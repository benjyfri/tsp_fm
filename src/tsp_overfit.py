import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# 1. Global Configuration
# ============================================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Running on {DEVICE}...")


# ============================================================
# 2. Sobolev Preconditioner
# ============================================================
class SobolevPreconditioner:
    """Filters gradients to enforce smooth, coherent movement."""

    def __init__(self, num_points, s=1.5):
        self.N = num_points
        self.s = s
        k = torch.arange(num_points, device=DEVICE).float()
        k_symmetric = torch.minimum(k, num_points - k)
        self.kernel = (1.0 + k_symmetric ** 2) ** (-self.s)
        self.kernel = self.kernel / self.kernel.mean()

    def apply(self, grad_data):
        if torch.isnan(grad_data).any():
            grad_data = torch.nan_to_num(grad_data, nan=0.0)
        grad_fft = torch.fft.fft(grad_data, dim=1)
        grad_sobolev_fft = grad_fft * self.kernel.view(1, -1, 1)
        return torch.fft.ifft(grad_sobolev_fft, dim=1).real


# ============================================================
# 3. Physics Core
# ============================================================
def tp_energy_loss(curve, alpha=2.0, beta=3.0):
    """
    Tangent-Point Energy.
    Uses Beta=3.0 for a balance between strong repulsion and stability.
    """
    B, N, D = curve.shape

    # 1. Tangents & Lengths
    curve_next = torch.roll(curve, shifts=-1, dims=1)
    curve_prev = torch.roll(curve, shifts=1, dims=1)
    tangents_raw = (curve_next - curve_prev) / 2.0
    dl = torch.norm(tangents_raw, dim=-1)
    tangents = tangents_raw / (dl.unsqueeze(-1) + 1e-8)

    # 2. Pairwise Differences
    u, v = curve.unsqueeze(2), curve.unsqueeze(1)
    diffs = u - v
    dists_sq = (diffs ** 2).sum(dim=-1)

    # 3. Cross Product Kernel |T x D|^2
    t_expand = tangents.unsqueeze(2).expand(-1, -1, N, -1)
    dot_prod = (t_expand * diffs).sum(dim=-1)
    # Clamp min to 0 to avoid numerical -1e-9
    cross_sq = torch.clamp(dists_sq - dot_prod ** 2, min=0.0)

    # 4. Energy Density
    numerator = cross_sq.pow(alpha / 2.0)
    # Strong epsilon (1e-3) prevents explosion when strands touch
    denominator = (dists_sq + 1e-3).pow(beta / 2.0)

    energy_mat = (numerator / denominator) * (dl.unsqueeze(2) * dl.unsqueeze(1))

    # 5. Mask Neighbors (Singularity handling)
    eye = torch.eye(N, device=DEVICE).bool()
    mask = eye | torch.roll(eye, 1, 0) | torch.roll(eye, -1, 0)

    return energy_mat[~mask.unsqueeze(0)].mean()


def compute_length(curve):
    return torch.norm(curve - torch.roll(curve, shifts=-1, dims=1), dim=-1).sum()


def flatness_loss(curve):
    return (curve[..., 3] ** 2).sum()


def get_trefoil(M):
    t = torch.linspace(0, 2 * np.pi, M + 1)[:-1].to(DEVICE)
    x = torch.sin(t) + 2 * torch.sin(2 * t)
    y = torch.cos(t) - 2 * torch.cos(2 * t)
    z = -torch.sin(3 * t)
    w = torch.zeros_like(t)
    return torch.stack([x, y, z, w], dim=-1).unsqueeze(0) / 3.0


# ============================================================
# 4. Experiment 1: The Topological Proof
# ============================================================
def run_4d_vs_3d_experiment():
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: 3D (Stuck) vs 4D (Untangling)")
    print("Using Hard Constraints for Length Stability")
    print("=" * 60)

    M = 200
    STEPS = 5000
    LR = 1e-2
    SAVE_EVERY = 50

    # Init Curves
    c_init = get_trefoil(M)

    # 3D Curve
    curve_3d = nn.Parameter(c_init.clone())

    # 4D Curve (Add initial 'lift' noise)
    c4 = c_init.clone()
    c4[..., 3] += 0.15 * torch.randn(M, device=DEVICE)
    curve_4d = nn.Parameter(c4)

    # Lock Target Length
    TARGET_LEN = compute_length(c_init).item()
    print(f"Target Length Locked: {TARGET_LEN:.4f}")

    opt_3d = torch.optim.SGD([curve_3d], lr=LR, momentum=0.9)
    opt_4d = torch.optim.SGD([curve_4d], lr=LR, momentum=0.9)

    sob = SobolevPreconditioner(M, s=1.5)
    curve_3d.register_hook(lambda g: sob.apply(g))
    curve_4d.register_hook(lambda g: sob.apply(g))

    hist_3d, hist_4d = [], []

    for step in range(STEPS):
        progress = step / STEPS

        # --- 3D Update ---
        opt_3d.zero_grad()
        curve_3d.data[..., 3] = 0.0  # Force 3D Manifold

        # Only Repulsive Energy (Length is handled by constraint)
        loss_3d = tp_energy_loss(curve_3d, beta=3.0)

        loss_3d.backward()
        curve_3d.grad[..., 3] = 0.0  # Kill W gradient
        # Clip to prevent explosion
        torch.nn.utils.clip_grad_norm_([curve_3d], 1.0)
        opt_3d.step()

        # Hard Constraint: Renormalize Length
        with torch.no_grad():
            curr_l = compute_length(curve_3d)
            curve_3d.data = curve_3d.data * (TARGET_LEN / curr_l)

        # --- 4D Update ---
        opt_4d.zero_grad()

        # Flatness Schedule: Wait until 60% done to start flattening
        w_flat = 0.05
        if progress > 0.6:
            # Quadratic ramp up
            w_flat += 50.0 * ((progress - 0.6) / 0.4) ** 2

        l_tp = tp_energy_loss(curve_4d, beta=3.0)
        l_flat = flatness_loss(curve_4d)

        loss_4d = l_tp + w_flat * l_flat

        loss_4d.backward()
        torch.nn.utils.clip_grad_norm_([curve_4d], 1.0)
        opt_4d.step()

        # Hard Constraint: Renormalize Length
        with torch.no_grad():
            curr_l = compute_length(curve_4d)
            curve_4d.data = curve_4d.data * (TARGET_LEN / curr_l)

        # Logging
        if step % SAVE_EVERY == 0:
            hist_3d.append(curve_3d.detach().cpu().numpy()[0])
            hist_4d.append(curve_4d.detach().cpu().numpy()[0])

        if step % 200 == 0:
            # 3D should be stuck (~high loss), 4D should drop
            print(
                f"Step {step} | 3D Loss: {loss_3d.item():.4f} | 4D Loss: {l_tp.item():.4f} (Flat: {l_flat.item():.2f})")

    # --- Plotting ---
    print("Generating Experiment 1 Animation...")
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("3D Optimization\n(Stuck Knotted)")
    ax1.set_xlim(-1, 1);
    ax1.set_ylim(-1, 1);
    ax1.set_zlim(-1, 1)
    line1, = ax1.plot([], [], [], 'r-', lw=3, alpha=0.8)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("4D Optimization\n(Untangles & Flattens)")
    ax2.set_xlim(-1, 1);
    ax2.set_ylim(-1, 1);
    ax2.set_zlim(-1, 1)
    line2, = ax2.plot([], [], [], 'g-', lw=3, alpha=0.8)

    def update(frame):
        d3 = hist_3d[frame];
        l3 = np.vstack([d3, d3[0]])
        line1.set_data(l3[:, 0], l3[:, 1]);
        line1.set_3d_properties(l3[:, 2])

        d4 = hist_4d[frame];
        l4 = np.vstack([d4, d4[0]])
        line2.set_data(l4[:, 0], l4[:, 1]);
        line2.set_3d_properties(l4[:, 2])

        angle = (frame / len(hist_3d)) * 360
        ax1.view_init(30, angle);
        ax2.view_init(30, angle)
        return line1, line2

    ani = animation.FuncAnimation(fig, update, frames=len(hist_3d), interval=30)
    ani.save('exp1_topology.gif', writer='pillow', fps=30)
    print("Saved exp1_topology.gif")


# ============================================================
# 5. Experiment 2: TSP Solver
# ============================================================
def run_tsp_solver():
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: TSP Solver")
    print("=" * 60)

    N = 50
    M = 200
    STEPS = 5000
    LR = 5e-2
    PRINT_EVERY = 500
    SAVE_EVERY = 50

    cities = torch.rand(1, N, 2, device=DEVICE)
    curve_init = torch.cat(
        [cities.repeat(1, M // N + 1, 1)[:, :M], 0.05 * torch.randn(1, M, 2, device=DEVICE)], dim=-1
    )
    curve = nn.Parameter(curve_init.detach().clone())

    optimizer = torch.optim.SGD([curve], lr=LR, momentum=0.9)
    sob = SobolevPreconditioner(M, s=1.5)
    curve.register_hook(lambda g: sob.apply(g))

    def transport_loss(cities, curve, epsilon=0.0005):
        dists_sq = torch.cdist(cities, curve[..., :2]) ** 2
        val = -epsilon * torch.logsumexp(-dists_sq / epsilon, dim=2)
        return val.mean()

    history = []
    loss_log = {'trsp': [], 'tp': [], 'len': [], 'flat': [], 'total': []}

    for step in range(STEPS):
        optimizer.zero_grad()
        progress = step / STEPS

        w_trsp = 2000.0 * np.exp(-10.0 * progress) + 10.0
        w_tp = 0.5
        w_len = 0.1 + 1.0 * progress
        w_flat = 1.0
        if progress > 0.75:
            w_flat = 50.0 * ((progress - 0.75) / 0.25) ** 2

        l_trsp = transport_loss(cities, curve)
        l_tp = tp_energy_loss(curve, beta=4.0)  # Higher beta for TSP local repulsion
        l_len = compute_length(curve)  # Use actual length function
        l_flat = flatness_loss(curve)

        total_loss = (w_trsp * l_trsp) + (w_tp * l_tp) + (w_len * l_len) + (w_flat * l_flat)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([curve], 1.0)
        optimizer.step()

        # Log
        norm_trsp = l_trsp.item() + (0.0005 * np.log(M))
        loss_log['trsp'].append(norm_trsp)
        loss_log['tp'].append(l_tp.item())
        loss_log['len'].append(l_len.item())
        loss_log['flat'].append(l_flat.item())

        if step % SAVE_EVERY == 0:
            history.append(curve.detach().cpu().numpy()[0])

        if step % PRINT_EVERY == 0:
            print(f"Step {step} | Trsp: {norm_trsp:.5f} | TP: {l_tp.item():.2f}")

    # Dashboard
    print("Generating TSP Dashboard...")
    cities_np = cities.detach().cpu().numpy()[0]
    fig = plt.figure(figsize=(16, 10))

    ax1 = fig.add_subplot(221)
    ax1.set_title("2D Projection")
    ax1.set_xlim(0, 1);
    ax1.set_ylim(0, 1)
    sc1 = ax1.scatter(cities_np[:, 0], cities_np[:, 1], c='k', s=20, zorder=5)
    ln1, = ax1.plot([], [], 'r-', lw=2)

    ax2 = fig.add_subplot(222)
    ax2.set_title("Dynamics")
    ax2.set_xlim(0, STEPS)
    ax2.set_yscale('symlog', linthresh=0.01)
    ax2.grid(True, alpha=0.3)
    l_ln, = ax2.plot([], [], label='Len', c='b')
    t_ln, = ax2.plot([], [], label='Trsp', c='g')
    tp_ln, = ax2.plot([], [], label='Repuls', c='orange')
    f_ln, = ax2.plot([], [], label='Flat', c='purple')
    ax2.legend()

    ax3 = fig.add_subplot(223, projection='3d')
    ax3.set_title("3D View")
    ax3.set_xlim(0, 1);
    ax3.set_ylim(0, 1);
    ax3.set_zlim(-0.2, 0.2)
    ln3, = ax3.plot([], [], [], 'g-', lw=1.5)

    ax4 = fig.add_subplot(224, projection='3d')
    ax4.set_title("4D View")
    ax4.set_xlim(0, 1);
    ax4.set_ylim(0, 1);
    ax4.set_zlim(-0.2, 0.2)
    ln4, = ax4.plot([], [], [], 'm-', lw=1.5)

    def update(frame):
        cur = frame * SAVE_EVERY
        d = history[frame];
        l = np.vstack([d, d[0]])
        ln1.set_data(l[:, 0], l[:, 1])
        ln3.set_data(l[:, 0], l[:, 1]);
        ln3.set_3d_properties(l[:, 2])
        ln4.set_data(l[:, 0], l[:, 1]);
        ln4.set_3d_properties(l[:, 3])

        angle = (frame / len(history)) * 360
        ax3.view_init(30, angle);
        ax4.view_init(30, angle + 90)

        if cur < len(loss_log['len']):
            x = np.arange(cur + 1)
            l_ln.set_data(x, loss_log['len'][:cur + 1])
            t_ln.set_data(x, loss_log['trsp'][:cur + 1])
            tp_ln.set_data(x, loss_log['tp'][:cur + 1])
            f_ln.set_data(x, loss_log['flat'][:cur + 1])

        return [ln1, ln3, ln4, l_ln, t_ln, tp_ln, f_ln]

    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=30)
    ani.save('exp2_tsp_solver.gif', writer='pillow', fps=30)
    print("Saved 'exp2_tsp_solver.gif'")


if __name__ == "__main__":
    run_4d_vs_3d_experiment()
    # run_tsp_solver()