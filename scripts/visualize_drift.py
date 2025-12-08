import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import collections as mc
from matplotlib import animation
from torchdiffeq import odeint

# --- CONFIGURATION ---
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'


# --- 1. GEOMETRY & INTERPOLANT CLASSES ---

class GeometryProvider:
    def __init__(self, n_points, ambient_dim=2):
        self.n_points = n_points
        self.ambient_dim = ambient_dim

    def center_and_norm(self, x):
        """Centers x and normalizes Frobenius norm to 1."""
        if torch.is_tensor(x):
            x_centered = x - torch.mean(x, dim=-2, keepdim=True)
            norm = torch.norm(x_centered, p='fro', dim=(-2, -1), keepdim=True)
            return x_centered / (norm + 1e-8)
        else:
            x_centered = x - np.mean(x, axis=-2, keepdims=True)
            norm = np.linalg.norm(x_centered, axis=(-2, -1), keepdims=True)
            return x_centered / (norm + 1e-8)


class KendallInterpolant:
    def __init__(self):
        pass

    def precompute(self, x0, x1):
        """Pre-compute geodesic parameters (theta, tangent vector u)."""
        x0_d = x0.double()
        x1_d = x1.double()

        # Inner product (assume centered/normalized inputs)
        inner_prod = torch.sum(x0_d * x1_d, dim=[-2, -1])
        cos_theta = torch.clamp(inner_prod, -1.0 + 1e-7, 1.0 - 1e-7)
        theta_geo = torch.acos(cos_theta)

        # Handle small angles
        mask = theta_geo < 1e-4
        scale_factor = torch.zeros_like(theta_geo)
        scale_factor[~mask] = theta_geo[~mask] / torch.sin(theta_geo[~mask])
        scale_factor[mask] = 1.0
        scale_factor = scale_factor.view(-1, 1, 1)

        # Log map (Tangent vector u at x0 pointing to x1)
        # Log_x0(x1) = (theta / sin(theta)) * (x1 - x0 * cos(theta))
        log_x1_x0 = scale_factor * (x1_d - x0_d * cos_theta.view(-1, 1, 1))

        return {
            'theta_geo': theta_geo.float(),
            'log_x1_x0': log_x1_x0.float(),
            'x0': x0.float()
        }

    def sample_analytical(self, params, t_scalar):
        """Returns x(t) using closed form geodesic formula."""
        x0 = params['x0']
        theta = params['theta_geo']
        u = params['log_x1_x0']

        angle = t_scalar * theta
        sin_angle = torch.sin(angle).view(-1, 1, 1)
        cos_angle = torch.cos(angle).view(-1, 1, 1)
        theta_view = theta.view(-1, 1, 1)

        # Geodesic: x(t) = x0 cos(t*theta) + (u / theta) sin(t*theta)
        xt = x0 * cos_angle + (u / (theta_view + 1e-8)) * sin_angle
        return xt

    def get_velocity_at_t(self, params, t_scalar):
        """Returns x'(t) (The vector field) at time t."""
        x0 = params['x0']
        theta = params['theta_geo']
        u = params['log_x1_x0']

        angle = t_scalar * theta
        sin_angle = torch.sin(angle).view(-1, 1, 1)
        cos_angle = torch.cos(angle).view(-1, 1, 1)
        theta_view = theta.view(-1, 1, 1)

        # Derivative: x'(t) = -x0*theta*sin(t*theta) + u*cos(t*theta)
        vt = -x0 * theta_view * sin_angle + u * cos_angle
        return vt


# --- 2. HELPER FUNCTIONS ---

def align_procrustes(source, target):
    """Aligns source to target using Procrustes analysis."""
    if torch.is_tensor(source): source = source.detach().cpu().numpy()
    if torch.is_tensor(target): target = target.detach().cpu().numpy()

    M = target.T @ source
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return torch.tensor(source @ R.T, dtype=torch.float32)


def create_standard_uniform(points, tour_indices):
    """Creates Standard GT (Uniform Circle)"""
    if torch.is_tensor(points): points = points.cpu().numpy()
    if torch.is_tensor(tour_indices): tour_indices = tour_indices.cpu().numpy()

    N = points.shape[0]
    R = 1.0 / np.sqrt(N)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    Q_ordered = R * np.stack([np.cos(theta), np.sin(theta)], axis=1)

    Q_final = np.zeros_like(Q_ordered)
    Q_final[tour_indices] = Q_ordered
    return torch.tensor(Q_final, dtype=torch.float32)


def get_tour_segments(points, tour_indices):
    if torch.is_tensor(points): points = points.cpu().numpy()
    if torch.is_tensor(tour_indices): tour_indices = tour_indices.cpu().numpy()
    tour_indices = tour_indices.astype(int)

    segments = []
    for i in range(len(tour_indices)):
        idx1 = tour_indices[i]
        idx2 = tour_indices[(i + 1) % len(tour_indices)]
        segments.append((points[idx1], points[idx2]))
    return segments


# --- 3. MAIN EXECUTION ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str,
                        default='/home/benjamin.fri/PycharmProjects/tsp_fm/data/processed_data_geom_val.pt')
    parser.add_argument('--output_file', type=str, default='drift_comparison.gif')
    parser.add_argument('--sample_idx', type=int, default=0)
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Data
    print(f"Loading data from {args.input_file}...")
    try:
        data = torch.load(args.input_file, weights_only=False)
        if isinstance(data, (list, tuple)):
            x0_all = data[0]
            tours_all = data[2]
        elif isinstance(data, dict):
            x0_all = data.get('data', data.get('x0'))
            tours_all = data.get('tours', data.get('gt_paths'))
        else:
            raise ValueError("Unknown data format")
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    # 2. Prepare Sample
    x0 = torch.tensor(x0_all['points']).to(DEVICE)
    gt_path = x0_all['path']

    # Normalize x0
    geo = GeometryProvider(x0.shape[0])
    x0 = geo.center_and_norm(x0)

    # Generate Target (Uniform Circle)
    x1_orig = create_standard_uniform(x0, gt_path).to(DEVICE)

    # Align Target to x0 (Required for minimal geodesic)
    x1_aligned = align_procrustes(x1_orig, x0).to(DEVICE)

    # 3. Setup Interpolant
    interpolant = KendallInterpolant()
    # Batch dimension needed for computations
    params = interpolant.precompute(x0.unsqueeze(0), x1_aligned.unsqueeze(0))

    t_span = torch.linspace(0, 1, args.steps).to(DEVICE)

    # --- A. Analytical Geodesic (Exact) ---
    print("Computing Analytical Trajectory...")
    traj_analytical = []
    for t in t_span:
        xt = interpolant.sample_analytical(params, t)
        traj_analytical.append(xt)
    traj_analytical = torch.stack(traj_analytical).squeeze(1).cpu()

    # --- B. ODE Solver with INACCURATE FIRST STEP ---
    print("Computing ODE Drift Trajectory (With bad initial step)...")

    def ode_func(t, x):
        # We use the vector field defined by the geodesic params at time t
        return interpolant.get_velocity_at_t(params, t)

    # 1. Manual Inaccurate Step 1
    # We take a simple Euler step and add noise to simulate inaccuracy
    dt = t_span[1] - t_span[0]
    v0 = interpolant.get_velocity_at_t(params, t_span[0])

    # Add significant noise to create the "inaccurate" start condition
    # x1_inaccurate implicitly becomes (1, N, 2) because v0 is (1, N, 2)
    noise = torch.randn_like(x0) * 0.05
    x1_inaccurate = x0 + (v0 * dt) + noise

    # 2. Accurate Integration for the rest (Steps 2 to N)
    # We start integrating from the inaccurate state x1 at time t_span[1]
    # FIX: Do NOT unsqueeze x1_inaccurate again, it is already (1, N, 2)
    traj_rest = odeint(ode_func, x1_inaccurate, t_span[1:], method='dopri5')

    # 3. Concatenate: [x0 (start), ...traj_rest (drifted path)]
    # x0 is (N, 2), we need (1, 1, N, 2) to match traj_rest's (T, 1, N, 2)
    traj_ode = torch.cat([x0.unsqueeze(0).unsqueeze(0), traj_rest], dim=0)
    traj_ode = traj_ode.squeeze(1).detach().cpu()

    # --- 4. VISUALIZATION ---
    print(f"Generating Animation: {args.output_file}...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    titles = ["Analytical Geodesic\n(Exact Manifold Movement)",
              "ODE Solver Integration\n(Inaccurate 1st Step -> Drift)"]
    trajs = [traj_analytical, traj_ode]

    # Calculate bounds
    all_pts = torch.cat([traj_analytical, traj_ode], dim=0).numpy()
    margin = 0.1
    x_min, x_max = all_pts[..., 0].min() - margin, all_pts[..., 0].max() + margin
    y_min, y_max = all_pts[..., 1].min() - margin, all_pts[..., 1].max() + margin

    scatters = []
    lines = []
    texts = []

    colors = cm.turbo(np.linspace(0, 1, x0.shape[0]))

    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=14, fontweight='bold')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.5)

        # Static elements
        target_np = x1_aligned.cpu().numpy()
        ax.scatter(target_np[:, 0], target_np[:, 1], c='gray', marker='x', alpha=0.3, label='Target')
        ax.scatter(x0.cpu().numpy()[:, 0], x0.cpu().numpy()[:, 1], c='gray', marker='s', alpha=0.3, label='Start')

        # Dynamic elements
        start_pts = trajs[i][0]
        scat = ax.scatter(start_pts[:, 0], start_pts[:, 1], c=colors, s=80, edgecolor='k', zorder=5)

        segs = get_tour_segments(start_pts, gt_path)
        lc = mc.LineCollection(segs, colors='black', linewidths=1.0, alpha=0.3)
        ax.add_collection(lc)

        txt = ax.text(0.02, 0.02, "", transform=ax.transAxes,
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'), fontsize=10)

        scatters.append(scat)
        lines.append(lc)
        texts.append(txt)

        if i == 0: ax.legend(loc='upper right')

    def update(frame):
        artists = []
        for i in range(2):
            current_pts = trajs[i][frame].numpy()

            scatters[i].set_offsets(current_pts)

            segs = get_tour_segments(current_pts, gt_path)
            lines[i].set_segments(segs)

            # Metric: Frobenius Norm (Should be exactly 1.0)
            frob_norm = np.linalg.norm(current_pts)
            drift = abs(frob_norm - 1.0)

            texts[i].set_text(f"t={t_span[frame]:.2f}\nNorm: {frob_norm:.6f}\nDrift Error: {drift:.6f}")

            artists.extend([scatters[i], lines[i], texts[i]])
        return artists

    ani = animation.FuncAnimation(fig, update, frames=args.steps, blit=True, interval=50)
    writer = animation.PillowWriter(fps=20)
    ani.save(args.output_file, writer=writer)
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()