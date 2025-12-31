import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import collections as mc
from matplotlib import animation
from matplotlib.lines import Line2D
from torchdiffeq import odeint
from sklearn.isotonic import IsotonicRegression

# --- FIX 0: Set Geomstats Backend ---
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
# --- FIX 1: Enforce Float32 ---
torch.set_default_dtype(torch.float32)

# --- 1. PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# --- Imports ---
from src.models import (
    VectorFieldModel,
    RoPEVectorFieldModel,
    CanonicalRoPEVectorField,
    CanonicalMLPVectorField,
    CanonicalRegressor,
    SpectralCanonMLP,
    SpectralCanonTransformer
)
from src.geometry import GeometryProvider
from src.dataset import load_data
# We use these utils for metrics, but use direct odeint for trajectory history
from src.utils import reconstruct_tour, calculate_tour_length

def is_valid_tour(tour, n_points):
    return len(set(tour)) == n_points

# --- INFERENCE CORE (FIXED) ---
@torch.no_grad()
def get_trajectory(model, x0, geometry, steps=40, device='cuda'):
    """
    Generates the flow trajectory.
    CHANGED: Uses 'rk4' (fixed step) instead of 'dopri5' (adaptive).
    This prevents the solver from getting stuck at stiff regions (e.g. t=0.22).
    """

    def ode_func(t, y):
        t_batch = t.expand(y.shape[0]).to(device)
        return model(y, t_batch, geometry=geometry)

    # We enforce exact steps for the animation frames
    t_span = torch.linspace(0., 1., steps=steps).to(device)

    # Using fixed-step RK4 ensures we don't get stuck in infinite refinement loops
    traj = odeint(
        ode_func,
        x0,
        t_span,
        method='rk4',
        options={'step_size': 1.0 / (steps - 1)}
    )
    return traj.squeeze(1)  # (Steps, N, 2)


# --- MATH HELPERS ---

def center_and_norm(x):
    """Centers x and normalizes Frobenius norm to 1 (Numpy)."""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = x - x.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(x)
    return x / (norm + 1e-8)


def center_and_norm_torch(x):
    """Centers x and normalizes Frobenius norm to 1 (Torch)."""
    x_centered = x - torch.mean(x, dim=-2, keepdim=True)
    norm = torch.norm(x_centered, p='fro', dim=(-2, -1), keepdim=True)
    return x_centered / (norm + 1e-8)


def align_procrustes(source, target):
    """
    Aligns source to target using Procrustes analysis (optimal rotation/reflection).
    """
    # Assume centered inputs
    M = target.T @ source
    U, S, V = np.linalg.svd(M)
    R = U @ V

    # Det check for reflection (ensure we stay in O(2))
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ V

    return source @ R.T


def get_circle_basis(p0, p1):
    """
    Returns (u, v, theta) such that the great circle connecting p0 and p1
    is given by: c(t) = cos(t)*u + sin(t)*v
    """
    p0_flat = p0.flatten()
    p1_flat = p1.flatten()

    u = p0_flat

    # Project p1 onto tangent space of p0 to find v direction
    dot_val = np.dot(p0_flat, p1_flat)

    # Clamp for numerical stability
    dot_val = np.clip(dot_val, -1.0, 1.0)
    theta = np.arccos(dot_val)

    v_raw = p1_flat - dot_val * p0_flat
    v_norm = np.linalg.norm(v_raw)

    if v_norm < 1e-6:
        # p0 and p1 are parallel or antipodal
        v = np.zeros_like(u)
        v[0] = 1.0  # arbitrary fallback
        v = v - np.dot(u, v) * u
        v = v / np.linalg.norm(v)
    else:
        v = v_raw / v_norm

    return u, v, theta


def get_tour_segments(points, tour_indices):
    """Helper to create line segments for a tour."""
    if torch.is_tensor(tour_indices):
        tour_indices = tour_indices.cpu().numpy()

    # Ensure indices are integers
    tour_indices = tour_indices.astype(int)

    segments = []
    for i in range(len(tour_indices)):
        idx1 = tour_indices[i]
        idx2 = tour_indices[(i + 1) % len(tour_indices)]
        segments.append((points[idx1], points[idx2]))
    return segments


# --- SUPER GT GENERATION ---

def create_super_gt(points, tour_indices, max_iter=100, tol=1e-7):
    """
    Generates the optimal 'Super GT' Q in Kendall Shape Space.
    """
    # Ensure numpy
    if torch.is_tensor(points): points = points.cpu().numpy()
    if torch.is_tensor(tour_indices): tour_indices = tour_indices.cpu().numpy()

    N = points.shape[0]

    # --- Step A: Preprocess Input P ---
    P_ordered = points[tour_indices]
    P_centered = P_ordered - np.mean(P_ordered, axis=0)
    norm_P = np.linalg.norm(P_centered)
    P_norm = P_centered / (norm_P + 1e-8)

    # --- Step B: Setup Target Q parameters ---
    R = 1.0 / np.sqrt(N)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    iso_reg = IsotonicRegression(increasing=True, out_of_bounds='clip')

    # --- Step C: Iterative Optimization ---
    for i in range(max_iter):
        Q = R * np.stack([np.cos(theta), np.sin(theta)], axis=1)

        M = P_norm.T @ Q
        U, _, Vt = np.linalg.svd(M)
        O = U @ Vt

        if np.linalg.det(O) < 0:
            U[:, -1] *= -1
            O = U @ Vt

        P_aligned_to_Q = P_norm @ O
        raw_angles = np.arctan2(P_aligned_to_Q[:, 1], P_aligned_to_Q[:, 0])

        diff = raw_angles - theta
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        target_angles = theta + diff
        new_theta = iso_reg.fit_transform(np.arange(N), target_angles)

        loss = np.mean((new_theta - theta) ** 2)
        theta = new_theta
        if loss < tol:
            break

    # --- Step D: Final Output Generation ---
    Q_canonical = R * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    M_final = Q_canonical.T @ P_norm
    U_f, _, Vt_f = np.linalg.svd(M_final)
    O_final = U_f @ Vt_f

    if np.linalg.det(O_final) < 0:
        U_f[:, -1] *= -1
        O_final = U_f @ Vt_f

    Q_final = Q_canonical @ O_final
    Q_original_indices = np.zeros_like(Q_final)
    Q_original_indices[tour_indices] = Q_final

    return torch.tensor(Q_original_indices, dtype=torch.float32)


# --- VISUALIZATION FUNCTIONS ---

def compare_super_gt_interpolations(x0, x_orig, x_super, gt_path, output_file="super_gt_comparison.gif", steps=60):
    """
    Side-by-side animation:
    Left: Interpolation Input -> Standard Uniform Circle
    Right: Interpolation Input -> Super GT Circle
    """
    print(f"Generating Super GT comparison: {output_file}...")

    if torch.is_tensor(x0): x0 = x0.detach().cpu().numpy()
    if torch.is_tensor(x_orig): x_orig = x_orig.detach().cpu().numpy()
    if torch.is_tensor(x_super): x_super = x_super.detach().cpu().numpy()

    # Pre-process
    x0 = center_and_norm(x0)
    x_orig_aligned = align_procrustes(x_orig, x0)
    x_super_aligned = align_procrustes(x_super, x0)

    # Compute Geodesic Bases
    u_orig, v_orig, theta_orig = get_circle_basis(x0, x_orig_aligned)
    u_super, v_super, theta_super = get_circle_basis(x0, x_super_aligned)

    # Setup Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    titles = ["Input -> Uniform Circle", "Input -> Super GT (Optimized)"]
    targets = [x_orig_aligned, x_super_aligned]
    bases = [(u_orig, v_orig, theta_orig), (u_super, v_super, theta_super)]

    scatters = []
    lines = []

    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=14, fontweight='bold')
        ax.set_aspect('equal')

        all_pts = np.concatenate([x0, targets[i]])
        margin = 0.1
        ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
        ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
        ax.grid(True, linestyle=':', alpha=0.4)

        ax.scatter(x0[:, 0], x0[:, 1], c='gray', alpha=0.3, marker='s', label='Input')
        ax.scatter(targets[i][:, 0], targets[i][:, 1], c='gray', alpha=0.3, marker='x', label='Target')

        scat = ax.scatter(x0[:, 0], x0[:, 1],
                          c=cm.turbo(np.linspace(0, 1, len(x0))),
                          s=80, edgecolor='k', zorder=5)

        init_segs = get_tour_segments(x0, gt_path)
        lc = mc.LineCollection(init_segs, colors='black', linewidths=1.0, alpha=0.2)
        ax.add_collection(lc)

        scatters.append(scat)
        lines.append(lc)
        if i == 0: ax.legend(loc='upper left')

    def update(frame):
        t = frame / (steps - 1)
        artists = []

        for i in range(2):
            u, v, theta = bases[i]
            angle = t * theta
            current_pos = (np.cos(angle) * u + np.sin(angle) * v).reshape(-1, 2)

            scatters[i].set_offsets(current_pos)
            segs = get_tour_segments(current_pos, gt_path)
            lines[i].set_segments(segs)
            artists.append(scatters[i])
            artists.append(lines[i])

        return artists

    ani = animation.FuncAnimation(fig, update, frames=steps, blit=True, interval=50)
    writer = animation.PillowWriter(fps=25)
    ani.save(output_file, writer=writer)
    plt.close(fig)
    print("Comparison saved.")


def animate_model_sensitivity(model, x0, geometry, gt_path, output_file="model_sensitivity.gif", noise_scale=0.05,
                              steps=40, device='cuda'):
    """
    NEW FUNCTION:
    Visualizes susceptibility to drift by running the model twice:
    Left: Standard Inference (Clean x0)
    Right: Inference with Perturbed Initial Condition (x0 + noise)
    """
    print(f"Generating Model Sensitivity Analysis: {output_file}...")
    x0 = x0.to(device)

    # 1. Trajectory 1: Clean
    traj_clean = get_trajectory(model, x0.unsqueeze(0), geometry, steps, device).detach().cpu().numpy()

    # 2. Trajectory 2: Perturbed
    # Create noise, add to x0, then re-project to manifold (Center & Norm)
    noise = torch.randn_like(x0) * noise_scale
    x0_noisy = x0 + noise
    x0_noisy = center_and_norm_torch(x0_noisy)

    traj_noisy = get_trajectory(model, x0_noisy.unsqueeze(0), geometry, steps, device).detach().cpu().numpy()

    # 3. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    titles = ["Model Inference (Clean)", f"Model Inference (Drifted/Noisy)\nSigma={noise_scale}"]
    trajs = [traj_clean, traj_noisy]

    # Determine bounds
    all_pts = np.concatenate([traj_clean, traj_noisy])
    margin = 0.1
    vmin, vmax = all_pts.min() - margin, all_pts.max() + margin

    scatters = []
    lines = []
    texts = []

    colors = cm.turbo(np.linspace(0, 1, x0.shape[0]))

    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.grid(True, linestyle=':', alpha=0.4)

        # Ghost of Start
        start_np = trajs[i][0]
        ax.scatter(start_np[:, 0], start_np[:, 1], c='gray', alpha=0.3, marker='s', label='Start')

        # Active Elements
        scat = ax.scatter(start_np[:, 0], start_np[:, 1], c=colors, s=80, edgecolor='k', zorder=5)
        segs = get_tour_segments(start_np, gt_path)
        lc = mc.LineCollection(segs, colors='black', linewidths=1.0, alpha=0.3)
        ax.add_collection(lc)

        # Drift Metric Text
        txt = ax.text(0.02, 0.02, "", transform=ax.transAxes,
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        scatters.append(scat)
        lines.append(lc)
        texts.append(txt)
        if i == 0: ax.legend(loc='upper left')

    def update(frame):
        artists = []
        # Calculate drift between Clean and Noisy at this frame
        current_clean = trajs[0][frame]
        current_noisy = trajs[1][frame]
        drift_dist = np.linalg.norm(current_clean - current_noisy)

        for i in range(2):
            data = trajs[i][frame]
            scatters[i].set_offsets(data)
            segs = get_tour_segments(data, gt_path)
            lines[i].set_segments(segs)

            t = frame / (steps - 1)
            texts[i].set_text(f"t={t:.2f}\nDrift (L2): {drift_dist:.4f}")

            artists.extend([scatters[i], lines[i], texts[i]])
        return artists

    ani = animation.FuncAnimation(fig, update, frames=steps, blit=True, interval=50)
    writer = animation.PillowWriter(fps=25)
    ani.save(output_file, writer=writer)
    plt.close(fig)
    print("Sensitivity animation saved.")


def animate_geodesic_diff(traj_model, x_gt, gt_path, output_file="geodesic_diff.gif"):
    """
    Animates the difference between the Learned Flow and the Ideal Geodesic.
    """
    print(f"Generating geodesic difference animation...")

    if torch.is_tensor(traj_model): traj_model = traj_model.detach().cpu()
    if torch.is_tensor(x_gt): x_gt = x_gt.detach().cpu()

    x0 = traj_model[0]
    steps = traj_model.shape[0]
    n_points = traj_model.shape[1]

    # 1. Compute Ideal Geodesic
    x0_np = x0.numpy()
    x_gt_np = x_gt.numpy()
    x_gt_aligned = align_procrustes(x_gt_np, x0_np)  # Align GT to Input

    traj_model_np = traj_model.numpy()

    # Ideal Geodesic (SLERP for Pre-shape sphere)
    u, v, theta = get_circle_basis(x0_np, x_gt_aligned)

    traj_ideal_np = np.zeros_like(traj_model_np)
    ts = np.linspace(0, 1, steps)

    for i, t in enumerate(ts):
        angle = t * theta
        point_flat = np.cos(angle) * u + np.sin(angle) * v
        traj_ideal_np[i] = point_flat.reshape(n_points, 2)

    # Setup Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    all_x = np.concatenate([traj_model_np[..., 0].flatten(), traj_ideal_np[..., 0].flatten()])
    all_y = np.concatenate([traj_model_np[..., 1].flatten(), traj_ideal_np[..., 1].flatten()])
    margin = 0.1
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)

    colors = cm.turbo(np.linspace(0, 1, n_points))

    # --- Initialize Artists ---
    segments_tour_init = get_tour_segments(traj_model_np[0], gt_path)
    lc_tour = mc.LineCollection(segments_tour_init, colors='black', linewidths=1.0, alpha=0.3, zorder=3)
    ax.add_collection(lc_tour)

    scat_model = ax.scatter(traj_model_np[0, :, 0], traj_model_np[0, :, 1],
                            c=colors, s=100, zorder=5, edgecolor='k', label='Learned Flow')
    scat_ideal = ax.scatter(traj_ideal_np[0, :, 0], traj_ideal_np[0, :, 1],
                            c=colors, s=80, marker='x', alpha=0.6, zorder=4, label='Ideal Geodesic')

    segments_err_init = [[tuple(traj_model_np[0, i]), tuple(traj_ideal_np[0, i])] for i in range(n_points)]
    lines_collection = mc.LineCollection(segments_err_init, colors='red', linewidths=1, alpha=0.5, linestyle=':')
    ax.add_collection(lines_collection)

    title_text = ax.text(0.5, 1.02, '', transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='Model Output'),
        Line2D([0], [0], marker='x', color='gray', linestyle='None', label='Ideal Geodesic (GT)'),
        Line2D([0], [0], color='black', alpha=0.3, label='GT Tour Edges'),
        Line2D([0], [0], color='red', linestyle=':', label='Deviation'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    def update(frame):
        current_model = traj_model_np[frame]
        current_ideal = traj_ideal_np[frame]

        error = np.linalg.norm(current_model - current_ideal, axis=1).mean()
        t = frame / (steps - 1)

        scat_model.set_offsets(current_model)
        scat_ideal.set_offsets(current_ideal)
        segments_err = [[tuple(current_model[i]), tuple(current_ideal[i])] for i in range(n_points)]
        lines_collection.set_segments(segments_err)
        segments_tour = get_tour_segments(current_model, gt_path)
        lc_tour.set_segments(segments_tour)
        title_text.set_text(f't={t:.2f} | Mean Deviation: {error:.4f}')
        return scat_model, scat_ideal, lines_collection, lc_tour, title_text

    ani = animation.FuncAnimation(fig, update, frames=steps, blit=True, interval=50)
    writer = animation.PillowWriter(fps=20)
    ani.save(output_file, writer=writer)
    plt.close(fig)
    print(f"Geodesic difference animation saved to {output_file}")


def animate_alternative_paths(x0, x_gt, gt_path, output_file="path_comparisons.gif", steps=60):
    """
    Creates a 2x2 animation comparing different movement types in Shape Space.
    """
    print("Generating alternative paths animation...")

    if torch.is_tensor(x0): x0 = x0.detach().cpu().numpy()
    if torch.is_tensor(x_gt): x_gt = x_gt.detach().cpu().numpy()

    n_points = x0.shape[0]
    x0 = center_and_norm(x0)
    x_gt_raw = center_and_norm(x_gt)
    x_gt_aligned = align_procrustes(x_gt_raw, x0)
    u, v, theta = get_circle_basis(x0, x_gt_aligned)

    trajs = {
        'Geodesic (Shortest)': [],
        'Geodesic (Complement)': [],
        'Euclidean (Shrinks)': [],
        'Normalized Linear (NLERP)': []
    }
    ts = np.linspace(0, 1, steps)

    for t in ts:
        # A. Geodesic Shortest
        angle_short = t * theta
        flat_short = np.cos(angle_short) * u + np.sin(angle_short) * v
        trajs['Geodesic (Shortest)'].append(flat_short.reshape(n_points, 2))
        # B. Geodesic Complement
        angle_long = t * (theta - 2 * np.pi)
        flat_long = np.cos(angle_long) * u + np.sin(angle_long) * v
        trajs['Geodesic (Complement)'].append(flat_long.reshape(n_points, 2))
        # C. Euclidean
        shape_linear = (1 - t) * x0 + t * x_gt_aligned
        trajs['Euclidean (Shrinks)'].append(shape_linear)
        # D. NLERP
        lerp = (1 - t) * x0 + t * x_gt_aligned
        nlerp = lerp / (np.linalg.norm(lerp) + 1e-8)
        trajs['Normalized Linear (NLERP)'].append(nlerp)

    for k in trajs:
        trajs[k] = np.array(trajs[k])

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    titles = list(trajs.keys())
    colors = cm.turbo(np.linspace(0, 1, n_points))

    all_data = np.concatenate([t for t in trajs.values()])
    margin = 0.15
    xlim = (all_data[..., 0].min() - margin, all_data[..., 0].max() + margin)
    ylim = (all_data[..., 1].min() - margin, all_data[..., 1].max() + margin)

    scatters = []
    line_collections = []
    texts = []

    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=16, fontweight='bold')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.4)

        ax.scatter(x0[:, 0], x0[:, 1], c='gray', alpha=0.2, marker='s', s=40)
        ax.scatter(x_gt_aligned[:, 0], x_gt_aligned[:, 1], c='gray', alpha=0.2, marker='x', s=40)

        init_pos = trajs[titles[i]][0]
        segs = get_tour_segments(init_pos, gt_path)
        lc = mc.LineCollection(segs, colors='black', linewidths=1.5, alpha=0.3)
        ax.add_collection(lc)
        line_collections.append(lc)

        scat = ax.scatter(init_pos[:, 0], init_pos[:, 1],
                          c=colors, s=80, edgecolor='k', linewidth=0.5, zorder=5)
        scatters.append(scat)

        txt = ax.text(0.05, 0.05, '', transform=ax.transAxes, fontsize=12,
                      verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        texts.append(txt)

    def update(frame):
        t = frame / (steps - 1)
        artists = []
        for i, key in enumerate(titles):
            data = trajs[key][frame]
            scatters[i].set_offsets(data)
            artists.append(scatters[i])
            segs = get_tour_segments(data, gt_path)
            line_collections[i].set_segments(segs)
            artists.append(line_collections[i])
            norm = np.linalg.norm(data)
            texts[i].set_text(f"t={t:.2f}\nNorm: {norm:.3f}")
            artists.append(texts[i])
        return artists

    ani = animation.FuncAnimation(fig, update, frames=steps, blit=True, interval=40)
    writer = animation.PillowWriter(fps=25)
    ani.save(output_file, writer=writer)
    plt.close(fig)
    print(f"Path comparison animation saved to {output_file}")


def plot_comparisons(original_x, traj, x1_gt, pred_tour, model_len, gt_path, gt_len, output_file):
    """
    Standard comparison plot (Flow, TSP Reconstructions, Deviation).
    """
    x0_norm = traj[0].cpu().numpy()
    x1_pred = traj[-1].cpu().numpy()
    orig = original_x.cpu().numpy()

    if torch.is_tensor(x1_gt):
        x1_gt = x1_gt.cpu().numpy()

    num_points = x0_norm.shape[0]
    point_colors = cm.turbo(np.linspace(0, 1, num_points))

    fig, axes = plt.subplots(1, 4, figsize=(28, 7))

    # --- 1. Shape Space Flow ---
    axes[0].set_title("Shape Space Flow\n(Colored by Point Identity)", fontsize=14)
    for i in range(num_points):
        path = traj[:, i, :].cpu().numpy()
        axes[0].plot(path[:, 0], path[:, 1], alpha=0.5, color=point_colors[i], linewidth=1.5)
    axes[0].scatter(x0_norm[:, 0], x0_norm[:, 1], c='black', alpha=0.3, s=20, label='Start')
    axes[0].set_aspect('equal')

    # Metrics
    gap_str = "N/A"
    if gt_len is not None and gt_len > 0:
        gap = ((model_len - gt_len) / gt_len) * 100
        gap_str = f"{gap:.2f}%"

    # --- 2. Model Prediction ---
    axes[1].set_title(f"Model Prediction\nLen: {model_len:.4f} | Gap: {gap_str}", fontsize=14)
    axes[1].scatter(orig[:, 0], orig[:, 1], c='blue', s=40, zorder=5)
    lines = [[tuple(orig[i]), tuple(orig[j])] for i, j in zip(pred_tour, pred_tour[1:])]
    lines.append([tuple(orig[pred_tour[-1]]), tuple(orig[pred_tour[0]])])
    lc = mc.LineCollection(lines, colors='blue', linewidths=2.0, alpha=0.7)
    axes[1].add_collection(lc)
    axes[1].set_aspect('equal')

    # --- 3. Ground Truth ---
    if gt_path is not None:
        axes[2].set_title(f"Ground Truth\nLen: {gt_len:.4f}", fontsize=14)
        axes[2].scatter(orig[:, 0], orig[:, 1], c='green', s=40, zorder=5)
        gt_indices = gt_path.cpu().numpy() if torch.is_tensor(gt_path) else gt_path
        gt_lines = [[tuple(orig[gt_indices[i]]), tuple(orig[gt_indices[(i + 1) % len(gt_indices)]])]
                    for i in range(len(gt_indices))]
        lc_gt = mc.LineCollection(gt_lines, colors='green', linewidths=2.0, alpha=0.7)
        axes[2].add_collection(lc_gt)
    else:
        axes[2].set_title("Ground Truth Not Available", fontsize=14)
    axes[2].set_aspect('equal')

    # --- 4. Deviation Analysis ---
    axes[3].set_title("Deviation Analysis\n(Square=Input -> Circle=Pred -> X=Target)", fontsize=14)
    for i in range(num_points):
        path = traj[:, i, :].cpu().numpy()
        axes[3].plot(path[:, 0], path[:, 1], alpha=0.2, color=point_colors[i], linewidth=1)
        axes[3].plot([x1_pred[i, 0], x1_gt[i, 0]], [x1_pred[i, 1], x1_gt[i, 1]],
                     alpha=0.6, color='black', linestyle=':', linewidth=1.0, zorder=2)

    axes[3].scatter(x0_norm[:, 0], x0_norm[:, 1], edgecolors=point_colors, facecolors='none',
                    s=60, marker='s', linewidth=1.5, zorder=3)
    axes[3].scatter(x1_pred[:, 0], x1_pred[:, 1], c=point_colors, s=40,
                    marker='o', edgecolors='white', linewidth=0.5, zorder=4)
    for i in range(num_points):
        axes[3].scatter(x1_gt[i, 0], x1_gt[i, 1], color=point_colors[i], s=50,
                        marker='x', linewidth=1.5, zorder=3)
    axes[3].set_aspect('equal')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_file, dpi=150)
    print(f"Comparison plot saved to {output_file}")


# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()

    # Defaults tailored to user script
    # parser.add_argument('--model_path', type=str,
    #                     default=r"/home/benjamin.fri/PycharmProjects/tsp_fm/checkpoints/canonical_rope-kendall_sfm-D512-L20-P63.2M/final_model.pt")
    parser.add_argument('--model_path', type=str,
                        default=r"/home/benjamin.fri/PycharmProjects/tsp_fm/checkpoints/spectral_trans_38.67M_L8_H8_D512_lr1e-04/final_model.pt")
    parser.add_argument('--input_file', type=str,
                        default='/home/benjamin.fri/PycharmProjects/tsp_fm/data/can_tsp50_val.pt')
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--sample_idx', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default="visualizations")

    parser.add_argument('--num_points', type=int, default=50)
    parser.add_argument('--interpolant', type=str, default='kendall')
    parser.add_argument('--model_type', type=str, default='rope')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    print(f"Loading Model from {args.model_path}...")
    try:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'args' in checkpoint:
        saved_args = checkpoint['args']
        if isinstance(saved_args, dict):
            model_args = argparse.Namespace(**saved_args)
        else:
            model_args = saved_args
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model_args = args

    n_points = getattr(model_args, 'num_points', args.num_points)
    interpolant_name = getattr(model_args, 'interpolant', args.interpolant)

    geo = None
    if 'kendall' in interpolant_name:
        geo = GeometryProvider(n_points)

    model_type = getattr(model_args, 'model_type', 'concat')
    print(f"Initializing {model_type} model...")

    if model_type == 'rope':
        model = RoPEVectorFieldModel(model_args).to(device)
    elif model_type == 'canonical_rope':
        model = CanonicalRoPEVectorField(model_args).to(device)
    elif model_type == 'canonical_mlp':
        model = CanonicalMLPVectorField(model_args).to(device)
        # --- NEW MODELS ---
    elif model_type == 'canonical_regressor':
        model = CanonicalRegressor(model_args).to(device)
    elif model_type == 'spectral_mlp':
        model = SpectralCanonMLP(model_args).to(device)
    elif model_type == 'spectral_trans':
        model = SpectralCanonTransformer(model_args).to(device)
    else:
        model = VectorFieldModel(model_args).to(device)

    # Load the state dict
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    # --- FIX: Remove 'torch.compile' prefix if it exists ---
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "")  # remove the prefix
        new_state_dict[name] = v

    # Load the cleaned state dict
    model.load_state_dict(new_state_dict)
    model.eval()

    # Load Data
    print(f"Loading data from {args.input_file}...")
    x0_all, x1_all, gt_paths_all, _ = load_data(args.input_file, device)

    if args.sample_idx >= len(x0_all):
        sys.exit(1)

    x0_raw = x0_all[args.sample_idx].to(dtype=torch.float32)
    x1_gt_actual = x1_all[args.sample_idx].to(dtype=torch.float32)

    # Note: x1_all contains the "Original" Uniform targets, but we will regenerate them
    # below to be absolutely sure of the Radius=1/sqrt(N) scaling for the comparison.
    gt_path = gt_paths_all[args.sample_idx]

    # --- NEW: Generate Super GT ---
    # print("Generating Super GT target (Optimized Circle)...")
    # x1_super_gt = create_super_gt(x0_raw, gt_path).to(device)

    # --- Generate Correct Uniform Baseline (Norm=1) for Comparison ---
    # We regenerate the Uniform Circle to strictly adhere to Radius = 1/sqrt(N)
    N = x0_raw.shape[0]
    R_uniform = 1.0 / np.sqrt(N)
    theta_uni = np.linspace(0, 2 * np.pi, N, endpoint=False)
    Q_uni_ordered = R_uniform * np.stack([np.cos(theta_uni), np.sin(theta_uni)], axis=1)

    # Map back to original indices
    if torch.is_tensor(gt_path):
        gt_path_np = gt_path.cpu().numpy()
    else:
        gt_path_np = gt_path

    Q_uni = np.zeros_like(Q_uni_ordered)
    Q_uni[gt_path_np] = Q_uni_ordered
    x1_target_uniform = torch.tensor(Q_uni, dtype=torch.float32).to(device)

    # Inference (Standard)
    print(f"Running flow generation...")
    x0_input = x0_raw.unsqueeze(0).to(device)
    use_geo = geo if ('kendall' in interpolant_name) else None

    # UPDATED: odeint now uses RK4 via function param, so we just request 'steps'
    traj = get_trajectory(model, x0_input, geometry=use_geo, steps=10, device=device)
    final_config = traj[-1].squeeze(0)

    # Metrics
    pred_tour = reconstruct_tour(final_config)
    if torch.is_tensor(pred_tour):
        pred_tour = pred_tour.cpu()

    if not is_valid_tour(pred_tour.tolist(), n_points):
        print("WARNING: Predicted tour is INVALID (contains duplicates)!")


    # print(f"gt_path: {gt_path}")
    # print(f"pred_tour: {pred_tour}")
    gt_len = calculate_tour_length(x0_raw, gt_path)
    model_len = calculate_tour_length(x0_raw, pred_tour)

    print(f"\n{'=' * 30}")
    print(f"Model Length: {model_len:.5f}")
    print(f"GT Length:    {gt_len:.5f}")
    print(f"{'=' * 30}\n")

    # Saving
    base_name = f"sample_{args.sample_idx}_{model_type}"
    plot_path = os.path.join(args.output_dir, f"{base_name}_plot.png")
    anim_path = os.path.join(args.output_dir, f"{base_name}_geodesic_compare.gif")
    paths_path = os.path.join(args.output_dir, f"{base_name}_paths_analysis.gif")
    super_gt_path = os.path.join(args.output_dir, f"{base_name}_super_gt_compare.gif")
    sensitivity_path = os.path.join(args.output_dir, f"{base_name}_model_sensitivity.gif")

    # Standard plots (Using uniform target as the reference for 'accuracy' to model training)
    plot_comparisons(x0_raw, traj, x1_gt_actual, pred_tour, model_len, gt_path, gt_len, plot_path)
    animate_geodesic_diff(traj, x1_gt_actual, gt_path, anim_path)
    # animate_alternative_paths(x0_raw, x1_gt_actual, gt_path, paths_path)

    # --- NEW: Run Super GT Comparison ---
    # Compare interpolation to Uniform (Original) vs Interpolation to Super GT
    # compare_super_gt_interpolations(x0_raw, x1_gt_actual, x1_super_gt, gt_path, super_gt_path)

    # --- NEW: Run Drift/Sensitivity Analysis ---
    # Compare Clean Inference vs Noisy Inference
    # animate_model_sensitivity(model, x0_raw, use_geo, gt_path, sensitivity_path, noise_scale=0.005, device=device)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    main()