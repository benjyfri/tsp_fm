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

# --- FIX 0: Set Geomstats Backend ---
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
# --- FIX 1: Enforce Float32 globally ---
torch.set_default_dtype(torch.float32)

# --- 1. PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# --- Imports ---
try:
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
    from src.utils import reconstruct_tour, calculate_tour_length
except ImportError:
    # Fallback for standalone testing if src isn't in path
    print("Warning: Could not import from src. Ensure script is run with project root in PYTHONPATH.")
    sys.exit(1)


def is_valid_tour(tour, n_points):
    return len(set(tour)) == n_points


# --- INFERENCE CORE ---
@torch.no_grad()
def get_trajectory(model, x0, geometry, steps=40, device='cuda'):
    """
    Generates the flow trajectory using RK4.
    """

    def ode_func(t, y):
        # Ensure t is float32
        t_val = t.item() if torch.is_tensor(t) else t
        t_batch = torch.tensor([t_val], dtype=torch.float32, device=device).expand(y.shape[0])
        return model(y, t_batch, geometry=geometry)

    t_span = torch.linspace(0., 1., steps=steps, dtype=torch.float32).to(device)

    traj = odeint(
        ode_func,
        x0,
        t_span,
        method='rk4',
        options={'step_size': 1.0 / (steps - 1)}
    )
    return traj.squeeze(1)  # (Steps, N, 2)


# --- MATH HELPERS ---

def align_procrustes(source, target):
    """
    Aligns source to target using Procrustes analysis (optimal rotation/reflection).
    """
    M = target.T @ source
    U, S, V = np.linalg.svd(M)
    R = U @ V

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
    dot_val = np.dot(p0_flat, p1_flat)
    dot_val = np.clip(dot_val, -1.0, 1.0)
    theta = np.arccos(dot_val)

    v_raw = p1_flat - dot_val * p0_flat
    v_norm = np.linalg.norm(v_raw)

    if v_norm < 1e-6:
        v = np.zeros_like(u)
        v[0] = 1.0
        v = v - np.dot(u, v) * u
        v = v / np.linalg.norm(v)
    else:
        v = v_raw / v_norm

    return u, v, theta


def get_tour_segments(points, tour_indices):
    """Helper to create line segments for a tour."""
    if torch.is_tensor(tour_indices):
        tour_indices = tour_indices.cpu().numpy()
    tour_indices = tour_indices.astype(int)

    segments = []
    for i in range(len(tour_indices)):
        idx1 = tour_indices[i]
        idx2 = tour_indices[(i + 1) % len(tour_indices)]
        segments.append((points[idx1], points[idx2]))
    return segments


def ensure_rope_cache_size(model, required_seq_len, device):
    """
    Dynamically resizes the precomputed RoPE frequency cache (freqs_cis)
    """
    # Check if model has the cache attribute directly
    if hasattr(model, 'freqs_cis'):
        current_len = model.freqs_cis.shape[0]
        if current_len < required_seq_len:
            print(f"Resizing RoPE cache: {current_len} -> {required_seq_len}")

            head_dim_complex = model.freqs_cis.shape[-1]
            head_dim = head_dim_complex * 2

            theta = 10000.0 ** (-2 * torch.arange(0, head_dim, 2).float() / head_dim)
            theta = theta.to(device)

            m = torch.arange(required_seq_len, device=device).float()
            freqs = torch.outer(m, theta)

            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
            model.freqs_cis = freqs_cis


# --- VISUALIZATION FUNCTIONS ---

def animate_geodesic_diff(traj_model, grid_data, particle_vectors_tensor, x_gt, gt_path,
                          output_file="geodesic_diff.gif"):
    """
    Animates:
    Left: Difference between Learned Flow and Ideal Geodesic.
    Right: VectorField (Grid) colored by magnitude + Moving Particles with their own vectors.

    grid_data: Tuple (X, Y, grid_vectors_tensor)
    particle_vectors_tensor: (Steps, N_points, 2)
    """
    print(f"Generating geodesic & grid vector field animation...")

    if torch.is_tensor(traj_model): traj_model = traj_model.detach().cpu()
    if torch.is_tensor(x_gt): x_gt = x_gt.detach().cpu()
    if torch.is_tensor(particle_vectors_tensor): particle_vectors_tensor = particle_vectors_tensor.detach().cpu()

    # Unpack grid data
    grid_X, grid_Y, grid_vectors_tensor = grid_data
    if torch.is_tensor(grid_vectors_tensor): grid_vectors_tensor = grid_vectors_tensor.detach().cpu()

    x0 = traj_model[0]
    steps = traj_model.shape[0]
    n_points = traj_model.shape[1]

    # --- 1. Compute Ideal Geodesic ---
    x0_np = x0.numpy()
    x_gt_np = x_gt.numpy()
    x_gt_aligned = align_procrustes(x_gt_np, x0_np)

    traj_model_np = traj_model.numpy()
    u, v, theta = get_circle_basis(x0_np, x_gt_aligned)

    traj_ideal_np = np.zeros_like(traj_model_np)
    ts = np.linspace(0, 1, steps)

    for i, t in enumerate(ts):
        angle = t * theta
        point_flat = np.cos(angle) * u + np.sin(angle) * v
        traj_ideal_np[i] = point_flat.reshape(n_points, 2)

    # --- 2. Setup Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Determine Limits
    margin = 0.05
    xlim = (grid_X.min() - margin, grid_X.max() + margin)
    ylim = (grid_Y.min() - margin, grid_Y.max() + margin)

    colors = cm.turbo(np.linspace(0, 1, n_points))

    # --- LEFT PLOT: Comparison ---
    ax_comp = axes[0]
    ax_comp.set_xlim(xlim)
    ax_comp.set_ylim(ylim)
    ax_comp.set_aspect('equal')
    ax_comp.grid(True, linestyle='--', alpha=0.3)
    ax_comp.set_title("Flow Trajectory vs Ideal Geodesic", fontsize=14, fontweight='bold')

    segments_tour_init = get_tour_segments(traj_model_np[0], gt_path)
    lc_tour = mc.LineCollection(segments_tour_init, colors='black', linewidths=1.0, alpha=0.3, zorder=3)
    ax_comp.add_collection(lc_tour)

    scat_model = ax_comp.scatter(traj_model_np[0, :, 0], traj_model_np[0, :, 1],
                                 c=colors, s=100, zorder=5, edgecolor='k', label='Learned Flow')
    scat_ideal = ax_comp.scatter(traj_ideal_np[0, :, 0], traj_ideal_np[0, :, 1],
                                 c=colors, s=80, marker='x', alpha=0.6, zorder=4, label='Ideal Geodesic')

    segments_err_init = [[tuple(traj_model_np[0, i]), tuple(traj_ideal_np[0, i])] for i in range(n_points)]
    lines_collection = mc.LineCollection(segments_err_init, colors='red', linewidths=1, alpha=0.5, linestyle=':')
    ax_comp.add_collection(lines_collection)

    text_comp = ax_comp.text(0.5, 0.95, '', transform=ax_comp.transAxes, ha='center',
                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='Model Output'),
        Line2D([0], [0], marker='x', color='gray', linestyle='None', label='Ideal Geodesic (GT)'),
        Line2D([0], [0], color='red', linestyle=':', label='Deviation'),
    ]
    ax_comp.legend(handles=legend_elements, loc='lower left')

    # --- RIGHT PLOT: Grid Vector Field + Particles ---
    ax_vec = axes[1]
    ax_vec.set_xlim(xlim)
    ax_vec.set_ylim(ylim)
    ax_vec.set_aspect('equal')
    ax_vec.grid(True, linestyle='--', alpha=0.3)
    ax_vec.set_title("Full Grid Vector Field + Particle Velocities", fontsize=14, fontweight='bold')

    # Ghost Trail
    lc_tour_vec = mc.LineCollection(segments_tour_init, colors='black', linewidths=1.0, alpha=0.1, zorder=2)
    ax_vec.add_collection(lc_tour_vec)

    # Particle Scatter
    scat_vec_points = ax_vec.scatter(traj_model_np[0, :, 0], traj_model_np[0, :, 1],
                                     c='black', s=20, alpha=0.6, zorder=5)

    # --- VECTOR FIELDS CONFIGURATION ---
    # 1. Grid Global Stats
    all_mags = torch.norm(grid_vectors_tensor, dim=-1).numpy()
    vmax = np.percentile(all_mags, 98)
    vmin = 0.0
    scale_val_grid = vmax * 15 if vmax > 1e-6 else 1.0

    # 2. Grid Quiver (Background)
    grid_V0 = grid_vectors_tensor[0].numpy()
    mags_0 = np.linalg.norm(grid_V0, axis=1)

    quiver_grid = ax_vec.quiver(grid_X, grid_Y,
                                grid_V0[:, 0], grid_V0[:, 1],
                                mags_0,
                                cmap='plasma',
                                clim=(vmin, vmax),
                                scale=scale_val_grid,
                                width=0.003,
                                alpha=0.9,
                                pivot='mid',
                                zorder=1)

    # 3. Particle Quiver (Foreground - Attached to points)
    # We use a fixed high-contrast color (Red) for these vectors
    part_V0 = particle_vectors_tensor[0].numpy()
    # Scale heuristic for particle vectors (often similar to grid, maybe slightly larger/smaller)
    scale_val_part = scale_val_grid

    quiver_particles = ax_vec.quiver(traj_model_np[0, :, 0], traj_model_np[0, :, 1],
                                     part_V0[:, 0], part_V0[:, 1],
                                     color='red',  # Fixed color
                                     scale=scale_val_part,
                                     width=0.006,  # Thicker than grid
                                     headwidth=4,
                                     headlength=4,
                                     alpha=1.0,
                                     pivot='mid',
                                     zorder=6)  # Topmost zorder

    # Add Colorbar for grid
    cbar = fig.colorbar(quiver_grid, ax=ax_vec, fraction=0.046, pad=0.04)
    cbar.set_label('Grid Vector Magnitude', rotation=270, labelpad=15)

    text_vec = ax_vec.text(0.5, 0.95, '', transform=ax_vec.transAxes, ha='center',
                           bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Update Function
    def update(frame):
        t = frame / (steps - 1)

        # 1. Update Left (Comparison)
        current_model = traj_model_np[frame]
        current_ideal = traj_ideal_np[frame]
        error = np.linalg.norm(current_model - current_ideal, axis=1).mean()

        scat_model.set_offsets(current_model)
        scat_ideal.set_offsets(current_ideal)

        segments_err = [[tuple(current_model[i]), tuple(current_ideal[i])] for i in range(n_points)]
        lines_collection.set_segments(segments_err)

        segments_tour = get_tour_segments(current_model, gt_path)
        lc_tour.set_segments(segments_tour)

        text_comp.set_text(f't={t:.2f} | Mean Deviation: {error:.4f}')

        # 2. Update Right (Grid + Particles)

        # A. Update Grid Vectors
        current_grid_vecs = grid_vectors_tensor[frame].numpy()
        current_grid_mags = np.linalg.norm(current_grid_vecs, axis=1)
        quiver_grid.set_UVC(current_grid_vecs[:, 0], current_grid_vecs[:, 1], current_grid_mags)

        # B. Update Particle Positions & Vectors
        scat_vec_points.set_offsets(current_model)
        lc_tour_vec.set_segments(segments_tour)

        current_part_vecs = particle_vectors_tensor[frame].numpy()
        # Important: For moving quivers, we must set offsets (position) AND UVC (direction)
        quiver_particles.set_offsets(current_model)
        quiver_particles.set_UVC(current_part_vecs[:, 0], current_part_vecs[:, 1])

        text_vec.set_text(f't={t:.2f}')

        return scat_model, scat_ideal, lines_collection, lc_tour, text_comp, scat_vec_points, quiver_grid, quiver_particles, text_vec

    # Duration logic
    min_duration_ms = 20000
    interval = int(max(50, min_duration_ms / steps))

    # Fix: use float division for fps to avoid ZeroDivisionError
    fps_val = 1000.0 / interval

    ani = animation.FuncAnimation(fig, update, frames=steps, blit=False, interval=interval)
    writer = animation.PillowWriter(fps=fps_val)
    ani.save(output_file, writer=writer)
    plt.close(fig)
    print(f"Animation saved to {output_file} (Duration: ~{steps * interval / 1000:.1f}s)")


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
    parser.add_argument('--model_path', type=str,
                        default=r"/home/benjamin.fri/PycharmProjects/tsp_fm/checkpoints/spectral_trans_76.48M_L16_H8_D512_lr3e-04/final_model.pt")
    parser.add_argument('--input_file', type=str,
                        default='/home/benjamin.fri/PycharmProjects/tsp_fm/data/can_tsp50_val.pt')
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--sample_idx', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default="visualizations")
    parser.add_argument('--num_points', type=int, default=50)
    parser.add_argument('--interpolant', type=str, default='kendall')
    parser.add_argument('--model_type', type=str, default='rope')
    parser.add_argument('--steps', type=int, default=40, help="Number of integration steps (frames)")

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
        model_args = argparse.Namespace(**saved_args) if isinstance(saved_args, dict) else saved_args
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
    elif model_type == 'canonical_regressor':
        model = CanonicalRegressor(model_args).to(device)
    elif model_type == 'spectral_mlp':
        model = SpectralCanonMLP(model_args).to(device)
    elif model_type == 'spectral_trans':
        model = SpectralCanonTransformer(model_args).to(device)
    else:
        model = VectorFieldModel(model_args).to(device)

    # Clean state dict
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "")
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    # --- FIX 2: FORCE FLOAT32 FOR WEIGHTS ---
    model = model.float()
    model.eval()

    # Load Data
    print(f"Loading data from {args.input_file}...")
    x0_all, x1_all, gt_paths_all, _ = load_data(args.input_file, device)

    if args.sample_idx >= len(x0_all):
        sys.exit(1)

    # Force float32 inputs
    x0_raw = x0_all[args.sample_idx].to(dtype=torch.float32)
    x1_gt_actual = x1_all[args.sample_idx].to(dtype=torch.float32)
    gt_path = gt_paths_all[args.sample_idx]

    # Inference
    print(f"Running flow generation ({args.steps} steps)...")
    x0_input = x0_raw.unsqueeze(0).to(device)
    use_geo = geo if ('kendall' in interpolant_name) else None

    # Get Trajectory
    traj = get_trajectory(model, x0_input, geometry=use_geo, steps=args.steps, device=device)

    # --- NEW: Compute Velocity Vectors for the Trajectory (Particles) ---
    print("Computing particle velocity vectors...")

    # RoPE Cache check: Ensure it fits the particle batch (usually 50)
    # This is usually smaller than grid, but we check anyway.
    ensure_rope_cache_size(model, traj.shape[1], device)

    particle_vectors_list = []
    time_steps = torch.linspace(0, 1, steps=traj.shape[0], dtype=torch.float32).to(device)

    for i, t in enumerate(time_steps):
        t_batch = t.unsqueeze(0)
        # Use trajectory state at this step: traj[i] shape (N, 2)
        # We need (1, N, 2) for model input
        current_state = traj[i].unsqueeze(0).to(device)

        with torch.no_grad():
            v_part = model(current_state, t_batch, geometry=use_geo)
        particle_vectors_list.append(v_part.squeeze(0).cpu())

    particle_vectors_tensor = torch.stack(particle_vectors_list)  # (Steps, N, 2)

    # --- SETUP GRID FOR VECTOR FIELD ---
    print("Computing grid vector field for animation...")

    # Define grid bounds based on trajectory with padding
    all_x = traj[..., 0]
    all_y = traj[..., 1]
    min_x, max_x = all_x.min().item(), all_x.max().item()
    min_y, max_y = all_y.min().item(), all_y.max().item()
    pad = 0.2

    grid_res = 20
    x_grid = np.linspace(min_x - pad, max_x + pad, grid_res)
    y_grid = np.linspace(min_y - pad, max_y + pad, grid_res)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    # Flatten to (1, N_grid, 2)
    grid_flat = np.stack([X_grid.flatten(), Y_grid.flatten()], axis=1)
    grid_input_tensor = torch.tensor(grid_flat, dtype=torch.float32).unsqueeze(0).to(device)

    # --- FIX 3: RESIZE ROPE CACHE FOR GRID SIZE ---
    ensure_rope_cache_size(model, grid_input_tensor.shape[1], device)

    grid_vectors_list = []

    # We reuse time_steps computed above
    for t in time_steps:
        t_batch = t.unsqueeze(0)
        with torch.no_grad():
            v_grid = model(grid_input_tensor, t_batch, geometry=use_geo)
        grid_vectors_list.append(v_grid.squeeze(0).cpu())

    grid_vectors_tensor = torch.stack(grid_vectors_list)  # (Steps, N_grid, 2)
    grid_data = (X_grid, Y_grid, grid_vectors_tensor)

    final_config = traj[-1].squeeze(0)

    # Metrics
    pred_tour = reconstruct_tour(final_config)
    if torch.is_tensor(pred_tour):
        pred_tour = pred_tour.cpu()

    if not is_valid_tour(pred_tour.tolist(), n_points):
        print("WARNING: Predicted tour is INVALID (contains duplicates)!")

    gt_len = calculate_tour_length(x0_raw, gt_path)
    model_len = calculate_tour_length(x0_raw, pred_tour)

    print(f"\n{'=' * 30}")
    print(f"Model Length: {model_len:.5f}")
    print(f"GT Length:    {gt_len:.5f}")
    print(f"{'=' * 30}\n")

    # Saving
    base_name = f"sample_{args.sample_idx}_{model_type}"
    plot_path = os.path.join(args.output_dir, f"{base_name}_plot.png")
    anim_path = os.path.join(args.output_dir, f"{base_name}_vis_combined.gif")

    plot_comparisons(x0_raw, traj, x1_gt_actual, pred_tour, model_len, gt_path, gt_len, plot_path)

    # Updated call with particle vectors
    animate_geodesic_diff(traj, grid_data, particle_vectors_tensor, x1_gt_actual, gt_path, anim_path)


if __name__ == "__main__":
    main()