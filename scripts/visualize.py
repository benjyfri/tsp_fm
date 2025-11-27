import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib import animation
from torchdiffeq import odeint
from pathlib import Path

# --- FIX 0: Set Geomstats Backend ---
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'

# --- 1. PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# --- FIX 1: Comprehensive Imports ---
from src.models import (
    VectorFieldModel,
    RoPEVectorFieldModel,
    CanonicalRoPEVectorField,
    CanonicalMLPVectorField
)
from src.geometry import GeometryProvider
# Use shared utils for consistency
from src.dataset import load_data
from src.utils import reconstruct_tour, calculate_tour_length

# --- INFERENCE CORE (Trajectory Version) ---
# We keep this local because standard ode_solve_euler usually
# returns only the final state, but here we need the full path for GIFs.
@torch.no_grad()
def get_trajectory(model, x0, geometry, steps=100, device='cuda'):
    """
    Solves the ODE but returns the entire trajectory (Steps, Batch, N, 2)
    instead of just the final state.
    """
    def ode_func(t, y):
        # t is a scalar tensor, model expects (Batch,)
        t_batch = t.expand(y.shape[0]).to(device)
        return model(y, t_batch, geometry=geometry)

    t_span = torch.linspace(0., 1., steps=steps).to(device)
    # x0 shape: (1, N, 2)
    traj = odeint(ode_func, x0, t_span, method='dopri5')
    # traj shape: (Steps, 1, N, 2)
    return traj.squeeze(1) # -> (Steps, N, 2)

# --- VISUALIZATION ---
def save_flow_animation(traj, output_file="flow_animation.gif"):
    print(f"Generating flow animation ({len(traj)} frames)...")
    traj_np = traj.cpu().numpy()

    # Setup Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    # Add some margin to the view
    x_min, x_max = traj_np[:, :, 0].min(), traj_np[:, :, 0].max()
    y_min, y_max = traj_np[:, :, 1].min(), traj_np[:, :, 1].max()
    margin = 0.1
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Elements to animate
    scat = ax.scatter([], [], c='blue', s=40, alpha=0.8, edgecolors='k', linewidth=0.5)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, fontweight='bold')

    def init():
        scat.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return scat, time_text

    def update(frame):
        current_points = traj_np[frame]
        scat.set_offsets(current_points)

        t = frame / (len(traj_np) - 1)
        time_text.set_text(f'Time t={t:.2f}')

        # Change color at the end to indicate "Done"
        if frame == len(traj_np) - 1:
            scat.set_color('red')
        else:
            scat.set_color('cornflowerblue')
        return scat, time_text

    ani = animation.FuncAnimation(fig, update, frames=len(traj_np),
                                  init_func=init, blit=True, interval=40)

    # Robust Writer Selection
    if output_file.endswith('.mp4'):
        try:
            writer = animation.FFMpegWriter(fps=25)
        except Exception:
            print("FFMpeg not found, falling back to Pillow/GIF")
            output_file = output_file.replace('.mp4', '.gif')
            writer = animation.PillowWriter(fps=25)
    else:
        writer = animation.PillowWriter(fps=25)

    ani.save(output_file, writer=writer)
    plt.close(fig)
    print(f"Animation saved to {output_file}")

def plot_comparisons(original_x, traj, pred_tour, model_len, gt_path, gt_len, output_file):
    x0_norm = traj[0].cpu().numpy()
    x1_pred = traj[-1].cpu().numpy()
    orig = original_x.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # 1. Flow Trajectories
    axes[0].set_title("Shape Space Flow (Projected)", fontsize=14)
    # Plot faint lines for every point's trajectory
    for i in range(x0_norm.shape[0]):
        path = traj[:, i, :].cpu().numpy()
        axes[0].plot(path[:, 0], path[:, 1], alpha=0.15, color='gray', linewidth=1)

    axes[0].scatter(x0_norm[:, 0], x0_norm[:, 1], c='green', label='Start (t=0)', s=25, alpha=0.6)
    axes[0].scatter(x1_pred[:, 0], x1_pred[:, 1], c='red', label='End (t=1)', s=25)
    axes[0].set_aspect('equal')
    axes[0].legend(loc='upper right')

    # Metrics String
    gap_str = "N/A"
    if gt_len is not None and gt_len > 0:
        gap = ((model_len - gt_len) / gt_len) * 100
        gap_str = f"{gap:.2f}%"

    # 2. Model Reconstruction
    axes[1].set_title(f"Model Prediction\nLen: {model_len:.4f} | Gap: {gap_str}", fontsize=14)
    axes[1].scatter(orig[:, 0], orig[:, 1], c='blue', s=40, zorder=5)

    # Draw predicted tour
    lines = [[tuple(orig[i]), tuple(orig[j])] for i, j in zip(pred_tour, pred_tour[1:])]
    lines.append([tuple(orig[pred_tour[-1]]), tuple(orig[pred_tour[0]])]) # Close loop

    lc = mc.LineCollection(lines, colors='blue', linewidths=2.0, alpha=0.7)
    axes[1].add_collection(lc)
    axes[1].set_aspect('equal')

    # 3. Ground Truth
    if gt_path is not None:
        axes[2].set_title(f"Ground Truth\nLen: {gt_len:.4f}", fontsize=14)
        axes[2].scatter(orig[:, 0], orig[:, 1], c='green', s=40, zorder=5)

        # Draw GT tour
        # Assuming gt_path is a list/tensor of INDICES
        gt_indices = gt_path.cpu().numpy() if torch.is_tensor(gt_path) else gt_path

        gt_lines = [[tuple(orig[gt_indices[i]]), tuple(orig[gt_indices[(i+1)%len(gt_indices)]])]
                    for i in range(len(gt_indices))]

        lc_gt = mc.LineCollection(gt_lines, colors='green', linewidths=2.0, alpha=0.7)
        axes[2].add_collection(lc_gt)
    else:
        axes[2].set_title("Ground Truth Not Available", fontsize=14)

    axes[2].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Comparison plot saved to {output_file}")

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/home/benjamin.fri/PycharmProjects/tsp_fm/checkpoints/kendall_ROPE_02/best_model.pt')
    parser.add_argument('--input_file', type=str, default='/home/benjamin.fri/PycharmProjects/tsp_fm/data/processed_data_geom_val.pt')
    parser.add_argument('--gpu_id', type=int, default=3)
    parser.add_argument('--sample_idx', type=int, default=0, help="Index of the sample to visualize")
    parser.add_argument('--output_dir', type=str, default="visualizations")

    # Fallback Defaults
    parser.add_argument('--num_points', type=int, default=50)
    parser.add_argument('--interpolant', type=str, default='kendall')
    parser.add_argument('--model_type', type=str, default='rope',
                        choices=['concat', 'rope', 'canonical_mlp', 'canonical_rope'])
    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # A. Load Model & Config
    print(f"Loading Model from {args.model_path}...")
    try:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'args' in checkpoint:
        print("Found hyperparameters in checkpoint.")
        saved_args = checkpoint['args']
        if isinstance(saved_args, dict):
            model_args = argparse.Namespace(**saved_args)
        else:
            model_args = saved_args
        state_dict = checkpoint['model_state_dict']
    else:
        print("WARNING: Using command line args for model config.")
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model_args = args

    # B. Initialize Geometry
    n_points = getattr(model_args, 'num_points', args.num_points)
    interpolant_name = getattr(model_args, 'interpolant', args.interpolant)

    geo = None
    if interpolant_name == 'kendall':
        geo = GeometryProvider(n_points)

    # C. Initialize Model
    model_type = getattr(model_args, 'model_type', 'concat')
    print(f"Initializing {model_type} model...")

    if model_type == 'rope':
        model = RoPEVectorFieldModel(model_args).to(device)
    elif model_type == 'canonical_rope':
        model = CanonicalRoPEVectorField(model_args).to(device)
    elif model_type == 'canonical_mlp':
        model = CanonicalMLPVectorField(model_args).to(device)
    else:
        model = VectorFieldModel(model_args).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    # D. Load Data (Standardized)
    print(f"Loading data from {args.input_file}...")
    # This uses the same logic as training/inference
    x0_all, _, _, gt_paths_all = load_data(args.input_file, device)

    # Select specific sample
    if args.sample_idx >= len(x0_all):
        print(f"Error: sample_idx {args.sample_idx} out of bounds (max {len(x0_all)-1})")
        sys.exit(1)

    x0_raw = x0_all[args.sample_idx].to(dtype=torch.float32) # (N, 2)
    gt_path = gt_paths_all[args.sample_idx] # Indices (N)

    # E. Calculate GT Length
    gt_len = calculate_tour_length(x0_raw, gt_path)
    print(f"Ground Truth Length for sample {args.sample_idx}: {gt_len:.5f}")

    # F. Inference
    x0_input = x0_raw.unsqueeze(0) # (1, N, 2)

    print(f"Running flow generation...")
    # Use geometry if kendall, else None
    use_geo = geo if interpolant_name == 'kendall' else None

    traj = get_trajectory(model, x0_input, geometry=use_geo, steps=100, device=device)

    # G. Evaluation & Reconstruction
    final_config = traj[-1].squeeze(0) # (N, 2)

    # Use shared utility for tour reconstruction
    pred_tour = reconstruct_tour(final_config)
    model_len = calculate_tour_length(x0_raw, pred_tour)

    print(f"\n{'='*30}")
    print(f"Model Length: {model_len:.5f}")
    gap = ((model_len - gt_len)/gt_len)*100
    print(f"Gap:          {gap:.2f}%")
    print(f"{'='*30}\n")

    # H. Saving
    base_name = f"sample_{args.sample_idx}_{model_type}"
    plot_path = os.path.join(args.output_dir, f"{base_name}_plot.png")
    anim_path = os.path.join(args.output_dir, f"{base_name}_flow.gif")

    plot_comparisons(x0_raw, traj, pred_tour, model_len, gt_path, gt_len, plot_path)
    save_flow_animation(traj, anim_path)

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    main()