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

# --- IMPORTS ---
from src.models import (
    VectorFieldModel,
    RoPEVectorFieldModel,
    CanonicalRoPEVectorField,
    CanonicalMLPVectorField,
    CanonicalRegressor
)
from src.geometry import GeometryProvider
from src.dataset import load_data
from src.utils import reconstruct_tour, calculate_tour_length

# --- INFERENCE CORE ---

@torch.no_grad()
def get_prediction(model, x0, geometry, steps=100, device='cuda'):
    """
    Handles both Regression (One-shot) and Flow Matching (ODE).
    Returns:
        traj: Tensor of shape (Steps, N, 2).
              For regression, this is just [x0, prediction].
    """
    # Ensure input is on the correct device
    x0 = x0.to(device)

    # 1. Check if Regression Model
    if isinstance(model, CanonicalRegressor):
        print("Detected Regression Model. Skipping ODE solver.")
        prediction = model(x0) # (1, N, 2)
        # Create a fake 2-step trajectory: [Start, End]
        traj = torch.stack([x0.squeeze(0), prediction.squeeze(0)], dim=0)
        return traj

    # 2. Flow Matching (ODE)
    def ode_func(t, y):
        # t is scalar, expand to batch
        t_batch = t.expand(y.shape[0]).to(device)
        return model(y, t_batch, geometry=geometry)

    t_span = torch.linspace(0., 1., steps=steps).to(device)
    # traj shape: (Steps, 1, N, 2) -> (Steps, N, 2)
    traj_raw = odeint(ode_func, x0, t_span, method='dopri5')
    return traj_raw.squeeze(1)

# --- VISUALIZATION ---

def save_flow_animation(traj, output_file="flow_animation.gif"):
    # If trajectory only has 2 steps (Regression), skip animation
    if len(traj) <= 2:
        print("Trajectory has <= 2 steps (Regression model). Skipping GIF generation.")
        return

    print(f"Generating flow animation ({len(traj)} frames)...")
    traj_np = traj.cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    x_min, x_max = traj_np[:, :, 0].min(), traj_np[:, :, 0].max()
    y_min, y_max = traj_np[:, :, 1].min(), traj_np[:, :, 1].max()
    margin = 0.1
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')

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
        if frame == len(traj_np) - 1:
            scat.set_color('red')
        else:
            scat.set_color('cornflowerblue')
        return scat, time_text

    ani = animation.FuncAnimation(fig, update, frames=len(traj_np),
                                  init_func=init, blit=True, interval=40)

    try:
        writer = animation.PillowWriter(fps=25)
        ani.save(output_file, writer=writer)
        print(f"Animation saved to {output_file}")
    except Exception as e:
        print(f"Could not save animation: {e}")
    finally:
        plt.close(fig)

def plot_comparisons(original_x, traj, pred_tour, model_len, gt_path, gt_len, output_file):
    x0_norm = traj[0].cpu().numpy()
    x1_pred = traj[-1].cpu().numpy()
    orig = original_x.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # 1. Flow/Mapping Visualization
    axes[0].set_title("Input -> Output Mapping", fontsize=14)

    # If it's a flow (many steps), draw faint lines. If regression (2 steps), draw arrows.
    if len(traj) > 2:
        for i in range(x0_norm.shape[0]):
            path = traj[:, i, :].cpu().numpy()
            axes[0].plot(path[:, 0], path[:, 1], alpha=0.15, color='gray', linewidth=1)
    else:
        # Draw arrows for regression
        for i in range(x0_norm.shape[0]):
            axes[0].arrow(x0_norm[i, 0], x0_norm[i, 1],
                          x1_pred[i, 0] - x0_norm[i, 0], x1_pred[i, 1] - x0_norm[i, 1],
                          alpha=0.2, color='gray', head_width=0.02)

    axes[0].scatter(x0_norm[:, 0], x0_norm[:, 1], c='green', label='Start', s=25, alpha=0.6)
    axes[0].scatter(x1_pred[:, 0], x1_pred[:, 1], c='red', label='Prediction', s=25)
    axes[0].set_aspect('equal')
    axes[0].legend(loc='upper right')

    # Metrics
    gap_str = "N/A"
    if gt_len is not None and gt_len > 0:
        gap = ((model_len - gt_len) / gt_len) * 100
        gap_str = f"{gap:.2f}%"

    # 2. Model Reconstruction
    axes[1].set_title(f"Model Tour\nLen: {model_len:.4f} | Gap: {gap_str}", fontsize=14)
    axes[1].scatter(orig[:, 0], orig[:, 1], c='blue', s=40, zorder=5)

    lines = [[tuple(orig[i]), tuple(orig[j])] for i, j in zip(pred_tour, pred_tour[1:])]
    lines.append([tuple(orig[pred_tour[-1]]), tuple(orig[pred_tour[0]])])
    lc = mc.LineCollection(lines, colors='blue', linewidths=2.0, alpha=0.7)
    axes[1].add_collection(lc)
    axes[1].set_aspect('equal')

    # 3. Ground Truth
    if gt_path is not None:
        axes[2].set_title(f"Ground Truth\nLen: {gt_len:.4f}", fontsize=14)
        axes[2].scatter(orig[:, 0], orig[:, 1], c='green', s=40, zorder=5)

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
    parser.add_argument('--model_path', type=str, default='/home/benjamin.fri/PycharmProjects/tsp_fm/scripts_direct/checkpoints_reg/Regression-D256-L12/best_model.pt')
    parser.add_argument('--input_file', type=str, default='data/processed_data_geom_val.pt')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--sample_idx', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default="visualizations")

    # Fallback options
    parser.add_argument('--model_type', type=str, default='rope',
                        choices=['concat', 'rope', 'canonical_mlp', 'canonical_rope', 'regression'])

    args = parser.parse_args()

    # Resolve paths
    input_path = Path(args.input_file)
    if not input_path.exists():
        input_path = Path(parent_dir) / args.input_file

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # A. Load Model
    print(f"Loading Model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)

    if 'args' in checkpoint:
        model_args = argparse.Namespace(**checkpoint['args'])
    else:
        model_args = args

    # Check for regression override
    model_type = getattr(model_args, 'model_type', args.model_type)
    if 'Reg' in args.model_path or 'regression' in str(model_type).lower():
        model_type = 'regression'

    print(f"Initializing {model_type} model...")

    if model_type == 'regression':
        model = CanonicalRegressor(model_args).to(device)
    elif model_type == 'rope':
        model = RoPEVectorFieldModel(model_args).to(device)
    elif model_type == 'canonical_rope':
        model = CanonicalRoPEVectorField(model_args).to(device)
    elif model_type == 'canonical_mlp':
        model = CanonicalMLPVectorField(model_args).to(device)
    else:
        model = VectorFieldModel(model_args).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # B. Load Data
    x0_all, _, _, gt_paths_all, _ = load_data(str(input_path), device, interpolant=None)

    # FIX: Ensure sample is moved to the correct device!
    x0_raw = x0_all[args.sample_idx].to(device=device, dtype=torch.float32)

    gt_path = gt_paths_all[args.sample_idx]
    gt_len = calculate_tour_length(x0_raw, gt_path)

    # C. Inference
    x0_input = x0_raw.unsqueeze(0) # (1, N, 2)

    geo = None
    if getattr(model_args, 'interpolant', 'kendall') == 'kendall':
        geo = GeometryProvider(x0_raw.shape[0])

    print("Running Inference...")
    traj = get_prediction(model, x0_input, geometry=geo, steps=100, device=device)

    # D. Reconstruct
    final_config = traj[-1].squeeze(0)
    pred_tour = reconstruct_tour(final_config)
    model_len = calculate_tour_length(x0_raw, pred_tour)

    print(f"Gap: {((model_len - gt_len)/gt_len)*100:.2f}%")

    # E. Save
    base_name = f"sample_{args.sample_idx}_{model_type}"
    plot_path = os.path.join(args.output_dir, f"{base_name}_plot.png")
    anim_path = os.path.join(args.output_dir, f"{base_name}_flow.gif")

    plot_comparisons(x0_raw, traj, pred_tour, model_len, gt_path, gt_len, plot_path)
    save_flow_animation(traj, anim_path)

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    main()





