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

# --- 1. PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# --- CHANGED: Import RoPEVectorFieldModel as well ---
from src.models import VectorFieldModel, RoPEVectorFieldModel
from src.geometry import GeometryProvider

# --- HELPER FUNCTIONS ---
def normalize_and_center(x):
    x = x - torch.mean(x, dim=0, keepdim=True)
    norm = torch.norm(x, p='fro')
    return x / norm

def get_angular_order(points):
    centroid = torch.mean(points, dim=0, keepdim=True)
    centered_points = points - centroid
    angles = torch.atan2(centered_points[:, 1], centered_points[:, 0])
    order = torch.argsort(angles)
    return order

def calculate_tour_length(points, order):
    if torch.is_tensor(order): order = order.cpu().numpy()
    if torch.is_tensor(points): points = points.cpu().numpy()
    ordered_points = points[order]
    next_points = np.roll(ordered_points, shift=-1, axis=0)
    diffs = ordered_points - next_points
    distances = np.linalg.norm(diffs, axis=1)
    return np.sum(distances)

# --- INFERENCE CORE ---
@torch.no_grad()
def solve_flow(model, x0, geometry, steps=100, device='cuda'):
    def ode_func(t, y):
        t_batch = t.expand(y.shape[0]).to(device)
        return model(y, t_batch, geometry=geometry)

    t_span = torch.linspace(0., 1., steps=steps).to(device)
    traj = odeint(ode_func, x0, t_span, method='dopri5')
    return traj.squeeze(1)

# --- VISUALIZATION ---
def save_flow_animation(original_x, traj, output_file="flow_animation.gif"):
    print("Generating flow animation...")
    traj_np = traj.cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(traj_np[:, :, 0].min() - 0.1, traj_np[:, :, 0].max() + 0.1)
    ax.set_ylim(traj_np[:, :, 1].min() - 0.1, traj_np[:, :, 1].max() + 0.1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    scat = ax.scatter([], [], c='blue', s=30, alpha=0.7)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        scat.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return scat, time_text

    def update(frame):
        current_points = traj_np[frame]
        scat.set_offsets(current_points)
        time_text.set_text(f'Time t={frame/(len(traj_np)-1):.2f}')
        scat.set_color('red' if frame == len(traj_np) - 1 else 'blue')
        return scat, time_text

    ani = animation.FuncAnimation(fig, update, frames=len(traj_np),
                                  init_func=init, blit=True, interval=50)
    # Fallback to Pillow if ffmpeg not available
    writer = animation.FFMpegWriter(fps=20) if output_file.endswith('.mp4') else animation.PillowWriter(fps=20)
    ani.save(output_file, writer=writer)
    plt.close(fig)
    print(f"Animation saved to {output_file}")

def plot_comparisons(original_x, traj, pred_order, model_len, gt_order, gt_len, output_file):
    x0_norm = traj[0].cpu().numpy()
    x1_pred = traj[-1].cpu().numpy()
    orig = original_x.cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    axes[0].set_title("Shape Space Flow")
    for i in range(x0_norm.shape[0]):
        path = traj[:, i, :].cpu().numpy()
        axes[0].plot(path[:, 0], path[:, 1], alpha=0.2, color='gray', linewidth=0.5)
    axes[0].scatter(x0_norm[:, 0], x0_norm[:, 1], c='green', label='t=0', s=20)
    axes[0].scatter(x1_pred[:, 0], x1_pred[:, 1], c='red', label='t=1', s=20)
    axes[0].set_aspect('equal')
    axes[0].legend()

    gap_str = ""
    if gt_len is not None and gt_len > 0:
        gap = ((model_len - gt_len) / gt_len) * 100
        gap_str = f"\nGap: {gap:.2f}%"

    axes[1].set_title(f"Model Prediction\nLen: {model_len:.4f}{gap_str}")
    axes[1].scatter(orig[:, 0], orig[:, 1], c='blue', s=30, zorder=5)
    ordered_orig = orig[pred_order.cpu().numpy()]
    lines = [[tuple(ordered_orig[i]), tuple(ordered_orig[(i+1)%len(ordered_orig)])]
             for i in range(len(ordered_orig))]
    lc = mc.LineCollection(lines, colors='blue', linewidths=1.5)
    axes[1].add_collection(lc)
    axes[1].set_aspect('equal')

    if gt_order is not None:
        axes[2].set_title(f"Ground Truth\nLen: {gt_len:.4f}")
        axes[2].scatter(orig[:, 0], orig[:, 1], c='green', s=30, zorder=5)
        gt_orig = orig[gt_order.cpu().numpy()]
        gt_lines = [[tuple(gt_orig[i]), tuple(gt_orig[(i+1)%len(gt_orig)])]
                    for i in range(len(gt_orig))]
        lc_gt = mc.LineCollection(gt_lines, colors='green', linewidths=1.5)
        axes[2].add_collection(lc_gt)
    else:
        axes[2].set_title("Ground Truth Not Available")
    axes[2].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Comparison plot saved to {output_file}")

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    # Default path pointed to your RoPE model
    parser.add_argument('--model_path', type=str, default='/home/benjamin.fri/PycharmProjects/tsp_fm/checkpoints/kendall_ROPE_02/best_model.pt')
    parser.add_argument('--input_file', type=str, default='/home/benjamin.fri/PycharmProjects/tsp_fm/data/processed_data_geom_val.pt')
    parser.add_argument('--gpu_id', type=int, default=5)

    # Defaults (used only if checkpoint doesn't have config)
    parser.add_argument('--num_points', type=int, default=50)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # A. Load Model & Config
    print(f"Loading Model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

    # Check if the checkpoint contains hyperparameters
    if isinstance(checkpoint, dict) and 'args' in checkpoint:
        print("Found hyperparameters in checkpoint. Initializing model from config.")
        model_args = argparse.Namespace(**checkpoint['args'])
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("WARNING: No 'args' found. Using default script arguments.")
        state_dict = checkpoint['model_state_dict']
        model_args = args
    else:
        state_dict = checkpoint
        print("WARNING: Raw state dict loaded. Using default script arguments.")
        model_args = args

    # B. Initialize Geometry
    n_points = getattr(model_args, 'num_points', 50)
    geo = GeometryProvider(n_points)
    interpolant_name = getattr(model_args, 'interpolant', 'kendall')

    # C. Initialize Model (Handling RoPE vs Concat)
    model_type = getattr(model_args, 'model_type', 'concat') # Default to concat if unknown

    # --- CHANGED: Logic to select the correct model architecture ---
    if model_type == 'rope':
        print(f"Detected Model Type: RoPE (Rotary Positional Embeddings)")
        model = RoPEVectorFieldModel(model_args).to(device)
    else:
        print(f"Detected Model Type: Standard (Concatenation)")
        model = VectorFieldModel(model_args).to(device)
    # -------------------------------------------------------------

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"\nERROR Loading state dict: {e}")
        print("Tip: If you are seeing size mismatches, ensure the training arguments (embed_dim, etc.) match exactly.")
        sys.exit(1)

    model.eval()
    print(f"Model Loaded. Architecture: {model_args.embed_dim} dim, {model_args.num_layers} layers.")

    # D. Load Data & Extract GT
    gt_order = None
    gt_len = None
    x0_raw = None

    if args.input_file and os.path.exists(args.input_file):
        print(f"Loading data from {args.input_file}...")
        data_obj = torch.load(args.input_file, weights_only=False)
        if isinstance(data_obj, list): sample = data_obj[1]
        else: sample = data_obj

        if isinstance(sample, dict):
            x0_raw = sample['points']
            if 'path' in sample: gt_order = sample['path']
        elif isinstance(sample, (list, tuple)):
            x0_raw = sample[0]
            if len(sample) > 1: gt_order = sample[1]
        else:
            x0_raw = sample

        if not torch.is_tensor(x0_raw): x0_raw = torch.tensor(x0_raw, dtype=torch.float32)
        if gt_order is not None and not torch.is_tensor(gt_order): gt_order = torch.tensor(gt_order, dtype=torch.long)
        if x0_raw.dim() == 3: x0_raw = x0_raw[0]
        if gt_order is not None and gt_order.dim() > 1: gt_order = gt_order[0]
    else:
        print("No input file provided or file not found. Generating random data.")
        x0_raw = torch.rand(n_points, 2)

    x0_raw = x0_raw.to(device)

    # E. Calculate GT Length
    if gt_order is not None:
        gt_len = calculate_tour_length(x0_raw, gt_order)

    # F. Inference
    x0_norm = normalize_and_center(x0_raw)
    x0_input = x0_norm.unsqueeze(0)

    print(f"Running flow matching (Steps=100, Interpolant={interpolant_name})...")
    use_geo = geo if interpolant_name == 'kendall' else None
    traj = solve_flow(model, x0_input, geometry=use_geo, steps=100, device=device)

    # G. Evaluation
    x1_pred = traj[-1]
    pred_order = get_angular_order(x1_pred)
    model_len = calculate_tour_length(x0_raw, pred_order)

    print(f"\n{'='*30}")
    print(f"Model Length: {model_len:.5f}")
    if gt_len is not None:
        print(f"GT Length:    {gt_len:.5f}")
        print(f"Gap:          {((model_len - gt_len)/gt_len)*100:.2f}%")
    else:
        print("GT Length:    N/A")
    print(f"{'='*30}\n")

    plot_comparisons(x0_raw, traj, pred_order, model_len, gt_order, gt_len, "tsp_comparison.png")
    save_flow_animation(x0_raw, traj, "tsp_flow.gif")

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    main()