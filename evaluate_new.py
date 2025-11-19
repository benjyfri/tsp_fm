import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib import animation
from torchdiffeq import odeint

# --- 1. SETUP GEOMSTATS ---
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
import geomstats.backend as gs
from geomstats.geometry.pre_shape import PreShapeSpace

# --- 2. MODEL IMPORT ---
from tsp_flow.models import KendallVectorFieldModel

# --- 3. GEOMETRY & TSP HELPERS ---

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
    """
    Calculate tour length using Euclidean distance between sequential points in 'order'.
    """
    # Ensure inputs are numpy for calculation
    if torch.is_tensor(order):
        order = order.cpu().numpy()
    if torch.is_tensor(points):
        points = points.cpu().numpy()

    # Reorder points
    ordered_points = points[order]

    # Calculate distances: dist(i, i+1) + dist(last, first)
    # np.roll with shift=-1 moves index 1 to 0, 2 to 1, ..., 0 to Last
    next_points = np.roll(ordered_points, shift=-1, axis=0)
    diffs = ordered_points - next_points
    distances = np.linalg.norm(diffs, axis=1)

    return np.sum(distances)

# --- 4. INFERENCE CORE ---

@torch.no_grad()
def solve_flow(model, x0, space, steps=100, device='cuda'):
    def ode_func(t, y):
        t_batch = t.expand(y.shape[0]).to(device)
        return model(y, t_batch, space)

    t_span = torch.linspace(0., 1., steps=steps).to(device)

    traj = odeint(
        ode_func,
        x0,
        t_span,
        method='dopri5'
    )
    return traj.squeeze(1)

# --- 5. VISUALIZATION ---

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

    writer = animation.FFMpegWriter(fps=20) if output_file.endswith('.mp4') else animation.PillowWriter(fps=20)
    ani.save(output_file, writer=writer)
    plt.close(fig)
    print(f"Animation saved to {output_file}")

def plot_comparisons(original_x, traj, pred_order, model_len, gt_order, gt_len, output_file):
    x0_norm = traj[0].cpu().numpy()
    x1_pred = traj[-1].cpu().numpy()
    orig = original_x.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: Flow
    axes[0].set_title("Shape Space Flow")
    for i in range(x0_norm.shape[0]):
        path = traj[:, i, :].cpu().numpy()
        axes[0].plot(path[:, 0], path[:, 1], alpha=0.2, color='gray', linewidth=0.5)
    axes[0].scatter(x0_norm[:, 0], x0_norm[:, 1], c='green', label='t=0', s=20)
    axes[0].scatter(x1_pred[:, 0], x1_pred[:, 1], c='red', label='t=1', s=20)
    axes[0].set_aspect('equal')
    axes[0].legend()

    # Panel 2: Model
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

    # Panel 3: Ground Truth
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

# --- 6. MAIN ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/home/benjamin.fri/PycharmProjects/tsp_fm/checkpoints_kendall_geomstats/best_model.pt')
    parser.add_argument('--input_file', type=str, default='/home/benjamin.fri/PycharmProjects/tsp_fm/data_old_scripts/processed_data_geom_val.pt')
    parser.add_argument('--num_points', type=int, default=50)
    parser.add_argument('--gpu_id', type=int, default=5)

    # Model Params (Match your trained model)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)

    args = parser.parse_args()

    # Device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # A. Load Model
    print("Loading Model...")
    space = PreShapeSpace(k_landmarks=args.num_points, ambient_dim=2)
    model = KendallVectorFieldModel(
        n_points=args.num_points,
        embed_dim=args.embed_dim,
        t_emb_dim=args.t_emb_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # B. Load Data & Extract GT
    gt_order = None
    gt_len = None

    if args.input_file and os.path.exists(args.input_file):
        print(f"Loading data from {args.input_file}...")
        data_obj = torch.load(args.input_file, weights_only=False)

        # 1. Handle Batch/List
        if isinstance(data_obj, list):
            sample = data_obj[1]
            print(f"  (Loaded item 0 from list of length {len(data_obj)})")
        else:
            sample = data_obj

        # 2. Parse Dictionary
        if isinstance(sample, dict):
            # Keys: 'points', 'circle', 'theta', 'path', 'edge_lengths'

            # Extract Points
            if 'points' in sample:
                x0_raw = sample['points']
            else:
                raise ValueError(f"Data dictionary missing 'points'. Keys: {sample.keys()}")

            # Extract GT Path
            if 'path' in sample:
                gt_order = sample['path']
                print("  -> Found GT path in sample['path'].")
            else:
                print(f"  -> Warning: 'path' key not found. GT comparison will be skipped. Keys: {sample.keys()}")

        elif isinstance(sample, (list, tuple)):
            # Fallback for tuple datasets
            x0_raw = sample[0]
            if len(sample) > 1: gt_order = sample[1]
        else:
            # Raw tensor
            x0_raw = sample

        # 3. Type Safety (Fix for 'numpy' has no attribute 'dim')
        if not torch.is_tensor(x0_raw):
            x0_raw = torch.tensor(x0_raw, dtype=torch.float32)

        if gt_order is not None and not torch.is_tensor(gt_order):
            gt_order = torch.tensor(gt_order, dtype=torch.long)

        # 4. Dimension Checks
        if x0_raw.dim() == 3:
            x0_raw = x0_raw[0] # Take first of batch

        if gt_order is not None:
            if gt_order.dim() > 1: gt_order = gt_order[0]

    else:
        print("Generating random data (No GT).")
        x0_raw = torch.rand(args.num_points, 2)

    x0_raw = x0_raw.to(device)

    # C. Calculate GT Length
    if gt_order is not None:
        gt_len = calculate_tour_length(x0_raw, gt_order)

    # D. Inference
    x0_norm = normalize_and_center(x0_raw)
    x0_input = x0_norm.unsqueeze(0)

    print("Running flow matching...")
    traj = solve_flow(model, x0_input, space, steps=100, device=device)

    # E. Evaluation
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