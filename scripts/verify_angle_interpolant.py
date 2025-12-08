import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, collections as mc
from pathlib import Path

# --- GEOMSTATS SETUP ---
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.interpolants import AngleInterpolant
from src.dataset import load_data

def save_flow_animation(traj, path_indices, output_file="flow_animation.gif"):
    """
    traj: (Frames, N, 2) - Points in original storage order
    path_indices: (N,) - The specific order to visit points to form the tour
    """
    print(f"Generating flow animation ({len(traj)} frames)...")
    traj_np = traj.cpu().numpy()

    # Ensure path indices are numpy and flat
    if torch.is_tensor(path_indices):
        path_indices = path_indices.cpu().numpy().flatten()

    fig, ax = plt.subplots(figsize=(6, 6))
    x_min, x_max = traj_np[:, :, 0].min(), traj_np[:, :, 0].max()
    y_min, y_max = traj_np[:, :, 1].min(), traj_np[:, :, 1].max()
    margin = 0.1
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_title("Angle Interpolant (Corrected Edges)")

    # 1. Scatter Plot (Nodes)
    scat = ax.scatter([], [], c='blue', s=40, alpha=0.8, edgecolors='k', linewidth=0.5, zorder=5)

    # 2. Line Collection (Edges)
    # initialized empty, will be updated every frame
    line_collection = mc.LineCollection([], colors='blue', linewidths=1.5, alpha=0.6, zorder=4)
    ax.add_collection(line_collection)

    # 3. Quiver (Velocity) - Optional, kept low visibility
    dummy = np.zeros(traj_np.shape[2])
    quiver = ax.quiver(dummy, dummy, dummy, dummy,
                       color='red', alpha=0.6, scale=1, scale_units='xy', width=0.005, zorder=3)

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, fontweight='bold')

    def init():
        scat.set_offsets(np.empty((0, 2)))
        line_collection.set_segments([])
        quiver.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return scat, line_collection, quiver, time_text

    def update(frame):
        current_points = traj_np[frame]

        # --- FIX: REORDER POINTS FOR LINES ---
        # The points are in storage order. We use path_indices to pick them
        # in the correct topological order to draw the polygon.
        ordered_points = current_points[path_indices]

        # Build segments: (p0, p1), (p1, p2), ... (pn, p0)
        segments = []
        for i in range(len(ordered_points)):
            segments.append((ordered_points[i], ordered_points[(i+1) % len(ordered_points)]))

        line_collection.set_segments(segments)
        scat.set_offsets(current_points)

        # Update Text
        t = frame / (len(traj_np) - 1)
        time_text.set_text(f'Time t={t:.2f}')

        # Colors change at end
        if frame == len(traj_np) - 1:
            scat.set_color('red')
            line_collection.set_colors('red')
        else:
            scat.set_color('cornflowerblue')
            line_collection.set_colors('cornflowerblue')

        return scat, line_collection, time_text

    # SLOW DOWN: interval=100ms (0.1s per frame)
    ani = animation.FuncAnimation(fig, update, frames=len(traj_np),
                                  init_func=init, blit=True, interval=100)
    try:
        # SLOW DOWN: fps=10
        writer = animation.PillowWriter(fps=10)
        ani.save(output_file, writer=writer)
        print(f"Animation saved to {output_file}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='/home/benjamin.fri/PycharmProjects/tsp_fm/data/processed_data_geom_val.pt')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--sample_idx', type=int, default=0)
    parser.add_argument('--frames', type=int, default=60)

    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # 1. Initialize Interpolant
    print("Initializing AngleInterpolant...")
    interpolant = AngleInterpolant()

    # 2. Load Data
    print(f"Loading data from {args.input_file}...")
    x0_all, x1_all, theta_all, gt_paths_all, _ = load_data(args.input_file, device, interpolant=None)

    # 3. Select Sample
    x0 = x0_all[args.sample_idx].unsqueeze(0).to(device)
    x1 = x1_all[args.sample_idx].unsqueeze(0).to(device)
    theta = theta_all[args.sample_idx].unsqueeze(0).to(device)
    path = gt_paths_all[args.sample_idx].unsqueeze(0).to(device)

    # 4. Precompute (PASSING PATH)
    print("Precomputing (Internal Sorting)...")
    pre_data = interpolant.precompute(x0, x1, theta, path=path)

    # Unpack
    l1 = pre_data['l1'].to(device)
    alpha1 = pre_data['alpha1'].to(device)
    phi1_start = pre_data['phi1_start'].to(device)
    d_l = pre_data['d_l'].to(device)
    d_alpha = pre_data['d_alpha'].to(device)
    d_phi = pre_data['d_phi'].to(device)
    # The precompute step stores path_indices in dict
    path_indices = pre_data['path_indices'].to(device)

    # 5. Generate Trajectory
    print(f"Generating {args.frames} frames...")
    traj = []
    t_steps = torch.linspace(0, 1, args.frames, device=device)

    for t_val in t_steps:
        # Sample returns unordered xt (matching x0)
        _, xt, _ = interpolant.sample(
            x0, x1, theta,
            l1, alpha1, phi1_start, d_l, d_alpha, d_phi, path_indices,
            device=device
        )
        traj.append(xt)

    traj = torch.stack(traj).squeeze(1) # (Frames, N, 2)

    # 6. Save (Passing the path indices for correct edge drawing)
    save_flow_animation(
        traj,
        gt_paths_all[args.sample_idx],
        f"verify_angle_slow_{args.sample_idx}.gif"
    )

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    main()