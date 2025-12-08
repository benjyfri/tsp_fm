import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import argparse

# --- Environment Setup (Must match your script) ---
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'

# Ensure we can find the src module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dataset import load_data
from src.interpolants import KendallInterpolant
from src.geometry import GeometryProvider

def visualize_interpolation(args):
    # 1. Setup Device & Types
    # Using CPU for visualization is usually sufficient and easier for Matplotlib interaction
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu_only else 'cpu')
    torch.set_default_dtype(torch.float32)

    print(f"Using device: {device}")

    # 2. Load Data
    print(f"Loading data from {args.data_path}...")
    # We load to CPU first.
    # --- CHANGE: Load paths as well ---
    x0_all, x1_all, paths_all, _ = load_data(args.data_path, torch.device('cpu'))

    # Select specific sample
    if args.sample_idx >= len(x0_all):
        raise ValueError(f"Sample index {args.sample_idx} out of bounds (Size: {len(x0_all)})")

    # Extract single sample (N, 2)
    x0_sample = x0_all[args.sample_idx].to(device)
    x1_sample = x1_all[args.sample_idx].to(device)
    # --- CHANGE: Extract path indices (keep on CPU for numpy indexing) ---
    path_indices = paths_all[args.sample_idx].numpy()
    N = x0_sample.shape[0]

    print(f"Visualizing Sample {args.sample_idx} with {N} cities.")

    # 3. Setup Geometry & Interpolant
    geo = GeometryProvider(N)
    interpolant = KendallInterpolant(geo)

    # 4. Prepare 'Batch-as-Time' for smooth animation
    T = args.num_frames

    # Create time steps [0, ..., 1]
    t_steps = torch.linspace(0, 1, T, device=device)

    # Expand x0, x1 to (T, N, 2)
    x0_batch = x0_sample.unsqueeze(0).repeat(T, 1, 1) # (T, N, 2)
    x1_batch = x1_sample.unsqueeze(0).repeat(T, 1, 1) # (T, N, 2)

    # # make visualization clearer
    # x0_batch = x0_batch * 5
    # x1_batch = x1_batch * 5

    # 5. Precompute Geometry
    print("Precomputing geodesic parameters...")
    precomputed22 = interpolant.precompute(x0_batch, x0_batch)
    precomputed = interpolant.precompute(x0_batch, x1_batch)

    # 6. Sample Trajectories
    print("Generating trajectories...")
    with torch.no_grad():
        theta_geo = precomputed['theta_geo']
        log_x1_x0 = precomputed['log_x1_x0']
        small_angle_mask = precomputed['small_angle_mask']

        t_view = t_steps.view(T, 1, 1)
        theta_view = theta_geo.view(T, 1, 1)

        # Exact logic from your KendallInterpolant.sample
        angle = t_view * theta_view
        sin_angle = torch.sin(angle)
        cos_angle = torch.cos(angle)

        xt = (x0_batch * cos_angle +
              (log_x1_x0 / (theta_view + 1e-8)) * sin_angle)

        # Apply mask fallback
        small_mask_view = small_angle_mask.view(T, 1, 1)
        if small_mask_view.any():
            lin_xt = (1 - t_view) * x0_batch + t_view * x1_batch
            xt = torch.where(small_mask_view, lin_xt, xt)

    # xt is now (T, N, 2) - The exact path of the points over time
    trajectories = xt.cpu().numpy()

    # 7. Create Animation
    print("Building animation...")
    # --- CHANGE: Increased figure size significantly ---
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot formatting
    ax.set_xlim(-1.5, 1.5) # Assuming normalized data
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Colors for points (gradient to see rotation/matching)
    colors = plt.cm.viridis(np.linspace(0, 1, N))

    # Initial positions used for setting up plots
    current_pos_0 = trajectories[0]

    # --- CHANGE: Add lines representing the tour ---
    # Reorder points according to the path indices
    ordered_points_0 = current_pos_0[path_indices]
    # Close the loop by appending the first point to the end
    closed_points_0 = np.vstack([ordered_points_0, ordered_points_0[0]])
    # Plot the line (unpack with comma to get the Line2D object)
    line, = ax.plot(closed_points_0[:, 0], closed_points_0[:, 1],
                    color='gray', alpha=0.6, linewidth=2, zorder=1)

    # Initial scatter (increased zorder to sit on top of lines)
    scat = ax.scatter(current_pos_0[:, 0], current_pos_0[:, 1],
                      c=colors, s=150, edgecolors='k', zorder=2)

    # Title
    title = ax.text(0.5, 1.02, f"Kendall Interpolation (t=0.00)",
                    bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                    transform=ax.transAxes, ha="center", fontsize=16)

    # Optional: Draw the target circle faint in background
    target = x1_sample.cpu().numpy()
    # Also reorder target for visualization
    target_ordered = target[path_indices]
    target_closed = np.vstack([target_ordered, target_ordered[0]])
    ax.plot(target_closed[:, 0], target_closed[:, 1], color='green', alpha=0.1, linewidth=1, zorder=0)
    ax.scatter(target[:, 0], target[:, 1], c=colors, s=50, alpha=0.2, marker='x', zorder=0)


    def update(frame):
        # Update point positions
        current_pos = trajectories[frame]
        scat.set_offsets(current_pos)

        # --- CHANGE: Update line positions ---
        # Reorder based on path indices
        ordered_points = current_pos[path_indices]
        # Close loop
        closed_points = np.vstack([ordered_points, ordered_points[0]])
        # Update line data
        line.set_data(closed_points[:, 0], closed_points[:, 1])

        # Update title
        t_val = t_steps[frame].item()
        title.set_text(f"Kendall Interpolation (t={t_val:.2f})")

        # --- CHANGE: Return line object for blitting ---
        return scat, title, line

    # Increased interval slightly for smoother viewing of larger image
    ani = animation.FuncAnimation(fig, update, frames=T, interval=60, blit=True)

    # Save
    save_path = f"kendall_vis_large_sample_{args.sample_idx}.gif"
    print(f"Saving animation to {save_path}...")
    # Increased FPS for smoother video
    ani.save(save_path, writer=PillowWriter(fps=25))
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Paths (Defaults match your context)
    parser.add_argument('--data_path', type=str,
                        default='/home/benjamin.fri/PycharmProjects/tsp_fm/data/processed_data_geom_val.pt')

    parser.add_argument('--sample_idx', type=int, default=0, help="Index of the sample to visualize")
    parser.add_argument('--num_frames', type=int, default=90, help="Number of frames in animation")
    parser.add_argument('--cpu_only', action='store_true', help="Force CPU usage")

    args = parser.parse_args()

    visualize_interpolation(args)