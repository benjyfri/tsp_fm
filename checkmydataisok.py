#!/usr/bin/env python3
"""
Visualizes the 'Angle FM' flow for a single TSP sample with enhanced aesthetics.

This script loads a processed .pt dataset, selects a single sample,
and animates the interpolation path defined by:
- Interpolating turning angles from X0 to X1 (shown in a bar chart).
- Using the *constant* edge lengths from X0.
- Reconstructing the polygon directly from interpolated angles and original lengths.
- [REVISION] Point cloud is centered at the first path-ordered node (p[path[0]]).

Features:
- Unique, consistent colors for each node.
- Standard white background for clarity.
- GIF output.
"""

import os
import argparse
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap # For gradient bars
# [FIXED] Removed deprecated import: 'from matplotlib.cm import get_cmap'

# ==============================================================================
# [CORE LOGIC]
# (Functions updated based on user request)
# ==============================================================================

def reconstruct_polygon_batch(batched_angles: torch.Tensor,
                              batched_lengths: torch.Tensor,
                              batched_path: torch.Tensor) -> torch.Tensor:
    """
    Differentiably reconstructs a batch of (N, 2) point clouds
    from their turning angles and edge lengths, provided in path-order.

    [REVISION] The returned point cloud is *not* centroid-centered.
    It is anchored at its first path-ordered point (p[path[0]]), which is
    placed at the origin (0,0).
    """
    B, N = batched_angles.shape
    device = batched_angles.device

    # 1. Calculate relative heading changes (turns) at each vertex
    rel_headings = batched_angles - math.pi  # (B, N)

    # 2. Set the first heading change to 0 (to define heading[0] as 0)
    rel_headings_shifted = torch.roll(rel_headings, shifts=1, dims=1)
    rel_headings_shifted[:, 0] = 0.0

    # 3. Compute absolute headings of each *edge vector*
    abs_headings = torch.cumsum(rel_headings_shifted, dim=1)  # (B, N)

    # 4. Create edge vectors (v_i) in path order
    vectors_x = batched_lengths * torch.cos(abs_headings)
    vectors_y = batched_lengths * torch.sin(abs_headings)
    vectors = torch.stack([vectors_x, vectors_y], dim=2)  # (B, N, 2)

    # 5. Reconstruct points in *path order* by cumsum
    points_path_order_shifted = torch.cumsum(vectors, dim=1)  # (B, N, 2)
    points_path_order = torch.roll(points_path_order_shifted, shifts=1, dims=1)
    # Set p[path[0]] = (0,0) - this is the ANCHOR point
    points_path_order[:, 0, :] = 0.0

    # 6. Scatter points back to *original 0..N-1 order*
    path_expanded = batched_path.unsqueeze(-1).expand(-1, -1, 2)  # (B, N, 2)
    points_original_order = torch.zeros_like(points_path_order)
    points_original_order.scatter_(dim=1, index=path_expanded, src=points_path_order)

    # 7. Center the point cloud [REMOVED PER REQUEST]
    # points_centered = points_original_order - torch.mean(points_original_order, dim=1, keepdim=True)
    # return points_centered

    # Return the anchor-centered points
    return points_original_order


# [FIX] Removed unused function pre_process_shape_torch
#
# def pre_process_shape_torch(X: torch.Tensor) -> torch.Tensor:
#    ... (function removed) ...


# ==============================================================================
# Animation Script
# ==============================================================================

def main(args):
    if not os.path.exists(args.data_file):
        print(f"Error: Data file not found at {args.data_file}")
        return

    print(f"Loading data from {args.data_file}...")
    try:
        # Load the list of dictionaries
        entries = torch.load(args.data_file, weights_only=False)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    if args.idx >= len(entries):
        print(f"Error: Index {args.idx} is out of bounds. File only has {len(entries)} samples.")
        return

    # --- 1. Load Data for the single sample ---
    print(f"Visualizing sample {args.idx}")
    entry = entries[args.idx]

    # Get ground-truth X0 and X1 (the points the model sees)
    # [FIX] These are loaded from the 'unprocessed' keys
    x0_gt = torch.from_numpy(entry['points_unprocessed'].astype(np.float32))
    x1_gt = torch.from_numpy(entry['circle_unprocessed'].astype(np.float32))

    # Get data required for Angle FM reconstruction
    angles_0 = torch.from_numpy(entry['turning_angles'].astype(np.float32))
    angles_1 = torch.from_numpy(entry['circle_turning_angles'].astype(np.float32))
    edge_lengths = torch.from_numpy(entry['edge_lengths'].astype(np.float32))
    path = torch.from_numpy(entry['path'].astype(np.int64))

    N = angles_0.shape[0]

    # --- 2. Batchify (add B=1 dimension) ---
    x0_gt = x0_gt.unsqueeze(0)
    x1_gt = x1_gt.unsqueeze(0)
    angles_0 = angles_0.unsqueeze(0)
    angles_1 = angles_1.unsqueeze(0)
    edge_lengths = edge_lengths.unsqueeze(0)
    path = path.unsqueeze(0)

    # --- 3. Setup the Plot ---
    # [Aesthetics] REVISION: Removed dark background, use default white
    # plt.style.use('dark_background')
    plt.rcParams.update({'font.size': 12})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), dpi=args.dpi) # Higher DPI for sharpness

    # [Aesthetics] Main title for the entire figure
    main_title_artist = fig.suptitle(f'Angle Flow Matching: Sample {args.idx}',
                                     color='black', fontsize=18, weight='bold') # REVISION: color='black'

    # --- Setup Ax1: Polygon Plot ---
    x0_np = x0_gt.squeeze(0).numpy()
    x1_np = x1_gt.squeeze(0).numpy()

    # [FIXED] Define all_points (the (2N, 2) array) *before* it is used
    all_points = np.vstack((x0_np, x1_np))

    # [Aesthetics] Colormap for unique node colors
    # [FIXED] Use plt.get_cmap() to avoid DeprecationWarning
    cmap = plt.get_cmap('viridis', N) # Use 'viridis' for distinct nodes up to N
    node_colors = [cmap(i) for i in range(N)]

    # Plot X0 and X1 as static references with node colors
    for i in range(N):
        color = node_colors[i]
        ax1.plot(x0_np[i, 0], x0_np[i, 1], 'o', color=color, alpha=0.2, markersize=8) # Faded X0 nodes
        ax1.plot(x1_np[i, 0], x1_np[i, 1], 'X', color=color, alpha=0.2, markersize=8) # Faded X1 nodes

    # Faded polygon lines for X0 and X1
    x0_path_order_np = x0_np[path.squeeze(0).numpy()]
    x0_closed = np.vstack((x0_path_order_np, x0_path_order_np[0]))
    ax1.plot(x0_closed[:, 0], x0_closed[:, 1], '-', color='gray', lw=1.5, alpha=0.3, label='X0 (Start)') # REVISION: alpha=0.3

    x1_path_order_np = x1_np[path.squeeze(0).numpy()]
    x1_closed = np.vstack((x1_path_order_np, x1_path_order_np[0]))
    ax1.plot(x1_closed[:, 0], x1_closed[:, 1], '-', color='lightgray', lw=1.5, alpha=0.3, label='X1 (Target)') # REVISION: alpha=0.3

    # Initialize the animated elements for the current polygon
    line, = ax1.plot([], [], '-', color='cyan', lw=3, alpha=0.9, zorder=5) # Current polygon line
    # REVISION: markeredgecolor='black'
    node_artists = [ax1.plot([], [], 'o', color=node_colors[i], markersize=6, markeredgecolor='black', markeredgewidth=0.5, zorder=6)[0] for i in range(N)]

    # Text for current time 't'
    time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, color='black', # REVISION: color='black'
                         fontsize=14, va='top', ha='left', weight='bold')

    ax1.set_title("Polygon Interpolation", fontsize=16, color='black') # REVISION: color='black'

    # [FIXED] Calculate padding and limits using the 'all_points' variable
    x_range = all_points[:,0].max() - all_points[:,0].min()
    y_range = all_points[:,1].max() - all_points[:,1].min()
    padding = max(x_range, y_range) * 0.15 # Use 15% padding

    min_x = all_points[:,0].min() - padding
    max_x = all_points[:,0].max() + padding
    min_y = all_points[:,1].min() - padding
    max_y = all_points[:,1].max() + padding

    ax1.set_xlim(min_x, max_x)
    ax1.set_ylim(min_y, max_y)
    ax1.set_aspect('equal', 'box')
    ax1.set_xlabel("X coordinate", color='black') # REVISION: color='black'
    ax1.set_ylabel("Y coordinate", color='black') # REVISION: color='black'
    ax1.tick_params(axis='x', colors='black') # REVISION: colors='black'
    ax1.tick_params(axis='y', colors='black') # REVISION: colors='black'
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='lower left', frameon=False, fontsize=10, labelcolor='black') # REVISION: labelcolor='black'


    # --- Setup Ax2: Angle Bar Chart ---
    angles_0_np = angles_0.squeeze(0).numpy()
    angles_1_np = angles_1.squeeze(0).numpy()
    x_indices = np.arange(N)

    # [Aesthetics] Plot target angles as a reference dashed line
    ax2.plot(x_indices, angles_1_np, '--', color='lime', lw=1.5, label='Target Angles (X1)', alpha=0.7)
    ax2.plot(x_indices, angles_0_np, '--', color='orange', lw=1.5, label='Start Angles (X0)', alpha=0.7)

    # Initialize the bars at t=0 with node-specific colors and gradient
    # A simple gradient from bottom (darker) to top (lighter)
    current_bars = []
    for k in range(N):
        # REVISION: edgecolor='black'
        bar = ax2.bar(x_indices[k], angles_0_np[k], color='gray', edgecolor='black', linewidth=0.5, alpha=0.8)[0]
        current_bars.append(bar)

    # Set fixed Y-limits for stability
    all_angles_np = np.concatenate((angles_0_np, angles_1_np))
    min_angle, max_angle = all_angles_np.min(), all_angles_np.max()
    ax2.set_ylim(min_angle * 0.9 - 0.1, max_angle * 1.1 + 0.1)

    ax2.set_xlabel("Vertex Index (Path Order)", color='black') # REVISION: color='black'
    ax2.set_ylabel("Angle (radians)", color='black') # REVISION: color='black'
    ax2.set_title("Turning Angles Interpolation", fontsize=16, color='black') # REVISION: color='black'
    ax2.legend(loc='upper right', frameon=False, fontsize=10, labelcolor='black') # REVISION: labelcolor='black'
    ax2.tick_params(axis='x', colors='black') # REVISION: colors='black'
    ax2.tick_params(axis='y', colors='black') # REVISION: colors='black'
    ax2.grid(True, linestyle=':', alpha=0.3)


    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle

    # --- 4. Define the Animation Function ---
    num_frames = args.frames

    def animate(i):
        t = i / (num_frames - 1)
        t_ = torch.tensor([[t]], dtype=angles_0.dtype) # Shape (B, 1)

        # --- Core 'Angle FM' path ---
        angles_t = (1 - t_) * angles_0 + t_ * angles_1
        lengths_t = edge_lengths

        # REVISION: reconstruct_polygon_batch now returns anchor-centered points
        unprocessed_xt = reconstruct_polygon_batch(angles_t, lengths_t, path)

        # [FIX] Do not apply Kendall (Frobenius) scaling.
        # We want to see the raw interpolation, which may change scale.
        xt = unprocessed_xt
        # --- End of path logic ---

        # === Update Ax1: Polygon ===
        xt_np = xt.squeeze(0).numpy()       # (N, 2)
        path_np = path.squeeze(0).numpy() # (N,)
        xt_path_order = xt_np[path_np]
        xt_closed = np.vstack((xt_path_order, xt_path_order[0]))

        line.set_data(xt_closed[:, 0], xt_closed[:, 1])
        for k in range(N):
            # [FIX] Wrap the x and y coordinates in lists
            node_artists[k].set_data([xt_np[k, 0]], [xt_np[k, 1]])

        time_text.set_text(f't = {t:.2f}')

        # === Update Ax2: Bar Chart ===
        angles_t_np = angles_t.squeeze(0).numpy()
        for k, bar in enumerate(current_bars):
            bar.set_height(angles_t_np[k])
            # [Aesthetics] Color bars based on node color
            bar.set_facecolor(node_colors[k])
            bar.set_edgecolor('black') # REVISION: edgecolor='black'

        # Return all artists that were modified
        return [line, time_text, *node_artists, *current_bars]

    # --- 5. Run and Save the Animation ---
    print(f"Generating {num_frames} frames at {args.fps} FPS. This may take a while...")
    print(f"Saving animation to {args.out}...")
    try:
        # Use blit=False for more reliable rendering with complex updates
        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=num_frames,
            interval=1000 / args.fps, # interval in ms
            blit=True # Important for complex updates
        )
        ani.save(args.out, writer='pillow', fps=args.fps)
        print("Done.")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Ensure 'pillow' (pip install pillow) is installed for GIF saving.")

    plt.close(fig) # Close the figure to free memory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Animate the Angle FM flow with enhanced aesthetics and save as GIF.")
    parser.add_argument('--data_file', type=str, default="data_old_scripts/processed_tsp_dataset_TSP50_val.pt",
                        help="Path to the processed .pt dataset file (e.g., processed_tsp_dataset_TSP50_train.pt)")
    parser.add_argument('--idx', type=int, default=0,
                        help="The index of the sample in the dataset to visualize.")
    parser.add_argument('--out', type=str, default="angle_fm_flow_slick.gif",
                        help="Output GIF file name.")
    parser.add_argument('--frames', type=int, default=100,
                        help="Number of frames to generate in the GIF.")
    parser.add_argument('--fps', type=int, default=20,
                        help="Frames per second for the output GIF.")
    parser.add_argument('--dpi', type=int, default=100,
                        help="Dots per inch for the animation output. Higher for sharper images (e.g., 150-200).")

    args = parser.parse_args()
    main(args)