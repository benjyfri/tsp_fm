import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import random

def sample_geodesic_np(x0, x1, theta, t, eps=1e-6):
    """
    [NEW] Numpy version of the geodesic sampling function.
    Computes the interpolated shape X(t).
    """
    # t is a scalar
    if theta < eps:
        # Fallback to linear interpolation if angle is too small
        return (1 - t) * x0 + t * x1

    a = (1 - t) * theta
    b = t * theta
    sin_theta = np.sin(theta)

    xt = (np.sin(a) / sin_theta) * x0 + (np.sin(b) / sin_theta) * x1
    return xt

def draw_shape(ax, shape_coords, path_indices, node_cmap,
               line_style='-', line_color='black', line_width=1.0,
               alpha=1.0, zorder=2, point_size=40):
    """
    [NEW] Helper function to draw a single shape (nodes + path) on an axis.
    """
    N = len(path_indices)

    for i in range(N):
        start_node_idx = path_indices[i]
        end_node_idx = path_indices[(i + 1) % N]
        color = node_cmap(start_node_idx % 20)

        # Draw path edge
        ax.plot(
            [shape_coords[start_node_idx, 0], shape_coords[end_node_idx, 0]],
            [shape_coords[start_node_idx, 1], shape_coords[end_node_idx, 1]],
            linestyle=line_style, color=line_color, alpha=alpha,
            linewidth=line_width, zorder=zorder
        )

        # Draw node
        ax.scatter(
            shape_coords[start_node_idx, 0],
            shape_coords[start_node_idx, 1],
            s=point_size, color=color, edgecolors='k',
            alpha=alpha, zorder=zorder+1
        )

    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(labelsize=8)

def plot_kendall_tsp_sample(axes, entry, point_size=40, title_prefix="Sample"):
    """
    [NEW] Plot one TSP datapoint on three provided subplots (ax1, ax2, ax3):
    - Left: Start Shape (X0)
    - Middle: Target Shape (X1)
    - Right: Geodesic Interpolation Path
    """
    ax1, ax2, ax3 = axes

    x0 = entry['points']
    x1 = entry['circle']
    path = entry['path']
    theta = entry['theta']

    N = len(x0)
    cmap = plt.get_cmap('tab20')

    # --- 1. Start Shape (X0) ---
    draw_shape(ax1, x0, path, cmap, line_color='black', point_size=point_size, zorder=2)
    ax1.set_title(f"{title_prefix} - Start Shape (X0)", fontsize=12)

    # --- 2. Target Shape (X1) ---
    draw_shape(ax2, x1, path, cmap, line_color='black', point_size=point_size, zorder=2)
    ax2.set_title(f"{title_prefix} - Target Shape (X1)", fontsize=12)

    # --- 3. Geodesic Path Overlay ---
    # Sample intermediate shapes
    xt_33 = sample_geodesic_np(x0, x1, theta, 0.33)
    xt_66 = sample_geodesic_np(x0, x1, theta, 0.66)

    # Plot all 4 shapes overlaid
    # t=0 (Start)
    draw_shape(ax3, x0, path, cmap, line_style='--', line_color='gray',
               alpha=0.7, zorder=1, point_size=point_size-10)
    # t=0.33
    draw_shape(ax3, xt_33, path, cmap, line_style=':', line_color='blue',
               alpha=0.7, zorder=2, point_size=point_size-10)
    # t=0.66
    draw_shape(ax3, xt_66, path, cmap, line_style=':', line_color='green',
               alpha=0.7, zorder=3, point_size=point_size-10)
    # t=1.0 (End)
    draw_shape(ax3, x1, path, cmap, line_style='-', line_color='black',
               alpha=1.0, zorder=4, point_size=point_size-10)

    ax3.set_title(f"{title_prefix} - Geodesic Path", fontsize=12)
    # Set axis limits based on the combined range of x0 and x1
    all_coords = np.vstack([x0, x1])
    min_val = all_coords.min() - 0.1
    max_val = all_coords.max() + 0.1
    ax3.set_xlim(min_val, max_val)
    ax3.set_ylim(min_val, max_val)


def main():
    parser = argparse.ArgumentParser(description="Visualize Kendall-processed TSP dataset.")
    parser.add_argument('--infile', default="processed_tsp_dataset_TSP50_train.pt",
                        help="Path to the processed .pt file (Kendall format).")
    parser.add_argument('--num-samples', type=int, default=3,
                        help="Number of random samples to plot.")
    args = parser.parse_args()

    try:
        entries = torch.load(args.infile, weights_only=False)
        print(f"Loaded {len(entries)} samples from {args.infile}")
    except FileNotFoundError:
        print(f"Error: File not found at {args.infile}")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    num_to_plot = min(args.num_samples, len(entries))
    if num_to_plot == 0:
        print("No entries to plot.")
        return

    # [MODIFIED] Create 3 columns per sample
    fig, axes = plt.subplots(num_to_plot, 3, figsize=(18, 6 * num_to_plot))
    if num_to_plot == 1:
        axes = np.expand_dims(axes, axis=0)

    print("\n" + "="*30)
    print(" Plotting and Verifying Samples")
    print("="*30)

    indices_to_plot = random.sample(range(len(entries)), num_to_plot)

    for i, sample_idx in enumerate(indices_to_plot):
        entry = entries[sample_idx]
        x0 = entry['points']
        x1 = entry['circle']

        print(f"\n--- Sample {i+1} (Index {sample_idx}) ---")
        print(f"  Theta: {entry['theta']:.6f} rad")
        print(f"  X0 Centroid: {np.mean(x0, axis=0)} (Should be ~0)")
        print(f"  X0 F-Norm:   {np.linalg.norm(x0, 'fro'):.6f} (Should be 1.0)")
        print(f"  X1 Centroid: {np.mean(x1, axis=0)} (Should be ~0)")
        print(f"  X1 F-Norm:   {np.linalg.norm(x1, 'fro'):.6f} (Should be 1.0)")

        plot_kendall_tsp_sample(
            axes[i],
            entry,
            title_prefix=f"Sample (Idx {sample_idx})"
        )

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()