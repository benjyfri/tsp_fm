import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import random
import os
import math
from matplotlib.animation import FuncAnimation

# --- Core Geodesic Function (Unchanged) ---

def sample_geodesic_np(x0, x1, theta, t, eps=1e-6):
    """
    Numpy version of the geodesic sampling function.
    Computes the interpolated shape X(t).
    """
    # t is a scalar
    if theta < eps:
        # Fallback to linear interpolation if angle is too small
        return (1 - t) * x0 + t * x1

    a = (1 - t) * theta
    b = t * theta
    sin_theta = np.sin(theta)

    if sin_theta == 0:
        return (1 - t) * x0 + t * x1

    xt = (np.sin(a) / sin_theta) * x0 + (np.sin(b) / sin_theta) * x1
    return xt

# --- Modern Drawing Functions ---

def _setup_modern_ax(ax, all_coords):
    """[Unchanged] Helper to style the axis for a clean, modern animation."""
    # Set tight limits
    min_val = all_coords.min() - 0.1
    max_val = all_coords.max() + 0.1
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # Remove all chart junk
    ax.set_aspect('equal', 'box')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for spine in ax.spines.values():
        spine.set_visible(False)

def draw_shape(ax, shape_coords, path_indices, node_cmap,
               line_style='-', line_color='black', line_width=2.5,
               alpha=1.0, zorder=2, point_size=50):
    """
    [Unchanged] Helper function to draw a single shape (nodes + path).
    Default styles are now bolder for a modern look.
    """
    N = len(path_indices)

    # Pre-calculate colors
    colors = [node_cmap(idx % 20) for idx in path_indices]

    # Draw all path edges first
    for i in range(N):
        start_node_idx = path_indices[i]
        end_node_idx = path_indices[(i + 1) % N]

        ax.plot(
            [shape_coords[start_node_idx, 0], shape_coords[end_node_idx, 0]],
            [shape_coords[start_node_idx, 1], shape_coords[end_node_idx, 1]],
            linestyle=line_style, color=line_color, alpha=alpha,
            linewidth=line_width, zorder=zorder
        )

    # Draw all nodes on top
    ax.scatter(
        shape_coords[path_indices, 0],
        shape_coords[path_indices, 1],
        s=point_size, color=colors, edgecolors='black',
        alpha=alpha, zorder=zorder+1, linewidths=1.0
    )

# --- [NEW] Turning Angle Annotation Function ---

def annotate_turning_angles(ax, shape_coords, path_indices, turning_angles):
    """
    [Unchanged] Annotates an *existing* plot axis with turning angles.
    Assumes turning_angles[i] corresponds to the angle at vertex path_indices[i].
    """
    N = len(path_indices)
    if turning_angles is None or len(turning_angles) != N:
        print(f"  Warning: Mismatch or missing turning angles. Skipping angle labels.")
        return

    # Get plot bounds to calculate a dynamic offset
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_offset = (xlim[1] - xlim[0]) * 0.015  # 1.5% of x-axis width
    y_offset = (ylim[1] - ylim[0]) * 0.015  # 1.5% of y-axis width

    for i in range(N):
        node_idx = path_indices[i]       # Get the node index (e.g., 3)
        coord = shape_coords[node_idx]     # Get coordinate for that node
        angle_deg = np.degrees(turning_angles[i]) # Get angle for this path step

        ax.text(
            coord[0] + x_offset,
            coord[1] + y_offset,
            f"{angle_deg:.1f}°",
            fontsize=8,
            color='black',
            ha='left',
            va='bottom',
            # Add a small white background for readability
            bbox=dict(facecolor='white', alpha=0.6, pad=0.1, boxstyle='round,pad=0.2')
        )

# --- New Animation Function (Unchanged) ---

def create_morph_animation(entry, sample_idx, out_file, duration_sec=2, fps=60, dpi=150, loop=True):
    """
    [Unchanged] Creates a full morphing animation for a single data entry
    and saves it to a file.
    """
    x0 = entry['points']
    x1 = entry['circle']
    path = entry['path']
    theta = entry['theta']
    N = len(x0)
    cmap = plt.get_cmap('tab20')

    # 1. Setup Figure
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    all_coords = np.vstack([x0, x1])

    # 2. Define Time Steps with Easing
    num_frames = int(duration_sec * fps)

    # Cosine easing (ease-in, ease-out) from 0 to 1
    t_linear = np.linspace(0, 1, num_frames)
    t_ease = 0.5 * (1 - np.cos(t_linear * math.pi))

    if loop:
        # Loop 0 -> 1 -> 0
        t_values = np.concatenate([t_ease, t_ease[::-1]])
    else:
        # Just 0 -> 1
        t_values = t_ease

    # 3. Define the Animation Update Function
    def update(t):
        ax.clear()
        _setup_modern_ax(ax, all_coords)

        # A. Draw "ghost" of start and end shapes
        draw_shape(ax, x0, path, cmap, line_style='--', line_color='gray',
                   alpha=0.1, zorder=1, point_size=20)
        draw_shape(ax, x1, path, cmap, line_style='--', line_color='black',
                   alpha=0.1, zorder=1, point_size=20)

        # B. Calculate and draw the intermediate shape
        xt = sample_geodesic_np(x0, x1, theta, t)
        draw_shape(ax, xt, path, cmap, line_color='#0077b6', # A nice blue
                   alpha=1.0, zorder=10)

        # C. Add a title (optional, but nice)
        ax.set_title(f"Sample {sample_idx} (t = {t:.3f})", fontsize=16)
        return (fig,)

    # 4. Create and Save Animation
    print(f"  Creating animation ({len(t_values)} frames) for sample {sample_idx}...")
    ani = FuncAnimation(
        fig,
        update,
        frames=t_values,
        blit=False
    )

    # Save the animation
    ani.save(out_file, writer='pillow', fps=fps, dpi=dpi)
    print(f"  Saved: {out_file}")
    plt.close(fig) # Close the figure to save memory

# --- [MODIFIED] Main Function ---

def main():
    parser = argparse.ArgumentParser(description="Create morphing animations from a Kendall-processed TSP dataset.")
    parser.add_argument('--infile', default="processed_tsp_dataset_TSP50_val_FULL.pt", # <-- Updated default
                        help="Path to the processed .pt file (Kendall format).")
    parser.add_argument('--num-samples', type=int, default=3,
                        help="Number of random samples to animate.")
    parser.add_argument('--out-dir', default="animations",
                        help="Directory to save output GIFs and PNGs.")
    parser.add_argument('--duration', type=float, default=2.0,
                        help="Duration of the 0->1 morph in seconds.")
    parser.add_argument('--fps', type=int, default=60,
                        help="Frames per second for the animation.")
    parser.add_argument('--dpi', type=int, default=150,
                        help="DPI for the output files (higher is sharper).")
    parser.add_argument('--no-loop', action='store_true',
                        help="Animate 0->1 only (default is 0->1->0 loop).")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        entries = torch.load(args.infile, weights_only=False)
        print(f"Loaded {len(entries)} samples from {args.infile}")
    except FileNotFoundError:
        print(f"Error: File not found at {args.infile}")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    num_to_animate = min(args.num_samples, len(entries))
    if num_to_animate == 0:
        print("No entries to animate.")
        return

    print("\n" + "="*30)
    print(f" Creating {num_to_animate} Animations & Plots")
    print("="*30)

    indices_to_animate = random.sample(range(len(entries)), num_to_animate)
    cmap = plt.get_cmap('tab20')

    for i, sample_idx in enumerate(indices_to_animate):
        entry = entries[sample_idx]
        x0 = entry['points']
        x1 = entry['circle']
        path = entry['path']

        # [MODIFIED] Check for BOTH turning_angles
        turning_angles_x0 = entry.get('turning_angles')
        turning_angles_x1 = entry.get('circle_turning_angles')

        if turning_angles_x0 is None or turning_angles_x1 is None:
            print(f"  Warning: 'turning_angles' or 'circle_turning_angles' not found in entry {sample_idx}. Skipping angle plot.")

        print(f"\n--- Sample {i+1}/{num_to_animate} (Index {sample_idx}) ---")
        print(f"  Theta: {entry['theta']:.6f} rad")
        print(f"  X0 F-Norm:   {np.linalg.norm(x0, 'fro'):.6f}")
        print(f"  X1 F-Norm:   {np.linalg.norm(x1, 'fro'):.6f}")

        # --- [MODIFIED] Create and save static turning angle plot for BOTH shapes ---
        if turning_angles_x0 is not None and turning_angles_x1 is not None:
            angle_plot_file = os.path.join(args.out_dir, f"sample_{sample_idx}_angles.png")
            print(f"  Creating turning angle plot for sample {sample_idx}...")

            # Setup figure with TWO subplots
            fig_static, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10), dpi=args.dpi)

            # Use same bounds for both plots
            all_coords = np.vstack([x0, x1])
            _setup_modern_ax(ax0, all_coords)
            _setup_modern_ax(ax1, all_coords)

            # --- Plot 1: Source Shape (X0) ---
            draw_shape(ax0, x0, path, cmap, line_color='#0077b6',
                       alpha=1.0, zorder=10, point_size=70)
            # Add X0 angle annotations
            annotate_turning_angles(ax0, x0, path, turning_angles_x0)
            ax0.set_title(f"Sample {sample_idx} - Initial Shape (X0) & Turning Angles", fontsize=16)

            # --- Plot 2: Target Shape (X1) ---
            draw_shape(ax1, x1, path, cmap, line_color='#d9534f', # A nice red
                       alpha=1.0, zorder=10, point_size=70)
            # Add X1 angle annotations
            annotate_turning_angles(ax1, x1, path, turning_angles_x1)
            ax1.set_title(f"Sample {sample_idx} - Target Circle (X1) & Turning Angles", fontsize=16)

            # Save the static plot
            try:
                fig_static.savefig(angle_plot_file, bbox_inches='tight', dpi=args.dpi)
                print(f"  Saved: {angle_plot_file}")
            except Exception as e:
                print(f"  Error saving static plot: {e}")

            plt.close(fig_static) # Close static fig to save memory
        # --- End of modified plot section ---

        # Create the animation (as before)
        out_file_gif = os.path.join(args.out_dir, f"sample_{sample_idx}.gif")

        create_morph_animation(
            entry,
            sample_idx,
            out_file_gif,
            duration_sec=args.duration,
            fps=args.fps,
            dpi=args.dpi,
            loop=not args.no_loop
        )

    print("\n" + "="*30)
    print("Animation and plot generation complete.")
    print(f"Files saved in: {os.path.abspath(args.out_dir)}")

if __name__ == '__main__':
    main()