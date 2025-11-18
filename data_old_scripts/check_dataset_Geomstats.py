import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import random
import os
import math
from matplotlib.animation import FuncAnimation

# ============================================================================
# Math & Geometry Helpers
# ============================================================================

def sample_geodesic_np(x0, x1, theta, t, eps=1e-6):
    """
    Computes the interpolated shape X(t) along the geodesic on the sphere.
    Formula: (sin((1-t)theta)/sin(theta))*x0 + (sin(t*theta)/sin(theta))*x1
    """
    if theta < eps:
        return (1 - t) * x0 + t * x1

    sin_theta = np.sin(theta)

    # Prevent division by zero if theta is extremely close to 0 or pi
    if abs(sin_theta) < eps:
        return (1 - t) * x0 + t * x1

    a = (1 - t) * theta
    b = t * theta

    xt = (np.sin(a) / sin_theta) * x0 + (np.sin(b) / sin_theta) * x1
    return xt

def compute_turning_angles_np(points, path):
    """
    Calculates exterior turning angles for a closed polygon.
    Returns angles in radians.
    """
    # Reorder points according to path
    ordered_points = points[path]
    N = len(ordered_points)
    angles = np.zeros(N)

    for i in range(N):
        # Previous, Current, Next vertices
        prev_p = ordered_points[i - 1]
        curr_p = ordered_points[i]
        next_p = ordered_points[(i + 1) % N]

        # Vectors corresponding to incoming and outgoing edges
        vec_in = curr_p - prev_p
        vec_out = next_p - curr_p

        # Angle calculation using arctan2
        # We want the angle change, not the internal angle
        angle_in = np.arctan2(vec_in[1], vec_in[0])
        angle_out = np.arctan2(vec_out[1], vec_out[0])

        # Calculate turning angle (delta)
        turn = angle_out - angle_in

        # Normalize to (-pi, pi]
        turn = (turn + np.pi) % (2 * np.pi) - np.pi
        angles[i] = turn

    return angles

# ============================================================================
# Visualization Helpers
# ============================================================================

def _setup_modern_ax(ax, all_coords):
    """Helper to style the axis for a clean, modern animation."""
    # Determine limits with padding
    min_xy = all_coords.min()
    max_xy = all_coords.max()
    pad = (max_xy - min_xy) * 0.1

    ax.set_xlim(min_xy - pad, max_xy + pad)
    ax.set_ylim(min_xy - pad, max_xy + pad)

    ax.set_aspect('equal', 'box')
    ax.axis('off') # Turn off axis lines and labels entirely

def draw_shape(ax, shape_coords, path_indices, node_cmap,
               line_style='-', line_color='black', line_width=2.0,
               alpha=1.0, zorder=2, point_size=40):
    """Draws nodes and edges of the TSP tour."""
    N = len(path_indices)

    # Create cyclic path for plotting lines
    # We append the first point to the end to close the loop
    plot_indices = np.concatenate([path_indices, [path_indices[0]]])
    path_coords = shape_coords[plot_indices]

    # Draw edges
    ax.plot(path_coords[:, 0], path_coords[:, 1],
            linestyle=line_style, color=line_color, alpha=alpha,
            linewidth=line_width, zorder=zorder)

    # Draw nodes
    colors = [node_cmap(i % 20) for i in range(len(shape_coords))]
    # Only scatter the points that are in the path (which is all of them, but organized)
    # Using original indices for coloring consistency
    ax.scatter(
        shape_coords[:, 0], shape_coords[:, 1],
        s=point_size, c=colors, edgecolors='black',
        alpha=alpha, zorder=zorder+1, linewidths=0.8
    )

def annotate_turning_angles(ax, shape_coords, path_indices, turning_angles):
    """Annotates vertices with their turning angles."""
    N = len(path_indices)

    # Get plot bounds to calculate a dynamic offset
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_offset = (xlim[1] - xlim[0]) * 0.02
    y_offset = (ylim[1] - ylim[0]) * 0.02

    for i in range(N):
        node_idx = path_indices[i]
        coord = shape_coords[node_idx]
        angle_deg = np.degrees(turning_angles[i])

        ax.text(
            coord[0] + x_offset,
            coord[1] + y_offset,
            f"{angle_deg:.0f}°",
            fontsize=7,
            color='black',
            ha='left', va='bottom',
            bbox=dict(facecolor='white', alpha=0.7, pad=0.1, boxstyle='round,pad=0.1')
        )

# ============================================================================
# Animation Logic
# ============================================================================

def create_morph_animation(entry, sample_idx, out_file, duration_sec=2, fps=30, dpi=150, loop=True):
    x0 = entry['points']
    x1 = entry['circle']
    path = entry['path']
    theta = entry['theta']

    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    all_coords = np.vstack([x0, x1])
    cmap = plt.get_cmap('tab20')

    # Easing function
    num_frames = int(duration_sec * fps)
    t_linear = np.linspace(0, 1, num_frames)
    t_ease = 0.5 * (1 - np.cos(t_linear * math.pi))

    if loop:
        t_values = np.concatenate([t_ease, t_ease[::-1]])
    else:
        t_values = t_ease

    def update(t):
        ax.clear()
        _setup_modern_ax(ax, all_coords)

        # Ghost shapes
        draw_shape(ax, x0, path, cmap, line_style=':', line_color='gray', alpha=0.15, zorder=1)
        draw_shape(ax, x1, path, cmap, line_style=':', line_color='black', alpha=0.15, zorder=1)

        # Interpolated shape
        xt = sample_geodesic_np(x0, x1, theta, t)

        # Color changes from Blue (start) to Red (target)
        current_color = plt.cm.coolwarm(t)

        draw_shape(ax, xt, path, cmap, line_color=current_color, line_width=2.5, alpha=1.0, zorder=10)

        ax.set_title(f"Interpolation t={t:.2f}\nGeodesic $\\theta$={theta:.3f}", fontsize=12)
        return (fig,)

    print(f"  Rendering animation to {out_file}...")
    ani = FuncAnimation(fig, update, frames=t_values, blit=False)
    ani.save(out_file, writer='pillow', fps=fps, dpi=dpi)
    plt.close(fig)

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize Kendall Shape Space Morphing")
    parser.add_argument('--infile', type=str, default="geom_demo_val_N10.pt")
    parser.add_argument('--num-samples', type=int, default=3, help="Number of samples to visualize")
    parser.add_argument('--out-dir', default="viz_output", help="Output directory")
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    print(f"Loading {args.infile}...")
    try:
        dataset = torch.load(args.infile, weights_only=False)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Dataset contains {len(dataset)} samples.")

    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    for i, idx in enumerate(indices):
        entry = dataset[idx]

        # Extract Data
        x0 = entry['points']
        x1 = entry['circle']
        path = entry['path']
        theta_file = entry['theta']

        print(f"\n=== Processing Sample {idx} ({i+1}/{len(indices)}) ===")

        # ---------------------------------------------------------
        # 1. Alignment Verification (Kendall Check)
        # ---------------------------------------------------------
        # In pre-shape space, <x0, x1> = Tr(x0.T @ x1).
        # If optimally aligned, this dot product should be cos(theta).

        # Frobenius dot product
        dot_prod = np.trace(x0.T @ x1)

        # Clamp for safety in arccos
        dot_prod_clamped = np.clip(dot_prod, -1.0, 1.0)
        theta_check = np.arccos(dot_prod_clamped)

        # Check norms (Should be 1.0 in pre-shape space)
        norm_x0 = np.linalg.norm(x0, 'fro')
        norm_x1 = np.linalg.norm(x1, 'fro')

        print(f"  [Alignment Check]")
        print(f"   > Norm x0:     {norm_x0:.6f} (Should be ~1.0)")
        print(f"   > Norm x1:     {norm_x1:.6f} (Should be ~1.0)")
        print(f"   > File Theta:  {theta_file:.6f}")
        print(f"   > Calc Theta:  {theta_check:.6f} (from arccos(Tr(x0.T @ x1)))")

        if abs(theta_file - theta_check) > 1e-4:
            print("   WARNING: Significant discrepancy in theta. Alignment might be off.")
        else:
            print("   STATUS: Alignment looks correct.")

        # ---------------------------------------------------------
        # 2. Calculate Turning Angles (Since not in file)
        # ---------------------------------------------------------
        angles_x0 = compute_turning_angles_np(x0, path)
        angles_x1 = compute_turning_angles_np(x1, path)

        # ---------------------------------------------------------
        # 3. Generate Static Plot
        # ---------------------------------------------------------
        fig_static, (ax_L, ax_R) = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        all_coords = np.vstack([x0, x1])
        cmap = plt.get_cmap('tab20')

        _setup_modern_ax(ax_L, all_coords)
        _setup_modern_ax(ax_R, all_coords)

        # Left: Initial
        draw_shape(ax_L, x0, path, cmap, line_color='#0077b6', point_size=60)
        annotate_turning_angles(ax_L, x0, path, angles_x0)
        ax_L.set_title("Initial TSP Solution (Pre-shape)", fontsize=14, fontweight='bold')

        # Right: Target Circle
        draw_shape(ax_R, x1, path, cmap, line_color='#d9534f', point_size=60)
        annotate_turning_angles(ax_R, x1, path, angles_x1)
        ax_R.set_title(f"Target Circle (Aligned)\nGeodesic Dist: {theta_file:.4f}", fontsize=14, fontweight='bold')

        static_path = os.path.join(args.out_dir, f"sample_{idx}_static.png")
        plt.savefig(static_path, bbox_inches='tight')
        print(f"  Saved static plot: {static_path}")
        plt.close(fig_static)

        # ---------------------------------------------------------
        # 4. Generate Animation
        # ---------------------------------------------------------
        anim_path = os.path.join(args.out_dir, f"sample_{idx}_morph.gif")
        create_morph_animation(entry, idx, anim_path, fps=args.fps)

if __name__ == "__main__":
    main()