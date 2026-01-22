#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
from scipy.interpolate import griddata
from matplotlib.patches import FancyArrowPatch

# ==========================================
# CONFIGURATION
# ==========================================
ORIGINAL_DATA_PATH = "../data/can_tsp50_val.pt"
CACHE_FILE = "sample_cachess.pt"
SAMPLE_IDX = 0
N_SUBPLOTS = 4
DPI = 300


# ==========================================
# 1. PROFESSIONAL STYLE SETUP
# ==========================================
def setup_professional_style():
    plt.style.use('seaborn-v0_8-paper')
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


# ==========================================
# 2. DATA LOADING
# ==========================================
def get_data_sample():
    if os.path.exists(CACHE_FILE):
        print(f"Loading from local cache: {CACHE_FILE}...")
        try:
            data = torch.load(CACHE_FILE, weights_only=False)
            return data['x0'], data['x1'], data['loop_idx'], data['N']
        except Exception as e:
            print(f"Cache load failed ({e}), falling back to main file.")

    print(f"Loading from original source: {ORIGINAL_DATA_PATH}...")
    if not os.path.exists(ORIGINAL_DATA_PATH):
        print(f"Error: {ORIGINAL_DATA_PATH} not found.")
        sys.exit(1)
    else:
        path_to_use = ORIGINAL_DATA_PATH

    try:
        dataset = torch.load(path_to_use, weights_only=False)
        x0 = dataset['points'][SAMPLE_IDX].to(torch.float64).numpy()
        x1 = dataset['circle'][SAMPLE_IDX].to(torch.float64).numpy()
        N = x0.shape[0]

        path_raw = dataset['path'][SAMPLE_IDX].numpy()
        s_perm = dataset['spectral_perm'][SAMPLE_IDX].numpy()

        inv_perm = np.argsort(s_perm)
        canonical_path = inv_perm[path_raw]
        loop_idx = np.append(canonical_path, canonical_path[0])

        cache_data = {'x0': x0, 'x1': x1, 'loop_idx': loop_idx, 'N': N}
        torch.save(cache_data, CACHE_FILE)
        return x0, x1, loop_idx, N

    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def get_bounds(x0, x1, pad=0.15):
    all_pts = np.concatenate([x0, x1], axis=0)
    return (all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad,
            all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)


# ==========================================
# 3. HELPER: VECTOR INTERPOLATION
# ==========================================
def get_interpolated_field(x0, x1, t, grid_res=30, pad=0.15):
    x_min, x_max, y_min, y_max = get_bounds(x0, x1, pad)
    gx = np.linspace(x_min, x_max, grid_res)
    gy = np.linspace(y_min, y_max, grid_res)
    GX, GY = np.meshgrid(gx, gy)

    xt = (1 - t) * x0 + t * x1
    V_total = x1 - x0

    vx_grid = griddata(xt, V_total[:, 0], (GX, GY), method='linear', fill_value=np.nan)
    vy_grid = griddata(xt, V_total[:, 1], (GX, GY), method='linear', fill_value=np.nan)

    mask = np.isnan(vx_grid)
    if np.any(mask):
        vx_grid[mask] = griddata(xt, V_total[:, 0], (GX, GY), method='nearest')[mask]
        vy_grid[mask] = griddata(xt, V_total[:, 1], (GX, GY), method='nearest')[mask]

    return GX, GY, vx_grid, vy_grid


# ==========================================
# 4. PLOTTING FUNCTIONS
# ==========================================
# def plot_3_combined(x0, x1, loop_idx, N, n_steps=4):
#     """
#     Plot 3: Combined View + Final GT Circle Panel
#     Colors derived from the angular position in the TARGET (x1) state.
#     """
#     # Steps for the flow part
#     t_steps = np.linspace(0, 0.75, n_steps)
#
#     # We add +1 for the final GT circle panel
#     total_panels = n_steps + 1
#
#     x_min, x_max, y_min, y_max = get_bounds(x0, x1)
#
#     # ========================================================
#     # COLOR LOGIC CHANGE:
#     # 1. Find centroid of the target circle (x1)
#     # 2. Calculate angle of every point relative to centroid
#     # 3. Map angle to color (using HSV for cyclic continuity)
#     # ========================================================
#     centroid_x1 = np.mean(x1, axis=0)
#     centroid_x1 = np.array([0,0])
#     # Calculate angles using arctan2 (returns -pi to pi)
#     angles = np.arctan2(x1[:, 1] - centroid_x1[1], x1[:, 0] - centroid_x1[0])
#
#     # Normalize angles to [0, 1] for the colormap
#     # (angles + pi) / 2pi maps [-pi, pi] to [0, 1]
#     norm_angles = (angles + np.pi) / (2 * np.pi)
#
#     # Generate colors using the cyclic 'hsv' map
#     colors = plt.get_cmap('vanimo')(norm_angles)
#
#     v_all = x1 - x0
#     max_mag = np.percentile(np.sqrt(np.sum(v_all ** 2, axis=1)), 95)
#
#     # Adjust figsize to accommodate the extra panel
#     fig, axes = plt.subplots(1, total_panels, figsize=(3 * total_panels, 3.5), constrained_layout=True)
#
#     for i, ax in enumerate(axes):
#         # Check if this is the regular flow sequence or the final GT panel
#         if i < n_steps:
#             # --- Interpolation Panels ---
#             t = t_steps[i]
#             xt = (1 - t) * x0 + t * x1
#             title_text = f"t={t:.2f}"
#
#             # Calculate and Plot Vector Field
#             GX, GY, vx, vy = get_interpolated_field(x0, x1, t)
#             magnitude = np.sqrt(vx ** 2 + vy ** 2)
#
#             ax.quiver(GX, GY, vx, vy, magnitude,
#                       cmap='plasma', alpha=0.9,
#                       pivot='mid',
#                       angles='xy', scale_units='xy', scale=max_mag * 15,
#                       width=0.005, headwidth=3, zorder=0)
#
#             ax.scatter(xt[:, 0], xt[:, 1], c=colors, s=60,
#                        edgecolor='black', linewidth=0.5, zorder=10)
#         else:
#             # --- Final Panel (Angular Sort) ---
#             xt = x1
#             title_text = "Solution"
#
#             # Use the calculated centroid
#             centroid = centroid_x1
#
#             # Draw Radial Lines (The "Spokes")
#             for pt in xt:
#                 ax.plot([centroid[0], pt[0]], [centroid[1], pt[1]],
#                         color='gray', linewidth=0.8, alpha=0.3, zorder=1)
#
#             # Draw Centroid
#             ax.scatter(centroid[0], centroid[1], c='black', marker='+', s=70, zorder=5)
#
#             # Add a curved arrow showing ordering
#             radius = np.mean(np.linalg.norm(xt - centroid, axis=1)) * 1.3
#             theta_start = np.radians(45)
#             theta_end = np.radians(135)
#             radius = radius / 3
#             # Adjust arrow positions based on centroid
#             posA = (centroid[0] + radius * np.cos(theta_start),
#                     centroid[1] + radius * np.sin(theta_start))
#             posB = (centroid[0] + radius * np.cos(theta_end),
#                     centroid[1] + radius * np.sin(theta_end))
#
#             arrow = FancyArrowPatch(
#                 posA=posA,
#                 posB=posB,
#                 connectionstyle="arc3,rad=.3",
#                 arrowstyle="Simple, tail_width=1.0, head_width=8, head_length=8",
#                 color='red',
#                 alpha=1.0,
#                 zorder=2
#             )
#             ax.add_patch(arrow)
#
#             ax.text(centroid[0], centroid[1] + radius , r"$\theta$",
#                     ha='center', va='bottom', fontsize=15, color='red')
#
#             # --- Points (Applied to all panels) ---
#             ax.scatter(xt[:, 0], xt[:, 1], c=colors, s=60,
#                        edgecolor='black', linewidth=0.5, zorder=10)
#
#         ax.set_title(title_text, fontweight='bold', color='#333333')
#         ax.set_aspect('equal')
#         ax.set_xlim(x_min, x_max)
#         ax.set_ylim(y_min, y_max)
#         ax.set_xticks([])
#         ax.set_yticks([])
#
#     plt.savefig("plot3_combined.png")
#     print("Generated plot3_combined.png")
#     plt.close()


def plot_3_combined(x0, x1, loop_idx, N, n_steps=4):
    """
    Plot 3: Combined View + Final GT Circle Panel
    Colors derived from the angular position in the TARGET (x1) state.
    """
    # Steps for the flow part
    t_steps = np.linspace(0, 0.75, n_steps)

    # We add +1 for the final GT circle panel
    total_panels = n_steps + 1

    x_min, x_max, y_min, y_max = get_bounds(x0, x1)

    # ========================================================
    # COLOR LOGIC
    # ========================================================
    centroid_x1 = np.array([0, 0])
    # Calculate angles using arctan2 (returns -pi to pi)
    angles = np.arctan2(x1[:, 1] - centroid_x1[1], x1[:, 0] - centroid_x1[0])

    # Normalize angles to [0, 1] for the colormap
    norm_angles = (angles + np.pi) / (2 * np.pi)

    # Generate colors using the cyclic 'hsv' map
    colors = plt.get_cmap('vanimo')(norm_angles)

    v_all = x1 - x0
    max_mag = np.percentile(np.sqrt(np.sum(v_all ** 2, axis=1)), 95)

    # Adjust figsize to accommodate the extra panel
    fig, axes = plt.subplots(1, total_panels, figsize=(3 * total_panels, 3.5), constrained_layout=True)

    for i, ax in enumerate(axes):
        # Check if this is the regular flow sequence or the final GT panel
        if i < n_steps:
            # --- Interpolation Panels ---
            t = t_steps[i]
            xt = (1 - t) * x0 + t * x1
            title_text = f"t = {t:.2f}"

            # Calculate and Plot Vector Field
            GX, GY, vx, vy = get_interpolated_field(x0, x1, t)
            magnitude = np.sqrt(vx ** 2 + vy ** 2)

            ax.quiver(GX, GY, vx, vy, magnitude,
                      cmap='plasma', alpha=1.0,
                      pivot='mid',
                      angles='xy', scale_units='xy', scale=max_mag * 15,
                      width=0.005, headwidth=3, zorder=0)

            ax.scatter(xt[:, 0], xt[:, 1], c=colors, s=60,
                       edgecolor='black', linewidth=0.5, zorder=10)

            # Hide ticks and grid for interpolation panels
            ax.set_xticks([])
            ax.set_yticks([])

        else:
            # --- Final Panel (Angular Sort) ---
            xt = x1
            title_text = f"t = 1.0"

            # Use the calculated centroid
            centroid = centroid_x1

            # 1. ADD GRID LINES HERE
            # We explicitly add the grid with specific styling
            ax.grid(True, which='major', color='#999999', linestyle='--', alpha=0.5, zorder=0)

            # Draw Radial Lines (The "Spokes")
            for pt in xt:
                ax.plot([centroid[0], pt[0]], [centroid[1], pt[1]],
                        color='gray', linewidth=0.8, alpha=0.3, zorder=1)

            # Draw Centroid
            ax.scatter(centroid[0], centroid[1], c='black', marker='+', s=100, zorder=5)

            # Add a curved arrow showing ordering
            radius = np.mean(np.linalg.norm(xt - centroid, axis=1)) * 1.3
            theta_start = np.radians(45)
            theta_end = np.radians(135)
            radius = radius / 3
            # Adjust arrow positions based on centroid
            posA = (centroid[0] + radius * np.cos(theta_start),
                    centroid[1] + radius * np.sin(theta_start))
            posB = (centroid[0] + radius * np.cos(theta_end),
                    centroid[1] + radius * np.sin(theta_end))

            arrow = FancyArrowPatch(
                posA=posA,
                posB=posB,
                connectionstyle="arc3,rad=.3",
                arrowstyle="Simple, tail_width=1.0, head_width=8, head_length=8",
                color='red',
                alpha=1.0,
                zorder=2
            )
            ax.add_patch(arrow)

            ax.text(centroid[0], centroid[1] + radius, r"$\theta$",
                    ha='center', va='bottom', fontsize=20, color='red')

            # --- Points (Applied to all panels) ---
            ax.scatter(xt[:, 0], xt[:, 1], c=colors, s=60,
                       edgecolor='black', linewidth=0.5, zorder=10)

            # 2. HIDE LABELS BUT KEEP TICKS (Required for grid to show)
            ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

        ax.set_title(title_text, fontsize=20,fontweight='bold', color='black')
        ax.set_aspect('equal')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    plt.savefig("plot3_combined.png")
    print("Generated plot3_combined.png")
    plt.close()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    setup_professional_style()

    # Load Data
    x0, x1, loop, N = get_data_sample()

    print("\n--- Generating Figures ---")
    plot_3_combined(x0, x1, loop, N, N_SUBPLOTS)

    print("\nDone. Created: plot3_combined.png")