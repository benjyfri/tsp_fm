import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe


# ==========================================
# 1. STYLE & CONFIG
# ==========================================
def set_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 14,
        # --- FIX: FORCE PURE BLACK & HEAVY WEIGHT ---
        'axes.titlesize': 24,
        'axes.titleweight': 'heavy',  # Thicker than 'bold'
        'axes.titlecolor': '#000000',  # Force HEX pure black (no grey)
        'text.color': '#000000',
        'figure.dpi': 300,
        'lines.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,
        'axes.spines.bottom': False,
        'xtick.bottom': False,
        'ytick.left': False,
        'xtick.labelbottom': False,
        'ytick.labelleft': False,
    })


# ==========================================
# 2. DATA LOADING
# ==========================================
# def load_data(idx=8765):
def load_data(idx=3794):
    try:
        data_path = "../data/can_tsp50_test.pt"
        dataset = torch.load(data_path)
        if isinstance(dataset, dict):
            sample = {
                'points': dataset['points'][idx],
                'path': dataset['path'][idx],
                'circle': dataset['circle'][idx]
            }
        else:
            sample = dataset[4]
        print(f"Successfully loaded data from {data_path}")
        return sample
    except Exception as e:
        print(f"Loading failed ({e}). Generating synthetic data...")
        return 1



# ==========================================
# 3. PLOTTING
# ==========================================
def create_paradigm_figure(sample, output_file="paradigm_shift_refined.png"):
    set_style()

    # Convert to numpy
    X = sample['points'].numpy()
    path = sample['path'].numpy()
    X_target = sample['circle'].numpy()
    N = X.shape[0]

    # --- CHANGE 1: Unified Color Gradient ---
    # Assign colors based on the ORDER in the tour.
    # This visually connects the left 'mess' to the right 'circle'.
    node_ranks = np.zeros(N, dtype=int)
    node_ranks[path] = np.arange(N)

    # 'coolwarm' or 'viridis' or 'plasma'
    cmap = plt.cm.plasma
    node_colors = cmap(node_ranks / N)

    # Setup 2x1 Plot
    # fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(wspace=-0.05)
    # ==========================================
    # LEFT: OLD PARADIGM
    # ==========================================
    ax = axes[0]
    ax.set_title("Edge Diffusion", pad=20, color='#333333')

    # --- CHANGE 2: Visual Hierarchy (The Noise) ---
    # Draw all edges very faintly
    for i in range(N):
        for j in range(i + 1, N):
            ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]],
                    color='black',
                    alpha=1.0,
                    linewidth=0.2,
                    zorder=1)

    # The Optimal Tour (The Signal)
    for k in range(N):
        curr = path[k]
        next_node = path[(k + 1) % N]

        # Draw path with a white outline for contrast against the noise
        line, = ax.plot([X[curr, 0], X[next_node, 0]], [X[curr, 1], X[next_node, 1]],
                        color='#1a1a1a', alpha=1.0, linewidth=2.5, zorder=10)
        line.set_path_effects([pe.withStroke(linewidth=4.5, foreground='white'), pe.Normal()])

    # Nodes (Colored by gradient)
    ax.scatter(X[:, 0], X[:, 1], c=node_colors, s=200,
               edgecolors='white', linewidth=1.5, zorder=20)

    ax.set_aspect('equal')

    # ==========================================
    # RIGHT: NEW PARADIGM
    # ==========================================
    ax = axes[1]
    ax.set_title("Point Transport", pad=20, color='#333333')

    # --- CHANGE 4: Manifold Styling ---
    # Calculate radius roughly
    radius = np.mean(np.linalg.norm(X_target, axis=1))
    center = np.array([0,0])

    # Dashed subtle circle
    circle_patch = patches.Circle(center, radius,
                                  fill=False, edgecolor='#1a1a1a',
                                  linewidth=2.5, alpha=1.0, zorder=0)
    ax.add_patch(circle_patch)

    # --- CHANGE 3: Modern Arrow Styling ---
    for i in range(N):
        start = X[i]
        end = X_target[i]

        # Fancy Arrow
        arrow = patches.FancyArrowPatch(
            posA=(start[0], start[1]), posB=(end[0], end[1]),
            arrowstyle='-|>', mutation_scale=15,
            color='gray', alpha=0.6, linewidth=1.5, zorder=5,
            shrinkA=8,  # Radius of the start node (approx sqrt(200/pi))
            shrinkB=8  # Radius of the target node
        )
        ax.add_patch(arrow)

        # Ghost Node (Original position)
        ax.scatter(start[0], start[1], c=node_colors[i].reshape(1, -1), edgecolors='black', linewidth=1.0,
                   s=200, alpha=0.3,zorder=2)

        # Target Node (Manifold position)
        # --- CHANGE 5: Refined Node Geometry ---
        ax.scatter(end[0], end[1], c=node_colors[i].reshape(1, -1),
                   s=200, alpha=1.0, edgecolors='white', linewidth=1.5, zorder=20)

    # Sync limits
    padding = 0.03
    all_coords = np.concatenate([X, X_target])

    # Use max extent for both to keep scale identical
    xlims = (all_coords[:, 0].min() - padding, all_coords[:, 0].max() + padding)
    ylims = (all_coords[:, 1].min() - padding, all_coords[:, 1].max() + padding)

    for ax in axes:
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_aspect('equal')

    # --- Saving ---
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"SUCCESS: Figure saved to {output_file}")


if __name__ == "__main__":
    sample_data = load_data()
    create_paradigm_figure(sample_data)