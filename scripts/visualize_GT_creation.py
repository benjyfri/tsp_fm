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
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.size': 14,
        'axes.titlesize': 30,
        'axes.titleweight': 'bold',
        'figure.dpi': 300,
        'savefig.dpi': 300,
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
# 2. GEOMETRY HELPERS
# ==========================================

def interpolate_arc(p1, p2, center=(0, 0), num_points=50):
    v1 = p1 - center
    v2 = p2 - center
    r = np.linalg.norm(v1)

    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])

    diff = angle2 - angle1
    if diff > np.pi:
        angle2 -= 2 * np.pi
    elif diff < -np.pi:
        angle2 += 2 * np.pi

    angles = np.linspace(angle1, angle2, num_points)
    x = center[0] + r * np.cos(angles)
    y = center[1] + r * np.sin(angles)
    return np.stack([x, y], axis=1)


# ==========================================
# 3. PLOTTING FUNCTION
# ==========================================

def create_figure(sample, output_file="icml_figure.png"):
    set_style()

    X = sample['points'].numpy()
    path = sample['path'].numpy()
    X_target = sample['circle'].numpy()
    N = X.shape[0]

    # --- COLOR STRATEGY ---
    node_ranks = np.zeros(N, dtype=int)
    node_ranks[path] = np.arange(N)
    cmap = plt.cm.vanimo
    node_colors = cmap(node_ranks / N)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)

    for ax in axes:
        # --- STYLE 1: SCIENTIFIC GRID ---
        # Turn on the grid, but push it to the back
        ax.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.7, zorder=0)
        ax.set_box_aspect(1)
        # Add a faint bounding box (spine) so the plot doesn't feel "floating"
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')  
            spine.set_linewidth(1.5)

    # ---------------------------------------------
    # HELPER: DRAW TEXT
    # ---------------------------------------------
    def draw_labels(ax, coords):
        for i in range(N):
            txt = ax.text(coords[i, 0], coords[i, 1], str(i),
                          fontsize=18, color='white',
                          ha='center', va='center', fontweight='bold', zorder=15)
            # Text Outline (Pop)
            txt.set_path_effects([
                pe.withStroke(linewidth=1.5, foreground='black'),
                pe.Normal()
            ])

    # ---------------------------------------------
    # PANEL A: INPUT
    # ---------------------------------------------
    ax = axes[0]
    ax.set_title("(A) Input TSP", pad=25)

    # Nodes
    ax.scatter(X[:, 0], X[:, 1], c=node_colors, s=800,
               edgecolors='black', linewidth=0.5, zorder=10)  # Increased linewidth for visibility

    draw_labels(ax, X)
    ax.set_aspect('equal')

    # ---------------------------------------------
    # PANEL B: GT PAIRING (EDGES POP HERE)
    # ---------------------------------------------
    ax = axes[1]
    ax.set_title("(B) Optimal Cycle", pad=25)

    for k in range(N):
        curr_idx = path[k]
        next_idx = path[(k + 1) % N]
        start = X[curr_idx]
        end = X[next_idx]
        edge_color = node_colors[curr_idx]

        # Draw Line
        line, = ax.plot([start[0], end[0]], [start[1], end[1]],
                        color=edge_color, alpha=1.0, linewidth=3.0, zorder=1)

        # --- EDGE POP EFFECT (CASING) ---
        line.set_path_effects([
            pe.withStroke(linewidth=4.0, foreground='black'),  # Black casing
            pe.Normal()
        ])

    # Nodes on top
    ax.scatter(X[:, 0], X[:, 1], c=node_colors, s=800,
               edgecolors='black', linewidth=0.5, zorder=10)

    draw_labels(ax, X)
    ax.set_aspect('equal')

    # ---------------------------------------------
    # PANEL C: TARGET MANIFOLD (EDGES POP HERE)
    # ---------------------------------------------
    ax = axes[2]
    ax.set_title("(C) Target Circle", pad=25)

    for k in range(N):
        curr_idx = path[k]
        next_idx = path[(k + 1) % N]
        edge_color = node_colors[curr_idx]
        arc_pts = interpolate_arc(X_target[curr_idx], X_target[next_idx])

        # Draw Arc
        line, = ax.plot(arc_pts[:, 0], arc_pts[:, 1],
                        color=edge_color, alpha=1.0, linewidth=3.0, zorder=1)

        # --- EDGE POP EFFECT (CASING) ---
        line.set_path_effects([
            pe.withStroke(linewidth=4.0, foreground='black'),
            pe.Normal()
        ])

    # Nodes
    ax.scatter(X_target[:, 0], X_target[:, 1], c=node_colors, s=800,
               edgecolors='black', linewidth=0.5, zorder=10)

    draw_labels(ax, X_target)

    # Limits
    ax.set_aspect('equal')
    R_mean = np.mean(np.linalg.norm(X_target, axis=1))
    lim = R_mean * 1.25
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    plt.savefig(output_file, bbox_inches='tight')
    print(f"SUCCESS: Saved figure with POP edges to {output_file}")
    plt.close()


# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    try:
        data_path = "../data/can_tsp20_test.pt"
        dataset = torch.load(data_path)
        if isinstance(dataset, dict):
            sample = {'points': dataset['points'][6], 'path': dataset['path'][6], 'circle': dataset['circle'][6]}
        else:
            sample = dataset[6]
        create_figure(sample)
    except Exception as e:
        print(f"Using synthetic data... ({e})")
        N = 20
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        path = np.arange(N)
        rng = np.random.default_rng(42)
        points = circle + rng.normal(0, 0.2, size=(N, 2))
        perm = rng.permutation(N)
        sample_synth = {
            'points': torch.tensor(points[perm]).float(),
            'path': torch.tensor(np.argsort(perm)).long(),
            'circle': torch.tensor(circle[perm]).float()
        }
        create_figure(sample_synth)