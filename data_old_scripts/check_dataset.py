import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_tsp_pointcloud_and_circle(ax1, ax2, entry, point_size=60, title_prefix="Sample"):
    """
    Plot one TSP datapoint on provided subplots (ax1, ax2):
    - Left: original point cloud with optimal path
    - Right: circle-mapped version with arcs
    """
    pts = entry['points']
    circle = entry['circle']
    path = entry['path']
    arc_coords = entry['arc_coords']

    N = len(pts)
    cmap = plt.get_cmap('tab20')

    # --- LEFT: original layout ---
    for i in range(N):
        start_idx = path[i]
        end_idx = path[(i + 1) % N]
        color = cmap(start_idx % 20)

        # draw edge
        ax1.plot(
            [pts[start_idx, 0], pts[end_idx, 0]],
            [pts[start_idx, 1], pts[end_idx, 1]],
            '-', color='black', alpha=0.5, linewidth=0.75
        )

        # draw node + label at correct position
        ax1.scatter(pts[start_idx, 0], pts[start_idx, 1], s=point_size, color=color, edgecolors='k', zorder=3)
        ax1.text(
            pts[start_idx, 0] + 0.02, pts[start_idx, 1] + 0.02,
            str(start_idx), fontsize=10, fontweight='bold', color=color, zorder=4
        )

    ax1.set_aspect('equal', 'box')
    ax1.set_title(f"{title_prefix} - Original Layout", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.tick_params(labelsize=8)

    # --- RIGHT: circle layout ---
    num_arc_segments = len(arc_coords) // N
    for i in range(N):
        node_idx = path[i]
        color = cmap(node_idx % 20)
        arc_segment = arc_coords[i * num_arc_segments : (i + 1) * num_arc_segments]

        ax2.plot(arc_segment[:, 0], arc_segment[:, 1], '-', color='black', alpha=0.5, linewidth=0.75)
        ax2.scatter(circle[node_idx, 0], circle[node_idx, 1], s=point_size, color=color, marker='o', edgecolors='k', zorder=3)
        ax2.text(
            circle[node_idx, 0] + 0.02, circle[node_idx, 1] + 0.02,
            str(node_idx), fontsize=10, fontweight='bold', color=color, zorder=4
        )

    ax2.set_aspect('equal', 'box')
    ax2.set_title(f"{title_prefix} - Circle Mapping", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.tick_params(labelsize=8)


def plot_multiple_tsp_samples(entries, num_samples=3, point_size=60):
    """
    Plot multiple TSP samples in one big figure.
    Each sample has two columns: Original | Circle
    """
    num_samples = min(num_samples, len(entries))
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 5 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)
    import random
    for i in range(num_samples):
        ax1, ax2 = axes[i]
        idx = int(random.uniform(0,len(entries)))
        print(np.mean(entries[idx]["circle"], axis=0))
        print(np.linalg.norm(entries[idx]["circle"]-entries[idx]["translation_vector"], axis=1))
        print(f'+++++++++++++++++++++')
        plot_tsp_pointcloud_and_circle(ax1, ax2, entries[idx], point_size=point_size, title_prefix=f"Sample {i}")

    plt.tight_layout()
    plt.show()


# --- Example usage ---
entries = torch.load("processed_tsp_dataset_TSP50_test.pt", weights_only=False)
plot_multiple_tsp_samples(entries, num_samples=5)
