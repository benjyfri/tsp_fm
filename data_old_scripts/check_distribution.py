import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from math import pi

# --- Global Constants for the Reference Function ---
NUM_ARC_SEGMENTS = 50


def euclidean_distance(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_cycle_distance(points, cycle_indices):
    """Calculates the total distance of a cycle defined by a sequence of point indices."""
    total_distance = 0
    num_points = len(cycle_indices)
    for i in range(num_points):
        p1_index = cycle_indices[i]
        # Connect the last point back to the first one (cycle)
        p2_index = cycle_indices[(i + 1) % num_points]
        total_distance += euclidean_distance(points[p1_index], points[p2_index])
    return total_distance


def solve_tsp(points):
    """
    Finds all possible cycle distances and identifies the shortest path and its length.
    Returns the shortest distance, the corresponding permutation, a list of ALL distances,
    and the list of unique distances.
    """
    n = len(points)
    point_indices = list(range(n))
    inner_indices = point_indices[1:]

    all_distances_list = []
    unique_distances_dict = {}
    min_distance = float('inf')
    shortest_path_indices = None

    # Iterate through all permutations of the inner points
    for perm in permutations(inner_indices):
        # A full cycle starts at 0, goes through the permutation, and returns to 0
        cycle = (0,) + perm
        distance = calculate_cycle_distance(points, cycle)

        all_distances_list.append(distance)
        unique_distances_dict[distance] = unique_distances_dict.get(distance, 0) + 1

        if distance < min_distance:
            min_distance = distance
            shortest_path_indices = cycle

    unique_distances = sorted(unique_distances_dict.keys())

    # Return list of all distances for statistics (mean/std)
    return min_distance, shortest_path_indices, all_distances_list, unique_distances


# --- Reference Function for Circle Mapping ---
def setup_path_data(points, path_indices, total_cycle_length, mode='shortest', num_arc_segments=NUM_ARC_SEGMENTS):
    """
    Sets up the new circle coordinates based on the shortest path.
    The core logic is preserved from the provided reference function.
    """
    points = np.asarray(points, dtype=np.float64)
    N = len(points)
    path_indices = np.asarray(path_indices, dtype=np.int64)

    if mode == 'shortest':
        p0_idx_in_path = np.where(path_indices == 0)[0][0]
        idx_clockwise_in_path = (p0_idx_in_path + 1) % N
        idx_ccw_in_path = (p0_idx_in_path - 1) % N
        node_idx_clockwise = path_indices[idx_clockwise_in_path]
        node_idx_ccw = path_indices[idx_ccw_in_path]
        x_cw = points[node_idx_clockwise, 0]
        x_ccw = points[node_idx_ccw, 0]
        if x_cw <= x_ccw:
            path_indices_list = path_indices.tolist()
            start_node = path_indices_list[0]
            rest_of_path = path_indices_list[1:]
            new_path_indices = [start_node] + rest_of_path[::-1]
            path_indices = np.array(new_path_indices, dtype=np.int64)

    path_coords = points[path_indices]

    # Circle mapping
    R = total_cycle_length / (2.0 * pi)

    # compute segment lengths (euclidean distances in original space) and central angles
    segment_lengths = []
    for i in range(N):
        start_idx = path_indices[i]
        end_idx = path_indices[(i + 1) % N]
        dist = np.linalg.norm(points[start_idx] - points[end_idx])
        segment_lengths.append(dist)
    segment_lengths = np.array(segment_lengths, dtype=np.float64)
    central_angles = segment_lengths / R

    # Calculate cumulative angles for circle placement
    start_angle = np.pi / 2.0
    cumulative_angles = [start_angle]
    current_angle = start_angle
    for ang in central_angles[:-1]:
        current_angle -= ang
        cumulative_angles.append(current_angle)
    cumulative_angles = np.array(cumulative_angles, dtype=np.float64)

    circle_x = R * np.cos(cumulative_angles)
    circle_y = R * np.sin(cumulative_angles)
    circle_coords = np.stack([circle_x, circle_y], axis=1)

    # Translate circle so P0 lines up with original P0 location
    P0_original = path_coords[0]
    P0_standard = circle_coords[0]
    translation_vector = P0_original - P0_standard

    target_coords = circle_coords + translation_vector

    return {
        'target_coords': target_coords.astype(np.float64),
        'R': float(R),
        'total_length': float(total_cycle_length),
        'path_indices': path_indices.astype(np.int64)
    }


# ----------------------------------------------------------------------
#                             MAIN SCRIPT
# ----------------------------------------------------------------------

# --- 1. Sample Random Points ---
np.random.seed(42)
N = 10# Number of points
original_points = np.random.uniform(-1, 1, size=(N, 2))

# --- 2. Solve TSP for Original Points ---
# Note the change: distances_original is now a list of ALL cycle lengths
shortest_dist_original, shortest_perm_original, all_distances_original, unique_distances_original = solve_tsp(
    original_points)
total_cycles = len(all_distances_original)

# --- Calculate Statistics for Original ---
mean_original = np.mean(all_distances_original)
std_original = np.std(all_distances_original)
num_unique_original = len(unique_distances_original)

# --- 3. Create Corresponding Circle Graph Placement ---
path_data = setup_path_data(
    points=original_points,
    path_indices=shortest_perm_original,
    total_cycle_length=shortest_dist_original
)
circle_points = path_data['target_coords']

# --- 4. Solve TSP for Circle Graph Placement ---
shortest_dist_circle, shortest_perm_circle, all_distances_circle, unique_distances_circle = solve_tsp(circle_points)

# --- Calculate Statistics for Circle ---
mean_circle = np.mean(all_distances_circle)
std_circle = np.std(all_distances_circle)
num_unique_circle = len(unique_distances_circle)

# --- 5. Plot Both Graphs Side by Side with Statistics ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

## 📊 Original Square Points TSP Cycles
ax0 = axes[0]
ax0.hist(all_distances_original, bins=np.linspace(min(all_distances_original), max(all_distances_original), 15),
         alpha=0.7, color='skyblue', edgecolor='black')
ax0.axvline(shortest_dist_original, color='red', linestyle='dashed', linewidth=2,
            label=f'Min: {shortest_dist_original:.3f}')
ax0.axvline(mean_original, color='green', linestyle='dotted', linewidth=2,
            label=f'Mean: {mean_original:.3f}')

# Add text statistics
stats_text_original = (
    f"Total Cycles: {total_cycles}\n"
    f"Unique Distances: {num_unique_original}\n"
    f"Mean ($\mu$): {mean_original:.3f}\n"
    f"Std ($\sigma$): {std_original:.3f}"
)
ax0.text(0.98, 0.98, stats_text_original,
         transform=ax0.transAxes,
         fontsize=10,
         verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

ax0.set_title(f'Setup 1: Original Square Points - Cycle Distances (N={N})')
ax0.set_xlabel('Cycle Distance')
ax0.set_ylabel('Frequency')
ax0.legend(loc='upper left')
ax0.grid(axis='y', linestyle='--')

## ⭕ Circle Placement Points TSP Cycles
ax1 = axes[1]
ax1.hist(all_distances_circle, bins=np.linspace(min(all_distances_circle), max(all_distances_circle), 15),
         alpha=0.7, color='lightcoral', edgecolor='black')
ax1.axvline(shortest_dist_circle, color='blue', linestyle='dashed', linewidth=2,
            label=f'Min: {shortest_dist_circle:.3f}')
ax1.axvline(mean_circle, color='darkorange', linestyle='dotted', linewidth=2,
            label=f'Mean: {mean_circle:.3f}')

# Add text statistics
stats_text_circle = (
    f"Total Cycles: {total_cycles}\n"
    f"Unique Distances: {num_unique_circle}\n"
    f"Mean ($\mu$): {mean_circle:.3f}\n"
    f"Std ($\sigma$): {std_circle:.3f}"
)
ax1.text(0.98, 0.98, stats_text_circle,
         transform=ax1.transAxes,
         fontsize=10,
         verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

ax1.set_title(f'Setup 2: Circle Placement Points - Cycle Distances (N={N})')
ax1.set_xlabel('Cycle Distance')
ax1.set_ylabel('Frequency')
ax1.legend(loc='upper left')
ax1.grid(axis='y', linestyle='--')

plt.tight_layout()
plt.show()

# --- Summary & Analysis Printout ---
print("\n" + "=" * 50)
print("             TSP Comparison Statistical Summary")
print("=" * 50)
print(f"Number of Points (N): {N}")
print(f"Total possible Hamiltonian Cycles: **{total_cycles}** (Since $N=5$, $(5-1)! = 24$ cycles)")
print("-" * 50)

print("1. Original Square Points:")
print(f"   Shortest Cycle Length (Min): **{shortest_dist_original:.6f}**")
print(f"   Mean Cycle Length ($\mu$): {mean_original:.6f}")
print(f"   Standard Deviation ($\sigma$): {std_original:.6f}")
print(f"   Number of Unique Distances: {num_unique_original}")
print(f"   Shortest Cycle Permutation (Indices): {shortest_perm_original}")
print("-" * 50)

print("2. Circle Placement Points:")
print(f"   Circle Circumference (based on original shortest): {path_data['total_length']:.6f}")
print(f"   Shortest Cycle Length (Min): **{shortest_dist_circle:.6f}**")
print(f"   Mean Cycle Length ($\mu$): {mean_circle:.6f}")
print(f"   Standard Deviation ($\sigma$): {std_circle:.6f}")
print(f"   Number of Unique Distances: {num_unique_circle}")
print("-" * 50)

print(f"\n**Statistical Insight:**")
print(f"The Circle setup ($\sigma_{{circle}} = {std_circle:.4f}$) generally shows a much **lower standard deviation**")
print(f"compared to the Random Square setup ($\sigma_{{original}} = {std_original:.4f}$).")
print("This indicates that placing points based on the shortest cycle path length, where segments become arc lengths,")
print("constrains the geometric freedom, leading to less variation across all possible cycle distances.")





