import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

# --- Configuration ---
N = 10  # Number of points
SEED = 40

# --- 1. Sample N points randomly in a 1x1 square ---
np.random.seed(SEED)
points = np.random.rand(N, 2)


# --- Helper Function for Distance Calculation ---
def calculate_distance(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# --- 2. Find, Calculate Lengths, and Sort all Hamiltonian Cycles ---

# List to store tuples: (cycle_length, cycle_path_indices)
cycle_data = []

# Get all permutations of the remaining N-1 points (1 to N-1)
cycle_permutations = permutations(range(1, N))

for perm in cycle_permutations:
    # A full cycle: [start] -> [perm] -> [start]
    cycle = [0] + list(perm) + [0]

    # Canonical check to avoid duplicates (e.g., [0, 1, 2, 0] vs [0, 2, 1, 0])
    inner_path = cycle[1:-1]
    if inner_path[0] < inner_path[-1]:

        # Calculate the length of this specific cycle
        total_length = 0

        for j in range(N):  # N edges in a cycle of N points
            p_start = points[cycle[j]]
            p_end = points[cycle[j + 1]]
            dist = calculate_distance(p_start, p_end)
            total_length += dist

        cycle_data.append((total_length, cycle))

# Sort the cycles based on their length (the first element of the tuple)
cycle_data.sort(key=lambda x: x[0])

# --- 3. Select the Shortest, Longest, and Median Cycles ---

# Total number of unique cycles (should be 12 for N=5)
num_unique_cycles = len(cycle_data)
# Index 0 is the shortest cycle
shortest_cycle = cycle_data[0]
# Index num_unique_cycles - 1 is the longest cycle
longest_cycle = cycle_data[-1]

# The median index for a list of even length L is typically (L/2) - 1 or L/2.
# For L=12, we choose index 5 (the 6th element) as the median.
median_index = num_unique_cycles // 2 - 1  # 12 // 2 - 1 = 5
median_cycle = cycle_data[median_index]

# The list of cycles to plot, along with a descriptive title
cycles_to_plot_info = [
    ("Shortest Cycle", shortest_cycle),
    ("Median Cycle", median_cycle),
    ("Longest Cycle", longest_cycle)
]

num_cycles_to_plot = len(cycles_to_plot_info)

# Set up a figure with 2 columns: Square Plot (left) and Circle Plot (right)
fig, axes = plt.subplots(num_cycles_to_plot, 2, figsize=(12, 4 * num_cycles_to_plot))

# --- 4. Plotting ---

# Loop through the selected cycles
for i, (title, (total_length, cycle)) in enumerate(cycles_to_plot_info):

    # --- Recalculate edge lengths (needed for the circle plot) ---
    edge_lengths = []
    for j in range(N):
        p_start = points[cycle[j]]
        p_end = points[cycle[j + 1]]
        edge_lengths.append(calculate_distance(p_start, p_end))

    # Path coordinates for plotting the square
    path_x = [points[j][0] for j in cycle]
    path_y = [points[j][1] for j in cycle]

    # -----------------------------------------------------------------
    # --- A. Plot the Hamiltonian Cycle on the 1x1 Square (Left Plot) ---
    # -----------------------------------------------------------------
    ax_square = axes[i, 0]
    ax_square.set_aspect('equal', adjustable='box')
    ax_square.set_title(f'{title} (Length: {total_length:.3f})')

    # Plot edges and points
    ax_square.plot(path_x, path_y, 'b-', label='Cycle Path')
    ax_square.plot(points[:, 0], points[:, 1], 'ko', markersize=5, zorder=5)
    ax_square.plot(points[cycle[0], 0], points[cycle[0], 1], 'ro', markersize=8, zorder=6, label='Start Point 0')

    # Add point labels
    for k, (x, y) in enumerate(points):
        ax_square.annotate(f'{k}', (x, y), textcoords="offset points", xytext=(5, 5), ha='center')

    # Set plot limits and labels
    ax_square.set_xlim(-0.05, 1.05)
    ax_square.set_ylim(-0.05, 1.05)
    ax_square.set_xlabel('X Coordinate')
    ax_square.set_ylabel('Y Coordinate')
    ax_square.grid(True, linestyle='--', alpha=0.6)

    # -------------------------------------------------------------
    # --- B. Plot the Edge Lengths on a Circle (Right Plot) ---
    # -------------------------------------------------------------
    ax_circle = axes[i, 1]
    ax_circle.set_aspect('equal', adjustable='box')
    ax_circle.set_title(f'{title} Mapping on Circle')

    # Calculate Radius R = C / (2 * pi)
    radius = total_length / (2 * np.pi)

    # Plot the full circle
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circ = radius * np.cos(theta)
    y_circ = radius * np.sin(theta)
    ax_circle.plot(x_circ, y_circ, 'k--')

    # Calculate angular positions for the points on the circle
    cumulative_length = 0
    angles = [np.pi / 2]  # Start at the top (pi/2 or 90 degrees)

    for length in edge_lengths:
        cumulative_length += length
        # Angle from the positive X-axis, moving clockwise (subtraction)
        angle = np.pi / 2 - (cumulative_length / radius)
        angles.append(angle)

    # Plot the points and label them
    for k, angle in enumerate(angles[:-1]):
        point_idx = cycle[k]

        # Coordinates on the circle
        px = radius * np.cos(angle)
        py = radius * np.sin(angle)

        # Plot the point
        color = 'r' if k == 0 else 'b'
        ax_circle.plot(px, py, color + 'o', markersize=8)

        # Add point label
        ax_circle.annotate(f'{point_idx}', (px, py), textcoords="offset points", xytext=(-5, 5), ha='right')

        # Draw the arc for the edge and label it
        if k < N:
            start_angle = angles[k]
            end_angle = angles[k + 1]

            # Create a small angular segment for the arc plot
            arc_angles = np.linspace(end_angle, start_angle, 50)
            arc_x = radius * np.cos(arc_angles)
            arc_y = radius * np.sin(arc_angles)
            ax_circle.plot(arc_x, arc_y, 'g-', linewidth=3, alpha=0.7)

            # Add edge length label
            mid_angle = (start_angle + end_angle) / 2
            label_radius = radius * 1.15
            label_x = label_radius * np.cos(mid_angle)
            label_y = label_radius * np.sin(mid_angle)

            ax_circle.annotate(f'{edge_lengths[k]:.2f}', (label_x, label_y),
                               ha='center', va='center', fontsize=9,
                               bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    # Set limits for the circle plot
    ax_circle.set_xlim(-radius * 1.3, radius * 1.3)
    ax_circle.set_ylim(-radius * 1.3, radius * 1.3)
    ax_circle.set_xticks([])
    ax_circle.set_yticks([])

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()