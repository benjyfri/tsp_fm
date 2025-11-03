import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from math import pi
from scipy.spatial.distance import pdist, squareform
import scipy.linalg

# --- Helper Functions from Original Script (for point generation) ---
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def calculate_cycle_distance(points, cycle_indices):
    total_distance = 0
    num_points = len(cycle_indices)
    for i in range(num_points):
        p1_index = cycle_indices[i]
        p2_index = cycle_indices[(i + 1) % num_points]
        total_distance += euclidean_distance(points[p1_index], points[p2_index])
    return total_distance

def solve_tsp(points):
    n = len(points)
    point_indices = list(range(n))
    inner_indices = point_indices[1:]

    min_distance = float('inf')
    shortest_path_indices = None

    for perm in permutations(inner_indices):
        cycle = (0,) + perm
        distance = calculate_cycle_distance(points, cycle)
        if distance < min_distance:
            min_distance = distance
            shortest_path_indices = cycle
    return min_distance, shortest_path_indices

def setup_path_data(points, path_indices, total_cycle_length, mode='shortest'):
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
    R = total_cycle_length / (2.0 * pi)
    segment_lengths = []
    for i in range(N):
        start_idx = path_indices[i]
        end_idx = path_indices[(i + 1) % N]
        dist = np.linalg.norm(points[start_idx] - points[end_idx])
        segment_lengths.append(dist)
    segment_lengths = np.array(segment_lengths, dtype=np.float64)
    central_angles = segment_lengths / R
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
    P0_original = path_coords[0]
    P0_standard = circle_coords[0]
    translation_vector = P0_original - P0_standard
    target_coords = circle_coords + translation_vector

    return {
        'target_coords': target_coords.astype(np.float64),
    }

# ----------------------------------------------------------------------
#                      Graph/Laplacian Functions
# ----------------------------------------------------------------------

def calculate_weighted_adjacency(points, sigma):
    dist_matrix = squareform(pdist(points, 'euclidean'))
    W = np.exp(-dist_matrix**2 / (2 * sigma**2))
    np.fill_diagonal(W, 0)
    return W

def calculate_laplacian(W):
    D = np.diag(W.sum(axis=1))
    L = D - W
    return L

def get_spectrum(L):
    eigenvalues = scipy.linalg.eigvalsh(L)
    return np.sort(eigenvalues)

# ----------------------------------------------------------------------
#                             MAIN SCRIPT
# ----------------------------------------------------------------------

# --- Global Settings ---
N = 11  # Number of points
p = 5 # Number of examples to plot

# Prepare a single figure with p rows and 3 columns
fig, axes = plt.subplots(p, 3, figsize=(21, 6 * p))
fig.suptitle(f'Graph Laplacian Decomposition Difference (N={N} Nodes, {p} Examples)', fontsize=20, y=0.99)

all_fiedler_values_original = []
all_fiedler_values_circle = []

for i in range(p):
    # --- 1. Sample Random Points ---
    original_points = np.random.uniform(-1, 1, size=(N, 2))

    # --- 2. Solve TSP (needed to get circle placement) ---
    shortest_dist_original, shortest_perm_original = solve_tsp(original_points)

    # --- 3. Create Corresponding Circle Graph Placement ---
    path_data = setup_path_data(
        points=original_points,
        path_indices=shortest_perm_original,
        total_cycle_length=shortest_dist_original
    )
    circle_points = path_data['target_coords']

    # --- 4. Calculate Laplacians and Spectra ---
    dist_matrix_original = squareform(pdist(original_points, 'euclidean'))
    pairwise_dists = dist_matrix_original[np.triu_indices(N, k=1)]
    sigma = np.mean(pairwise_dists) + 1e-9 # Add epsilon to avoid div by zero

    W_original = calculate_weighted_adjacency(original_points, sigma)
    L_original = calculate_laplacian(W_original)
    spec_original = get_spectrum(L_original)

    W_circle = calculate_weighted_adjacency(circle_points, sigma)
    L_circle = calculate_laplacian(W_circle)
    spec_circle = get_spectrum(L_circle)

    # Store Fiedler values for summary
    all_fiedler_values_original.append(spec_original[1])
    all_fiedler_values_circle.append(spec_circle[1])

    # --- 5. Plot Point Clouds and Spectra in current row ---
    indices = np.arange(N)

    # Plot 1: Original Points
    ax0 = axes[i, 0]
    ax0.scatter(original_points[:, 0], original_points[:, 1], s=100, c=indices, cmap='viridis', edgecolors='k')
    ax0.set_title(f'Ex {i+1}: Original Random Points (Sigma={sigma:.2f})')
    ax0.set_xlabel('X')
    ax0.set_ylabel('Y')
    ax0.axis('equal')
    ax0.grid(True, linestyle='--')

    # Plot 2: Circle Points
    ax1 = axes[i, 1]
    ax1.scatter(circle_points[:, 0], circle_points[:, 1], s=100, c=indices, cmap='viridis', edgecolors='k')
    ax1.set_title(f'Ex {i+1}: Circle Placement Points')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.axis('equal')
    ax1.grid(True, linestyle='--')

    # Plot 3: Laplacian Spectrum Comparison
    ax2 = axes[i, 2]
    ax2.plot(indices, spec_original, 'o-', label='Original Spectrum', markersize=6)
    ax2.plot(indices, spec_circle, 'x-', label='Circle Spectrum', markersize=6)
    ax2.set_title(f'Ex {i+1}: Laplacian Spectrum ($\lambda_1$ Orig={spec_original[1]:.2f}, Circ={spec_circle[1]:.2f})')
    ax2.set_xlabel('Eigenvalue Index ($k$)')
    ax2.set_ylabel('Eigenvalue ($\lambda_k$)')
    ax2.set_xticks(indices)
    ax2.legend(loc='upper left')
    ax2.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for the super title
plt.show()

# --- Summary & Analysis Printout for all examples ---
print("\n" + "=" * 60)
print(f"        Summary of Fiedler Values (N={N}, {p} Examples)")
print("=" * 60)
for i in range(p):
    print(f"Example {i+1}:")
    print(f"  Original Fiedler Value ($\lambda_1$): {all_fiedler_values_original[i]:.6f}")
    print(f"  Circle Fiedler Value ($\lambda_1$):   {all_fiedler_values_circle[i]:.6f}")
    print(f"  Difference: {all_fiedler_values_original[i] - all_fiedler_values_circle[i]:.6f}")
    print("-" * 20)

print("\n**General Observation:** The Fiedler value (second smallest eigenvalue) of the Laplacian")
print("is an indicator of graph connectivity and 'how well-connected' the graph is.")
print("A smaller Fiedler value typically suggests a more 'path-like' or 'linear' structure,")
print("while a larger value indicates better connectivity and more 'spreading' capability.")
print("Comparing the two setups, the circle arrangement often leads to a different spectral")
print("signature than the randomly placed points, reflecting its more constrained geometry.")