import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import permutations
import string

# --- 1. RECEIVE 2D POINTS & SETUP CONSTANTS ---
points = np.array([
    [0.0, 0.0],  # A - corner
    [4.0, 0.0],  # B - corner
    [4.0, 4.0],  # C - corner
    [0.0, 4.0],  # D - corner
    [1.0, 1.0],  # E - inner spiral
    [3.0, 1.0],  # F - inner spiral
    [3.0, 3.0],  # G - inner spiral
    [1.0, 3.0],  # H - inner spiral
    [2.0, 2.0],  # I - center trap
])
# points = np.array([
#     [4.0, 4.0],  # A - corner
#     [4.0, 0.0],  # B - corner
#     [0.0, 0.0],  # C - corner
#     [0.0, 4.0],  # D - corner
#     [1.0, 1.0],  # E - inner spiral
#     [3.0, 1.0],  # F - inner spiral
#     [3.0, 3.0],  # G - inner spiral
#     [1.0, 3.0],  # H - inner spiral
#     [2.0, 2.0],  # I - center trap
# ])
# points = np.array([
#     # Left column (close together)
#     [0.0, 0.0],  # 0
#     [0.2, 2.0],  # 1
#     [0.0, 4.0],  # 2
#     [0.3, 6.0],  # 3
#     [0.1, 8.0],  # 4
#     [0.2, 10.0],  # 5
#     # Right column (close together)
#     [5.0, 0.5],  # 6
#     [5.1, 2.5],  # 7
#     [5.2, 4.5],  # 8
#     [5.0, 6.5],  # 9
#     [5.1, 8.5],  # 10
#     [5.0, 10.5],  # 11
#     # Trap points in the middle (these will confuse nearest neighbor)
#     [2.4, 1.0],  # 12 - trap near bottom
#     [2.6, 5.0],  # 13 - trap in middle
#     [2.5, 9.0],  # 14 - trap near top
# ])
max_mag_ind = np.argmax(np.linalg.norm(points,axis=1))
temp = (points[max_mag_ind]).copy()
points[max_mag_ind] = points[0]
points[0] = temp
num_points = len(points)

# Labels are based on the original index of the point (A=0, B=1, etc.)
node_labels = [string.ascii_uppercase[i] for i in range(num_points)]

# Fix for MatplotlibDeprecationWarning and color setup
try:
    cmap = plt.colormaps.get_cmap('Spectral')
except AttributeError:
    cmap = plt.cm.get_cmap('Spectral')

# Setup fixed colors for nodes based on their original index (0 to num_points-1)
node_colors_map = {i: cmap(i / (num_points - 1)) for i in range(num_points)}


# --- HELPER FUNCTIONS ---

def calculate_distance_matrix(pts):
    return np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :]) ** 2, axis=2))


def calculate_cycle_length(pts, cycle):
    dist_matrix = calculate_distance_matrix(pts)
    length = 0
    for i in range(len(cycle)):
        start_node = cycle[i]
        end_node = cycle[(i + 1) % len(cycle)]
        length += dist_matrix[start_node, end_node]
    return length


# --- 2. FIND THE OPTIMAL CYCLE AND NEAREST NEIGHBOR CYCLE ---

def optimal_tsp_solver(pts):
    """Find the shortest TSP cycle using brute force"""
    num_pts = len(pts)
    dist_matrix = calculate_distance_matrix(pts)
    indices_to_permute = range(1, num_pts)  # fix start at 0

    best_path = []
    shortest_length = np.inf

    for perm in permutations(indices_to_permute):
        current_path = [0] + list(perm)
        current_length = 0
        for i in range(num_pts):
            start_node = current_path[i]
            end_node = current_path[(i + 1) % num_pts]
            current_length += dist_matrix[start_node, end_node]

        if current_length < shortest_length:
            shortest_length = current_length
            best_path = current_path

    return np.array(best_path), shortest_length


def nearest_neighbor_tsp(pts, start_node=0):
    """Find a TSP cycle using the nearest neighbor heuristic"""
    num_pts = len(pts)
    dist_matrix = calculate_distance_matrix(pts)

    unvisited = set(range(num_pts))
    path = [start_node]
    unvisited.remove(start_node)

    current_node = start_node
    total_length = 0

    while unvisited:
        # Find the nearest unvisited neighbor
        nearest_node = min(unvisited, key=lambda node: dist_matrix[current_node, node])
        total_length += dist_matrix[current_node, nearest_node]
        path.append(nearest_node)
        unvisited.remove(nearest_node)
        current_node = nearest_node

    # Add distance back to start
    total_length += dist_matrix[current_node, start_node]

    return np.array(path), total_length


# --- 3. DATA GENERATION FUNCTION ---

def setup_path_data(points, path_indices, total_cycle_length, mode='general'):
    N = len(points)

    # Optional orientation enforcement for shortest
    if mode == 'shortest':
        p0_idx_in_path = np.where(path_indices == 0)[0][0]  # should be 0
        idx_clockwise_in_path = (p0_idx_in_path + 1) % N
        idx_ccw_in_path = (p0_idx_in_path - 1) % N
        node_idx_clockwise = path_indices[idx_clockwise_in_path]
        node_idx_ccw = path_indices[idx_ccw_in_path]
        x_cw = points[node_idx_clockwise, 0]
        x_ccw = points[node_idx_ccw, 0]
        if x_cw <= x_ccw:
            new_path_indices = [path_indices[0]] + path_indices[1:][::-1].tolist()
            path_indices = np.array(new_path_indices)

    path_coords = points[path_indices]

    # Circle mapping
    R = total_cycle_length / (2 * np.pi)

    segment_lengths = []
    for i in range(N):
        start_idx = path_indices[i]
        end_idx = path_indices[(i + 1) % N]
        dist = np.linalg.norm(points[start_idx] - points[end_idx])
        segment_lengths.append(dist)
    central_angles = np.array(segment_lengths) / R

    start_angle = np.pi / 2
    cumulative_angles = [start_angle]
    current_angle = start_angle
    for ang in central_angles[:-1]:
        current_angle -= ang
        cumulative_angles.append(current_angle)

    circle_x = R * np.cos(cumulative_angles)
    circle_y = R * np.sin(cumulative_angles)
    circle_coords = np.stack([circle_x, circle_y], axis=1)

    # ARC AND CHORD COORDS
    NUM_ARC_SEGMENTS = 20
    center_x, center_y = 0, 0

    def get_arc_coords(R, angle_start, angle_end, center_x, center_y, num_segments):
        if angle_end > angle_start:
            angle_end -= 2 * np.pi
        arc_angles = np.linspace(angle_start, angle_end, num_segments)
        x_coords = R * np.cos(arc_angles) + center_x
        y_coords = R * np.sin(arc_angles) + center_y
        return np.stack([x_coords, y_coords], axis=1)

    target_arc_coords = []
    straight_chord_coords = []

    for i in range(N):
        p_start_idx = i
        p_end_idx = (i + 1) % N
        angle_start = cumulative_angles[p_start_idx]
        angle_end = cumulative_angles[p_end_idx]
        start_point = path_coords[p_start_idx]
        end_point = path_coords[p_end_idx]

        arc_segment = get_arc_coords(R, angle_start, angle_end, center_x, center_y, NUM_ARC_SEGMENTS)

        t_interp = np.linspace(0, 1, NUM_ARC_SEGMENTS)
        chord_x = (1 - t_interp) * start_point[0] + t_interp * end_point[0]
        chord_y = (1 - t_interp) * start_point[1] + t_interp * end_point[1]
        chord_segment = np.stack([chord_x, chord_y], axis=1)

        straight_chord_coords.append(chord_segment)
        target_arc_coords.append(arc_segment)

    target_arc_coords = np.vstack(target_arc_coords)
    straight_chord_coords = np.vstack(straight_chord_coords)

    # translate circle so P0 lines up with original P0 location
    P0_original = path_coords[0]
    P0_standard = circle_coords[0]
    translation_vector = P0_original - P0_standard

    target_coords = circle_coords + translation_vector
    target_arc_coords_translated = target_arc_coords + translation_vector

    return {
        'path_coords': path_coords,
        'R': R,
        'target_coords': target_coords,
        'straight_chord_coords': straight_chord_coords,
        'target_arc_coords_translated': target_arc_coords_translated,
        'translation_vector': translation_vector,
        'total_length': total_cycle_length,
        'path_indices': path_indices
    }


# --- 4. EXECUTE SOLVERS AND GENERATE DATA ---

path_indices_shortest, total_length_shortest = optimal_tsp_solver(points)
data_shortest = setup_path_data(points, path_indices_shortest, total_length_shortest, mode='shortest')

path_indices_nn, total_length_nn = nearest_neighbor_tsp(points, start_node=0)
data_nn = setup_path_data(points, path_indices_nn, total_length_nn)

print(f"Total cycle length (Optimal TSP): {data_shortest['total_length']:.2f}")
print(f"Total cycle length (Nearest Neighbor): {data_nn['total_length']:.2f}")

# --- 5. PLOT SIDE BY SIDE (Static Visualization) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 14))
fig.subplots_adjust(hspace=0.35, wspace=0.25)  # increase vertical/horizontal spacing between subplots


def plot_static(ax_orig, ax_circle, data, title_prefix, path_indices):
    R = data['R']
    path_coords = data['path_coords']
    target_coords_std = data['target_coords'] - data['translation_vector']
    arc_coords_std = data['target_arc_coords_translated'] - data['translation_vector']

    ax_orig.set_title(f'{title_prefix} Path (Length: {data["total_length"]:.2f})', fontsize=12)
    ax_orig.plot(path_coords[:, 0], path_coords[:, 1], '-', color='gray', zorder=2)
    ax_orig.plot([path_coords[-1, 0], path_coords[0, 0]], [path_coords[-1, 1], path_coords[0, 1]], '-', color='gray',
                 zorder=2)

    node_colors_path = [node_colors_map[i] for i in path_indices]
    ax_orig.scatter(path_coords[:, 0], path_coords[:, 1], c=node_colors_path, s=80, zorder=3)

    for i in range(len(path_coords)):
        x, y = path_coords[i]
        label_idx = path_indices[i]
        ax_orig.annotate(node_labels[label_idx], (x, y), textcoords="offset points", xytext=(6, 6), ha='center',
                         fontsize=10, weight='bold', zorder=4)

    ax_orig.set_aspect('equal', adjustable='box')
    ax_orig.grid(True, linestyle='--', alpha=0.6)
    ax_orig.set_xlabel('')
    ax_orig.set_ylabel('')

    ax_circle.set_title(f'{title_prefix} Mapped Circle (Radius: {R:.2f})', fontsize=12)
    theta = np.linspace(0, 2 * np.pi, 100)
    circ_x_std = R * np.cos(theta)
    circ_y_std = R * np.sin(theta)
    ax_circle.plot(circ_x_std, circ_y_std, 'k--', alpha=0.3)
    ax_circle.plot(arc_coords_std[:, 0], arc_coords_std[:, 1], '-', color='gray', zorder=2)
    ax_circle.scatter(target_coords_std[:, 0], target_coords_std[:, 1], c=node_colors_path, s=80, zorder=3)

    for i in range(len(target_coords_std)):
        x, y = target_coords_std[i]
        label_idx = path_indices[i]
        ax_circle.annotate(node_labels[label_idx], (x, y), textcoords="offset points", xytext=(6, 6), ha='center',
                           fontsize=10, weight='bold', zorder=4)

    ax_circle.set_aspect('equal', adjustable='box')
    ax_circle.grid(True, linestyle='--', alpha=0.6)
    ax_circle.set_xlabel(' ')
    ax_circle.set_ylabel(' ')

    max_R = max(data_shortest['R'], data_nn['R'])
    limit = max_R * 1.5
    ax_circle.set_xlim(-limit, limit)
    ax_circle.set_ylim(-limit, limit)


plot_static(axes[0, 0], axes[0, 1], data_shortest, 'Optimal (Shortest)', data_shortest['path_indices'])
plot_static(axes[1, 0], axes[1, 1], data_nn, 'Nearest Neighbor', data_nn['path_indices'])

fig.suptitle('TSP Path Analysis', fontsize=16, y=0.98)  # move title slightly higher
plt.tight_layout(rect=[0, 0, 1, 0.95])  # ensure subplots stay clear of suptitle



# --- 6. CREATE SIDE-BY-SIDE AND COMBINED ANIMATIONS ---

def interpolate_path(t, start_path, end_path):
    return (1 - t) * start_path + t * end_path


def setup_animation_artists(ax, data, title, path_indices, reference_points=None):
    """
    Initializes artists for one animation subplot (used for the individual shortest/nn axes).
    This function DOES NOT create combined-axis artists.
    """
    N = len(path_indices)

    all_data_coords = np.vstack([data_shortest['path_coords'], data_shortest['target_coords'],
                                 data_nn['path_coords'], data_nn['target_coords']])
    x_min, x_max = np.min(all_data_coords[:, 0]) - 1, np.max(all_data_coords[:, 0]) + 1
    y_min, y_max = np.min(all_data_coords[:, 1]) - 1, np.max(all_data_coords[:, 1]) + 1

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title(title, fontsize=14)

    if reference_points is not None:
        ax.scatter(reference_points[:, 0], reference_points[:, 1], marker='x', color='black', alpha=0.4, s=60,
                   label='Original Points', zorder=1)
        for i, (x, y) in enumerate(reference_points):
            ax.annotate(node_labels[i], (x, y), textcoords="offset points", xytext=(8, 8), ha='center', fontsize=9,
                        color='black', alpha=0.6, zorder=1)

    R = data['R']
    theta = np.linspace(0, 2 * np.pi, 100)
    circ_x_std = R * np.cos(theta)
    circ_y_std = R * np.sin(theta)
    translated_circ_x = circ_x_std + data['translation_vector'][0]
    translated_circ_y = circ_y_std + data['translation_vector'][1]
    ax.plot(translated_circ_x, translated_circ_y, 'k--', alpha=0.2)

    node_colors_path = [node_colors_map[i] for i in path_indices]

    line_path, = ax.plot([], [], '-', color='gray', zorder=2, label=title.split()[0] + ' Path')
    scatter_pts = ax.scatter(data['path_coords'][:, 0], data['path_coords'][:, 1], c=node_colors_path, s=80, zorder=3)
    start_node_marker, = ax.plot([], [], 's', markersize=8, color='red', zorder=5, label='Start Node')

    annotations = []  # disabled dynamic annotations

    return line_path, scatter_pts, start_node_marker, annotations


# Setup the main animation figure with 3 subplots
fig_anim, axes_anim = plt.subplots(1, 3, figsize=(24, 8))
fig_anim.subplots_adjust(wspace=0.3, top=0.88)  # add horizontal space and top margin for title

ax_shortest = axes_anim[0]
ax_nn = axes_anim[1]
ax_combined = axes_anim[2]

# Shortest individual subplot artists
line_path_s, scatter_pts_s, start_node_marker_s, annotations_s = setup_animation_artists(
    ax_shortest, data_shortest, f'Optimal (Length: {data_shortest["total_length"]:.2f})', data_shortest['path_indices']
)

# Nearest Neighbor individual subplot artists
line_path_nn, scatter_pts_nn, start_node_marker_nn, annotations_nn = setup_animation_artists(
    ax_nn, data_nn, f'Nearest Neighbor (Length: {data_nn["total_length"]:.2f})', data_nn['path_indices']
)
for i in range(num_points):
    ax_shortest.scatter(points[i, 0], points[i, 1], marker='x', s=80,
                        color=node_colors_map[i], alpha=0.9, zorder=1)
for i in range(num_points):
    ax_nn.scatter(points[i, 0], points[i, 1], marker='x', s=80,
                        color=node_colors_map[i], alpha=0.9, zorder=1)
# Combined plot: we'll create separate animated scatters for shortest and nn,
# plus static X markers colored per original node.
# 1) static X markers for original points (colored by original index)
for i in range(num_points):
    ax_combined.scatter(points[i, 0], points[i, 1], marker='x', s=80,
                        color=node_colors_map[i], alpha=0.9, zorder=1)

# 2) background circles (optional guides)
R_s = data_shortest['R']
theta = np.linspace(0, 2 * np.pi, 100)
circ_x_s = R_s * np.cos(theta) + data_shortest['translation_vector'][0]
circ_y_s = R_s * np.sin(theta) + data_shortest['translation_vector'][1]
ax_combined.plot(circ_x_s, circ_y_s, '--', alpha=0.15, zorder=0)

R_nn = data_nn['R']
circ_x_nn = R_nn * np.cos(theta) + data_nn['translation_vector'][0]
circ_y_nn = R_nn * np.sin(theta) + data_nn['translation_vector'][1]
ax_combined.plot(circ_x_nn, circ_y_nn, '--', alpha=0.12, zorder=0)

# 3) dynamic elements for combined plot (two separate scatters)
# Colors per node according to original indices, but ordered per path order
node_colors_short_ordered = [node_colors_map[i] for i in data_shortest['path_indices']]
node_colors_nn_ordered = [node_colors_map[i] for i in data_nn['path_indices']]

# Initialize the two animated scatters on combined axis (distinct markers)
scatter_pts_c_s = ax_combined.scatter(data_shortest['path_coords'][:, 0],
                                      data_shortest['path_coords'][:, 1],
                                      c=node_colors_short_ordered, s=110, marker='o', edgecolors='k', zorder=4,
                                      label='Optimal Nodes')

scatter_pts_c_nn = ax_combined.scatter(data_nn['path_coords'][:, 0],
                                       data_nn['path_coords'][:, 1],
                                       c=node_colors_nn_ordered, s=80, marker='D', edgecolors='k', zorder=3,
                                       label='NN Nodes')

# Lines for the interpolating arcs/chords
line_path_c_s, = ax_combined.plot([], [], '-', color='blue', linewidth=2, zorder=2, label='Optimal Path')
line_path_c_nn, = ax_combined.plot([], [], '-', color='red', linewidth=2, zorder=2, label='NN Path')

# Start markers
start_node_marker_c_s, = ax_combined.plot([], [], 's', markersize=10, color='blue', zorder=5)
start_node_marker_c_nn, = ax_combined.plot([], [], 's', markersize=8, color='red', zorder=5)

ax_combined.set_title(f'Combined (Optimal: {data_shortest["total_length"]:.2f}, NN: {data_nn["total_length"]:.2f})',
                      fontsize=14)
ax_combined.set_aspect('equal', adjustable='box')
ax_combined.grid(True, linestyle='--', alpha=0.6)
# ax_combined.legend(loc='lower left')

# set combined axis limits consistent with others
all_data_coords = np.vstack([data_shortest['path_coords'], data_shortest['target_coords'],
                             data_nn['path_coords'], data_nn['target_coords']])
x_min, x_max = np.min(all_data_coords[:, 0]) - 1, np.max(all_data_coords[:, 0]) + 1
y_min, y_max = np.min(all_data_coords[:, 1]) - 1, np.max(all_data_coords[:, 1]) + 1
ax_combined.set_xlim(x_min, x_max)
ax_combined.set_ylim(y_min, y_max)


def init_all():
    all_artists = []

    # Shortest individual init
    line_path_s.set_data([], [])
    scatter_pts_s.set_visible(False)
    start_node_marker_s.set_data([], [])
    all_artists.extend([line_path_s, scatter_pts_s, start_node_marker_s])

    # NN individual init
    line_path_nn.set_data([], [])
    scatter_pts_nn.set_visible(False)
    start_node_marker_nn.set_data([], [])
    all_artists.extend([line_path_nn, scatter_pts_nn, start_node_marker_nn])

    # Combined shortest
    line_path_c_s.set_data([], [])
    scatter_pts_c_s.set_offsets(data_shortest['path_coords'])  # initial
    start_node_marker_c_s.set_data([], [])
    all_artists.extend([line_path_c_s, scatter_pts_c_s, start_node_marker_c_s])

    # Combined NN
    line_path_c_nn.set_data([], [])
    scatter_pts_c_nn.set_offsets(data_nn['path_coords'])  # initial
    start_node_marker_c_nn.set_data([], [])
    all_artists.extend([line_path_c_nn, scatter_pts_c_nn, start_node_marker_c_nn])

    return all_artists


def update_all(frame):
    t = frame
    all_artists = []

    # Shortest interpolations
    interpolated_path_s = interpolate_path(t, data_shortest['straight_chord_coords'],
                                           data_shortest['target_arc_coords_translated'])
    interpolated_node_s = interpolate_path(t, data_shortest['path_coords'], data_shortest['target_coords'])

    # NN interpolations
    interpolated_path_nn = interpolate_path(t, data_nn['straight_chord_coords'],
                                            data_nn['target_arc_coords_translated'])
    interpolated_node_nn = interpolate_path(t, data_nn['path_coords'], data_nn['target_coords'])

    # Update shortest individual subplot
    line_path_s.set_data(interpolated_path_s[:, 0], interpolated_path_s[:, 1])
    scatter_pts_s.set_offsets(interpolated_node_s)
    scatter_pts_s.set_visible(True)
    start_node_marker_s.set_data([interpolated_node_s[0, 0]], [interpolated_node_s[0, 1]])
    all_artists.extend([line_path_s, scatter_pts_s, start_node_marker_s])

    # Update NN individual subplot
    line_path_nn.set_data(interpolated_path_nn[:, 0], interpolated_path_nn[:, 1])
    scatter_pts_nn.set_offsets(interpolated_node_nn)
    scatter_pts_nn.set_visible(True)
    start_node_marker_nn.set_data([interpolated_node_nn[0, 0]], [interpolated_node_nn[0, 1]])
    all_artists.extend([line_path_nn, scatter_pts_nn, start_node_marker_nn])

    # Combined: shortest (blue)
    line_path_c_s.set_data(interpolated_path_s[:, 0], interpolated_path_s[:, 1])
    scatter_pts_c_s.set_offsets(interpolated_node_s)
    start_node_marker_c_s.set_data([interpolated_node_s[0, 0]], [interpolated_node_s[0, 1]])
    all_artists.extend([line_path_c_s, scatter_pts_c_s, start_node_marker_c_s])

    # Combined: NN (red)
    line_path_c_nn.set_data(interpolated_path_nn[:, 0], interpolated_path_nn[:, 1])
    scatter_pts_c_nn.set_offsets(interpolated_node_nn)
    start_node_marker_c_nn.set_data([interpolated_node_nn[0, 0]], [interpolated_node_nn[0, 1]])
    all_artists.extend([line_path_c_nn, scatter_pts_c_nn, start_node_marker_c_nn])

    fig_anim.suptitle('TSP Path Morphing Comparison (Optimal vs Nearest Neighbor)', fontsize=16)
    return all_artists


# Animation setup
frames = np.linspace(0, 1, 100)
ani = FuncAnimation(
    fig_anim,
    update_all,
    frames=frames,
    init_func=init_all,
    blit=True,
    interval=50
)

plt.show()