import argparse
import numpy as np
import torch
from math import pi, acos
import os
import sys
from tqdm import tqdm
from numpy.linalg import norm, svd  # <-- Import norm and svd

# Define global constant (no longer used by default, but kept if needed)
NUM_ARC_SEGMENTS = 20


# ---------- Core Data Processing Functions ----------

def normalize_points(points: np.ndarray) -> np.ndarray:
    """
    [LEGACY] Centers the point cloud at the origin and scales it so that
    the point with the maximum norm has a norm of 1.
    This is used to create the initial geometry before Kendall pre-processing.

    Args:
        points: A 2D NumPy array of shape (N, 2).

    Returns:
        A new (N, 2) NumPy array representing the normalized
        point cloud.
    """
    # 1. Center the point cloud
    points_centered = points - np.mean(points, axis=0)

    # 2. Find the maximum norm
    max_norm = np.max(np.linalg.norm(points_centered, axis=1))
    if max_norm == 0:
        return points_centered  # Avoid division by zero

    # 3. Scale the point cloud
    normalized_points = points_centered / max_norm
    return normalized_points


def pre_process_shape(X: np.ndarray) -> np.ndarray:
    """
    [NEW] Pre-processes a shape for Kendall's shape space.
    1. Centers the shape at the origin.
    2. Scales the shape to have a unit Frobenius norm.

    Args:
        X: A (N, 2) NumPy array of vertex coordinates.

    Returns:
        A new (N, 2) NumPy array (centered, unit F-norm).
    """
    # 1. Center
    X_c = X - np.mean(X, axis=0)

    # 2. Scale to unit Frobenius norm
    f_norm = norm(X_c, 'fro')
    if f_norm < 1e-9:
        return X_c  # Avoid division by zero

    X_normed = X_c / f_norm
    return X_normed


def find_optimal_rotation(A: np.ndarray, B: np.ndarray):
    """
    [NEW] Finds the optimal rotation to align B to A (Procrustes).
    Assumes A and B are already centered and scaled.
    Minimizes: || A - B @ Rot ||^2

    Args:
        A: The (N, 2) target point cloud (pre-processed).
        B: The (N, 2) source point cloud (pre-processed).

    Returns:
        A tuple containing:
        - B_aligned (np.ndarray): The (N, 2) aligned cloud (B @ Rot).
        - rotation_matrix (np.ndarray): The (2, 2) optimal rotation matrix.
    """
    # Compute covariance matrix M = B.T @ A
    M = B.T @ A

    # Compute SVD of M
    try:
        U, s, Vh = svd(M)
    except np.linalg.LinAlgError:
        # Fallback to identity rotation if SVD fails
        Rot = np.identity(2)
    else:
        # Compute optimal rotation Rot, ensuring det(Rot) = 1 (no reflection)
        S_fix = np.diag([1.0, np.linalg.det(U @ Vh)])
        Rot = U @ S_fix @ Vh  # This is the 2x2 Rotation Matrix

    # Apply the optimal rotation
    B_aligned = B @ Rot

    return B_aligned, Rot


def calculate_edge_lengths(points: np.ndarray, path_indices: np.ndarray) -> np.ndarray:
    """
    [NEW] Computes the Euclidean length of each edge in the TSP cycle.

    Args:
        points: The (N, 2) point cloud.
        path_indices: The (N,) array of node indices in TSP order.

    Returns:
        (N,) np.ndarray of edge lengths.
    """
    N = len(path_indices)
    segment_lengths = []
    for i in range(N):
        start_idx = path_indices[i]
        end_idx = path_indices[(i + 1) % N]
        dist = norm(points[start_idx] - points[end_idx])
        segment_lengths.append(dist)
    return np.array(segment_lengths, dtype=np.float64)


def calculate_turning_angles(points: np.ndarray, path_indices: np.ndarray) -> np.ndarray:
    """
    [NEW] Computes the interior turning angle at each vertex of the polygon path.

    Args:
        points: The (N, 2) point cloud.
        path_indices: The (N,) array of node indices in TSP order.

    Returns:
        (N,) np.ndarray of turning angles in radians.
    """
    N = len(path_indices)
    path_coords = points[path_indices]  # Points in path order
    turning_angles = []

    for i in range(N):
        p_prev = path_coords[(i - 1) % N]
        p_curr = path_coords[i]
        p_next = path_coords[(i + 1) % N]

        v1 = p_prev - p_curr  # Incoming vector
        v2 = p_next - p_curr  # Outgoing vector

        norm_v1 = norm(v1)
        norm_v2 = norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            angle = 0.0  # Handle collinear/degenerate case
        else:
            v1_u = v1 / norm_v1
            v2_u = v2 / norm_v2
            dot_prod = np.dot(v1_u, v2_u)
            angle = acos(np.clip(dot_prod, -1.0, 1.0))

        turning_angles.append(angle)

    return np.array(turning_angles, dtype=np.float64)


def create_circle_nodes(normalized_points: np.ndarray, path_indices: np.ndarray,
                        segment_lengths: np.ndarray, total_cycle_length: float):
    """
    Creates the initial (un-rotated, origin-centered) circle coordinates
    based on the TSP path and total length. Enforces 'shortest' orientation.

    Args:
        normalized_points: The (N, 2) normalized input point cloud.
        path_indices: The (N,) array of node indices in TSP order.
        segment_lengths: The (N,) array of path segment lengths.
        total_cycle_length: The total length of the TSP path.

    Returns:
        A tuple containing:
        - circle_coords (np.ndarray): The (N, 2) coordinates on the circle, centered at origin.
        - path_coords (np.ndarray): The (N, 2) normalized points, ordered by the path.
        - path_indices (np.ndarray): The (potentially re-oriented) path indices.
        - segment_lengths (np.ndarray): The (potentially re-oriented) segment lengths.
    """
    N = len(normalized_points)

    # 1. Enforce 'shortest' orientation
    p0_idx_in_path = np.where(path_indices == 0)[0][0]  # index position where node 0 appears
    idx_clockwise_in_path = (p0_idx_in_path + 1) % N
    idx_ccw_in_path = (p0_idx_in_path - 1) % N
    node_idx_clockwise = path_indices[idx_clockwise_in_path]
    node_idx_ccw = path_indices[idx_ccw_in_path]

    x_cw = normalized_points[node_idx_clockwise, 0]
    x_ccw = normalized_points[node_idx_ccw, 0]
    if x_cw <= x_ccw:
        # Reverse the path and segment lengths
        new_path_indices = [path_indices[0]] + path_indices[1:][::-1].tolist()
        path_indices = np.array(new_path_indices, dtype=np.int64)

        # segment_lengths[i] corresponds to edge (path_indices[i], path_indices[i+1])
        # Reversing the path means the segments are also re-ordered
        # The new segment_lengths[0] is old segment_lengths[0] (p0 -> p_last)
        # The new segment_lengths[1] is old segment_lengths[N-1] (p_last -> p_N-2)
        # etc.
        new_segment_lengths = [segment_lengths[0]] + segment_lengths[1:][::-1].tolist()
        segment_lengths = np.array(new_segment_lengths, dtype=np.float64)


    # 2. Get points in path order
    path_coords = normalized_points[path_indices]  # shape (N,2)

    # 3. Calculate circle radius
    R = total_cycle_length / (2.0 * pi)  # This is the scalar Radius

    # 4. Compute central angles
    central_angles = segment_lengths / R  # shape (N,)

    # 5. Calculate cumulative angles for each node
    start_angle = np.pi / 2.0
    cumulative_angles = [start_angle]
    current_angle = start_angle
    for ang in central_angles[:-1]:
        current_angle -= ang
        cumulative_angles.append(current_angle)
    cumulative_angles = np.array(cumulative_angles, dtype=np.float64)  # shape (N,)

    # 6. Create origin-centered circle coordinates
    circle_x = R * np.cos(cumulative_angles)
    circle_y = R * np.sin(cumulative_angles)
    circle_coords = np.stack([circle_x, circle_y], axis=1)  # (N,2)

    return circle_coords, path_coords, path_indices, segment_lengths


# ---------- Helper Utilities ----------
# (calculate_total_cycle_length is now replaced by calculate_edge_lengths and np.sum)

# ---------- .txt parsing function ----------
# (Unchanged)
def parse_tsp_lib_line(line: str):
    line = line.strip()
    if not line:
        return None
    parts = line.split(" output ")
    if len(parts) != 2:
        return None
    coord_str, sol_str = parts
    try:
        coords_flat = [float(x) for x in coord_str.split()]
        N = len(coords_flat) // 2
        if len(coords_flat) % 2 != 0 or N == 0:
            return None
        points = np.array(coords_flat, dtype=np.float64).reshape(N, 2)
        path_1_indexed = [int(x) for x in sol_str.split()]
        if len(path_1_indexed) < 2 or path_1_indexed[0] != path_1_indexed[-1]:
            return None
        path_0_indexed = [x - 1 for x in path_1_indexed[:-1]]
        if len(path_0_indexed) != N:
            return None
        if set(path_0_indexed) != set(range(N)):
            return None
        return points, np.array(path_0_indexed, dtype=np.int64)
    except Exception:
        return None


# ---------- NEW: dataset construction from .txt file ----------

def process_txt_to_dataset(txt_path):
    """
    Reads a .txt file (tsp-20/tsp-50 format) and returns a list of dicts
    (one dict per valid line / problem instance).
    Each dict contains torch-friendly numpy arrays (float32/int64).
    """
    dataset_entries = []

    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        if not lines:
            print(f"Warning: File {txt_path} is empty.")
            return []
    except FileNotFoundError:
        print(f"Error: File not found at {txt_path}")
        return []
    except Exception as e:
        print(f"Error reading file {txt_path}: {e}")
        return []

    file_name = os.path.basename(txt_path)

    for line_idx, line in enumerate(tqdm(lines, desc=f"Processing {file_name}", unit=" sample")):
        parsed_data = parse_tsp_lib_line(line)
        if parsed_data is None:
            tqdm.write(f"Skipping line {line_idx+1} (invalid format).")
            continue

        points, path_indices = parsed_data

        # --- Start of Core Geometric Logic ---

        # 1. Normalize the points (legacy method) for geometric calcs
        normalized_points = normalize_points(points)

        # 2. Calculate edge lengths and turning angles (on legacy normalized points)
        segment_lengths = calculate_edge_lengths(normalized_points, path_indices)
        turning_angles = calculate_turning_angles(normalized_points, path_indices)
        total_length = np.sum(segment_lengths)

        # 3. Create the initial, origin-centered circle nodes
        #    This also re-orients the path and segments if needed.
        circle_coords, path_coords, path_indices, segment_lengths = \
            create_circle_nodes(normalized_points, path_indices, segment_lengths, total_length)

        # 4. [NEW] Pre-process both shapes for Kendall's space
        #    Both `path_coords` and `circle_coords` are in path order.
        X0_path = pre_process_shape(path_coords)
        X1_path = pre_process_shape(circle_coords)

        # 5. [NEW] Find optimal alignment (rotation only)
        X1_aligned, Rot = find_optimal_rotation(X0_path, X1_path)

        # 6. [NEW] Compute Procrustes distance (theta)
        #    We clip to avoid numerical issues with acos
        dot_prod = np.clip(np.trace(X0_path.T @ X1_aligned), -1.0, 1.0)
        theta = acos(dot_prod)

        # 7. Re-order the final clouds to match the *original* point cloud order
        inv_path = [i for i in range(len(path_indices))]
        for i, ind in enumerate(path_indices):
            inv_path[ind] = i

        X0_final = X0_path[inv_path]
        X1_final = X1_aligned[inv_path]

        # --- End of Core Geometric Logic ---

        entry = {
            'points': X0_final.astype(np.float32),          # (N,2) X0
            'circle': X1_final.astype(np.float32),          # (N,2) X1 (aligned)
            'path': path_indices.astype(np.int64),          # (N,) ordering
            'theta': float(theta),                          # Procrustes distance
            'edge_lengths': segment_lengths.astype(np.float32), # (N,)
            'turning_angles': turning_angles.astype(np.float32) # (N,)
        }
        dataset_entries.append(entry)

    return dataset_entries


# ---------- Small plotting helper (optional) ----------
# (Unchanged)
def quick_plot_sample(entry, show=True, ax=None):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available, skipping plot.")
        return
    pts = entry['points']
    circle = entry['circle']
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    N = len(pts)
    cmap = plt.get_cmap('tab20')
    for i in range(N):
        ax.scatter(pts[i, 0], pts[i, 1], marker='o', s=40, color=cmap(i % 20))
        ax.scatter(circle[i, 0], circle[i, 1], marker='x', s=40, color=cmap(i % 20))
    ax.set_aspect('equal', 'box')
    ax.set_title('points (o) and circle nodes (x) (Kendall Pre-Processed)')
    if show:
        plt.show()


# ---------- Main / CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Convert TSP .txt dataset -> PyTorch dataset for flow matching.")
    parser.add_argument('--infile', default="../tsp50_test_concorde.txt",
                        help="Path to .txt file (e.g., tsp20.txt or tsp50.txt).")
    parser.add_argument('--out', default='processed_tsp_dataset_TSP50_test.pt',
                        help="Output .pt file to save processed dataset.")
    # 'num_arc_segments' is no longer used but kept to avoid breaking old CLI calls
    parser.add_argument('--num-arc-segments', type=int, default=NUM_ARC_SEGMENTS,
                        help="[DEPRECATED] No longer used.")
    parser.add_argument('--preview-first', action='store_true',
                        help="Show a quick plot of the first sample (matplotlib required).")
    args = parser.parse_args()

    if not os.path.exists(args.infile):
        print(f"Error: Input file not found at {args.infile}")
        print("Please download the tsp-20 or tsp-50 .txt dataset file.")
        return

    print("Reading input file:", args.infile)
    try:
        entries = process_txt_to_dataset(args.infile)
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"Error processing file: {e}")
        return

    if not entries:
        print("No valid entries were processed. Check file format.")
        return

    print(f"Processed {len(entries)} instances. Saving to {args.out} ...")
    torch.save(entries, args.out)
    print("Saved.")

    if args.preview_first and len(entries) > 0:
        quick_plot_sample(entries[0])


if __name__ == '__main__':
    main()