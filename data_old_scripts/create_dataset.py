import argparse
import numpy as np
import pandas as pd
import torch
from itertools import permutations
from math import pi
import os
import sys

# Define global constant
NUM_ARC_SEGMENTS = 20


# ---------- Core Data Processing Functions ----------

def normalize_points(points: np.ndarray) -> np.ndarray:
    """
    Centers the point cloud at the origin and scales it so that
    the point with the maximum norm has a norm of 1.

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


def create_circle_nodes(normalized_points: np.ndarray, path_indices: np.ndarray,
                        total_cycle_length: float):
    """
    Creates the initial (un-rotated, origin-centered) circle coordinates
    based on the TSP path and total length. Enforces 'shortest' orientation.

    Args:
        normalized_points: The (N, 2) normalized input point cloud.
        path_indices: The (N,) array of node indices in TSP order.
        total_cycle_length: The total length of the TSP path.

    Returns:
        A tuple containing:
        - circle_coords (np.ndarray): The (N, 2) coordinates on the circle, centered at origin.
        - path_coords (np.ndarray): The (N, 2) normalized points, ordered by the path.
        - R (float): The calculated radius of the circle.
        - cumulative_angles (np.ndarray): The (N,) array of angles for each node.
        - path_indices (np.ndarray): The (potentially re-oriented) path indices.
    """
    N = len(normalized_points)

    # 1. Enforce 'shortest' orientation
    p0_idx_in_path = np.where(path_indices == 0)[0][0]  # index position where node 0 appears
    idx_clockwise_in_path = (p0_idx_in_path + 1) % N
    idx_ccw_in_path = (p0_idx_in_path - 1) % N
    node_idx_clockwise = path_indices[idx_clockwise_in_path]
    node_idx_ccw = path_indices[idx_ccw_in_path]

    # Use points from the *normalized* cloud for the check
    x_cw = normalized_points[node_idx_clockwise, 0]
    x_ccw = normalized_points[node_idx_ccw, 0]
    if x_cw <= x_ccw:
        new_path_indices = [path_indices[0]] + path_indices[1:][::-1].tolist()
        path_indices = np.array(new_path_indices, dtype=np.int64)

    # 2. Get points in path order
    path_coords = normalized_points[path_indices]  # shape (N,2)

    # 3. Calculate circle radius
    R = total_cycle_length / (2.0 * pi)  # This is the scalar Radius

    # 4. Compute segment lengths and central angles
    segment_lengths = []
    for i in range(N):
        start_idx = path_indices[i]
        end_idx = path_indices[(i + 1) % N]
        dist = np.linalg.norm(normalized_points[start_idx] - normalized_points[end_idx])
        segment_lengths.append(dist)
    segment_lengths = np.array(segment_lengths, dtype=np.float64)
    central_angles = segment_lengths / R  # shape (N,)

    # 5. Calculate cumulative angles for each node
    # start angle = pi/2 (top of the circle)
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

    return circle_coords, path_coords, R, cumulative_angles, path_indices


def align_clouds_procrustes(A_cloud: np.ndarray, B_cloud: np.ndarray):
    """
    Finds the optimal rotation and translation to align B_cloud to A_cloud
    using Orthogonal Procrustes analysis.

    Minimizes: || A_cloud - (B_cloud @ Rot + t) ||^2

    Args:
        A_cloud: The (N, 2) target point cloud (e.g., path_coords).
        B_cloud: The (N, 2) source point cloud (e.g., initial circle_coords).

    Returns:
        A tuple containing:
        - B_cloud_aligned (np.ndarray): The (N, 2) aligned cloud.
        - rotation_matrix (np.ndarray): The (2, 2) optimal rotation matrix.
        - translation_vector (np.ndarray): The (1, 2) optimal translation vector.
    """
    # 1. Center both point clouds
    A_centroid = np.mean(A_cloud, axis=0)
    B_centroid = np.mean(B_cloud, axis=0)  # B_cloud is at [0,0] but we compute for generality

    A_c = A_cloud - A_centroid
    B_c = B_cloud - B_centroid

    # 2. Compute covariance matrix M = B_c.T @ A_c
    M = B_c.T @ A_c

    # 3. Compute SVD of M
    try:
        U, s, Vh = np.linalg.svd(M)
    except np.linalg.LinAlgError:
        # Fallback to identity rotation if SVD fails
        Rot = np.identity(2)
    else:
        # 4. Compute optimal rotation Rot, ensuring det(Rot) = 1 (no reflection)
        S_fix = np.diag([1.0, np.linalg.det(U @ Vh)])
        Rot = U @ S_fix @ Vh  # This is the 2x2 Rotation Matrix

    # 5. Compute optimal translation
    # t = A_centroid - B_centroid @ Rot
    translation_vector = A_centroid - (B_centroid @ Rot)

    # 6. Apply the optimal rotation Rot and translation t
    B_cloud_aligned = (B_cloud @ Rot) + translation_vector

    return B_cloud_aligned, Rot, translation_vector


def create_arc_data(path_coords: np.ndarray, R: float, cumulative_angles: np.ndarray,
                    rotation_matrix: np.ndarray, translation_vector: np.ndarray,
                    num_arc_segments: int = 20):
    """
    Generates high-resolution arc and chord coordinates.

    Args:
        path_coords: The (N, 2) normalized points, ordered by the path.
        R: The circle radius.
        cumulative_angles: The (N,) array of angles for each node.
        rotation_matrix: The (2, 2) rotation matrix from Procrustes.
        translation_vector: The (1, 2) translation vector from Procrustes.
        num_arc_segments: The number of segments for each arc.

    Returns:
        A tuple containing:
        - target_arc_coords_translated (np.ndarray): (N*num_seg, 2) aligned arc points.
        - straight_chord_coords (np.ndarray): (N*num_seg, 2) straight line points.
    """
    N = len(path_coords)
    center_x, center_y = 0.0, 0.0  # Initial circle is at origin

    def get_arc_coords(R_, angle_start, angle_end, center_x_, center_y_, num_segments):
        # ensure we go clockwise (angles decreasing)
        if angle_end > angle_start:
            angle_end -= 2 * np.pi
        arc_angles = np.linspace(angle_start, angle_end, num_segments)
        x_coords = R_ * np.cos(arc_angles) + center_x_
        y_coords = R_ * np.sin(arc_angles) + center_y_
        return np.stack([x_coords, y_coords], axis=1)  # (num_segments, 2)

    target_arc_coords = []
    straight_chord_coords = []

    for i in range(N):
        p_start_idx = i
        p_end_idx = (i + 1) % N
        angle_start = cumulative_angles[p_start_idx]
        angle_end = cumulative_angles[p_end_idx]
        start_point = path_coords[p_start_idx]
        end_point = path_coords[p_end_idx]

        # Arc segment (at origin)
        arc_segment = get_arc_coords(R, angle_start, angle_end, center_x, center_y, num_arc_segments)

        # Straight line segment (chord)
        t_interp = np.linspace(0, 1, num_arc_segments)
        chord_x = (1 - t_interp) * start_point[0] + t_interp * end_point[0]
        chord_y = (1 - t_interp) * start_point[1] + t_interp * end_point[1]
        chord_segment = np.stack([chord_x, chord_y], axis=1)

        straight_chord_coords.append(chord_segment)
        target_arc_coords.append(arc_segment)

    target_arc_coords = np.vstack(target_arc_coords)  # (N*num_segments, 2)
    straight_chord_coords = np.vstack(straight_chord_coords)  # same shape

    # Apply the *same* Procrustes alignment to the high-res arc points
    target_arc_coords_translated = (target_arc_coords @ rotation_matrix) + translation_vector

    return target_arc_coords_translated, straight_chord_coords


# ---------- Helper Utilities ----------

def parse_sol_string(sol_str):
    """
    Parse sol string like "0-1-12-11-2-5-8-13-14-3-10-9-7-6-4-0".
    Return numpy array of length N containing node indices (unique, 0..N-1).
    If sol_str is invalid return None.
    """
    if pd.isna(sol_str):
        return None
    try:
        parts = [int(x) for x in str(sol_str).strip().split('-') if x != '']
    except Exception:
        return None
    if len(parts) < 2:
        return None
    # If last equals first, drop the final duplicate to get a single cycle listing.
    if parts[0] == parts[-1]:
        parts = parts[:-1]
    return np.array(parts, dtype=np.int64)


def calculate_total_cycle_length(points, path_indices):
    """Compute total euclidean cycle length for the given path order (closed cycle)."""
    N = len(path_indices)
    total = 0.0
    for i in range(N):
        a = points[path_indices[i]]
        b = points[path_indices[(i + 1) % N]]
        total += np.linalg.norm(a - b)
    return float(total)


def greedy_tsp(points, start=0):
    """
    Very simple greedy TSP (fast, approximate): start at node 0, always go to nearest unvisited.
    Returns a cycle ordering array of length N.
    """
    N = len(points)
    visited = [False] * N
    order = [start]
    visited[start] = True
    current = start
    for _ in range(N - 1):
        dists = np.linalg.norm(points - points[current], axis=1)
        dists = np.where(np.array(visited), np.inf, dists)
        nxt = int(np.argmin(dists))
        order.append(nxt)
        visited[nxt] = True
        current = nxt
    return np.array(order, dtype=np.int64)


# ---------- CSV parsing / dataset construction ----------

def process_csv_to_dataset(csv_path, recompute_missing=False, num_arc_segments=NUM_ARC_SEGMENTS):
    """
    Reads CSV and returns a list of dicts (one dict per row / problem instance).
    Each dict contains torch-friendly numpy arrays (float32/int64).
    """
    df = pd.read_csv(csv_path)
    # detect N by counting columns that start with 'X_' or 'Y_'
    x_cols = [c for c in df.columns if c.startswith('X_')]
    y_cols = [c for c in df.columns if c.startswith('Y_')]
    if len(x_cols) == 0 or len(y_cols) == 0:
        raise ValueError("CSV must contain 'X_i' and 'Y_i' columns.")
    # sort columns to ensure X_0..X_{N-1} order
    x_cols = sorted(x_cols, key=lambda s: int(s.split('_')[1]))
    y_cols = sorted(y_cols, key=lambda s: int(s.split('_')[1]))

    N = len(x_cols)
    if len(y_cols) != N:
        raise ValueError("Number of X and Y columns mismatch.")

    dataset_entries = []
    for idx, row in df.iterrows():
        # build points (N,2)
        xs = row[x_cols].to_numpy(dtype=np.float64)
        ys = row[y_cols].to_numpy(dtype=np.float64)
        points = np.stack([xs, ys], axis=1)  # (N,2)

        # parse sol
        sol = row.get('sol', None) if 'sol' in df.columns else None
        path_indices = parse_sol_string(sol)
        if path_indices is None:
            if recompute_missing:
                path_indices = greedy_tsp(points, start=0)  # Use raw points for greedy
            else:
                continue
        else:
            # basic validation
            unique_nodes = np.unique(path_indices)
            if len(unique_nodes) != N or set(unique_nodes.tolist()) != set(range(N)):
                if recompute_missing:
                    path_indices = greedy_tsp(points, start=0)  # Use raw points for greedy
                else:
                    continue

        # --- Start of Refactored Logic ---

        # 1. Normalize the points
        normalized_points = normalize_points(points)

        # 2. Calculate total length on the *normalized* points
        total_length = calculate_total_cycle_length(normalized_points, path_indices)

        # 3. Create the initial, origin-centered circle nodes
        circle_coords, path_coords, R, cum_angles, path_indices = \
            create_circle_nodes(normalized_points, path_indices, total_length)

        # 4. Find optimal alignment (Procrustes)
        #    We align the circle_coords (B) to the path_coords (A)
        aligned_circle_nodes, Rot, t = align_clouds_procrustes(path_coords, circle_coords)

        # 5. Create high-resolution arc and chord data
        arc_coords, chord_coords = create_arc_data(path_coords, R, cum_angles, Rot, t, num_arc_segments)

        # 6. Re-order the aligned nodes to match the *original* point cloud order
        #    'aligned_circle_nodes' is currently in path order.
        inv_path = [i for i in range(len(path_indices))]
        for i, ind in enumerate(path_indices):
            inv_path[ind] = i

        final_circle_nodes = (aligned_circle_nodes)[inv_path]

        # --- End of Refactored Logic ---

        entry = {
            'points': normalized_points.astype(np.float32),  # original point cloud (N,2)
            'circle': final_circle_nodes.astype(np.float32),  # target nodes placed on circle (N,2)
            'path': path_indices.astype(np.int64),  # ordering of nodes
            'total_length': float(total_length),
            'R': float(R),
            'arc_coords': arc_coords.astype(np.float32),
            'chord_coords': chord_coords.astype(np.float32),
            'translation_vector': t.astype(np.float32)
        }
        dataset_entries.append(entry)
    return dataset_entries


# ---------- Small plotting helper (optional) ----------

def quick_plot_sample(entry, show=True, ax=None):
    """
    entry: dict as returned by process_csv_to_dataset (numpy arrays).
    This plots original points and circle nodes (no connecting lines), different colors per node.
    Requires matplotlib.
    """
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
    # color each point by index
    cmap = plt.get_cmap('tab20')
    for i in range(N):
        ax.scatter(pts[i, 0], pts[i, 1], marker='o', s=40, color=cmap(i % 20))
        ax.scatter(circle[i, 0], circle[i, 1], marker='x', s=40, color=cmap(i % 20))
    ax.set_aspect('equal', 'box')
    ax.set_title('points (o) and circle nodes (x)')
    if show:
        plt.show()


# ---------- Main / CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Convert TSP CSV -> PyTorch dataset for flow matching.")
    parser.add_argument('--csv', default=r"tspData.csv", help="Path to CSV file (with 'sol', 'X_i', 'Y_i' columns).")
    parser.add_argument('--out', default='processed_tsp_dataset.pt', help="Output .pt file to save processed dataset.")
    parser.add_argument('--recompute-missing', action='store_true',
                        help="If sol is missing/invalid, use greedy TSP fallback.")
    parser.add_argument('--num-arc-segments', type=int, default=NUM_ARC_SEGMENTS,
                        help="Number of arc segments per edge.")
    parser.add_argument('--preview-first', action='store_true',
                        help="Show a quick plot of the first sample (matplotlib required).")
    args = parser.parse_args()

    # Continue with a default CSV path if args.csv is the default and doesn't exist
    default_csv_path = "tspData.csv"
    if args.csv == default_csv_path and not os.path.exists(default_csv_path):
        print(f"Warning: Default '{default_csv_path}' not found.")
        # Try to find a common alternative path for local/colab testing
        alt_csv_path = r"C:\Users\Benjy\Downloads\archive (1)\tspData.csv"
        if os.path.exists(alt_csv_path):
            print(f"Using alternative path: {alt_csv_path}")
            args.csv = alt_csv_path
        else:
            print("No valid CSV found. Please specify --csv.")
            return

    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found at {args.csv}")
        return

    print("Reading CSV:", args.csv)
    # Added error handling for missing 'sol' column, per your original script's logic
    try:
        entries = process_csv_to_dataset(args.csv, recompute_missing=args.recompute_missing,
                                         num_arc_segments=args.num_arc_segments)
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"Error processing CSV: {e}")
        print("If 'sol' column is missing or invalid, try running with --recompute-missing")
        return

    if not entries:
        print("No valid entries were processed.")
        return

    print(f"Processed {len(entries)} instances. Saving to {args.out} ...")
    # Save raw list of dicts (numpy arrays) with torch.save
    torch.save(entries, args.out)
    print("Saved.")

    if args.preview_first and len(entries) > 0:
        quick_plot_sample(entries[0])


if __name__ == '__main__':
    main()