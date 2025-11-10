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


import numpy as np
from numpy.linalg import svd, norm


import numpy as np
from numpy.linalg import svd, norm, det

def find_optimal_transform(A: np.ndarray, B: np.ndarray):
    """
    [CORRECTED] Finds the optimal *rotation* (det(Q)=+1)
    to align B to A (Orthogonal Procrustes problem).
    Assumes A and B are already centered and scaled.
    Minimizes: || A - B @ Q ||^2

    Args:
        A: The (N, 2) target point cloud (pre-processed).
        B: The (N, 2) source point cloud (pre-processed).

    Returns:
        A tuple containing:
        - B_aligned (np.ndarray): The (N, 2) aligned cloud (B @ Q).
        - Q (np.ndarray): The (2, 2) optimal rotation matrix.
    """
    # Compute covariance matrix M = A.T @ B
    M = A.T @ B  # Shape: (2, N) @ (N, 2) = (2, 2)

    # Compute SVD of M
    try:
        U, s, Vh = svd(M)
    except np.linalg.LinAlgError:
        return B, np.identity(2)  # Fallback

    V = Vh.T  # Vh is V-transpose, so we transpose it back to get V

    # --- THIS IS THE CRITICAL FIX ---
    # 1. Calculate the optimal orthogonal matrix
    Q = V @ U.T

    # 2. Check for a reflection (det(Q) == -1)
    if det(Q) < 0:
        # print("Reflection detected, correcting...") # Optional: for debugging

        # We must "flip" the solution to force a rotation.
        # We do this by flipping the sign of the last column of V
        # (which corresponds to the smallest singular value).
        V_corrected = V.copy()
        V_corrected[:, -1] *= -1

        # 3. Re-calculate Q with the corrected V
        Q = V_corrected @ U.T

        # Double-check: det(Q) should now be +1
        # assert np.isclose(det(Q), 1.0)

    # Apply the optimal *rotation*: B_aligned = B @ Q
    B_aligned = B @ Q  # Shape: (N, 2) @ (2, 2) = (N, 2)

    return B_aligned, Q
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


# In createdata.py

def calculate_turning_angles(points: np.ndarray, path_indices: np.ndarray) -> np.ndarray:
    """
    [FIXED LOGIC] Computes the interior turning angle (up to 2*pi).
    This version flips the convex/reflex logic to match the (likely)
    Clockwise (CW) winding order of the TSP solver.
    """
    N = len(path_indices)
    path_coords = points[path_indices]  # Points in path order
    turning_angles = []

    for i in range(N):
        p_prev = path_coords[(i - 1) % N]
        p_curr = path_coords[i]
        p_next = path_coords[(i + 1) % N]

        v1 = p_prev - p_curr  # Vector from current to previous
        v2 = p_next - p_curr  # Vector from current to next

        norm_v1 = norm(v1)
        norm_v2 = norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            angle = 0.0  # Handle collinear/degenerate case
        else:
            v1_u = v1 / norm_v1
            v2_u = v2 / norm_v2

            # 1. Calculate the convex angle (always <= pi or 180 deg)
            dot_prod = np.dot(v1_u, v2_u)
            convex_angle = acos(np.clip(dot_prod, -1.0, 1.0))

            # 2. Calculate the "turn direction" using 2D cross-product (determinant)
            turn_direction_z = v1[0] * v2[1] - v1[1] * v2[0]

            # 3. [THE FIX IS HERE] Determine the final interior angle
            # Both polygons are now CCW.
            # For CCW, the interior (convex) angle has a NEGATIVE z-turn.

            if turn_direction_z > 0: # <-- This is the fix
                # "Left Turn" -> Convex angle (e.g., 30 deg)
                final_angle = convex_angle
            else:
                # "Right Turn" (or collinear) -> Reflex angle (e.g., 330 deg)
                final_angle = 2.0 * np.pi - convex_angle

            # A small correction: if the angle is ~360, it should be 0
            if np.isclose(final_angle, 2.0 * np.pi) and np.isclose(convex_angle, 0.0):
                final_angle = 0.0

            angle = final_angle

        turning_angles.append(angle)

    return np.array(turning_angles, dtype=np.float64)
def create_circle_nodes(path_indices: np.ndarray,
                        segment_lengths: np.ndarray,
                        total_cycle_length: float):
    """
    [NEW] Creates the initial (un-rotated, origin-centered) circle coordinates
    in the *original 0-to-N-1 point cloud order*.

    Args:
        path_indices: The (N,) array of node indices in TSP order.
        segment_lengths: The (N,) array of path segment lengths, corresponding
                         to the edges in path_indices.
        total_cycle_length: The total length of the TSP path.

    Returns:
        circle_coords_original_order (np.ndarray): The (N, 2) coordinates on
            the circle, centered at the origin, and ordered such that
            index 'i' corresponds to the original point cloud's index 'i'.
    """
    N = len(path_indices)

    # 1. Calculate circle radius
    R = total_cycle_length / (2.0 * pi)  # This is the scalar Radius

    # 2. Compute central angles for each path segment
    central_angles = segment_lengths / R  # shape (N,)

    # 3. Calculate cumulative angles for each node *in path order*
    start_angle = np.pi / 2.0  # Start at the top
    cumulative_angles = [start_angle]
    current_angle = start_angle
    for ang in central_angles[:-1]:
        current_angle -= ang
        cumulative_angles.append(current_angle)

    # cumulative_angles[i] is the angle for node path_indices[i]
    cumulative_angles = np.array(cumulative_angles, dtype=np.float64)  # shape (N,)

    # 4. Create origin-centered circle coordinates *in path order*
    circle_x = R * np.cos(cumulative_angles)
    circle_y = R * np.sin(cumulative_angles)
    circle_coords_path_order = np.stack([circle_x, circle_y], axis=1)  # (N,2)

    # 5. Create the inverse permutation (the "un-scrambler")
    # This maps the path order back to the original 0..N-1 order.
    inv_path = np.empty_like(path_indices)
    inv_path[path_indices] = np.arange(N)

    # 6. Apply the inverse permutation to get the 0..N-1 order
    circle_coords_original_order = circle_coords_path_order[inv_path]

    return circle_coords_original_order

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
    [REVISED] Reads a .txt file and returns a list of dicts.
    This version fixes the data inconsistency bug.
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

        # --- [FIX] Start of Corrected Geometric Logic ---

        # 1. [FIX] Create the final x0 FIRST.
        #    Use pre_process_shape on the *original* points.
        #    This is our single source of truth for x0.
        X0_processed = pre_process_shape(points)

        # 2. [FIX] Calculate all geometry from THIS x0.
        #    (Do NOT use normalized_points anymore)
        segment_lengths = calculate_edge_lengths(X0_processed, path_indices)
        turning_angles = calculate_turning_angles(X0_processed, path_indices)
        total_length = np.sum(segment_lengths)

        # 3. [FIX] Create the circle nodes from x0's actual geometry.
        circle_coords = create_circle_nodes(path_indices, segment_lengths, total_length)

        # 4. Calculate turning angles for the target circle
        #    (This part was fine, just uses the new circle_coords)
        circle_turning_angles = calculate_turning_angles(circle_coords, path_indices)

        # 5. [FIX] Pre-process the target circle. This is x1 (un-aligned).
        X1_processed = pre_process_shape(circle_coords)

        # 6. Find optimal alignment (rotation only)
        #    (This part was fine)
        X1_aligned, Rot = find_optimal_transform(X0_processed, X1_processed)

        # 7. Compute Procrustes distance (theta)
        #    (This part was fine)
        dot_prod = np.clip(np.sum(X0_processed * X1_aligned), -1.0, 1.0)
        theta = acos(dot_prod)

        # 8. Set final shapes
        X0_final = X0_processed
        X1_final = X1_aligned

        # --- End of Corrected Geometric Logic ---

        entry = {
            'points': X0_final.astype(np.float32),          # (N,2) X0
            'circle': X1_final.astype(np.float32),          # (N,2) X1 (aligned)
            'path': path_indices.astype(np.int64),          # (N,) ordering
            'theta': float(theta),                          # Procrustes distance
            'edge_lengths': segment_lengths.astype(np.float32), # (N,) from X0
            'turning_angles': turning_angles.astype(np.float32), # (N,) from X0

            # --- NEWLY ADDED PER REQUEST ---
            'circle_unprocessed': circle_coords.astype(np.float32),
            'circle_turning_angles': circle_turning_angles.astype(np.float32)
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
    parser.add_argument('--infile', default="../tsp50_val_concorde.txt",
                        help="Path to .txt file (e.g., tsp20.txt or tsp50.txt).")
    parser.add_argument('--out', default='processed_tsp_dataset_TSP50_val.pt', # <-- Updated default filename
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