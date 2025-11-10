import argparse
import numpy as np
import torch
from math import pi, acos
import os
import sys
from tqdm import tqdm
from numpy.linalg import norm, svd, det
import multiprocessing

# ---------- Core Data Processing Functions (Vectorized) ----------

def pre_process_shape(X: np.ndarray) -> np.ndarray:
    """
    [NEW] Pre-processes a shape for Kendall's shape space.
    """
    X_c = X - np.mean(X, axis=0)
    f_norm = norm(X_c, 'fro')
    if f_norm < 1e-9:
        return X_c
    return X_c / f_norm

def find_optimal_transform(A: np.ndarray, B: np.ndarray):
    """
    [YOUR ORIGINAL LOGIC] Finds the optimal *rotation* (det(Q)=+1)
    to align B to A.
    """
    M = A.T @ B
    try:
        U, s, Vh = svd(M)
    except np.linalg.LinAlgError:
        return B, np.identity(2)

    V = Vh.T
    Q = V @ U.T

    if det(Q) < 0:
        V_corrected = V.copy()
        V_corrected[:, -1] *= -1
        Q = V_corrected @ U.T

    B_aligned = B @ Q
    return B_aligned, Q

def calculate_edge_lengths(points: np.ndarray, path_indices: np.ndarray) -> np.ndarray:
    """
    [VECTORIZED] Computes the Euclidean length of each edge in the TSP cycle.
    """
    path_coords = points[path_indices]
    next_coords = np.roll(path_coords, -1, axis=0)
    diffs = path_coords - next_coords
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return segment_lengths.astype(np.float64)


def calculate_turning_angles(points: np.ndarray, path_indices: np.ndarray) -> np.ndarray:
    """
    [VECTORIZED & CW-FIXED]
    Computes the interior turning angle for CW polygons.
    """
    N = len(path_indices)
    path_coords = points[path_indices]
    p_prev = np.roll(path_coords, 1, axis=0)
    p_curr = path_coords
    p_next = np.roll(path_coords, -1, axis=0)

    v1 = p_prev - p_curr
    v2 = p_next - p_curr
    norm_v1 = np.linalg.norm(v1, axis=1)
    norm_v2 = np.linalg.norm(v2, axis=1)

    valid_mask = (norm_v1 > 1e-9) & (norm_v2 > 1e-9)
    turning_angles = np.zeros(N, dtype=np.float64)

    if np.any(valid_mask):
        v1_u = v1[valid_mask] / norm_v1[valid_mask, np.newaxis]
        v2_u = v2[valid_mask] / norm_v2[valid_mask, np.newaxis]

        dot_prod = np.sum(v1_u * v2_u, axis=1)
        convex_angle = np.arccos(np.clip(dot_prod, -1.0, 1.0))

        v1_valid = v1[valid_mask]
        v2_valid = v2[valid_mask]
        turn_direction_z = v1_valid[:, 0] * v2_valid[:, 1] - v1_valid[:, 1] * v2_valid[:, 0]

        # --- [THIS IS THE CW FIX] ---
        # For a CW polygon, the interior angle has turn_direction_z < 0
        final_angle = np.where(turn_direction_z > 0,
                               convex_angle,
                               2.0 * np.pi - convex_angle)
        # --- END OF FIX ---

        # Correction for 0-degree angles
        zero_angle_mask = np.isclose(final_angle, 2.0 * np.pi) & np.isclose(convex_angle, 0.0)
        final_angle[zero_angle_mask] = 0.0

        turning_angles[valid_mask] = final_angle

    return turning_angles

def create_circle_nodes(path_indices: np.ndarray,
                        segment_lengths: np.ndarray,
                        total_cycle_length: float):
    """
    [VECTORIZED & YOUR ORIGINAL CW LOGIC] Creates a CW circle.
    """
    N = len(path_indices)
    R = total_cycle_length / (2.0 * pi)
    central_angles = segment_lengths / R

    start_angle = np.pi / 2.0
    cumulative_angles = np.empty(N, dtype=np.float64)
    cumulative_angles[0] = start_angle

    # We SUBTRACT angles to move CW around the circle
    cumulative_angles[1:] = start_angle - np.cumsum(central_angles[:-1])

    circle_x = R * np.cos(cumulative_angles)
    circle_y = R * np.sin(cumulative_angles)
    circle_coords_path_order = np.stack([circle_x, circle_y], axis=1)

    inv_path = np.empty_like(path_indices)
    inv_path[path_indices] = np.arange(N)

    circle_coords_original_order = circle_coords_path_order[inv_path]
    return circle_coords_original_order


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


# ---------- [RESTORED] Worker Function for Multiprocessing ----------
def process_line(line_tuple):
    """
    Worker function that processes a single line.

    [RESTORED as per user request]
    This function now performs TWO calculations:
    1.  The KENDALL'S SHAPE SPACE calculation to get the
        aligned 'points' (X0_final) and 'circle' (X1_final) and 'theta'.
    2.  The MAX-MAGNITUDE NORMALIZATION calculation to
        calculate 'edge_lengths', 'turning_angles', 'circle_unprocessed',
        and 'circle_turning_angles' as requested.
    """
    line_idx, line = line_tuple

    try:
        parsed_data = parse_tsp_lib_line(line)
        if parsed_data is None:
            return line_idx, "Invalid format"

        # 'points' here are the raw, original coordinates
        points, path_indices = parsed_data

        # --- 1. KENDALL'S SHAPE SPACE LOGIC (Stays the Same) ---
        # This path calculates the values for 'points', 'circle', and 'theta'

        X0_processed = pre_process_shape(points)

        # We must build the KENDALL-based circle for alignment
        # Note: We use the CW-corrected turning angle function here
        kendall_segment_lengths = calculate_edge_lengths(X0_processed, path_indices)
        kendall_total_length = np.sum(kendall_segment_lengths)
        kendall_circle_coords = create_circle_nodes(path_indices,
                                                    kendall_segment_lengths,
                                                    kendall_total_length)

        X1_processed = pre_process_shape(kendall_circle_coords)
        X1_aligned, Rot = find_optimal_transform(X0_processed, X1_processed)

        dot_prod = np.clip(np.sum(X0_processed * X1_aligned), -1.0, 1.0)
        theta = acos(dot_prod)

        X0_final = X0_processed
        X1_final = X1_aligned
        # --- End of Kendall Logic ---


        # --- 2. NEW NORMALIZATION LOGIC (User Request) ---
        # This path calculates the values for 'edge_lengths', 'turning_angles',
        # 'circle_unprocessed', and 'circle_turning_angles'.

        # a. The original point cloud will be centered at (0,0)
        points_centered = points - np.mean(points, axis=0)

        # b. Then, it will be normalized such that the largest point magnitude is 1
        magnitudes = np.linalg.norm(points_centered, axis=1)
        max_mag = np.max(magnitudes)

        if max_mag < 1e-9:
            points_norm_new = points_centered # Handle degenerate case
        else:
            points_norm_new = points_centered / max_mag

        # c. Calculate edge lengths from this new normalized shape
        #    (Use CW-corrected angle function)
        new_edge_lengths = calculate_edge_lengths(points_norm_new, path_indices)

        # d. Calculate turning angles from this new normalized shape
        new_turning_angles = calculate_turning_angles(points_norm_new, path_indices)

        # c. (continued) Build the corresponding circle
        new_total_length = np.sum(new_edge_lengths)
        new_circle_coords = create_circle_nodes(path_indices,
                                                new_edge_lengths,
                                                new_total_length)

        # d. (continued) Calculate turning angles for the new circle
        #    (Use CW-corrected angle function)
        new_circle_turning_angles = calculate_turning_angles(new_circle_coords,
                                                             path_indices)
        # --- End of New Normalization Logic ---


        # --- 3. ASSEMBLE FINAL DICTIONARY ---
        entry = {
            # From Kendall Logic
            'points': X0_final.astype(np.float32),
            'circle': X1_final.astype(np.float32),
            'theta': float(theta),

            # From New Normalization Logic
            'edge_lengths': new_edge_lengths.astype(np.float32),
            'points_unprocessed': points_norm_new.astype(np.float32),
            'turning_angles': new_turning_angles.astype(np.float32),
            'circle_unprocessed': new_circle_coords.astype(np.float32),
            'circle_turning_angles': new_circle_turning_angles.astype(np.float32),

            # Other data
            'path': path_indices.astype(np.int64)
        }
        return line_idx, entry

    except Exception as e:
        return line_idx, f"Error: {str(e)}"

# ---------- NEW: Multiprocessed & Cluster-Aware dataset construction ----------
# (Unchanged)
def process_txt_to_dataset(txt_path, num_workers_arg):
    """
    [MULTIPROCESSED & CLUSTER-AWARE] Reads a .txt file.
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
    lines_with_indices = list(enumerate(lines))
    num_lines = len(lines_with_indices)

    # --- Cluster-Aware Core Counting ---
    num_cores = num_workers_arg

    if num_cores is None:
        slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
        if slurm_cpus:
            try:
                num_cores = int(slurm_cpus)
                print(f"Info: Detected SLURM_CPUS_PER_TASK. Using {num_cores} cores.")
            except ValueError:
                pass

    if num_cores is None:
        num_cores = multiprocessing.cpu_count()
        print(f"Warning: No --num-workers or SLURM_CPUS_PER_TASK detected.")
        print(f"         Falling back to all available cores on node: {num_cores}.")
        print(f"         On a cluster, it is highly recommended to use:")
        print(f"         --num-workers $SLURM_CPUS_PER_TASK")
    # --- End of Core Counting ---

    print(f"Starting processing of {num_lines} samples on {num_cores} cores...")

    with multiprocessing.Pool(processes=num_cores) as pool:
        results_iter = pool.imap_unordered(process_line, lines_with_indices)

        for line_idx, result in tqdm(results_iter, total=num_lines, desc=f"Processing {file_name}"):
            if isinstance(result, dict):
                dataset_entries.append(result)
            else:
                tqdm.write(f"Skipping line {line_idx+1}: {result}")

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


# ---------- [FIXED] Main / CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Convert TSP .txt dataset -> PyTorch dataset for flow matching.")
    parser.add_argument('--infile', default="../tsp50_train_concorde.txt",
                        help="Path to .txt file (e.g., tsp20.txt or tsp50.txt).")
    parser.add_argument('--out', default='processed_tsp_dataset_TSP50_train.pt',
                        help="Output .pt file to save processed dataset.")

    # --- CLUSTER-AWARE ARGUMENT ---
    parser.add_argument('--num-workers', type=int, default=32,
                        help="Number of cores to use. On SLURM, use $SLURM_CPUS_PER_TASK")

    parser.add_argument('--num-arc-segments', type=int, default=20,
                        help="[DEPRECATED] No longer used.")
    parser.add_argument('--preview-first', action='store_true',
                        help="Show a quick plot of the first sample (matplotlib required).")
    args = parser.parse_args()

    if not os.path.exists(args.infile):
        print(f"Error: Input file not found at {args.infile}")
        return

    print("Reading input file:", args.infile)
    try:
        entries = process_txt_to_dataset(args.infile, args.num_workers)
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        return

    if not entries:
        print("No valid entries were processed. Check file format or errors above.")
        return

    print(f"Processed {len(entries)} instances. Saving to {args.out} ...")
    torch.save(entries, args.out)
    print("Saved.")

    # --- [FIXED LINE] ---
    # Use args.preview_first (with an underscore)
    if args.preview_first and len(entries) > 0:
        quick_plot_sample(entries[0])


if __name__ == '__main__':
    main()