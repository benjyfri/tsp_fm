#!/usr/bin/env python3
"""
Data preprocessing using geomstats for Kendall shape space.

REVISION NOTES:
1. Forces all TSP solutions to be Counter-Clockwise (CCW) by checking signed area.
2. Builds all Target Circles Counter-Clockwise (CCW).
3. Ensures consistent orientation for turning angle calculations.
4. Added option to create full 'real' dataset without slicing.
"""

import argparse
import numpy as np
import torch
import os
import sys
from tqdm import tqdm
from math import pi
import multiprocessing

# Geomstats imports
import geomstats.backend as gs
from geomstats.geometry.pre_shape import PreShapeSpace
from geomstats.geometry.matrices import Matrices


# ============================================================================
# Geometry Helpers
# ============================================================================

def get_signed_area(coords: np.ndarray) -> float:
    """
    Calculates the signed area of a polygon to determine winding direction.
    Coords must be ordered by the tour path.

    Returns:
        Positive (> 0): Counter-Clockwise (CCW)
        Negative (< 0): Clockwise (CW)
    """
    x = coords[:, 0]
    y = coords[:, 1]
    # Shoelace formula (vectorized)
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def parse_tsp_lib_line(line: str):
    """
    Parse a single line from TSP .txt file.
    Format: "x1 y1 ... xN yN output i1 ... iN i1"
    """
    line = line.strip()
    if not line:
        return None

    parts = line.split(" output ")
    if len(parts) != 2:
        return None

    coord_str, sol_str = parts

    try:
        # Parse coordinates
        coords_flat = [float(x) for x in coord_str.split()]
        N = len(coords_flat) // 2

        if len(coords_flat) % 2 != 0 or N == 0:
            return None

        points = np.array(coords_flat, dtype=np.float64).reshape(N, 2)

        # Parse path (1-indexed, closed)
        path_1_indexed = [int(x) for x in sol_str.split()]

        if len(path_1_indexed) < 2 or path_1_indexed[0] != path_1_indexed[-1]:
            return None

        # Convert to 0-indexed, remove closing vertex
        path_0_indexed = [x - 1 for x in path_1_indexed[:-1]]

        if len(path_0_indexed) != N:
            return None

        if set(path_0_indexed) != set(range(N)):
            return None

        return points, np.array(path_0_indexed, dtype=np.int64)

    except Exception:
        return None


# ============================================================================
# Circle Construction (Strictly CCW)
# ============================================================================

def calculate_edge_lengths(points: np.ndarray, path_indices: np.ndarray) -> np.ndarray:
    """Calculate edge lengths along TSP tour."""
    path_coords = points[path_indices]
    next_coords = np.roll(path_coords, -1, axis=0)
    diffs = path_coords - next_coords
    edge_lengths = np.linalg.norm(diffs, axis=1)
    return edge_lengths.astype(np.float64)


def create_circle_from_lengths(path_indices: np.ndarray,
                               edge_lengths: np.ndarray) -> np.ndarray:
    """
    Create circle configuration preserving edge lengths.
    Constructed strictly Counter-Clockwise (CCW).
    """
    N = len(path_indices)
    total_length = np.sum(edge_lengths)

    # Radius of circle
    R = total_length / (2.0 * pi)

    # Central angles for each edge
    central_angles = edge_lengths / R

    # Place points on circle (starting at (0, R))
    start_angle = pi / 2.0
    cumulative_angles = np.zeros(N, dtype=np.float64)
    cumulative_angles[0] = start_angle

    # --- FORCE CCW CONSTRUCTION ---
    # Use (+) to move Counter-Clockwise from pi/2
    for i in range(1, N):
        cumulative_angles[i] = cumulative_angles[i-1] + central_angles[i-1]

    # Convert to Cartesian coordinates
    circle_x = R * np.cos(cumulative_angles)
    circle_y = R * np.sin(cumulative_angles)
    circle_coords_path_order = np.stack([circle_x, circle_y], axis=1)

    # Reorder to original vertex order so indices match 'points'
    inv_path = np.empty(N, dtype=np.int64)
    inv_path[path_indices] = np.arange(N)
    circle_coords_original_order = circle_coords_path_order[inv_path]

    return circle_coords_original_order


# ============================================================================
# Geomstats Processing
# ============================================================================

def process_with_geomstats(points: np.ndarray,
                           circle: np.ndarray,
                           space: PreShapeSpace) -> dict:
    """
    Process points and circle using geomstats.
    Both inputs are expected to be CCW aligned before entering here.
    """
    # Project points onto pre-shape space
    x0 = space.projection(points)
    x1_raw = space.projection(circle)

    # Align circle to points using optimal rotation
    x1 = Matrices.align_matrices(x1_raw, x0)

    # Compute geodesic distance (angle on sphere)
    metric = space.metric
    theta = metric.dist(x0, x1)

    return {
        'points': x0.astype(np.float32),  # x0
        'circle': x1.astype(np.float32),  # x1
        'theta': float(theta),
    }


# ============================================================================
# Worker Function
# ============================================================================

def process_line_worker(args_tuple):
    """
    Worker function:
    1. Parses line
    2. CHECKS WINDING of TSP -> Reverses path if Clockwise
    3. Generates CCW Circle
    4. Aligns in Shape Space
    """
    line_idx, line, n_points, ambient_dim = args_tuple

    try:
        # Parse line
        parsed = parse_tsp_lib_line(line)
        if parsed is None:
            return line_idx, "Invalid format"

        points, path = parsed

        # Verify number of points
        if len(points) != n_points:
            return line_idx, f"Expected {n_points} points, got {len(points)}"

        # --- STEP 1: ENFORCE CCW ORIENTATION ON TSP ---
        # We check the signed area of the points in the order of the path
        current_winding = get_signed_area(points[path])

        if current_winding < 0:
            # It is Clockwise. We must reverse the path.
            # Reversing the path makes the traversal Counter-Clockwise.
            path = path[::-1]

            # Note: We do NOT change the 'points' array coordinates,
            # only the order in which we visit them (the path).
        # ----------------------------------------------

        # Create geomstats space
        space = PreShapeSpace(k_landmarks=n_points, ambient_dim=ambient_dim, equip=True)

        # Calculate edge lengths (using the potentially reversed, now CCW path)
        edge_lengths = calculate_edge_lengths(points, path)

        # Create circle
        # (This function now builds strictly CCW using the '+' operator)
        circle = create_circle_from_lengths(path, edge_lengths)

        # Process with geomstats
        result = process_with_geomstats(points, circle, space)

        # Add path and edge lengths
        result['path'] = path.astype(np.int64)
        result['edge_lengths'] = edge_lengths.astype(np.float32)

        return line_idx, result

    except Exception as e:
        return line_idx, f"Error: {str(e)}"


# ============================================================================
# Main Processing
# ============================================================================

def process_txt_to_dataset(txt_path: str,
                           num_points: int,
                           num_workers: int = None) -> list:
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        if not lines:
            print(f"Warning: File {txt_path} is empty")
            return []
    except Exception as e:
        print(f"Error reading file {txt_path}: {e}")
        return []

    print(f"Processing {len(lines)} samples from {txt_path}")

    if num_workers is None:
        slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
        if slurm_cpus:
            num_workers = int(slurm_cpus)
        else:
            num_workers = multiprocessing.cpu_count()

    print(f"Using {num_workers} workers")

    args_list = [
        (i, line, num_points, 2)
        for i, line in enumerate(lines)
    ]

    dataset = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        results_iter = pool.imap_unordered(process_line_worker, args_list)

        for line_idx, result in tqdm(results_iter, total=len(lines), desc="Processing"):
            if isinstance(result, dict):
                dataset.append(result)
            else:
                # Log first few errors
                if line_idx < 5:
                    tqdm.write(f"Line {line_idx}: {result}")

    return dataset


def plot_sample(entry, save_path=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    points = entry['points']
    circle = entry['circle']
    path = entry['path']
    theta = entry['theta']

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot TSP path to verify orientation visually
    tsp_ordered = points[path]
    # Close the loop
    tsp_plot = np.vstack([tsp_ordered, tsp_ordered[0]])

    ax.plot(tsp_plot[:,0], tsp_plot[:,1], 'b-', label='TSP Path (CCW)', alpha=0.5)
    ax.scatter(points[:, 0], points[:, 1], c='b', s=20)

    # Plot Circle
    circle_ordered = circle[path]
    circle_plot = np.vstack([circle_ordered, circle_ordered[0]])
    ax.plot(circle_plot[:,0], circle_plot[:,1], 'r--', label='Circle (CCW)', alpha=0.5)

    ax.set_aspect('equal')
    ax.set_title(f'Geodesic Distance θ = {theta:.4f}')
    ax.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess TSP data using geomstats (Forces CCW Orientation)"
    )
    # Input Config
    parser.add_argument('--infile', type=str, default="../tsp50_val_concorde.txt")
    parser.add_argument('--num_points', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=32)

    # Real Dataset Creation Options
    parser.add_argument('--create_real_dataset', action='store_true',
                        help="If set, processes the entire file into one .pt file (ignores demo sizes).")
    parser.add_argument('--real_out_path', type=str, default='processed_data_geom_val.pt',
                        help="Output path when creating the real dataset.")

    # Demo Slice Options (Used if --create_real_dataset is NOT set)
    parser.add_argument('--demo-train-size', type=int, default=100)
    parser.add_argument('--demo-val-size', type=int, default=10)
    parser.add_argument('--demo-train-out', type=str, default='geom_demo_train_N100.pt')
    parser.add_argument('--demo-val-out', type=str, default='geom_demo_val_N10.pt')

    # Debugging / Plotting
    parser.add_argument('--plot_first', action='store_true')
    parser.add_argument('--plot_path', type=str, default='first_sample.png')

    args = parser.parse_args()

    if not os.path.exists(args.infile):
        print(f"Error: Input file not found: {args.infile}")
        sys.exit(1)

    # 1. Process the entire file (or as much as possible)
    dataset = process_txt_to_dataset(
        txt_path=args.infile,
        num_points=args.num_points,
        num_workers=args.num_workers
    )

    if not dataset:
        print("No valid samples processed. Exiting.")
        sys.exit(1)

    print(f"\nSuccessfully processed {len(dataset)} samples (All CCW oriented)")

    # 2. Optional: Plot first sample
    if args.plot_first and len(dataset) > 0:
        plot_sample(dataset[0], save_path=args.plot_path)

    # 3. Save Output
    if args.create_real_dataset:
        # --- REAL DATASET MODE ---
        print(f"Saving FULL dataset ({len(dataset)} samples) to {args.real_out_path}...")
        torch.save(dataset, args.real_out_path)
        print("Done.")
    else:
        # --- DEMO / SLICING MODE ---
        required_size = args.demo_train_size + args.demo_val_size
        if len(dataset) < required_size:
            print(f"Warning: Not enough samples ({len(dataset)}) for requested demo sizes ({required_size}).")
            sys.exit(1)

        demo_train = dataset[0 : args.demo_train_size]
        demo_val = dataset[args.demo_train_size : required_size]

        torch.save(demo_train, args.demo_train_out)
        torch.save(demo_val, args.demo_val_out)
        print(f"Saved demo slices: {args.demo_train_out} and {args.demo_val_out}")

if __name__ == '__main__':
    main()