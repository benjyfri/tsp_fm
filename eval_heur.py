import torch
import numpy as np
import os
import argparse
import scipy.spatial
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from collections import defaultdict

# --- Import all heuristic solvers from python-tsp ---
from python_tsp.heuristics import (
    solve_tsp_local_search,
    solve_tsp_simulated_annealing,
    solve_tsp_lin_kernighan,
    solve_tsp_record_to_record
)

# --- DATA LOADER CLASS (copied from previous script) ---

class PointCloudDataset(torch.utils.data.Dataset):
    """
    Loads the 'processed_tsp_dataset.pt' file.
    Each item returns the (noisy_input, tar_circle, total_length) tuple.
    """
    def __init__(self, data_file):
        try:
            self.entries = torch.load(data_file, weights_only=False)
            print(f"Successfully loaded {len(self.entries)} total data entries from {data_file}")
        except FileNotFoundError:
            print(f"Error: File not found at {data_file}. Please check the path.")
            self.entries = []
        except Exception as e:
            print(f"Error loading file: {e}")
            self.entries = []

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # The flow starts at 'points' (x0) and ends at 'circle' (x1)
        x0 = torch.from_numpy(entry['points'].astype(np.float32))
        x1 = torch.from_numpy(entry['circle'].astype(np.float32))

        # Get the total_length (scalar)
        total_length = torch.tensor(entry['total_length'], dtype=torch.float32)

        return x0, x1, total_length

# --- END OF DATA LOADER ---


def main(args):
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Data
    print("Loading data...")
    full_dataset = PointCloudDataset(args.data_path)

    if not full_dataset.entries:
        print("Failed to load dataset. Exiting.")
        return

    # Replicate the train/test split
    test_size = len(full_dataset) - args.train_size
    if test_size <= 0:
        print(f"Error: Train size {args.train_size} is >= total dataset size {len(full_dataset)}")
        return

    print(f"Splitting dataset: {args.train_size} train, {test_size} test.")
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, test_dataset = random_split(
        full_dataset,
        [args.train_size, test_size],
        generator=generator
    )

    # Create a DataLoader for the test set
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # --- Define all heuristics to be tested ---

    # Local Search has multiple perturbation schemes
    local_search_perturbations = [
        'two_opt', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6'
    ]

    # Dictionary to hold the error lists for each method
    # defaultdict(list) automatically creates an empty list for new keys
    all_percentage_errors = defaultdict(list)

    print("Calculating statistics for all python-tsp heuristics...")

    # Wrap test_loader with tqdm for a progress bar
    progress_bar = tqdm(test_loader, desc="Evaluating All Heuristics", unit="batch")

    for batch in progress_bar:
        # Get data from batch
        x0_batch, _, gt_length_batch = batch

        # Iterate over each sample in the batch
        for i in range(x0_batch.shape[0]):
            x0_sample_tensor = x0_batch[i]
            gt_length_sample = gt_length_batch[i].item()

            # Convert to NumPy for python-tsp and scipy
            x0_sample_np = x0_sample_tensor.cpu().numpy()

            # 1. Create the (N, N) Euclidean distance matrix
            distance_matrix = scipy.spatial.distance.cdist(
                x0_sample_np,
                x0_sample_np,
                'euclidean'
            )

            # --- 2. Solve with each heuristic ---

            # Helper function to calculate and store error
            def eval_and_store(name, distance):
                error = (distance - gt_length_sample) / gt_length_sample
                all_percentage_errors[name].append(error)

            # --- Local Search (all 7 perturbations) ---
            for scheme in local_search_perturbations:
                name = f"Local Search ({scheme})"
                _, distance = solve_tsp_local_search(
                    distance_matrix,
                    perturbation_scheme=scheme
                )
                eval_and_store(name, distance)

            # --- Simulated Annealing ---
            _, sa_distance = solve_tsp_simulated_annealing(distance_matrix)
            eval_and_store("Simulated Annealing", sa_distance)

            # --- Lin-Kernighan ---
            # Lin-Kernighan implementation might be sensitive, wrap in try-except
            try:
                _, lk_distance = solve_tsp_lin_kernighan(distance_matrix)
                eval_and_store("Lin-Kernighan", lk_distance)
            except Exception as e:
                # Store NaN or skip if it fails on an instance
                all_percentage_errors["Lin-Kernighan"].append(np.nan)


            # --- Record-to-Record ---
            try:
                _, rtr_distance = solve_tsp_record_to_record(distance_matrix)
                eval_and_store("Record-to-Record", rtr_distance)
            except Exception as e:
                all_percentage_errors["Record-to-Record"].append(np.nan)

    # --- 3. Compute and print statistics ---

    print("\n" + "="*80)
    print(f"--- TSP Heuristic Evaluation (Evaluated on {test_size} test samples) ---")
    print("Percentage Error: (Heuristic_Length - GT_Length) / GT_Length")
    print(f"{'-'*80}")
    print(f"{'Heuristic':<30} | {'Mean Error (%)':<15} | {'Std Dev (%)':<15} | {'Min Error (%)':<15} | {'Max Error (%)':<15}")
    print(f"{'-'*80}")

    # Sort keys for a consistent print order
    sorted_keys = sorted(all_percentage_errors.keys())

    for heuristic_name in sorted_keys:
        errors_array = np.array(all_percentage_errors[heuristic_name])

        # Filter out potential NaNs if a solver failed
        errors_array = errors_array[~np.isnan(errors_array)]
        if len(errors_array) == 0:
            print(f"{heuristic_name:<30} | {'Failed on all samples'.center(66)}")
            continue

        mean_err = np.mean(errors_array)
        std_err = np.std(errors_array)
        min_err = np.min(errors_array)
        max_err = np.max(errors_array)

        print(f"{heuristic_name:<30} | {mean_err*100:^15.4f} | {std_err*100:^15.4f} | {min_err*100:^15.4f} | {max_err*100:^15.4f}")

    print("="*80 + "\n")
    print("Heuristic evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ALL python-tsp heuristics on the TSP dataset")

    # Data Args
    parser.add_argument('--data_path', type=str, default="data_old_scripts/processed_tsp_dataset.pt", help='Path to the processed_tsp_dataset.pt file')
    parser.add_argument('--train_size', type=int, default=15000,
                        help='Number of samples for the training set (to ensure correct test split)')

    # Eval Args
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loading')
    # Removed --perturbation arg

    args = parser.parse_args()
    main(args)