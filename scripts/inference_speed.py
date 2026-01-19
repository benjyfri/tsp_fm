import torch
import os
import sys
import time
import argparse
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# --- Environment Setup ---
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
torch.set_default_dtype(torch.float32)

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from src.models import VectorFieldModel
from src.utils import ode_solve_euler


# ==============================================================================
#  PART 1: PURE TORCH RECONSTRUCTION (GPU)
# ==============================================================================
def reconstruct_tour_gpu(final_configs):
    """
    Reconstructs the tour from the final point locations purely on GPU.
    Assumes final_configs is (B, N, 2) and forms a circle.
    """
    # 1. Calculate centroid
    centroid = final_configs.mean(dim=1, keepdim=True)  # (B, 1, 2)

    # 2. Center the points
    centered = final_configs - centroid

    # 3. Compute angles (atan2)
    # Result is (B, N) values in [-pi, pi]
    angles = torch.atan2(centered[..., 1], centered[..., 0])

    # 4. Sort indices by angle to recover the sequence
    # argsort returns the indices that would sort the array
    tour_indices = torch.argsort(angles, dim=1)  # (B, N)

    return tour_indices


# ==============================================================================
#  PART 2: FULLY VECTORIZED 2-OPT (NO LOOPS)
# ==============================================================================
@torch.jit.script
def batched_two_opt_vectorized(points, tour, max_iterations: int = 1000):
    """
    Fully vectorized 2-opt. Performs the 'flip' using a masked gather,
    avoiding per-batch python loops entirely.

    Args:
        points: (B, N, 2) coordinates
        tour:   (B, N+1) indices (cyclic)
    """
    batch_size, num_points_plus_1 = tour.shape
    num_points = num_points_plus_1 - 1
    device = points.device

    # Base grid for indexing: [[0, 1, ... N], [0, 1, ... N], ...]
    idx_grid = torch.arange(num_points_plus_1, device=device).unsqueeze(0).expand(batch_size, -1)

    iterator = 0
    while iterator < max_iterations:
        # --- 1. GATHER COORDINATES ---
        # tour shape: (B, N+1)
        # We want edges (i, i+1) vs (j, j+1)

        # tour indices excluding the last cyclic point for 'i' logic
        # But for 'gather' we need the full cyclic tour to get coords
        p_ordered = torch.gather(points, 1, tour[..., :-1].unsqueeze(-1).expand(-1, -1, 2))  # (B, N, 2)
        p_next = torch.roll(p_ordered, -1, dims=1)  # (B, N, 2) -> (i+1)

        # To optimize N^2 check, we use the gathered coordinates directly
        # Points i: (B, N, 1, 2)
        # Points j: (B, 1, N, 2)

        P_i = p_ordered.unsqueeze(2)
        P_j = p_ordered.unsqueeze(1)
        P_i_next = p_next.unsqueeze(2)
        P_j_next = p_next.unsqueeze(1)

        # --- 2. COMPUTE COST CHANGES ---
        # dist(i, j)
        d_ij = torch.norm(P_i - P_j, dim=-1)
        # dist(i+1, j+1)
        d_next = torch.norm(P_i_next - P_j_next, dim=-1)
        # dist(i, i+1) - current edge 1
        d_curr1 = torch.norm(P_i - P_i_next, dim=-1)
        # dist(j, j+1) - current edge 2
        d_curr2 = torch.norm(P_j - P_j_next, dim=-1)

        change = d_ij + d_next - d_curr1 - d_curr2

        # Mask invalid swaps (diagonal and lower triangle)
        # We only want i < j, and strictly non-adjacent edges usually
        # triu(k=2) excludes diagonal (0) and adjacent (1)
        valid_change = torch.triu(change, diagonal=2)

        # --- 3. FIND BEST MOVE ---
        min_change_val, flatten_argmin = torch.min(valid_change.view(batch_size, -1), dim=-1)

        # Check if we are done
        if (min_change_val >= -1e-6).all():
            break

        # --- 4. PERFORM SWAP (VECTORIZED FLIP) ---
        # Decode indices i and j
        # Note: argmin returns index in N*N flattened array
        best_i = torch.div(flatten_argmin, num_points, rounding_mode='floor')
        best_j = torch.remainder(flatten_argmin, num_points)

        # The 2-opt move reverses the segment [i+1, j]
        start = best_i + 1
        end = best_j

        # We need to reverse indices between 'start' and 'end'
        # Logic: New_Index[k] = (start + end) - k   IF   start <= k <= end
        #        New_Index[k] = k                   ELSE

        # Create masks for the range [start, end]
        # shape (B, N+1) to match index grid
        mask = (idx_grid >= start.unsqueeze(1)) & (idx_grid <= end.unsqueeze(1))

        # Calculate reversed indices
        # We need to clamp 'start+end' to avoid weird broadcasting, though shapes align (B, 1)
        sum_bound = start.unsqueeze(1) + end.unsqueeze(1)
        reversed_idxs = sum_bound - idx_grid

        # Apply the flip only where mask is True, otherwise keep original index
        target_gather_idxs = torch.where(mask, reversed_idxs, idx_grid)

        # Gather the new tour order
        # We are re-ordering the *values* in the tour tensor based on the calculated indices
        tour = torch.gather(tour, 1, target_gather_idxs)

        iterator += 1

    return tour


# ==============================================================================
#  BENCHMARKING LOGIC
# ==============================================================================
def benchmark_model(model_path, gpu_id=0, num_samples=1000, batch_size=None, force_N=None):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {os.path.basename(os.path.dirname(model_path))}")
    print(f"{'=' * 60}")

    # 1. Load Model
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except:
        checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'args' in checkpoint:
        model_args = argparse.Namespace(**checkpoint['args'])
        state_dict = checkpoint['model_state_dict']
    else:
        print("❌ Could not parse checkpoint args.")
        return

    # Clean state dict keys
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "").replace("module.", "")
        if "freqs_cis" in name: continue
        new_state_dict[name] = v

    # Initialize Model
    try:
        model = VectorFieldModel(model_args).to(device)
        model.load_state_dict(new_state_dict, strict=False)
    except:
        print("Warning: Could not load exact architecture, using generic VectorFieldModel")
        model = VectorFieldModel(model_args).to(device)

    model.eval()

    # OPTIONAL: Compile model as suggested by your friend
    # model = torch.compile(model)

    # Determine N
    if force_N is not None:
        N = force_N
        print(f"--> Overriding N with manually provided value: {N}")
    else:
        N = getattr(model_args, 'num_points', 100)

    if batch_size is None:
        batch_size = 512 if N < 200 else 64

    print(f"Config: N={N} | Batch Size={batch_size} | Total Samples={num_samples}")

    # 2. Generate Data (Stay on GPU)
    print("Generating synthetic data on GPU...")
    # Create dataset on CPU to save VRAM, move to GPU in loop
    x0 = torch.rand(num_samples, N, 2)
    dataset = TensorDataset(x0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 3. Warmup
    print("Warming up GPU...")
    with torch.no_grad():
        dummy = torch.rand(2, N, 2).to(device)
        _ = ode_solve_euler(model, dummy, steps=50)

    # 4. Run Benchmark
    print("Running inference...")
    inference_times = []
    reconstruct_times = []
    opt_times = []
    total_times = []

    if torch.cuda.is_available(): torch.cuda.synchronize()

    for batch in loader:
        batch_x0 = batch[0].to(device)
        B = batch_x0.shape[0]

        # --- Timer 1: ODE Inference ---
        t0 = time.perf_counter()
        with torch.no_grad():
            final_configs = ode_solve_euler(model, batch_x0, steps=100)

        if torch.cuda.is_available(): torch.cuda.synchronize()
        t1 = time.perf_counter()

        # --- Timer 2: Reconstruction (Now on GPU) ---
        batch_tours = reconstruct_tour_gpu(final_configs)

        # Make Cyclic (Append start to end)
        # batch_tours is (B, N). We want (B, N+1)
        start_nodes = batch_tours[:, 0:1]  # (B, 1)
        tour_cyclic = torch.cat([batch_tours, start_nodes], dim=1)  # (B, N+1)

        if torch.cuda.is_available(): torch.cuda.synchronize()
        t2 = time.perf_counter()

        # --- Timer 3: 2-Opt (Now Vectorized GPU) ---
        # Pass pure tensors. No CPU conversion.
        refined_tour = batched_two_opt_vectorized(batch_x0, tour_cyclic)

        if torch.cuda.is_available(): torch.cuda.synchronize()
        t3 = time.perf_counter()

        # Stats (Summing total time for the batch)
        inference_times.append(t1 - t0)
        reconstruct_times.append(t2 - t1)
        opt_times.append(t3 - t2)
        total_times.append(t3 - t0)

    # 5. Report
    # Times are total for all batches. Divide by num_samples to get per-instance.
    avg_inf = np.sum(inference_times) / num_samples
    avg_rec = np.sum(reconstruct_times) / num_samples
    avg_opt = np.sum(opt_times) / num_samples
    avg_tot = np.sum(total_times) / num_samples

    print(f"\n--- Results for TSP-{N} ---")
    print(f"Model Inference:  {avg_inf * 1000:.3f} ms/instance")
    print(f"Reconstruction:   {avg_rec * 1000:.3f} ms/instance")
    print(f"2-Opt Refinement: {avg_opt * 1000:.3f} ms/instance")
    print(f"-------------------------------------------")
    print(f"TOTAL TIME:       {avg_tot * 1000:.3f} ms/instance")
    print(f"Throughput:       {1.0 / avg_tot:.2f} samples/sec")


if __name__ == "__main__":
    # Define models
    model_path = "/home/benjamin.fri/PycharmProjects/tsp_fm/checkpoints/lin_trans_100_06/best_model.pt"

    # Run Benchmark
    for N in [20, 50, 100, 500, 1000, 5000]:
        print(f"+++++++++++++++++++ N={N} +++++++++++++++++++")
        # Adjust batch size down for large N to avoid OOM
        bs = 512
        if N >= 500: bs = 64
        if N >= 2000: bs = 16
        if N >= 5000: bs = 4

        benchmark_model(model_path, gpu_id=6, num_samples=512, force_N=N, batch_size=bs)