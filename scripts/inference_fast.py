import torch
import argparse
import sys
import os
import numpy as np
import wandb
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

# --- FIX 0: Set Geomstats Backend ---
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'

# --- FIX 1: Enforce Float32 Globally ---
torch.set_default_dtype(torch.float32)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import (
    VectorFieldModel, RoPEVectorFieldModel, CanonicalRoPEVectorField,
    CanonicalMLPVectorField, CanonicalRegressor, SpectralCanonMLP,
    SpectralCanonTransformer, EquivariantDiffTransformer
)
from src.dataset import load_data
from src.utils import ode_solve_euler, calculate_tour_length
from src.geometry import GeometryProvider


# ==============================================================================
#  OPTIMIZATION 1: PURE TORCH RECONSTRUCTION (GPU)
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

    # 3. Compute angles (atan2) -> (B, N) values in [-pi, pi]
    angles = torch.atan2(centered[..., 1], centered[..., 0])

    # 4. Sort indices by angle to recover the sequence
    tour_indices = torch.argsort(angles, dim=1)  # (B, N)

    return tour_indices


# ==============================================================================
#  OPTIMIZATION 2: FULLY VECTORIZED 2-OPT (NO PYTHON LOOPS)
# ==============================================================================
@torch.jit.script
def batched_two_opt_vectorized(points, tour, max_iterations: int = 1000):
    """
    Fully vectorized 2-opt. Performs the 'flip' using a masked gather,
    avoiding per-batch python loops entirely.

    Args:
        points: (B, N, 2) Coordinates
        tour:   (B, N+1) Cyclic Tour Indices
    """
    batch_size, num_points_plus_1 = tour.shape
    num_points = num_points_plus_1 - 1
    device = points.device

    # Base grid for indexing: [[0, 1, ... N], [0, 1, ... N], ...]
    idx_grid = torch.arange(num_points_plus_1, device=device).unsqueeze(0).expand(batch_size, -1)

    iterator = 0
    while iterator < max_iterations:
        # --- 1. GATHER COORDINATES ---
        # Get coordinates for the current tour order
        p_ordered = torch.gather(points, 1, tour[..., :-1].unsqueeze(-1).expand(-1, -1, 2))  # (B, N, 2)
        p_next = torch.roll(p_ordered, -1, dims=1)  # (B, N, 2) -> (i+1)

        # Reshape for broadcasting N^2 comparison
        P_i = p_ordered.unsqueeze(2)  # (B, N, 1, 2)
        P_j = p_ordered.unsqueeze(1)  # (B, 1, N, 2)
        P_i_next = p_next.unsqueeze(2)
        P_j_next = p_next.unsqueeze(1)

        # --- 2. COMPUTE COST CHANGES ---
        d_ij = torch.norm(P_i - P_j, dim=-1)
        d_next = torch.norm(P_i_next - P_j_next, dim=-1)
        d_curr1 = torch.norm(P_i - P_i_next, dim=-1)
        d_curr2 = torch.norm(P_j - P_j_next, dim=-1)

        change = d_ij + d_next - d_curr1 - d_curr2

        # Mask invalid swaps (diagonal and lower triangle)
        valid_change = torch.triu(change, diagonal=2)

        # --- 3. FIND BEST MOVE ---
        # Find the minimum change for each batch item
        min_change_val, flatten_argmin = torch.min(valid_change.view(batch_size, -1), dim=-1)

        # If all batch items have non-negative min_change, we are optimal
        if (min_change_val >= -1e-6).all():
            break

        # --- 4. PERFORM SWAP (VECTORIZED FLIP) ---
        best_i = torch.div(flatten_argmin, num_points, rounding_mode='floor')
        best_j = torch.remainder(flatten_argmin, num_points)

        # The segment to reverse is [i+1, j]
        start = best_i + 1
        end = best_j

        # Create mask for the range [start, end]
        mask = (idx_grid >= start.unsqueeze(1)) & (idx_grid <= end.unsqueeze(1))

        # Calculate reversed indices: New_Index[k] = (start + end) - k
        sum_bound = start.unsqueeze(1) + end.unsqueeze(1)
        reversed_idxs = sum_bound - idx_grid

        # Apply flip only where mask is True
        target_gather_idxs = torch.where(mask, reversed_idxs, idx_grid)
        tour = torch.gather(tour, 1, target_gather_idxs)

        iterator += 1

    return tour


# ==============================================================================
#  HELPER: METRICS
# ==============================================================================
def get_edges(tour):
    edges = set()
    for i in range(len(tour)):
        u, v = tour[i], tour[(i + 1) % len(tour)]
        if u > v: u, v = v, u
        edges.add((u, v))
    return edges


# ==============================================================================
#  MAIN EVALUATION LOOP
# ==============================================================================
def evaluate(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Robust Model Loading
    print(f"Loading checkpoint from {args.model_path}...")
    try:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.model_path, map_location=device)

    # Determine Model Configuration
    if isinstance(checkpoint, dict) and 'args' in checkpoint:
        saved_args = checkpoint['args']
        if isinstance(saved_args, dict):
            model_args = argparse.Namespace(**saved_args)
        else:
            model_args = saved_args
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_args = args
        state_dict = checkpoint['model_state_dict']
    else:
        model_args = args
        state_dict = checkpoint

    # 2. Setup Geometry
    interpolant = getattr(model_args, 'interpolant', args.interpolant)
    num_points = getattr(model_args, 'num_points', args.num_points)

    geo = None
    if 'kendall' in interpolant:
        geo = GeometryProvider(num_points)
        print(f"Initialized Kendall Shape Space Geometry (N={num_points}).")

    # 3. Initialize Model
    model_type = getattr(model_args, 'model_type', 'concat')

    if model_type == 'rope':
        model = RoPEVectorFieldModel(model_args)
    elif model_type == 'canonical_rope':
        model = CanonicalRoPEVectorField(model_args)
    elif model_type == 'canonical_mlp':
        model = CanonicalMLPVectorField(model_args)
    elif model_type == 'canonical_regressor':
        model = CanonicalRegressor(model_args)
    elif model_type == 'spectral_mlp':
        model = SpectralCanonMLP(model_args)
    elif model_type == 'spectral_trans':
        model = SpectralCanonTransformer(model_args)
    elif model_type == 'equivariant_transformer':
        model = EquivariantDiffTransformer(model_args)
    else:
        model = VectorFieldModel(model_args)

    model = model.to(device)

    # Clean state dict
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    print("\n--- Diagnostic: Checking State Dict Keys ---")
    for k, v in state_dict.items():
        name = k
        if name.startswith("_orig_mod."): name = name[10:]
        if name.startswith("module."): name = name[7:]
        # Skip RoPE cache if present
        if "freqs_cis" in name or "freqs_base" in name: continue
        new_state_dict[name] = v

    print(f"Loading {len(new_state_dict)} keys into model...")
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("✅ SUCCESS: Weights loaded exactly.")
    except RuntimeError as e:
        print(f"\n❌ CRITICAL FAILURE: Weights did not load!")
        print(str(e))
        sys.exit(1)

    model.float()
    model.eval()

    # OPTIONAL: Compile for speed (PyTorch 2.0+)
    # model = torch.compile(model)

    # 4. Load Data
    print(f"Loading test data from {args.test_data}...")

    class MockInterpolant:
        pass

    mock_interp = MockInterpolant()
    mock_interp.__class__.__name__ = "KendallInterpolant" if 'kendall' in interpolant else "Linear"

    x0, x1, gt_paths, static_signals, precomputed = load_data(
        args.test_data, torch.device('cpu'), interpolant=mock_interp
    )

    x0 = x0.to(dtype=torch.float32)

    if model_type == 'equivariant_transformer':
        if static_signals is None:
            raise ValueError("Model is EquivariantTransformer but dataset has no static signals.")
        static_signals = static_signals.to(dtype=torch.float32)
        dataset = TensorDataset(x0, torch.arange(len(x0)), static_signals)
    else:
        dataset = TensorDataset(x0, torch.arange(len(x0)))

    print(f"Creating DataLoader with batch_size={args.batch_size}...")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Evaluating on {len(x0)} samples...")
    print(f"Steps: {args.steps} | 2-Opt: {args.run_2opt}")

    # Metrics Containers
    optimality_gaps = []
    dists_output_gt = []
    rel_dists_output_gt = []
    edge_overlaps = []
    output_cohesion_ratios = []
    valid_reconstructions = 0

    # 5. Batched Inference Loop
    for batch in tqdm(loader, desc="Batched Inference"):

        # Unpack batch
        batch_x0 = batch[0].to(device)
        batch_indices = batch[1]
        batch_signals = None
        if len(batch) > 2:
            batch_signals = batch[2].to(device)

        # --- A. SOLVE ODE (GPU) ---
        with torch.no_grad():
            final_configs = ode_solve_euler(model, batch_x0, geometry=geo, signals=batch_signals, steps=args.steps)

        # --- B. RECONSTRUCT (GPU) ---
        # No more CPU/Numpy here!
        batch_tours = reconstruct_tour_gpu(final_configs)  # (B, N)

        # --- C. 2-OPT (GPU - Vectorized) ---
        if args.run_2opt:
            # Make tour cyclic (append start to end)
            start_nodes = batch_tours[:, 0:1]  # (B, 1)
            tour_cyclic = torch.cat([batch_tours, start_nodes], dim=1)  # (B, N+1)

            # Run fast vectorized 2-opt
            refined_cyclic = batched_two_opt_vectorized(batch_x0, tour_cyclic)

            # Remove cyclic endpoint
            batch_tours = refined_cyclic[:, :-1]

        # --- D. METRICS (CPU) ---
        # Only now do we move data to CPU for reporting
        final_configs_cpu = final_configs.cpu()
        original_cities_cpu = batch_x0.cpu()
        batch_tours_cpu = batch_tours.cpu().tolist()  # Convert tensors to lists for easy iteration

        for i in range(len(batch_indices)):
            idx = batch_indices[i].item()
            pred_config = final_configs_cpu[i]
            original_cities = original_cities_cpu[i]
            pred_tour = batch_tours_cpu[i]

            # 1. Standard TSP Metrics
            pred_len = calculate_tour_length(original_cities, pred_tour)

            if len(set(pred_tour)) == len(original_cities):
                valid_reconstructions += 1

            gt_path = gt_paths[idx]
            gt_len = calculate_tour_length(original_cities, gt_path)

            if gt_len < 1e-6: gt_len = 1.0
            gap = (pred_len - gt_len) / gt_len
            optimality_gaps.append(gap * 100)

            # 2. Structural Metrics
            pred_edges = get_edges(pred_tour)
            gt_edges = get_edges(gt_path)

            overlap_count = len(pred_edges.intersection(gt_edges))
            edge_overlaps.append((overlap_count / len(original_cities)) * 100)

            if geo is not None:
                # Cohesion
                pred_dist_sum = sum(torch.norm(pred_config[u] - pred_config[v]).item() for u, v in pred_edges)
                mean_pred_edge_dist = pred_dist_sum / (len(pred_edges) + 1e-8)
                gt_dist_sum = sum(torch.norm(pred_config[u] - pred_config[v]).item() for u, v in gt_edges)
                mean_gt_edge_dist = gt_dist_sum / (len(gt_edges) + 1e-8)
                output_cohesion_ratios.append(mean_gt_edge_dist / (mean_pred_edge_dist + 1e-8))

                # Shape Metrics
                gt_config = original_cities[gt_path]
                d_out_gt = geo.distance(pred_config, gt_config).item()
                d_in_gt = geo.distance(original_cities, gt_config).item()
                dists_output_gt.append(d_out_gt)
                scale = d_in_gt + 1e-8
                rel_dists_output_gt.append(d_out_gt / scale)

    # --- E. REPORTING ---
    mean_gap = np.mean(optimality_gaps)
    validity_rate = (valid_reconstructions / len(x0)) * 100
    mean_overlap = np.mean(edge_overlaps)

    print(f"\n=== Results ===")
    print(f"Average Optimality Gap:  {mean_gap:.4f}%")
    print(f"Validity Rate:           {validity_rate:.2f}%")
    print(f"GT Edge Overlap:         {mean_overlap:.2f}%")

    wandb_logs = {
        "test_gap_percentage": mean_gap,
        "validity_rate": validity_rate,
        "gt_edge_overlap_percentage": mean_overlap
    }

    if geo is not None and len(dists_output_gt) > 0:
        mean_dist_gt = np.mean(dists_output_gt)
        mean_rel_output_gt = np.mean(rel_dists_output_gt)
        mean_cohesion = np.mean(output_cohesion_ratios)

        print(f"\n--- Shape Space Metrics ---")
        print(f"Dist (Out vs GT):           {mean_dist_gt:.4f}")
        print(f"Relative Dist (Out vs GT):  {mean_rel_output_gt:.4f}")
        print(f"Output Cohesion Ratio:      {mean_cohesion:.4f}")

        wandb_logs.update({
            "kendall_dist_output_gt": mean_dist_gt,
            "kendall_rel_dist_output_gt": mean_rel_output_gt,
            "output_cohesion_ratio": mean_cohesion
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--model_path', type=str,
                        default=r"/home/benjamin.fri/PycharmProjects/tsp_fm/checkpoints/lin_trans_100_06/best_model.pt")
    parser.add_argument('--test_data', type=str,
                        default='/home/benjamin.fri/PycharmProjects/tsp_fm/data/can_tsp50_test.pt')

    # Batching & Hardware
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_id', type=int, default=6)

    # Run Config
    parser.add_argument('--interpolant', type=str, default='linear')
    parser.add_argument('--run_2opt', action='store_true', help="Run 2-opt local search on predicted tours")

    # Overrides
    parser.add_argument('--model_type', type=str, default=None)
    parser.add_argument('--num_points', type=int, default=100)

    # Solver
    parser.add_argument('--method', type=str, default='euler', choices=['euler', 'rk4'])
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)

    args = parser.parse_args()
    evaluate(args)