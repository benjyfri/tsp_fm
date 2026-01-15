import torch
import argparse
import sys
import os
import numpy as np
import wandb
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torchdiffeq import odeint

from src.utils import ode_solve_euler, reconstruct_tour, calculate_tour_length, ode_solve_rk4_exp, ode_solve_adaptive

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
from src.utils import reconstruct_tour, calculate_tour_length
from src.geometry import GeometryProvider


# --- 2-OPT FUNCTION ---
def batched_two_opt_torch(points, tour, max_iterations=1000, device="cpu"):
    iterator = 0
    tour = tour.copy()
    with torch.inference_mode():
        cuda_points = torch.from_numpy(points).to(device)
        cuda_tour = torch.from_numpy(tour).to(device)
        batch_size = cuda_tour.shape[0]
        min_change = -1.0
        while min_change < 0.0:
            points_i = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, -1, 1, 2))
            points_j = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, 1, -1, 2))
            points_i_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, -1, 1, 2))
            points_j_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, 1, -1, 2))

            A_ij = torch.sqrt(torch.sum((points_i - points_j) ** 2, axis=-1))
            A_i_plus_1_j_plus_1 = torch.sqrt(torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, axis=-1))
            A_i_i_plus_1 = torch.sqrt(torch.sum((points_i - points_i_plus_1) ** 2, axis=-1))
            A_j_j_plus_1 = torch.sqrt(torch.sum((points_j - points_j_plus_1) ** 2, axis=-1))

            change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
            valid_change = torch.triu(change, diagonal=2)

            min_change = torch.min(valid_change)
            flatten_argmin_index = torch.argmin(valid_change.reshape(batch_size, -1), dim=-1)
            min_i = torch.div(flatten_argmin_index, len(points), rounding_mode='floor')
            min_j = torch.remainder(flatten_argmin_index, len(points))

            if min_change < -1e-6:
                for i in range(batch_size):
                    cuda_tour[i, min_i[i] + 1:min_j[i] + 1] = torch.flip(cuda_tour[i, min_i[i] + 1:min_j[i] + 1],
                                                                         dims=(0,))
                iterator += 1
            else:
                break

            if iterator >= max_iterations:
                break
        tour = cuda_tour.cpu().numpy()
    return tour, iterator


# --- BATCHED ODE SOLVER WRAPPER ---
@torch.no_grad()
def solve_flow(model, x0, signals, geometry, steps=11, method='rk4', device='cuda'):
    """
    Uses torchdiffeq to solve the flow.
    """

    # Define the ODE function that captures 'signals' in its closure
    def ode_func(t, y):
        t_batch = t.expand(y.shape[0]).to(device)

        # Check if the model is the equivariant one that needs signals
        if isinstance(model, EquivariantDiffTransformer):
            return model(y, t_batch, static_signals=signals, geometry=geometry)
        else:
            return model(y, t_batch, geometry=geometry)

    t_span = torch.linspace(0., 1., steps=steps).to(device)
    options = {'step_size': 1.0 / (steps - 1)} if method in ['rk4', 'euler'] else None

    traj = odeint(
        ode_func,
        x0,
        t_span,
        method=method,
        options=options
    )
    return traj[-1]


def get_edges(tour):
    edges = set()
    for i in range(len(tour)):
        u, v = tour[i], tour[(i + 1) % len(tour)]
        if u > v: u, v = v, u
        edges.add((u, v))
    return edges


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

    # Factory logic
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

    # Clean state dict (remove _orig_mod prefix)
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k.replace("_orig_mod.", "")
    #
    #     # --- CRITICAL FIX: Filter out stale RoPE buffers ---
    #     if "freqs_cis" in name:
    #         continue
    #     # ---------------------------------------------------
    #
    #     new_state_dict[name] = v
    #
    # # Strict=False to allow ignoring the missing freqs_cis (which we want to re-init)
    # model.load_state_dict(new_state_dict, strict=True)

    # ... (inside evaluate function) ...

    # 1. Clean state dict carefully
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    print("\n--- Diagnostic: Checking State Dict Keys ---")
    ckpt_keys = list(state_dict.keys())
    print(f"First 3 Checkpoint Keys: {ckpt_keys[:3]}")

    for k, v in state_dict.items():
        name = k
        # Strip torch.compile prefix
        if name.startswith("_orig_mod."):
            name = name[10:]

        # Strip DataParallel prefix (common in multi-gpu)
        if name.startswith("module."):
            name = name[7:]

        # Skip RoPE cache if present (it's non-persistent)
        if "freqs_cis" in name or "freqs_base" in name:
            continue

        new_state_dict[name] = v

    # 2. Force Strict Loading
    print(f"Loading {len(new_state_dict)} keys into model...")
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("✅ SUCCESS: Weights loaded exactly.")
    except RuntimeError as e:
        print(f"\n❌ CRITICAL FAILURE: Weights did not load!")
        print(str(e))

        # Help identify the mismatch
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(new_state_dict.keys())

        missing = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys

        if missing:
            print(f"\nMISSING KEYS (In Model, not in Checkpoint):\n{list(missing)[:5]}")
        if unexpected:
            print(f"\nUNEXPECTED KEYS (In Checkpoint, not in Model):\n{list(unexpected)[:5]}")

        sys.exit(1)

    model.float()
    model.eval()

    # 4. Load Data
    print(f"Loading test data from {args.test_data}...")

    class MockInterpolant:
        pass

    mock_interp = MockInterpolant()
    mock_interp.__class__.__name__ = "KendallInterpolant" if 'kendall' in interpolant else "Linear"

    # Load 5 items: x0, x1, paths, signals, precomputed
    x0, x1, gt_paths, static_signals, precomputed = load_data(
        args.test_data, torch.device('cpu'), interpolant=mock_interp
    )

    x0 = x0.to(dtype=torch.float32)

    # Ensure signals exist if model needs them
    if model_type == 'equivariant_transformer':
        if static_signals is None:
            raise ValueError("Model is EquivariantTransformer but dataset has no static signals.")
        static_signals = static_signals.to(dtype=torch.float32)

        # Create dataset WITH signals
        dataset = TensorDataset(x0, torch.arange(len(x0)), static_signals)
    else:
        # Standard dataset
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
    print(f"Amount of ODE solver steps: {args.steps}")
    if args.run_2opt:
        print(">> Option enabled: 2-opt post-processing.")

    # Metrics Containers
    optimality_gaps = []
    dists_output_gt = []
    rel_dists_output_gt = []
    edge_overlaps = []
    output_cohesion_ratios = []
    valid_reconstructions = 0

    # 5. Batched Inference Loop
    for batch in tqdm(loader, desc="Batched Inference"):

        # Unpack batch safely
        batch_x0 = batch[0].to(device)
        batch_indices = batch[1]

        batch_signals = None
        if len(batch) > 2:
            batch_signals = batch[2].to(device)

        with torch.no_grad():
            final_configs = ode_solve_euler(model, batch_x0, geometry=geo, signals=batch_signals, steps=args.steps)
            # final_configs = solve_flow(
            #     model,
            #     batch_x0,
            #     signals=batch_signals,  # Pass signals here
            #     geometry=geo,
            #     steps=args.steps,
            #     method=args.method,
            #     device=device
            # )

        # # Project back to Manifold if needed
        # if geo is not None:
        #     final_configs = geo._project(final_configs)

        # C. Reconstruct & Evaluate
        final_configs_cpu = final_configs.cpu()
        original_cities_cpu = batch_x0.cpu()

        # Generate initial tours for the batch
        batch_tours = [reconstruct_tour(cfg) for cfg in final_configs_cpu]

        # --- 2-OPT OPTIONAL BLOCK ---
        if args.run_2opt:
            # We iterate through the batch because the 2-opt function provided expects
            # points to represent the geometry of a single instance (or shared geometry).
            # Since TSP instances in the batch have unique geometries, we process them individually.
            for k in range(len(batch_tours)):
                points_np = original_cities_cpu[k].numpy()
                tour_idxs = batch_tours[k]

                # --- FIX: Convert Tensor to list before concatenation ---
                if isinstance(tour_idxs, torch.Tensor):
                    tour_idxs = tour_idxs.tolist()
                elif isinstance(tour_idxs, np.ndarray):
                    tour_idxs = tour_idxs.tolist()
                # -----------------------------------------------------

                # Make tour cyclic for 2-opt (append start to end)
                tour_cyclic = np.array([tour_idxs + [tour_idxs[0]]])

                # Run 2-opt
                refined_tour, _ = batched_two_opt_torch(points_np, tour_cyclic, max_iterations=1000, device=device)

                # Remove the cyclic endpoint and update
                batch_tours[k] = list(refined_tour[0][:-1])
        # ----------------------------

        for i in range(len(batch_indices)):
            idx = batch_indices[i].item()
            pred_config = final_configs_cpu[i]
            original_cities = original_cities_cpu[i]

            # 1. Standard TSP Metrics
            # Use the (potentially optimized) tour from our batch list
            pred_tour = batch_tours[i]

            pred_len = calculate_tour_length(original_cities, pred_tour)

            if len(set(pred_tour)) == len(original_cities):
                valid_reconstructions += 1

            gt_path = gt_paths[idx]
            gt_len = calculate_tour_length(original_cities, gt_path)

            # Prevent division by zero
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

    # --- Aggregation & Reporting ---
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
    # Default paths for easy testing
    parser.add_argument('--model_path', type=str,
                        default=r"/home/benjamin.fri/PycharmProjects/tsp_fm/checkpoints/lin_trans_100_03/best_model.pt")
    parser.add_argument('--test_data', type=str,
                        default='/home/benjamin.fri/PycharmProjects/tsp_fm/data/can_tsp500_test.pt')

    # Batching & Hardware
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_id', type=int, default=3)

    # Run Config
    parser.add_argument('--interpolant', type=str, default='linear')
    parser.add_argument('--run_2opt', action='store_true', help="Run 2-opt local search on predicted tours")

    # Overrides
    parser.add_argument('--model_type', type=str, default=None)
    parser.add_argument('--num_points', type=int, default=500)

    # Solver
    parser.add_argument('--method', type=str, default='euler', choices=['euler', 'rk4'])
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)  # <--- Default
    args = parser.parse_args()
    evaluate(args)