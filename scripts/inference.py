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
    VectorFieldModel,
    RoPEVectorFieldModel,
    CanonicalMLPVectorField,
    CanonicalRoPEVectorField,
    # --- NEW IMPORTS ---
    CanonicalRegressor,
    SpectralCanonMLP,
    SpectralCanonTransformer
)
from src.dataset import load_data
from src.utils import ode_solve_euler, reconstruct_tour, calculate_tour_length, ode_solve_rk4_exp, ode_solve_adaptive
from src.geometry import GeometryProvider


def get_edges(tour):
    """
    Convert a tour (list of indices) into a set of sorted edges (tuples).
    Ex: [0, 1, 2] -> {(0,1), (1,2), (0,2)}
    """
    edges = set()
    for i in range(len(tour)):
        u, v = tour[i], tour[(i + 1) % len(tour)]
        if u > v:
            u, v = v, u
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
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

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
    # if interpolant == 'kendall':
    if 'kendall' in interpolant:
        geo = GeometryProvider(num_points)
        print(f"Initialized Kendall Shape Space Geometry (N={num_points}).")

    # 3. Initialize Model
    model_type = getattr(model_args, 'model_type', 'concat')

    if model_type == 'rope':
        model = RoPEVectorFieldModel(model_args).to(device)
    elif model_type == 'canonical_rope':
        model = CanonicalRoPEVectorField(model_args).to(device)
    elif model_type == 'canonical_mlp':
        model = CanonicalMLPVectorField(model_args).to(device)
    # --- NEW MODELS ---
    elif model_type == 'canonical_regressor':
        model = CanonicalRegressor(model_args).to(device)
    elif model_type == 'spectral_mlp':
        model = SpectralCanonMLP(model_args).to(device)
    elif model_type == 'spectral_trans':
        model = SpectralCanonTransformer(model_args).to(device)
    else:
        model = VectorFieldModel(model_args).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    # 4. Load Data & Create DataLoader
    print(f"Loading test data from {args.test_data}...")
    x0, _, gt_paths, _ = load_data(args.test_data, torch.device('cpu'))

    # x0 = x0[:10000, :, :]
    # gt_paths = gt_paths[:10000]
    # Ensure float32
    x0 = x0.to(dtype=torch.float32)

    print(f"Creating DataLoader with batch_size={args.batch_size}, num_workers={args.num_workers}...")
    indices = torch.arange(len(x0))
    dataset = TensorDataset(x0, indices)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Evaluating on {len(x0)} samples using '{interpolant}' interpolant...")

    # Metrics Containers
    optimality_gaps = []

    # New Shape Metrics
    dists_output_input = []
    dists_output_gt = []
    dists_input_gt = []

    rel_dists_output_input = []
    rel_dists_output_gt = []

    # New Structure Metrics
    edge_overlaps = []
    output_cohesion_ratios = []

    valid_reconstructions = 0

    # 5. Batched Inference Loop
    for batch_x0, batch_indices in tqdm(loader, desc="Batched Inference"):

        # A. Move batch to GPU
        batch_x0 = batch_x0.to(device)

        # B. Flow Match (Batched)
        with torch.no_grad():
            final_configs = ode_solve_euler(model, batch_x0, geometry=geo, steps=5)
            # final_configs = ode_solve_rk4_exp(model, batch_x0, geometry=geo, steps=100)

        # C. Reconstruct & Evaluate
        final_configs_cpu = final_configs.cpu()
        original_cities_cpu = batch_x0.cpu()

        for i in range(len(batch_indices)):
            idx = batch_indices[i].item()
            pred_config = final_configs_cpu[i]  # (N, 2)
            original_cities = original_cities_cpu[i]  # (N, 2)

            # 1. Standard TSP Metrics
            pred_tour = reconstruct_tour(pred_config)
            pred_len = calculate_tour_length(original_cities, pred_tour)

            # Check validity
            if len(set(pred_tour)) == len(original_cities):
                valid_reconstructions += 1

            # Ground truth lookup
            gt_path = gt_paths[idx]
            gt_len = calculate_tour_length(original_cities, gt_path)

            gap = (pred_len - gt_len) / gt_len
            optimality_gaps.append(gap * 100)

            # 2. Structural Metrics (New)
            pred_edges = get_edges(pred_tour)
            gt_edges = get_edges(gt_path)

            # Intersection count
            overlap_count = len(pred_edges.intersection(gt_edges))
            edge_overlaps.append((overlap_count / len(original_cities)) * 100)

            # Output Cohesion (Distances in Output Space)
            if geo is not None:
                # A. Mean distance of edges chosen by Model
                pred_dist_sum = 0
                for u, v in pred_edges:
                    d = torch.norm(pred_config[u] - pred_config[v])
                    pred_dist_sum += d.item()
                mean_pred_edge_dist = pred_dist_sum / len(pred_edges)

                # B. Mean distance of edges chosen by GT
                gt_dist_sum = 0
                for u, v in gt_edges:
                    d = torch.norm(pred_config[u] - pred_config[v])
                    gt_dist_sum += d.item()
                mean_gt_edge_dist = gt_dist_sum / len(gt_edges)

                output_cohesion_ratios.append(mean_gt_edge_dist / (mean_pred_edge_dist + 1e-8))

            # 3. Kendall Shape Space Metrics
            if geo is not None:
                gt_config = original_cities[gt_path]

                d_out_in = geo.distance(pred_config, original_cities)
                d_out_gt = geo.distance(pred_config, gt_config)
                d_in_gt = geo.distance(original_cities, gt_config)  # Scale

                dists_output_input.append(d_out_in.item())
                dists_output_gt.append(d_out_gt.item())
                dists_input_gt.append(d_in_gt.item())

                scale = d_in_gt.item() + 1e-8
                rel_dists_output_input.append(d_out_in.item() / scale)
                rel_dists_output_gt.append(d_out_gt.item() / scale)

    # --- Aggregation & Reporting ---
    mean_gap = np.mean(optimality_gaps)
    validity_rate = (valid_reconstructions / len(x0)) * 100
    mean_overlap = np.mean(edge_overlaps)

    print(f"\n=== Results ===")
    print(f"Average Optimality Gap:  {mean_gap:.4f}%")
    print(f"Validity Rate:           {validity_rate:.2f}%")
    print(f"GT Edge Overlap:         {mean_overlap:.2f}% (Edges matched with GT)")

    wandb_logs = {
        "test_gap_percentage": mean_gap,
        "validity_rate": validity_rate,
        "gt_edge_overlap_percentage": mean_overlap
    }

    if geo is not None and len(dists_output_gt) > 0:
        mean_dist_input = np.mean(dists_output_input)
        mean_dist_gt = np.mean(dists_output_gt)
        mean_dist_baseline = np.mean(dists_input_gt)

        # Calculate Means for Relative Metrics
        mean_rel_output_input = np.mean(rel_dists_output_input)
        mean_rel_output_gt = np.mean(rel_dists_output_gt)

        mean_cohesion = np.mean(output_cohesion_ratios)

        convergence_ratio = mean_dist_gt / (mean_dist_input + 1e-8)

        print(f"\n--- Shape Space Metrics (Absolute) ---")
        print(f"Dist (Input vs GT) [Scale]: {mean_dist_baseline:.4f}")
        print(f"Dist (Out vs Input):        {mean_dist_input:.4f}")
        print(f"Dist (Out vs GT):           {mean_dist_gt:.4f}")

        print(f"\n--- Shape Space Metrics (Relative) ---")
        print(f"Relative Dist (Out vs In):  {mean_rel_output_input:.4f}  (Normalized by baseline)")
        print(f"Relative Dist (Out vs GT):  {mean_rel_output_gt:.4f}  (Normalized by baseline)")
        print(f"Output Cohesion Ratio:      {mean_cohesion:.4f}      (1.0 = Perfect Structure)")

        wandb_logs.update({
            "kendall_dist_output_input": mean_dist_input,
            "kendall_dist_output_gt": mean_dist_gt,
            "kendall_dist_baseline_scale": mean_dist_baseline,
            "kendall_rel_dist_output_input": mean_rel_output_input,
            "kendall_rel_dist_output_gt": mean_rel_output_gt,
            "kendall_convergence_ratio": convergence_ratio,
            "output_cohesion_ratio": mean_cohesion
        })

    if args.use_wandb:
        wandb_config = vars(args)
        wandb_config['model_architecture'] = vars(model_args)
        wandb.init(project=args.project_name, job_type="inference", config=wandb_config)
        wandb.log(wandb_logs)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    parser = argparse.ArgumentParser()
    # Paths
    # parser.add_argument('--model_path', type=str,
    #                     default='/home/benjamin.fri/PycharmProjects/tsp_fm/checkpoints/kendall_ROPE_02/best_model_best.pt')

    parser.add_argument('--model_path', type=str,
                        default=r"/home/benjamin.fri/PycharmProjects/tsp_fm/checkpoints/spectral_MLP_17_1M_linear_sfm_01/best_model.pt")
    parser.add_argument('--test_data', type=str,
                        default='/home/benjamin.fri/PycharmProjects/tsp_fm/data/processed_data_geom_test.pt')

    # Batching & Hardware Args
    parser.add_argument('--batch_size', type=int, default=2048, help="Number of samples to process at once on GPU")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of CPU workers for data loading")

    # Run Config
    parser.add_argument('--interpolant', type=str, choices=['kendall', 'linear', 'angle'], default='kendall')
    parser.add_argument('--gpu_id', type=int, default=4)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--project_name', type=str, default="tsp-flow-matching")

    # Fallback Args
    parser.add_argument('--model_type', type=str, default='rope',
                        choices=['concat', 'rope', 'canonical_mlp', 'canonical_rope'])
    parser.add_argument('--num_points', type=int, default=50)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--t_emb_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)

    args = parser.parse_args()
    evaluate(args)