import torch
import argparse
import sys
import os
import numpy as np
import wandb
from tqdm import tqdm

# Add parent dir to path to import src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import VectorFieldModel
from src.dataset import load_data
from src.utils import ode_solve_euler, reconstruct_tour, calculate_tour_length
from src.geometry import GeometryProvider

def evaluate(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # 1. Setup Geometry
    # Crucial: If we are in Kendall mode, we need the geometry object
    # to project inputs/outputs onto the manifold during the ODE solve.
    geo = None
    if args.interpolant == 'kendall':
        geo = GeometryProvider(args.num_points)
        print("Initialized Kendall Shape Space Geometry.")

    # 2. Load Model
    print(f"Loading model from {args.model_path}...")
    model = VectorFieldModel(args).to(device)

    # Robust loading: handles both direct state_dict and checkpoint dicts
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 3. Load Data
    print(f"Loading test data from {args.test_data}...")
    x0, _, _, gt_paths = load_data(args.test_data, device)

    print(f"Evaluating on {len(x0)} samples using '{args.interpolant}' interpolant...")

    gaps = []

    # 4. Inference Loop
    for i in tqdm(range(len(x0)), desc="Inference"):
        # A. Get inputs
        points_start = x0[i].unsqueeze(0) # (1, N, 2)

        # B. Flow Match: Solve ODE t=0 -> t=1
        # We pass the 'geo' object. If it's not None (Kendall),
        # the solver will project velocities to tangent space and positions to the manifold.
        final_config = ode_solve_euler(model, points_start, geometry=geo, steps=100)
        final_config = final_config.squeeze(0) # (N, 2)

        # C. Reconstruct Tour from resulting "circle"
        pred_tour = reconstruct_tour(final_config)

        # D. Calculate Lengths
        # Note: use original 'points_start' for distance calculation, not the flowed points!
        original_cities = points_start.squeeze(0)

        pred_len = calculate_tour_length(original_cities, pred_tour)
        gt_len = calculate_tour_length(original_cities, gt_paths[i])

        # E. Calculate Gap
        # Gap = (Predicted - Optimal) / Optimal
        gap = (pred_len - gt_len) / gt_len
        gaps.append(gap * 100) # Convert to Percentage

    mean_gap = np.mean(gaps)
    print(f"\nResults:")
    print(f"Average Optimality Gap: {mean_gap:.4f}%")

    if args.use_wandb:
        wandb.init(project=args.project_name, job_type="inference", config=args)
        wandb.log({"test_gap_percentage": mean_gap})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--test_data', type=str, required=True)

    # We must specify the interpolant to know if we need geometry
    parser.add_argument('--interpolant', type=str, choices=['kendall', 'linear', 'angle'], default='kendall')

    # Model Args (Must match training config)
    parser.add_argument('--num_points', type=int, default=50)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--t_emb_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--project_name', type=str, default="tsp-flow-matching")

    args = parser.parse_args()
    evaluate(args)