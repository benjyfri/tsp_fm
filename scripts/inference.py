import torch
import argparse
import sys
import os
import numpy as np
import wandb
from tqdm import tqdm

# --- FIX 1: Enforce Float32 Globally for Geomstats compatibility ---
torch.set_default_dtype(torch.float32)

# Add parent dir to path to import src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Updated imports to include RoPE model
from src.models import VectorFieldModel, RoPEVectorFieldModel
from src.dataset import load_data
from src.utils import ode_solve_euler, reconstruct_tour, calculate_tour_length
from src.geometry import GeometryProvider

def evaluate(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # 1. Robust Model Loading
    print(f"Loading checkpoint from {args.model_path}...")
    try:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

    # Determine Model Configuration
    if isinstance(checkpoint, dict) and 'args' in checkpoint:
        print("Found hyperparameters in checkpoint. Initializing model from saved config.")
        saved_args = checkpoint['args']
        if isinstance(saved_args, dict):
            model_args = argparse.Namespace(**saved_args)
        else:
            model_args = saved_args
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("WARNING: No 'args' found in checkpoint. Using command line args.")
        model_args = args
        state_dict = checkpoint['model_state_dict']
    else:
        print("WARNING: Raw state dict loaded.")
        model_args = args
        state_dict = checkpoint

    # 2. Setup Geometry
    interpolant = getattr(model_args, 'interpolant', args.interpolant)
    num_points = getattr(model_args, 'num_points', args.num_points)

    geo = None
    if interpolant == 'kendall':
        geo = GeometryProvider(num_points)
        print(f"Initialized Kendall Shape Space Geometry (N={num_points}).")

    # 3. Initialize & Load Model based on type
    # Default to 'concat' if model_type is missing (legacy checkpoints)
    model_type = getattr(model_args, 'model_type', 'concat')

    if model_type == 'rope':
        print("Initializing RoPE Vector Field Model...")
        model = RoPEVectorFieldModel(model_args).to(device)
    else:
        print("Initializing Standard Vector Field Model (Concat)...")
        model = VectorFieldModel(model_args).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    # 4. Load Data
    print(f"Loading test data from {args.test_data}...")
    x0, _, _, gt_paths = load_data(args.test_data, device)

    # --- FIX 2: Explicitly cast input data to Float32 ---
    # This ensures the inputs match the dtype Geomstats expects based on the global default set above
    x0 = x0.to(dtype=torch.float32)

    print(f"Evaluating on {len(x0)} samples using '{interpolant}' interpolant...")

    gaps = []

    # 5. Inference Loop
    for i in tqdm(range(len(x0)), desc="Inference"):
        # A. Get inputs
        points_start = x0[i].unsqueeze(0) # (1, N, 2)

        # B. Flow Match
        final_config = ode_solve_euler(model, points_start, geometry=geo, steps=100)
        final_config = final_config.squeeze(0) # (N, 2)

        # C. Reconstruct
        pred_tour = reconstruct_tour(final_config)

        # D. Calculate Lengths (Use original points for distance)
        original_cities = points_start.squeeze(0)

        pred_len = calculate_tour_length(original_cities, pred_tour)
        gt_len = calculate_tour_length(original_cities, gt_paths[i])

        # E. Calculate Gap
        gap = (pred_len - gt_len) / gt_len
        gaps.append(gap * 100)

    mean_gap = np.mean(gaps)
    print(f"\nResults:")
    print(f"Average Optimality Gap: {mean_gap:.4f}%")

    if args.use_wandb:
        wandb_config = vars(args)
        wandb_config['model_architecture'] = vars(model_args)
        wandb.init(project=args.project_name, job_type="inference", config=wandb_config)
        wandb.log({"test_gap_percentage": mean_gap})

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/home/benjamin.fri/PycharmProjects/tsp_fm/checkpoints/full_train_01/best_model.pt')
    parser.add_argument('--test_data', type=str, default='/home/benjamin.fri/PycharmProjects/tsp_fm/data/processed_data_geom_val.pt')
    parser.add_argument('--interpolant', type=str, choices=['kendall', 'linear', 'angle'], default='kendall')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--project_name', type=str, default="tsp-flow-matching")

    # Fallback Args
    parser.add_argument('--model_type', type=str, default='rope', choices=['concat', 'rope'])
    parser.add_argument('--num_points', type=int, default=50)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--t_emb_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)

    args = parser.parse_args()
    evaluate(args)