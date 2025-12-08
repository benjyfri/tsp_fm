import torch
import argparse
import sys
import os
import numpy as np
import wandb
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

# --- FIX 0: Set Geomstats Backend ---
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
torch.set_default_dtype(torch.float32)

# Add Parent Dir
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.models import (
    VectorFieldModel,
    RoPEVectorFieldModel,
    CanonicalMLPVectorField,
    CanonicalRoPEVectorField,
    CanonicalRegressor
)
from src.dataset import load_data
from src.utils import ode_solve_euler, reconstruct_tour, calculate_tour_length
from src.geometry import GeometryProvider

def resolve_path(path_str):
    if os.path.exists(path_str): return path_str
    alt_path = os.path.join(parent_dir, path_str)
    if os.path.exists(alt_path): return alt_path
    return path_str

def evaluate(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Checkpoint
    print(f"Loading checkpoint from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)

    if 'args' in checkpoint:
        model_args = argparse.Namespace(**checkpoint['args'])
    else:
        model_args = args

    # 2. Determine Model Type
    # Priority: Command line > Checkpoint Args > Heuristic
    model_type = getattr(model_args, 'model_type', args.model_type)
    if 'Reg' in args.model_path or 'regression' in str(model_type).lower():
        model_type = 'regression'

    print(f"Detected Model Type: {model_type}")

    # 3. Init Geometry (Only needed for Flow Matching)
    geo = None
    interpolant = getattr(model_args, 'interpolant', 'kendall')
    if model_type != 'regression' and interpolant == 'kendall':
        n_points = getattr(model_args, 'num_points', 50)
        geo = GeometryProvider(n_points)
        print("Initialized Geometry for Flow Matching.")

    # 4. Initialize Model
    if model_type == 'regression':
        model = CanonicalRegressor(model_args).to(device)
    elif model_type == 'rope':
        model = RoPEVectorFieldModel(model_args).to(device)
    elif model_type == 'canonical_rope':
        model = CanonicalRoPEVectorField(model_args).to(device)
    elif model_type == 'canonical_mlp':
        model = CanonicalMLPVectorField(model_args).to(device)
    else:
        model = VectorFieldModel(model_args).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(dtype=torch.float32)
    model.eval()

    # 5. Load Data
    data_path = resolve_path(args.test_data)
    print(f"Loading test data from {data_path}...")

    # load_data returns: x0, x1, theta, paths, precomputed
    x0, _, _, gt_paths, _ = load_data(data_path, torch.device('cpu'), interpolant=None)
    x0 = x0.to(dtype=torch.float32)

    # 6. Dataloader
    dataset = TensorDataset(x0, torch.arange(len(x0)))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    gaps = []

    # 7. Inference Loop
    print("Starting Inference...")
    for batch_x0, batch_idx in tqdm(loader, desc="Evaluating"):
        batch_x0 = batch_x0.to(device)

        with torch.no_grad():
            if isinstance(model, CanonicalRegressor):
                # Direct Regression
                pred_configs = model(batch_x0)
            else:
                # Flow Matching ODE
                pred_configs = ode_solve_euler(model, batch_x0, geometry=geo, steps=100)

        # CPU Reconstruction
        pred_configs_cpu = pred_configs.cpu()
        orig_cpu = batch_x0.cpu()

        for i in range(len(batch_idx)):
            idx = batch_idx[i].item()

            # Reconstruct
            pred_tour = reconstruct_tour(pred_configs_cpu[i])
            pred_len = calculate_tour_length(orig_cpu[i], pred_tour)

            # Ground Truth
            gt_path = gt_paths[idx]
            gt_len = calculate_tour_length(orig_cpu[i], gt_path)

            gap = (pred_len - gt_len) / gt_len
            gaps.append(gap * 100)

    mean_gap = np.mean(gaps)
    print(f"\n{'='*30}")
    print(f"Results for {model_type}")
    print(f"Average Optimality Gap: {mean_gap:.4f}%")
    print(f"{'='*30}\n")

    if args.use_wandb:
        wandb.init(project=args.project_name, job_type="inference", config=vars(args))
        wandb.log({"test_gap_percentage": mean_gap})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/home/benjamin.fri/PycharmProjects/tsp_fm/scripts_direct/checkpoints_reg/Regression-D256-L12/best_model.pt')
    parser.add_argument('--test_data', type=str, default='data/processed_data_geom_val.pt')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_id', type=int, default=7)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--project_name', type=str, default="tsp-inference")

    # Optional overrides
    parser.add_argument('--model_type', type=str, default=None,
                        help="Force model type: regression, rope, etc.")

    args = parser.parse_args()
    evaluate(args)