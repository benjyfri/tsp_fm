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
    CanonicalRoPEVectorField
)
from src.dataset import load_data
from src.utils import ode_solve_euler, reconstruct_tour, calculate_tour_length
from src.geometry import GeometryProvider

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
    if interpolant == 'kendall':
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
    else:
        model = VectorFieldModel(model_args).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    # 4. Load Data & Create DataLoader (OPTIMIZATION START)
    print(f"Loading test data from {args.test_data}...")

    # Load raw tensors
    # We move them to CPU first to avoid VRAM bloat, they will be moved to GPU by the loader or loop
    x0, _, _, gt_paths = load_data(args.test_data, torch.device('cpu'))

    # Ensure float32
    x0 = x0.to(dtype=torch.float32)

    # If gt_paths is a tensor, we can batch it. If it's a list, we might need to handle it differently.
    # Assuming gt_paths corresponds 1-to-1 with x0.
    # We'll stick to indexing gt_paths if it's not a tensor, but usually in these datasets it is.
    # If gt_paths is a list of lists, we just keep it aside and index it.

    print(f"Creating DataLoader with batch_size={args.batch_size}, num_workers={args.num_workers}...")

    # Create a dataset. We pass indices so we can look up the ground truth paths later
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

    gaps = []

    # 5. Batched Inference Loop
    # We wrap the loader with tqdm
    for batch_x0, batch_indices in tqdm(loader, desc="Batched Inference"):

        # A. Move batch to GPU
        batch_x0 = batch_x0.to(device) # Shape: (B, N, 2)

        # B. Flow Match (Batched)
        # ode_solve_euler should automatically handle the batch dimension
        # as long as your model accepts (B, N, 2) input.
        with torch.no_grad():
            final_configs = ode_solve_euler(model, batch_x0, geometry=geo, steps=100)
            # final_configs shape: (B, N, 2)

        # C. Reconstruct & Evaluate (Iterate over batch results)
        # Since reconstruction is usually CPU-bound and heuristic, we loop here.
        # Moving back to CPU for reconstruction avoids overhead of many small GPU access calls.
        final_configs_cpu = final_configs.cpu()
        original_cities_cpu = batch_x0.cpu()

        for i in range(len(batch_indices)):
            idx = batch_indices[i].item()
            pred_config = final_configs_cpu[i]     # (N, 2)
            original_cities = original_cities_cpu[i] # (N, 2)

            # Reconstruct Tour
            pred_tour = reconstruct_tour(pred_config)

            # Calculate Lengths
            pred_len = calculate_tour_length(original_cities, pred_tour)

            # Ground truth lookup
            gt_path = gt_paths[idx]
            gt_len = calculate_tour_length(original_cities, gt_path)

            # Calculate Gap
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
    # Paths
    parser.add_argument('--model_path', type=str, default='/home/benjamin.fri/PycharmProjects/tsp_fm/checkpoints/kendall_ROPE_02/best_model.pt')
    parser.add_argument('--test_data', type=str, default='/home/benjamin.fri/PycharmProjects/tsp_fm/data/processed_data_geom_val.pt')

    # Batching & Hardware Args
    parser.add_argument('--batch_size', type=int, default=128, help="Number of samples to process at once on GPU")
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