import torch
import argparse
import sys
import os
import numpy as np
import wandb
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

# --- Import your specific modules ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models_regression import EquivariantAngleRegressor
from dataset_regression import load_data
from src.utils import calculate_tour_length

# --- Set Global Precision ---
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_default_dtype(torch.float32)


def get_edges(tour):
    """Helper to convert a tour list into a set of edges for overlap calculation."""
    edges = set()
    for i in range(len(tour)):
        u, v = tour[i], tour[(i + 1) % len(tour)]
        if u > v: u, v = v, u
        edges.add((u, v))
    return edges


@torch.no_grad()
def evaluate(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 1. Load Checkpoint & Model
    # -------------------------------------------------------------------------
    print(f"Loading checkpoint from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)

    # Handle nested arguments in checkpoint (common if saved with 'args')
    if isinstance(checkpoint, dict) and 'args' in checkpoint:
        saved_args = checkpoint['args']
        # Convert dict to Namespace if necessary
        if isinstance(saved_args, dict):
            saved_args = argparse.Namespace(**saved_args)

        # Merge args: Override saved model args with critical inference args if needed
        model_args = saved_args
        # You might want to force embed_dim/heads if they aren't in saved_args
        if not hasattr(model_args, 'embed_dim'): model_args.embed_dim = args.embed_dim
    else:
        model_args = args

    state_dict = checkpoint.get('model_state_dict', checkpoint)

    print("Initializing EquivariantAngleRegressor...")
    model = EquivariantAngleRegressor(model_args)
    model = model.to(device)

    # --- Clean State Dict (Prefix removal) ---
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k
        if name.startswith("_orig_mod."): name = name[10:]  # Remove compile prefix
        if name.startswith("module."): name = name[7:]  # Remove DDP prefix
        new_state_dict[name] = v

    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("✅ SUCCESS: Weights loaded exactly.")
    except RuntimeError as e:
        print(f"⚠️ WARNING: Strict loading failed. Attempting strict=False...")
        print(f"Error detail: {e}")
        model.load_state_dict(new_state_dict, strict=False)

    model.eval()

    # -------------------------------------------------------------------------
    # 2. Load Data
    # -------------------------------------------------------------------------
    print(f"Loading test data from {args.test_data}...")

    # load_data returns: (x0, x1, path, signals, precomputed)
    # We only need x0 (cities), path (GT), and signals
    x0, _, gt_paths, static_signals, _ = load_data(
        args.test_data, torch.device('cpu'), interpolant=None
    )

    x0 = x0.to(dtype=torch.float32)
    if static_signals is None:
        raise ValueError("Dataset must contain static signals for Equivariant Model.")
    static_signals = static_signals.to(dtype=torch.float32)

    dataset = TensorDataset(x0, torch.arange(len(x0)), static_signals)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Evaluating on {len(x0)} samples...")

    # -------------------------------------------------------------------------
    # 3. Inference Loop
    # -------------------------------------------------------------------------
    optimality_gaps = []
    edge_overlaps = []
    valid_reconstructions = 0

    for batch in tqdm(loader, desc="Inference"):
        batch_x0 = batch[0].to(device)
        batch_indices = batch[1]
        batch_signals = batch[2].to(device)

        # A. Forward Pass
        # Returns: (B, N, 2) vectors representing direction on unit circle
        pred_vectors = model(batch_x0, batch_signals)

        # B. Convert Vectors to Angles (Radians)
        # atan2 returns values in [-pi, pi]
        pred_angles = torch.atan2(pred_vectors[..., 1], pred_vectors[..., 0])

        # C. Reconstruct Tour via Sorting
        # Sorting angles gives the order of cities on the circle
        _, pred_tours = torch.sort(pred_angles, dim=1)

        # D. CPU Projection for Metrics
        pred_tours_cpu = pred_tours.cpu()
        original_cities_cpu = batch_x0.cpu()

        for i in range(len(batch_indices)):
            idx = batch_indices[i].item()
            pred_tour = pred_tours_cpu[i].tolist()  # Indices in order
            original_cities = original_cities_cpu[i]

            # 1. Metric: Length & Gap
            pred_len = calculate_tour_length(original_cities, pred_tour)

            gt_path = gt_paths[idx]
            # Ensure gt_path is a list of indices
            if isinstance(gt_path, torch.Tensor):
                gt_path = gt_path.tolist()

            gt_len = calculate_tour_length(original_cities, gt_path)

            if gt_len < 1e-6: gt_len = 1.0
            gap = (pred_len - gt_len) / gt_len
            optimality_gaps.append(gap * 100)

            # 2. Metric: Validity (Always valid for sorting, but good sanity check)
            if len(set(pred_tour)) == len(original_cities):
                valid_reconstructions += 1

            # 3. Metric: Edge Overlap
            pred_edges = get_edges(pred_tour)
            gt_edges = get_edges(gt_path)

            overlap_count = len(pred_edges.intersection(gt_edges))
            edge_overlaps.append((overlap_count / len(original_cities)) * 100)

    # -------------------------------------------------------------------------
    # 4. Results
    # -------------------------------------------------------------------------
    mean_gap = np.mean(optimality_gaps)
    validity_rate = (valid_reconstructions / len(x0)) * 100
    mean_overlap = np.mean(edge_overlaps)

    print(f"\n=== Regression Inference Results ===")
    print(f"Average Optimality Gap:  {mean_gap:.4f}%")
    print(f"Validity Rate:           {validity_rate:.2f}%")
    print(f"GT Edge Overlap:         {mean_overlap:.2f}%")

    # Optional: Save results to simple dict or JSON if needed
    results = {
        "mean_gap": mean_gap,
        "validity_rate": validity_rate,
        "mean_overlap": mean_overlap
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument('--model_path', type=str,
                        default=r"/home/benjamin.fri/PycharmProjects/tsp_fm/scripts_direct/checkpoints/VecReg_01/best_model.pt")
    parser.add_argument('--test_data', type=str,
                        default='/home/benjamin.fri/PycharmProjects/tsp_fm/data/can_tsp50_rope_test.pt')

    # Hardware & Batching
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_id', type=int, default=2)

    # --- UPDATED MODEL CONFIG TO MATCH CHECKPOINT ---
    parser.add_argument('--embed_dim', type=int, default=384)  # Changed from 256
    parser.add_argument('--num_layers', type=int, default=8)  # Changed from 4
    parser.add_argument('--num_heads', type=int, default=12)  # Likely 12 (384/32) or 6 (384/64)

    args = parser.parse_args()

    evaluate(args)
