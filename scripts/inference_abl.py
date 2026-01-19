import torch
import argparse
import sys
import os
import numpy as np
import wandb
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torchdiffeq import odeint

# --- FIX 0: Set Geomstats Backend ---
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'

# --- FIX 1: Enforce Float32 Globally ---
torch.set_default_dtype(torch.float32)

# Ensure 'scripts' and root are in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

# --- IMPORTS ---
try:
    import train_trans_NERF as nerf_pkg

    print("✅ Imported train_trans_NERF")
except ImportError:
    pass

try:
    import train_trans_LITEPT as litept_pkg

    print("✅ Imported train_trans_LITEPT")
except ImportError:
    pass

try:
    import train_etrans as etrans_pkg

    print("✅ Imported train_etrans")
except ImportError:
    print("⚠️  Could not import train_etrans.py")

from src.dataset import load_data
from src.utils import reconstruct_tour, calculate_tour_length


# ==============================================================================
#  ODE WRAPPER
# ==============================================================================
class ODEFunc(torch.nn.Module):
    def __init__(self, model, signals=None):
        super().__init__()
        self.model = model
        self.signals = signals

    def forward(self, t, x):
        t_batch = t.repeat(x.shape[0]).to(x.device)
        # Check signature compatibility before passing signals
        if self.signals is not None:
            # Try/Except block to handle models that don't accept signals
            try:
                return self.model(x, t_batch, signals=self.signals)
            except TypeError:
                return self.model(x, t_batch)
        return self.model(x, t_batch)


# ==============================================================================
#  HELPER FUNCTIONS
# ==============================================================================
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


def get_edges(tour):
    edges = set()
    for i in range(len(tour)):
        u, v = tour[i], tour[(i + 1) % len(tour)]
        if u > v: u, v = v, u
        edges.add((u, v))
    return edges


# ==============================================================================
#  EVALUATION LOOP
# ==============================================================================
def evaluate(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Checkpoint
    print(f"Loading checkpoint from {args.model_path}...")
    try:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.model_path, map_location=device)

    # --- CHECKPOINT LOADING (FIXED) ---
    checkpoint = torch.load(args.model_path, map_location=device)

    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise RuntimeError(
            "Checkpoint format invalid. Expected dict with key 'model_state_dict'."
        )

    state_dict = checkpoint['model_state_dict']
    model_args = args  # architecture must come from CLI, not checkpoint

    # --- FALLBACKS ---
    if not hasattr(model_args, 'embed_dim'): model_args.embed_dim = getattr(args, 'embed_dim', 256)
    if not hasattr(model_args, 'num_layers'): model_args.num_layers = getattr(args, 'num_layers', 6)
    if not hasattr(model_args, 'num_heads'): model_args.num_heads = getattr(args, 'num_heads', 8)
    if not hasattr(model_args, 'fourier_scale'): model_args.fourier_scale = getattr(args, 'fourier_scale', 16.0)
    if not hasattr(model_args, 'mlp_ratio'): model_args.mlp_ratio = getattr(args, 'mlp_ratio', 4.0)

    # --- SANITIZATION BLOCK ---
    def sanitize(val):
        if isinstance(val, (list, tuple)): return int(val[0])
        return val

    model_args.embed_dim = sanitize(model_args.embed_dim)
    model_args.num_layers = sanitize(model_args.num_layers)
    model_args.num_heads = sanitize(model_args.num_heads)

    # 2. INTELLIGENT MODEL DETECTION
    model_type = args.model_type
    if model_type is None:
        if hasattr(model_args, 'model_type') and model_args.model_type is not None:
            model_type = model_args.model_type
        if model_type is None or model_type == 'concat':
            fname = args.model_path.lower()
            if 'etrans' in fname or 'equivariant' in fname:
                model_type = 'etrans'
            elif 'nerf' in fname or 'fourier' in fname:
                model_type = 'nerf'
            elif 'litept' in fname or 'rope' in fname:
                model_type = 'litept'
            else:
                model_type = 'nerf'

    print(f"Initializing Model Type: {model_type} | Dim: {model_args.embed_dim} | Layers: {model_args.num_layers}")

    # 3. Model Factory
    model = None

    if model_type == 'nerf':
        model = nerf_pkg.PointTransformerFlowMatching(
            hidden_dim=model_args.embed_dim, depth=model_args.num_layers,
            num_heads=model_args.num_heads, mlp_ratio=model_args.mlp_ratio,
            fourier_scale=model_args.fourier_scale
        )
    elif model_type == 'litept':
        model = litept_pkg.PointTransformerFlowMatching(
            hidden_dim=model_args.embed_dim, depth=model_args.num_layers,
            num_heads=model_args.num_heads, mlp_ratio=model_args.mlp_ratio
        )
    elif model_type in ['etrans', 'equivariant_transformer']:
        #
        # CRITICAL FIX: The code you provided defines PointTransformerFlowMatching, NOT EquivariantDiffTransformer.
        # We must instantiate the class that actually exists in your training script.

        if hasattr(etrans_pkg, 'PointTransformerFlowMatching'):
            # This matches the training script you shared
            model = etrans_pkg.PointTransformerFlowMatching(
                hidden_dim=model_args.embed_dim,
                depth=model_args.num_layers,
                num_heads=model_args.num_heads,
                mlp_ratio=model_args.mlp_ratio
                # Note: Your training script does NOT take fourier_scale
            )
        elif hasattr(etrans_pkg, 'EquivariantDiffTransformer'):
            model = etrans_pkg.EquivariantDiffTransformer(model_args)
        else:
            raise ValueError("Could not find a valid model class in train_etrans.py")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)

    # 4. Strict State Dict Loading
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "").replace("module.", "")
        # if "freqs_cis" in name or "freqs_base" in name or "inv_freq" in name: continue
        new_state_dict[name] = v

    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("✅ SUCCESS: Weights loaded exactly.")
    except RuntimeError as e:
        print(f"\n❌ CRITICAL: Weights mismatch for {model_type}!")
        print(f"   Error details: {e}")
        sys.exit(1)

    model.eval()

    # 5. Data Loading
    print(f"Loading test data from {args.test_data}...")

    class MockInterpolant:
        pass

    mock_interp = MockInterpolant()
    mock_interp.__class__.__name__ = "Linear"

    x0, x1, gt_paths, static_signals, _ = load_data(
        args.test_data, torch.device('cpu'), interpolant=mock_interp
    )
    x0 = x0.to(dtype=torch.float32)

    # Note: Since we determined your 'etrans' model is actually a Standard Transformer,
    # we don't strictly need signals. We load them just in case, but treat them as optional.
    if static_signals is not None:
        dataset = TensorDataset(x0, torch.arange(len(x0)), static_signals)
    else:
        dataset = TensorDataset(x0, torch.arange(len(x0)))

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 6. Inference Loop
    valid_reconstructions = 0
    optimality_gaps = []
    edge_overlaps = []

    print(f"Starting inference on {len(x0)} samples...")
    print(f"running for {args.steps} steps")

    for batch in tqdm(loader, desc="Inference"):
        batch_x0 = batch[0].to(device)
        batch_idx = batch[1]
        batch_sigs = None
        if len(batch) > 2:
            batch_sigs = batch[2].to(device)

        # Wrap and Solve
        ode_func = ODEFunc(model, signals=batch_sigs)
        t_span = torch.linspace(0, 1, args.steps).to(device)

        with torch.no_grad():
            traj = odeint(ode_func, batch_x0, t_span, method='euler')
            final_configs = traj[-1]

        final_configs_cpu = final_configs.cpu()
        batch_x0_cpu = batch_x0.cpu()
        batch_tours = [reconstruct_tour(cfg) for cfg in final_configs_cpu]

        if args.run_2opt:
            for k in range(len(batch_tours)):
                tour_idxs = batch_tours[k]
                if isinstance(tour_idxs, torch.Tensor):
                    tour_idxs = tour_idxs.tolist()
                elif isinstance(tour_idxs, np.ndarray):
                    tour_idxs = tour_idxs.tolist()

                tour_cyclic = np.array([tour_idxs + [tour_idxs[0]]])
                points_np = batch_x0_cpu[k].numpy()
                refined, _ = batched_two_opt_torch(points_np, tour_cyclic, device=device)
                batch_tours[k] = list(refined[0][:-1])

        for i, idx in enumerate(batch_idx):
            idx = idx.item()
            pred_tour = batch_tours[i]
            orig_cities = batch_x0_cpu[i]

            pred_len = calculate_tour_length(orig_cities, pred_tour)
            gt_path = gt_paths[idx]
            gt_len = calculate_tour_length(orig_cities, gt_path)

            if gt_len < 1e-6: gt_len = 1.0
            gap = (pred_len - gt_len) / gt_len
            optimality_gaps.append(gap * 100)

            if len(set(pred_tour)) == len(orig_cities):
                valid_reconstructions += 1

            pred_edges = get_edges(pred_tour)
            gt_edges = get_edges(gt_path)
            overlap = len(pred_edges.intersection(gt_edges)) / len(orig_cities)
            edge_overlaps.append(overlap * 100)

    print(f"\n=== Results ===")
    print(f"Optimality Gap:   {np.mean(optimality_gaps):.4f}%")
    print(f"Validity Rate:    {valid_reconstructions / len(x0) * 100:.2f}%")
    print(f"Edge Overlap:     {np.mean(edge_overlaps):.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Updated paths for robustness
    parser.add_argument('--model_path', type=str,
                        default=r"/home/benjamin.fri/PycharmProjects/tsp_fm/scripts/checkpoints_best/etrans_best_02/best_model.pt")
    parser.add_argument('--test_data', type=str, default='../data/tsp50_test.pt')
    parser.add_argument('--model_type', type=str, default=None)

    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--gpu_id', type=int, default=3)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--run_2opt', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)

    # Architecture Defaults
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--fourier_scale', type=float, default=16.0)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)

    args = parser.parse_args()
    evaluate(args)
