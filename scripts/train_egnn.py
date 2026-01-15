#!/usr/bin/env python3
"""
train_egnn_complete.py
The Definitive Sparse EGNN Flow Matching for TSP.

CHANGELOG:
1. Data Loading: Now uses efficient TensorDataset mapping for the dictionary format.
2. Coordinate Head: Uses normalized unit vectors (rij / d) * w.
3. Aggregation: Uses Mean (Sum / k) to be invariant to k-NN size.
4. CoM: Enforces zero-mean centering consistent with Procrustes data.
5. Stability: LayerNorm + Residuals + Tanh gating.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader  # Added for efficient loading
import wandb
from tqdm import tqdm

# --- 1. L40S/Ampere Optimization ---
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_default_dtype(torch.float32)

# Try importing torch_cluster for fast GPU kNN
try:
    from torch_cluster import knn_graph as pyg_knn_graph

    HAS_PYG_CLUSTER = True
except ImportError:
    HAS_PYG_CLUSTER = False

# --- 2. Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# ==============================================================================
#  DATA LOADING (Efficient)
# ==============================================================================

class TSPFlowDataset(Dataset):
    """
    Efficiently loads the preprocessed dictionary of tensors.
    Structure: {'points': (N, P, 2), 'circle': (N, P, 2), ...}
    """

    def __init__(self, path):
        super().__init__()
        print(f"-> Loading dataset from {path}...")

        # Load the dictionary of tensors (on CPU RAM)
        # weights_only=False is used to allow loading the dictionary structure safely
        data = torch.load(path, weights_only=False)

        # Extract inputs (x0) and targets (x1)
        self.x0 = data['points'].float()
        self.x1 = data['circle'].float()

        # Free up memory from other keys (paths, perms, edges) if they exist
        del data

        self.N = len(self.x0)
        print(f"   Loaded {self.N} samples.")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # Returns simple tuple, DataLoader handles batching
        return self.x0[idx], self.x1[idx]


# ==============================================================================
#  UTILITIES
# ==============================================================================

def get_knn_graph(x, k):
    """
    Computes kNN graph.
    x: (B, N, 2)
    Returns: edge_index (2, E) -> (Receiver, Sender)
    """
    B, N, _ = x.shape
    device = x.device

    # --- Method 1: PyTorch Geometric (Fastest) ---
    if HAS_PYG_CLUSTER:
        x_flat = x.view(-1, 2)
        batch = torch.arange(B, device=device).repeat_interleave(N)

        # PyG returns (Source, Target) -> (Sender, Receiver)
        edge_index = pyg_knn_graph(x_flat, k, batch=batch, loop=False)

        # Flip to get (Receiver, Sender) for index_add_
        return edge_index.flip(0)

    # --- Method 2: Batched cdist (Fallback) ---
    with torch.no_grad():
        dist_sq = torch.cdist(x, x, p=2) ** 2
        _, idx = dist_sq.topk(k + 1, largest=False)
        idx = idx[:, :, 1:]  # remove self

    row = torch.arange(N, device=device)[None, :, None].expand(B, N, k)
    batch_offset = torch.arange(B, device=device)[:, None, None] * N

    row = (row + batch_offset).reshape(-1)
    col = (idx + batch_offset).reshape(-1)

    return torch.stack([row, col], dim=0)


# ==============================================================================
#  MODEL COMPONENTS
# ==============================================================================

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        return self.net(t[:, None])


class FiLM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim * 2)

    def forward(self, h, t_emb):
        # h: (B*N, D), t_emb: (B, D)
        gamma, beta = self.proj(t_emb).chunk(2, dim=-1)
        return h * (1 + gamma[:, None, :]) + beta[:, None, :]


class SparseEGNNLayer(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        self.k = k
        self.norm = nn.LayerNorm(dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(dim * 2 + 1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, h, x, edge_index):
        row, col = edge_index

        rij = x[row] - x[col]
        dij_sq = (rij ** 2).sum(dim=-1, keepdim=True)

        m_ij = self.edge_mlp(
            torch.cat([h[row], h[col], dij_sq], dim=-1)
        )

        agg = torch.zeros_like(h)
        agg.index_add_(0, row, m_ij)

        # Mean Aggregation
        agg = agg / self.k

        update = self.node_mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h + update)


class SparseCoordinateHead(nn.Module):
    def __init__(self, dim, k, weight_temp=10.0):
        super().__init__()
        self.k = k
        self.weight_temp = weight_temp

        self.mlp = nn.Sequential(
            nn.Linear(dim * 2 + 1, dim),
            nn.SiLU(),
            nn.Linear(dim, 1, bias=False)
        )
        nn.init.zeros_(self.mlp[-1].weight)

    def forward(self, h, x, edge_index):
        row, col = edge_index

        rij = x[row] - x[col]
        dij_sq = (rij ** 2).sum(dim=-1, keepdim=True)

        # Compute Weight w_ij
        w = self.mlp(
            torch.cat([h[row], h[col], dij_sq], dim=-1)
        )
        w = torch.tanh(w / self.weight_temp)

        # Unit Vector Normalization
        dij = torch.sqrt(dij_sq + 1e-8)
        v_ij = w * (rij / dij)

        # Aggregation
        v = torch.zeros_like(x)
        v.index_add_(0, row, v_ij)

        # Mean Aggregation
        v = v / self.k

        return v


# ==============================================================================
#  FULL MODEL
# ==============================================================================

class SparseEGNNFlowMatching(nn.Module):
    def __init__(self, hidden_dim=128, depth=6, k=16, weight_temp=10.0):
        super().__init__()
        self.k = k
        self.hidden_dim = hidden_dim

        self.node_embedding = nn.Parameter(torch.randn(1, hidden_dim) * 0.01)
        self.time_emb = TimeEmbedding(hidden_dim)
        self.film = FiLM(hidden_dim)

        self.layers = nn.ModuleList([
            SparseEGNNLayer(hidden_dim, k) for _ in range(depth)
        ])

        self.coord_head = SparseCoordinateHead(hidden_dim, k, weight_temp)

    def forward(self, x, t, edge_index):
        B, N, _ = x.shape
        x_flat = x.reshape(B * N, 2)

        # 1. Initialize Features
        h = self.node_embedding.expand(B * N, -1)

        # 2. Time Embedding
        t_emb = self.time_emb(t)

        # 3. EGNN Body
        for layer in self.layers:
            h_view = h.view(B, N, -1)
            h_view = self.film(h_view, t_emb)
            h = h_view.view(B * N, -1)
            h = layer(h, x_flat, edge_index)

        # 4. Velocity Prediction
        v = self.coord_head(h, x_flat, edge_index)
        v = v.view(B, N, 2)

        # 5. Output CoM Removal
        v = v - v.mean(dim=1, keepdim=True)

        return v


# ==============================================================================
#  TRAINING
# ==============================================================================

def sample_linear_interpolant(x0, x1, device):
    """
    x_t = (1-t)x0 + t x1
    u_t = x1 - x0
    """
    B = x0.shape[0]
    t = torch.rand(B, device=device)
    t_exp = t.view(B, 1, 1)

    xt = (1 - t_exp) * x0 + t_exp * x1
    ut = x1 - x0
    return t, xt, ut


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val': best_val
    }, path)


def train(args):
    # --- Reproducibility ---
    if args.deterministic:
        print("!! Deterministic Mode Enabled !!")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # --- Setup ---
    _dummy = SparseEGNNFlowMatching(args.embed_dim, args.num_layers, args.k, args.weight_temp)
    num_params = sum(p.numel() for p in _dummy.parameters()) / 1e6
    del _dummy

    if args.run_name is None:
        knn_type = "PyG" if HAS_PYG_CLUSTER else "Native"
        args.run_name = f"EGNN_Final_{knn_type}_{num_params:.2f}M_L{args.num_layers}_K{args.k}"

    wandb.init(project=args.project_name, name=args.run_name, config=args)
    wandb.config.update({"num_params_M": num_params, "knn_backend": "PyG" if HAS_PYG_CLUSTER else "Native"},
                        allow_val_change=True)
    # --- ADD THIS BLOCK ---
    # 1. Define 'epoch' as a metric
    wandb.define_metric("epoch")
    # 2. Set 'epoch' as the x-axis for all other metrics
    wandb.define_metric("*", step_metric="epoch")
    # ----------------------
    config = wandb.config

    device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Params: {num_params:.2f}M")

    save_dir = Path(config.save_dir) / config.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Data Loading (Optimized) ---
    print("\nInitializing DataLoaders...")
    try:
        train_set = TSPFlowDataset(config.train_data)
        val_set = TSPFlowDataset(config.val_data)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)

    # Optimized DataLoader settings for High-Performance Training
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,  # Prefetch in parallel
        pin_memory=True,  # Faster CPU->GPU transfer
        persistent_workers=True  # Don't kill workers between epochs
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # --- Model ---
    model = SparseEGNNFlowMatching(
        hidden_dim=config.embed_dim,
        depth=config.num_layers,
        k=config.k,
        weight_temp=config.weight_temp
    ).to(device)

    print("Compiling model...")
    try:
        model = torch.compile(model, mode='default')
    except Exception as e:
        print(f"Compilation skipped: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    eta_min = config.lr * config.eta_min_factor
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=eta_min)

    best_val = float('inf')

    # --- Training Loop ---
    print("\nStarting training...")
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", ncols=100)

        for batch in pbar:
            # Batch Unpacking (Dataset returns x0, x1)
            b_x0, b_x1 = batch

            b_x0 = b_x0.to(device, non_blocking=True)
            b_x1 = b_x1.to(device, non_blocking=True)

            # --- Consistency Check ---
            # Data is already centered, but we enforce it here to avoid drift
            b_x0 = b_x0 - b_x0.mean(dim=1, keepdim=True)
            b_x1 = b_x1 - b_x1.mean(dim=1, keepdim=True)

            # 1. Interpolate
            t, xt, ut = sample_linear_interpolant(b_x0, b_x1, device)

            # 2. Graph Construction
            edge_index = get_knn_graph(xt, config.k)

            optimizer.zero_grad(set_to_none=True)

            # 3. Forward
            vt = model(xt, t, edge_index)

            loss = torch.mean((vt - ut) ** 2)

            if torch.isnan(loss):
                print(f"\n❌ CRITICAL: Loss is NaN.")
                wandb.finish(exit_code=1)
                sys.exit(1)

            loss.backward()

            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

            optimizer.step()

            epoch_train_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train = epoch_train_loss / max(1, num_batches)

        # --- Validation ---
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                b_x0, b_x1 = batch
                b_x0 = b_x0.to(device, non_blocking=True)
                b_x1 = b_x1.to(device, non_blocking=True)

                b_x0 = b_x0 - b_x0.mean(dim=1, keepdim=True)
                b_x1 = b_x1 - b_x1.mean(dim=1, keepdim=True)

                t, xt, ut = sample_linear_interpolant(b_x0, b_x1, device)
                edge_index = get_knn_graph(xt, config.k)

                vt = model(xt, t, edge_index)

                val_loss_accum += torch.mean((vt - ut) ** 2).item()
                val_batches += 1

        avg_val = val_loss_accum / max(1, val_batches)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Train {avg_train:.6f} | Val {avg_val:.6f} | LR {current_lr:.3e}")
        wandb.log({"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val, "lr": current_lr})

        # --- Save ---
        if avg_val < best_val:
            best_val = avg_val
            save_checkpoint(save_dir / "best_model.pt", model, optimizer, scheduler, epoch, best_val)

        # if epoch % config.checkpoint_freq == 0:
        #     save_checkpoint(save_dir / "last_checkpoint.pt", model, optimizer, scheduler, epoch, best_val)

    save_checkpoint(save_dir / "final_model.pt", model, optimizer, scheduler, config.epochs, best_val)
    print(f"Training complete. Best Val: {best_val:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sparse EGNN (Complete)")

    parser.add_argument('--project_name', type=str, default="tsp_FM_EGNN_Final")
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default="./checkpoints")

    # Model Hyperparameters
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--weight_temp', type=float, default=10.0)

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--eta_min_factor', type=float, default=1e-3)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    parser.add_argument('--gpu_id', type=int, default=7)
    parser.add_argument('--checkpoint_freq', type=int, default=10)

    # Flags
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true')

    args = parser.parse_args()
    train(args)