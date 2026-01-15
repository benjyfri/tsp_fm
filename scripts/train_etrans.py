#!/usr/bin/env python3
"""
train_transformer_complete.py
Point Order Equivariant Transformer Flow Matching for TSP.

CHANGELOG:
1. Model: Replaced EGNN with Global Point Transformer (Set Transformer).
2. Connectivity: Removed KNN/Graph. Uses Dense O(N^2) Self-Attention.
3. Equivariance: Permutation Equivariant (Order of points does not matter).
4. Conditioning: Adaptive LayerNorm (AdaLN) for Time injection.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import tqdm
import math

# --- 1. L40S/Ampere Optimization ---
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_default_dtype(torch.float32)

# --- 2. Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# ==============================================================================
#  DATA LOADING
# ==============================================================================

class TSPFlowDataset(Dataset):
    """
    Efficiently loads the preprocessed dictionary of tensors.
    Structure: {'points': (N_samples, N_points, 2), 'circle': (N_samples, N_points, 2)}
    """

    def __init__(self, path):
        super().__init__()
        print(f"-> Loading dataset from {path}...")

        # weights_only=False allows loading the dictionary structure
        data = torch.load(path, weights_only=False)

        self.x0 = data['points'].float()
        self.x1 = data['circle'].float()

        # Free memory
        del data

        self.N = len(self.x0)
        print(f"   Loaded {self.N} samples.")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.x0[idx], self.x1[idx]


# ==============================================================================
#  MODEL COMPONENTS: Point Transformer (DiT Style)
# ==============================================================================

def modulate(x, shift, scale):
    """
    Adaptive Layer Norm modulation.
    x: (B, N, D)
    shift, scale: (B, D) -> broadcast to (B, 1, D)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        t: (B,)
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class PointTransformerBlock(nn.Module):
    """
    A DiT-style Transformer block adapted for Point Sets.
    Permutation Equivariant: No positional encodings on the sequence dim.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

        # Adaptive Layer Norm modulation parameters
        # Predicts gamma1, beta1, alpha1, gamma2, beta2, alpha2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, t_emb):
        """
        x: (B, N, D) - Point features
        t_emb: (B, D) - Time embedding
        """
        # 1. Regress modulation parameters from time
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(t_emb).chunk(6, dim=1)

        # 2. Self-Attention Block
        # Modulate Pre-Norm
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)

        # Standard Global Self-Attention (Permutation Equivariant)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)

        # Residual + Gating
        x = x + gate_msa.unsqueeze(1) * attn_out

        # 3. MLP Block
        # Modulate Pre-Norm
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)

        # MLP
        mlp_out = self.mlp(x_norm)

        # Residual + Gating
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class PointTransformerFlowMatching(nn.Module):
    """
    Full Flow Matching Model using Point Transformers.
    Input: (B, N, 2)
    Output: (B, N, 2) velocity field
    """

    def __init__(self,
                 hidden_dim=128,
                 depth=6,
                 num_heads=8,
                 mlp_ratio=4.0):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 1. Input Projections
        # Map 2D coordinates to Hidden Dim. 
        # Since we use Self-Attention on this, it's inherently position-aware 
        # relative to other points, but permutation invariant regarding index.
        self.x_embedder = nn.Linear(2, hidden_dim)

        # Time Embedder
        self.t_embedder = TimestepEmbedder(hidden_dim)

        # 2. Transformer Body
        self.blocks = nn.ModuleList([
            PointTransformerBlock(hidden_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # 3. Final Layer (DiT Style)
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
        self.final_linear = nn.Linear(hidden_dim, 2)

        # Initialize final layer to zero for stability
        nn.init.constant_(self.final_linear.weight, 0)
        nn.init.constant_(self.final_linear.bias, 0)

    def forward(self, x, t):
        """
        x: (B, N, 2)
        t: (B,)
        """
        # Embedding
        x_h = self.x_embedder(x)  # (B, N, D)
        t_emb = self.t_embedder(t)  # (B, D)

        # Transformer Blocks
        for block in self.blocks:
            x_h = block(x_h, t_emb)

        # Final Head
        shift, scale = self.adaLN_modulation(t_emb).chunk(2, dim=1)
        x_h = modulate(self.final_norm(x_h), shift, scale)
        velocity = self.final_linear(x_h)  # (B, N, 2)

        # Remove Center of Mass (CoM) - Enforce translation invariance of the flow
        velocity = velocity - velocity.mean(dim=1, keepdim=True)

        return velocity


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
    # Calculates params
    _dummy = PointTransformerFlowMatching(args.embed_dim, args.num_layers, args.num_heads)
    num_params = sum(p.numel() for p in _dummy.parameters()) / 1e6
    del _dummy

    if args.run_name is None:
        args.run_name = f"Trans_FM_{num_params:.2f}M_L{args.num_layers}_H{args.embed_dim}"

    wandb.init(project=args.project_name, name=args.run_name, config=args)
    wandb.config.update({"num_params_M": num_params, "type": "Transformer"}, allow_val_change=True)

    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    config = wandb.config

    device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Params: {num_params:.2f}M")

    save_dir = Path(config.save_dir) / config.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Data Loading ---
    print("\nInitializing DataLoaders...")
    try:
        train_set = TSPFlowDataset(config.train_data)
        val_set = TSPFlowDataset(config.val_data)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
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
    model = PointTransformerFlowMatching(
        hidden_dim=config.embed_dim,
        depth=config.num_layers,
        num_heads=config.num_heads
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
            # Batch Unpacking
            b_x0, b_x1 = batch

            b_x0 = b_x0.to(device, non_blocking=True)
            b_x1 = b_x1.to(device, non_blocking=True)

            # Center Data (Remove Translation)
            b_x0 = b_x0 - b_x0.mean(dim=1, keepdim=True)
            b_x1 = b_x1 - b_x1.mean(dim=1, keepdim=True)

            # 1. Interpolate
            t, xt, ut = sample_linear_interpolant(b_x0, b_x1, device)

            optimizer.zero_grad(set_to_none=True)

            # 2. Forward (No Edge Index, purely Transformer)
            vt = model(xt, t)

            # 3. Loss
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

                vt = model(xt, t)

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

    save_checkpoint(save_dir / "final_model.pt", model, optimizer, scheduler, config.epochs, best_val)
    print(f"Training complete. Best Val: {best_val:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Point Transformer Flow Matching")

    parser.add_argument('--project_name', type=str, default="tsp_FM_Transformer")
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default="./checkpoints")

    # Model Hyperparameters
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--eta_min_factor', type=float, default=1e-3)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    parser.add_argument('--gpu_id', type=int, default=0)

    # Flags
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true')

    args = parser.parse_args()
    train(args)