#!/usr/bin/env python3
"""
LR Finder for Kendall Shape Space Flow Matching.
Runs for maximum 1 epoch to determine optimal starting learning rate.
"""

import os
import sys
import math
import argparse
from pathlib import Path

# --- SET BACKEND *BEFORE* GEOMSTATS IMPORT ---
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
# ---------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Geomstats imports
import geomstats.backend as gs
from geomstats.geometry.pre_shape import PreShapeSpace

# Plotting
import matplotlib
matplotlib.use('Agg') # Server-safe backend
import matplotlib.pyplot as plt

# Set backend to PyTorch
gs.random.seed(42)

# ============================================================================
# (Include same Model Classes as previous script)
# ============================================================================

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.max_period = 10000

    def forward(self, t):
        t_scalar = t.flatten()
        half_dim = self.dim // 2
        device = t.device
        if half_dim == 0: return torch.zeros(t.shape[0], self.dim, device=device)
        log_max_period = torch.log(torch.tensor(self.max_period, device=device, dtype=torch.float32))
        denominator = torch.tensor(half_dim - 1, dtype=torch.float32, device=device).clamp(min=1.0)
        freqs = torch.exp(torch.arange(0, half_dim, device=device, dtype=torch.float32) * (-log_max_period / denominator))
        args = t_scalar[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding.view(t.shape[0], self.dim)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)
        return x

class KendallVectorFieldModel(nn.Module):
    def __init__(self, n_points=50, embed_dim=256, t_emb_dim=128, num_layers=4, num_heads=8, dropout=0.0):
        super().__init__()
        self.time_embed = TimeEmbedding(t_emb_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(2 + t_emb_dim, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim)
        )
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, 2)
        )
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def forward(self, x, t, space):
        t_emb = self.time_embed(t)
        t_expanded = t_emb.unsqueeze(1).repeat(1, x.shape[1], 1)
        inp = torch.cat([x, t_expanded], dim=2)
        h = self.input_projection(inp)
        for block in self.transformer_blocks: h = block(h)
        v_raw = self.output_head(h)
        return space.to_tangent(v_raw, x)

# ============================================================================
# Helpers
# ============================================================================

def load_dataset(data_path, device):
    print(f"Loading data from: {data_path}")
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    x0_list = [torch.from_numpy(d['points']).float() for d in data]
    x1_list = [torch.from_numpy(d['circle']).float() for d in data]
    return torch.stack(x0_list).to(device), torch.stack(x1_list).to(device)

def setup_geomstats_space(n_points):
    space = PreShapeSpace(k_landmarks=n_points, ambient_dim=2, equip=True)
    return space, space.metric

def sample_geodesic(x0, x1, metric, device):
    B = x0.shape[0]
    t = torch.rand(B, device=device, dtype=x0.dtype)
    log_x1_x0 = metric.log(x1, x0)
    t_broadcast = t.view(B, 1, 1)
    geodesic_segment = t_broadcast * log_x1_x0
    xt = metric.exp(geodesic_segment, x0)
    ut = metric.parallel_transport(log_x1_x0, x0, direction=geodesic_segment)
    return t, xt, ut

# ============================================================================
# LR Finder Logic
# ============================================================================

def run_lr_finder(model, x0, x1, space, metric, device, args):
    # Configuration
    start_lr = 1e-7
    end_lr = 10.0
    beta = 0.98  # Smoothing factor for loss

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=start_lr)

    # Calculate steps
    n_samples = x0.shape[0]
    batch_size = args.batch_size
    num_steps = n_samples // batch_size

    # Calculate multiplier
    mult = (end_lr / start_lr) ** (1 / num_steps)

    # Storage
    lrs = []
    losses = []
    avg_loss = 0.0
    best_loss = 0.0

    model.train()

    # Indices
    indices = torch.randperm(n_samples)

    print(f"Starting LR Range Test: {start_lr:.1e} -> {end_lr}")
    print(f"Steps: {num_steps}")

    pbar = tqdm(range(num_steps), desc="LR Finder")

    for i in pbar:
        # Get batch
        idx = indices[i*batch_size : (i+1)*batch_size]
        batch_x0 = x0[idx].to(device)
        batch_x1 = x1[idx].to(device)

        # Sample Geodesic
        t, xt, ut = sample_geodesic(batch_x0, batch_x1, metric, device)

        # Step
        optimizer.zero_grad()
        vt = model(xt, t, space)
        loss = torch.mean((vt - ut) ** 2)

        # Compute smoothed loss
        current_loss = loss.item()
        avg_loss = beta * avg_loss + (1 - beta) * current_loss
        smoothed_loss = avg_loss / (1 - beta**(i + 1))

        # Stop if loss explodes
        if i > 0 and smoothed_loss > 4 * best_loss:
            print("\nLoss diverging, stopping early.")
            break

        # Record best loss
        if smoothed_loss < best_loss or i == 0:
            best_loss = smoothed_loss

        # Store values
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(smoothed_loss)

        # Update LR
        loss.backward()
        optimizer.step()

        current_lr = optimizer.param_groups[0]['lr']
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr * mult

        pbar.set_postfix({'lr': f"{current_lr:.1e}", 'loss': f"{smoothed_loss:.4f}"})

    return lrs, losses

def plot_lr_finder(lrs, losses, output_dir):
    """Plots the LR vs Loss graph."""

    # Find numerical gradient (steepest descent)
    # We look for the point with the steepest negative gradient
    grads = np.gradient(losses)
    # We want the steepest decline, so minimum gradient
    steepest_idx = np.argmin(grads)
    suggested_lr = lrs[steepest_idx]

    # Find minimum loss
    min_loss_idx = np.argmin(losses)
    min_lr = lrs[min_loss_idx]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.semilogx(lrs, losses)

    # Plot markers
    plt.scatter(suggested_lr, losses[steepest_idx], c='red', label=f'Steepest Desc. ({suggested_lr:.2e})')
    plt.scatter(min_lr, losses[min_loss_idx], c='green', label=f'Min Loss ({min_lr:.2e})')

    plt.xlabel("Learning Rate (Log Scale)")
    plt.ylabel("Loss (Smoothed)")
    plt.title("Learning Rate Finder")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)

    path = Path(output_dir) / 'lr_finder_plot.png'
    plt.savefig(path)
    print(f"\nPlot saved to: {path}")

    # Heuristic for suggestion:
    # Usually 1/10th of the min loss LR, or the steepest descent point
    print("\nAnalysis:")
    print(f"  1. Minimum Loss found at LR: {min_lr:.2e}")
    print(f"  2. Steepest Descent found at LR: {suggested_lr:.2e}")
    print(f"  -> Recommendation: Try an LR around {suggested_lr:.2e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default="data_old_scripts/processed_data_geom_train.pt")
    parser.add_argument('--num_points', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='checkpoints_kendall_geomstats')
    args = parser.parse_args()

    # Setup
    if torch.cuda.is_available():
        device = f"cuda:{args.gpu_id}"
        torch.set_default_device(device)
    else:
        device = 'cpu'

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    space, metric = setup_geomstats_space(args.num_points)
    x0, x1 = load_dataset(args.train_data_path, device)

    # Init Model (same dim as main script)
    model = KendallVectorFieldModel(
        n_points=args.num_points,
        embed_dim=256,
        t_emb_dim=128,
        num_layers=4
    ).to(device)

    # Run Finder
    lrs, losses = run_lr_finder(model, x0, x1, space, metric, device, args)

    # Plot
    plot_lr_finder(lrs, losses, args.output_dir)

if __name__ == '__main__':
    main()