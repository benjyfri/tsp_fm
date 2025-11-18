#!/usr/bin/env python3
"""
Robust Flow Matching Training for Kendall Shape Space using Geomstats.

This script properly uses geomstats to handle:
- Pre-shape space (sphere)
- Kendall shape space (quotient by rotations)
- Riemannian metrics
- Geodesics and parallel transport
- Tangent space projections
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path

# --- FIX 1: SET BACKEND *BEFORE* GEOMSTATS IMPORT ---
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
# ---------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from tqdm import tqdm # This was unused

# Geomstats imports
import geomstats.backend as gs
from geomstats.geometry.pre_shape import PreShapeSpace, PreShapeMetric
from geomstats.geometry.matrices import Matrices

# --- NEW: Add plotting imports ---
# Use 'Agg' backend for non-interactive saving (server-safe)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# ---------------------------------

# Set backend to PyTorch
gs.random.seed(42)

# ============================================================================
# Model Definition
# ============================================================================

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.max_period = 10000

    def forward(self, t):
        """t: (B,) -> (B, dim)"""
        t_scalar = t.flatten()
        half_dim = self.dim // 2
        device = t.device

        if half_dim == 0:
            return torch.zeros(t.shape[0], self.dim, device=device)

        log_max_period = torch.log(torch.tensor(self.max_period, device=device, dtype=torch.float32))
        denominator = torch.tensor(half_dim - 1, dtype=torch.float32, device=device).clamp(min=1.0)
        freqs = torch.exp(torch.arange(0, half_dim, device=device, dtype=torch.float32) *
                          (-log_max_period / denominator))
        args = t_scalar[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding.view(t.shape[0], self.dim)


class TransformerBlock(nn.Module):
    """Standard Transformer encoder block with self-attention."""
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
        """x: (B, N, embed_dim)"""
        attn_out, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)
        return x


class KendallVectorFieldModel(nn.Module):
    """
    Vector field model for Kendall shape space.

    Key features:
    - Outputs vectors in the tangent space to pre-shape space (sphere)
    - Properly projects to tangent space using geomstats
    - Permutation equivariant via Transformer
    """
    def __init__(self, n_points=50, embed_dim=256, t_emb_dim=128,
                 num_layers=4, num_heads=8, dropout=0.0):
        super().__init__()
        self.n_points = n_points
        self.embed_dim = embed_dim
        self.input_dim = 2  # 2D coordinates

        # Time embedding
        self.time_embed = TimeEmbedding(t_emb_dim)

        # Input projection
        input_feature_dim = self.input_dim + t_emb_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_feature_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Output head - predicts velocity in ambient space
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, self.input_dim)
        )

        # Zero-initialize output head
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def forward(self, x, t, space):
        """
        Args:
            x: (B, N, 2) - points on pre-shape space
            t: (B,) - time
            space: PreShapeSpace instance for tangent projection

        Returns:
            v_tangent: (B, N, 2) - velocity in tangent space
        """
        B, N, D = x.shape

        # Time embedding
        t_emb = self.time_embed(t)  # (B, t_emb_dim)
        t_expanded = t_emb.unsqueeze(1).repeat(1, N, 1)  # (B, N, t_emb_dim)

        # Concatenate coordinates with time
        inp = torch.cat([x, t_expanded], dim=2)  # (B, N, D + t_emb_dim)

        # Input projection
        h = self.input_projection(inp)  # (B, N, embed_dim)

        # Transformer blocks
        for block in self.transformer_blocks:
            h = block(h)

        # Output head
        v_raw = self.output_head(h)  # (B, N, 2)

        # --- FIX 3: PROJECT USING PYTORCH BACKEND ---
        # Project to tangent space using geomstats (PyTorch backend)
        # This operation is now differentiable and stays on the device.
        v_tangent = space.to_tangent(v_raw, x)
        # --------------------------------------------

        return v_tangent


# ============================================================================
# Data Loading
# ============================================================================

def load_dataset(data_path, device):
    """Load processed TSP dataset."""
    print(f"Loading data from: {data_path}")
    try:
        data = torch.load(data_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    if not isinstance(data, list) or len(data) == 0:
        print("Error: Dataset is empty or invalid")
        sys.exit(1)

    print(f"Loaded {len(data)} samples")

    # Extract x0 (points) and x1 (circle)
    x0_list = []
    x1_list = []
    theta_list = []

    for entry in data:
        x0_list.append(torch.from_numpy(entry['points']).float())
        x1_list.append(torch.from_numpy(entry['circle']).float())
        theta_list.append(entry['theta'])

    x0 = torch.stack(x0_list).to(device)
    x1 = torch.stack(x1_list).to(device)
    theta = torch.tensor(theta_list, dtype=torch.float32).to(device)

    return x0, x1, theta


def create_batches(x0, x1, theta, batch_size, shuffle=True):
    """Create batches from dataset."""
    n_samples = x0.shape[0]
    indices = torch.randperm(n_samples) if shuffle else torch.arange(n_samples)

    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        yield {
            'x0': x0[batch_indices],
            'x1': x1[batch_indices],
            'theta': theta[batch_indices]
        }


# ============================================================================
# Geomstats Integration
# ============================================================================

def setup_geomstats_space(n_points, ambient_dim=2):
    """
    Set up Kendall shape space using geomstats.

    Returns:
        space: PreShapeSpace instance
        metric: PreShapeMetric instance
    """
    # Create pre-shape space (sphere of centered configurations)
    space = PreShapeSpace(k_landmarks=n_points, ambient_dim=ambient_dim, equip=True)

    # Get metric
    metric = space.metric

    print(f"Created PreShapeSpace:")
    print(f"  - Dimension: {space.dim}")
    print(f"  - Number of landmarks: {space.k_landmarks}")
    print(f"  - Ambient dimension: {space.ambient_dim}")

    return space, metric


def sample_geodesic_geomstats(x0, x1, metric, device):
    """
    Sample geodesic between x0 and x1 using geomstats (PyTorch backend).

    Args:
        x0: (B, N, 2) - start points (torch tensor)
        x1: (B, N, 2) - end points (torch tensor)
        metric: PreShapeMetric instance
        device: torch device

    Returns:
        t: (B,) - random times
        xt: (B, N, 2) - points on geodesic
        ut: (B, N, 2) - velocities (tangent vectors)
    """
    B = x0.shape[0]

    # --- FIX 3 (cont.): VECTORIZED GEODESIC SAMPLING ---
    # No more numpy conversions, all ops are torch-native

    # Sample random times
    t = torch.rand(B, device=device, dtype=x0.dtype)

    # Compute log map: tangent vector from x0 to x1
    # log_x1_x0 has shape (B, N, 2)
    log_x1_x0 = metric.log(x1, x0)

    # We need to reshape t for broadcasting: (B,) -> (B, 1, 1)
    t_broadcast = t.view(B, 1, 1)

    # Geodesic segment vector from x0
    # (B, 1, 1) * (B, N, 2) -> (B, N, 2)
    geodesic_segment = t_broadcast * log_x1_x0

    # Point on geodesic: exp(geodesic_segment) at base point x0
    # xt shape: (B, N, 2)
    xt = metric.exp(geodesic_segment, x0)

    # Velocity: parallel transport of log_x1_x0 (the full velocity vector)
    # along the geodesic_segment
    # ut shape: (B, N, 2)
    ut = metric.parallel_transport(
        log_x1_x0,
        x0,
        direction=geodesic_segment
    )
    # ----------------------------------------------------

    return t, xt, ut


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, x0, x1, theta, space, metric, optimizer, args, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in create_batches(x0, x1, theta, args.batch_size, shuffle=True):
        # Move to device
        batch_x0 = batch['x0'].to(device)
        batch_x1 = batch['x1'].to(device)

        # Sample geodesic using geomstats
        t, xt, ut = sample_geodesic_geomstats(batch_x0, batch_x1, metric, device)

        # Forward pass
        optimizer.zero_grad()
        vt = model(xt, t, space)

        # Loss: MSE between predicted and true velocity
        loss = torch.mean((vt - ut) ** 2)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate(model, x0, x1, theta, space, metric, args, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in create_batches(x0, x1, theta, args.batch_size, shuffle=False):
            batch_x0 = batch['x0'].to(device)
            batch_x1 = batch['x1'].to(device)

            # Sample geodesic
            t, xt, ut = sample_geodesic_geomstats(batch_x0, batch_x1, metric, device)

            # Forward pass
            vt = model(xt, t, space)

            # Loss
            loss = torch.mean((vt - ut) ** 2)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


# --- NEW: Plotting utility function ---
# ============================================================================
# Plotting Utility
# ============================================================================

def save_loss_plot(train_losses, test_losses, output_path):
    """Saves a plot of training and test loss."""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 7))
    plt.plot(epochs, train_losses, 'bo-', label='Train Loss', markersize=4)
    plt.plot(epochs, test_losses, 'ro-', label='Test Loss', markersize=4)

    plt.title('Training and Test Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    try:
        plt.savefig(output_path)
        print(f"\nLoss plot saved to: {output_path}")
    except Exception as e:
        print(f"\nError saving loss plot: {e}")
    finally:
        plt.close() # Free memory
# ----------------------------------------


def main(args):
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device setup
    if torch.cuda.is_available():
        device = f"cuda:{args.gpu_id}"
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu_id)}")

        torch.set_default_device(device)
    else:
        device = 'cpu'
        print("Using CPU")
        torch.set_default_device(device)

    # Create output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup geomstats space
    space, metric = setup_geomstats_space(args.num_points, ambient_dim=2)

    # Load data
    train_x0, train_x1, train_theta = load_dataset(args.train_data_path, device)
    test_x0, test_x1, test_theta = load_dataset(args.test_data_path, device)

    print(f"Train samples: {train_x0.shape[0]}")
    print(f"Test samples: {test_x0.shape[0]}")

    # Verify data is on pre-shape space
    print("\nVerifying data preprocessing...")
    for i in range(min(5, train_x0.shape[0])):
        x_np = train_x0[i].cpu().numpy()
        # Check centering
        mean = np.mean(x_np, axis=0)
        print(f"  Sample {i}: mean = {mean}, norm = {np.linalg.norm(x_np, 'fro'):.6f}")

    # Create model
    model = KendallVectorFieldModel(
        n_points=args.num_points,
        embed_dim=args.embed_dim,
        t_emb_dim=args.t_emb_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    # Training loop
    best_test_loss = float('inf')
    train_losses = []
    test_losses = []

    print(f"\nStarting training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(
            model, train_x0, train_x1, train_theta,
            space, metric, optimizer, args, device
        )
        train_losses.append(train_loss)

        # Validate
        test_loss = validate(
            model, test_x0, test_x1, test_theta,
            space, metric, args, device
        )
        test_losses.append(test_loss)

        # Update learning rate
        scheduler.step()

        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss = {train_loss:.6f}, "
              f"Test Loss = {test_loss:.6f}, "
              f"LR = {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
            }, out_dir / 'best_model.pt')
            print(f"  -> Saved best model (test loss: {test_loss:.6f})")

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses,
    }, out_dir / 'final_model.pt')

    print("\nTraining complete!")
    print(f"Best test loss: {best_test_loss:.6f}")
    print(f"Models saved to: {out_dir}")

    # --- NEW: Save the loss plot ---
    plot_path = out_dir / 'loss_plot.png'
    save_loss_plot(train_losses, test_losses, plot_path)
    # -------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train flow matching model for Kendall shape space")

    # Data
    parser.add_argument('--train_data_path', type=str, default="data_old_scripts/processed_data_geom_train.pt",
                        help="Path to training data (.pt file)")
    parser.add_argument('--test_data_path', type=str, default="data_old_scripts/processed_data_geom_val.pt",
                        help="Path to test data (.pt file)")
    parser.add_argument('--num_points', type=int, default=50,
                        help="Number of points in each shape")

    # Model architecture
    parser.add_argument('--embed_dim', type=int, default=256,
                        help="Transformer embedding dimension")
    parser.add_argument('--t_emb_dim', type=int, default=128,
                        help="Time embedding dimension")
    parser.add_argument('--num_layers', type=int, default=4,
                        help="Number of Transformer layers")
    parser.add_argument('--num_heads', type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument('--dropout', type=float, default=0.0,
                        help="Dropout rate")

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=256,
                        help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-7,
                        help="Learning rate")
    parser.add_argument('--grad_clip_norm', type=float, default=5.0,
                        help="Gradient clipping norm")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed")

    # Hardware
    parser.add_argument('--gpu_id', type=int, default=0,
                        help="GPU ID to use")

    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints_kendall_geomstats',
                        help="Directory to save models and logs")

    args = parser.parse_args()
    main(args)