#!/usr/bin/env python3
"""
Robust train.py for Flow Matching (Kendall, Linear, & Angle).

- Stable Kendall geodesic sampling
- Differentiable Angle/Length reconstruction
"""

import os
import time
import math
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.autograd.functional import jvp  # [NEW] For Angle FM

from torchcfm import ConditionalFlowMatcher
# Ensure this path is correct for your project structure
from tsp_flow.data_loader import get_loaders
from tsp_flow.models import StrongEquivariantVectorField
from tsp_flow.utils import count_parameters, plot_loss_curves


# ---------- [NEW] Angle FM Reconstruction & Processing ----------

def reconstruct_polygon_batch(batched_angles: torch.Tensor,
                              batched_lengths: torch.Tensor,
                              batched_path: torch.Tensor) -> torch.Tensor:
    """
    Differentiably reconstructs a batch of centered (N, 2) point clouds
    from their turning angles and edge lengths, provided in path-order.

    Args:
        batched_angles: (B, N) tensor of turning angles (in path order).
        batched_lengths: (B, N) tensor of edge lengths (in path order).
        batched_path: (B, N) tensor of path indices.

    Returns:
        (B, N, 2) tensor of centered point clouds in *original 0..N-1 order*.
    """
    B, N = batched_angles.shape
    device = batched_angles.device

    # 1. Calculate relative heading changes (turns) at each vertex
    rel_headings = batched_angles - math.pi  # (B, N)

    # 2. Set the first heading change to 0 (to define heading[0] as 0)
    rel_headings_shifted = torch.roll(rel_headings, shifts=1, dims=1)
    rel_headings_shifted[:, 0] = 0.0

    # 3. Compute absolute headings of each *edge vector*
    abs_headings = torch.cumsum(rel_headings_shifted, dim=1)  # (B, N)

    # 4. Create edge vectors (v_i) in path order
    vectors_x = batched_lengths * torch.cos(abs_headings)
    vectors_y = batched_lengths * torch.sin(abs_headings)
    vectors = torch.stack([vectors_x, vectors_y], dim=2)  # (B, N, 2)

    # 5. Reconstruct points in *path order* by cumsum
    points_path_order_shifted = torch.cumsum(vectors, dim=1)  # (B, N, 2)
    points_path_order = torch.roll(points_path_order_shifted, shifts=1, dims=1)
    points_path_order[:, 0, :] = 0.0  # Set p[path[0]] = (0,0)

    # 6. Scatter points back to *original 0..N-1 order*
    path_expanded = batched_path.unsqueeze(-1).expand(-1, -1, 2)  # (B, N, 2)
    points_original_order = torch.zeros_like(points_path_order)
    points_original_order.scatter_(dim=1, index=path_expanded, src=points_path_order)

    # 7. Center the point cloud
    points_centered = points_original_order - torch.mean(points_original_order, dim=1, keepdim=True)

    return points_centered


def pre_process_shape_torch(X: torch.Tensor) -> torch.Tensor:
    """
    [NEW] PyTorch version of the Kendall pre-processing from data_create.py.
    Pre-processes a batch of shapes.
    1. Centers the shape at the origin.
    2. Scales the shape to have a unit Frobenius norm.

    Args:
        X: A (B, N, 2) tensor of vertex coordinates.

    Returns:
        A new (B, N, 2) tensor (centered, unit F-norm).
    """
    # 1. Center (re-center just in case, though reconstruct_polygon_batch does it)
    X_c = X - torch.mean(X, dim=1, keepdim=True)

    # 2. Scale to unit Frobenius norm
    f_norm = torch.norm(X_c, p='fro', dim=(1, 2), keepdim=True) # (B, 1, 1)

    # Avoid division by zero
    f_norm = torch.clamp(f_norm, min=1e-9)

    X_normed = X_c / f_norm
    return X_normed


# ---------- Geodesic (Kendall) FM ----------

def sample_geodesic_stable(x0, x1, theta, device, eps=1e-6):
    """Stable Kendall geodesic sampling. Returns (t, xt, ut)."""
    B = x0.shape[0]
    t = torch.rand(B, device=device, dtype=x0.dtype)
    t_ = t.view(B, 1, 1)

    theta = theta.to(device=device, dtype=x0.dtype).view(B, 1, 1)
    theta = torch.clamp(theta, min=eps, max=(math.pi - eps))

    a = (1 - t_) * theta
    b = t_ * theta

    sin_theta_safe = torch.clamp(torch.sin(theta), min=eps)

    sa = torch.sin(a)
    sb = torch.sin(b)
    ca = torch.cos(a)
    cb = torch.cos(b)

    xt_geo = (sa / sin_theta_safe) * x0 + (sb / sin_theta_safe) * x1
    ut_geo = (theta / sin_theta_safe) * (cb * x1 - ca * x0)

    mask_near = (theta <= eps) | ((math.pi - theta) <= eps)
    mask_near = mask_near.to(dtype=x0.dtype)

    xt_lin = (1 - t_) * x0 + t_ * x1
    ut_lin = x1 - x0

    xt = xt_geo * (1 - mask_near) + xt_lin * mask_near
    ut = ut_geo * (1 - mask_near) + ut_lin * mask_near

    return t, xt, ut


# ---------- [REVISED] Flow Sampling ----------

def get_flow_sample(batch: dict, method: str, device: str, cfm_obj=None, geo_eps=1e-6):
    """
    [REVISED] Return t, xt, ut for the chosen method.
    Batch is now a dictionary.
    """

    # Common data
    x0 = batch['x0'].to(device)
    x1 = batch['x1'].to(device)
    B, N, _ = x0.shape

    if method == 'linear':
        if cfm_obj is None:
            raise ValueError("ConditionalFlowMatcher required for linear method")
        t, xt, ut = cfm_obj.sample_location_and_conditional_flow(x0, x1)

    elif method == 'kendall':
        theta = batch['theta'].to(device)
        t, xt, ut = sample_geodesic_stable(x0, x1, theta, device, geo_eps)

    elif method == 'angle':
        # [NEW] Angle FM implementation

        # 1. Get required data
        angles_0 = batch['angles_0'].to(device)       # (B, N)
        angles_1 = batch['angles_1'].to(device)       # (B, N)
        lengths  = batch['edge_lengths'].to(device)   # (B, N) <-- [CORRECTED]
        path = batch['path'].to(device)           # (B, N)

        # 2. Sample time t
        t = torch.rand(B, device=device, dtype=x0.dtype)
        t_ = t.view(B, 1) # (B, 1) for broadcasting

        # 3. Interpolate angles. Lengths are constant.
        angles_t = (1 - t_) * angles_0 + t_ * angles_1
        lengths_t = lengths # <-- [CORRECTED] Constant, as requested

        # 4. Define velocities in angle/length space
        ut_angles = angles_1 - angles_0
        ut_lengths = torch.zeros_like(lengths_t) # <-- [CORRECTED] Velocity is zero

        # 5. Define the reconstruction function wrapper for jvp
        #    This wrapper *must* include the pre-processing step
        #    to ensure xt lives in the same (unit-norm) space as x0/x1.
        def _reconstruct_wrapper(a, l):
            unprocessed_xt = reconstruct_polygon_batch(a, l, path)
            processed_xt = pre_process_shape_torch(unprocessed_xt)
            return processed_xt

        # 6. Compute (xt, ut) using Jacobian-vector product
        #    xt = processed_xt(angles_t, lengths_t)
        #    ut = J_processed * [ut_angles, ut_lengths]
        xt, ut = jvp(
            _reconstruct_wrapper,
            (angles_t, lengths_t),
            (ut_angles, ut_lengths) # Pass in ut_lengths=0
        )

    else:
        raise NotImplementedError(f"FM Method {method} not implemented")

    return t, xt, ut


# ---------- [MODIFIED] Main Training Loop ----------

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"  -> GPU: {torch.cuda.get_device_name(0)}")

    out_dir = args.output_dir or f"checkpoints_{args.fm_method}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading data from: train={args.train_data_path}, test={args.test_data_path}")
    train_loader, test_loader, _ = get_loaders(args.train_data_path, args.test_data_path, args.batch_size)
    if train_loader is None:
        raise RuntimeError("Data loaders could not be created")

    model = StrongEquivariantVectorField(
        n_points=args.num_points,
        embed_dim=args.embed_dim,
        t_emb_dim=args.t_emb_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)

    print(f"Model params: {count_parameters(model):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=args.lr_factor, patience=args.lr_patience)

    cfm = None
    if args.fm_method == 'linear':
        cfm = ConditionalFlowMatcher(sigma=0.0)

    stability_eps = 1e-6

    train_losses = []
    test_losses = []
    best_test_loss = float('inf')

    print(f"Starting training for {args.log_epochs} epochs")

    for epoch in range(args.log_epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.log_epochs}", leave=False)

        # [MODIFIED] Batch is now a dict, but get_flow_sample handles it.
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()

            # get_flow_sample now handles the dict 'batch'
            t, xt, ut = get_flow_sample(batch, args.fm_method, device, cfm_obj=cfm, geo_eps=stability_eps)

            vt = model(xt, t)
            loss = torch.mean((vt - ut) ** 2)

            loss.backward()

            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)

            optimizer.step()

            epoch_loss += loss.item()
            if batch_idx % 10 == 0:
                pbar.set_postfix(loss=loss.item())

        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)

        # validation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                # [MODIFIED] Batch is a dict, get_flow_sample handles it.
                t, xt, ut = get_flow_sample(batch, args.fm_method, device, cfm_obj=cfm, geo_eps=stability_eps)
                vt = model(xt, t)
                test_loss += float(((vt - ut) ** 2).mean().item())

        avg_test = test_loss / len(test_loader)
        test_losses.append(avg_test)

        scheduler.step(avg_test)

        print(f"Epoch {epoch+1}/{args.log_epochs}: Train {avg_train:.6f}, Test {avg_test:.6f}, LR {optimizer.param_groups[0]['lr']:.2e}")

        if avg_test < best_test_loss:
            best_test_loss = avg_test
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pt'))
            print(f"  -> New best saved: {best_test_loss:.6f}")

    torch.save(model.state_dict(), os.path.join(out_dir, 'final_model.pt'))
    plot_loss_curves(train_losses, test_losses, os.path.join(out_dir, 'loss_curve.png'))
    print("Training complete. Models & loss curve saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # [MODIFIED] Added 'angle' to choices
    parser.add_argument('--fm_method', type=str, required=True, choices=['linear', 'kendall', 'angle'])

    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--test_data_path', type=str, required=True)

    parser.add_argument('--num_points', type=int, default=50)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--t_emb_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--log_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    # [FIXED] argparse had '--add_Falsergument'
    parser.add_argument('--lr_patience', type=int, default=10)
    parser.add_argument('--lr_factor', type=float, default=0.5)

    parser.add_argument('--output_dir', type=str, default=None)

    args = parser.parse_args()
    main(args)