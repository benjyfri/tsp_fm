#!/usr/bin/env python3
"""
Robust train.py for Flow Matching (Kendall, Linear, & Angle).
- [FIXED] Model now outputs tangent vector directly.
- [FIXED] Removed redundant projection from loss calculation.
"""

import os
import time
import math
import argparse
import sys

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.autograd.functional import jvp  # For Angle FM

from torchcfm import ConditionalFlowMatcher
from tsp_flow.data_loader import get_loaders
from tsp_flow.models import StrongEquivariantVectorField
from tsp_flow.utils import count_parameters, plot_loss_curves
import torch.nn as nn

# [All helper functions: reconstruct_polygon_batch, pre_process_shape_torch,
#  sample_geodesic_stable, get_flow_sample, check_data_sanity...
#  all remain UNCHANGED from your last script. Omitted for brevity.]
# ...
def reconstruct_polygon_batch(batched_angles: torch.Tensor,
                              batched_lengths: torch.Tensor,
                              batched_path: torch.Tensor) -> torch.Tensor:
    B, N = batched_angles.shape
    device = batched_angles.device
    rel_headings = batched_angles - math.pi
    rel_headings_shifted = torch.roll(rel_headings, shifts=1, dims=1)
    rel_headings_shifted[:, 0] = 0.0
    abs_headings = torch.cumsum(rel_headings_shifted, dim=1)
    vectors_x = batched_lengths * torch.cos(abs_headings)
    vectors_y = batched_lengths * torch.sin(abs_headings)
    vectors = torch.stack([vectors_x, vectors_y], dim=2)
    points_path_order_shifted = torch.cumsum(vectors, dim=1)
    points_path_order = torch.roll(points_path_order_shifted, shifts=1, dims=1)
    points_path_order[:, 0, :] = 0.0
    path_expanded = batched_path.unsqueeze(-1).expand(-1, -1, 2)
    points_original_order = torch.zeros_like(points_path_order)
    points_original_order.scatter_(dim=1, index=path_expanded, src=points_path_order)
    points_centered = points_original_order - torch.mean(points_original_order, dim=1, keepdim=True)
    return points_centered

def pre_process_shape_torch(X: torch.Tensor) -> torch.Tensor:
    X_c = X - torch.mean(X, dim=1, keepdim=True)
    f_norm = torch.norm(X_c, p='fro', dim=(1, 2), keepdim=True)
    f_norm = torch.clamp(f_norm, min=1e-9)
    X_normed = X_c / f_norm
    return X_normed

def sample_geodesic_stable(x0, x1, theta, device, eps=1e-6):
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

def get_flow_sample(batch: dict, method: str, device: str, cfm_obj=None, geo_eps=1e-6):
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
        angles_0 = batch['angles_0'].to(device)
        angles_1 = batch['angles_1'].to(device)
        lengths  = batch['edge_lengths'].to(device)
        path = batch['path'].to(device)
        t = torch.rand(B, device=device, dtype=x0.dtype)
        t_ = t.view(B, 1)
        angles_t = (1 - t_) * angles_0 + t_ * angles_1
        lengths_t = lengths
        ut_angles = angles_1 - angles_0
        ut_lengths = torch.zeros_like(lengths_t)
        def _reconstruct_wrapper(a, l):
            unprocessed_xt = reconstruct_polygon_batch(a, l, path)
            processed_xt = pre_process_shape_torch(unprocessed_xt)
            return processed_xt
        xt, ut = jvp(
            _reconstruct_wrapper,
            (angles_t, lengths_t),
            (ut_angles, ut_lengths)
        )
    else:
        raise NotImplementedError(f"FM Method {method} not implemented")
    return t, xt, ut

def check_data_sanity(loader, device, eps=1e-5):
    print("\n--- [DATA SANITY CHECK] ---")
    is_sane = True
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Checking data sanity")):
            x0 = batch['x0'].to(device); x1 = batch['x1'].to(device); theta = batch['theta'].to(device)
            x0_mean = torch.mean(x0, dim=1); x1_mean = torch.mean(x1, dim=1)
            if not torch.allclose(x0_mean, torch.zeros_like(x0_mean), atol=eps):
                print(f"  [FAIL] Batch {batch_idx}: x0 is not centered! Max mean norm: {torch.max(torch.norm(x0_mean, dim=1)).item()}"); is_sane = False
            if not torch.allclose(x1_mean, torch.zeros_like(x1_mean), atol=eps):
                print(f"  [FAIL] Batch {batch_idx}: x1 is not centered! Max mean norm: {torch.max(torch.norm(x1_mean, dim=1)).item()}"); is_sane = False
            x0_norm_F = torch.norm(x0, p='fro', dim=(1, 2)); x1_norm_F = torch.norm(x1, p='fro', dim=(1, 2))
            if not torch.allclose(x0_norm_F, torch.ones_like(x0_norm_F), atol=eps):
                print(f"  [FAIL] Batch {batch_idx}: x0 does not have unit norm! Norms: {x0_norm_F}"); is_sane = False
            if not torch.allclose(x1_norm_F, torch.ones_like(x1_norm_F), atol=eps):
                print(f"  [FAIL] Batch {batch_idx}: x1 does not have unit norm! Norms: {x1_norm_F}"); is_sane = False
            inner_prod = torch.sum(x0 * x1, dim=(1, 2)); inner_prod = torch.clamp(inner_prod, -1.0, 1.0)
            theta_recalc = torch.acos(inner_prod)
            if not torch.allclose(theta, theta_recalc, atol=eps):
                print(f"  [FAIL] Batch {batch_idx}: Theta is inconsistent!"); print(f"    Loaded theta (sample 0): {theta[0].item()}"); print(f"    Recalc theta (sample 0): {theta_recalc[0].item()}"); is_sane = False
            if not is_sane: print("  [HALT] Sanity check failed on first bad batch."); return False
    if is_sane: print("  [SUCCESS] All batches passed sanity checks (Centering, Unit Norm, Theta).")
    print("--- [END DATA SANITY CHECK] ---\n")
    return True

# ---------- [MODIFIED] Main Training Loop ----------

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        if args.gpu_id >= torch.cuda.device_count():
            print(f"Error: GPU ID {args.gpu_id} requested... defaulting to 0.")
            args.gpu_id = 0
        device = f"cuda:{args.gpu_id}"
        print(f"Using device: {device}")
        print(f"  -> GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = 'cpu'
        print(f"Using device: {device} (CUDA not available)")

    out_dir = args.output_dir or f"checkpoints_{args.fm_method}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading data from: train={args.train_data_path}, test={args.test_data_path}")
    train_loader, test_loader, _ = get_loaders(args.train_data_path, args.test_data_path, args.batch_size)
    if train_loader is None:
        raise RuntimeError("Data loaders could not be created")

    if args.fm_method == 'kendall':
        if not check_data_sanity(train_loader, device):
            print("Data sanity check failed. Exiting.")
            sys.exit(1)

    model = StrongEquivariantVectorField(
        n_points=args.num_points,
        embed_dim=args.embed_dim,
        t_emb_dim=args.t_emb_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)

    print("Zero-initializing output head...")
    try:
        model.output_head[4].weight.data.zero_()
        model.output_head[4].bias.data.zero_()
        print("  -> Success (Zero-init).")
    except Exception as e:
        print(f"  -> FAILED to zero-init output_head[4]: {e}")
        sys.exit(1)

    print(f"Model params: {count_parameters(model):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    cfm = None
    if args.fm_method == 'linear':
        cfm = ConditionalFlowMatcher(sigma=0.0)

    stability_eps = 1e-6
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    global_step = 0

    print(f"Starting training for {args.log_epochs} epochs")
    print(f"  -> Target LR: {args.lr:.2e}, Warmup Steps: {args.warmup_steps}")
    print(f"  -> NOTE: Scheduler is DISABLED for this demo.")

    for epoch in range(args.log_epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.log_epochs}", leave=False)

        for batch_idx, batch in enumerate(pbar):
            if global_step < args.warmup_steps:
                current_lr = args.lr * (global_step + 1) / args.warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            elif global_step == args.warmup_steps:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr

            optimizer.zero_grad()

            t, xt, ut = get_flow_sample(batch, args.fm_method, device, cfm_obj=cfm, geo_eps=stability_eps)
            t = t.to(dtype=xt.dtype, device=xt.device).view(xt.shape[0], 1)

            # --- [FIX] vt is now the tangent vector vt_tan ---
            vt_tan = model(xt, t)
            # --- [END FIX] ---

            dot = torch.sum(xt * ut, dim=(1,2), keepdim=True)
            ut_tan = ut - dot * xt

            # --- [FIX] REMOVED REDUNDANT PROJECTION ---
            # dot_v = torch.sum(xt * vt, dim=(1,2), keepdim=True)
            # vt_tan = vt - dot_v * xt
            # --- [END FIX] ---

            # [KENDALL DEBUG]
            if args.fm_method == 'kendall' and batch_idx == 0:
                with torch.no_grad():
                    vt_tan_norm = torch.norm(vt_tan.view(vt_tan.shape[0], -1), dim=1)
                    ut_tan_norm = torch.norm(ut_tan.view(ut_tan.shape[0], -1), dim=1)
                    cos_ut_vt_tan = (torch.sum(ut_tan * vt_tan, dim=(1,2)) /
                                     (ut_tan_norm * vt_tan_norm + 1e-12)).clamp(-1, 1)

                    print(f"\n[KENDALL DEBUG] Epoch {epoch+1} Batch {batch_idx:04d}")
                    print(f"  ut_tan_norm  mean={ut_tan_norm.mean():.4e} ±{ut_tan_norm.std():.4e}")
                    print(f"  vt_tan_norm  mean={vt_tan_norm.mean():.4e} ±{vt_tan_norm.std():.4e}")
                    print(f"  cos(ut_tan,vt_tan) mean={cos_ut_vt_tan.mean():+.3e}")

            # --- [FIX] Loss is now (ut_tan - vt_tan) ---
            loss = torch.mean((ut_tan - vt_tan) ** 2)
            # --- [END FIX] ---

            loss.backward()

            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)

            optimizer.step()
            global_step += 1
            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)

        # validation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                t, xt, ut = get_flow_sample(batch, args.fm_method, device, cfm_obj=cfm, geo_eps=stability_eps)
                t = t.to(dtype=xt.dtype, device=xt.device).view(xt.shape[0], 1)

                # --- [FIX] vt is now the tangent vector vt_tan ---
                vt_tan = model(xt, t)
                # --- [END FIX] ---

                dot = torch.sum(xt * ut, dim=(1,2), keepdim=True)
                ut_tan = ut - dot * xt

                # --- [FIX] REMOVED REDUNDANT PROJECTION ---
                # dot_v = torch.sum(xt * vt, dim=(1,2), keepdim=True)
                # vt_tan = vt - dot_v * xt
                # --- [END FIX] ---

                # --- [FIX] Loss is now (ut_tan - vt_tan) ---
                loss_val = torch.mean((ut_tan - vt_tan) ** 2)
                # --- [END FIX] ---
                test_loss += loss_val.item()

        avg_test = test_loss / len(test_loader)
        test_losses.append(avg_test)

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
    parser.add_argument('--fm_method', type=str, required=True, choices=['linear', 'kendall', 'angle'])
    parser.add_argument('--gpu-id', type=int, default=7, help="ID of the GPU to use (e.g., 0, 1, 2, ...)")
    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--test_data_path', type=str, required=True)
    parser.add_argument('--num_points', type=int, default=50)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--log_epochs', type=int, default=100) # Reverted to 100 for a quick test
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size. 32 is good for the 100-sample demo.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-4, help="Target learning rate after warmup. 1e-4 is a stable default.")
    parser.add_argument('--warmup_steps', type=int, default=10, help="Number of steps for linear LR warmup. 10 is good for this demo.")
    parser.add_argument('--grad_clip_norm', type=float, default=5.0, help="Max norm for gradient clipping")
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    main(args)