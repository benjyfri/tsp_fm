#!/usr/bin/env python3
"""
COMPREHENSIVE DIAGNOSTIC TRAINING SCRIPT FOR KENDALL FLOW MATCHING

This script performs exhaustive checks to identify why Kendall FM fails to train
while Linear FM succeeds. It checks:
1. Data consistency and numerical properties
2. Geodesic sampling stability
3. Gradient flow and magnitude
4. Loss landscape properties
5. Model output characteristics
6. Tangent space projection accuracy
7. Time embedding distribution
8. Comparison with linear baseline
"""

import os
import time
import math
import argparse
import sys
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.autograd.functional import jvp

from torchcfm import ConditionalFlowMatcher
# Assuming these are in your python path
from tsp_flow.data_loader import get_loaders
from tsp_flow.models import StrongEquivariantVectorField
from tsp_flow.utils import count_parameters, plot_loss_curves
import torch.nn as nn

# ============================================================================
# HELPER FUNCTIONS (unchanged from original)
# ============================================================================

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

    theta_dev = theta.to(device=device, dtype=x0.dtype).view(B, 1, 1)

    # Clamp theta to be slightly away from 0 and pi
    theta_clamped = torch.clamp(theta_dev, min=eps, max=(math.pi - eps))

    a = (1 - t_) * theta_clamped
    b = t_ * theta_clamped

    sin_theta_safe = torch.sin(theta_clamped) # Already clamped, so > 0

    sa = torch.sin(a)
    sb = torch.sin(b)
    ca = torch.cos(a)
    cb = torch.cos(b)

    xt_geo = (sa / sin_theta_safe) * x0 + (sb / sin_theta_safe) * x1
    ut_geo = (theta_clamped / sin_theta_safe) * (cb * x1 - ca * x0)

    # Use linear interpolation for angles very close to 0 or pi
    mask_near = (theta_dev <= eps) | ((math.pi - theta_dev) <= eps)
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
        xt, ut = jvp(_reconstruct_wrapper, (angles_t, lengths_t), (ut_angles, ut_lengths))
    else:
        raise NotImplementedError(f"FM Method {method} not implemented")
    return t, xt, ut

# ============================================================================
# DIAGNOSTIC FUNCTIONS
# ============================================================================

def check_data_sanity_detailed(loader, device, eps=1e-5):
    """Enhanced data sanity check with detailed statistics."""
    print("\n" + "="*80)
    print("🔬 DETAILED DATA SANITY CHECK")
    print("="*80)

    all_stats = {
        'x0_centering_error': [],
        'x1_centering_error': [],
        'x0_norm_error': [],
        'x1_norm_error': [],
        'theta_consistency_error': [],
        'theta_values': [],
        'inner_products': []
    }
    is_sane = True

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Checking data")):
            x0 = batch['x0'].to(device)
            x1 = batch['x1'].to(device)
            theta = batch['theta'].to(device)

            # Check centering
            x0_mean = torch.mean(x0, dim=1)
            x1_mean = torch.mean(x1, dim=1)
            all_stats['x0_centering_error'].append(torch.norm(x0_mean, dim=1).cpu())
            all_stats['x1_centering_error'].append(torch.norm(x1_mean, dim=1).cpu())

            # Check normalization
            x0_norm = torch.norm(x0, p='fro', dim=(1, 2))
            x1_norm = torch.norm(x1, p='fro', dim=(1, 2))
            all_stats['x0_norm_error'].append((x0_norm - 1.0).abs().cpu())
            all_stats['x1_norm_error'].append((x1_norm - 1.0).abs().cpu())

            # Check theta consistency
            inner_prod = torch.sum(x0 * x1, dim=(1, 2)).clamp(-1.0, 1.0)
            theta_recalc = torch.acos(inner_prod)
            theta_error = (theta - theta_recalc).abs()
            all_stats['theta_consistency_error'].append(theta_error.cpu())
            all_stats['theta_values'].append(theta.cpu())
            all_stats['inner_products'].append(inner_prod.cpu())

            if torch.max(theta_error) > 1e-3:
                is_sane = False

    # Compute and display statistics
    for key in ['x0_centering_error', 'x1_centering_error', 'x0_norm_error',
                'x1_norm_error', 'theta_consistency_error']:
        vals = torch.cat(all_stats[key])
        print(f"\n{key}:")
        print(f"  Mean: {vals.mean():.6e}")
        print(f"  Std:  {vals.std():.6e}")
        print(f"  Max:  {vals.max():.6e}")
        print(f"  Min:  {vals.min():.6e}")

    # Theta distribution
    theta_vals = torch.cat(all_stats['theta_values'])
    print(f"\nTheta distribution (in radians):")
    print(f"  Mean: {theta_vals.mean():.4f}")
    print(f"  Std:  {theta_vals.std():.4f}")
    print(f"  Min:  {theta_vals.min():.4f}")
    print(f"  Max:  {theta_vals.max():.4f}")
    print(f"  Median: {theta_vals.median():.4f}")
    print(f"  % near 0 (theta < 0.01): {torch.sum(theta_vals < 0.01) / len(theta_vals) * 100:.2f}%")
    print(f"  % near pi (theta > 3.13): {torch.sum(theta_vals > 3.13) / len(theta_vals) * 100:.2f}%")

    inner_prods = torch.cat(all_stats['inner_products'])
    print(f"\nInner product <x0, x1> distribution:")
    print(f"  Mean: {inner_prods.mean():.4f}")
    print(f"  Std:  {inner_prods.std():.4f}")
    print(f"  Min:  {inner_prods.min():.4f}")
    print(f"  Max:  {inner_prods.max():.4f}")

    if not is_sane:
        print("\n[WARNING] Theta consistency check failed! Max error > 1e-3.")
    else:
        print("\n[SUCCESS] Data sanity checks passed.")

    print("="*80 + "\n")
    return is_sane

def check_geodesic_sampling(batch, device, eps=1e-6, num_checks=10):
    """Verify geodesic sampling produces valid points on the sphere."""
    print("\n" + "="*80)
    print("🔬 GEODESIC SAMPLING DIAGNOSTICS")
    print("="*80)

    x0 = batch['x0'].to(device)
    x1 = batch['x1'].to(device)
    theta = batch['theta'].to(device)

    issues = []
    ut_norms = []

    for _ in range(num_checks):
        t, xt, ut = sample_geodesic_stable(x0, x1, theta, device, eps)

        # Check xt normalization
        xt_norm = torch.norm(xt, p='fro', dim=(1, 2))
        norm_error = (xt_norm - 1.0).abs()
        if norm_error.max() > 1e-4:
            issues.append(f"xt norm error: max={norm_error.max():.6e}, mean={norm_error.mean():.6e}")

        # Check xt centering
        xt_mean = torch.mean(xt, dim=1)
        center_error = torch.norm(xt_mean, dim=1)
        if center_error.max() > 1e-4:
            issues.append(f"xt centering error: max={center_error.max():.6e}")

        # Check ut is tangent (orthogonal to xt)
        # This is the "ambient" ut, not the projected ut_tan, but for geodesics,
        # the velocity vector ut should already be tangent to xt.
        dot_xt_ut = torch.sum(xt * ut, dim=(1, 2))
        if dot_xt_ut.abs().max() > 1e-3:
            issues.append(f"ut not tangent: max dot product={dot_xt_ut.abs().max():.6e}")

        ut_norms.append(torch.norm(ut.view(ut.shape[0], -1), dim=1).cpu())

    if issues:
        print("\n[ISSUES FOUND]:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ All geodesic sampling checks passed!")

    # Check ut magnitude distribution
    all_ut_norms = torch.cat(ut_norms)
    print(f"\n'ut' (geodesic velocity) norm distribution:")
    print(f"  Mean: {all_ut_norms.mean():.4e}")
    print(f"  Std:  {all_ut_norms.std():.4e}")
    print(f"  Min:  {all_ut_norms.min():.4e}")
    print(f"  Max:  {all_ut_norms.max():.4e}")

    print("="*80 + "\n")
    return len(issues) == 0

def check_gradient_flow(model, batch, device, method, cfm_obj=None):
    """Check gradient magnitudes and flow through the network."""
    print("\n" + "="*80)
    print("🔬 GRADIENT FLOW DIAGNOSTICS (1 Batch)")
    print("="*80)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Dummy optimizer
    optimizer.zero_grad()

    t, xt, ut = get_flow_sample(batch, method, device, cfm_obj)
    t = t.to(dtype=xt.dtype, device=xt.device).view(xt.shape[0], 1)

    # --- THIS IS THE CRITICAL SECTION ---
    vt_raw = model(xt, t)  # Raw model output

    # Project raw output to tangent space
    dot_xt_vt = torch.sum(xt * vt_raw, dim=(1,2), keepdim=True)
    vt_tan = vt_raw - dot_xt_vt * xt

    # Project target to tangent space
    dot_xt_ut = torch.sum(xt * ut, dim=(1,2), keepdim=True)
    ut_tan = ut - dot_xt_ut * xt
    # --- END CRITICAL SECTION ---

    loss = torch.mean((ut_tan - vt_tan) ** 2)

    loss.backward()

    print(f"\nLoss: {loss.item():.6e}")
    print(f"Gradient statistics by layer:")

    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.abs().mean().item()
            grad_max = param.grad.abs().max().item()
            total_norm += grad_norm ** 2
            if "output_head" in name or "time_embed" in name:
                print(f"  {name}:")
                print(f"    Norm: {grad_norm:.6e}, Mean: {grad_mean:.6e}, Max: {grad_max:.6e}")
        else:
            print(f"  {name}: [NO GRADIENT]")

    total_norm = math.sqrt(total_norm)
    print(f"\nTotal gradient norm: {total_norm:.6e}")

    zero_grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().max() < 1e-10)
    print(f"Parameters with near-zero gradients: {zero_grads}")

    optimizer.zero_grad() # Clean up
    print("="*80 + "\n")
    return total_norm

def check_model_output_properties(model, batch, device, method, cfm_obj=None, prefix=""):
    """Analyze model output characteristics."""
    print("\n" + "="*80)
    print(f"🔬 {prefix} MODEL OUTPUT DIAGNOSTICS (1 Batch)")
    print("="*80)

    model.eval()
    with torch.no_grad():
        t, xt, ut = get_flow_sample(batch, method, device, cfm_obj)
        t = t.to(dtype=xt.dtype, device=xt.device).view(xt.shape[0], 1)

        # --- THIS IS THE CRITICAL SECTION ---
        vt_raw = model(xt, t)  # Raw model output

        # Project raw output to tangent space
        dot_xt_vt_raw = torch.sum(xt * vt_raw, dim=(1,2), keepdim=True)
        vt_tan = vt_raw - dot_xt_vt_raw * xt

        # Project target to tangent space
        dot_xt_ut = torch.sum(xt * ut, dim=(1,2), keepdim=True)
        ut_tan = ut - dot_xt_ut * xt
        # --- END CRITICAL SECTION ---

        # Compute statistics
        vt_raw_norm = torch.norm(vt_raw.view(vt_raw.shape[0], -1), dim=1)
        vt_tan_norm = torch.norm(vt_tan.view(vt_tan.shape[0], -1), dim=1)
        ut_tan_norm = torch.norm(ut_tan.view(ut_tan.shape[0], -1), dim=1)

        print(f"\nTarget (ut_tan) statistics:")
        print(f"  Mean norm: {ut_tan_norm.mean():.6e}")
        print(f"  Std norm:  {ut_tan_norm.std():.6e}")
        print(f"  Min norm:  {ut_tan_norm.min():.6e}")
        print(f"  Max norm:  {ut_tan_norm.max():.6e}")

        print(f"\nModel RAW output (vt_raw) statistics:")
        print(f"  Mean norm: {vt_raw_norm.mean():.6e}")
        print(f"  Std norm:  {vt_raw_norm.std():.6e}")

        print(f"\nModel TANGENT output (vt_tan) statistics:")
        print(f"  Mean norm: {vt_tan_norm.mean():.6e}")
        print(f"  Std norm:  {vt_tan_norm.std():.6e}")

        # Check alignment
        cos_sim = (torch.sum(ut_tan * vt_tan, dim=(1,2)) /
                   (ut_tan_norm * vt_tan_norm + 1e-12)).clamp(-1, 1)

        print(f"\nCosine similarity between ut_tan and vt_tan:")
        print(f"  Mean: {cos_sim.mean():.4f}")
        print(f"  Std:  {cos_sim.std():.4f}")
        print(f"  Min:  {cos_sim.min():.4f}")
        print(f"  Max:  {cos_sim.max():.4f}")

        # Check tangent space constraint
        dot_xt_vt = torch.sum(xt * vt_tan, dim=(1,2))
        print(f"\nTangent space constraint check <xt, vt_tan>:")
        print(f"  Mean: {dot_xt_vt.mean():.6e}")
        print(f"  Max abs: {dot_xt_vt.abs().max():.6e}")

        loss = torch.mean((ut_tan - vt_tan) ** 2)
        print(f"\nMSE Loss: {loss.item():.6e}")

    print("="*80 + "\n")
    return loss.item()

def check_time_embedding_distribution(model, device):
    """Check time embedding behavior."""
    print("\n" + "="*80)
    print("🔬 TIME EMBEDDING DIAGNOSTICS")
    print("="*80)

    model.eval()
    with torch.no_grad():
        t_values = torch.linspace(0, 1, 11, device=device)
        for t_val in t_values:
            t_batch = t_val.view(1)
            t_emb = model.time_embed(t_batch) # Use internal method
            print(f"t={t_val:.2f}: emb_mean={t_emb.mean():.4f}, emb_std={t_emb.std():.4f}, "
                  f"emb_norm={t_emb.norm():.4f}")
    print("="*80 + "\n")

def compare_linear_vs_kendall(train_loader, device):
    """Direct comparison of Linear vs Kendall sampling."""
    print("\n" + "="*80)
    print("🔬 LINEAR VS KENDALL TARGET COMPARISON")
    print("="*80)

    cfm = ConditionalFlowMatcher(sigma=0.0)
    batch = next(iter(train_loader))

    # Linear sampling
    t_lin, xt_lin, ut_lin = get_flow_sample(batch, 'linear', device, cfm_obj=cfm)
    dot_lin = torch.sum(xt_lin * ut_lin, dim=(1,2), keepdim=True)
    ut_tan_lin = ut_lin - dot_lin * xt_lin
    ut_tan_lin_norm = torch.norm(ut_tan_lin.view(ut_tan_lin.shape[0], -1), dim=1)

    # Kendall sampling
    t_ken, xt_ken, ut_ken = get_flow_sample(batch, 'kendall', device)
    dot_ken = torch.sum(xt_ken * ut_ken, dim=(1,2), keepdim=True)
    ut_tan_ken = ut_ken - dot_ken * xt_ken
    ut_tan_ken_norm = torch.norm(ut_tan_ken.view(ut_tan_ken.shape[0], -1), dim=1)

    print("\n--- Target Tangent Vector Norms (ut_tan) ---")
    print("\nLinear method:")
    print(f"  Mean norm: {ut_tan_lin_norm.mean():.6e}")
    print(f"  Std norm:  {ut_tan_lin_norm.std():.6e}")
    print(f"  Min norm:  {ut_tan_lin_norm.min():.6e}")
    print(f"  Max norm:  {ut_tan_lin_norm.max():.6e}")

    print("\nKendall method:")
    print(f"  Mean norm: {ut_tan_ken_norm.mean():.6e}")
    print(f"  Std norm:  {ut_tan_ken_norm.std():.6e}")
    print(f"  Min norm:  {ut_tan_ken_norm.min():.6e}")
    print(f"  Max norm:  {ut_tan_ken_norm.max():.6e}")

    ratio = ut_tan_ken_norm.mean() / (ut_tan_lin_norm.mean() + 1e-9)
    print(f"\nRatio (Kendall / Linear): {ratio:.4f}")
    if ratio < 0.1 or ratio > 10:
        print("[WARNING] Target vector magnitudes are significantly different!")
        print("          This could require different learning rates.")
    else:
        print("[INFO] Target vector magnitudes are comparable.")

    print("="*80 + "\n")

# ============================================================================
# MAIN DIAGNOSTIC TRAINING LOOP
# ============================================================================

def main(args):
    print("\n" + "="*80)
    print("COMPREHENSIVE KENDALL FLOW MATCHING DIAGNOSTICS")
    print(f"Method: {args.fm_method}")
    print("="*80 + "\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        if args.gpu_id >= torch.cuda.device_count():
            print(f"Error: GPU ID {args.gpu_id} requested... defaulting to 0.")
            args.gpu_id = 0
        device = f"cuda:{args.gpu_id}"
        print(f"Using device: {device} ({torch.cuda.get_device_name(args.gpu_id)})")
    else:
        device = 'cpu'
        print(f"Using device: {device}")

    out_dir = args.output_dir or f"diagnostics_{args.fm_method}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving results to: {out_dir}")

    # Load data
    print(f"\nLoading data...")
    train_loader, test_loader, _ = get_loaders(
        args.train_data_path, args.test_data_path, args.batch_size
    )
    if train_loader is None:
        raise RuntimeError("Data loaders could not be created")

    # Get first batch for diagnostics
    first_batch = next(iter(train_loader))

    # ========== DIAGNOSTIC PHASE 1: DATA ==========
    print("\n" + "#"*80)
    print("# PHASE 1: DATA DIAGNOSTICS (Method: Kendall)")
    print("#"*80)

    if not check_data_sanity_detailed(train_loader, device):
        print("[ERROR] Data sanity check failed. Exiting.")
        sys.exit(1)

    check_geodesic_sampling(first_batch, device)
    compare_linear_vs_kendall(train_loader, device)

    # ========== DIAGNOSTIC PHASE 2: MODEL INITIALIZATION ==========
    print("\n" + "#"*80)
    print("# PHASE 2: MODEL INITIALIZATION")
    print("#"*80)

    model = StrongEquivariantVectorField(
        n_points=args.num_points,
        embed_dim=args.embed_dim,
        t_emb_dim=args.t_emb_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)

    print(f"\nModel parameters: {count_parameters(model):,}")

    # Zero-initialize output
    print("Zero-initializing output head...")
    try:
        with torch.no_grad():
            model.output_head[4].weight.data.zero_()
            model.output_head[4].bias.data.zero_()
        print("  -> Success (Zero-init).")
    except Exception as e:
        print(f"  -> FAILED to zero-init output_head[4]: {e}")
        sys.exit(1)

    check_time_embedding_distribution(model, device)

    # ========== DIAGNOSTIC PHASE 3: INITIAL MODEL BEHAVIOR ==========
    print("\n" + "#"*80)
    print("# PHASE 3: INITIAL MODEL BEHAVIOR (BEFORE TRAINING)")
    print("#"*80)

    cfm = ConditionalFlowMatcher(sigma=0.0) if args.fm_method == 'linear' else None

    initial_loss = check_model_output_properties(model, first_batch, device, args.fm_method, cfm, prefix="INITIAL")
    initial_grad_norm = check_gradient_flow(model, first_batch, device, args.fm_method, cfm)

    # ========== TRAINING PHASE ==========
    print("\n" + "#"*80)
    print("# PHASE 4: TRAINING WITH CONTINUOUS MONITORING")
    print("#"*80)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    test_losses = []
    grad_norms = []
    best_test_loss = float('inf')
    global_step = 0

    diagnostic_interval = max(1, len(train_loader) // 2) # Run 2 times per epoch

    for epoch in range(args.log_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_grad_norm = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.log_epochs}", leave=False)

        for batch_idx, batch in enumerate(pbar):
            # Warmup
            if global_step < args.warmup_steps:
                current_lr = args.lr * (global_step + 1) / args.warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            elif global_step == args.warmup_steps:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr


            optimizer.zero_grad()

            t, xt, ut = get_flow_sample(batch, args.fm_method, device, cfm_obj=cfm)
            t = t.to(dtype=xt.dtype, device=xt.device).view(xt.shape[0], 1)

            # --- [CRITICAL FIX] ---
            # The model outputs a raw vector vt_raw in ambient space
            vt_raw = model(xt, t)

            # We MUST project this raw vector to the tangent space of xt
            dot_xt_vt = torch.sum(xt * vt_raw, dim=(1,2), keepdim=True)
            vt_tan = vt_raw - dot_xt_vt * xt

            # We also project the target vector ut to the tangent space of xt
            # (ut is not guaranteed to be perfectly tangent due to numerics)
            dot_xt_ut = torch.sum(xt * ut, dim=(1,2), keepdim=True)
            ut_tan = ut - dot_xt_ut * xt

            # The loss compares the two TANGENT vectors
            loss = torch.mean((ut_tan - vt_tan) ** 2)
            # --- [END CRITICAL FIX] ---

            loss.backward()

            # Track gradient norm before clipping
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.norm().item()
                    total_norm += param_norm ** 2
            total_norm = math.sqrt(total_norm)
            epoch_grad_norm += total_norm

            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)

            optimizer.step()
            global_step += 1
            epoch_loss += loss.item()

            pbar.set_postfix(loss=loss.item(), grad_norm=total_norm, lr=optimizer.param_groups[0]['lr'])

            # Periodic detailed diagnostics
            if (batch_idx > 0 and batch_idx % diagnostic_interval == 0) or \
                    (epoch == 0 and batch_idx == 0):
                print(f"\n--- Detailed check at epoch {epoch+1}, batch {batch_idx} ---")
                check_model_output_properties(model, batch, device, args.fm_method, cfm, prefix="MID-TRAIN")

        avg_train = epoch_loss / len(train_loader)
        avg_grad_norm = epoch_grad_norm / len(train_loader)
        train_losses.append(avg_train)
        grad_norms.append(avg_grad_norm)

        # Validation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                t, xt, ut = get_flow_sample(batch, args.fm_method, device, cfm_obj=cfm)
                t = t.to(dtype=xt.dtype, device=xt.device).view(xt.shape[0], 1)

                # --- [CRITICAL FIX] (Same as in training) ---
                vt_raw = model(xt, t)
                dot_xt_vt = torch.sum(xt * vt_raw, dim=(1,2), keepdim=True)
                vt_tan = vt_raw - dot_xt_vt * xt

                dot_xt_ut = torch.sum(xt * ut, dim=(1,2), keepdim=True)
                ut_tan = ut - dot_xt_ut * xt

                loss_val = torch.mean((ut_tan - vt_tan) ** 2)
                # --- [END CRITICAL FIX] ---
                test_loss += loss_val.item()

        avg_test = test_loss / len(test_loader)
        test_losses.append(avg_test)

        print(f"\nEpoch {epoch+1}/{args.log_epochs}:")
        print(f"  Train Loss: {avg_train:.6e}")
        print(f"  Test Loss:  {avg_test:.6e}")
        print(f"  Grad Norm:  {avg_grad_norm:.6e}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        if avg_test < best_test_loss:
            best_test_loss = avg_test
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pt'))
            print(f"  ✓ New best model saved!")

        # Check for training failure
        if epoch > 10 and avg_train > initial_loss:
            print("\n" + "!"*80)
            print("WARNING: Training appears to be failing! Loss is not decreasing.")
            print(f"Initial loss: {initial_loss:.6e}, Current loss: {avg_train:.6e}")
            print("!"*80)

    # ========== FINAL DIAGNOSTICS ==========
    print("\n" + "#"*80)
    print("# PHASE 5: FINAL DIAGNOSTICS (AFTER TRAINING)")
    print("#"*80)

    final_loss = check_model_output_properties(model, first_batch, device, args.fm_method, cfm, prefix="FINAL")

    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Initial loss: {initial_loss:.6e}")
    print(f"Final loss (train): {train_losses[-1]:.6e}")
    print(f"Final loss (test):  {test_losses[-1]:.6e}")
    print(f"Best test loss: {best_test_loss:.6e}")
    print(f"Initial grad norm: {initial_grad_norm:.6e}")
    print(f"Mean grad norm:  {np.mean(grad_norms):.6e}")
    print("="*80 + "\n")

    # Save final model and plots
    torch.save(model.state_dict(), os.path.join(out_dir, 'final_model.pt'))

    # Save diagnostic data
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Diagnostic Plots - {args.fm_method.upper()} Method', fontsize=16)

    # Plot 1: Loss curves
    axes[0, 0].plot(train_losses, label='Train', linewidth=2)
    axes[0, 0].plot(test_losses, label='Test', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Test Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')

    # Plot 2: Gradient norms
    axes[0, 1].plot(grad_norms, linewidth=2, color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Gradient Norm')
    axes[0, 1].set_title('Avg. Gradient Norm per Epoch')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')

    # Plot 3: Loss ratio (test/train)
    loss_ratios = [v / t if t > 1e-9 else 0 for t, v in zip(train_losses, test_losses)]
    axes[1, 0].plot(loss_ratios, linewidth=2, color='red')
    axes[1, 0].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Test Loss / Train Loss')
    axes[1, 0].set_title('Overfitting Indicator')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 10) # Clip y-axis for readability

    # Plot 4: LR
    # We can't easily plot LR from inside main, but we can plot loss improvement
    if len(train_losses) > 1:
        train_improvements = [train_losses[i-1] - train_losses[i] for i in range(1, len(train_losses))]
        axes[1, 1].plot(train_improvements, linewidth=2, color='purple')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Improvement (Delta)')
        axes[1, 1].set_title('Per-Epoch Loss Improvement')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(out_dir, 'diagnostic_plots.png'), dpi=150)
    plt.close()

    # Save numerical results to file
    with open(os.path.join(out_dir, 'diagnostic_results.txt'), 'w') as f:
        f.write("="*80 + "\n")
        f.write("KENDALL FLOW MATCHING DIAGNOSTIC RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Method: {args.fm_method}\n")
        f.write(f"Epochs: {args.log_epochs}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Batch size: {args.batch_size}\n\n")
        f.write(f"Initial loss: {initial_loss:.6e}\n")
        f.write(f"Final train loss: {train_losses[-1]:.6e}\n")
        f.write(f"Final test loss: {test_losses[-1]:.6e}\n")
        f.write(f"Best test loss: {best_test_loss:.6e}\n\n")
        f.write(f"Initial gradient norm: {initial_grad_norm:.6e}\n")
        f.write(f"Final gradient norm: {grad_norms[-1]:.6e}\n")
        f.write(f"Mean gradient norm: {np.mean(grad_norms):.6e}\n\n")
        f.write("Train losses by epoch:\n")
        for i, loss in enumerate(train_losses):
            f.write(f"  Epoch {i+1}: {loss:.6e}\n")
        f.write("\nTest losses by epoch:\n")
        for i, loss in enumerate(test_losses):
            f.write(f"  Epoch {i+1}: {loss:.6e}\n")

    print(f"Diagnostic results saved to {out_dir}/")
    print("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comprehensive Kendall FM Diagnostics')
    parser.add_argument('--fm_method', type=str, default='kendall',
                        choices=['linear', 'kendall', 'angle'],
                        help='Flow matching method (default: kendall)')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--train_data_path', type=str, required=True,
                        help='Path to training data')
    parser.add_argument('--test_data_path', type=str, required=True,
                        help='Path to test data')
    parser.add_argument('--num_points', type=int, default=50,
                        help='Number of points in point cloud')
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--t_emb_dim', type=int, default=64,
                        help='Time embedding dimension')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=16,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate')
    parser.add_argument('--log_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=10,
                        help='Number of warmup steps')
    parser.add_argument('--grad_clip_norm', type=float, default=5.0,
                        help='Gradient clipping norm')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (default: diagnostics_[fm_method])')

    args = parser.parse_args()

    if args.fm_method != 'kendall':
        print(f"Warning: This script is designed to debug the 'kendall' method. You are running '{args.fm_method}'.")

    main(args)