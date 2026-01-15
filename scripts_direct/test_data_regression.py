#!/usr/bin/env python3
"""
Revised Data Verification Script
Checks compatibility for Angle Regression models.
- Uses src.dataset.load_data to ensure Angles are loaded.
- Saves all visualizations to 'visualizations/' directory.
- Performs rigorous geometric and signal integrity tests.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as path_effects
from matplotlib import collections as mc
import os
import sys
import logging
from geomstats.geometry.matrices import Matrices

# Silence Matplotlib Animation logs
logging.getLogger('matplotlib.animation').setLevel(logging.WARNING)

# --- Path Setup to import src ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from dataset_regression import load_data
except ImportError:
    print("❌ Critical Error: Could not import 'load_data' from src.dataset")
    print(f"Ensure your project structure is correct. Current sys.path: {sys.path}")
    sys.exit(1)

# --- Configuration ---
VIZ_DIR = "visualizations"
os.makedirs(VIZ_DIR, exist_ok=True)


# ============================================================================
# 1. SPECTRAL CANONICALIZATION (Source of Truth for Input X0)
# ============================================================================
@torch.no_grad()
def get_spectral_canonicalization(x, sigma_kernel=1.0, epsilon=1e-8):
    """
    Applies deterministic canonicalization to verify invariance of X0.
    """
    B, N, D = x.shape
    device, dtype = x.device, x.dtype

    # Centering & Scaling
    centroid = x.mean(dim=1, keepdim=True)
    x_centered = x - centroid
    dist = torch.cdist(x_centered, x_centered, p=2)
    avg_dist = dist.sum(dim=(1, 2)) / (N * (N - 1))
    scale = (avg_dist / sigma_kernel).view(B, 1, 1) + epsilon
    x_norm = x_centered / scale

    # Laplacian
    dist_sq = torch.cdist(x_norm, x_norm, p=2).pow(2)
    W = torch.exp(-dist_sq / (sigma_kernel ** 2))
    W.diagonal(dim1=-2, dim2=-1).zero_()
    D_vec = W.sum(dim=2) + epsilon
    D_inv_sqrt_vec = torch.rsqrt(D_vec)
    W_norm = W * D_inv_sqrt_vec.unsqueeze(1) * D_inv_sqrt_vec.unsqueeze(2)

    L_sym = torch.eye(N, device=device, dtype=dtype).unsqueeze(0) - W_norm
    vals, vecs = torch.linalg.eigh(L_sym)
    fiedler = vecs[:, :, 1] * D_inv_sqrt_vec

    # Sign Fix
    skew = (fiedler ** 3).sum(dim=1, keepdim=True)
    sign_flip = torch.where(torch.sign(skew) == 0, torch.ones_like(skew), torch.sign(skew))
    fiedler = fiedler * sign_flip

    # Permutation
    perm = torch.argsort(fiedler, dim=1)
    x_ordered = x_norm.gather(1, perm.unsqueeze(-1).expand(-1, -1, D))

    # Rotation
    weights = torch.linspace(-1, 1, N, device=device, dtype=dtype).view(1, N, 1)
    u = F.normalize((x_ordered * weights).sum(dim=1), dim=1, eps=epsilon)
    cos_t, sin_t = u[:, 1:2], -u[:, 0:1]
    R1 = torch.stack([torch.cat([cos_t, -sin_t], dim=1),
                      torch.cat([sin_t, cos_t], dim=1)], dim=1)
    x_rot = torch.bmm(x_ordered, R1)

    # Reflection
    upper_centroid_x = (x_rot[..., 0:1] * (weights > 0).float()).sum(dim=1)
    ref_sign = torch.where(torch.sign(upper_centroid_x) == 0, torch.ones_like(upper_centroid_x),
                           torch.sign(upper_centroid_x)).view(B, 1, 1)
    R2 = torch.zeros(B, 2, 2, device=device, dtype=dtype)
    R2[:, 0, 0], R2[:, 1, 1] = ref_sign.squeeze(), 1.0

    R_total = torch.bmm(R1, R2)
    return torch.bmm(x_ordered, R_total), R_total, perm, scale


# ============================================================================
# 2. VISUALIZATION HELPERS
# ============================================================================
def add_indices(ax, coords, color='black', fontsize=7):
    """Helper to add text indices to scatter points."""
    x_range = coords[:, 0].max() - coords[:, 0].min()
    offset = x_range * 0.03
    N = coords.shape[0]
    for k in range(N):
        txt = ax.text(coords[k, 0] + offset, coords[k, 1] + offset, str(k),
                      fontsize=fontsize, fontweight='bold', color=color, zorder=20,
                      ha='left', va='bottom')
        txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='white', alpha=0.8)])


def add_tour(ax, coords, path):
    """Helper to draw the TSP path."""
    loop_p = np.append(path, path[0])
    ax.plot(coords[loop_p, 0], coords[loop_p, 1], 'k--', linewidth=0.7, alpha=0.4, zorder=1)


def visualize_side_by_side(x0, x1_coords, path, idx):
    """
    Strict Side-by-Side comparison: Input vs GT Circle.
    """
    N = x0.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Sample {idx}: Input vs Ground Truth", fontsize=14, fontweight='bold')

    # Left: Input
    axes[0].scatter(x0[:, 0], x0[:, 1], c=np.arange(N), cmap='hsv', s=100, edgecolors='k')
    add_tour(axes[0], x0, path)
    add_indices(axes[0], x0)
    axes[0].set_title("Input Point Cloud (X0)")
    axes[0].axis('equal')

    # Right: GT (Circle)
    axes[1].scatter(x1_coords[:, 0], x1_coords[:, 1], c=np.arange(N), cmap='hsv', s=100, edgecolors='k')
    add_tour(axes[1], x1_coords, path)
    add_indices(axes[1], x1_coords)
    axes[1].set_title("Ground Truth Angles\n(Projected to Unit Circle)")
    axes[1].axis('equal')
    axes[1].add_artist(plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--'))

    save_path = os.path.join(VIZ_DIR, f"side_by_side_{idx}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()


def visualize_rope_signals(x0, x1_coords, signals, path, idx):
    """
    Detailed RoPE Signal analysis.
    """
    N = x0.shape[0]

    # Unpack signals: [Radius, Sin, Cos, Hull]
    r_vals = signals[:, 0]
    sin_vals = signals[:, 1]
    cos_vals = signals[:, 2]
    hull_vals = signals[:, 3]
    hull_mask = hull_vals < 1e-4

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(f"RoPE Signal Analysis (Sample {idx})", fontsize=16, fontweight='bold')

    # 1. Radial
    sc1 = axes[0].scatter(x0[:, 0], x0[:, 1], c=r_vals, cmap='viridis', s=80, edgecolors='k')
    add_tour(axes[0], x0, path)
    add_indices(axes[0], x0)
    axes[0].set_title("Signal 1: Radius")
    axes[0].axis('equal')
    plt.colorbar(sc1, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Hull
    sc2 = axes[1].scatter(x0[:, 0], x0[:, 1], c=hull_vals, cmap='plasma_r', s=80, edgecolors='k')
    if hull_mask.any():
        axes[1].scatter(x0[hull_mask, 0], x0[hull_mask, 1], s=180, facecolors='none', edgecolors='cyan', linewidths=2.5)
    add_tour(axes[1], x0, path)
    add_indices(axes[1], x0)
    axes[1].set_title("Signal 4: Hull Depth")
    axes[1].axis('equal')
    plt.colorbar(sc2, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Angular Signals (Cos/Sin)
    true_theta = np.arctan2(x0[:, 1], x0[:, 0])
    sc3 = axes[2].scatter(cos_vals, sin_vals, c=true_theta, cmap='hsv', s=80, edgecolors='k')
    add_tour(axes[2], np.stack([cos_vals, sin_vals], axis=1), path)
    add_indices(axes[2], np.stack([cos_vals, sin_vals], axis=1))
    axes[2].set_title("Signal 2 & 3 (Cos/Sin)")
    axes[2].axis('equal')
    axes[2].add_artist(plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--'))

    # 4. GT Circle
    axes[3].scatter(x1_coords[:, 0], x1_coords[:, 1], c=np.arange(N), cmap='hsv', s=80, edgecolors='k')
    add_tour(axes[3], x1_coords, path)
    add_indices(axes[3], x1_coords)
    axes[3].set_title("GT Angles on Circle")
    axes[3].axis('equal')
    axes[3].add_artist(plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--'))

    save_path = os.path.join(VIZ_DIR, f"signals_{idx}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def generate_animation(x0, x1_coords, path, idx, frames=30):
    """Linear interpolation animation."""
    N = x0.shape[0]
    fig, ax = plt.subplots(figsize=(6, 6))

    all_pts = np.concatenate([x0, x1_coords], axis=0)
    ax.set_xlim(all_pts[:, 0].min() - 0.2, all_pts[:, 0].max() + 0.2)
    ax.set_ylim(all_pts[:, 1].min() - 0.2, all_pts[:, 1].max() + 0.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Shadow target
    ax.scatter(x1_coords[:, 0], x1_coords[:, 1], c='gray', marker='o', s=20, alpha=0.2)

    scat = ax.scatter(x0[:, 0], x0[:, 1], c=np.arange(N), cmap='hsv', s=60, edgecolors='k', zorder=5)
    lc = mc.LineCollection([], colors='black', linewidths=1.0, alpha=0.3)
    ax.add_collection(lc)

    def update(frame):
        t = frame / (frames - 1)
        xt = (1 - t) * x0 + t * x1_coords
        scat.set_offsets(xt)

        p = path.astype(int)
        loop_p = np.append(p, p[0])
        segs = [(xt[loop_p[i]], xt[loop_p[i + 1]]) for i in range(len(p))]
        lc.set_segments(segs)
        ax.set_title(f"Sample {idx} | t={t:.2f}")
        return scat, lc

    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True)
    save_path = os.path.join(VIZ_DIR, f"animation_{idx}.gif")
    ani.save(save_path, writer=animation.PillowWriter(fps=15))
    plt.close()


# ============================================================================
# 3. MAIN RIGOROUS TEST SUITE
# ============================================================================
def run_rigorous_tests(data_path, num_samples=5, num_viz=3, sigma_target=1.0):
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    print(f"Loading {data_path} via src.dataset.load_data...")

    # --- LOAD DATA VIA DATASET LOADER ---
    # We pass 'cpu' as device to keep it simple for numpy conversion
    # Note: load_data returns (x0, x1, paths, signals, precomputed)
    # x1 SHOULD be angles now.
    try:
        x0_all, x1_all, paths_all, signals_all, _ = load_data(data_path, 'cpu', interpolant=None)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    total_len = len(x0_all)
    indices = np.random.choice(total_len, min(num_samples, total_len), replace=False)

    print("\n" + "=" * 115)
    print(
        f"{'Idx':<5} | {'Type':<6} | {'SpecErr':<8} | {'CentErr':<8} | {'ScaleErr':<8} | {'AlignErr':<8} | {'HullOK':<6} | {'Result'}")
    print("=" * 115)

    for i, idx in enumerate(indices):
        # Extract sample
        x0 = x0_all[idx].double()  # Input (N, 2)
        x1 = x1_all[idx].double()  # Target (N) or (N, 1) or (N, 2) ?
        path = paths_all[idx].numpy()
        signals = signals_all[idx].numpy() if signals_all is not None else None

        N = x0.shape[0]

        # --- TEST 1: STRICT ANGLE CHECK ---
        # We expect x1 to be shape (N) or (N, 1). If it is (N, 2), conversion failed.
        is_angles = (x1.ndim == 1) or (x1.ndim == 2 and x1.shape[1] == 1)

        if not is_angles:
            print(f"❌ CRITICAL FAIL: Dataset loader returned Coordinates for x1, expected Angles. Shape: {x1.shape}")
            sys.exit(1)

        data_type_str = "ANGLE"

        # --- CONVERT TO COORDS FOR GEOMETRIC TESTS ---
        x1_angles = x1.squeeze()
        x1_coords = torch.stack([torch.cos(x1_angles), torch.sin(x1_angles)], dim=1)

        # --- TEST 2: Canonicalization (Input X0) ---
        centroid_err = torch.norm(x0.mean(dim=0)).item()

        dist_mat = torch.cdist(x0, x0, p=2)
        actual_scale = (dist_mat.sum() / (N * (N - 1))).item()
        scale_err = abs(actual_scale - sigma_target)

        # Spectral Reconstruction (Checks internal consistency)
        x_rec, _, _, _ = get_spectral_canonicalization(x0.unsqueeze(0).float(), sigma_kernel=sigma_target)
        x_rec = x_rec.squeeze(0).double()
        spec_err = torch.mean((x0 - x_rec) ** 2).item()

        # --- TEST 3: Alignment (Output X1 vs X0) ---
        base_dist = torch.mean(torch.norm(x0 - x1_coords, dim=1)).item()
        x1_np = x1_coords.numpy()
        x0_np = x0.numpy()

        # Optimal rotation to align Circle back to Point Cloud
        x1_realigned = Matrices.align_matrices(x1_np, x0_np)
        align_err = base_dist - np.mean(np.linalg.norm(x0_np - x1_realigned, axis=1))

        # --- TEST 4: Signals ---
        rope_status = "N/A"
        if signals is not None:
            # Hull is index 3
            hull_vals = signals[:, 3]
            if hull_vals.min() < 1e-2:
                rope_status = "OK"
            else:
                rope_status = "WEAK"

        # --- VERDICT ---
        # Allow small tolerance for alignment error (geometry drift)
        is_pass = (spec_err < 1e-3) and (centroid_err < 1e-4) and (scale_err < 1e-4) and (abs(align_err) < 1.5)
        status = "✅ PASS" if is_pass else "❌ FAIL"

        print(
            f"{idx:<5} | {data_type_str:<6} | {spec_err:<8.1e} | {centroid_err:<8.1e} | {scale_err:<8.1e} | {align_err:<8.1e} | {rope_status:<6} | {status}")

        # --- VISUALIZATION ---
        if i < num_viz:
            # 1. Side by Side
            visualize_side_by_side(x0.numpy(), x1_coords.numpy(), path, idx)
            # 2. Detailed Signals
            if signals is not None:
                visualize_rope_signals(x0.numpy(), x1_coords.numpy(), signals, path, idx)
            # 3. Animation
            generate_animation(x0.numpy(), x1_coords.numpy(), path, idx)

    print(f"\nVerification Complete. Visualizations saved to '{VIZ_DIR}/'.")


if __name__ == "__main__":
    # Point this to your actual .pt file
    # Adjust absolute path if necessary
    DATA_PATH = "../data/can_tsp50_rope_val.pt"

    run_rigorous_tests(DATA_PATH, num_samples=5, num_viz=3, sigma_target=1.0)