import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# --- 1. Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from src.models import SpectralCanonTransformer, get_spectral_canonicalization

    print("Successfully imported Spectral Models")
except ImportError as e:
    print(f"Error importing models: {e}")
    sys.exit(1)


# --- 2. Mock Configuration ---
class Config:
    def __init__(self, num_points=50):
        self.num_points = num_points
        self.embed_dim = 128
        self.num_heads = 4
        self.num_layers = 3
        self.dropout = 0.0


# --- 3. Geometric Statistics ---
def get_cloud_stats(x):
    dist_matrix = torch.cdist(x, x)
    diameter = torch.max(dist_matrix).item()
    dist_matrix.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))
    avg_nn_dist = torch.mean(torch.min(dist_matrix, dim=-1)[0]).item()
    return diameter, avg_nn_dist


# --- 4. Main Robustness Test ---
def run_robustness_suite(N=100):
    dtype = torch.float32
    torch.set_default_dtype(dtype)

    # We will test different noise levels while also applying massive scale changes
    noise_levels = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2]
    config = Config(num_points=N)
    model = SpectralCanonTransformer(config).eval().to(dtype)

    # Baseline context at unit scale
    sample_x = torch.randn(1, N, 2, dtype=dtype)
    base_diameter, base_nn_dist = get_cloud_stats(sample_x)

    print(f"\n{'=' * 115}")
    print(f"STRESS TEST: SCALE + ROTATION + REFLECTION + NOISE (N={N})")
    print(f"Unit Scale Context: Diameter: {base_diameter:.3f} | d_NN: {base_nn_dist:.3f}")
    print(f"{'=' * 115}")
    print(
        f"{'Noise Std':<10} | {'Global Scale':<12} | {'Rel. Noise':<12} | {'Avg MSE':<10} | {'Rel. MSE':<10} | {'Status'}")
    print("-" * 115)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    plt.suptitle(f"Spectral Model: Scale & Rotation Invariance Stress Test (N={N})\n"
                 f"Blue: Unit Scale Canonical | Orange: Scaled/Noisy/Rotated Canonical", fontsize=16)

    for i, sigma in enumerate(noise_levels):
        # 1. Generate Original Data (Unit Scale)
        x_orig = torch.randn(1, N, 2, dtype=dtype)
        x_orig = x_orig - x_orig.mean(dim=1, keepdim=True)
        t = torch.tensor([0.5], dtype=dtype)

        # 2. Random Global Scale (Log-uniform from 0.5 to 50.0)
        global_scale = 10 ** np.random.uniform(np.log10(0.5), np.log10(50.0))

        # 3. Create Rotation/Reflection Matrix
        angle = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(angle), np.sin(angle)
        rot_mat = torch.tensor([[c, -s], [s, c]], dtype=dtype)
        if np.random.rand() > 0.5:
            M = torch.tensor([[-1., 0.], [0., 1.]], dtype=dtype) @ rot_mat
        else:
            M = rot_mat

        perm = torch.randperm(N)

        # 4. Apply Transform + Scale + Noise
        # noise is scaled by the global_scale to keep relative noise consistent
        noise = torch.randn(1, N, 2, dtype=dtype) * (sigma * global_scale)
        x_trans = ((x_orig[:, perm, :] @ M.t()) * global_scale) + noise

        # RE-CENTER (Ensures translation invariance doesn't interfere with scale/rot check)
        x_trans = x_trans - x_trans.mean(dim=1, keepdim=True)

        # 5. Model Forward Pass
        with torch.no_grad():
            v_orig = model(x_orig, t)
            v_trans = model(x_trans, t)

            # Internal Canonical States for Viz
            x_canon_orig, _, _, _ = get_spectral_canonicalization(x_orig)
            x_canon_trans, _, _, _ = get_spectral_canonicalization(x_trans)

        # 6. Global Frame Equivariance Check
        # To compare v_trans to v_orig:
        # 1. Multiply v_orig by global_scale (Ground truth velocity should be larger)
        # 2. Or: Divide v_trans by global_scale (Bring it back to unit magnitude)
        v_final = torch.zeros_like(v_trans)
        v_final[:, perm, :] = (v_trans / global_scale) @ M  # <--- SCALE REVERSED HERE

        mse = torch.mean((v_orig - v_final) ** 2).item()
        rel_noise_pct = sigma / base_nn_dist

        # MSE is relative to the unit scale coordinates
        rel_mse = mse / (base_diameter ** 2)
        status = "STABLE" if mse < 1e-3 else "FLIPPED"

        print(
            f"{sigma:<10.1e} | {global_scale:<12.2f} | {rel_noise_pct:<12.2%} | {mse:<10.2e} | {rel_mse:<10.2e} | {status}")

        # 7. Plotting
        ax = axes[i]
        pts_orig = x_canon_orig[0].cpu().numpy()
        pts_trans = x_canon_trans[0].cpu().numpy()

        ax.scatter(pts_orig[:, 0], pts_orig[:, 1], alpha=0.5, label='Unit Scale', s=25, color='#1f77b4')
        ax.scatter(pts_trans[:, 0], pts_trans[:, 1], alpha=0.5, label='Scaled/Noisy', s=25, color='#ff7f0e')

        # Connections to check if spectral ordering is identical
        for j in range(0, N, max(1, N // 15)):
            ax.plot([pts_orig[j, 0], pts_trans[j, 0]], [pts_orig[j, 1], pts_trans[j, 1]], 'k-', alpha=0.1)

        ax.set_title(f"Scale: {global_scale:.1f}x | Noise: {rel_noise_pct:.1%}\nMSE: {mse:.2e}")
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.3)
        if i == 0: ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("scale_robustness_report.png", dpi=200)
    print(f"\n[FINISH] Report saved to: scale_robustness_report.png")
    plt.show()


if __name__ == "__main__":
    run_robustness_suite(N=100)