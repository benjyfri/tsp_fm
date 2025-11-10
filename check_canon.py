"""
Canonicalization clarifications + improved test script.

Key changes:
- Explicitly center each sampled cloud: subtract centroid BEFORE canonicalization.
- Optional Laplacian (spectral) canonicalization implementation (toggle with use_laplacian).
- Visualizes pre-centered (faint), centered, rotated (R1) and canonical (R2) for geometric method;
  also shows spectral canonical result side-by-side when enabled.
- Batch mode: multiple samples in one big figure.
"""

import math
from typing import Tuple, Dict, Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors  # for kNN graph used by Laplacian (pip install scikit-learn)

torch.set_printoptions(precision=4, sci_mode=False)


def center_cloud(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Center the cloud to zero mean.
    Args:
        x: (B, N, 2)
    Returns:
        x_centered: (B, N, 2)
        centroid: (B, 1, 2)
    """
    centroid = x.mean(dim=1, keepdim=True)  # (B,1,2)
    x_centered = x - centroid
    return x_centered, centroid


def geometric_canonicalize(x: torch.Tensor, epsilon: float = 1e-8
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    The original geometric canonicalization:
    - pick p_n (farthest point)
    - rotate so p_n -> (0, ||p_n||) (on +y axis)
    - pick p_x (closest to original p_n) and reflect if needed
    Returns:
        final_x (B,N,2), rotated_x (after R1), p_n (B,2), info dict
    """
    B, N, D = x.shape
    magnitudes = torch.norm(x, dim=2)  # (B,N)
    max_mag_indices = torch.argmax(magnitudes, dim=1)  # (B,)
    p_n = torch.gather(x, 1, max_mag_indices.view(B, 1, 1).expand(-1, -1, 2)).squeeze(1)  # (B,2)

    p_n_norm = torch.norm(p_n, dim=1, keepdim=True) + epsilon
    px_norm = p_n[:, 0:1] / p_n_norm
    py_norm = p_n[:, 1:2] / p_n_norm

    cos_alpha = py_norm
    sin_alpha = px_norm

    R1 = torch.zeros(B, 2, 2, device=x.device, dtype=x.dtype)
    R1[:, 0, 0] = cos_alpha.squeeze(-1)
    R1[:, 0, 1] = -sin_alpha.squeeze(-1)
    R1[:, 1, 0] = sin_alpha.squeeze(-1)
    R1[:, 1, 1] = cos_alpha.squeeze(-1)

    rotated_x = torch.bmm(x, R1)

    # find p_x using distances in original (not rotated) domain, then take its rotated coords
    dists = torch.norm(x - p_n.unsqueeze(1), dim=2)
    dists = dists.clone()
    dists.scatter_(1, max_mag_indices.unsqueeze(1), float('inf'))
    closest_indices = torch.argmin(dists, dim=1)
    p_x_rotated = torch.gather(rotated_x, 1, closest_indices.view(B, 1, 1).expand(-1, -1, 2)).squeeze(1)

    reflection_mask = torch.where(p_x_rotated[:, 0] < 0, -1.0, 1.0).to(x.dtype)
    R2 = torch.eye(2, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B, 1, 1)
    R2[:, 0, 0] = reflection_mask

    final_x = torch.bmm(rotated_x, R2)
    info = {
        "max_mag_indices": max_mag_indices,
        "closest_indices": closest_indices,
        "R1": R1,
        "R2": R2,
        "p_x_rotated": p_x_rotated
    }
    return final_x, rotated_x, p_n, info


def laplacian_spectral_canonicalize(x: torch.Tensor, k: int = 8, epsilon: float = 1e-8
                                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Simple spectral canonicalization using graph Laplacian:
    - build kNN graph (symmetric)
    - compute normalized Laplacian L_sym = I - D^{-1/2} W D^{-1/2}
    - get second smallest eigenvector (Fiedler vector) of L_sym (or principal eigenvector of adjacency)
    - use sign of projection on Fiedler to define an axis and rotate cloud to align that axis with +y
    - resolve reflection by a similar heuristic (use a point's x sign)
    NOTE: For small N this is fine; for large N you may want iterative eigensolver.
    """
    # Convert to numpy for sklearn convenience (single batch)
    assert x.shape[0] == 1, "This simple spectral function handles batch size 1"
    X = x[0].cpu().numpy()  # (N,2)
    N = X.shape[0]

    # build k-NN adjacency (symmetric)
    k = min(k, N - 1)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)  # indices includes the point itself at col 0
    W = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in indices[i, 1:]:
            W[i, j] = 1.0
            W[j, i] = 1.0  # symmetric

    # degree and normalized Laplacian
    d = np.maximum(W.sum(axis=1), 1e-12)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    L_sym = np.eye(N) - D_inv_sqrt @ W @ D_inv_sqrt

    # eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(L_sym)
    # eigvals sorted ascending; the second smallest is index 1 (if N>1)
    idx = 1 if N > 1 else 0
    fiedler = eigvecs[:, idx]  # (N,)

    # Use the Fiedler vector as a scalar embedding; compute its 1D "orientation" vector
    # For a rough axis direction, compute covariance-weighted direction:
    weights = np.abs(fiedler)
    weighted_mean = (weights[:, None] * X).sum(axis=0) / (weights.sum() + 1e-12)
    centered = X - weighted_mean
    # PCA on (weighted) centered points to get principal axis
    C = (weights[:, None] * centered).T @ centered
    vals, vecs = np.linalg.eigh(C)
    principal = vecs[:, np.argmax(vals)]  # 2-vector

    # principal points along some angle theta; we want to rotate so that this axis maps to +y
    theta = math.atan2(principal[1], principal[0])  # angle of principal axis
    alpha = math.pi / 2 - theta
    cos_a = math.cos(alpha)
    sin_a = math.sin(alpha)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=float)

    rotated = (X @ R.T)
    # reflection check: pick a point with large |fiedler| and reflect if its x<0
    idx_max = int(np.argmax(np.abs(fiedler)))
    if rotated[idx_max, 0] < 0:
        rotated[:, 0] *= -1.0

    # pack back to torch tensors (batch dim)
    rotated_t = torch.from_numpy(rotated.astype(np.float32)).unsqueeze(0)
    final_t = rotated_t.clone()  # in this simple method rotated==final after reflection fix
    info = {
        "fiedler_index": idx_max,
        "fiedler_values": fiedler,
        "R": R
    }
    return final_t, rotated_t, torch.from_numpy(weighted_mean.astype(np.float32)), info


# --- plotting & batch runner ---

def make_cloud(n_points: int, noise=0.02, seed: Optional[int] = None) -> torch.Tensor:
    """
    Make a non-centered ellipse-like cloud to illustrate centering.
    Returns shape (1,N,2)
    """
    if seed is not None:
        torch.manual_seed(seed)

    angles = torch.linspace(0, 2 * math.pi, n_points + 1)[:-1]
    radii = 0.3 + 0.9 * torch.rand(n_points)
    xs = (radii * torch.cos(angles)).unsqueeze(1)
    ys = (0.7 * radii * torch.sin(angles)).unsqueeze(1)
    cloud = torch.cat([xs, ys], dim=1)
    cloud += noise * torch.randn_like(cloud)

    # optionally shift cloud away from origin to show centering step is needed
    shift = torch.tensor([0.2, -0.15])
    cloud = cloud + shift  # <<< THIS CREATES A NON-CENTERED SAMPLE
    return cloud.unsqueeze(0)  # (1,N,2)


def plot_batch(num_samples=5, n_points=80, device='cpu', seeds=None, use_laplacian=False):
    if seeds is None:
        seeds = list(range(num_samples))
    assert len(seeds) == num_samples

    cmap = plt.get_cmap("tab20")
    point_colors = [cmap(i % 20) for i in range(n_points)]
    point_colors = np.array(point_colors)

    # Columns: Pre-centered (faint), After R1 (geom), Canonical (geom)  [optionally spectral extra column]
    ncols = 4 if use_laplacian else 3
    fig, axes = plt.subplots(num_samples, ncols, figsize=(4.5 * ncols, 3.2 * num_samples), squeeze=False)
    fig.suptitle(f"Centering + Canonicalization tests — {num_samples} samples, {n_points} points each", fontsize=16, y=0.94)

    for i_sample in range(num_samples):
        x_raw = make_cloud(n_points, seed=seeds[i_sample]).to(device)  # (1,N,2) **not centered**
        # 1) center — THIS IS WHERE WE FORCE mean 0,0
        x_centered, centroid = center_cloud(x_raw)  # centroid shape (1,1,2)

        # geometric canonicalization (works on centered cloud)
        final_geo, rotated_geo, p_n, info_geo = geometric_canonicalize(x_centered)

        # spectral canonicalization (if requested) - expects batch size 1
        if use_laplacian:
            final_spec, rotated_spec, spec_mean, info_spec = laplacian_spectral_canonicalize(x_centered, k=8)

        orig_np = x_raw[0].cpu().numpy()
        centered_np = x_centered[0].cpu().numpy()
        rotated_np = rotated_geo[0].cpu().numpy()
        canon_np = final_geo[0].cpu().numpy()

        # Column 0: original (not-centered), faint + centered on top
        ax0 = axes[i_sample, 0]
        ax0.set_title("Original (raw) + Centered")
        ax0.scatter(orig_np[:, 0], orig_np[:, 1], s=28, c='lightgray', edgecolor='none', alpha=0.75)
        ax0.scatter(centered_np[:, 0], centered_np[:, 1], s=36, c=point_colors, edgecolor='k', linewidth=0.2)
        ax0.scatter([centroid[0, 0, 0].item()], [centroid[0, 0, 1].item()], c='brown', marker='x', s=80, label='centroid')
        ax0.legend(fontsize=8)
        ax0.set_aspect('equal')
        ax0.grid(alpha=0.25)

        # Column 1: rotated (R1) from geometric method
        ax1 = axes[i_sample, 1]
        ax1.set_title("After R1 (geometric rotate)")
        ax1.scatter(rotated_np[:, 0], rotated_np[:, 1], s=36, c=point_colors, edgecolor='k', linewidth=0.2)
        # show where p_n landed: on +y at its radius
        p_n_radius = float(torch.norm(p_n[0]).cpu().numpy())
        ax1.scatter([0.0], [p_n_radius], marker='x', s=80, c='red', linewidths=2)
        ax1.set_aspect('equal')
        ax1.grid(alpha=0.25)

        # Column 2: canonical result (geometric)
        ax2 = axes[i_sample, 2]
        ax2.set_title("Canonical (geometric)")
        ax2.scatter(canon_np[:, 0], canon_np[:, 1], s=36, c=point_colors, edgecolor='k', linewidth=0.2)
        # arrows original(centered) -> canonical
        for j in range(n_points):
            ax2.arrow(centered_np[j, 0], centered_np[j, 1],
                      (canon_np[j, 0] - centered_np[j, 0]),
                      (canon_np[j, 1] - centered_np[j, 1]),
                      head_width=0.005, head_length=0.008, length_includes_head=True,
                      alpha=0.8, linewidth=0.5, color=point_colors[j])
        ax2.set_aspect('equal')
        ax2.grid(alpha=0.25)
        ax2.text(0.01, 0.98, f"p_n index={info_geo['max_mag_indices'].item()}, p_x index={info_geo['closest_indices'].item()}",
                 transform=ax2.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7), fontsize=8)

        # Column 3 (optional): spectral
        if use_laplacian:
            ax3 = axes[i_sample, 3]
            rotated_sp_np = rotated_spec[0].cpu().numpy()
            canon_sp_np = final_spec[0].cpu().numpy()
            ax3.set_title("Spectral (Laplacian) canonical")
            ax3.scatter(canon_sp_np[:, 0], canon_sp_np[:, 1], s=36, c=point_colors, edgecolor='k', linewidth=0.2)
            # arrows centered -> spectral canonical
            for j in range(n_points):
                ax3.arrow(centered_np[j, 0], centered_np[j, 1],
                          (canon_sp_np[j, 0] - centered_np[j, 0]),
                          (canon_sp_np[j, 1] - centered_np[j, 1]),
                          head_width=0.005, head_length=0.008, length_includes_head=True,
                          alpha=0.8, linewidth=0.5, color=point_colors[j])
            ax3.set_aspect('equal')
            ax3.grid(alpha=0.25)
            ax3.text(0.01, 0.98, f"fiedler idx={info_spec['fiedler_index']}", transform=ax3.transAxes,
                     verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7), fontsize=8)

        # invariance test (geometric) — rotate and reflect the centered input and re-canonicalize
        theta = 1.2345
        R = torch.tensor([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=x_centered.dtype).to(x_centered.device)
        rotated_input = (x_centered @ R)
        final_from_rot, _, _, _ = geometric_canonicalize(rotated_input)

        reflect_mat = torch.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=x_centered.dtype).to(x_centered.device)
        reflected_input = (x_centered @ reflect_mat)
        final_from_reflect, _, _, _ = geometric_canonicalize(reflected_input)

        diff_rot = float(torch.max(torch.abs(final_geo - final_from_rot)))
        diff_ref = float(torch.max(torch.abs(final_geo - final_from_reflect)))

        ax2.text(0.01, 0.02, f"Δrot={diff_rot:.1e}\nΔref={diff_ref:.1e}", transform=ax2.transAxes,
                 verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.8), fontsize=8)

        print(f"Sample {i_sample:02d}: centroid={centroid.cpu().numpy().ravel()}, "
              f"geom Δrot={diff_rot:.2e}, Δref={diff_ref:.2e}")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    NUM_SAMPLES = 5
    N_POINTS = 120  # more points as you asked
    SEEDS = [10, 23, 42, 99, 123]
    plot_batch(num_samples=NUM_SAMPLES, n_points=N_POINTS, seeds=SEEDS, use_laplacian=True)
