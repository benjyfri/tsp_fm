import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import sys
import os


# ==========================================
# 1. H-MDS & DECIPHERING LOGIC
# ==========================================

def compute_hmds(x0):
    """
    Computes Pure Hyperbolic MDS embedding.
    """
    x = x0[0].double()

    # 1. Pairwise Distance Matrix
    diff = x.unsqueeze(1) - x.unsqueeze(0)
    D = torch.norm(diff, dim=-1)

    # Scale if necessary to prevent cosh overflow/underflow
    max_dist = D.max()
    scale = 1.0
    if max_dist > 15.0:
        scale = 15.0 / max_dist
    elif max_dist < 1.0:
        scale = 5.0  # Stretch small clusters

    D_scaled = D * scale

    # 2. Pseudo-Gram Matrix
    Y = torch.cosh(D_scaled)

    # 3. Eigendecomposition of -Y
    vals, vecs = torch.linalg.eigh(-Y)

    # Top 2 positive eigenvalues
    vals_r = vals[-2:]
    vecs_r = vecs[:, -2:]

    # Clamp and Recover
    vals_r = torch.clamp(vals_r, min=1e-9)
    X_hyp = vecs_r @ torch.diag(torch.sqrt(vals_r))

    # 4. Project to Ball
    X_norm_sq = torch.sum(X_hyp ** 2, dim=-1, keepdim=True)
    denom = 1 + torch.sqrt(1 + X_norm_sq)
    x_ball = X_hyp / denom

    return x_ball.float().unsqueeze(0)


def decipher_tour_from_geometry(x_emb):
    """
    DECIPHERS the tour by sorting points angularly in the Poincaré disk.
    Args:
        x_emb: (1, N, 2) tensor in Poincaré disk
    Returns:
        tour_indices: (N,) tensor of indices sorting the tour
    """
    x = x_emb[0]  # (N, 2)

    # 1. Compute Polar Angle: atan2(y, x)
    # Range is [-pi, pi]
    angles = torch.atan2(x[:, 1], x[:, 0])

    # 2. Sort indices by angle
    sorted_indices = torch.argsort(angles)

    return sorted_indices


def align_procrustes(source, target):
    """Aligns source to target for cleaner visual comparison."""
    M = target.t() @ source
    U, S, V = torch.svd(M)
    R = U @ V.t()
    if torch.det(R) < 0:
        U[:, -1] *= -1
        R = U @ V.t()
    return source @ R.t()


# ==========================================
# 2. GEOMETRY BACKENDS
# ==========================================

class EuclideanBackend:
    def compute_trajectory(self, x0, x1, t):
        return (1 - t) * x0 + t * x1


class PoincareBackend:
    def __init__(self, eps=1e-5):
        self.eps = eps

    def mobius_add(self, x, y):
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        num = (1 + 2 * xy + y2) * x + (1 - x2) * y
        denom = 1 + 2 * xy + x2 * y2
        return num / (denom + self.eps)

    def log_map(self, x, y):
        mx = -x
        y_prime = self.mobius_add(mx, y)
        y_norm = torch.norm(y_prime, dim=-1, keepdim=True)
        scale = torch.atanh(torch.clamp(y_norm, max=1 - self.eps))
        lambda_x = 2.0 / (1 - torch.sum(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (2.0 / lambda_x) * (scale / (y_norm + self.eps)) * y_prime

    def exp_map(self, x, v):
        lambda_x = 2.0 / (1 - torch.sum(x ** 2, dim=-1, keepdim=True) + self.eps)
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        s = torch.tanh(lambda_x * v_norm / 2.0)
        u = (s / (v_norm + self.eps)) * v
        return self.mobius_add(x, u)

    def compute_trajectory(self, x0, x1, t):
        x0 = torch.clamp(x0, -0.999, 0.999)
        x1 = torch.clamp(x1, -0.999, 0.999)
        v = self.log_map(x0, x1)
        return self.exp_map(x0, t * v)


# ==========================================
# 3. DATA LOADING
# ==========================================

def generate_synthetic_data(N=20):
    # Generate points in a loop to ensure a "decipherable" structure exists
    theta = torch.linspace(0, 2 * np.pi, N + 1)[:-1]
    # Scramble them in space but keep topology
    r = 0.5 + 0.1 * torch.randn(N)
    x_circle = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)

    # Scramble order
    perm = torch.randperm(N)
    x0 = x_circle[perm].unsqueeze(0)

    # The "GT" path is the inverse of the permutation
    gt_path = torch.argsort(perm)

    return x0, gt_path


def load_data_and_decipher(path, index=0):
    """Loads data, computes H-MDS, and DECIPHERS the tour from H-MDS."""

    # 1. Load or Generate
    if not os.path.exists(path):
        print(f"File {path} not found. Using Synthetic Data.")
        x0, gt_path = generate_synthetic_data(N=25)
    else:
        try:
            data = torch.load(path, map_location='cpu')
            if isinstance(data, list):
                sample = data[index]
            else:
                sample = data

            x0_raw = sample.get('x0') if isinstance(sample, dict) else sample[0]
            gt_path = sample.get('tour') if isinstance(sample, dict) else sample[1]

            x0 = x0_raw.float()
            if x0.dim() == 2: x0 = x0.unsqueeze(0)
            gt_path = gt_path.long()

            # Center and Scale
            x0 = x0 - x0.mean(dim=1, keepdim=True)
            x0 = x0 / (x0.norm(dim=-1).max() * 2.2)

        except:
            x0, gt_path = generate_synthetic_data(N=25)

    # 2. Compute Target x1 using H-MDS
    print("Computing H-MDS Embedding...")
    x1_hyp = compute_hmds(x0)

    # 3. DECIPHER THE TOUR from x1_hyp
    print("Deciphering Tour from Hyperbolic Geometry...")
    deciphered_indices = decipher_tour_from_geometry(x1_hyp)

    # 4. Align for visualization
    # We rotate the H-MDS result to align visually with the input,
    # just so the animation looks like a flow, not a random rotation.
    x1_aligned = align_procrustes(x1_hyp[0], x0[0]).unsqueeze(0)

    return x0, x1_aligned, gt_path, deciphered_indices


def get_edges(points, tour_indices):
    segments = []
    pts = points[tour_indices]
    for i in range(len(pts) - 1):
        segments.append([pts[i], pts[i + 1]])
    segments.append([pts[-1], pts[0]])
    return segments


# ==========================================
# 4. ANIMATION
# ==========================================

def run_visualization():
    FILE_PATH = 'data/processed_data_geom_val.pt'
    SAMPLE_IDX = 0
    NUM_FRAMES = 100

    # Load everything
    x0, x1, gt_path, deciphered_path = load_data_and_decipher(FILE_PATH, SAMPLE_IDX)

    # Check Accuracy
    print(f"GT Path:   {gt_path[:10]}...")
    print(f"Deciphered:{deciphered_path[:10]}...")

    # Convert paths to numpy for plotting
    path_indices = [gt_path.cpu().numpy(), deciphered_path.cpu().numpy()]

    backends = [EuclideanBackend(), PoincareBackend()]
    titles = ["Original Flow (Using GT Tour)", "Hyperbolic Flow (Using Deciphered Tour)"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    artists = []

    for i, ax in enumerate(axes):
        ax.set_aspect('equal')
        ax.set_title(titles[i], fontsize=11, fontweight='bold')
        ax.axis('off')

        # Disk Boundary
        boundary = Circle((0, 0), 1.0, color='black', fill=False, linestyle='--', alpha=0.3)
        ax.add_patch(boundary)
        ax.set_xlim(-1.1, 1.1);
        ax.set_ylim(-1.1, 1.1)

        # Target Ghost
        x1_np = x1[0].numpy()
        ax.scatter(x1_np[:, 0], x1_np[:, 1], c='gray', alpha=0.1, marker='x')

        # Particles
        scat = ax.scatter([], [], c='#3498db', s=60, edgecolors='white', zorder=5)

        # Edges
        # Note: Left plot uses path_indices[0] (GT), Right plot uses path_indices[1] (Deciphered)
        color = '#27ae60' if i == 1 else '#e74c3c'  # Green for deciphered, Red for GT
        lc = LineCollection([], colors=color, linewidths=2.0, alpha=0.8, zorder=4)
        ax.add_collection(lc)

        artists.append({'scat': scat, 'lc': lc, 'path': path_indices[i]})

    def update(frame):
        t = frame / (NUM_FRAMES - 1)
        t_tensor = torch.tensor([t]).float().view(1, 1, 1)

        rets = []
        for i, backend in enumerate(backends):
            # Interpolate
            xt = backend.compute_trajectory(x0, x1, t_tensor)
            xt_np = xt[0].numpy()

            # Update Points
            artists[i]['scat'].set_offsets(xt_np)

            # Update Edges using the SPECIFIC path for that plot
            # (GT for Left, Deciphered for Right)
            segs = get_edges(xt_np, artists[i]['path'])
            artists[i]['lc'].set_segments(segs)

            rets.extend([artists[i]['scat'], artists[i]['lc']])
        return rets

    print("Generating Animation...")
    ani = animation.FuncAnimation(fig, update, frames=NUM_FRAMES, interval=40, blit=True)

    out_file = "hmds_deciphering_proof.gif"
    ani.save(out_file, writer=animation.PillowWriter(fps=25))
    print(f"Proof saved to {out_file}")
    plt.show()


if __name__ == "__main__":
    run_visualization()