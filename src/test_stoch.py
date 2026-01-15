import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import os
import sys

# --- 1. Import your existing class ---
from interpolants import KendallSFMInterpolant


# --- 2. Wrapper Class (Fixes missing geometry & adds helper) ---
class VisualizerKendallSFM(KendallSFMInterpolant):
    """
    Wraps your existing KendallSFMInterpolant to:
    1. Handle the missing 'geometry' argument (we pass None).
    2. Add the 'get_trajectory_at_t' helper method needed for animation.
    """

    def __init__(self, g=0.1):
        # Pass None for geometry since we use the robust math directly
        super().__init__(geometry=None, g=g)

    def get_trajectory_at_t(self, x0, x1, t_val):
        B = x0.shape[0]
        device = x0.device

        # Precompute direction
        v_x0_to_x1 = self._log_map(x0, x1)

        # 1. Time formatting
        t = torch.tensor([t_val], device=device).float().view(1, 1, 1)

        # 2. Compute Mean Position (Geodesic)
        tangent_vec_at_t = t * v_x0_to_x1
        mu_t = self._exp_map(x0, tangent_vec_at_t)

        # 3. Add Noise
        sigma_t = self.g * torch.sqrt(t * (1 - t))

        # Sample ambient Gaussian noise
        noise_ambient = torch.randn_like(mu_t)

        # Center noise
        noise_centered = noise_ambient - torch.mean(noise_ambient, dim=-2, keepdim=True)

        # Project noise to tangent space of mu_t
        proj_comp = torch.sum(noise_centered * mu_t, dim=[-2, -1], keepdim=True) * mu_t
        noise_tangent = noise_centered - proj_comp

        # Compute raw zt via exponential map
        zt_raw = self._exp_map(mu_t, sigma_t * noise_tangent)

        # Project zt back to manifold
        zt = self._project_to_manifold(zt_raw)

        return mu_t, zt


# --- 3. Data Loading & Utils ---

def load_real_sample(path, index=0, device='cpu'):
    """
    Robustly loads a sample from the .pt file.
    Handles Tuple/List/Dict formats and Numpy/Torch data types.
    """
    print(f"Loading data from {path}...")
    try:
        data = torch.load(path, map_location=device, weights_only=False)

        # --- DETECT FORMAT ---
        x0_raw = None
        gt_path_raw = None

        # Case A: List of 10000 individual samples
        if isinstance(data, list) and len(data) > 10:
            print(f"Detected list of {len(data)} samples.")
            sample = data[index]

            # Try common keys for points
            if isinstance(sample, dict):
                for key in ['x0', 'points', 'nodes', 'loc', 'graph']:
                    if key in sample:
                        x0_raw = sample[key]
                        break
                for key in ['gt_indices', 'tour', 'solution', 'path']:
                    if key in sample:
                        gt_path_raw = sample[key]
                        break
            elif isinstance(sample, (tuple, list)):
                x0_raw = sample[0]
                gt_path_raw = sample[1]

                # Case B: Tuple of batch tensors (e.g. length 3 or 4)
        elif isinstance(data, (tuple, list)) and len(data) <= 10:
            print("Detected batch tuple format.")
            x0_raw = data[0][index]
            gt_path_raw = data[2][index]

        # Case C: Dictionary of batches
        elif isinstance(data, dict):
            print("Detected batch dictionary format.")
            for key in ['x0', 'points', 'nodes']:
                if key in data:
                    x0_raw = data[key][index]
                    break
            for key in ['gt_indices', 'tour', 'tours']:
                if key in data:
                    gt_path_raw = data[key][index]
                    break

        if x0_raw is None:
            raise ValueError(f"Could not extract 'x0' from sample {index}")
        if gt_path_raw is None:
            raise ValueError(f"Could not extract 'tour' from sample {index}")

        # --- CONVERT NUMPY TO TORCH IF NEEDED ---
        if isinstance(x0_raw, np.ndarray):
            x0_raw = torch.from_numpy(x0_raw)
        if isinstance(gt_path_raw, np.ndarray):
            gt_path_raw = torch.from_numpy(gt_path_raw)

        # --- PREPARE TENSORS ---
        x0 = x0_raw.float()
        if x0.dim() == 2: x0 = x0.unsqueeze(0)  # (1, N, 2)

        gt_path = gt_path_raw.long()

        print(f"Loaded Sample {index}: N={x0.shape[1]} points")

    except Exception as e:
        print(f"CRITICAL ERROR loading file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- Construct Target x1 (Uniform Circle) ---
    N = x0.shape[1]

    # 1. Generate Canonical Circle
    R = 1.0 / np.sqrt(N)  # Radius for unit frobenius norm
    theta = torch.linspace(0, 2 * np.pi, N + 1)[:-1]  # (N,)
    circle_points = R * torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)  # (N, 2)

    # 2. Align Circle to x0 Indices using GT Path
    x1 = torch.zeros_like(x0[0])  # (N, 2)
    x1[gt_path] = circle_points
    x1 = x1.unsqueeze(0)  # (1, N, 2)

    # --- Pre-processing ---
    # Center and Norm x0
    x0_centered = x0 - x0.mean(dim=1, keepdim=True)
    x0_norm = x0_centered / x0_centered.norm(dim=(1, 2), keepdim=True)

    # Procrustes Align x1 to x0 for visual stability
    x1_aligned = align_procrustes(x1[0], x0_norm[0]).unsqueeze(0)

    return x0_norm, x1_aligned, gt_path


def align_procrustes(source, target):
    M = target.t() @ source
    U, S, V = torch.svd(M)
    R = U @ V.t()

    if torch.det(R) < 0:
        U[:, -1] *= -1
        R = U @ V.t()

    return source @ R.t()


def get_edges(points, tour_indices):
    segments = []
    pts = points[tour_indices]
    for i in range(len(pts) - 1):
        segments.append([pts[i], pts[i + 1]])
    segments.append([pts[-1], pts[0]])
    return segments


# --- 4. Visualization Core ---

def run_visualization():
    # --- CONFIG ---
    FILE_PATH = '/home/benjamin.fri/PycharmProjects/tsp_fm/data/processed_data_geom_val.pt'
    SAMPLE_IDX = 0
    SCALES = [0.01, 0.02, 0.03,0.04, 0.05]
    NUM_FRAMES = 100

    # 1. Load Data
    device = torch.device('cpu')
    x0, x1, gt_path = load_real_sample(FILE_PATH, SAMPLE_IDX, device)

    # 2. Setup Plot
    fig, axes = plt.subplots(1, len(SCALES), figsize=(5 * len(SCALES), 5))
    if len(SCALES) == 1: axes = [axes]

    interpolants = []
    artists = []

    gt_path_np = gt_path.cpu().numpy()

    # 3. Initialize Subplots
    for i, g in enumerate(SCALES):
        ax = axes[i]
        interp = VisualizerKendallSFM(g=g)
        interpolants.append(interp)

        # Style
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.3)
        ax.set_title(f"Noise Scale $g={g}$", fontsize=12, fontweight='bold')

        # Calculate limits based on data
        all_pts = torch.cat([x0, x1], dim=1).squeeze(0).numpy()
        margin = 0.15
        ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
        ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)

        # A. Static Elements
        x1_np = x1[0].numpy()
        ax.scatter(x1_np[:, 0], x1_np[:, 1], c='gray', marker='x', alpha=0.2, label='Target')

        # B. Geodesic (Green)
        scat_geo = ax.scatter([], [], c='#2ecc71', s=30, alpha=0.8, label='Geodesic', zorder=3)
        lc_geo = LineCollection([], colors='#2ecc71', linewidths=1.5, alpha=0.5)
        ax.add_collection(lc_geo)

        # C. Stochastic (Red)
        scat_stoch = ax.scatter([], [], c='#e74c3c', s=30, alpha=0.8, label='Stochastic', zorder=4)
        lc_stoch = LineCollection([], colors='#e74c3c', linewidths=1.5, alpha=0.5)
        ax.add_collection(lc_stoch)

        txt = ax.text(0.05, 0.95, '', transform=ax.transAxes, verticalalignment='top')

        if i == 0:
            ax.legend(loc='lower left', fontsize=8)

        artists.append({
            'scat_geo': scat_geo, 'lc_geo': lc_geo,
            'scat_stoch': scat_stoch, 'lc_stoch': lc_stoch,
            'text': txt
        })

    # 4. Update Function
    def update(frame):
        t = frame / (NUM_FRAMES - 1)
        t_safe = np.clip(t, 0.001, 0.999)

        rets = []

        for i, g in enumerate(SCALES):
            interp = interpolants[i]
            art = artists[i]

            mu_t, zt = interp.get_trajectory_at_t(x0, x1, t_safe)

            mu_np = mu_t[0].numpy()
            zt_np = zt[0].numpy()

            # Update Green
            art['scat_geo'].set_offsets(mu_np)
            segs_geo = get_edges(mu_np, gt_path_np)
            art['lc_geo'].set_segments(segs_geo)

            # Update Red
            art['scat_stoch'].set_offsets(zt_np)
            segs_stoch = get_edges(zt_np, gt_path_np)
            art['lc_stoch'].set_segments(segs_stoch)

            art['text'].set_text(f"t={t:.2f}")

            rets.extend([art['scat_geo'], art['lc_geo'], art['scat_stoch'], art['lc_stoch'], art['text']])

        return rets

    # 5. Save
    print("Generating Animation...")
    ani = animation.FuncAnimation(fig, update, frames=NUM_FRAMES, interval=50, blit=True)

    outfile = "sfm_noise_analysis.gif"
    ani.save(outfile, writer=animation.PillowWriter(fps=20))
    print(f"Comparison saved to {outfile}")
    plt.close()


if __name__ == "__main__":
    run_visualization()