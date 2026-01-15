import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import sys

# --- 1. Import your class ---
from interpolants import LinearSFMInterpolant


# --- 2. Wrapper Class ---
class VisualizerLinearSFM(LinearSFMInterpolant):
    """
    Wraps LinearSFMInterpolant to provide trajectory snapshots for animation.
    """

    def __init__(self, g=0.1):
        super().__init__(g=g)

    def get_trajectory_at_t(self, x0, x1, t_val):
        """
        Returns the Mean (Deterministic) position and the Noisy (Stochastic) position at time t.
        """
        device = x0.device
        B = x0.shape[0]

        # Time formatting
        t = torch.tensor([t_val], device=device).float().view(1, 1, 1)

        # 1. Deterministic Path (Mean)
        mu_t = (1 - t) * x0 + t * x1

        # 2. Add Brownian Bridge Noise
        sigma_t = self.g * torch.sqrt(t * (1 - t))
        noise = torch.randn_like(x0)

        zt = mu_t + sigma_t * noise

        return mu_t, zt


# --- 3. Data Loading & Utils (Same as your Kendall script) ---

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
    SAMPLE_IDX = 10
    # Increased scales slightly for Linear because it is visually subtle compared to Kendall
    SCALES = [0.001,0.002,0.003,0.004,0.005]
    NUM_FRAMES = 100

    # 1. Load Data
    x0, x1, gt_path = load_real_sample(FILE_PATH, SAMPLE_IDX)
    gt_path_np = gt_path.numpy()

    # 2. Setup Plot
    fig, axes = plt.subplots(1, len(SCALES), figsize=(5 * len(SCALES), 5))
    if len(SCALES) == 1: axes = [axes]

    interpolants = []
    artists = []

    for i, g in enumerate(SCALES):
        ax = axes[i]
        interp = VisualizerLinearSFM(g=g)
        interpolants.append(interp)

        ax.set_aspect('equal')
        ax.set_title(f"Linear Noise g={g}", fontsize=12, fontweight='bold')

        # Set limits
        all_pts = torch.cat([x0, x1], dim=1).squeeze(0).numpy()
        margin = 0.1
        ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
        ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)

        # Target phantom
        x1_np = x1[0].numpy()
        ax.scatter(x1_np[:, 0], x1_np[:, 1], c='gray', marker='x', alpha=0.2)

        # Plots
        scat_geo = ax.scatter([], [], c='#3498db', s=30, alpha=0.6, label='Deterministic', zorder=3)
        lc_geo = LineCollection([], colors='#3498db', linewidths=1.5, alpha=0.4)

        scat_stoch = ax.scatter([], [], c='#e74c3c', s=30, alpha=0.9, label='Stochastic', zorder=4)
        lc_stoch = LineCollection([], colors='#e74c3c', linewidths=1.5, alpha=0.6)

        ax.add_collection(lc_geo)
        ax.add_collection(lc_stoch)

        txt = ax.text(0.05, 0.95, '', transform=ax.transAxes, verticalalignment='top')

        if i == 0: ax.legend(loc='lower left', fontsize=8)

        artists.append(
            {'scat_geo': scat_geo, 'lc_geo': lc_geo, 'scat_stoch': scat_stoch, 'lc_stoch': lc_stoch, 'text': txt})

    def update(frame):
        t = frame / (NUM_FRAMES - 1)
        t_safe = np.clip(t, 0.001, 0.999)  # Avoid exact endpoints for clean math

        rets = []
        for i, g in enumerate(SCALES):
            interp = interpolants[i]
            art = artists[i]

            mu_t, zt = interp.get_trajectory_at_t(x0, x1, t_safe)
            mu_np = mu_t[0].numpy()
            zt_np = zt[0].numpy()

            art['scat_geo'].set_offsets(mu_np)
            art['lc_geo'].set_segments(get_edges(mu_np, gt_path_np))

            art['scat_stoch'].set_offsets(zt_np)
            art['lc_stoch'].set_segments(get_edges(zt_np, gt_path_np))

            art['text'].set_text(f"t={t:.2f}")
            rets.extend([art['scat_geo'], art['lc_geo'], art['scat_stoch'], art['lc_stoch'], art['text']])
        return rets

    print("Generating Linear SFM Animation...")
    ani = animation.FuncAnimation(fig, update, frames=NUM_FRAMES, interval=50, blit=True)
    ani.save("linear_sfm_noise.gif", writer=animation.PillowWriter(fps=20))
    print("Saved linear_sfm_noise.gif")
    plt.close()


if __name__ == "__main__":
    run_visualization()