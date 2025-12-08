import torch
import matplotlib.pyplot as plt
import numpy as np
from models import  _get_canonical_rotation_optimized


# --- 2. Helper for Random Rotation ---
def get_random_rotation_matrix(batch_size):
    theta = torch.rand(batch_size) * 2 * torch.pi
    c, s = torch.cos(theta), torch.sin(theta)
    row1 = torch.stack([c, -s], dim=1)
    row2 = torch.stack([s, c], dim=1)
    return torch.stack([row1, row2], dim=1)

# --- 3. Data Generation & Checking ---
def run_check(num_points=50, seed=None):
    if seed:
        torch.manual_seed(seed)

    B, N, D = 1, num_points, 2 # Process 1 example at a time for clarity

    # A. Create Data
    x_original = torch.randn(B, N, D)

    # B. Create Rotated Version
    Q = get_random_rotation_matrix(B)
    x_rotated = torch.bmm(x_original, Q)

    # C. Create Permuted (Shuffled) Version
    perm = torch.randperm(N)
    x_shuffled = x_original[:, perm, :]

    # --- PROCESS ---
    # 1. Canonicalize Original
    R_orig = _get_canonical_rotation_optimized(x_original)
    out_orig = torch.bmm(x_original, R_orig)

    # 2. Canonicalize Rotated
    R_rot = _get_canonical_rotation_optimized(x_rotated)
    out_rot = torch.bmm(x_rotated, R_rot)

    # 3. Canonicalize Shuffled
    R_shuf = _get_canonical_rotation_optimized(x_shuffled)
    out_shuf = torch.bmm(x_shuffled, R_shuf)

    # Re-order shuffled output to match original indices for error checking
    inv_perm = torch.argsort(perm)
    out_shuf_restored = out_shuf[:, inv_perm, :]

    # Calculate Errors
    err_rot = (out_orig - out_rot).abs().max().item()
    err_perm = (out_orig - out_shuf_restored).abs().max().item()

    return {
        "x_orig": x_original[0],
        "x_rot": x_rotated[0],
        "out_orig": out_orig[0],
        "out_rot": out_rot[0],
        "err_rot": err_rot,
        "err_perm": err_perm
    }

# --- 4. Improved Visualization ---
def visualize_results(experiments):
    num_ex = len(experiments)
    fig, axes = plt.subplots(num_ex, 3, figsize=(18, 5 * num_ex))

    if num_ex == 1: axes = axes[None, :] # Handle single row case

    for i, data in enumerate(experiments):
        x_orig = data["x_orig"].detach().numpy()
        x_rot = data["x_rot"].detach().numpy()
        out_orig = data["out_orig"].detach().numpy()
        out_rot = data["out_rot"].detach().numpy()

        N = x_orig.shape[0]
        # Use a spectral map so index 0 is red, index N is blue/purple
        colors = plt.cm.nipy_spectral(np.linspace(0, 0.9, N))

        # --- Col 1: Input Space ---
        ax = axes[i, 0]
        ax.set_title(f"Example {i+1}: Input vs Rotated Input")
        # Draw lines connecting original to rotated to visualize the rigid transform
        # (Only drawing lines for first 10 points to avoid clutter, but coloring all)
        for p in range(N):
            ax.plot([x_orig[p,0], x_rot[p,0]], [x_orig[p,1], x_rot[p,1]],
                    color='gray', alpha=0.1, linewidth=0.5)

        ax.scatter(x_orig[:, 0], x_orig[:, 1], c=colors, s=80, label='Original', edgecolors='k', alpha=0.6)
        ax.scatter(x_rot[:, 0], x_rot[:, 1], c=colors, marker='^', s=80, label='Rotated', edgecolors='k', alpha=0.6)
        ax.legend(loc='upper right')
        ax.axis('equal')

        # --- Col 2: Canonical Space (Target View) ---
        ax = axes[i, 1]
        ax.set_title(f"Canonical Space (Rotation Invariant)\nMax Error: {data['err_rot']:.1e}")

        # Plot "Original" as large hollow circles
        ax.scatter(out_orig[:, 0], out_orig[:, 1], s=200, facecolors='none', edgecolors=colors,
                   linewidth=2, label='Canonical (from Orig)')

        # Plot "Rotated" as small solid dots (Should fit inside the circles)
        ax.scatter(out_rot[:, 0], out_rot[:, 1], c=colors, s=30, marker='o',
                   label='Canonical (from Rotated)')

        ax.legend(loc='upper right')
        ax.axis('equal')
        ax.grid(True, linestyle='--', alpha=0.5)

        # --- Col 3: Permutation Check (Residuals) ---
        ax = axes[i, 2]
        ax.set_title(f"Permutation Consistency\nMax Error: {data['err_perm']:.1e}")

        # Ideally this is just 0,0. We add noise to x-axis just to visualize count if they stack perfectly
        residuals = (out_orig - out_rot)
        ax.scatter(residuals[:, 0], residuals[:, 1], c=colors, alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlim(-1e-4, 1e-4)
        ax.set_ylim(-1e-4, 1e-4)
        ax.text(0, 0, "Perfect Match\n(Zero Error)", ha='center', va='center', fontweight='bold', alpha=0.3)
        ax.set_xlabel("Residual X")
        ax.set_ylabel("Residual Y")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Generate 3 distinct examples
    seeds = [42, 101, 999]
    results = [run_check(num_points=50, seed=s) for s in seeds]

    # Print Summary
    print(f"{'='*60}")
    print(f"{'Example':<10} | {'Rot Error':<15} | {'Perm Error':<15} | {'Status':<10}")
    print(f"{'-'*60}")
    for i, res in enumerate(results):
        status = "PASSED" if res['err_rot'] < 1e-5 and res['err_perm'] < 1e-5 else "FAILED"
        print(f"{i+1:<10} | {res['err_rot']:.2e}        | {res['err_perm']:.2e}        | {status}")
    print(f"{'='*60}")

    # Plot
    visualize_results(results)