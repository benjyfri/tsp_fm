import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import os
import time

# --- 1. Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from src.models import get_spectral_canonicalization

    print("Successfully imported get_spectral_canonicalization")
except ImportError as e:
    print(f"Error importing from src.models: {e}")
    sys.exit(1)


# --- 2. Forensic Analysis Helper ---
def analyze_stability_metrics(x, sigma=1.0, epsilon=1e-8):
    """
    Re-runs the math to extract the exact values used for decisions.
    """
    B, N, D = x.shape
    device = x.device

    # Center
    centroid = x.mean(dim=1, keepdim=True)
    x_centered = x - centroid

    # Laplacian
    dist_sq = torch.cdist(x_centered, x_centered, p=2) ** 2
    W = torch.exp(-dist_sq / (sigma ** 2))
    mask = torch.eye(N, device=device).unsqueeze(0).expand(B, N, N)
    W = W * (1 - mask)
    D_vec = W.sum(dim=2) + epsilon
    D_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(D_vec))
    I = torch.eye(N, device=device).unsqueeze(0).expand(B, N, N)
    L_sym = I - torch.bmm(torch.bmm(D_inv_sqrt, W), D_inv_sqrt)

    # Eigen
    vals, vecs = torch.linalg.eigh(L_sym)
    fiedler_sym = vecs[:, :, 1]
    fiedler = torch.bmm(D_inv_sqrt, fiedler_sym.unsqueeze(-1)).squeeze(-1)

    # Metric 1: Skewness (Sign Decision)
    skew = torch.sum(fiedler ** 3, dim=1).item()

    # Fix sign for rotation calc (Internal logic check)
    sign = np.sign(skew) if abs(skew) > 1e-9 else 1.0
    fiedler_fixed = fiedler * sign

    # Sort
    perm = torch.argsort(fiedler_fixed, dim=1)
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
    x_ordered = x_centered[batch_idx, perm]

    # Metric 2: Rotation Vector (u)
    weights = torch.linspace(-1, 1, N, device=device).view(1, N, 1)
    weighted_direction = torch.sum(x_ordered * weights, dim=1)  # (B, 2)
    u = weighted_direction / (torch.norm(weighted_direction, dim=1, keepdim=True) + epsilon)
    u_np = u[0].cpu().numpy()

    # Align for Reflection Calc
    cos_t, sin_t = u[:, 1:2], -u[:, 0:1]
    R1 = torch.stack([
        torch.cat([cos_t, -sin_t], dim=1),
        torch.cat([sin_t, cos_t], dim=1)
    ], dim=1)
    x_rot = torch.bmm(x_ordered, R1)

    # Metric 3: Reflection Score (Upper Half X-Centroid)
    upper_mask = (weights > 0).float()
    upper_centroid_x = torch.sum(x_rot[..., 0:1] * upper_mask, dim=1).item()

    return {
        'skew': skew,
        'rot_vec': u_np,
        'ref_score': upper_centroid_x,
        'fiedler_head': fiedler_fixed[0, :3].cpu().numpy(),
        'fiedler_tail': fiedler_fixed[0, -3:].cpu().numpy(),
        'eigenvalues': vals[0, :4].cpu().numpy()
    }


def plot_failure_case(ax_left, ax_right, x_c1, x_c2, mse, title_suffix=""):
    """
    Visualizes a failure case with diagnostic arrows.
    """

    # Helper to calculate diagnostic arrows on the *canonical* output
    def get_arrows(x_canon):
        N = x_canon.shape[0]
        x = torch.tensor(x_canon)
        weights = torch.linspace(-1, 1, N).view(N, 1)
        rot_vec = torch.sum(x * weights, dim=0).numpy()
        upper_mask = (weights > 0).float()
        ref_vec = torch.sum(x * upper_mask, dim=0).numpy()
        return rot_vec, ref_vec

    r1, ref1 = get_arrows(x_c1)
    r2, ref2 = get_arrows(x_c2)

    # Left: Canonical from Original

    ax_left.scatter(x_c1[:, 0], x_c1[:, 1], c='blue', s=40, alpha=0.6, label='Points')
    ax_left.arrow(0, 0, r1[0], r1[1], color='red', width=0.05, label='Rot Vec')
    ax_left.arrow(0, 0, ref1[0], ref1[1], color='green', width=0.03, label='Ref Vec')
    ax_left.set_title(f"Canonical (Orig)\n{title_suffix}")
    ax_left.grid(True, alpha=0.3)
    ax_left.set_aspect('equal')

    # Right: Canonical from Transformed
    ax_right.scatter(x_c2[:, 0], x_c2[:, 1], c='orange', s=40, alpha=0.6)
    ax_right.arrow(0, 0, r2[0], r2[1], color='red', width=0.05)
    ax_right.arrow(0, 0, ref2[0], ref2[1], color='green', width=0.03)
    ax_right.set_title(f"Canonical (Trans)\nMSE: {mse:.4f}")
    ax_right.grid(True, alpha=0.3)
    ax_right.set_aspect('equal')

    # Unify limits
    all_x = np.concatenate([x_c1[:, 0], x_c2[:, 0]])
    all_y = np.concatenate([x_c1[:, 1], x_c2[:, 1]])
    margin = 0.5
    lims = (min(all_x) - margin, max(all_x) + margin, min(all_y) - margin, max(all_y) + margin)
    ax_left.set_xlim(lims[0], lims[1]);
    ax_left.set_ylim(lims[2], lims[3])
    ax_right.set_xlim(lims[0], lims[1]);
    ax_right.set_ylim(lims[2], lims[3])


# --- 3. Main Stress Test ---
def stress_test(num_trials=100, N=50):
    print(f"\n{'=' * 80}")
    print(f"DEBUG STRESS TEST (N={N}) - Printing Numerical Instabilities")
    print(f"{'=' * 80}")

    stats = {
        'total': num_trials,
        'failed': 0,
        'max_mse': 0.0,
        'mse_list': []
    }

    failures_to_plot = []

    for i in range(num_trials):
        # 1. Generate & Transform
        x_orig = torch.randn(1, N, 2)
        x_orig = x_orig - x_orig.mean(dim=1, keepdim=True)

        perm_idx = torch.randperm(N)
        angle = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(angle), np.sin(angle)

        rot_mat = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)

        is_reflected = False
        if np.random.rand() > 0.5:
            is_reflected = True
            ref_mat = torch.tensor([[-1., 0.], [0., 1.]], dtype=torch.float32)
            M = ref_mat @ rot_mat
        else:
            M = rot_mat

        # Apply Transform
        x_trans = x_orig[:, perm_idx, :] @ M.t()

        # 2. Canonicalize
        with torch.no_grad():
            x_c1, _, _, _ = get_spectral_canonicalization(x_orig)
            x_c2, _, _, _ = get_spectral_canonicalization(x_trans)

        # 3. Check Error
        mse = torch.mean((x_c1 - x_c2) ** 2).item()
        stats['mse_list'].append(mse)
        if mse > stats['max_mse']: stats['max_mse'] = mse

        if mse > 1e-3:
            stats['failed'] += 1
            print(f"\n>>> ITERATION {i} FAILED | MSE: {mse:.5f}")
            print(f"    Transform applied: Rot={np.degrees(angle):.1f}°, Reflected={is_reflected}")

            # 4. GET THE NUMBERS
            m1 = analyze_stability_metrics(x_orig)
            m2 = analyze_stability_metrics(x_trans)

            print(f"\n    {'METRIC':<20} | {'ORIGINAL (C1)':<25} | {'TRANSFORMED (C2)':<25} | {'STATUS'}")
            print(f"    {'-' * 80}")

            # A. Check Eigenvalue Gap (Mode Mixing)
            gap1 = m1['eigenvalues'][2] - m1['eigenvalues'][1]
            gap2 = m2['eigenvalues'][2] - m2['eigenvalues'][1]
            status_gap = "STABLE" if gap1 > 1e-2 else "UNSTABLE (Tiny Gap!)"
            print(f"    {'Eigen Gap (v2-v1)':<20} | {gap1:<25.2e} | {gap2:<25.2e} | {status_gap}")

            # B. Check Skewness (Sign Flip)
            s1 = m1['skew']
            s2 = m2['skew']
            sign_match = np.sign(s1) == np.sign(s2)
            status_skew = "OK" if sign_match else "FLIPPED (Sign Ambiguity!)"
            print(f"    {'Skewness':<20} | {s1:<25.5f} | {s2:<25.5f} | {status_skew}")

            # C. Reflection Score consistency
            r1 = m1['ref_score']
            r2 = m2['ref_score']

            flip_consistent = np.sign(r1) == np.sign(r2)
            status_ref = "OK" if flip_consistent else "AMBIGUOUS (Near 0?)"
            print(f"    {'Refl Score':<20} | {r1:<25.5f} | {r2:<25.5f} | {status_ref}")

            print(f"\n    {'Fiedler Head (Low)':<20} | {m1['fiedler_head']} \n    {'':<20} | {m2['fiedler_head']}")
            print(f"    {'Fiedler Tail (High)':<20} | {m1['fiedler_tail']} \n    {'':<20} | {m2['fiedler_tail']}")

            # Logic Diagnosis
            diagnosis = "Unknown"
            if not sign_match:
                diagnosis = "Fiedler Sign Flip"
                print("\n    [DIAGNOSIS] -> Fiedler Sign Flip. Skewness was likely close to 0.")
            elif not flip_consistent:
                diagnosis = "Reflection Flip"
                print("\n    [DIAGNOSIS] -> Reflection Flip. Weighted X-centroid was likely close to 0.")
            elif status_gap.startswith("UNSTABLE"):
                diagnosis = "Mode Mixing"
                print("\n    [DIAGNOSIS] -> Mode Mixing. Eigenvalue gap is too small.")
            else:
                diagnosis = "Rotation Instability"
                print("\n    [DIAGNOSIS] -> Rotation Instability? (Weighted Direction might be unstable).")

            # Store for plotting (Up to 4)
            if len(failures_to_plot) < 4:
                failures_to_plot.append({
                    'x1': x_c1[0].cpu().numpy(),
                    'x2': x_c2[0].cpu().numpy(),
                    'mse': mse,
                    'diagnosis': diagnosis
                })

    # --- Summary Statistics ---
    avg_mse = np.mean(stats['mse_list'])
    pass_rate = 100 * (1 - stats['failed'] / stats['total'])

    print(f"\n{'=' * 80}")
    print(f"STRESS TEST SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total Trials: {stats['total']}")
    print(f"Failures:     {stats['failed']}")
    print(f"Pass Rate:    {pass_rate:.1f}%")
    print(f"Avg MSE:      {avg_mse:.2e}")
    print(f"Max MSE:      {stats['max_mse']:.2e}")  # Corrected string formatting

    # --- Visualization ---
    if len(failures_to_plot) > 0:
        print(f"\nGenerative visual report for {len(failures_to_plot)} failures...")
        fig, axes = plt.subplots(len(failures_to_plot), 2, figsize=(10, 4 * len(failures_to_plot)))
        if len(failures_to_plot) == 1: axes = [axes]  # Handle single case

        plt.suptitle("Canonicalization Failure Analysis", fontsize=16)

        for i, fail_data in enumerate(failures_to_plot):
            row_axes = axes[i] if len(failures_to_plot) > 1 else axes
            plot_failure_case(
                row_axes[0], row_axes[1],
                fail_data['x1'], fail_data['x2'],
                fail_data['mse'],
                title_suffix=f"Diag: {fail_data['diagnosis']}"
            )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig("stress_test_report.png")
        print("Report saved to 'stress_test_report.png'")
        plt.show()
    elif stats['failed'] == 0:
        print("\n[SUCCESS] No failures to visualize. System is stable.")


if __name__ == "__main__":
    stress_test(num_trials=100, N=100)