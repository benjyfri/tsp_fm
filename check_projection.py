import torch
import numpy as np
import math
from math import pi, acos

# --- Import functions from your old scripts ---
# (I've renamed them to avoid file conflicts)
try:
    from data_old_scripts.create_dataset_TSP50 import pre_process_shape, find_optimal_transform
    from train_old import sample_geodesic_stable
except ImportError:
    print("ERROR: Please save your data_create.py as 'old_data_create.py'")
    print("and your train.py as 'old_train.py' in the same directory.")
    exit()

# ---
# ADDING THE "MISSING TOOL" FOR VERIFICATION
# We need these functions to *test* for horizontality.
# This logic is based on the kendall_pytorch.py class we reviewed.
# ---

def _skew_matrix_from_scalar(a: torch.Tensor) -> torch.Tensor:
    """Helper to create a batch of 2x2 skew-symmetric matrices."""
    B = a.shape[0]
    A = torch.zeros(B, 2, 2, device=a.device, dtype=a.dtype)
    A[:, 0, 1] = -a
    A[:, 1, 0] = a
    return A

def horizontal_projection_torch(X: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    (TESTING TOOL) Projects a batch of tangent vectors V at points X
    to their horizontal component.
    """
    if X.dim() == 2:
        X = X.unsqueeze(0)
        V = V.unsqueeze(0)

    B = X.shape[0]
    # Center V
    Vc = V - torch.mean(V, dim=-2, keepdim=True)

    # M = Vc.T @ X - X.T @ Vc
    M = torch.bmm(Vc.transpose(-2, -1), X) - torch.bmm(X.transpose(-2, -1), Vc)

    # G = X.T @ X
    G = torch.bmm(X.transpose(-2, -1), X)

    # Solve for a = -M[0,1] / tr(G)
    denom = G[:, 0, 0] + G[:, 1, 1] # tr(G) == ||X||_F^2 == 1.0
    safe_denom = torch.where(torch.abs(denom) > 1e-8, denom, torch.ones_like(denom))
    a = -M[:, 0, 1] / safe_denom

    # V_vert = X @ A
    A = _skew_matrix_from_scalar(a)
    V_vert = torch.bmm(X, A)

    V_h = Vc - V_vert
    return V_h.squeeze(0) if B == 1 else V_h

def vertical_component_torch(X: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    (TESTING TOOL) Returns the vertical (rotational) component of V at X.
    """
    # Center V first (as horizontal_projection does)
    Vc = V - torch.mean(V, dim=-2, keepdim=True)
    Vh = horizontal_projection_torch(X, Vc)
    return Vc - Vh

# ---
# MAIN TEST SCRIPT
# ---
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    N_POINTS = 50
    BATCH_SIZE = 4

    print(f"--- Kendall Geodesic Horizontality Test ---")
    print(f"Using device: {device}")
    print("Goal: Show that the velocity 'ut' from 'sample_geodesic_stable'")
    print("is 'horizontal' (has no rotational component) at its point 'xt'.\n")

    # 1. Create a BATCH of CORRECTLY processed and aligned data
    print(f"Generating {BATCH_SIZE} valid (x0, x1) pairs...")

    # Generate random numpy data
    x0_list_np = [pre_process_shape(np.random.rand(N_POINTS, 2)) for _ in range(BATCH_SIZE)]
    x1_list_np = [pre_process_shape(np.random.rand(N_POINTS, 2)) for _ in range(BATCH_SIZE)]

    # Align x1 to x0
    x1_aligned_list_np = [find_optimal_transform(x0_list_np[i], x1_list_np[i])[0]
                          for i in range(BATCH_SIZE)]

    # Convert to batched torch tensors
    x0 = torch.from_numpy(np.stack(x0_list_np)).to(device, dtype=torch.float32)
    x1 = torch.from_numpy(np.stack(x1_aligned_list_np)).to(device, dtype=torch.float32)

    # Calculate the true geodesic distance theta
    inner_prod = torch.sum(x0 * x1, dim=(-2, -1))
    theta = torch.acos(torch.clamp(inner_prod, -1.0, 1.0))

    print("Data generation complete.")
    print(f"  x0 shape: {x0.shape}")
    print(f"  x1 shape: {x1.shape}")
    print(f"  Avg. theta: {theta.mean().item():.4f}\n")

    # 2. Test at different time steps
    test_times = [0.0, 0.25, 0.5, 0.75, 1.0]

    for t_val in test_times:
        print(f"--- Testing at t = {t_val:.2f} ---")

        # Manually run the logic from sample_geodesic_stable
        # This lets us fix 't' instead of getting a random one.
        t = torch.full((BATCH_SIZE,), t_val, device=device, dtype=torch.float32)

        # --- Start of sample_geodesic_stable logic ---
        B = x0.shape[0]
        t_ = t.view(B, 1, 1)

        theta_ = theta.to(device=device, dtype=x0.dtype).view(B, 1, 1)
        eps = 1e-6
        theta_ = torch.clamp(theta_, min=eps, max=(math.pi - eps))

        a = (1 - t_) * theta_
        b = t_ * theta_

        sin_theta_safe = torch.clamp(torch.sin(theta_), min=eps)

        sa = torch.sin(a)
        sb = torch.sin(b)
        ca = torch.cos(a)
        cb = torch.cos(b)

        xt_geo = (sa / sin_theta_safe) * x0 + (sb / sin_theta_safe) * x1
        ut_geo = (theta_ / sin_theta_safe) * (cb * x1 - ca * x0)

        mask_near = (theta_ <= eps) | ((math.pi - theta_) <= eps)
        mask_near = mask_near.to(dtype=x0.dtype)

        xt_lin = (1 - t_) * x0 + t_ * x1
        ut_lin = x1 - x0

        xt = xt_geo * (1 - mask_near) + xt_lin * mask_near
        ut = ut_geo * (1 - mask_near) + ut_lin * mask_near
        # --- End of sample_geodesic_stable logic ---

        # 3. Perform the Horizontality Test
        # Get the vertical (rotational) component of ut at xt
        v_vert = vertical_component_torch(xt, ut)

        # Calculate its norm. If ut is horizontal, this norm should be ~0.
        norm_v_vert = torch.norm(v_vert.view(B, -1), p=2, dim=-1)

        print(f"  Avg. Norm of xt (should be ~1.0): {torch.norm(xt.view(B,-1), p=2, dim=-1).mean().item():.6f}")
        print(f"  Avg. Norm of Vertical Component (should be ~0): {norm_v_vert.mean().item():.2e}")

        # 4. (Optional) Sanity check velocity norm
        ut_norm = torch.norm(ut.view(B, -1), p=2, dim=-1)
        if t_val == 0.0:
            # At t=0, ||ut|| should be equal to theta
            print(f"  Avg. Norm of ut at t=0 (should == theta): {ut_norm.mean().item():.4f}")
            print(f"  Avg. Theta (for comparison):             {theta.mean().item():.4f}")
        else:
            print(f"  Avg. Norm of ut: {ut_norm.mean().item():.4f}")