import argparse
import numpy as np
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from math import pi
import math  ### NEW: Added math import for sqrt/clamping logic


# ============================================================================
# Geometry Helpers
# ============================================================================

### NEW SECTION: Helper functions for Static RoPE signal calculation
def compute_approx_hull_depth(coords):
    """
    Computes 'Hull Depth' with dynamic resolution based on N.
    """
    B, N, _ = coords.shape
    device = coords.device

    # Dynamic resolution: sqrt(N), clamped [32, 128]
    num_directions = int(math.sqrt(N))
    num_directions = max(32, min(128, num_directions))

    angles = torch.linspace(0, 2 * math.pi, num_directions + 1, device=device)[:-1]
    directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

    # Project & Find Fence
    projections = coords @ directions.T
    max_vals = projections.amax(dim=1, keepdim=True)
    dists_to_boundary = max_vals - projections

    # Hull Depth = Min distance to ANY boundary wall
    hull_depth = dists_to_boundary.amin(dim=-1)
    return hull_depth


def compute_invariant_signals(x0):
    """
    Calculates the 4 raw invariant topological signals.
    Returns: (B, N, 4) tensor containing [R, Sin, Cos, Hull]
    """
    B, N, _ = x0.shape
    density_scale = math.sqrt(N)

    # 1. Center
    centroid = x0.mean(dim=1, keepdim=True)
    centered = x0 - centroid

    # 2. Compute Signals
    # A. Polar (Unscaled, Continuous)
    theta = torch.atan2(centered[..., 1], centered[..., 0])
    pos_sin = torch.sin(theta)
    pos_cos = torch.cos(theta)

    # B. Radial (Scaled)
    pos_r = torch.sqrt(centered[..., 0] ** 2 + centered[..., 1] ** 2) * density_scale

    # C. Hull Depth (Scaled)
    raw_depth = compute_approx_hull_depth(centered)
    pos_hull = raw_depth * density_scale

    # Stack: (B, N, 4)
    signals = torch.stack([pos_r, pos_sin, pos_cos, pos_hull], dim=-1)
    return signals


### END NEW SECTION

def get_signed_area_batched(points, paths):
    """
    Calculates signed area for a batch of tours to determine winding.
    Positive: CCW, Negative: CW.
    points: (B, N, 2), paths: (B, N)
    """
    B, N, _ = points.shape
    ordered = torch.gather(points, 1, paths.unsqueeze(-1).expand(-1, -1, 2))
    x = ordered[..., 0]
    y = ordered[..., 1]

    # Shoelace formula
    x_next = torch.roll(x, -1, dims=1)
    y_next = torch.roll(y, -1, dims=1)
    area = 0.5 * torch.sum(x * y_next - x_next * y, dim=1)
    return area


# ============================================================================
# Batched Preprocessing logic
# ============================================================================

### CHANGED: Added 'calc_rope' argument to function signature
def process_batch(points_batch, paths_batch, calc_rope=True, sigma_kernel=1.0, epsilon=1e-12, device="cuda"):
    B, N, D = points_batch.shape
    t_x0 = points_batch.to(device).to(torch.float64)
    t_paths = paths_batch.to(device)

    # --- STEP 1: ENFORCE CCW WINDING ---
    areas = get_signed_area_batched(t_x0, t_paths)
    cw_mask = areas < 0
    if cw_mask.any():
        t_paths[cw_mask] = torch.flip(t_paths[cw_mask], dims=[1])

    # 1. Centering
    t_x0 = t_x0 - t_x0.mean(dim=1, keepdim=True)

    # 2. Distance-Based Scale Normalization (Sets x0 scale)
    dist = torch.cdist(t_x0, t_x0, p=2)
    avg_dist = dist.sum(dim=(1, 2)) / (N * (N - 1))
    scale = (avg_dist / sigma_kernel).view(B, 1, 1) + epsilon
    t_x0_norm = t_x0 / scale

    # 3. Create Circle Targets
    p_ordered = torch.gather(t_x0_norm, 1, t_paths.unsqueeze(-1).expand(-1, -1, D))
    next_p = torch.roll(p_ordered, -1, dims=1)
    edge_lengths = torch.norm(p_ordered - next_p, dim=2)

    total_len = edge_lengths.sum(dim=1, keepdim=True)
    R = total_len / (2.0 * pi)
    angles = torch.cumsum(edge_lengths / (R + epsilon), dim=1) + (pi / 2.0)
    angles = torch.roll(angles, 1, dims=1)
    angles[:, 0] = pi / 2.0

    circle_raw = torch.stack([R * torch.cos(angles), R * torch.sin(angles)], dim=2)
    inv_paths = torch.argsort(t_paths, dim=1)
    t_x1_norm = torch.gather(circle_raw, 1, inv_paths.unsqueeze(-1).expand(-1, -1, D))

    # 4. Procrustes Alignment
    t_x1_centered = t_x1_norm - t_x1_norm.mean(dim=1, keepdim=True)
    H = torch.bmm(t_x1_centered.transpose(1, 2), t_x0_norm)
    U, S, V = torch.linalg.svd(H)
    R_proc = torch.bmm(U, V)
    d = torch.det(R_proc)
    V_mod = V.clone()
    V_mod[:, 1, :] *= d.view(B, 1)
    R_proc = torch.bmm(U, V_mod)
    t_x1_aligned = torch.bmm(t_x1_norm, R_proc)

    # --- MATCH FROBENIUS NORMS ---
    norm_x0 = torch.linalg.norm(t_x0_norm, ord='fro', dim=(1, 2), keepdim=True)
    norm_x1 = torch.linalg.norm(t_x1_aligned, ord='fro', dim=(1, 2), keepdim=True)
    t_x1_scaled = t_x1_aligned * (norm_x0 / (norm_x1 + epsilon))

    # 5. Spectral Ordering
    dist_sq = torch.cdist(t_x0_norm, t_x0_norm, p=2).pow(2)
    W = torch.exp(-dist_sq / (sigma_kernel ** 2))
    mask = torch.eye(N, device=device).bool().unsqueeze(0)
    W.masked_fill_(mask, 0)

    D_vec = W.sum(dim=2) + epsilon
    D_inv_sqrt = torch.rsqrt(D_vec)
    W_norm = W * D_inv_sqrt.unsqueeze(1) * D_inv_sqrt.unsqueeze(2)
    L_sym = torch.eye(N, device=device, dtype=torch.float64) - W_norm

    vals, vecs = torch.linalg.eigh(L_sym)
    fiedler = vecs[:, :, 1] * D_inv_sqrt

    skew = (fiedler ** 3).sum(dim=1, keepdim=True)
    sign_flip = torch.where(skew >= 0, torch.ones_like(skew), -torch.ones_like(skew))
    fiedler = fiedler * sign_flip

    perm = torch.argsort(fiedler, dim=1)
    x0_ord = torch.gather(t_x0_norm, 1, perm.unsqueeze(-1).expand(-1, -1, D))
    x1_ord = torch.gather(t_x1_scaled, 1, perm.unsqueeze(-1).expand(-1, -1, D))

    # 6. Final Canonical Rotation & Reflection
    weights = torch.linspace(-1, 1, N, device=device, dtype=torch.float64).view(1, N, 1)
    u = F.normalize((x0_ord * weights).sum(dim=1), dim=1, eps=epsilon)
    cos_t, sin_t = u[:, 1:2], -u[:, 0:1]

    Rot1 = torch.stack([torch.cat([cos_t, -sin_t], dim=1), torch.cat([sin_t, cos_t], dim=1)], dim=1)
    x0_rot = torch.bmm(x0_ord, Rot1)

    upper_mask = (weights > 0).to(torch.float64)
    upper_cx = (x0_rot[..., 0:1] * upper_mask).sum(dim=1)
    ref_sign = torch.where(upper_cx >= 0, torch.ones_like(upper_cx), -torch.ones_like(upper_cx)).squeeze()

    Rot2 = torch.zeros(B, 2, 2, device=device, dtype=torch.float64)
    Rot2[:, 0, 0] = ref_sign
    Rot2[:, 1, 1] = 1.0

    R_total = torch.bmm(Rot1, Rot2)
    x0_final = torch.bmm(x0_ord, R_total)
    x1_final = torch.bmm(x1_ord, R_total)

    inv_perm = torch.argsort(perm, dim=1)
    paths_canonical = torch.gather(inv_perm, 1, t_paths)

    ### NEW: COMPUTE STATIC SIGNALS (If requested)
    # We use x0_final because this is the canonical input state the model will see.
    static_signals = None
    if calc_rope:
        # We calculate (R, Sin, Cos, Hull) here. Independent of dimensions.
        static_signals = compute_invariant_signals(x0_final.float())

    ### CHANGED: Return signature now includes static_signals
    return x0_final.to(torch.float32).cpu(), \
        x1_final.to(torch.float32).cpu(), \
        perm.cpu(), \
        edge_lengths.to(torch.float32).cpu(), \
        paths_canonical.cpu(), \
        static_signals.cpu() if static_signals is not None else None


# ============================================================================
# Boilerplate Parsing and Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, default="../tsp1000_test_concorde.txt")
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_points', type=int, default=100)
    parser.add_argument('--out', type=str, default='can_tsp1000_test.pt')
    parser.add_argument('--no_rope', action='store_true', help="Disable calculation of static RoPE signals")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # --- Loading Data ---
    all_points, all_paths = [], []
    with open(args.infile, 'r') as f:
        for line in f:
            if " output " not in line: continue
            c_str, s_str = line.split(" output ")
            pts = np.fromstring(c_str, sep=' ').reshape(-1, 2)
            pth = np.fromstring(s_str, sep=' ', dtype=np.int64) - 1
            all_points.append(pts)
            all_paths.append(pth[:-1])

    p_tensor = torch.from_numpy(np.array(all_points))
    s_tensor = torch.from_numpy(np.array(all_paths))

    # --- Storage Lists (Store Batches, not Items) ---
    store_x0, store_x1, store_perm, store_edges, store_path, store_signals = [], [], [], [], [], []

    calc_rope = not args.no_rope
    print(f"Processing data... (Calc RoPE Signals: {calc_rope})")

    # --- Processing Loop ---
    for i in tqdm(range(0, len(p_tensor), args.batch_size)):
        b_p = p_tensor[i: i + args.batch_size]
        b_s = s_tensor[i: i + args.batch_size]
        if len(b_p) == 0: break

        x0, x1, perms, edges, paths_ccw, static_signals = process_batch(
            b_p, b_s, calc_rope=calc_rope, device=device
        )

        # Append WHOLE BATCHES to lists (CPU tensors)
        store_x0.append(x0)
        store_x1.append(x1)
        store_perm.append(perms)
        store_edges.append(edges)
        store_path.append(paths_ccw)
        if static_signals is not None:
            store_signals.append(static_signals)

    # --- Fast Collate & Save ---
    print("Collating tensors...")
    final_dict = {
        'points': torch.cat(store_x0, dim=0),
        'circle': torch.cat(store_x1, dim=0),
        'spectral_perm': torch.cat(store_perm, dim=0),
        'edge_lengths': torch.cat(store_edges, dim=0),
        'path': torch.cat(store_path, dim=0),
    }

    if len(store_signals) > 0:
        final_dict['static_signals'] = torch.cat(store_signals, dim=0)

    print(f"Saving to {args.out}...")
    torch.save(final_dict, args.out)
    print("Done.")


if __name__ == '__main__':
    main()