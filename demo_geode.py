#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import torch

# --- GEOMSTATS SETUP ---
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
import geomstats.backend as gs
from geomstats.geometry.pre_shape import PreShapeSpace
from geomstats.geometry.matrices import Matrices

# ============================================================================
# 1. Geometry Logic (Robust)
# ============================================================================

def sample_geodesic_trajectory(x0, x1, n_steps=100):
    """Calculates the Geodesic path and Tangent Velocity."""
    device = x0.device
    x0_d = x0.double()
    x1_d = x1.double()

    t_seq = torch.linspace(0, 1, n_steps, dtype=torch.float64, device=device)

    # Log Map
    inner_prod = torch.sum(x0_d * x1_d, dim=[-2, -1])
    cos_theta = torch.clamp(inner_prod, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)

    if theta < 1e-6:
        scale_factor = 1.0
    else:
        scale_factor = theta / torch.sin(theta)

    v_initial = scale_factor * (x1_d - x0_d * cos_theta)

    xt_list = []
    ut_list = []

    for t in t_seq:
        if theta < 1e-6:
            xt = x0_d
            ut = v_initial
        else:
            # Geodesic & Parallel Transport
            xt = (x0_d * torch.cos(t * theta) + (v_initial / theta) * torch.sin(t * theta))
            ut = (-x0_d * theta * torch.sin(t * theta) + v_initial * torch.cos(t * theta))

        xt_list.append(xt.cpu().numpy())
        ut_list.append(ut.cpu().numpy())

    return t_seq.cpu().numpy(), np.array(xt_list), np.array(ut_list)


def get_3d_projection(xt_seq, ut_seq):
    """Projects 6D trajectory onto 3D Sphere using PCA."""
    T, N, D = xt_seq.shape
    X_flat = xt_seq.reshape(T, -1)
    U_flat = ut_seq.reshape(T, -1)

    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X_flat)
    # Project velocities into same basis
    U_3d = pca.transform(U_flat)

    # Normalize for perfect visual sphere
    norms = np.linalg.norm(X_3d, axis=1, keepdims=True)
    X_3d = X_3d / norms
    return X_3d, U_3d

# ============================================================================
# 2. Data & Setup
# ============================================================================

def create_triangles():
    # T1: Equilateral
    t1 = np.array([[0.0, 1.0], [-0.866, -0.5], [0.866, -0.5]])
    # T2: Flattened & Shifted
    t2 = np.array([[0.0, 0.2], [-2.0, -0.2], [2.0, -0.2]])
    return torch.tensor(t1).float(), torch.tensor(t2).float()

def normalize_and_center(x):
    x = x - torch.mean(x, dim=0, keepdim=True)
    return x / torch.norm(x, p='fro')

# ============================================================================
# 3. Main Dashboard
# ============================================================================

def main():
    space = PreShapeSpace(k_landmarks=3, ambient_dim=2, equip=True)

    # Data Prep
    raw_x0, raw_x1 = create_triangles()
    x0 = normalize_and_center(raw_x0)
    x1_unaligned = normalize_and_center(raw_x1)

    # Alignment
    x1 = Matrices.align_matrices(x1_unaligned, x0)

    # Compute Flow
    print("Computing Geodesic...")
    times, xt_seq, ut_seq = sample_geodesic_trajectory(x0, x1, n_steps=100)
    X_3d, U_3d = get_3d_projection(xt_seq, ut_seq)

    # --- FIGURE SETUP ---
    # Use a wide layout: [Static | Sphere | Dynamic]
    fig = plt.figure(figsize=(18, 6), dpi=100)
    plt.style.use('ggplot') # Better aesthetics

    # --- PANEL 1: STATIC COMPARISON ---
    ax_static = fig.add_subplot(131)
    ax_static.set_title("Static Reference (Aligned)", fontsize=12, fontweight='bold')
    ax_static.set_aspect('equal')
    ax_static.grid(True, alpha=0.3)

    # Plot Start (Green)
    s0 = x0.cpu().numpy()
    poly0 = np.vstack([s0, s0[0]])
    ax_static.plot(poly0[:,0], poly0[:,1], 'o-', color='#2ecc71', lw=2, label='Start ($x_0$)')
    ax_static.fill(poly0[:,0], poly0[:,1], color='#2ecc71', alpha=0.1)

    # Plot End (Red)
    s1 = x1.cpu().numpy()
    poly1 = np.vstack([s1, s1[0]])
    ax_static.plot(poly1[:,0], poly1[:,1], 'o-', color='#e74c3c', lw=2, label='Target ($x_1$)')
    ax_static.fill(poly1[:,0], poly1[:,1], color='#e74c3c', alpha=0.1)

    ax_static.legend(loc='upper right')
    ax_static.set_xlim(-0.8, 0.8)
    ax_static.set_ylim(-0.8, 0.8)

    # --- PANEL 2: MANIFOLD SPHERE ---
    ax_sphere = fig.add_subplot(132, projection='3d')
    ax_sphere.set_title("Kendall Shape Space ($S^2$)", fontsize=12, fontweight='bold')

    # Draw nice wireframe sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax_sphere.plot_wireframe(x, y, z, color="gray", alpha=0.1, linewidth=0.5)

    # Plot Trajectory
    ax_sphere.plot(X_3d[:,0], X_3d[:,1], X_3d[:,2], color='black', linestyle='--', alpha=0.5, lw=1)
    ax_sphere.scatter(X_3d[0,0], X_3d[0,1], X_3d[0,2], color='#2ecc71', s=80, label='Start')
    ax_sphere.scatter(X_3d[-1,0], X_3d[-1,1], X_3d[-1,2], color='#e74c3c', s=80, label='Target')

    # Dynamic Sphere Elements
    sphere_dot, = ax_sphere.plot([], [], [], 'bo', markersize=10, mec='white', mew=1.5)

    # We will use a Quiver for the 3D arrow, but we must update it carefully
    # Initialize with dummy data
    q_3d = ax_sphere.quiver(0,0,0, 0,0,0, color='#3498db', lw=3, arrow_length_ratio=0.3)

    ax_sphere.set_box_aspect([1,1,1]) # Crucial for spherical look
    ax_sphere.axis('off')

    # --- PANEL 3: DYNAMIC MORPH ---
    ax_morph = fig.add_subplot(133)
    ax_morph.set_title("Geodesic Flow (Real-Time)", fontsize=12, fontweight='bold')
    ax_morph.set_aspect('equal')
    ax_morph.grid(True, alpha=0.3)
    ax_morph.set_xlim(-0.8, 0.8)
    ax_morph.set_ylim(-0.8, 0.8)

    poly_morph, = ax_morph.plot([], [], color='#3498db', lw=3)
    nodes_morph, = ax_morph.plot([], [], 'ko', markersize=6)

    # --- FIX IS HERE: INITIALIZE QUIVER WITH DATA, NOT EMPTY ---
    # Matplotlib locks the number of arrows at init. We must provide N=3 arrows.
    q_morph = ax_morph.quiver(
        xt_seq[0,:,0], xt_seq[0,:,1],
        ut_seq[0,:,0], ut_seq[0,:,1],
        color='#e67e22',
        scale=1,
        units='xy',
        width=0.02,       # Thicker shaft
        headwidth=4,      # Wider head
        headlength=5,     # Longer head
        label='Velocity ($u_t$)'
    )
    # -----------------------------------------------------------

    ax_morph.legend(loc='lower right')

    # --- UPDATE FUNCTION ---
    def init():
        sphere_dot.set_data([], [])
        sphere_dot.set_3d_properties([])
        poly_morph.set_data([], [])
        nodes_morph.set_data([], [])
        # Do NOT reset q_morph to empty here!
        return sphere_dot, poly_morph, nodes_morph, q_morph

    def update(frame):
        # 1. SPHERE UPDATE
        x, y, z = X_3d[frame]
        u, v, w = U_3d[frame]

        sphere_dot.set_data([x], [y])
        sphere_dot.set_3d_properties([z])

        # Update 3D Arrow (Remove old, create new)
        # This is the only robust way to animate arrows in Matplotlib 3D
        nonlocal q_3d
        q_3d.remove()
        # Scale arrow for visibility (0.5x magnitude)
        scale_3d = 0.5
        q_3d = ax_sphere.quiver(x, y, z, u*scale_3d, v*scale_3d, w*scale_3d,
                                color='#3498db', lw=2.5, arrow_length_ratio=0.2)

        # 2. MORPH UPDATE
        curr_x = xt_seq[frame]
        curr_u = ut_seq[frame]

        # Polygon
        px = np.append(curr_x[:,0], curr_x[0,0])
        py = np.append(curr_x[:,1], curr_x[0,1])
        poly_morph.set_data(px, py)
        nodes_morph.set_data(curr_x[:,0], curr_x[:,1])

        # Quiver (Velocity)
        q_morph.set_offsets(curr_x)
        q_morph.set_UVC(curr_u[:,0], curr_u[:,1])

        # Update Title with Norm info
        # Norm should be roughly constant in flow matching along geodesic
        vel_norm = np.linalg.norm(curr_u)
        ax_morph.set_xlabel(f"Time: {times[frame]:.2f} | ||u_t||: {vel_norm:.2f}")

        return sphere_dot, poly_morph, nodes_morph, q_morph, q_3d

    print("Generating Animation (this may take a moment)...")
    anim = FuncAnimation(fig, update, frames=len(times), init_func=init, blit=False, interval=50)

    save_name = "kendall_dashboard.gif"
    anim.save(save_name, writer='pillow', fps=20)
    print(f"Dashboard saved to: {save_name}")

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    main()