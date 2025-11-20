#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.decomposition import PCA
import torch

# --- GEOMSTATS SETUP ---
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
import geomstats.backend as gs
from geomstats.geometry.pre_shape import PreShapeSpace
from geomstats.geometry.matrices import Matrices

# ---------------------------
# 1. VISUAL CONFIGURATION
# ---------------------------
def configure_apple_style():
    """Configures Matplotlib for maximum screen usage and clean aesthetics."""
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    plt.rcParams['axes.linewidth'] = 0
    plt.rcParams['xtick.major.width'] = 0
    plt.rcParams['ytick.major.width'] = 0
    plt.rcParams['axes.grid'] = False # Clean background for max focus

# Apple Human Interface Colors
COLORS = {
    'bg': '#FFFFFF',
    'text': '#1D1D1F',
    'subtext': '#86868B',
    'primary': '#007AFF',
    'secondary': '#FF3B30',
    'accent': '#FF9500',
    'path': '#1D1D1F',
    'sphere': '#F2F2F7',
    'wireframe': '#D1D1D6',
}

# ---------------------------
# 2. MATH & GEOMETRY
# ---------------------------

def sample_geodesic_trajectory(x0, x1, n_steps=100):
    device = x0.device
    x0_d = x0.double(); x1_d = x1.double()
    t_seq = torch.linspace(0, 1, n_steps, dtype=torch.float64, device=device)

    inner_prod = torch.sum(x0_d * x1_d, dim=[-2, -1])
    cos_theta = torch.clamp(inner_prod, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)

    if theta < 1e-6:
        scale_factor = 1.0
        v_initial = torch.zeros_like(x0_d)
    else:
        scale_factor = theta / torch.sin(theta)
        v_initial = scale_factor * (x1_d - x0_d * cos_theta)

    xt_list = []; ut_list = []
    for t in t_seq:
        if theta < 1e-6:
            xt = x0_d
            ut = v_initial
        else:
            xt = (x0_d * torch.cos(t * theta) + (v_initial / theta) * torch.sin(t * theta))
            ut = (-x0_d * theta * torch.sin(t * theta) + v_initial * torch.cos(t * theta))
        xt_list.append(xt.cpu().numpy())
        ut_list.append(ut.cpu().numpy())

    return t_seq.cpu().numpy(), np.array(xt_list), np.array(ut_list)

def get_3d_projection(xt_seq, ut_seq):
    T, N, D = xt_seq.shape
    X_flat = xt_seq.reshape(T, -1)
    U_flat = ut_seq.reshape(T, -1)
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X_flat)
    U_3d = pca.transform(U_flat)
    norms = np.linalg.norm(X_3d, axis=1, keepdims=True)
    return X_3d / norms, U_3d, pca

def create_triangles():
    t1 = np.array([[0.0, 1.0], [-0.866, -0.5], [0.866, -0.5]])
    t2 = np.array([[0.0, 0.2], [-2.0, -0.2], [2.0, -0.2]])
    return torch.tensor(t1).float(), torch.tensor(t2).float()

def normalize_and_center(x):
    x = x - torch.mean(x, dim=0, keepdim=True)
    return x / torch.norm(x, p='fro')

# ---------------------------
# 3. PLOTTING HELPERS
# ---------------------------

def draw_solid_sphere(ax):
    u = np.linspace(0, 2 * np.pi, 80)
    v = np.linspace(0, np.pi, 40)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color=COLORS['sphere'], alpha=0.15,
                    shade=True, rcount=80, ccount=80, antialiased=True)

def draw_sphere_wireframe(ax):
    u = np.linspace(0, 2 * np.pi, 18)
    v = np.linspace(0, np.pi, 12)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color=COLORS['wireframe'], alpha=0.3, linewidth=0.5)

def setup_3d_axis(ax, title):
    # Reduced padding for title to save space
    ax.set_title(title.upper(), fontsize=10, fontweight='bold',
                 color=COLORS['subtext'], pad=5)

    # Tighter limits to remove whitespace around sphere
    limit = 1.1
    ax.set_xlim([-limit, limit]); ax.set_ylim([-limit, limit]); ax.set_zlim([-limit, limit])
    ax.set_box_aspect([1,1,1])
    ax.axis('off')
    ax.set_facecolor(COLORS['bg'])
    ax.view_init(elev=20, azim=40)

def style_2d_axis(ax, title):
    ax.set_title(title.upper(), fontsize=10, fontweight='bold',
                 color=COLORS['subtext'], pad=5)
    ax.set_aspect('equal')

    # Turn off everything for maximum clean look
    ax.axis('off')

    # Tighter limits
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)

# ---------------------------
# 4. MAIN EXECUTION
# ---------------------------

def main():
    configure_apple_style()

    space = PreShapeSpace(k_landmarks=3, ambient_dim=2, equip=True)

    raw_x0, raw_x1 = create_triangles()
    x0 = normalize_and_center(raw_x0)
    x1_unaligned = normalize_and_center(raw_x1)
    x1 = Matrices.align_matrices(x1_unaligned, x0)

    print("Computing Geodesic...")
    times, xt_seq, ut_seq = sample_geodesic_trajectory(x0, x1, n_steps=120)
    X_3d, U_3d, pca_model = get_3d_projection(xt_seq, ut_seq)

    # Changed to a square-ish aspect ratio for 2x2 grid
    fig = plt.figure(figsize=(12, 12), dpi=150, facecolor=COLORS['bg'])

    # --------------------------
    # PANEL 1 (Top Left): Boundary Conditions
    # --------------------------
    ax_static = fig.add_subplot(221)
    style_2d_axis(ax_static, "Boundary Conditions")

    s0 = x0.cpu().numpy(); poly0 = np.vstack([s0, s0[0]])
    s1 = x1.cpu().numpy(); poly1 = np.vstack([s1, s1[0]])

    ax_static.plot(poly0[:,0], poly0[:,1], '-', color=COLORS['primary'], lw=2.5, alpha=0.8)
    ax_static.fill(poly0[:,0], poly0[:,1], color=COLORS['primary'], alpha=0.1)

    ax_static.plot(poly1[:,0], poly1[:,1], '-', color=COLORS['secondary'], lw=2.5, alpha=0.8)
    ax_static.fill(poly1[:,0], poly1[:,1], color=COLORS['secondary'], alpha=0.1)

    # --------------------------
    # PANEL 2 (Top Right): Interpolation
    # --------------------------
    ax_morph = fig.add_subplot(222)
    style_2d_axis(ax_morph, "Interpolation")

    poly_morph, = ax_morph.plot([], [], color=COLORS['primary'], lw=3.0, zorder=3)
    poly_fill = ax_morph.fill([], [], color=COLORS['primary'], alpha=0.15)[0]

    ax_morph.plot(poly0[:,0], poly0[:,1], ':', color=COLORS['subtext'], lw=1.0, alpha=0.4)
    ax_morph.plot(poly1[:,0], poly1[:,1], ':', color=COLORS['subtext'], lw=1.0, alpha=0.4)

    # --------------------------
    # PANEL 3 (Bottom Left): Velocity Field
    # --------------------------
    ax_velocity = fig.add_subplot(223, projection='3d')
    setup_3d_axis(ax_velocity, "Tangent Velocity")
    draw_solid_sphere(ax_velocity)
    draw_sphere_wireframe(ax_velocity)

    ax_velocity.plot(X_3d[:,0], X_3d[:,1], X_3d[:,2], color=COLORS['path'],
                     linestyle='-', alpha=0.3, lw=1.5)

    ax_velocity.scatter(X_3d[0,0], X_3d[0,1], X_3d[0,2], color=COLORS['primary'], s=60, alpha=1.0, edgecolors='none')
    ax_velocity.scatter(X_3d[-1,0], X_3d[-1,1], X_3d[-1,2], color=COLORS['secondary'], s=60, alpha=1.0, edgecolors='none')

    pos_marker_vel, = ax_velocity.plot([], [], [], 'o', color='white', markersize=6,
                                       markeredgecolor=COLORS['accent'], markeredgewidth=2.0, zorder=10)
    velocity_arrow = None

    # --------------------------
    # PANEL 4 (Bottom Right): Transport
    # --------------------------
    ax_triangle = fig.add_subplot(224, projection='3d')
    setup_3d_axis(ax_triangle, "Manifold Transport")
    draw_solid_sphere(ax_triangle)
    draw_sphere_wireframe(ax_triangle)

    ax_triangle.plot(X_3d[:,0], X_3d[:,1], X_3d[:,2], color=COLORS['path'],
                     linestyle='-', alpha=0.3, lw=1.5)

    pos_marker_tri, = ax_triangle.plot([], [], [], 'o', color='white', markersize=5,
                                       markeredgecolor=COLORS['primary'], markeredgewidth=1.5, zorder=9)

    triangle_poly = Poly3DCollection([], alpha=0.5, facecolor=COLORS['primary'],
                                     edgecolor=COLORS['primary'], linewidths=0.5)
    ax_triangle.add_collection3d(triangle_poly)

    # --------------------------
    # UI ELEMENTS
    # --------------------------
    # Floating pill in the absolute center of the 2x2 grid
    time_text = fig.text(0.5, 0.5, '', fontsize=11, color=COLORS['text'],
                         ha='center', va='center', family='monospace', weight='bold',
                         bbox=dict(boxstyle='round,pad=0.6', fc='#F2F2F7', ec='none', alpha=0.9),
                         zorder=100)

    # --------------------------
    # ANIMATION LOOP
    # --------------------------
    def init():
        return (pos_marker_vel, pos_marker_tri, triangle_poly, poly_morph, poly_fill)

    def update(frame):
        nonlocal velocity_arrow
        x, y, z = X_3d[frame]
        u, v, w = U_3d[frame]

        # Update Velocity Marker
        pos_marker_vel.set_data([x], [y]); pos_marker_vel.set_3d_properties([z])

        # Update Quiver
        if velocity_arrow: velocity_arrow.remove()
        vec = np.array([u, v, w], dtype=float)
        mag = np.linalg.norm(vec)
        if mag > 1e-8:
            unit = vec / mag * 0.4
            velocity_arrow = ax_velocity.quiver(x, y, z, unit[0], unit[1], unit[2],
                                                color=COLORS['accent'], lw=2.0, arrow_length_ratio=0.3)

        # Update 3D Triangle
        pos_marker_tri.set_data([x], [y]); pos_marker_tri.set_3d_properties([z])

        curr_triangle = xt_seq[frame]
        normal = np.array([x, y, z])
        arb = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t1 = np.cross(normal, arb); t1 /= np.linalg.norm(t1)
        t2 = np.cross(normal, t1); t2 /= np.linalg.norm(t2)

        tri_verts_3d = []
        scale_tri = 0.35
        for i in range(3):
            pt = normal + scale_tri * (curr_triangle[i,0]*t1 + curr_triangle[i,1]*t2)
            tri_verts_3d.append(pt / np.linalg.norm(pt))

        triangle_poly.set_verts([tri_verts_3d])

        # Update 2D Morph
        px = np.append(curr_triangle[:,0], curr_triangle[0,0])
        py = np.append(curr_triangle[:,1], curr_triangle[0,1])
        poly_morph.set_data(px, py)
        poly_fill.set_xy(np.column_stack([px, py]))

        # Update UI
        t_val = times[frame]
        time_text.set_text(f" t = {t_val:.2f} ")

        return (pos_marker_vel, velocity_arrow, pos_marker_tri, triangle_poly, poly_morph, poly_fill, time_text)

    print("\n" + "="*50)
    print("   GENERATING MAXIMIZED LAYOUT ASSET")
    print("="*50)

    # Tight layout with minimal padding (rect=[left, bottom, right, top])
    plt.tight_layout(pad=0.5, rect=[0, 0, 1, 1])

    anim = FuncAnimation(fig, update, frames=len(times), init_func=init, blit=False, interval=40)

    save_name = "apple_style_maximized.gif"
    print(f"  Rendering High-DPI frames...")
    anim.save(save_name, writer='pillow', fps=30, dpi=150)
    print(f"  Done: {save_name}")
    print("="*50 + "\n")

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    main()