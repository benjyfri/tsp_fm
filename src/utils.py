import torch
import numpy as np


from torchdiffeq import odeint

def ode_solve_adaptive(model, x_start, geometry, method='dopri5', rtol=1e-5, atol=1e-7):
    """
    Adaptive ODE solver for flows on manifolds.

    Args:
        model: Flow model
        x_start: Initial points on manifold
        geometry: Manifold geometry object
        method: Integration method ('dopri5', 'dopri8', 'adaptive_heun', etc.)
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    def vector_field(t, x):
        # t is a scalar, expand to batch
        t_batch = t.repeat(x.shape[0]) if t.dim() == 0 else t.repeat(x.shape[0])
        v = model(x, t_batch, geometry=geometry)
        return v

    t_span = torch.tensor([0.0, 1.0], device=x_start.device)

    # odeint returns trajectory at t_span points
    x_traj = odeint(
        vector_field,
        x_start,
        t_span,
        method=method,
        rtol=rtol,
        atol=atol
    )

    # Project final point back to manifold
    return geometry.space.projection(x_traj[-1])

# In src/utils.py
def ode_solve_euler(model, x_start, geometry=None, steps=100):
    dt = 1.0 / steps
    x = x_start.clone()
    model.eval()
    times = torch.linspace(0, 1, steps+1, device=x.device)

    with torch.no_grad():
        for i in range(steps):
            t = times[i].repeat(x.shape[0])

            # Pass geometry to model for tangent projection
            v = model(x, t, geometry=geometry)
            x = x + v * dt

            # If on Kendall manifold, project back to sphere
            if geometry is not None:
                x = geometry.space.projection(x)
    return x

import torch

def ode_solve_rk4_exp(model, x_start, geometry, steps=50):
    """
    RK4 integrator for flows on the pre-shape sphere, using:
      - geometry.to_tangent(vector, base_point) to ensure tangent vectors
      - geometry.space.projection(...) to project back to the preshape sphere

    model(x, t, geometry=...) should accept x and a 1D tensor of times (len=batch).
    """
    # preserve input dtype, but do geometry ops in double for stability
    orig_dtype = x_start.dtype
    device = x_start.device

    dt = 1.0 / steps
    x = x_start.clone()
    model.eval()
    times = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=orig_dtype)

    with torch.no_grad():
        for i in range(steps):
            # current time scalar (as a 1D tensor matching batch dim)
            t_scalar = times[i].item()
            t = torch.full((x.shape[0],), t_scalar, device=device, dtype=orig_dtype)

            # Cast to double for geometry ops
            x_double = x.to(torch.float64)

            # k1
            v1 = model(x, t, geometry=geometry)
            v1 = geometry.to_tangent(v1, x)  # project to tangent at x
            k1 = v1

            # k2: evaluate at x + 0.5*dt*k1 (projected)
            x2 = geometry.space.projection((x_double + (k1.to(torch.float64) * (dt * 0.5))))
            t2 = torch.full((x.shape[0],), t_scalar + 0.5 * dt, device=device, dtype=orig_dtype)
            v2 = model(x2.to(orig_dtype), t2, geometry=geometry)
            v2 = geometry.to_tangent(v2, x2.to(orig_dtype))
            k2 = v2

            # k3
            x3 = geometry.space.projection((x_double + (k2.to(torch.float64) * (dt * 0.5))))
            t3 = torch.full((x.shape[0],), t_scalar + 0.5 * dt, device=device, dtype=orig_dtype)
            v3 = model(x3.to(orig_dtype), t3, geometry=geometry)
            v3 = geometry.to_tangent(v3, x3.to(orig_dtype))
            k3 = v3

            # k4
            x4 = geometry.space.projection((x_double + (k3.to(torch.float64) * dt)))
            t4 = torch.full((x.shape[0],), t_scalar + dt, device=device, dtype=orig_dtype)
            v4 = model(x4.to(orig_dtype), t4, geometry=geometry)
            v4 = geometry.to_tangent(v4, x4.to(orig_dtype))
            k4 = v4

            # combine increments (in original dtype), treat result as tangent at x
            increment = ((k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0) * dt

            # step in the ambient space then project to manifold (do projection in double)
            x = geometry.space.projection((x_double + increment.to(torch.float64))).to(orig_dtype)

    return x



def reconstruct_tour(final_points):
    """
    Given points that should form a circle (x1),
    reconstruct the tour based on angular ordering.
    """
    # Calculate angles relative to center (0,0)
    angles = torch.atan2(final_points[:, 1], final_points[:, 0])
    # Sort indices by angle
    tour_order = torch.argsort(angles)
    return tour_order

def calculate_tour_length(points, tour_indices):
    """Calculate TSP length given coordinates and order."""
    ordered_points = points[tour_indices]
    # Distance to next point
    next_points = torch.roll(ordered_points, -1, dims=0)
    dists = torch.norm(ordered_points - next_points, dim=1)
    return torch.sum(dists).item()