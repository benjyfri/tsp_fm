import torch
import numpy as np

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