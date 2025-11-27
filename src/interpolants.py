# interpolants.py
import torch
import math

class BaseInterpolant:
    def precompute(self, x0, x1, theta):
        """
        Optional: Pre-compute expensive operations.
        Returns dict of tensors to be added to dataset, or None.
        """
        return None

    def sample(self, *args, device):
        """
        Sample interpolation.
        Args will be (x0, x1, theta, device) + any precomputed tensors.
        """
        raise NotImplementedError


class KendallInterpolant(BaseInterpolant):
    def __init__(self, geometry):
        self.geometry = geometry

    def precompute(self, x0, x1, theta):
        """Pre-compute geodesic components to avoid doing this every batch.

        IMPORTANT: keep results in double precision and return them as double.
        This avoids repeated float <-> double conversions in the hot training loop.
        """
        print("  Computing geodesics (one-time cost)...")

        # Work in double for numerical stability
        x0_d = x0.double()
        x1_d = x1.double()

        # inner product across points (B,)
        inner_prod = torch.sum(x0_d * x1_d, dim=[-2, -1])
        # clamp numerical noise
        cos_theta = torch.clamp(inner_prod, -1.0 + 1e-7, 1.0 - 1e-7)
        theta_geo = torch.acos(cos_theta)  # double

        # For stable division: compute sin(theta) view and mask small angles
        sin_theta_view = torch.sin(theta_geo.view(-1, 1, 1))
        small_angle_mask = (theta_geo < 1e-6)  # boolean mask, shape (B,)

        # scale_factor = theta / sin(theta) except for very small angles
        scale_factor = torch.where(
            small_angle_mask.view(-1, 1, 1),
            torch.ones_like(sin_theta_view),
            theta_geo.view(-1, 1, 1) / sin_theta_view
        )

        # log map in double
        log_x1_x0 = scale_factor * (x1_d - x0_d * cos_theta.view(-1, 1, 1))

        # Return double tensors (do NOT cast to float)
        return {
            'theta_geo': theta_geo.float(),
            'log_x1_x0': log_x1_x0.float(),
            'small_angle_mask': small_angle_mask, # Bool is fine
            'x0_fixed': x0.float()
        }

    def sample(self, x0, x1, theta, theta_geo, log_x1_x0, small_angle_mask, x0_fixed, device):
        """
        Pure Float32 sampling on GPU.
        No casting, no double precision.
        """
        B = x0.shape[0]

        # t is float32 by default
        t = torch.rand(B, device=device)

        t_view = t.view(B, 1, 1)
        theta_view = theta_geo.view(B, 1, 1)

        # All inputs here are already float32 because of precompute()
        angle = t_view * theta_view
        sin_angle = torch.sin(angle)
        cos_angle = torch.cos(angle)

        # Geodesic (Exp map)
        # Using a small epsilon in division just in case, though mask handles it
        xt = (x0_fixed * cos_angle +
              (log_x1_x0 / (theta_view + 1e-8)) * sin_angle)

        # Parallel Transport (Velocity)
        ut = (-x0_fixed * theta_view * sin_angle +
              log_x1_x0 * cos_angle)

        # Apply mask for small angles (Linear fallback)
        small_mask_view = small_angle_mask.view(B, 1, 1)
        if small_mask_view.any():
            xt = torch.where(small_mask_view, x0_fixed, xt)
            ut = torch.where(small_mask_view, log_x1_x0, ut)

        return t, xt, ut

class LinearInterpolant(BaseInterpolant):
    """
    Standard Euclidean Flow Matching.
    No pre-computation needed - already fast!
    """
    def sample(self, x0, x1, theta, device):
        B = x0.shape[0]
        t = torch.rand(B, device=device, dtype=torch.float32)
        t_view = t.view(B, 1, 1)

        xt = (1 - t_view) * x0 + t_view * x1
        ut = x1 - x0

        return t, xt, ut


class AngleInterpolant(BaseInterpolant):
    """
    Angular interpolation.
    Could pre-compute polar coordinates, but relatively fast as-is.
    """
    def precompute(self, x0, x1, theta):
        """Optional: pre-compute polar coordinates."""
        def to_polar(x):
            r = torch.norm(x, dim=-1, keepdim=True)
            phi = torch.atan2(x[..., 1:], x[..., 0:1])
            return r, phi

        r0, phi0 = to_polar(x0)
        r1, phi1 = to_polar(x1)

        # Pre-compute angular differences (shortest path)
        diff = phi1 - phi0
        diff = (diff + math.pi) % (2 * math.pi) - math.pi

        return {
            'r0': r0,
            'phi0': phi0,
            'r1': r1,
            'phi_diff': diff
        }

    def sample(self, x0, x1, theta, r0, phi0, r1, phi_diff, device):
        """Fast sampling with pre-computed polar coordinates."""
        B, N, _ = x0.shape
        t = torch.rand(B, device=device, dtype=torch.float32)
        t_view = t.view(B, 1, 1)

        # Interpolate using pre-computed values
        r_t = (1 - t_view) * r0 + t_view * r1
        phi_t = phi0 + t_view * phi_diff

        # Convert back to Cartesian
        xt = torch.cat([r_t * torch.cos(phi_t), r_t * torch.sin(phi_t)], dim=-1)

        # Velocity
        dr = r1 - r0
        u_x = dr * torch.cos(phi_t) - r_t * torch.sin(phi_t) * phi_diff
        u_y = dr * torch.sin(phi_t) + r_t * torch.cos(phi_t) * phi_diff
        ut = torch.cat([u_x, u_y], dim=-1)

        return t, xt, ut


def get_interpolant(name, geometry):
    if name == 'kendall': return KendallInterpolant(geometry)
    if name == 'linear': return LinearInterpolant()
    if name == 'angle': return AngleInterpolant()
    raise ValueError(f"Unknown interpolant: {name}")