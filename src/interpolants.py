import torch
import math

class BaseInterpolant:
    def sample(self, x0, x1, theta, device):
        """
        Returns:
            t: (B,) time
            xt: (B, N, 2) position at time t
            ut: (B, N, 2) target velocity field at time t
        """
        raise NotImplementedError

class KendallInterpolant(BaseInterpolant):
    def __init__(self, geometry):
        self.geometry = geometry

    def sample(self, x0, x1, theta, device):
        # Upcast for geometric stability
        x0_d = x0.double()
        x1_d = x1.double()
        B, N, _ = x0_d.shape

        t = torch.rand(B, device=device, dtype=torch.float64)

        # --- Geodesic Logic (from your original script) ---
        inner_prod = torch.sum(x0_d * x1_d, dim=[-2, -1])
        cos_theta = torch.clamp(inner_prod, -1.0 + 1e-7, 1.0 - 1e-7)

        # Use pre-calculated theta if reliable, or re-calc
        theta_geo = torch.acos(cos_theta)
        theta_view = theta_geo.view(B, 1, 1)
        sin_theta_view = torch.sin(theta_view)

        # Log map
        small_angle_mask = theta_view < 1e-6
        scale_factor = torch.where(small_angle_mask, torch.ones_like(theta_view), theta_view / sin_theta_view)
        log_x1_x0 = scale_factor * (x1_d - x0_d * cos_theta.view(B, 1, 1))

        # Geodesic (Exp map)
        t_view = t.view(B, 1, 1)
        xt = (x0_d * torch.cos(t_view * theta_view) + (log_x1_x0 / theta_view) * torch.sin(t_view * theta_view))
        xt = torch.where(small_angle_mask, x0_d, xt)

        # Parallel Transport (Velocity)
        ut = (-x0_d * theta_view * torch.sin(t_view * theta_view) + log_x1_x0 * torch.cos(t_view * theta_view))
        ut = torch.where(small_angle_mask, log_x1_x0, ut)

        return t.float(), xt.float(), ut.float()

class LinearInterpolant(BaseInterpolant):
    """Standard Euclidean Flow Matching (Optimal Transport path)."""
    def sample(self, x0, x1, theta, device):
        B = x0.shape[0]
        t = torch.rand(B, device=device, dtype=torch.float32)
        t_view = t.view(B, 1, 1)

        # Path: x_t = (1-t)x_0 + t x_1
        # Velocity: u_t = x_1 - x_0
        xt = (1 - t_view) * x0 + t_view * x1
        ut = x1 - x0

        return t, xt, ut

class AngleInterpolant(BaseInterpolant):
    """
    Interpolates strictly in angular domain.
    Assumes x0 and x1 are centered.
    Preserves radius of x0 linearly transforming to radius of x1 (or keeping R fixed).
    """
    def sample(self, x0, x1, theta, device):
        B, N, _ = x0.shape
        t = torch.rand(B, device=device, dtype=torch.float32)
        t_view = t.view(B, 1, 1)

        # Convert to Polar
        def to_polar(x):
            r = torch.norm(x, dim=-1, keepdim=True)
            # atan2 returns values in [-pi, pi]
            phi = torch.atan2(x[..., 1:], x[..., 0:1])
            return r, phi

        r0, phi0 = to_polar(x0)
        r1, phi1 = to_polar(x1)

        # Handle angular wrap-around (shortest path)
        diff = phi1 - phi0
        diff = (diff + math.pi) % (2 * math.pi) - math.pi

        # Interpolate R and Phi
        r_t = (1 - t_view) * r0 + t_view * r1
        phi_t = phi0 + t_view * diff

        # Convert back to Cartesian
        xt = torch.cat([r_t * torch.cos(phi_t), r_t * torch.sin(phi_t)], dim=-1)

        # Velocity calculation (chain rule)
        # dx/dt = d/dt(r_t * cos(phi_t)) = dr/dt * cos - r * sin * dphi/dt
        dr = r1 - r0
        dphi = diff

        u_x = dr * torch.cos(phi_t) - r_t * torch.sin(phi_t) * dphi
        u_y = dr * torch.sin(phi_t) + r_t * torch.cos(phi_t) * dphi
        ut = torch.cat([u_x, u_y], dim=-1)

        return t, xt, ut

def get_interpolant(name, geometry):
    if name == 'kendall': return KendallInterpolant(geometry)
    if name == 'linear': return LinearInterpolant()
    if name == 'angle': return AngleInterpolant()
    raise ValueError(f"Unknown interpolant: {name}")