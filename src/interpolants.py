# interpolants.py
import torch
import math

class BaseInterpolant:
    def precompute(self, x0, x1):
        """
        Optional: Pre-compute expensive operations.
        Returns dict of tensors to be added to dataset, or None.
        """
        return None

    def sample(self, *args, device):
        """
        Sample interpolation.
        Args will be (x0, x1, device) + any precomputed tensors unpacked.
        """
        raise NotImplementedError


class WITHGUARDKendallInterpolant(BaseInterpolant):
    def __init__(self, geometry):
        self.geometry = geometry

    def precompute(self, x0, x1):
        """Pre-compute geodesic components to avoid doing this every batch."""
        # Work in double for numerical stability during geometry calcs
        x0_d = x0.double()
        x1_d = x1.double()

        # inner product across points (B,)
        inner_prod = torch.sum(x0_d * x1_d, dim=[-2, -1])
        # clamp numerical noise
        cos_theta = torch.clamp(inner_prod, -1.0 + 1e-7, 1.0 - 1e-7)
        theta_geo = torch.acos(cos_theta)  # double

        # For stable division: compute sin(theta) view and mask small angles
        sin_theta_view = torch.sin(theta_geo.view(-1, 1, 1))
        small_angle_mask = (theta_geo < 1e-3)  # boolean mask, shape (B,)
        print(f'Max: {theta_geo.max()} , Min: {theta_geo.min()}')
        # scale_factor = theta / sin(theta) except for very small angles
        scale_factor = torch.where(
            small_angle_mask.view(-1, 1, 1),
            torch.ones_like(sin_theta_view),
            theta_geo.view(-1, 1, 1) / sin_theta_view
        )

        # log map in double
        # Log_x0(x1) formula: (theta / sin(theta)) * (x1 - x0 * cos(theta))
        log_x1_x0 = scale_factor * (x1_d - x0_d * cos_theta.view(-1, 1, 1))

        # Return strictly necessary tensors in float32
        return {
            'theta_geo': theta_geo.float(),
            'log_x1_x0': log_x1_x0.float(),
            'small_angle_mask': small_angle_mask,
        }

    def sample(self, x0, x1, theta_geo, log_x1_x0, small_angle_mask, device):
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
        # Formula: x_t = x0 * cos(t*theta) + (u / theta) * sin(t*theta)
        # Note: log_x1_x0 as calculated in precompute includes the (theta/sin(theta)) term,
        # so it effectively represents the tangent vector u.
        xt = (x0 * cos_angle +
              (log_x1_x0 / (theta_view + 1e-8)) * sin_angle)

        # Parallel Transport (Velocity / Vector Field)
        # u_t = -x0 * theta * sin(t*theta) + u * cos(t*theta)
        ut = (-x0 * theta_view * sin_angle +
              log_x1_x0 * cos_angle)

        # Apply mask for small angles (Linear/Static fallback)
        small_mask_view = small_angle_mask.view(B, 1, 1)
        if small_mask_view.any():
            # Standard Euclidean Interpolation for numerical stability at poles
            lin_xt = (1 - t_view) * x0 + t_view * x1
            lin_ut = x1 - x0

            xt = torch.where(small_mask_view, lin_xt, xt)
            ut = torch.where(small_mask_view, lin_ut, ut)

        return t, xt, ut

#Safe guards were never relevant (as Kendall dist was always big enough)
class KendallInterpolant(BaseInterpolant):
    def __init__(self, geometry):
        self.geometry = geometry

    def precompute(self, x0, x1):
        """Pre-compute geodesic components to avoid doing this every batch."""
        # Work in double for numerical stability during geometry calcs
        x0_d = x0.double()
        x1_d = x1.double()

        # inner product across points (B,)
        inner_prod = torch.sum(x0_d * x1_d, dim=[-2, -1])
        # clamp numerical noise
        cos_theta = torch.clamp(inner_prod, -1.0 + 1e-7, 1.0 - 1e-7)
        theta_geo = torch.acos(cos_theta)  # double

        # For stable division: compute sin(theta) view
        sin_theta_view = torch.sin(theta_geo.view(-1, 1, 1))

        # Direct computation of scale factor
        scale_factor = theta_geo.view(-1, 1, 1) / sin_theta_view

        # log map in double
        # Log_x0(x1) formula: (theta / sin(theta)) * (x1 - x0 * cos(theta))
        log_x1_x0 = scale_factor * (x1_d - x0_d * cos_theta.view(-1, 1, 1))

        # Return strictly necessary tensors in float32
        return {
            'theta_geo': theta_geo.float(),
            'log_x1_x0': log_x1_x0.float(),
        }

    def sample(self, x0, x1, theta_geo, log_x1_x0, device):
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
        # Formula: x_t = x0 * cos(t*theta) + (u / theta) * sin(t*theta)
        # Note: log_x1_x0 as calculated in precompute includes the (theta/sin(theta)) term,
        # so it effectively represents the tangent vector u.
        xt = (x0 * cos_angle +
              (log_x1_x0 / (theta_view)) * sin_angle)

        # Parallel Transport (Velocity / Vector Field)
        # u_t = -x0 * theta * sin(t*theta) + u * cos(t*theta)
        ut = (-x0 * theta_view * sin_angle +
              log_x1_x0 * cos_angle)

        return t, xt, ut


class LinearInterpolant(BaseInterpolant):
    """
    Standard Euclidean Flow Matching.
    No pre-computation needed.
    """
    def sample(self, x0, x1, device):
        B = x0.shape[0]
        t = torch.rand(B, device=device, dtype=torch.float32)
        t_view = t.view(B, 1, 1)

        xt = (1 - t_view) * x0 + t_view * x1
        ut = x1 - x0

        return t, xt, ut


class KendallSFMInterpolant(BaseInterpolant):
    """
    Stochastic Flow Matching on Kendall Shape Space (Hypersphere).

    Assumptions:
    1. x0 and x1 are ALREADY ALIGNED (centered, normalized, and rotated).
    2. Uses geomstats for stable Exp/Log maps.
    """

    def __init__(self, geometry, g=0.1):
        self.geometry = geometry
        self.g = g  # Stochasticity scale

    def precompute(self, x0, x1):
        """
        Calculate the tangent vector from x0 to x1 ONCE.
        This defines the 'mean' geodesic path.
        """
        # geomstats log: log(point, base_point) -> vector at base_point
        # We calculate the velocity vector at x0 pointing towards x1
        v_x0_to_x1 = self.geometry.metric.log(x1, x0)

        return {
            'v_x0_to_x1': v_x0_to_x1
        }

    def sample(self, x0, x1, v_x0_to_x1, device):
        """
        Sample z_t (noisy path) and u_t (conditional vector field).
        """
        B = x0.shape[0]

        # 1. Sample Time t
        # CRITICAL: Clamp t to avoid division by zero at t=1.0
        t = torch.rand(B, device=device).type_as(x0)
        t = torch.clamp(t, min=0.001, max=0.999)
        t_view = t.view(B, 1, 1)  # (B, 1, 1)

        # 2. Compute Mean Position mu_t (Deterministic Geodesic)
        # mu_t = Exp_x0(t * v)
        # We use the precomputed tangent vector v scaled by t
        tangent_vec_at_t = t_view * v_x0_to_x1
        mu_t = self.geometry.metric.exp(tangent_vec_at_t, x0)

        # 3. Add Noise (Stochastic Bridge)
        # sigma_t = g * sqrt(t * (1-t))
        sigma_t = self.g * torch.sqrt(t * (1 - t))
        sigma_t_view = sigma_t.view(B, 1, 1)

        # Sample ambient Gaussian noise
        noise_ambient = torch.randn_like(mu_t)

        # Project noise to the tangent space of mu_t
        # This ensures the noise moves us ALONG the sphere, not off it
        noise_tangent = self.geometry.space.to_tangent(noise_ambient, mu_t)

        # Compute noisy sample z_t
        # z_t = Exp_mu_t(sigma_t * noise_tangent)
        zt = self.geometry.metric.exp(sigma_t_view * noise_tangent, mu_t)

        # 4. Compute Conditional Vector Field u_t
        # The vector field should point from z_t towards x1.
        # u_t = Log_zt(x1) / (1 - t)

        # Vector at zt pointing to x1
        direction_to_target = self.geometry.metric.log(x1, zt)

        # Scale by time-to-go
        ut = direction_to_target / (1.0 - t_view)

        return t, zt, ut


def get_interpolant(name, geometry, stochasticity_scale=0.1):
    if name == 'kendall': return KendallInterpolant(geometry)
    if name == 'linear': return LinearInterpolant()
    if name == 'kendall_sfm': return KendallSFMInterpolant(geometry, g=stochasticity_scale)
    # if name == 'angle': return AngleInterpolant()
    raise ValueError(f"Unknown interpolant: {name}")