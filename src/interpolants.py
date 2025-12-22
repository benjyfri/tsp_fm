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
    Implements robust, device-safe Exp/Log maps to avoid geomstats CUDA bugs.
    """

    def __init__(self, geometry, g=0.005):
        self.geometry = geometry
        self.g = g  # Stochasticity scale

    def _project_to_manifold(self, x):
        """
        Force x to be strictly on the PreShapeSpace manifold:
        1. Centered (mean = 0)
        2. Unit Norm (frobenius norm = 1)
        """
        # 1. Center
        x_centered = x - torch.mean(x, dim=-2, keepdim=True)

        # 2. Normalize
        norm = torch.norm(x_centered, dim=(-2, -1), keepdim=True)
        # Avoid division by zero
        x_proj = x_centered / (norm + 1e-8)
        return x_proj

    def _log_map(self, x, y):
        """
        Logarithm map on the Hypersphere: Log_x(y)
        Returns the tangent vector at x pointing towards y.
        """
        # Inner product (B, 1, 1)
        inner = torch.sum(x * y, dim=[-2, -1], keepdim=True)
        cos_theta = torch.clamp(inner, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta)

        # u = y - x * cos_theta (Projection of y onto tangent space of x)
        u_raw = y - x * cos_theta

        # Scale factor: theta / sin(theta)
        sin_theta = torch.sqrt(1.0 - cos_theta ** 2)

        scale = torch.where(
            theta < 1e-4,
            torch.ones_like(theta),
            theta / (sin_theta + 1e-8)
        )

        return scale * u_raw

    def _exp_map(self, x, v):
        """
        Exponential map on the Hypersphere: Exp_x(v)
        """
        v_norm = torch.norm(v, dim=[-2, -1], keepdim=True)

        cos_norm = torch.cos(v_norm)
        sin_norm = torch.sin(v_norm)

        sinc_norm = torch.where(
            v_norm < 1e-4,
            torch.ones_like(v_norm),
            sin_norm / (v_norm + 1e-8)
        )

        return x * cos_norm + v * sinc_norm

    def precompute(self, x0, x1):
        """
        Calculate the tangent vector from x0 to x1 ONCE.
        """
        v_x0_to_x1 = self._log_map(x0, x1)
        return {'v_x0_to_x1': v_x0_to_x1}

    def sample(self, x0, x1, v_x0_to_x1, device):
        """
        Sample z_t (noisy path) and u_t (conditional vector field).
        """
        B = x0.shape[0]

        # 1. Sample Time t (Clamped)
        t = torch.rand(B, device=device).type_as(x0)
        t = torch.clamp(t, min=0.001, max=0.999)
        t_view = t.view(B, 1, 1)

        # 2. Compute Mean Position mu_t (Deterministic Geodesic)
        tangent_vec_at_t = t_view * v_x0_to_x1
        mu_t = self._exp_map(x0, tangent_vec_at_t)

        # 3. Add Noise (Stochastic Bridge)
        sigma_t = self.g * torch.sqrt(t * (1 - t))
        sigma_t_view = sigma_t.view(B, 1, 1)

        # Sample ambient Gaussian noise
        noise_ambient = torch.randn_like(mu_t)

        # CRITICAL: Ensure noise is centered (remove translation component)
        noise_centered = noise_ambient - torch.mean(noise_ambient, dim=-2, keepdim=True)

        # Project centered noise to tangent space of mu_t (remove radial component)
        # v_proj = v - <v, mu_t> * mu_t
        proj_comp = torch.sum(noise_centered * mu_t, dim=[-2, -1], keepdim=True) * mu_t
        noise_tangent = noise_centered - proj_comp

        # Compute raw zt via exponential map
        zt_raw = self._exp_map(mu_t, sigma_t_view * noise_tangent)

        # CRITICAL FIX: Explicitly project zt back to manifold
        # This fixes numerical drift (norm != 1 or mean != 0) that causes geomstats to crash
        zt = self._project_to_manifold(zt_raw)

        # 4. Compute Conditional Vector Field u_t
        # u_t = Log_zt(x1) / (1 - t)

        direction_to_target = self._log_map(zt, x1)
        ut = direction_to_target / (1.0 - t_view)

        return t, zt, ut


class LinearSFMInterpolant(BaseInterpolant):
    """
    Stochastic Flow Matching (Brownian Bridge) in Euclidean Space.
    Formula:
      x_t = (1-t)x0 + t*x1 + g*sqrt(t(1-t))*noise
      u_t = (x1 - x_t) / (1-t)
    """

    def __init__(self, g=0.005):
        self.g = g

    def sample(self, x0, x1, device):
        B = x0.shape[0]
        # Clamp t to prevent division by zero at t=1
        t = torch.rand(B, device=device).type_as(x0)
        t = torch.clamp(t, min=0.001, max=0.999)
        t_view = t.view(B, 1, 1)

        # 1. Deterministic Path (Mean)
        mu_t = (1 - t_view) * x0 + t_view * x1

        # 2. Add Brownian Bridge Noise
        # sigma_t = g * sqrt(t * (1-t))
        sigma_t = self.g * torch.sqrt(t * (1 - t))
        sigma_t_view = sigma_t.view(B, 1, 1)

        noise = torch.randn_like(x0)
        xt = mu_t + sigma_t_view * noise

        # 3. Conditional Vector Field (Targeting x1)
        # u_t(x|x1) = (x1 - x) / (1 - t)
        ut = (x1 - xt) / (1 - t_view)

        return t, xt, ut

def get_interpolant(name, geometry, stochasticity_scale=0.005):
    if name == 'kendall': return KendallInterpolant(geometry)
    if name == 'linear': return LinearInterpolant()
    if name == 'kendall_sfm': return KendallSFMInterpolant(geometry, g=stochasticity_scale)
    if name == 'linear_sfm': return LinearSFMInterpolant(g=stochasticity_scale)
    raise ValueError(f"Unknown interpolant: {name}")