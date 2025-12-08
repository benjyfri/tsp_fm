# kendall_pytorch.py
import torch
from typing import Tuple, Optional


class KendallShapeSpace:
    """
    Pure PyTorch Kendall shape space (planar shapes, dim=2).
    - Shapes are expected in pre-shape form: centered and unit-Frobenius norm.
      Many methods call `to_coords` internally when shapes may not be pre-shape.
    - Batch semantics: inputs may be (N,2) or (B,N,2). Outputs preserve batching.
    """

    def __init__(self, n_points: int, dim: int = 2, device: Optional[str] = None):
        assert dim == 2, "This implementation is specialized to 2D planar shapes."
        self.n_points = n_points
        self.dim = dim
        self.manifold_dim = n_points * dim - dim - dim * (dim - 1) // 2 - 1
        # device is only a default; methods prefer input.device
        self.default_device = device

    # ----------------------
    # Utilities / preprocessing
    # ----------------------
    def _ensure_batch(self, X: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Return (X_batch, squeezed) where squeezed indicates original had no batch dim."""
        if X.dim() == 2:
            return X.unsqueeze(0), True
        return X, False

    def _restore_squeeze(self, X: torch.Tensor, squeezed: bool) -> torch.Tensor:
        return X.squeeze(0) if squeezed else X

    def to_coords(self, X: torch.Tensor) -> torch.Tensor:
        """
        Center and scale to unit Frobenius norm.
        Input: (..., N, dim)
        Output: same shape, centered and unit-norm per batch element.
        """
        if X.ndim < 2:
            raise ValueError("Expected X of shape (N, dim) or (B,N,dim)")
        if X.shape[-1] != self.dim:
            raise ValueError(f"Last dimension must be {self.dim} (got {X.shape[-1]})")

        dev = X.device
        dtype = X.dtype

        # center along points axis (-2)
        Xc = X - torch.mean(X, dim=-2, keepdim=True)

        # Frobenius norm per batch: sqrt(sum over last two dims)
        f_norm = torch.norm(Xc.view(*Xc.shape[:-2], -1), p=2, dim=-1, keepdim=True)
        # f_norm has shape (...,1). Need to broadcast to (...,1,1)
        f_norm = f_norm.unsqueeze(-1)
        f_norm = torch.clamp(f_norm, min=1e-8)

        Xn = Xc / f_norm
        return Xn

    # ----------------------
    # Procrustes alignment
    # ----------------------
    def optimal_rotation(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Optimal rotation matrices R that align Y to X (so Y @ R ≈ X).
        Works for shapes X,Y of shape (B,N,2) or (N,2).
        Returns R of shape (B,2,2) or (2,2) matching input squeeze.
        """
        X_b, squeeze = self._ensure_batch(X)
        Y_b, _ = self._ensure_batch(Y)

        if X_b.shape != Y_b.shape:
            raise ValueError("X and Y must have the same shape")

        dev = X_b.device
        dtype = X_b.dtype
        B = X_b.shape[0]

        # M = X^T @ Y  -> (B,2,2)
        M = torch.bmm(X_b.transpose(-2, -1), Y_b)

        # Try SVD; if fails, add tiny jitter and retry.
        try:
            U, S, Vt = torch.linalg.svd(M)
        except RuntimeError:
            jitter = 1e-8 * torch.eye(2, device=dev, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
            U, S, Vt = torch.linalg.svd(M + jitter)

        V = Vt.transpose(-2, -1)
        R = torch.bmm(V, U.transpose(-2, -1))

        # Correct reflections: if det(R) < 0, flip last column of V
        detR = torch.det(R)
        needs_corr = detR < 0
        if needs_corr.any():
            V_corr = V.clone()
            V_corr[needs_corr, :, -1] *= -1.0
            R[needs_corr] = torch.bmm(V_corr[needs_corr], U[needs_corr].transpose(-2, -1))

        return R.squeeze(0) if squeeze else R

    def wellpos(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Return Y rotated to best align with X.
        Input shapes: (B,N,2) or (N,2)
        """
        X_b, squeeze = self._ensure_batch(X)
        Y_b, _ = self._ensure_batch(Y)
        R = self.optimal_rotation(X_b, Y_b)  # (B,2,2) or (2,2)
        # Broadcast multiply: Y_b @ R
        # If R is (2,2) and B>1, matmul will broadcast; ensure R has batch dim
        if R.dim() == 2 and X_b.shape[0] > 1:
            R = R.unsqueeze(0).expand(X_b.shape[0], -1, -1)
        Y_aligned = torch.bmm(Y_b, R)
        return Y_aligned.squeeze(0) if squeeze else Y_aligned

    # ----------------------
    # Distance and inner product
    # ----------------------
    def _fro_inner(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Frobenius inner product sum_{i,j} X_ij * Y_ij across last two dims -> (...,)"""
        prod = torch.sum(X * Y, dim=(-2, -1))
        return prod

    def dist(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Procrustes distance (geodesic distance on pre-shape sphere) between X and Y.
        Assumes X, Y are pre-shape (centered + unit-norm). If not, callers should call to_coords.
        Outputs a tensor of shape (B,) or scalar.
        """
        X_b, squeeze = self._ensure_batch(X)
        Y_b, _ = self._ensure_batch(Y)

        # Align Y to X
        Y_aligned = self.wellpos(X_b, Y_b)
        inner = self._fro_inner(X_b, Y_aligned)
        inner_clamped = torch.clamp(inner, -1.0, 1.0)
        theta = torch.acos(inner_clamped)
        return theta.squeeze(0) if squeeze else theta

    # ----------------------
    # Vertical / Horizontal decomposition
    # ----------------------
    @staticmethod
    def _skew_matrix_from_scalar(a: torch.Tensor) -> torch.Tensor:
        """
        Build a 2x2 skew matrix [[0, -a],[a, 0]] for a tensor a of shape (B,).
        Returns (B,2,2).
        """
        B = a.shape[0]
        A = torch.zeros(B, 2, 2, device=a.device, dtype=a.dtype)
        A[:, 0, 1] = -a
        A[:, 1, 0] = a
        return A

    def horizontal_projection(self, X: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Project tangent V at base X to horizontal subspace (remove rotational component).
        Inputs: X, V of shape (B,N,2) or (N,2). Returns V_horizontal of same shape.
        Algebra derived from: solve A G + G A = M for skew A, with G = X^T X (2x2).
        For 2D and A=[[0,-a],[a,0]], one obtains:
            a = - M[0,1] / (G[0,0] + G[1,1])
        where M = V^T X - X^T V (2x2 skew).
        """
        X_b, squeeze = self._ensure_batch(X)
        V_b, _ = self._ensure_batch(V)
        if X_b.shape != V_b.shape:
            raise ValueError("X and V must have same shape")

        B = X_b.shape[0]
        dev = X_b.device
        dtype = X_b.dtype

        # center V (horizontal tangent vectors must be centered)
        V_centered = V_b - torch.mean(V_b, dim=-2, keepdim=True)

        # Compute M = V^T X - X^T V  (B,2,2)
        M = torch.bmm(V_centered.transpose(-2, -1), X_b) - torch.bmm(X_b.transpose(-2, -1), V_centered)

        # Compute Gram G = X^T X  (B,2,2)
        G = torch.bmm(X_b.transpose(-2, -1), X_b)

        # Solve for scalar a:
        # denom = G00 + G11  (B,)
        g00 = G[:, 0, 0]
        g11 = G[:, 1, 1]
        denom = g00 + g11
        safe_denom = torch.where(torch.abs(denom) > 1e-8, denom, torch.ones_like(denom))
        a = -M[:, 0, 1] / safe_denom  # (B,)

        # Build A matrices and vertical component
        A = self._skew_matrix_from_scalar(a)
        # V_vertical = (A @ X^T)^T = X @ A^T? we'll compute directly
        V_vertical = torch.bmm(A, X_b.transpose(-2, -1)).transpose(-2, -1)

        V_horizontal = V_centered - V_vertical

        return V_horizontal.squeeze(0) if squeeze else V_horizontal

    # For convenience, expose vertical computation (small helper)
    def vertical_component(self, X: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        Vh = self.horizontal_projection(X, V)
        return V - Vh

    # ----------------------
    # Log / Exp / Geodesics
    # ----------------------
    def log(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Log map at X of Y: returns a horizontal tangent vector V at X such that exp_X(V)=Y (mod numerical).
        Ensures Y is well-positioned and projects the result to horizontal.
        """
        X_b, squeeze = self._ensure_batch(X)
        Y_b, _ = self._ensure_batch(Y)

        # Pre-shape check: ensure X and Y are centered+united
        Xb = self.to_coords(X_b)
        Yb = self.to_coords(Y_b)

        # Well-position (align Y to X)
        Y_aligned = self.wellpos(Xb, Yb)

        # inner product and theta
        inner = self._fro_inner(Xb, Y_aligned).unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
        inner = torch.clamp(inner, -1.0, 1.0)
        theta = torch.acos(inner.squeeze(-1).squeeze(-1))  # (B,)

        # Prepare broadcasting
        theta_b = theta.view(-1, 1, 1)
        sin_theta = torch.sin(theta_b)
        eps = 1e-6

        # Robust coefficient: theta / sin(theta) with linearization
        coef = torch.where(torch.abs(sin_theta) > eps, theta_b / sin_theta, torch.ones_like(theta_b))

        # V = coef * (Y_aligned - <X,Y_aligned> * X)
        V = coef * (Y_aligned - inner * Xb)

        # Project to horizontal to remove rotational component (precise)
        V_horizontal = self.horizontal_projection(Xb, V)

        return V_horizontal.squeeze(0) if squeeze else V_horizontal

    def exp(self, X: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Exponential map: returns endpoint of geodesic starting at X with initial tangent V (unit time).
        V will be projected to horizontal to guarantee horizontality of the geodesic.
        """
        X_b, squeeze = self._ensure_batch(X)
        V_b, _ = self._ensure_batch(V)

        # Ensure X on pre-shape sphere
        Xb = self.to_coords(X_b)

        # Project V to horizontal
        Vh = self.horizontal_projection(Xb, V_b)

        # Compute Frobenius norm of V per batch
        Vnorm = torch.norm(Vh.view(Vh.shape[0], -1), p=2, dim=-1, keepdim=True)  # (B,1)
        Vnorm = Vnorm.unsqueeze(-1)  # (B,1,1)
        eps = 1e-6
        small = Vnorm < eps

        # Direction safe divide
        Vdir = Vh / (Vnorm + 1e-24)

        cos_t = torch.cos(Vnorm)
        sin_t = torch.sin(Vnorm)

        result = cos_t * Xb + sin_t * Vdir
        # Linearize when Vnorm small
        result = torch.where(small, Xb + Vh, result)

        # Re-project to ensure unit pre-shape
        result = self.to_coords(result)

        return result.squeeze(0) if squeeze else result

    def geopoint(self, X: torch.Tensor, Y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Point on the horizontal geodesic from X to Y at time t in [0,1].
        Uses log+exp to guarantee horizontality exactly.
        t may be a scalar or a tensor of shape (B,) or (1,).
        """
        X_b, squeeze = self._ensure_batch(X)
        Y_b, _ = self._ensure_batch(Y)

        # compute initial velocity (horizontal)
        V = self.log(X_b, Y_b)  # (B,N,2)
        # ensure V has batch dim
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=X_b.device, dtype=X_b.dtype)
        # shape t to (B,1,1)
        if t.ndim == 0:
            t = t.view(1)
        if t.ndim == 1 and t.shape[0] == 1 and X_b.shape[0] > 1:
            # expand scalar to batch
            t = t.expand(X_b.shape[0])
        t_b = t.view(-1, 1, 1)

        Xt = self.exp(X_b, V * t_b)  # exp(X, t*V)
        return Xt.squeeze(0) if squeeze else Xt


# ----------------------
# Batch geodesic sampling
# ----------------------
def sample_kendall_geodesic_batch(
        manifold: KendallShapeSpace,
        x0_batch: torch.Tensor,
        x1_batch: torch.Tensor,
        device: Optional[str] = None,
        t_sample: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample points and horizontal velocities along Kendall geodesics for a batch of pairs.
    - x0_batch, x1_batch: (B,N,2) or (N,2). They will be to_coords() internally.
    - t_sample: optional tensor of shape (B,) with times in [0,1]. If None, sample uniform random times.
    Returns: (t, xt, ut) where
      t: (B,) sampled times
      xt: (B,N,2) points at time t
      ut: (B,N,2) horizontal velocities (d/dt xt)
    Analytical velocity formula (exact for sphere geodesic):
      Let V = log(x0, x1) with norm theta.
      xt = cos(t theta) X + sin(t theta) (V/theta)
      ut = cos(t theta) * V - theta * sin(t theta) * X
    """
    X0_b, squeeze = KendallShapeSpace._ensure_batch.__get__(manifold, None)(x0_batch)
    X1_b, _ = KendallShapeSpace._ensure_batch.__get__(manifold, None)(x1_batch)

    dev = X0_b.device if device is None else (torch.device(device))
    dtype = X0_b.dtype

    B = X0_b.shape[0]

    if t_sample is None:
        t = torch.rand(B, device=dev, dtype=dtype)
    else:
        t = t_sample.to(dev=device, dtype=dtype)
        # normalize shape
        if t.ndim == 0:
            t = t.view(1).expand(B)
        elif t.ndim == 1 and t.shape[0] == 1 and B > 1:
            t = t.expand(B)

    # Ensure inputs are pre-shape
    X0_b = manifold.to_coords(X0_b)
    X1_b = manifold.to_coords(X1_b)

    # Compute horizontal initial velocity V (B,N,2)
    V = manifold.log(X0_b, X1_b)

    # Norm (theta) per batch
    theta = torch.norm(V.view(B, -1), p=2, dim=-1)  # (B,)
    theta_b = theta.view(B, 1, 1)

    # shape t to (B,1,1)
    t_b = t.view(B, 1, 1)

    # Avoid dividing by zero for theta when forming V/theta
    small = theta_b.abs() < 1e-8
    Vdir = V / (theta_b + 1e-24)

    cos_term = torch.cos(t_b * theta_b)
    sin_term = torch.sin(t_b * theta_b)

    xt = cos_term * X0_b + sin_term * Vdir
    xt = manifold.to_coords(xt)

    # Analytical time derivative (horizontal)
    # ut = cos(t theta) * V - theta * sin(t theta) * X0
    ut = cos_term * V - (theta_b * sin_term) * X0_b
    ut = manifold.horizontal_projection(xt, ut)  # safety: ensure horizontality at xt

    return t, xt.squeeze(0) if squeeze else xt, ut.squeeze(0) if squeeze else ut


# ----------------------
# Quick sanity tests
# ----------------------
if __name__ == "__main__":
    # Use CPU for test to avoid GPU dependency
    device = torch.device("cpu")

    # small two-point polygon example (N=2)
    N = 3
    M = KendallShapeSpace(n_points=N, dim=2)

    # create simple shape X and rotated Y
    X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.get_default_dtype(), device=device)
    X = M.to_coords(X)  # center + unit
    # rotate X by 90 degrees
    R90 = torch.tensor([[0.0, -1.0], [1.0, 0.0]], device=device)
    Y = X @ R90
    Y = M.to_coords(Y)

    # optimal rotation should invert rotation
    R = M.optimal_rotation(X, Y)
    Y_aligned = M.wellpos(X, Y)
    print("Det(R) ~ 1:", torch.det(R).item())
    print("Alignment error (should be ~0):", torch.norm(Y_aligned - X).item())

    # horizontal projection: vertical vector A@X should be removed
    # pick small skew a
    a = 0.3
    A = torch.tensor([[0.0, -a], [a, 0.0]], dtype=X.dtype, device=device)
    V_vert = torch.bmm(A.unsqueeze(0).expand(1, -1, -1), X.unsqueeze(0).transpose(-2, -1)).transpose(-2, -1)
    V_proj = M.horizontal_projection(X, V_vert)
    print("Norm of vertical after projection (should be ~0):", torch.norm(V_proj).item())

    # exp(log) roundtrip
    V = M.log(X, Y)
    Y_rec = M.exp(X, V)
    print("exp(log) error (should be small):", torch.norm(Y_rec - Y).item())

    # sample batch
    x0 = X.unsqueeze(0).expand(4, -1, -1)
    x1 = Y.unsqueeze(0).expand(4, -1, -1)
    t_samples, xt, ut = sample_kendall_geodesic_batch(M, x0, x1)
    print("Sampled times:", t_samples)
    print("xt shape:", xt.shape, "ut shape:", ut.shape)
    # ensure horizontality of ut
    hv = M.horizontal_projection(xt, ut)
    print("Velocity horizontal residual (should be small):", torch.norm(hv - ut).item())

    print("All quick checks done.")
