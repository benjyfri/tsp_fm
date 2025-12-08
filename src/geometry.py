import os

# Must set backend before importing geomstats
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'

import torch
import geomstats.backend as gs
from geomstats.geometry.pre_shape import PreShapeSpace


class GeometryProvider:
    def __init__(self, n_points, ambient_dim=2):
        self.n_points = n_points
        self.ambient_dim = ambient_dim
        # Create pre-shape space (hypersphere of centered configurations)
        self.space = PreShapeSpace(k_landmarks=n_points, ambient_dim=ambient_dim, equip=True)
        self.metric = self.space.metric

    def to_tangent(self, vector, base_point):
        """Project a vector to the tangent space at base_point."""
        return self.space.to_tangent(vector, base_point)

    def _project(self, x):
        """
        Manually project raw coordinates to Pre-Shape Space:
        1. Center (remove translation)
        2. Normalize (remove scaling)
        """
        # Center
        x_centered = x - torch.mean(x, dim=-2, keepdim=True)

        # Normalize (Frobenius norm = 1)
        # We clamp min value to avoid division by zero
        norm = torch.norm(x_centered, p='fro', dim=(-2, -1), keepdim=True)
        x_normed = x_centered / (norm + 1e-8)

        return x_normed

    def _align(self, source, target):
        """
        Optimal rotation (Kabsch Algorithm) to align 'source' to 'target'.
        Assumes inputs are already centered.
        """
        # 1. Compute covariance matrix
        # shapes: (N, 2)
        cov = torch.matmul(source.transpose(-2, -1), target)

        # 2. SVD
        U, S, Vh = torch.linalg.svd(cov)

        # 3. Optimal Rotation R = U @ Vh
        # Note: Vh is V.T in PyTorch
        R = torch.matmul(U, Vh)

        # 4. Correction for Reflection (Ensure strictly SO(2))
        # If determinant is negative, we have a reflection, not a rotation.
        d = torch.linalg.det(R)

        # If det < 0, flip the last singular vector
        if d < 0:
            diag = torch.ones(self.ambient_dim, device=source.device)
            diag[-1] = -1
            # Recalculate R with the flip
            R = U @ torch.diag(diag) @ Vh

        # 5. Apply Rotation
        # source (N, 2) @ R (2, 2) -> (N, 2)
        # Note: We treat points as row vectors, so we multiply R on the right?
        # Standard: y = R x. If x is column.
        # Here x is (N, 2). So x_new = x @ R.T?
        # Let's check dims: (N, 2) x (2, 2).
        return torch.matmul(source, R.transpose(-2, -1))

    def distance(self, point_a, point_b):
        """
        Compute the robust Kendall Shape Space distance.
        Handling Centering -> Normalizing -> Aligning -> Geodesic
        """
        # 1. Project both to Pre-Shape Space (Center & Normalize)
        # This handles the fact that 'original_cities' has arbitrary scale
        a_proj = self._project(point_a)
        b_proj = self._project(point_b)

        # 2. Align point_b to point_a (Remove Rotation)
        # We do this manually to avoid attribute errors in older geomstats versions
        b_aligned = self._align(b_proj, a_proj)

        # 3. Compute distance on the Hypersphere (Pre-shape metric)
        # Since we manually aligned them, this is now the Kendall distance
        return self.metric.dist(a_proj, b_aligned)

    @property
    def dim(self):
        return self.space.dim