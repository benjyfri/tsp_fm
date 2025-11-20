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
        # Create pre-shape space (sphere of centered configurations)
        self.space = PreShapeSpace(k_landmarks=n_points, ambient_dim=ambient_dim, equip=True)
        self.metric = self.space.metric

    def to_tangent(self, vector, base_point):
        """Project a vector to the tangent space at base_point."""
        return self.space.to_tangent(vector, base_point)

    @property
    def dim(self):
        return self.space.dim