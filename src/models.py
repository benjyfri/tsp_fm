import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, t):
        t_scalar = t.flatten()
        half_dim = self.dim // 2
        # Standard positional embedding logic
        freqs = torch.exp(torch.arange(half_dim, device=t.device) * (-math.log(10000) / (half_dim - 1)))
        args = t_scalar[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return (self.scale * embedding).view(t.shape[0], self.dim)

class VectorFieldModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_points = config.num_points
        self.input_dim = 2

        # 1. Time Embedding
        self.time_embed = TimeEmbedding(config.t_emb_dim)

        # 2. Input Projection (Coordinates + Time)
        input_feature_dim = self.input_dim + config.t_emb_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_feature_dim, config.embed_dim),
            nn.GELU(),
            nn.LayerNorm(config.embed_dim)
        )

        # 3. Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embed_dim*4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # 4. Output Head
        self.output_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 2),
            nn.GELU(),
            nn.Linear(config.embed_dim * 2, self.input_dim)
        )

        # Zero initialization helps flow matching stability at start
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def forward(self, x, t, geometry=None):
        """
        Args:
            x: (B, N, 2) Current points
            t: (B,) Current time
            geometry: (Optional) The GeometryProvider object.
                      If provided, projects output to tangent space.
        """
        B, N, _ = x.shape

        # Embed time and expand to (B, N, t_dim)
        t_emb = self.time_embed(t).unsqueeze(1).repeat(1, N, 1)

        # Concatenate x and t
        inp = torch.cat([x, t_emb], dim=2)

        # Forward pass
        h = self.input_projection(inp)
        h = self.transformer(h)
        v_raw = self.output_head(h)

        # --- OPTIONAL PROJECTION (The "Shape" logic) ---
        if geometry is not None:
            # This enforces the vector field to stay on the manifold
            return geometry.to_tangent(v_raw, x)

        return v_raw

class ConfigurableMLP(nn.Module):
    """A flexible MLP module with configurable layers and dimensions."""

    def __init__(self, input_size, num_hidden_layers, hidden_dim, output_size, activation=nn.ReLU):
        super().__init__()
        layers = []

        # Input Layer
        layers.append(nn.Linear(input_size, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(activation())

        # Hidden Layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation())

        # Output Layer
        layers.append(nn.Linear(hidden_dim, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (B, input_size)
        Returns: (B, output_size)
        """
        return self.model(x)

# --- New Model (Requirements 1-6) ---

class CanonicalMLPVectorField(nn.Module):
    """
    A vector field model for flow matching that first moves the 2D point cloud
    into a canonical pose before processing it with an MLP.

    Assumes input point clouds are centered at (0,0).
    """

    def __init__(self, n_points, t_emb_dim=128, mlp_hidden_layers=4, mlp_hidden_dim=256):
        """
        Args:
            n_points (int): Number of points in the input cloud (e.g., 15).
            t_emb_dim (int): Dimension for the time embedding.
            mlp_hidden_layers (int): Number of hidden layers in the MLP.
            mlp_hidden_dim (int): Number of neurons in each hidden layer.
        """
        super().__init__()
        self.n_points = n_points
        self.input_dim = 2  # 2D coordinates

        # 1. Receives time 't'
        self.time_embed = TimeEmbedding(t_emb_dim)

        # 2. Accounts for (N points + 1 CoM point) * 2 dims + time
        mlp_input_size = (self.n_points + 1) * self.input_dim + t_emb_dim

        # 6. Output is velocity for the N original points
        mlp_output_size = self.n_points * self.input_dim

        # 5. Flexible MLP
        self.mlp = ConfigurableMLP(
            input_size=mlp_input_size,
            num_hidden_layers=mlp_hidden_layers,
            hidden_dim=mlp_hidden_dim,
            output_size=mlp_output_size
        )

    def _canonicalize(self, x, epsilon=1e-8):
        """
        Applies the specified canonical reordering and pose fixing.
        (Implements Requirement 4)

        x shape: (B, N, 2)
        """
        B, N, D = x.shape

        # --- 4a. Find p_n and rotate to positive y-axis ---

        # Find p_n: point with largest magnitude
        magnitudes = torch.norm(x, dim=2) # (B, N)
        max_mag_indices = torch.argmax(magnitudes, dim=1) # (B,)

        # Get p_n
        p_n = torch.gather(x, 1, max_mag_indices.view(B, 1, 1).expand(-1, -1, 2)).squeeze(1) # (B, 2)

        # Calculate rotation R1 to move p_n to (0, ||p_n||)
        p_n_norm = torch.norm(p_n, dim=1, keepdim=True) + epsilon # (B, 1)

        # Normalized coordinates of p_n: (cos(theta_pn), sin(theta_pn))
        px_norm = p_n[:, 0:1] / p_n_norm # (B, 1)
        py_norm = p_n[:, 1:2] / p_n_norm # (B, 1)

        # We want to rotate by alpha = pi/2 - theta_pn
        # cos(alpha) = cos(pi/2)cos(theta) + sin(pi/2)sin(theta) = sin(theta)
        # sin(alpha) = sin(pi/2)cos(theta) - cos(pi/2)sin(theta) = cos(theta)
        cos_alpha = py_norm # (B, 1)
        sin_alpha = px_norm # (B, 1)

        # Build rotation matrix R1 (B, 2, 2)
        R1 = torch.zeros(B, 2, 2, device=x.device, dtype=x.dtype)
        R1[:, 0, 0] = cos_alpha.squeeze(-1)
        R1[:, 0, 1] = -sin_alpha.squeeze(-1)
        R1[:, 1, 0] = sin_alpha.squeeze(-1)
        R1[:, 1, 1] = cos_alpha.squeeze(-1)

        # Apply first rotation
        rotated_x = torch.bmm(x, R1)

        # --- 4b. Find p_x and fix reflection ambiguity ---

        # Find p_x: point closest to *original* p_n (excluding p_n itself)
        dists = torch.norm(x - p_n.unsqueeze(1), dim=2) # (B, N)
        dists.scatter_(1, max_mag_indices.unsqueeze(1), float('inf')) # Ignore p_n

        closest_indices = torch.argmin(dists, dim=1) # (B,)

        # Get the *rotated* version of p_x
        p_x_rotated = torch.gather(rotated_x, 1, closest_indices.view(B, 1, 1).expand(-1, -1, 2)).squeeze(1) # (B, 2)

        # Check ambiguity: if p_x_rotated.x < 0, reflect across y-axis
        reflection_mask = torch.where(p_x_rotated[:, 0] < 0, -1.0, 1.0).to(x.dtype) # (B,)

        # Build reflection matrix R2 (B, 2, 2)
        R2 = torch.eye(2, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B, 1, 1)
        R2[:, 0, 0] = reflection_mask # Multiplies x-coord by -1 or 1

        # Apply second rotation (reflection)
        final_x = torch.bmm(rotated_x, R2)

        return final_x

    def forward(self, x, t):
        """
        x: (B, N, 2) - current state xt (Requirement 1)
        t: (B,)      - time (Requirement 1)
        Output: (B, N, 2) - velocity vector vt (Requirement 6)
        """
        B, N, D = x.shape
        if N != self.n_points:
            raise ValueError(f"Input has {N} points, but model expects {self.n_points}")
        if D != self.input_dim:
            raise ValueError(f"Input has {D} dims, but model expects {self.input_dim}")

        # 4. Canonicalize (uses only coordinates - Req 3)
        x_canonical = self._canonicalize(x) # (B, N, 2)

        # 2. Add CoM point (assumed 0,0)
        com_point = torch.zeros(B, 1, D, device=x.device, dtype=x.dtype)
        x_full = torch.cat([x_canonical, com_point], dim=1) # (B, N+1, 2)

        # 1. Embed time
        t_emb = self.time_embed(t) # (B, t_emb_dim)

        # 5. Prepare input for flexible MLP
        # Flatten point cloud
        x_flat = x_full.reshape(B, -1) # (B, (N+1)*2)

        # Concatenate time
        mlp_input = torch.cat([x_flat, t_emb], dim=1) # (B, (N+1)*2 + t_emb_dim)

        # Run MLP
        mlp_output = self.mlp(mlp_input) # (B, N*2)

        # 6. Reshape output to match flow matching target
        v_out = mlp_output.reshape(B, N, D) # (B, N, 2)

        return v_out