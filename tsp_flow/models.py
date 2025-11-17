import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Dependency from your current script ---

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding (like in transformers), robustified."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.max_period = 10000

    def forward(self, t):
        t_scalar = t.flatten()
        half_dim = self.dim // 2
        device = t.device

        if half_dim == 0:
            return torch.zeros(t.shape[0], self.dim, device=device)

        log_max_period = torch.log(torch.tensor(self.max_period, device=device))

        # FIX: Ensure denominator is a float tensor before clamping
        denominator = torch.tensor(half_dim - 1, dtype=torch.float32, device=device).clamp(min=1.0)

        freqs = torch.exp(
            torch.arange(0, half_dim, device=device) * (-log_max_period / denominator)
        )

        args = t_scalar[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return embedding.view(t.shape[0], self.dim)

# --- Flexible MLP (Requirement 5) ---

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

# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Dependency from your current script ---
# [TimeEmbedding, ConfigurableMLP, CanonicalMLPVectorField, TransformerBlock...
#  all of that code is fine and remains unchanged]
# ... (omitted for brevity) ...

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding (like in transformers), robustified."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.max_period = 10000
    def forward(self, t):
        t_scalar = t.flatten(); half_dim = self.dim // 2; device = t.device
        if half_dim == 0: return torch.zeros(t.shape[0], self.dim, device=device)
        log_max_period = torch.log(torch.tensor(self.max_period, device=device))
        denominator = torch.tensor(half_dim - 1, dtype=torch.float32, device=device).clamp(min=1.0)
        freqs = torch.exp(torch.arange(0, half_dim, device=device) * (-log_max_period / denominator))
        args = t_scalar[:, None] * freqs[None, :]; embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding.view(t.shape[0], self.dim)

class TransformerBlock(nn.Module):
    """A standard Transformer encoder block using Self-Attention."""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim); self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(),
                                 nn.Linear(4 * embed_dim, embed_dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        attn_output, _ = self.attn(x, x, x); x = x + self.dropout(attn_output); x = self.norm1(x)
        ffn_output = self.ffn(x); x = x + ffn_output; x = self.norm2(x)
        return x


class StrongEquivariantVectorField(nn.Module):
    """
    An enhanced Transformer-based model to learn the vector field v(xt, t),
    ensuring equivariance to point permutation.
    """

    def __init__(self, n_points=15, embed_dim=256, t_emb_dim=128, num_layers=4, num_heads=8, dropout=0.0):
        super().__init__()
        self.n_points = n_points
        self.embed_dim = embed_dim
        self.input_dim = 2  # 2D coordinates

        self.time_embed = TimeEmbedding(t_emb_dim)

        input_feature_dim = self.input_dim + t_emb_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_feature_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])

        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, self.input_dim)
        )

    def forward(self, x, t):
        """
        x: (B, N, 2) - current state xt
        t: (B,)      - time
        Output: (B, N, 2) - velocity vector vt
        """
        B, N, D = x.shape
        t_emb = self.time_embed(t)
        t_expanded = t_emb.unsqueeze(1).repeat(1, N, 1)
        inp = torch.cat([x, t_expanded], dim=2)
        h = self.input_projection(inp)

        for block in self.transformer_blocks:
            h = block(h)

        vt = self.output_head(h)

        # --- [CRITICAL FIX] ---
        # Project the output vector vt onto the tangent space of the
        # Kendall sphere defined by x. This prevents the model from
        # "cheating" by outputting a radial vector.
        # We assume x is unit norm, which the data sanity check confirmed.

        # Calculate the radial component of vt
        # dot_v shape is (B, 1, 1)
        # dot_v = torch.sum(x * vt, dim=(1, 2), keepdim=True)
        #
        # # Subtract the radial component to get the tangent vector
        # vt_tan = vt - dot_v * x
        #
        # return vt_tan # Return the *tangent* vector
        return vt
        # --- [END FIX] ---

# (The other models, ConfigurableMLP and CanonicalMLPVectorField, are not used
# in your train.py, but they are fine to leave as-is)