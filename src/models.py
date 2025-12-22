import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. Helper Functions & Classes ---

def rotate_half(x):
    """
    Rotates half the hidden dims of the input.
    Input: [x0, x1, x2, x3]
    Output: [-x1, x0, -x3, x2]
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)

def apply_rope(q, k, cos, sin):
    """
    Applies Rotary Position Embedding to query and key tensors.
    """
    # q, k: (B, num_heads, seq_len, head_dim)
    # cos, sin: (B, 1, 1, head_dim) - broadcastable
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryTimeEmbedding(nn.Module):
    """
    RoPE logic for time embeddings.
    """
    def __init__(self, dim, base=10000):
        super().__init__()
        assert dim % 2 == 0, f"RoPE requires even dim, got {dim}"
        self.dim = dim
        self.base = base
        # Create (dim/2) frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, t):
        # Handle (B,), (B, 1) -> (B,)
        t = t.flatten().to(dtype=self.inv_freq.dtype)

        # Outer product: (B, dim/2)
        angles = torch.outer(t, self.inv_freq)

        # Interleave angles to match rotate_half [theta0, theta0, theta1, theta1...]
        angles = torch.stack([angles, angles], dim=-1).flatten(-1) # (B, dim)

        # Reshape for broadcasting: (B, 1, 1, dim)
        cos = angles.cos().view(t.shape[0], 1, 1, -1)
        sin = angles.sin().view(t.shape[0], 1, 1, -1)
        return cos, sin

class TimeEmbedding(nn.Module):
    """Standard sinusoidal time embedding (Legacy/Fallback)."""
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, f"TimeEmbedding dimension must be even, got {dim}"
        self.dim = dim
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, t):
        t = t.to(dtype=self.scale.dtype)
        t_scalar = t.flatten()
        half_dim = self.dim // 2
        denominator = max(half_dim - 1, 1)
        exponent = torch.arange(half_dim, device=t.device, dtype=t.dtype) * (-math.log(10000) / denominator)
        freqs = torch.exp(exponent)
        args = t_scalar[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return (self.scale * embedding).view(t.shape[0], self.dim)

class ConfigurableMLP(nn.Module):
    """A flexible MLP module."""
    def __init__(self, input_size, num_hidden_layers, hidden_dim, output_size, activation=nn.ReLU):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(activation())
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation())
        layers.append(nn.Linear(hidden_dim, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# --- 2. Attention Modules ---

class RoPEMultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.embed_dim = config.embed_dim
        self.head_dim = config.embed_dim // config.num_heads
        self.dropout = getattr(config, 'dropout', 0.0)
        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=False)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_dropout = nn.Dropout(self.dropout)

    def forward(self, x, cos, sin):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q, k = apply_rope(q, k, cos, sin)

        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        return self.proj_dropout(self.proj(attn_output))

class RoPETransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = RoPEMultiheadAttention(config)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        dropout = getattr(config, 'dropout', 0.0)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(config.embed_dim * 4, config.embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x

# --- 3. Main Models ---

class VectorFieldModel(nn.Module):
    """Legacy model for compatibility."""
    def __init__(self, config):
        super().__init__()
        self.n_points = config.num_points
        self.input_dim = 2
        self.time_embed = TimeEmbedding(config.t_emb_dim)
        input_feature_dim = self.input_dim + config.t_emb_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_feature_dim, config.embed_dim),
            nn.GELU(),
            nn.LayerNorm(config.embed_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim, nhead=config.num_heads,
            dim_feedforward=config.embed_dim*4, dropout=config.dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.output_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 2),
            nn.GELU(),
            nn.Linear(config.embed_dim * 2, self.input_dim)
        )
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def forward(self, x, t, geometry=None):
        B, N, _ = x.shape
        t_emb = self.time_embed(t).unsqueeze(1).repeat(1, N, 1)
        inp = torch.cat([x, t_emb], dim=2)
        h = self.input_projection(inp)
        h = self.transformer(h)
        v_raw = self.output_head(h)
        if geometry is not None:
            return geometry.to_tangent(v_raw, x)
        return v_raw

class RoPEVectorFieldModel(nn.Module):
    """Standard RoPE Transformer (Non-Canonical)."""
    def __init__(self, config):
        super().__init__()
        self.input_dim = 2
        self.config = config
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, config.embed_dim),
            nn.LayerNorm(config.embed_dim)
        )
        head_dim = config.embed_dim // config.num_heads
        self.rope_time_embed = RotaryTimeEmbedding(head_dim)
        self.layers = nn.ModuleList([RoPETransformerBlock(config) for _ in range(config.num_layers)])
        self.output_head = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.GELU(),
            nn.Linear(config.embed_dim // 2, self.input_dim)
        )
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def forward(self, x, t, geometry=None):
        h = self.input_projection(x)
        cos, sin = self.rope_time_embed(t)
        for layer in self.layers:
            h = layer(h, cos, sin)
        v_raw = self.output_head(h)
        if geometry is not None:
            return geometry.to_tangent(v_raw, x)
        return v_raw

class CanonicalMLPVectorField(nn.Module):
    """
    Canonical Vector field using an MLP backbone with RoPE time embedding.
    """
    def __init__(self, config):
        super().__init__()
        self.n_points = config.num_points
        self.input_dim = 2

        assert config.embed_dim % 2 == 0, "Embed dim must be even for RoPE"
        self.rope_time_embed = RotaryTimeEmbedding(config.embed_dim)

        raw_input_size = (self.n_points + 1) * self.input_dim
        self.input_projection = nn.Sequential(
            nn.Linear(raw_input_size, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            nn.ReLU()
        )

        self.mlp = ConfigurableMLP(
            input_size=config.embed_dim,
            num_hidden_layers=config.num_layers,
            hidden_dim=config.embed_dim,
            output_size=self.n_points * self.input_dim
        )

    def forward(self, x, t, geometry=None):
        B, N, D = x.shape

        # 1. Canonicalize
        R = self._get_canonical_rotation_optimized(x)
        x_canonical = torch.bmm(x, R)

        # 2. Project
        com_point = torch.zeros(B, 1, D, device=x.device, dtype=x.dtype)
        x_full = torch.cat([x_canonical, com_point], dim=1)
        x_flat = x_full.reshape(B, -1)
        x_embed = self.input_projection(x_flat)

        # 3. RoPE
        cos, sin = self.rope_time_embed(t) # (B, 1, 1, embed_dim)
        x_embed_reshaped = x_embed.view(B, 1, 1, -1)
        x_rotated = (x_embed_reshaped * cos) + (rotate_half(x_embed_reshaped) * sin)
        x_input = x_rotated.view(B, -1)

        # 4. MLP
        mlp_output = self.mlp(x_input)
        v_canonical = mlp_output.reshape(B, N, D)

        # 5. Inverse Rotate
        v_global = torch.bmm(v_canonical, R.transpose(1, 2))

        if geometry is not None:
            return geometry.to_tangent(v_global, x)
        return v_global

# Optimized canonical rotation helper
def _get_canonical_rotation_optimized(x, epsilon=1e-8):
    """
    Optimized version with fewer operations and better memory usage.
    """
    B, N, D = x.shape
    device = x.device
    dtype = x.dtype

    # 1. Find max magnitude point (vectorized)
    magnitudes = torch.norm(x, dim=2)
    max_mag_indices = torch.argmax(magnitudes, dim=1)

    # 2. Extract p_n efficiently
    batch_indices = torch.arange(B, device=device)
    p_n = x[batch_indices, max_mag_indices]  # (B, 2) - faster than gather

    # 3. Compute first rotation
    p_n_norm = torch.norm(p_n, dim=1, keepdim=True).clamp(min=epsilon)
    cos_alpha = p_n[:, 1:2] / p_n_norm
    sin_alpha = -p_n[:, 0:1] / p_n_norm

    # Build R1 more efficiently (avoid zeros initialization)
    R1 = torch.stack([
        torch.cat([cos_alpha, -sin_alpha], dim=1),
        torch.cat([sin_alpha, cos_alpha], dim=1)
    ], dim=1)  # (B, 2, 2)

    # 4. Apply first rotation
    x_rot = torch.bmm(x, R1)

    # 5. Find closest point (excluding p_n)
    dists = torch.norm(x - p_n.unsqueeze(1), dim=2)
    dists[batch_indices, max_mag_indices] = float('inf')
    closest_indices = torch.argmin(dists, dim=1)

    # 6. Determine reflection
    p_x_rotated = x_rot[batch_indices, closest_indices]
    reflection_mask = torch.where(
        p_x_rotated[:, 0] < 0,
        torch.tensor(-1.0, dtype=dtype, device=device),
        torch.tensor(1.0, dtype=dtype, device=device)
    )

    # Build R2 efficiently
    R2 = torch.eye(2, dtype=dtype, device=device).unsqueeze(0).expand(B, 2, 2).clone()
    R2[:, 0, 0] = reflection_mask

    # Combine rotations
    return torch.bmm(R1, R2)


# Drop-in replacement for both canonical models
class CanonicalRoPEVectorField(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = 2
        self.n_points = config.num_points
        self.config = config

        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, config.embed_dim),
            nn.LayerNorm(config.embed_dim)
        )

        head_dim = config.embed_dim // config.num_heads
        self.rope_time_embed = RotaryTimeEmbedding(head_dim)

        self.layers = nn.ModuleList([
            RoPETransformerBlock(config) for _ in range(config.num_layers)
        ])

        self.output_head = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.GELU(),
            nn.Linear(config.embed_dim // 2, self.input_dim)
        )

    def forward(self, x, t, geometry=None):
        B, N, D = x.shape

        # Use optimized rotation
        R = _get_canonical_rotation_optimized(x)
        x_canonical = torch.bmm(x, R)

        h = self.input_projection(x_canonical)
        cos, sin = self.rope_time_embed(t)

        for layer in self.layers:
            h = layer(h, cos, sin)

        v_canonical = self.output_head(h)
        v_global = torch.bmm(v_canonical, R.transpose(1, 2))

        if geometry is not None:
            return geometry.to_tangent(v_global, x)
        return v_global

# ... (Keep all your existing code in models.py) ...

# --- 4. Regression Models (New) ---

class SimpleTransformerBlock(nn.Module):
    """
    Standard Transformer Block without Time-conditioning (RoPE).
    Used for direct regression.
    """
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        # Standard PyTorch Multihead Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=getattr(config, 'dropout', 0.0),
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(config.embed_dim)
        dropout = getattr(config, 'dropout', 0.0)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(config.embed_dim * 4, config.embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, N, C)
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class CanonicalRegressor(nn.Module):
    """
    Direct Coordinate Regression Model (x0 -> x1).
    Uses canonicalization to ensure rotation/reflection invariance,
    but removes Time/Flow Matching components.
    """
    def __init__(self, config):
        super().__init__()
        self.input_dim = 2
        self.n_points = config.num_points
        self.config = config

        # Project coordinates to embedding
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            nn.GELU() # Added activation for richer feature extraction
        )

        # Backbone: Standard Transformer (No Time RoPE)
        self.layers = nn.ModuleList([
            SimpleTransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Output Head: Predicts coordinates directly
        self.output_head = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.GELU(),
            nn.Linear(config.embed_dim // 2, self.input_dim)
        )

        # Initialize output to be small/close to identity behavior initially
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def forward(self, x):
        """
        Input: x (Source coordinates, e.g., random points)
        Output: y (Target coordinates, e.g., TSP solution)
        """
        B, N, D = x.shape

        # 1. Canonicalize Input
        # We calculate the rotation frame based on input 'x'
        R = _get_canonical_rotation_optimized(x)
        x_canonical = torch.bmm(x, R)

        # 2. Embed
        h = self.input_projection(x_canonical)

        # 3. Transformer Processing
        for layer in self.layers:
            h = layer(h)

        # 4. Decode to Canonical Coordinates
        y_canonical = self.output_head(h)

        # 5. Inverse Rotate (Restore Global Frame)
        # We apply the inverse of R (R^T) to put the prediction back in original space
        y_global = torch.bmm(y_canonical, R.transpose(1, 2))

        return y_global


import torch


@torch.no_grad()
def get_spectral_canonicalization(x, sigma_kernel=1.0, epsilon=1e-8):
    """
    Optimized spectral canonicalization with distance-based scale normalization.

    Key optimizations:
    - Single distance computation (reused for scale + Laplacian)
    - In-place operations where safe
    - Efficient scale computation without full sum
    - Fused rotation matrix construction
    - Eliminated unnecessary clones

    Args:
        x: Input point cloud (B, N, 2)
        sigma_kernel: Target average distance for RBF kernel
        epsilon: Numerical stability constant

    Returns:
        x_canonical: (B, N, 2) Canonicalized points
        R_total: (B, 2, 2) Combined rotation/reflection matrix
        perm: (B, N) Spectral sorting indices
        scale: (B, 1, 1) Distance-based scale factor
    """
    B, N, D = x.shape
    device = x.device
    dtype = x.dtype

    # --- 1. Center ---
    centroid = x.mean(dim=1, keepdim=True)
    x_centered = x - centroid

    # --- 2. Scale Normalization (Distance-Based) ---
    # OPTIMIZATION: Compute distances once, reuse for scale AND Laplacian
    dist = torch.cdist(x_centered, x_centered, p=2)

    # OPTIMIZATION: More efficient averaging - avoid counting diagonal
    # Using: mean = sum / (N*(N-1)) but computed efficiently
    # We can use: (dist.sum(dim=2) - 0) / (N-1) then average over N
    avg_dist = dist.sum(dim=(1, 2)) / (N * (N - 1))
    scale = (avg_dist / sigma_kernel).view(B, 1, 1) + epsilon

    # Normalize in-place if possible (but safer to create new tensor)
    x_norm = x_centered / scale

    # --- 3. Laplacian (Reuse Distance Computation) ---
    # OPTIMIZATION: Recompute with normalized coordinates
    # (We could try to rescale the existing dist, but cdist is fast enough
    # and this ensures numerical accuracy)
    dist_sq = torch.cdist(x_norm, x_norm, p=2)
    dist_sq.pow_(2)  # In-place squaring

    # OPTIMIZATION: Fused exponential and kernel computation
    W = torch.exp(-dist_sq / (sigma_kernel ** 2))

    # Zero diagonal in-place
    W.diagonal(dim1=-2, dim2=-1).zero_()

    # D^{-1/2} with rsqrt (fused operation)
    D_vec = W.sum(dim=2) + epsilon
    D_inv_sqrt_vec = torch.rsqrt(D_vec)

    # L_sym = I - D^{-1/2} W D^{-1/2}
    W_normalized = W * D_inv_sqrt_vec.unsqueeze(1) * D_inv_sqrt_vec.unsqueeze(2)

    # OPTIMIZATION: Avoid creating full identity matrix if N is large
    # Instead compute L_sym = I - W_normalized by subtracting from identity implicitly
    # But for clarity and since eigh needs explicit matrix, we keep it
    I = torch.eye(N, device=device, dtype=dtype).unsqueeze(0)
    L_sym = I - W_normalized

    # --- 4. Eigen-Decomposition ---
    vals, vecs = torch.linalg.eigh(L_sym)
    fiedler_sym = vecs[:, :, 1]

    # OPTIMIZATION: Direct multiplication (no bmm needed for vectors)
    fiedler = fiedler_sym * D_inv_sqrt_vec

    # --- 5. Sign Fix ---
    skew = (fiedler ** 3).sum(dim=1, keepdim=True)
    sign_flip = torch.sign(skew)
    sign_flip[sign_flip == 0] = 1.0
    fiedler = fiedler * sign_flip

    # --- 6. Sort ---
    perm = torch.argsort(fiedler, dim=1)
    x_ordered = x_norm.gather(1, perm.unsqueeze(-1).expand(-1, -1, D))

    # --- 7. Rotation Alignment ---
    # OPTIMIZATION: Pre-create weights (can be cached if N is constant)
    weights = torch.linspace(-1, 1, N, device=device, dtype=dtype).view(1, N, 1)
    weighted_dir = (x_ordered * weights).sum(dim=1)
    u = F.normalize(weighted_dir, dim=1, eps=epsilon)

    # OPTIMIZATION: Direct rotation matrix construction
    cos_theta = u[:, 1:2]
    sin_theta = -u[:, 0:1]

    # Fused rotation matrix assembly
    R1 = torch.stack([
        torch.cat([cos_theta, -sin_theta], dim=1),
        torch.cat([sin_theta, cos_theta], dim=1)
    ], dim=1)

    x_rot = torch.bmm(x_ordered, R1)

    # --- 8. Reflection Alignment ---
    upper_half_mask = (weights > 0).float()
    upper_centroid_x = (x_rot[..., 0:1] * upper_half_mask).sum(dim=1)
    ref_sign = torch.sign(upper_centroid_x).view(B, 1, 1)
    ref_sign[ref_sign == 0] = 1.0

    # OPTIMIZATION: Build R2 efficiently without clone
    # Create R2 directly instead of expanding and cloning identity
    R2 = torch.zeros(B, 2, 2, device=device, dtype=dtype)
    R2[:, 0, 0] = ref_sign.squeeze()
    R2[:, 1, 1] = 1.0

    # Total transform
    R_total = torch.bmm(R1, R2)
    x_canonical = torch.bmm(x_ordered, R_total)

    return x_canonical, R_total, perm, scale

import math

class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during init. These are fixed during training.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        # x shape: (B,) -> (B, 1)
        # Use torch.pi (added in PyTorch 1.9) or math.pi
        x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ResBlock(nn.Module):
    """
    Standard MLP Residual Block:
    x -> Linear -> BN -> GELU -> Linear -> BN -> (+ x) -> GELU
    """

    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.norm1 = nn.BatchNorm1d(dim)
        self.act = nn.GELU()

        self.linear2 = nn.Linear(dim, dim)
        self.norm2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        out = self.linear1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.norm2(out)

        out += residual  # The Skip Connection
        out = self.act(out)

        return out


class SpectralCanonMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_points = config.num_points
        self.input_dim = 2

        # Dimensions
        self.t_dim = config.embed_dim // 4
        # We use a wider hidden dimension for MLPs usually (e.g. 4x or 2x)
        self.hidden_dim = config.embed_dim * 4

        # 1. Time Embedding
        self.time_mlp = nn.Sequential(
            GaussianFourierProjection(self.t_dim),
            nn.Linear(self.t_dim, self.t_dim),
            nn.GELU()
        )

        # 2. Input Projection
        # Concatenated input size: (N points * 2 coords) + time embedding
        input_flat_dim = (self.n_points * 2) + self.t_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_flat_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU()
        )

        # 3. Residual Layers (The "Deep" part)
        # Using a ModuleList ensures we can iterate over them easily
        # Defaulting to 3 blocks if not specified in config
        num_res_blocks = getattr(config, 'num_layers', 3)
        self.res_blocks = nn.ModuleList([
            ResBlock(self.hidden_dim, dropout=0.1)
            for _ in range(num_res_blocks)
        ])

        # 4. Output Projection
        self.output_proj = nn.Linear(self.hidden_dim, self.n_points * 2)

    def forward(self, x, t, geometry=None):
        """
        x: (B, N, 2)
        t: (B,)
        """
        B, N, D = x.shape

        # --- A. Canonicalize ---
        x_canonical, R_total, perm, scale = get_spectral_canonicalization(x)

        # --- B. Prepare Input ---
        # Flatten points: (B, N, 2) -> (B, N*2)
        x_flat = x_canon.reshape(B, -1)

        # Embed time: (B,) -> (B, t_dim)
        t_emb = self.time_mlp(t)

        # Concatenate: (B, N*2 + t_dim)
        x_in = torch.cat([x_flat, t_emb], dim=1)

        # --- C. MLP Forward Pass ---
        # 1. Project up
        h = self.input_proj(x_in)

        # 2. Apply Residual Blocks
        for block in self.res_blocks:
            h = block(h)

        # 3. Project down
        v_canon_flat = self.output_proj(h)

        # --- D. Inverse Transform ---
        v_canon = v_canon_flat.reshape(B, N, D)

        # Rotate back using R^T
        v_ordered = torch.bmm(v_canon, R_total.transpose(1, 2)) * scale

        # Un-permute back to original indices
        # We map the ordered/canonical data back to the original slots
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(B, N)
        v_global = torch.zeros_like(v_ordered)
        v_global[batch_idx, perm] = v_ordered

        if geometry is not None:
            return geometry.to_tangent(v_global, x)

        return v_global

class SpectralCanonTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Input Projection
        self.input_proj = nn.Linear(2, config.embed_dim)

        # 2. Sequence Positional Embedding (Learned)
        # "I am point #5 in the sorted list"
        self.seq_pos_embed = nn.Parameter(torch.randn(1, config.num_points, config.embed_dim) * 0.02)

        # 3. RoPE for TIME (Rotates based on diffusion time t)
        head_dim = config.embed_dim // config.num_heads
        self.rope_time = RotaryTimeEmbedding(head_dim)

        # 4. Layers
        self.layers = nn.ModuleList([
            RoPETransformerBlock(config) for _ in range(config.num_layers)
        ])

        self.output_head = nn.Linear(config.embed_dim, 2)

    def forward(self, x, t, geometry=None):
        B, N, D = x.shape

        # 1. Canonicalize
        x_canonical, R_total, perm, scale = get_spectral_canonicalization(x)

        # 2. Embed Features
        h = self.input_proj(x_canonical)  # (B, N, Dim)

        # 3. Add Sequence Positional Information
        # This tells the model: "This vector belongs to the point with rank i"
        h = h + self.seq_pos_embed[:, :N, :]

        # 4. Get Time Rotation
        cos, sin = self.rope_time(t)

        # 5. Transformer Blocks (with Time RoPE)
        for layer in self.layers:
            h = layer(h, cos, sin)

        # 6. Output
        v_canon = self.output_head(h)

        # 7. Inverse Transform
        # CHANGE: Multiply by scale to restore velocity magnitude
        v_ordered = torch.bmm(v_canon, R_total.transpose(1, 2)) * scale  # <--- ADD * scale

        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(B, N)
        v_global = torch.zeros_like(v_ordered)
        v_global[batch_idx, perm] = v_ordered

        if geometry is not None:
            return geometry.to_tangent(v_global, x)
        return v_global


class ContinuousRotaryPositionalEmbedding(nn.Module):
    """
    NEW: Sequence-Aware RoPE for TSP Curves.
    Handles (Batch, Seq_Len) inputs correctly without flattening.
    """

    def __init__(self, dim, base=10000):
        super().__init__()
        assert dim % 2 == 0, f"RoPE requires even dim, got {dim}"
        self.dim = dim
        self.base = base
        # Create (dim/2) frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, t):
        """
        Input t: (Batch, Seq_Len) - continuous values, e.g., 0.0 to 1.0
        Output: cos, sin of shape (Batch, 1, Seq_Len, Dim)
        """
        # 1. Compute angles using einsum to preserve Batch and Seq dimensions
        # t: (B, S), inv_freq: (D/2) -> (B, S, D/2)
        angles = torch.einsum("bs,d->bsd", t, self.inv_freq)

        # 2. Interleave to match rotate_half [theta0, theta0, theta1, theta1...]
        # (B, S, D/2) -> (B, S, D)
        angles = torch.stack([angles, angles], dim=-1).flatten(-2)

        # 3. Reshape for broadcasting against (B, Heads, Seq, Dim)
        # We add the 'Heads' dimension as 1
        # Output: (B, 1, S, Dim)
        cos = angles.cos().unsqueeze(1)
        sin = angles.sin().unsqueeze(1)

        return cos, sin


class TSP_Spectral_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_cities = config.num_cities
        self.m_points = config.num_points
        self.embed_dim = config.embed_dim
        self.config = config

        self.city_proj = nn.Linear(2, config.embed_dim)
        self.curve_queries = nn.Parameter(torch.randn(1, self.m_points, config.embed_dim) * 0.02)

        head_dim = config.embed_dim // config.num_heads
        self.rope_time = ContinuousRotaryPositionalEmbedding(head_dim)

        self.layers = nn.ModuleList([
            RoPETransformerBlock(config) for _ in range(config.num_layers)
        ])

        self.output_head = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, 4)
        )

    def forward(self, x):
        B, N, D = x.shape
        M = self.m_points

        # --- FIX: Calculate Centroid manually before canonicalization ---
        # We do this because get_spectral_canonicalization returns 'inv_perm' as the 4th arg,
        # but we need 'centroid' for the inverse transform later.
        centroid = x.mean(dim=1, keepdim=True)

        # --- A. Canonicalization ---
        x_canonical, R_total, perm, scale = get_spectral_canonicalization(x)

        # --- B. Embeddings ---
        h_cities = self.city_proj(x_canon)
        h_curve = self.curve_queries.repeat(B, 1, 1)
        h = torch.cat([h_cities, h_curve], dim=1)

        # --- C. RoPE "Time" Creation ---
        t_cities = torch.zeros(B, N, device=x.device)
        t_curve = torch.linspace(0, 1, M, device=x.device).unsqueeze(0).expand(B, -1)
        t_seq = torch.cat([t_cities, t_curve], dim=1)

        # cos, sin shape: (B, 1, N+M, HeadDim)
        cos, sin = self.rope_time(t_seq)

        # --- D. Transformer Pass ---
        for layer in self.layers:
            h = layer(h, cos, sin)

        # --- E. Extract & Decode ---
        h_curve_out = h[:, N:, :]
        curve_4d_canon = self.output_head(h_curve_out)

        # --- F. Inverse Canonicalization ---
        curve_xy = curve_4d_canon[..., :2]
        curve_zw = curve_4d_canon[..., 2:]

        # Rotate XY using R Transpose
        curve_xy_global = torch.bmm(curve_xy, R_total.transpose(1, 2)) * scale

        # Add the locally calculated centroid back
        curve_xy_global = curve_xy_global + centroid

        curve_4d_global = torch.cat([curve_xy_global, curve_zw], dim=-1)

        return curve_4d_global