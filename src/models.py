import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==========================================
# Shared Components
# ==========================================

class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        ### CHANGED: Deterministic seeding for reproducibility
        g_cpu = torch.Generator()
        g_cpu.manual_seed(42)
        self.register_buffer("W", torch.randn(embed_dim // 2, generator=g_cpu) * scale)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# ==========================================
# Rotary Positional Embeddings (RoPE)
# ==========================================

def precompute_freqs_cis(dim, end, theta=10000.0):
    """
    Precompute Cos/Sin for Interleaved RoPE (Backward Compatible).
    Returns: Tensor shape (end, dim/2, 2) -> [cos, sin]
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # (end, dim/2)

    # Stack Cos and Sin in the last dimension
    # Shape: (end, dim/2, 2)
    return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
# ==========================================
# Transformer Components (AdaLN-Zero)
# ==========================================


class DiTBlockRoPE(nn.Module):
    """
    Transformer Block integrating AdaLN-Zero and RoPE.
    """

    def __init__(self, hidden_size, num_heads, time_dim, mlp_ratio=4.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.adaLN_modulation = AdaLNZero(hidden_size, time_dim)

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.attn_out = nn.Linear(hidden_size, hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size)
        )

    # DELETE apply_rotary_emb_old (Make sure it is gone!)

    def apply_rotary_emb_compatible(self, x, freqs_cis):
        """
        Apply Interleaved RoPE using Real Math (Compiler Friendly).
        x: (B, N, H, D)
        freqs_cis: (N, D/2, 2) containing [cos, sin]
        """
        # 1. Reshape x to isolate the adjacent pairs: (B, N, H, D/2, 2)
        # This matches the 'view_as_complex' layout
        x_shaped = x.view(*x.shape[:-1], -1, 2)

        # 2. Reshape frequencies to broadcast
        # Input: (N, D/2, 2) -> Target: (1, N, 1, D/2, 2)
        freqs = freqs_cis.view(1, x.shape[1], 1, -1, 2)
        cos = freqs[..., 0]
        sin = freqs[..., 1]

        # 3. Split input into Real (even indices) and Imag (odd indices) parts
        x_real = x_shaped[..., 0]
        x_imag = x_shaped[..., 1]

        # 4. Apply Rotation Formula (Real-valued complex multiplication)
        # (a + ib)(cos + isin) = (a*cos - b*sin) + i(a*sin + b*cos)
        out_real = x_real * cos - x_imag * sin
        out_imag = x_real * sin + x_imag * cos

        # 5. Stack back together and flatten
        return torch.stack([out_real, out_imag], dim=-1).flatten(-2)

    def forward(self, x, t, freqs_cis):
        B, N, C = x.shape
        norm_layer, (gamma1, beta1, alpha1, gamma2, beta2, alpha2) = self.adaLN_modulation(x, t)

        x_norm1 = norm_layer(x) * (1 + gamma1) + beta1
        qkv = self.qkv(x_norm1).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        # --- USE THE COMPATIBLE FUNCTION ---
        q = self.apply_rotary_emb_compatible(q, freqs_cis)
        k = self.apply_rotary_emb_compatible(k, freqs_cis)
        # -----------------------------------

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        x_attn = F.scaled_dot_product_attention(q, k, v)
        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)
        x_attn = self.attn_out(x_attn)

        x = x + alpha1 * x_attn
        x_norm2 = norm_layer(x) * (1 + gamma2) + beta2
        x_mlp = self.mlp(x_norm2)
        x = x + alpha2 * x_mlp

        return x


class SpectralCanonTransformer(nn.Module):
    """
    Flow Matching Transformer using:
    1. AdaLN-Zero for Time Injection.
    2. RoPE for Spatial/Sequence Encoding.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.embed_dim

        # Time Embedding Network
        self.time_embed_dim = dim
        self.time_mlp = nn.Sequential(
            GaussianFourierProjection(dim // 4),
            nn.Linear(dim // 4, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

        # Input Projection (Points -> Embeddings)
        self.input_proj = nn.Linear(2, dim)

        # RoPE Frequencies
        head_dim = dim // config.num_heads
        # Precompute for max points. persistent=False so it's not saved in state_dict (it's fixed)
        self.register_buffer("freqs_cis", precompute_freqs_cis(head_dim, config.num_points), persistent=False)

        # Transformer Blocks
        self.layers = nn.ModuleList([
            DiTBlockRoPE(dim, config.num_heads, self.time_embed_dim)
            for _ in range(config.num_layers)
        ])

        # Final Output Logic
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Linear(self.time_embed_dim, 2 * dim, bias=True)
        nn.init.constant_(self.final_adaLN.weight, 0)
        nn.init.constant_(self.final_adaLN.bias, 0)

        self.output_head = nn.Linear(dim, 2)
        # Zero init output head for pure noise prediction start
        nn.init.constant_(self.output_head.weight, 0)
        nn.init.constant_(self.output_head.bias, 0)

    def forward(self, x, t, geometry=None):
        B, N, D = x.shape

        # 1. Embed Time
        t_emb = self.time_mlp(t)

        # 2. Embed Spatial Inputs
        h = self.input_proj(x)

        # 3. Apply Blocks with RoPE
        # 3. Apply Blocks with RoPE
        if N > self.freqs_cis.shape[0]:
            # FIX: Get head_dim directly from the first layer to ensure consistency
            # with the loaded weights, ignoring potentially mismatched 'config' args.
            head_dim = self.layers[0].head_dim

            # Generate new frequencies
            new_freqs = precompute_freqs_cis(head_dim, N).to(x.device)

            # UPDATE THE OBJECT STATE
            self.freqs_cis = new_freqs

        # Slice freqs_cis to current sequence length
        freqs_cis = self.freqs_cis[:N]
        # Slice freqs_cis to current sequence length
        freqs_cis = self.freqs_cis[:N]

        for layer in self.layers:
            h = layer(h, t_emb, freqs_cis)

        # 4. Final Norm and Output
        # AdaLN for final layer (usually just scale/shift, no gate)
        style = self.final_adaLN(F.silu(t_emb))
        gamma, beta = style.chunk(2, dim=1)
        h = self.final_norm(h) * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        v_canon = self.output_head(h)

        if geometry is not None:
            return geometry.to_tangent(v_canon, x)
        return v_canon


# ==========================================
# MLP Components (FiLM + Fourier Inputs)
# ==========================================

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).
    Scales and shifts features based on time embedding.
    """

    def __init__(self, in_dim, time_dim):
        super().__init__()
        self.style = nn.Linear(time_dim, 2 * in_dim)
        # Initialize close to identity
        nn.init.constant_(self.style.weight, 0)
        nn.init.constant_(self.style.bias, 0)

    def forward(self, x, t_emb):
        # x: (B, D)
        # t_emb: (B, time_dim)
        style = self.style(t_emb)
        gamma, beta = style.chunk(2, dim=1)
        # x * (1 + gamma) + beta
        return x * (1 + gamma) + beta


class ResFiLMBlock(nn.Module):
    """
    Residual Block with FiLM modulation.
    """

    def __init__(self, dim, time_dim, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.norm1 = nn.BatchNorm1d(dim)
        self.act = nn.GELU()
        self.film1 = FiLMLayer(dim, time_dim)

        self.linear2 = nn.Linear(dim, dim)
        self.norm2 = nn.BatchNorm1d(dim)
        self.film2 = FiLMLayer(dim, time_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t_emb):
        residual = x

        out = self.linear1(x)
        out = self.norm1(out)
        out = self.film1(out, t_emb)  # Inject time via FiLM
        out = self.act(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.norm2(out)
        out = self.film2(out, t_emb)  # Inject time via FiLM

        out += residual
        out = self.act(out)
        return out


class SpectralCanonMLP(nn.Module):
    """
    Flow Matching MLP using:
    1. Spatial Fourier Features for input coordinates.
    2. FiLM for Time Injection.
    """

    def __init__(self, config):
        super().__init__()
        self.n_points = config.num_points
        self.hidden_dim = config.embed_dim * 4

        # Time Embedding
        self.time_embed_dim = config.embed_dim
        self.time_mlp = nn.Sequential(
            GaussianFourierProjection(self.time_embed_dim // 4),
            nn.Linear(self.time_embed_dim // 4, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )

        # Spatial Fourier Encoding
        # Maps 2D coords -> fourier_dim (e.g., 64)
        self.fourier_dim = 64
        self.spatial_enc = SpatialFourierFeatures(2, self.fourier_dim)

        # Input Projection
        # Input is flattened: N * fourier_dim
        input_flat_dim = self.n_points * self.fourier_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_flat_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU()
        )

        # ResNet Backbone with FiLM
        self.blocks = nn.ModuleList([
            ResFiLMBlock(self.hidden_dim, self.time_embed_dim, dropout=0.1)
            for _ in range(getattr(config, 'num_layers', 3))
        ])

        # Output Projection
        self.output_proj = nn.Linear(self.hidden_dim, self.n_points * 2)

    def forward(self, x, t, geometry=None):
        B, N, D = x.shape

        # 1. Embed Time
        t_emb = self.time_mlp(t)
        t_emb = nn.SiLU()(t_emb)

        # 2. Fourier Encode Spatial Coords
        # x: (B, N, 2) -> (B, N, fourier_dim)
        x_enc = self.spatial_enc(x)

        # 3. Flatten for Global MLP
        x_flat = x_enc.reshape(B, -1)

        # 4. Backbone
        h = self.input_proj(x_flat)
        for block in self.blocks:
            h = block(h, t_emb)

        v_flat = self.output_proj(h)
        v_canon = v_flat.reshape(B, N, D)

        if geometry is not None:
            return geometry.to_tangent(v_canon, x)
        return v_canon

class SpatialFourierFeatures(nn.Module):
    def __init__(self, input_dim=2, embed_dim=32, scale=1.0):
        super().__init__()
        ### CHANGED: Deterministic seeding
        g_cpu = torch.Generator()
        g_cpu.manual_seed(1337)
        self.register_buffer("W", torch.randn(input_dim, embed_dim // 2, generator=g_cpu) * scale)

    def forward(self, x):
        x_proj = x @ self.W * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class AdaLNZero(nn.Module):
    def __init__(self, hidden_size, time_dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(time_dim, 6 * hidden_size, bias=True)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(self, x, t):
        vals = self.linear(self.silu(t))
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = vals.chunk(6, dim=1)
        params = [p.unsqueeze(1) for p in (gamma1, beta1, alpha1, gamma2, beta2, alpha2)]
        return self.norm, params


### NEW: Helper for Real-Valued RoPE
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


### CHANGED: Replaced Complex logic with Real logic
def apply_rotary_emb(xq, xk, freqs_cis):
    """
    Apply RoPE using real-valued arithmetic (safer for AMP).

    Args:
        xq, xk: (B, N, H, D)
        freqs_cis: (B, N, 1, D) -- Contains [cos, sin] concatenated
    """
    # freqs_cis is technically [cos, sin] here from the compute_frequencies function below
    cos, sin = freqs_cis.chunk(2, dim=-1)

    # Standard RoPE formula: (x * cos) + (rotate_half(x) * sin)
    # This works entirely in the native precision (float16/bfloat16) without casting
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)

    return xq_out, xk_out
# --- The New Equivariant Model ---
# ==========================================
# 3. Equivariant Transformer
# ==========================================

class EquivariantDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, time_dim, mlp_ratio=4.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.adaLN_modulation = AdaLNZero(hidden_size, time_dim)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.attn_out = nn.Linear(hidden_size, hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size)
        )

    def forward(self, x, t, freqs_cis):
        B, N, C = x.shape
        norm_layer, (gamma1, beta1, alpha1, gamma2, beta2, alpha2) = self.adaLN_modulation(x, t)

        x_norm1 = norm_layer(x) * (1 + gamma1) + beta1
        qkv = self.qkv(x_norm1).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        # Apply Real-Valued RoPE
        q, k = apply_rotary_emb(q, k, freqs_cis)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        x_attn = F.scaled_dot_product_attention(q, k, v)
        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)

        x = x + alpha1 * self.attn_out(x_attn)
        x_norm2 = norm_layer(x) * (1 + gamma2) + beta2
        x = x + alpha2 * self.mlp(x_norm2)
        return x


class EquivariantDiffTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.embed_dim
        self.num_heads = config.num_heads

        head_dim = dim // self.num_heads
        assert head_dim % 8 == 0, \
            "head_dim must be divisible by 8 (2 * num_signals)"

        self.num_signals = 4
        self.pairs_per_signal = (head_dim // 2) // self.num_signals
        self.band_dim = self.pairs_per_signal * 2

        # --- Time embedding ---
        self.time_embed_dim = dim
        self.time_mlp = nn.Sequential(
            GaussianFourierProjection(dim // 4),
            nn.Linear(dim // 4, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # --- Input embedding ---
        self.input_proj = SpatialFourierFeatures(2, dim, scale=1.0)
        self.input_mixer = nn.Linear(dim, dim)

        # --- Transformer ---
        self.layers = nn.ModuleList(
            [EquivariantDiTBlock(dim, self.num_heads, dim) for _ in range(config.num_layers)]
        )

        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_adaLN = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.final_adaLN.weight)
        nn.init.zeros_(self.final_adaLN.bias)

        self.output_head = nn.Linear(dim, 2)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

        # --- Shared RoPE frequency base (correct size) ---
        rope_base_dim = self.pairs_per_signal * 2
        self.register_buffer(
            "freqs_base",
            1.0 / (10000.0 ** (torch.arange(0, rope_base_dim, 2).float() / rope_base_dim)),
            persistent=False,
        )
    def compute_frequencies(self, signals):
        """
        signals: (B, N, 4) = [radius, sinθ, cosθ, hull_dist]

        Returns:
        freqs_cis: (B, N, 1, 2 * head_dim)
        """
        B, N, S = signals.shape
        assert S == self.num_signals

        device = signals.device
        bands = []

        for s in range(self.num_signals):
            base = signals[..., s]  # (B, N)

            theta = base[..., None] * self.freqs_base  # (B, N, pairs_per_signal)

            cos = torch.cos(theta)
            sin = torch.sin(theta)

            cos = torch.cat([cos, cos], dim=-1)
            sin = torch.cat([sin, sin], dim=-1)

            bands.append((cos, sin))

        cos_full = torch.cat([b[0] for b in bands], dim=-1)
        sin_full = torch.cat([b[1] for b in bands], dim=-1)

        cos_full = cos_full.unsqueeze(2)
        sin_full = sin_full.unsqueeze(2)

        return torch.cat([cos_full, sin_full], dim=-1)
    def forward(self, x, t, static_signals=None, precomputed_freqs=None, geometry=None):
        if precomputed_freqs is not None:
            freqs_cis = precomputed_freqs
        else:
            if static_signals is None:
                raise ValueError("static_signals required if freqs not precomputed")
            freqs_cis = self.compute_frequencies(static_signals)

        B, N, _ = x.shape
        density_scale = math.sqrt(N)

        t_emb = self.time_mlp(t)
        h = self.input_proj(x * density_scale)
        h = self.input_mixer(h)

        for layer in self.layers:
            h = layer(h, t_emb, freqs_cis)

        gamma, beta = self.final_adaLN(F.silu(t_emb)).chunk(2, dim=-1)
        h = self.final_norm(h) * (1 + gamma[:, None]) + beta[:, None]

        v = self.output_head(h)
        if geometry is not None:
            return geometry.to_tangent(v, x)
        return v




class VectorFieldModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    def forward(self, x, t, geometry=None):
        return x

class RoPEVectorFieldModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    def forward(self, x, t, geometry=None):
        return x

class CanonicalMLPVectorField(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    def forward(self, x, t, geometry=None):
        return x

class CanonicalMLPVectorField(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    def forward(self, x, t, geometry=None):
        return x

class CanonicalRoPEVectorField(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    def forward(self, x, t, geometry=None):
        return x

class CanonicalRegressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    def forward(self, x, t, geometry=None):
        return x


# ==========================================
# Simple Equivariant Block
# ==========================================

class EquivariantDiTBlockSimple(nn.Module):
    def __init__(self, hidden_size, num_heads, time_dim, mlp_ratio=4.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.adaLN_modulation = AdaLNZero(hidden_size, time_dim)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.attn_out = nn.Linear(hidden_size, hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size)
        )

    # ### CHANGED: Removed freqs_cis argument
    def forward(self, x, t):
        B, N, C = x.shape
        norm_layer, (gamma1, beta1, alpha1, gamma2, beta2, alpha2) = self.adaLN_modulation(x, t)

        x_norm1 = norm_layer(x) * (1 + gamma1) + beta1
        qkv = self.qkv(x_norm1).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        # ### CHANGED: Removed apply_rotary_emb (RoPE)
        # No rotation logic here, just standard Attention preparation
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        x_attn = F.scaled_dot_product_attention(q, k, v)
        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)

        x = x + alpha1 * self.attn_out(x_attn)
        x_norm2 = norm_layer(x) * (1 + gamma2) + beta2
        x = x + alpha2 * self.mlp(x_norm2)
        return x


# ==========================================
# Simple Equivariant Transformer
# ==========================================

class EquivariantDiffTransformerSimple(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.embed_dim
        self.num_heads = config.num_heads

        # ### CHANGED: Removed all RoPE specific dimension checks and signal definitions
        # (Removed num_signals, pairs_per_signal, band_dim checks)

        # --- Time embedding ---
        self.time_embed_dim = dim
        self.time_mlp = nn.Sequential(
            GaussianFourierProjection(dim // 4),
            nn.Linear(dim // 4, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # --- Input embedding ---
        # ### CHANGED: Replaced SpatialFourierFeatures with simple Linear projection
        # "PE is only the projection of the coordinates"
        self.input_proj = nn.Linear(2, dim)
        self.input_mixer = nn.Linear(dim, dim)

        # --- Transformer ---
        # ### CHANGED: Using EquivariantDiTBlockSimple
        self.layers = nn.ModuleList(
            [EquivariantDiTBlockSimple(dim, self.num_heads, dim) for _ in range(config.num_layers)]
        )

        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_adaLN = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.final_adaLN.weight)
        nn.init.zeros_(self.final_adaLN.bias)

        self.output_head = nn.Linear(dim, 2)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

        # ### CHANGED: Removed freqs_base buffer entirely

    # ### CHANGED: Removed compute_frequencies method entirely

    # ### CHANGED: Removed static_signals and precomputed_freqs from forward
    def forward(self, x, t, geometry=None):
        B, N, _ = x.shape
        # density_scale = math.sqrt(N) # Optional: Can keep or remove scaling depending on preference, removing for simplicity

        t_emb = self.time_mlp(t)

        # ### CHANGED: Simple projection of coordinates
        h = self.input_proj(x)
        h = self.input_mixer(h)

        for layer in self.layers:
            # ### CHANGED: Removed freqs_cis argument passed to layers
            h = layer(h, t_emb)

        gamma, beta = self.final_adaLN(F.silu(t_emb)).chunk(2, dim=-1)
        h = self.final_norm(h) * (1 + gamma[:, None]) + beta[:, None]

        v = self.output_head(h)
        if geometry is not None:
            return geometry.to_tangent(v, x)
        return v