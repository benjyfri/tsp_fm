import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==========================================
# Shared Components
# ==========================================

class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier embeddings for noise levels (time).
    """

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # W shape: (embed_dim // 2)
        self.register_buffer("W", torch.randn(embed_dim // 2) * scale)

    def forward(self, x):
        # x shape: (B,) -> (B, 1)
        x_proj = x[:, None] * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SpatialFourierFeatures(nn.Module):
    """
    Fourier features for spatial coordinates (x, y).
    Used as the input encoding for the MLP model.
    """

    def __init__(self, input_dim=2, embed_dim=32, scale=10.0):
        super().__init__()
        self.register_buffer("W", torch.randn(input_dim, embed_dim // 2) * scale)

    def forward(self, x):
        # x: (B, N, 2)
        # Output: (B, N, embed_dim)
        x_proj = x @ self.W * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# ==========================================
# Rotary Positional Embeddings (RoPE)
# ==========================================

def precompute_freqs_cis(dim, end, theta=10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    # Create complex numbers: cos(m*theta) + i*sin(m*theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):
    """
    Apply RoPE to queries and keys.
    xq, xk: (B, N, num_heads, head_dim)
    freqs_cis: (N, head_dim/2)
    """
    # Reshape freqs for broadcasting: (1, N, 1, head_dim/2)
    freqs_cis = freqs_cis.view(1, xq.size(1), 1, xq.size(-1) // 2)

    # View xq/xk as complex numbers in the last dimension
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Apply rotation
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# ==========================================
# Transformer Components (AdaLN-Zero)
# ==========================================

class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization Zero (adaLN-Zero).
    Regresses scale (gamma), shift (beta), and gate (alpha) from time embedding.
    Initialized to zero to act as identity at the start of training.
    """

    def __init__(self, hidden_size, time_dim):
        super().__init__()
        self.silu = nn.SiLU()
        # Regress 6 parameters per dimension:
        # (gamma1, beta1, alpha1) for Attention
        # (gamma2, beta2, alpha2) for MLP
        self.linear = nn.Linear(time_dim, 6 * hidden_size, bias=True)

        # Zero initialization
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(self, x, t):
        # t: (B, time_dim)
        vals = self.linear(self.silu(t))  # (B, 6*D)

        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = vals.chunk(6, dim=1)

        # Reshape for broadcasting: (B, 1, D)
        def reshape_params(*args):
            return [a.unsqueeze(1) for a in args]

        params = reshape_params(gamma1, beta1, alpha1, gamma2, beta2, alpha2)
        return self.norm, params


class DiTBlockRoPE(nn.Module):
    """
    Transformer Block integrating AdaLN-Zero and RoPE.
    """

    def __init__(self, hidden_size, num_heads, time_dim, mlp_ratio=4.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.adaLN_modulation = AdaLNZero(hidden_size, time_dim)

        # Note: We use manual attention to inject RoPE easily
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

        # --- 1. Attention ---
        # Apply AdaLN
        x_norm1 = norm_layer(x) * (1 + gamma1) + beta1

        # QKV Projections
        qkv = self.qkv(x_norm1).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # (B, N, H, D)

        # Apply RoPE
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # Attention (Flash Attention if available, else standard)
        # Transpose to (B, H, N, D)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        x_attn = F.scaled_dot_product_attention(q, k, v)
        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)
        x_attn = self.attn_out(x_attn)

        # Residual with Scale Gate (alpha1)
        x = x + alpha1 * x_attn

        # --- 2. MLP ---
        x_norm2 = norm_layer(x) * (1 + gamma2) + beta2
        x_mlp = self.mlp(x_norm2)

        # Residual with Scale Gate (alpha2)
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