import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --- 1. Deterministic Fourier Features ---
class SpatialFourierFeatures(nn.Module):
    def __init__(self, input_dim, output_dim, scale=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scale = scale

        # FIX: Deterministic seed for the projection basis so runs are reproducible
        g_cpu = torch.Generator()
        g_cpu.manual_seed(42)
        self.register_buffer(
            "W", torch.randn(input_dim, output_dim // 2, generator=g_cpu) * scale
        )

    def forward(self, x):
        # x: (B, N, 2)
        proj = x @ self.W  # (B, N, D/2)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


# --- 2. RoPE Helpers ---
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(xq, xk, freqs_cis):
    """Apply RoPE using real-valued arithmetic."""
    cos, sin = freqs_cis.chunk(2, dim=-1)
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out, xk_out


# --- 3. Equivariant Block ---
class EquivariantBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # FIX: Non-affine LayerNorms to preserve strict equivariance
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.attn_out = nn.Linear(hidden_size, hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size)
        )

    def forward(self, x, freqs_cis):
        B, N, C = x.shape

        # Self-Attention Branch
        x_norm1 = self.norm1(x)
        qkv = self.qkv(x_norm1).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        x_attn = F.scaled_dot_product_attention(q, k, v)
        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)

        x = x + self.attn_out(x_attn)

        # MLP Branch
        x_norm2 = self.norm2(x)
        x = x + self.mlp(x_norm2)
        return x


# --- 4. The Angle Regressor Model ---
class EquivariantAngleRegressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.embed_dim
        self.num_heads = config.num_heads

        head_dim = dim // self.num_heads
        assert head_dim % 8 == 0, f"head_dim ({head_dim}) must be divisible by 8 for 4-signal RoPE"

        self.num_signals = 4
        self.pairs_per_signal = (head_dim // 2) // self.num_signals

        # Input embedding
        self.input_proj = SpatialFourierFeatures(2, dim, scale=1.0)
        self.input_mixer = nn.Linear(dim, dim)

        # Transformer
        self.layers = nn.ModuleList(
            [EquivariantBlock(dim, self.num_heads) for _ in range(config.num_layers)]
        )

        # Final Norm (also non-affine)
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)

        # ARCHITECTURE CHANGE: Predict 2D vector (x, y) per point
        # We do NOT normalize here. The loss function will handle the manifold projection.
        self.output_head = nn.Linear(dim, 2)

        # RoPE Base
        rope_base_dim = self.pairs_per_signal * 2

        self.register_buffer(
            "freqs_base",
            1.0 / (10000.0 ** (torch.arange(0, rope_base_dim, 2).float() / rope_base_dim)),
            persistent=False,
        )

    def compute_frequencies(self, signals):
        B, N, S = signals.shape
        assert S == self.num_signals

        bands = []
        for s in range(self.num_signals):
            base = signals[..., s]
            theta = base[..., None] * self.freqs_base
            cos = torch.cos(theta)
            sin = torch.sin(theta)
            # Repeat to match real-valued RoPE format (x1, x1, x2, x2...)
            cos = torch.cat([cos, cos], dim=-1)
            sin = torch.cat([sin, sin], dim=-1)
            bands.append((cos, sin))

        cos_full = torch.cat([b[0] for b in bands], dim=-1)
        sin_full = torch.cat([b[1] for b in bands], dim=-1)

        # SAFETY: Assert dimensions match before squeeze
        expected_width = self.config.embed_dim // self.config.num_heads
        assert cos_full.shape[-1] == expected_width, \
            f"RoPE Width Mismatch: Got {cos_full.shape[-1]}, Expected {expected_width}"

        cos_full = cos_full.unsqueeze(2)
        sin_full = sin_full.unsqueeze(2)
        return torch.cat([cos_full, sin_full], dim=-1)

    def forward(self, x, static_signals):
        """
        x: (B, N, 2) Input coordinates
        static_signals: (B, N, 4) Geometric signals
        """
        freqs_cis = self.compute_frequencies(static_signals)
        B, N, _ = x.shape
        density_scale = math.sqrt(N)

        h = self.input_proj(x * density_scale)
        h = self.input_mixer(h)

        for layer in self.layers:
            h = layer(h, freqs_cis)

        h = self.final_norm(h)

        # Output raw logits (B, N, 2)
        vectors = self.output_head(h)
        return vectors