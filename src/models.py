import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Helper Classes ---

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
        return self.model(x)

class RotaryTimeEmbedding(nn.Module):
    """RoPE logic."""
    def __init__(self, head_dim, base=10000):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires even head_dim"
        self.head_dim = head_dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, t):
        angles = torch.outer(t, self.inv_freq)
        angles = torch.stack([angles, angles], dim=-1).flatten(-2)
        cos = angles.cos().view(t.shape[0], 1, 1, -1)
        sin = angles.sin().view(t.shape[0], 1, 1, -1)
        return cos, sin

def rotate_half(x):
    x1 = x[..., ::2]; x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)

def apply_rope(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

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


# --- Main Models ---

class VectorFieldModel(nn.Module):
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
    Canonical Vector field using an MLP backbone.
    """
    def __init__(self, config):
        super().__init__()
        self.n_points = config.num_points
        self.input_dim = 2

        self.time_embed = TimeEmbedding(config.t_emb_dim)

        # (N points + 1 CoM point) * 2 dims + time
        mlp_input_size = (self.n_points + 1) * self.input_dim + config.t_emb_dim
        mlp_output_size = self.n_points * self.input_dim

        self.mlp = ConfigurableMLP(
            input_size=mlp_input_size,
            num_hidden_layers=config.num_layers,
            hidden_dim=config.embed_dim,
            output_size=mlp_output_size
        )

    def _canonicalize(self, x, epsilon=1e-8):
        B, N, D = x.shape
        magnitudes = torch.norm(x, dim=2)
        max_mag_indices = torch.argmax(magnitudes, dim=1)
        p_n = torch.gather(x, 1, max_mag_indices.view(B, 1, 1).expand(-1, -1, 2)).squeeze(1)

        p_n_norm = torch.norm(p_n, dim=1, keepdim=True) + epsilon
        px_norm = p_n[:, 0:1] / p_n_norm
        py_norm = p_n[:, 1:2] / p_n_norm

        # FIX: Ensure scalars for batch (B,) without redundant squeeze
        cos_alpha = py_norm.view(-1)
        sin_alpha = px_norm.view(-1)

        R1 = torch.zeros(B, 2, 2, device=x.device, dtype=x.dtype)
        # FIX: Removed redundant .squeeze(-1)
        R1[:, 0, 0] = cos_alpha
        R1[:, 0, 1] = sin_alpha
        R1[:, 1, 0] = -sin_alpha
        R1[:, 1, 1] = cos_alpha

        rotated_x = torch.bmm(x, R1)

        dists = torch.norm(x - p_n.unsqueeze(1), dim=2)
        dists.scatter_(1, max_mag_indices.unsqueeze(1), float('inf'))
        closest_indices = torch.argmin(dists, dim=1)
        p_x_rotated = torch.gather(rotated_x, 1, closest_indices.view(B, 1, 1).expand(-1, -1, 2)).squeeze(1)

        reflection_mask = torch.where(p_x_rotated[:, 0] < 0, -1.0, 1.0).to(x.dtype)
        R2 = torch.eye(2, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B, 1, 1)
        R2[:, 0, 0] = reflection_mask

        return torch.bmm(rotated_x, R2)

    def forward(self, x, t, geometry=None):
        B, N, D = x.shape
        x_canonical = self._canonicalize(x)
        com_point = torch.zeros(B, 1, D, device=x.device, dtype=x.dtype)
        x_full = torch.cat([x_canonical, com_point], dim=1)

        t_emb = self.time_embed(t)
        x_flat = x_full.reshape(B, -1)
        mlp_input = torch.cat([x_flat, t_emb], dim=1)

        mlp_output = self.mlp(mlp_input)
        v_raw = mlp_output.reshape(B, N, D)

        if geometry is not None:
            return geometry.to_tangent(v_raw, x)
        return v_raw

class CanonicalRoPEVectorField(nn.Module):
    """
    Canonical Vector field using RoPE Transformer backbone.
    """
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
        self.layers = nn.ModuleList([RoPETransformerBlock(config) for _ in range(config.num_layers)])

        self.output_head = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.GELU(),
            nn.Linear(config.embed_dim // 2, self.input_dim)
        )
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def _get_canonical_rotation(self, x, epsilon=1e-8):
        B, N, D = x.shape
        magnitudes = torch.norm(x, dim=2)
        max_mag_indices = torch.argmax(magnitudes, dim=1)
        p_n = torch.gather(x, 1, max_mag_indices.view(B, 1, 1).expand(-1, -1, 2)).squeeze(1)

        p_n_norm = torch.norm(p_n, dim=1, keepdim=True) + epsilon
        px_norm = p_n[:, 0:1] / p_n_norm
        py_norm = p_n[:, 1:2] / p_n_norm

        # FIX: Ensure scalars for batch (B,) without redundant squeeze
        cos_alpha = py_norm.view(-1)
        sin_alpha = px_norm.view(-1)

        R1 = torch.zeros(B, 2, 2, device=x.device, dtype=x.dtype)
        # FIX: Removed redundant .squeeze(-1)
        R1[:, 0, 0] = cos_alpha
        R1[:, 0, 1] = sin_alpha
        R1[:, 1, 0] = -sin_alpha
        R1[:, 1, 1] = cos_alpha

        x_rot = torch.bmm(x, R1)
        dists = torch.norm(x - p_n.unsqueeze(1), dim=2)
        dists.scatter_(1, max_mag_indices.unsqueeze(1), float('inf'))
        closest_indices = torch.argmin(dists, dim=1)
        p_x_rotated = torch.gather(x_rot, 1, closest_indices.view(B, 1, 1).expand(-1, -1, 2)).squeeze(1)

        reflection_mask = torch.where(p_x_rotated[:, 0] < 0, -1.0, 1.0).to(x.dtype)
        R2 = torch.eye(2, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B, 1, 1)
        R2[:, 0, 0] = reflection_mask

        return torch.bmm(R1, R2)

    def forward(self, x, t, geometry=None):
        B, N, D = x.shape

        # 1. Get R and rotate to Canonical
        R = self._get_canonical_rotation(x)
        x_canonical = torch.bmm(x, R)

        # 2. Forward pass
        h = self.input_projection(x_canonical)
        cos, sin = self.rope_time_embed(t)
        for layer in self.layers:
            h = layer(h, cos, sin)
        v_canonical = self.output_head(h)

        # 3. Un-rotate Velocity to Global Frame
        v_global = torch.bmm(v_canonical, R.transpose(1, 2))

        if geometry is not None:
            return geometry.to_tangent(v_global, x)
        return v_global