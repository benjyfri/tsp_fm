import torch
import torch.nn as nn
import torch.nn.functional as F


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


class TransformerBlock(nn.Module):
    """A standard Transformer encoder block using Self-Attention."""

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (B, N, embed_dim)
        attn_output, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm2(x)
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
        return vt