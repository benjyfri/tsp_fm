import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier embeddings for noise levels (time).
    """
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during init. These are fixed during training.
        # We use register_buffer so they are saved with the state_dict but not updated by optimizer.
        self.register_buffer("W", torch.randn(embed_dim // 2) * scale)

    def forward(self, x):
        # x shape: (B,) -> (B, 1)
        # 2 * pi ensures the period matches the scale
        x_proj = x[:, None] * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN).
    Regresses scale (gamma) and shift (beta) from the time embedding.
    """
    def __init__(self, embed_dim, time_embed_dim):
        super().__init__()
        self.emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * embed_dim, bias=True)
        )
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)

    def forward(self, x, t_emb):
        # x: (B, N, D)
        # t_emb: (B, time_dim)

        # Predict scale and shift from time embedding
        # chunk(2) splits the output into gamma and beta
        style = self.emb(t_emb)
        gamma, beta = style.chunk(2, dim=1)

        # Reshape for broadcasting: (B, 1, D)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        # Apply normalization with adaptive parameters
        return self.norm(x) * (1 + gamma) + beta

class AdaLNTransformerBlock(nn.Module):
    """
    Transformer Block with Adaptive LayerNorm for time conditioning.
    Structure:
    x = x + Attn(AdaLN(x, t))
    x = x + MLP(AdaLN(x, t))
    """
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        # We assume the time embedding dimension passed to the block is the same as embed_dim
        # (or you can parameterize this if your global time embedding size differs)
        time_dim = config.embed_dim

        self.norm1 = AdaLN(config.embed_dim, time_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=getattr(config, 'dropout', 0.0),
            batch_first=True
        )

        self.norm2 = AdaLN(config.embed_dim, time_dim)

        dropout = getattr(config, 'dropout', 0.0)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(config.embed_dim * 4, config.embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, t_emb):
        # 1. Self-Attention with AdaLN
        x_norm = self.norm1(x, t_emb)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # 2. MLP with AdaLN
        x_norm = self.norm2(x, t_emb)
        x = x + self.mlp(x_norm)

        return x

class SpectralCanonTransformer(nn.Module):
    """
    Flow Matching Transformer using AdaLN.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Input Projection
        # TSP coordinates are 2D
        self.input_proj = nn.Linear(2, config.embed_dim)

        # 2. Sequence Positional Embedding (Learned)
        self.seq_pos_embed = nn.Parameter(torch.randn(1, config.num_points, config.embed_dim) * 0.02)

        # 3. Time Embedding Network
        # Projects scalar t -> vector embedding
        self.time_embed_dim = config.embed_dim
        self.time_mlp = nn.Sequential(
            GaussianFourierProjection(self.time_embed_dim // 4),
            nn.Linear(self.time_embed_dim // 4, self.time_embed_dim),
            nn.GELU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )

        # 4. Transformer Blocks with AdaLN
        self.layers = nn.ModuleList([
            AdaLNTransformerBlock(config) for _ in range(config.num_layers)
        ])

        # 5. Final Output Head
        # Often good practice to have a final Norm before the head, using AdaLN here too is valid
        self.final_norm = AdaLN(config.embed_dim, self.time_embed_dim)
        self.output_head = nn.Linear(config.embed_dim, 2)

        # Initialize output head to zero (DiT trick) for better convergence
        nn.init.constant_(self.output_head.weight, 0)
        nn.init.constant_(self.output_head.bias, 0)

    def forward(self, x, t, geometry=None):
        """
        x: (B, N, 2) Points
        t: (B,) Time [0, 1]
        """
        B, N, D = x.shape

        # 1. Project Input and Add Sequence Position
        h = self.input_proj(x) # (B, N, embed_dim)
        h = h + self.seq_pos_embed[:, :N, :]

        # 2. Embed Time
        t_emb = self.time_mlp(t) # (B, embed_dim)

        # 3. Apply Blocks
        for layer in self.layers:
            h = layer(h, t_emb)

        # 4. Final Norm and Output
        h = self.final_norm(h, t_emb)
        v_canon = self.output_head(h)

        if geometry is not None:
            return geometry.to_tangent(v_canon, x)
        return v_canon

# --- Legacy/Helper classes preserved and fixed below ---

class ResBlock(nn.Module):
    """
    Standard MLP Residual Block for the MLP model.
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
        out += residual
        out = self.act(out)
        return out

class SpectralCanonMLP(nn.Module):
    """
    The MLP-based vector field. Fixed indentation bugs.
    Kept closer to original but fixed GaussianFourierProjection calls.
    """
    def __init__(self, config):
        super().__init__()
        self.n_points = config.num_points

        # Dimensions
        self.t_dim = config.embed_dim // 4
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

        # 3. Residual Layers
        num_res_blocks = getattr(config, 'num_layers', 3)
        self.res_blocks = nn.ModuleList([
            ResBlock(self.hidden_dim, dropout=0.1)
            for _ in range(num_res_blocks)
        ])

        # 4. Output Projection
        self.output_proj = nn.Linear(self.hidden_dim, self.n_points * 2)

    def forward(self, x, t, geometry=None):
        B, N, D = x.shape
        # Flatten points
        x_flat = x.reshape(B, -1)
        # Embed time
        t_emb = self.time_mlp(t)
        # Concatenate
        x_in = torch.cat([x_flat, t_emb], dim=1)

        # Forward
        h = self.input_proj(x_in)
        for block in self.res_blocks:
            h = block(h)
        v_canon_flat = self.output_proj(h)

        v_canon = v_canon_flat.reshape(B, N, D)
        if geometry is not None:
            return geometry.to_tangent(v_canon, x)
        return v_canon