"""
Embedding modules for time and pilot signals.
"""

import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding with MLP projection.

    Args:
        dim: Output dimension of time embedding
    """
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

        # MLP to project sinusoidal embedding
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, timesteps):
        """
        Args:
        timesteps: Timestep values (B, ) or (B, 1)

        Returns:
            Time embeddings (B, dim)
        """
        # Ensure timesteps is 1D
        if timesteps.dim() > 1:
            timesteps = timesteps.squeeze(-1)

        # Create sinusoidal embedding
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.aragne(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        emb = self.mlp(emb)

        return emb
    
class PilotEncoder(nn.Module):
    """
    Encoder for pilot signals.
    Compresses (18, 2) pilot signal into fixed-size feature vector.

    Major change: 4 Convs -> 2 Convs

    Args:
        in_channels: # of input channels (2 for real/img - but they are both real vector representation)
        out_dim: Output feature dimension
    """
    def __init__(self, in_channels=2, out_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.self_attn = nn.MultiheadAttention(128, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(out_dim)

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, pilot):
        """       
        Args:
        pilot: Pilot signal (B, 2, 18, 2)
        """
        h = self.encoder(pilot)
        B, C, H, W = h.shape
        h_flat = h.view(B, C, H * W).permute(0, 2, 1) # B, H * W, C
        h_attn, _ = self.self_attn(h_flat, h_flat, h_flat) # B, 36, C
        h_attn = self.norm(h_attn + h_flat) # Res + Norm

        h = h_attn.permute(0, 2, 1).view(B, C, H, W) # B, 36, C -> B, C, H, W

        features = self.pool(h) # B, C, 1, 1
        features = features.view(B, -1) # B, C

        return features
