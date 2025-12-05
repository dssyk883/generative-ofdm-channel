"""
attention.py
Cross-Attention for pilot conditioning
"""
import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    """
    Cross-Attention layer for conditioning on pilot features.
    Query: current channel features
    Key/Value: pilot encoder features

    Args:
        dim: Dimension of input features
        context_dim: Dimension of conditions - pilot features
        num_heads: Number of attention heads
    """
    def __init__(self, dim, context_dim, num_heads=8):
        # 18x2 -> 128 dim compression
        super().__init__()
        # 8 parallel attentions
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(context_dim, dim)
        self.to_v = nn.Linear(context_dim, dim)
        self.to_out = nn.Linear(dim, dim)

        # Layer norm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GeLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, context):
        """
        FFN
        
        Args:
        x: Input features (B, C, H, W)
        context: Conditioning context (B, context_dim)

        Returns:
            Output features (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Reshape for attention: (B, C, H, W) -> (B, H * W, C)
        # e.g.) down1: res -> cross attention: B, 128, 120, 14
        x_flat = x.view(B, C, H * W).transpose(1, 2) # (B, H * W, C)

        # Store residual
        residual = x_flat

        x_norm = self.norm1(x_flat)

        # Expand context for attention: (B, context_dim) -> (B, 1, context_dim)
        context = context.unsqueeze(1)

        # Linear projection
        q = self.to_q(x_norm) # B, H * W, C
        k = self.to_k(context) # B, 1, C
        v = self.to_v(context) # B, 1, C

        # Reshape for multi-head attention
        q = q.view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2) # B, heads, H * W, head_dim
        k = k.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2) # B, heads, 1, head_dim
        v = v.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2) # B, heads, 1, head_dim

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale # B, heads, H * W, 1
        attn = attn.softmax(dim=-1)

        # Apply attention to values
        out = attn @ v # B, heads, H * W, head_dim
        
        # Reshape out
        out = out.transpose(1, 2).contiguous().view(B, H * W, C) # B, H * W, heads, head_dim -> B, H*W, C
        out = self.to_out(out)

        out = out + residual

        # FFN with res
        out = out + self.ffn(self.norm2(out))

        # Reshape back to spatial: B, H * W, C -> B, C, H, W
        out = out.permute(0, 2, 1).view(B, C, H, W)

        return out