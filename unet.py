"""
Conditional U-Net for pilot-conditioned channel estimation.
Adapted from Imagen architecture for non-square channel data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CrossAttention
from .embeddings import PilotEncoder, TimeEmbedding

class ResBlock(nn.Module):
    """
    Residual block with time embedding injection. 
    Major Change:
        BatchNorm2d -> GroupNorm
    
    Args:
        in_channels: # of input channels
        out_channels: # of output channels
        time_emb_dim: Dimension of time embedding
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        # Time embedding projection
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
        
    def forward(self, x, time_emb):
        """
        Args:
        x: Input tensor (B, in_channels, H, W)
        time_emb: Time embedding (B, time_emb_dim)

        Returns:
            Output tensor (B, out_channels, H, W)
        """
        h = self.conv1(x)
        # Inject time embedding
        time_emb = self.time_mlp(time_emb) # out_channels
        h = h + time_emb[:, :, None, None] # Broadcast to B, C, H, W

        h = self.conv2(h)

        # Res connection
        return h + self.residual_conv(x)
    
class DownBlock(nn.Module):
    """
    Downsampling block: ResBlock + Cross-Attention + Downsample

    Args:
        in_channels: # of input channels
        out_channels: # of output channels
        time_emb_dim: Dimension of time embedding
        pilot_dim: Dimension of pilot features
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, pilot_dim):
        super().__init__()

        self.resblock = ResBlock(in_channels, out_channels, time_emb_dim)
        self.cross_attn = CrossAttention(out_channels, pilot_dim)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, time_emb, pilot_features):
        """
        Args        
        x: Input tensor (B, in_channels, H, W)
        time_emb: Time embedding (B, time_emb_dim)
        pilot_features: Pilot features (B, pilot_dim)

        Returns:
            Output tensor (B, out_channels, H//2, W//2)
        """
        h = self.resblock(x, time_emb)
        h = self.cross_attn(h, pilot_features)
        h = self.downsample(h)
        return h

class UpBlock(nn.Module):
    """
    Upsampling block: Upsample + concat skip + Res block + Cross-attention

    Args:
        in_channels: # of input channels
        skip_channels: # of skip connection channels
        out_channels: # of output channels
        time_emb_dim: Dimension of time embedding
        pilot_dim: Dimension of pilot features
    """
    def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim, pilot_dim):
        super().__init__()

        self.upsample_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.resblock = ResBlock(in_channels + skip_channels, out_channels, time_emb_dim)
        self.cross_attn = CrossAttention(out_channels, pilot_dim)

        def forward(self, x, skip, time_emb, pilot_features, target_size):
            """
            Args:
                x: Description
                skip: Description
                time_emb: Description
                pilot_features: Description
                target_size: Description
            Returns:
                Output tensor (B, out_channels, H_target, W_target)
            """
            # Upsampling
            h = F.interpolate(x, size=target_size, mode='nearest')
            h = self.upsample_conv(h)

            # Concatenate with skip connection
            h = torch.cat([h, skip], dim=1)

            # Process
            h = self.resblock(h, time_emb) # forwarding
            h = self.cross_attn(h, pilot_features) # forwarding

            return h
        

class ConditionalUNet(nn.Module):
    """
    U-Net with cross-attention conditioning on pilot signals.

    Architecture:
        - Initial Conv: 2 -> 64 channels
        - Encoder: 3 Down blocks (64 -> 128 -> 256 -> 512)
        - Bottleneck: 2 resblocks + cross-attention
        - Decoder: 3 Up blocks (512 -> 256 -> 128 -> 64)
        - Final Conv: 64 -> 2 channels
    
    Args:
        in_channels: # of input channels
        out_channels: # of output channels
        time_emb_dim: Dimension of time embedding
        pilot_dim: Dimension of pilot features
        channels: List of channel counts for each level [64, 128, 256, 512]
    """
    def __init__(
            self,
            in_channels=2,
            out_channels=2,
            time_emb_dim=512,
            pilot_dim=128,
            channels=[64, 128, 256, 512]
        ):
        super().__init__()

        self.channels = channels

        self.time_embed = TimeEmbedding(time_emb_dim)
        self.pilot_encoder = PilotEncoder(in_channels=2, out_dim=pilot_dim)

        # Init convolution
        self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        # Encoders
        self.down_block = nn.ModuleList([
            DownBlock(channels[0], channels[1], time_emb_dim, pilot_dim), # 64 -> 128
            DownBlock(channels[1], channels[2], time_emb_dim, pilot_dim), # 128 -> 256
            DownBlock(channels[2], channels[3], time_emb_dim, pilot_dim), # 256 -> 512
        ])

        # Bottleneck
        self.mid_block1 = ResBlock(channels[3], channels[3], time_emb_dim)
        self.mid_attn = CrossAttention(channels[3], pilot_dim)
        self.mid_block2 = ResBlock(channels[3], channels[3], time_emb_dim)

        # Decoder
        self.up_blocks = nn.ModuleList([
            # in_channels, skip_channels, out_channels
            UpBlock(channels[3], channels[2], channels[2], time_emb_dim, pilot_dim), # 512 + 256 -> 256
            UpBlock(channels[2], channels[1], channels[1], time_emb_dim, pilot_dim), # 256 + 128 -> 128
            UpBlock(channels[1], channels[0], channels[0], time_emb_dim, pilot_dim), # 128 + 64 -> 64
        ])

        # Final conv
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, pilot, timesteps):
        """
        Forward pass
        
        Args:
        x: Noisy channel (B, 2, 120, 14)
        pilot: pilot signal (B, 2, 18, 2)
        timesteps: Diffusion timesteps (B, ) or (B, 1)

        Returns:
            Predicted noise (B, 2, 120, 14)
        """
        # Embeddings
        time_emb = self.time_embed(timesteps) # B, time_emb_dim
        pilot_features = self.pilot_encoder(pilot) # B, pilot_dim

        # Initial Conv
        h = self.conv_in(x) # B, 64, 120, 14

        # Encoder
        skip_connections = []

        # Down 1: 64, 120, 14 -> 128, 60, 7
        h = self.down_blocks[0](h, time_emb, pilot_features)
        skip_connections.append(h)

        # Down 2: 128, 60, 7 -> 256, 30, 3
        h = self.down_blocks[1](h, time_emb, pilot_features)
        skip_connections.append(h)

        # Down 3: 256, 30, 3 -> 512, 15, 1
        h = self.down_blocks[2](h, time_emb, pilot_features)
        skip_connections.append(h)

        # Bottleneck: 512, 15, 1 -> 512, 15, 1
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h, pilot_features)
        h = self.mid_block2(h, time_emb)

        # Decoder
        # Up 1: 512, 15, 1 + skip2: 256, 30, 3 -> 256, 30, 3
        h = self.up_blocks[0](h, skip_connections[2], time_emb, pilot_features, target_size=(30, 3))

        # Up 2: 256, 30, 3 + skip1: 128, 60, 7 -> 128, 60, 7
        h = self.up_blocks[1](h, skip_connections[1], time_emb, pilot_features, target_size=(60, 7)) 

        # Up 3: 128, 60, 7 + skip0: 64, 120, 14 -> 64, 120, 14
        h = self.up_blocks[2](h, skip_connections[0], time_emb, pilot_features, target_size=(120, 14))

        # Final Conv
        out = self.conv_out(h) # B, 2, 120, 14

        return out