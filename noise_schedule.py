"""
Noise scheduling utilities for DDPM: cosine beta schedule as proposed in https://arxiv.org/abs/2102.09672
"""

import torch
import numpy as np

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Args:
    timesteps: # of diffusion steps
    s: Small offset to prevent beta from being too small at t=0

    Returns:
        betas: Beta values for each timestep (timesteps,)
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class NoiseScheduler:
    """
    Handles noise scheduling for DDPM training and inference.

    Args:
        num_timesteps: # of diffusion timesteps
        schedule_type: Type of noise schedule ('cosine' or 'linear')
    """
    def __init__(self, num_timesteps=1000):
        self.num_timesteps = num_timesteps
        betas = cosine_beta_schedule(num_timesteps)

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])

        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def add_noise(self, x_0, t, noise=None):
        """
        Add noise to clean data using forward diffusion process
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

        Args:        
            x_0: Clean data (B, C, H, W)
            t: Timesteps (B, )
            noise: Optional pre-generated noise (B, C, H, W)

        Returns:
            Noisy data x_t and the noise used
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        x_t = sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise

        return x_t, noise
    
    def denoise_step(self, x_t, predicted_noise, t):
        """
        Single denoising step during inference.
        Sample from p(x_{t-1} | x_t) using predicted noise.

        Args:
            x_t: Noisy data at timestep t (B, C, H, W)
            predicted_noise: Noise predicted by model (B, C, H, W)
            t: Current timestep (scalar)
        """
        # Extract values for this timestep
        alpha_t = self.alphas[t]
        alpha_prod_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]

        # Predict x_0
        pred_x_0 = (x_t - torch.sqrt(1 - alpha_prod_t) * predicted_noise) / torch.sqrt(alpha_prod_t)

        # Clip predicted x_0 for stability
        pred_x_0 = torch.clmap(pred_x_0, -1, 1)

        # Calculate x_{t-1}
        if t > 0:
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt(self.posterior_variance[t])
        else:
            noise = 0
            sigma_t = 0

        x_t_minus_1 = (
            torch.sqrt(self.alphas_cumprod_prev[t]) * beta_t / (1 - alpha_prod_t) * pred_x_0 +
            torch.sqrt(alpha_t) * (1 - self.alphas_cumprod_prev[t]) / (1 - alpha_prod_t) * x_t + 
            sigma_t * noise
        )

        return x_t_minus_1
    
    def to(self, device):
        """Move all tensors to specified device"""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self