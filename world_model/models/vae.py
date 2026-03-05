"""Convolutional Variational Autoencoder for visual encoding."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """
    Encoder: 64x64x3 → z=32 via 4 stride-2 conv layers
    Decoder: z=32 → 64x64x3 via transposed convolutions
    """

    def __init__(self, latent_dim: int = 32, input_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 64 -> 31 -> 14 -> 6 -> 2
        self.enc1 = nn.Conv2d(input_channels, 32, 4, stride=2, padding=1)  # -> 32x32
        self.enc2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)              # -> 16x16
        self.enc3 = nn.Conv2d(64, 128, 4, stride=2, padding=1)             # -> 8x8
        self.enc4 = nn.Conv2d(128, 256, 4, stride=2, padding=1)            # -> 4x4

        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        # Decoder: 4 -> 8 -> 16 -> 32 -> 64
        self.fc_dec = nn.Linear(latent_dim, 256 * 4 * 4)
        self.dec4 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)   # -> 8x8
        self.dec3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)    # -> 16x16
        self.dec2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)     # -> 32x32
        self.dec1 = nn.ConvTranspose2d(32, input_channels, 4, stride=2, padding=1)  # -> 64x64

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input image to (mu, log_var)."""
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        h = F.relu(self.enc3(h))
        h = F.relu(self.enc4(h))
        h = h.reshape(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstructed image."""
        h = F.relu(self.fc_dec(z))
        h = h.reshape(-1, 256, 4, 4)
        h = F.relu(self.dec4(h))
        h = F.relu(self.dec3(h))
        h = F.relu(self.dec2(h))
        return torch.sigmoid(self.dec1(h))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning (reconstruction, mu, logvar)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon: torch.Tensor, target: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor, beta: float = 0.5) -> tuple[torch.Tensor, dict]:
    """VAE loss = MSE reconstruction + beta * KL divergence."""
    recon_loss = F.mse_loss(recon, target, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl_loss
    return total, {"recon_loss": recon_loss.item(), "kl_loss": kl_loss.item(), "total_loss": total.item()}
