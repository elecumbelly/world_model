"""Linear controller: maps (z, hidden) → action. Optimized via CMA-ES."""

import numpy as np
import torch
import torch.nn as nn


class Controller(nn.Module):
    """
    Simple linear policy: action = argmax(W @ concat(z, h) + b)
    With latent_dim=32, hidden_dim=256 → 288 inputs, 4 outputs = 1,156 params
    """

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 256, action_dim: int = 4):
        super().__init__()
        self.input_dim = latent_dim + hidden_dim
        self.action_dim = action_dim
        self.fc = nn.Linear(self.input_dim, action_dim)

    def forward(self, z: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, latent_dim)
            hidden: (batch, hidden_dim)
        Returns:
            action_logits: (batch, action_dim)
        """
        x = torch.cat([z, hidden], dim=-1)
        return self.fc(x)

    def act(self, z: torch.Tensor, hidden: torch.Tensor) -> int:
        """Select action greedily (no gradient)."""
        with torch.no_grad():
            logits = self.forward(z, hidden)
            return logits.argmax(dim=-1).item()

    @property
    def num_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_params(self) -> np.ndarray:
        """Flatten all parameters to 1D numpy array."""
        params = []
        for p in self.parameters():
            params.append(p.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def set_params(self, flat_params: np.ndarray):
        """Set parameters from 1D numpy array."""
        idx = 0
        for p in self.parameters():
            n = p.numel()
            p.data = torch.tensor(
                flat_params[idx:idx + n].reshape(p.shape),
                dtype=p.dtype, device=p.device,
            )
            idx += n
