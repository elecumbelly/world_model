"""GRU-based recurrent world model for next-state prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldRNN(nn.Module):
    """
    Takes (z_t, action_t) → predicts (z_{t+1}, reward, done).
    Uses a GRU for temporal modeling.
    """

    def __init__(self, latent_dim: int = 32, action_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        input_dim = latent_dim + action_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

        self.fc_z = nn.Linear(hidden_dim, latent_dim)
        self.fc_reward = nn.Linear(hidden_dim, 1)
        self.fc_done = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor, action: torch.Tensor,
                hidden: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (batch, seq_len, latent_dim) - latent observations
            action: (batch, seq_len, action_dim) - one-hot actions
            hidden: (1, batch, hidden_dim) - GRU hidden state

        Returns:
            next_z: (batch, seq_len, latent_dim) - predicted next latent
            reward: (batch, seq_len, 1) - predicted reward
            done: (batch, seq_len, 1) - predicted done logit
            hidden: (1, batch, hidden_dim) - updated hidden state
        """
        x = torch.cat([z, action], dim=-1)
        output, hidden = self.gru(x, hidden)

        next_z = self.fc_z(output)
        reward = self.fc_reward(output)
        done = self.fc_done(output)

        return next_z, reward, done, hidden

    def predict_step(self, z: torch.Tensor, action: torch.Tensor,
                     hidden: torch.Tensor | None = None
                     ) -> tuple[torch.Tensor, float, float, torch.Tensor]:
        """Single-step prediction for inference/dreaming.

        Args:
            z: (1, 1, latent_dim)
            action: (1, 1, action_dim)
            hidden: (1, 1, hidden_dim)

        Returns:
            next_z, reward_value, done_prob, hidden
        """
        next_z, reward, done_logit, hidden = self.forward(z, action, hidden)
        reward_val = reward.squeeze().item()
        done_prob = torch.sigmoid(done_logit).squeeze().item()
        return next_z, reward_val, done_prob, hidden

    def initial_hidden(self, batch_size: int = 1, device: torch.device | None = None) -> torch.Tensor:
        """Create zero-initialized hidden state."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)


def rnn_loss(next_z_pred: torch.Tensor, next_z_target: torch.Tensor,
             reward_pred: torch.Tensor, reward_target: torch.Tensor,
             done_pred: torch.Tensor, done_target: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Combined loss for world model RNN."""
    z_loss = F.mse_loss(next_z_pred, next_z_target)
    r_loss = F.mse_loss(reward_pred.squeeze(-1), reward_target)
    d_loss = F.binary_cross_entropy_with_logits(done_pred.squeeze(-1), done_target)
    total = z_loss + r_loss + d_loss
    return total, {"z_loss": z_loss.item(), "r_loss": r_loss.item(), "d_loss": d_loss.item(), "total_loss": total.item()}
