"""Tests for the WorldRNN."""

import torch
import pytest

from world_model.models.rnn import WorldRNN, rnn_loss


class TestWorldRNN:
    def setup_method(self):
        self.model = WorldRNN(latent_dim=32, action_dim=4, hidden_dim=256)
        self.batch_size = 4
        self.seq_len = 10

    def test_forward_shapes(self):
        z = torch.randn(self.batch_size, self.seq_len, 32)
        actions = torch.zeros(self.batch_size, self.seq_len, 4)
        actions[:, :, 0] = 1.0  # all UP

        next_z, reward, done, hidden = self.model(z, actions)
        assert next_z.shape == (self.batch_size, self.seq_len, 32)
        assert reward.shape == (self.batch_size, self.seq_len, 1)
        assert done.shape == (self.batch_size, self.seq_len, 1)
        assert hidden.shape == (1, self.batch_size, 256)

    def test_predict_step(self):
        z = torch.randn(1, 1, 32)
        action = torch.zeros(1, 1, 4)
        action[0, 0, 2] = 1.0
        hidden = self.model.initial_hidden(1)

        next_z, reward_val, done_prob, new_hidden = self.model.predict_step(z, action, hidden)
        assert next_z.shape == (1, 1, 32)
        assert isinstance(reward_val, float)
        assert isinstance(done_prob, float)
        assert 0.0 <= done_prob <= 1.0

    def test_initial_hidden(self):
        hidden = self.model.initial_hidden(batch_size=8)
        assert hidden.shape == (1, 8, 256)
        assert torch.all(hidden == 0)

    def test_hidden_state_persistence(self):
        z = torch.randn(1, 1, 32)
        action = torch.zeros(1, 1, 4)
        action[0, 0, 0] = 1.0
        hidden = self.model.initial_hidden(1)

        _, _, _, h1 = self.model(z, action, hidden)
        _, _, _, h2 = self.model(z, action, h1)
        assert not torch.allclose(h1, h2)

    def test_rnn_loss(self):
        z = torch.randn(4, 10, 32)
        actions = torch.zeros(4, 10, 4)
        next_z_pred, reward_pred, done_pred, _ = self.model(z, actions)
        next_z_target = torch.randn(4, 10, 32)
        rewards_target = torch.randn(4, 10)
        dones_target = torch.zeros(4, 10)

        loss, details = rnn_loss(next_z_pred, next_z_target, reward_pred, rewards_target, done_pred, dones_target)
        assert loss.requires_grad
        assert "z_loss" in details
        assert "r_loss" in details
        assert "d_loss" in details
