"""Tests for the Controller."""

import numpy as np
import torch
import pytest

from world_model.models.controller import Controller


class TestController:
    def setup_method(self):
        self.controller = Controller(latent_dim=32, hidden_dim=256, action_dim=4)

    def test_forward_shape(self):
        z = torch.randn(1, 32)
        h = torch.randn(1, 256)
        logits = self.controller(z, h)
        assert logits.shape == (1, 4)

    def test_act_returns_valid_action(self):
        z = torch.randn(1, 32)
        h = torch.randn(1, 256)
        action = self.controller.act(z, h)
        assert 0 <= action <= 3

    def test_num_params(self):
        # (32 + 256) * 4 + 4 = 1156
        assert self.controller.num_params == 1156

    def test_get_set_params_roundtrip(self):
        params = self.controller.get_params()
        assert params.shape == (1156,)

        # Modify and set back
        new_params = np.random.randn(1156).astype(np.float32)
        self.controller.set_params(new_params)
        retrieved = self.controller.get_params()
        np.testing.assert_allclose(retrieved, new_params, atol=1e-6)

    def test_deterministic_action(self):
        z = torch.randn(1, 32)
        h = torch.randn(1, 256)
        a1 = self.controller.act(z, h)
        a2 = self.controller.act(z, h)
        assert a1 == a2  # Same input = same action (greedy)
