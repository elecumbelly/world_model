"""Tests for the DreamEngine."""

import numpy as np
import torch
import pytest

from world_model.models.vae import ConvVAE
from world_model.models.rnn import WorldRNN
from world_model.models.controller import Controller
from world_model.dreaming.dream_engine import DreamEngine
from world_model.envs.grid_world import GridWorld


class TestDreamEngine:
    def setup_method(self):
        self.vae = ConvVAE(latent_dim=32)
        self.rnn = WorldRNN(latent_dim=32, action_dim=4, hidden_dim=256)
        self.controller = Controller(latent_dim=32, hidden_dim=256, action_dim=4)
        self.device = torch.device("cpu")
        self.engine = DreamEngine(self.vae, self.rnn, self.controller, self.device)

    def test_encode_observation(self):
        obs = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        z = self.engine.encode_observation(obs)
        assert z.shape == (1, 32)

    def test_decode_latent(self):
        z = torch.randn(1, 32)
        img = self.engine.decode_latent(z)
        assert img.shape == (64, 64, 3)
        assert img.dtype == np.uint8

    def test_dream_produces_trajectory(self):
        obs = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = self.engine.dream(obs, num_steps=5)

        assert len(result["observations"]) >= 2  # At least initial + 1
        assert len(result["actions"]) >= 1
        assert len(result["rewards"]) >= 1
        assert all(isinstance(a, int) for a in result["actions"])
        assert all(0 <= a <= 3 for a in result["actions"])

    def test_dream_and_compare(self):
        env = GridWorld(grid_size=16, render_size=64, max_steps=20)
        result = self.engine.dream_and_compare(env, num_steps=5, seed=42)

        assert len(result["real_observations"]) >= 2
        assert len(result["dream_observations"]) >= 2
        assert len(result["actions"]) >= 1
        # Real and dream should have same number of frames
        assert len(result["real_observations"]) == len(result["dream_observations"])
