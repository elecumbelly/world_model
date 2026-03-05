"""End-to-end mini pipeline test with tiny data."""

import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

from world_model.envs.grid_world import GridWorld
from world_model.data.rollout_collector import collect_rollouts
from world_model.data.datasets import ObservationDataset, SequenceDataset
from world_model.models.vae import ConvVAE
from world_model.models.rnn import WorldRNN
from world_model.models.controller import Controller
from world_model.training.train_vae import train_vae, encode_dataset
from world_model.training.train_rnn import train_rnn
from world_model.dreaming.dream_engine import DreamEngine


class TestMiniPipeline:
    """Runs the full pipeline with tiny data to verify everything connects."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp = tmp_path
        self.rollout_dir = str(tmp_path / "rollouts")
        self.encoded_dir = str(tmp_path / "encoded")
        self.vae_ckpt = str(tmp_path / "vae")
        self.rnn_ckpt = str(tmp_path / "rnn")
        self.ctrl_ckpt = str(tmp_path / "controller")

    def test_full_pipeline(self):
        # Step 1: Collect tiny rollouts
        files = collect_rollouts(
            num_rollouts=5,
            save_dir=self.rollout_dir,
            max_steps=20,
            seed=42,
        )
        assert len(files) == 5
        data = np.load(files[0])
        assert "observations" in data
        assert "actions" in data

        # Step 2: Train VAE (1 epoch)
        history = train_vae(
            rollout_dir=self.rollout_dir,
            checkpoint_dir=self.vae_ckpt,
            epochs=1,
            batch_size=16,
        )
        assert "total_loss" in history
        assert Path(self.vae_ckpt, "best.pt").exists()

        # Step 3: Encode dataset
        encode_dataset(
            rollout_dir=self.rollout_dir,
            encoded_dir=self.encoded_dir,
            checkpoint_path=f"{self.vae_ckpt}/best.pt",
        )
        encoded_files = list(Path(self.encoded_dir).glob("*.npz"))
        assert len(encoded_files) == 5

        # Step 4: Train RNN (1 epoch)
        history = train_rnn(
            encoded_dir=self.encoded_dir,
            checkpoint_dir=self.rnn_ckpt,
            epochs=1,
            batch_size=4,
            seq_len=5,
        )
        assert "total_loss" in history
        assert Path(self.rnn_ckpt, "best.pt").exists()

        # Step 5: Dream (skip CMA-ES, just verify dreaming works)
        device = torch.device("cpu")
        vae = ConvVAE(latent_dim=32).to(device)
        vae.load_state_dict(torch.load(f"{self.vae_ckpt}/best.pt", map_location=device, weights_only=True))
        rnn = WorldRNN(latent_dim=32, action_dim=4, hidden_dim=256).to(device)
        rnn.load_state_dict(torch.load(f"{self.rnn_ckpt}/best.pt", map_location=device, weights_only=True))
        controller = Controller(latent_dim=32, hidden_dim=256, action_dim=4)

        engine = DreamEngine(vae, rnn, controller, device)
        env = GridWorld(max_steps=20)
        obs = env.reset(seed=42)
        result = engine.dream(obs, num_steps=5)

        assert len(result["observations"]) >= 2
        assert len(result["actions"]) >= 1


class TestDatasetPreflightChecks:
    """Verify datasets fail fast with actionable messages on bad data."""

    def test_observation_dataset_missing_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Rollout directory not found"):
            ObservationDataset(str(tmp_path / "nonexistent"))

    def test_observation_dataset_empty_dir(self, tmp_path):
        empty_dir = tmp_path / "empty_rollouts"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No rollout_\\*\\.npz files"):
            ObservationDataset(str(empty_dir))

    def test_sequence_dataset_missing_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Encoded directory not found"):
            SequenceDataset(str(tmp_path / "nonexistent"))

    def test_sequence_dataset_empty_dir(self, tmp_path):
        empty_dir = tmp_path / "empty_encoded"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No encoded_\\*\\.npz files"):
            SequenceDataset(str(empty_dir))

    def test_sequence_dataset_too_short(self, tmp_path):
        """All rollouts shorter than seq_len should raise ValueError."""
        enc_dir = tmp_path / "short_encoded"
        enc_dir.mkdir()
        # Create a file with only 3 steps (need seq_len+1 = 51)
        np.savez(
            enc_dir / "encoded_00000.npz",
            latents=np.zeros((4, 32)),
            actions=np.zeros(3, dtype=np.int32),
            rewards=np.zeros(3),
            dones=np.zeros(3, dtype=bool),
        )
        with pytest.raises(ValueError, match="No sequences long enough"):
            SequenceDataset(str(enc_dir), seq_len=50)
