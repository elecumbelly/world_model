"""PyTorch Dataset classes for VAE (observations) and RNN (sequences)."""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class ObservationDataset(Dataset):
    """Dataset of individual observations for VAE training.

    Loads all observations from rollout .npz files into memory.
    Returns observations as (3, H, W) float32 tensors in [0, 1].
    """

    def __init__(self, rollout_dir: str = "data/rollouts", max_files: int | None = None):
        self.observations = []
        rollout_path = Path(rollout_dir)
        files = sorted(rollout_path.glob("rollout_*.npz"))
        if max_files is not None:
            files = files[:max_files]

        for f in files:
            data = np.load(f)
            self.observations.append(data["observations"])

        self.observations = np.concatenate(self.observations, axis=0)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> torch.Tensor:
        obs = self.observations[idx].astype(np.float32) / 255.0
        return torch.from_numpy(obs).permute(2, 0, 1)  # (3, H, W)


class SequenceDataset(Dataset):
    """Dataset of latent sequences for RNN training.

    Loads encoded latent sequences from .npz files.
    Returns fixed-length chunks of (z, action, reward, done).
    """

    def __init__(self, encoded_dir: str = "data/encoded", seq_len: int = 50,
                 action_dim: int = 4, max_files: int | None = None):
        self.seq_len = seq_len
        self.action_dim = action_dim
        self.sequences = []

        encoded_path = Path(encoded_dir)
        files = sorted(encoded_path.glob("encoded_*.npz"))
        if max_files is not None:
            files = files[:max_files]

        for f in files:
            data = np.load(f)
            z = data["latents"]        # (T+1, latent_dim)
            actions = data["actions"]   # (T,)
            rewards = data["rewards"]   # (T,)
            dones = data["dones"]       # (T,)

            T = len(actions)
            if T >= seq_len + 1:
                self.sequences.append({
                    "latents": z,
                    "actions": actions,
                    "rewards": rewards,
                    "dones": dones,
                })

    def __len__(self) -> int:
        total = 0
        for seq in self.sequences:
            T = len(seq["actions"])
            total += max(1, T - self.seq_len)
        return total

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Find which sequence and offset
        for seq in self.sequences:
            T = len(seq["actions"])
            n_chunks = max(1, T - self.seq_len)
            if idx < n_chunks:
                start = idx
                end = start + self.seq_len

                z = torch.tensor(seq["latents"][start:end], dtype=torch.float32)
                z_next = torch.tensor(seq["latents"][start + 1:end + 1], dtype=torch.float32)

                # One-hot encode actions
                actions_idx = seq["actions"][start:end]
                actions = np.zeros((self.seq_len, self.action_dim), dtype=np.float32)
                for t, a in enumerate(actions_idx):
                    actions[t, a] = 1.0

                rewards = torch.tensor(seq["rewards"][start:end], dtype=torch.float32)
                dones = torch.tensor(seq["dones"][start:end], dtype=torch.float32)

                return {
                    "z": z,
                    "z_next": z_next,
                    "actions": torch.from_numpy(actions),
                    "rewards": rewards,
                    "dones": dones,
                }
            idx -= n_chunks

        raise IndexError("Index out of range")
