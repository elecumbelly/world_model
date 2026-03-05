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
        if not rollout_path.exists():
            raise FileNotFoundError(
                f"Rollout directory not found: {rollout_path.resolve()}. "
                f"Run 01_collect_rollouts.py first."
            )
        files = sorted(rollout_path.glob("rollout_*.npz"))
        if max_files is not None:
            files = files[:max_files]
        if not files:
            raise FileNotFoundError(
                f"No rollout_*.npz files in {rollout_path.resolve()}. "
                f"Run 01_collect_rollouts.py first."
            )

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
        if not encoded_path.exists():
            raise FileNotFoundError(
                f"Encoded directory not found: {encoded_path.resolve()}. "
                f"Run 03_encode_dataset.py first."
            )
        files = sorted(encoded_path.glob("encoded_*.npz"))
        if max_files is not None:
            files = files[:max_files]
        if not files:
            raise FileNotFoundError(
                f"No encoded_*.npz files in {encoded_path.resolve()}. "
                f"Run 03_encode_dataset.py first."
            )

        skipped = 0
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
            else:
                skipped += 1

        if not self.sequences:
            raise ValueError(
                f"No sequences long enough for seq_len={seq_len}. "
                f"All {len(files)} files had fewer than {seq_len + 1} steps. "
                f"Reduce seq_len or collect longer rollouts."
            )

        # Precompute flat index map: (seq_idx, start_offset) for O(1) lookup
        self._index_map = []
        for seq_idx, seq in enumerate(self.sequences):
            T = len(seq["actions"])
            n_chunks = max(1, T - self.seq_len)
            for offset in range(n_chunks):
                self._index_map.append((seq_idx, offset))

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq_idx, start = self._index_map[idx]
        seq = self.sequences[seq_idx]
        end = start + self.seq_len

        z = torch.from_numpy(seq["latents"][start:end].astype(np.float32))
        z_next = torch.from_numpy(seq["latents"][start + 1:end + 1].astype(np.float32))

        # Vectorized one-hot encoding
        actions_idx = seq["actions"][start:end]
        actions = np.zeros((self.seq_len, self.action_dim), dtype=np.float32)
        actions[np.arange(self.seq_len), actions_idx] = 1.0

        rewards = torch.from_numpy(seq["rewards"][start:end].astype(np.float32))
        dones = torch.from_numpy(seq["dones"][start:end].astype(np.float32))

        return {
            "z": z,
            "z_next": z_next,
            "actions": torch.from_numpy(actions),
            "rewards": rewards,
            "dones": dones,
        }
