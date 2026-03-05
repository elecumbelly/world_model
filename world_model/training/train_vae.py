"""VAE training loop and dataset encoding."""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from world_model.models.vae import ConvVAE, vae_loss
from world_model.data.datasets import ObservationDataset
from world_model.utils.device import get_device
from world_model.utils.logging_utils import get_logger

log = get_logger(__name__)


def train_vae(
    rollout_dir: str = "data/rollouts",
    checkpoint_dir: str = "checkpoints/vae",
    latent_dim: int = 32,
    beta: float = 0.5,
    lr: float = 1e-3,
    batch_size: int = 64,
    epochs: int = 20,
    max_files: int | None = None,
) -> dict:
    """Train the ConvVAE on observation images.

    Returns dict with training history (loss curves).
    """
    device = get_device()
    log.info(f"Training VAE on {device}")

    dataset = ObservationDataset(rollout_dir, max_files=max_files)
    log.info(f"Dataset: {len(dataset)} observations")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = ConvVAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    history = {"recon_loss": [], "kl_loss": [], "total_loss": []}
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_losses = {"recon_loss": 0, "kl_loss": 0, "total_loss": 0}
        n_batches = 0

        for batch in tqdm(loader, desc=f"VAE Epoch {epoch+1}/{epochs}", leave=False):
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            loss, losses = vae_loss(recon, batch, mu, logvar, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k, v in losses.items():
                epoch_losses[k] += v
            n_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= n_batches
            history[k].append(epoch_losses[k])

        log.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"recon={epoch_losses['recon_loss']:.6f} | "
            f"kl={epoch_losses['kl_loss']:.4f} | "
            f"total={epoch_losses['total_loss']:.6f}"
        )

        if epoch_losses["total_loss"] < best_loss:
            best_loss = epoch_losses["total_loss"]
            torch.save(model.state_dict(), ckpt_path / "best.pt")

    torch.save(model.state_dict(), ckpt_path / "final.pt")
    log.info(f"VAE training complete. Best loss: {best_loss:.6f}")
    return history


def encode_dataset(
    rollout_dir: str = "data/rollouts",
    encoded_dir: str = "data/encoded",
    checkpoint_path: str = "checkpoints/vae/best.pt",
    latent_dim: int = 32,
    batch_size: int = 256,
    max_files: int | None = None,
):
    """Encode all rollout observations to latent vectors using trained VAE.

    Saves one .npz per rollout with:
        latents: (T+1, latent_dim) float32
        actions: (T,) int32
        rewards: (T,) float32
        dones: (T,) bool
    """
    device = get_device()
    model = ConvVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    rollout_path = Path(rollout_dir)
    encoded_path = Path(encoded_dir)
    encoded_path.mkdir(parents=True, exist_ok=True)

    files = sorted(rollout_path.glob("rollout_*.npz"))
    if max_files is not None:
        files = files[:max_files]

    log.info(f"Encoding {len(files)} rollouts to {encoded_dir}")

    for f in tqdm(files, desc="Encoding"):
        data = np.load(f)
        obs = data["observations"]  # (T+1, H, W, 3)

        # Encode in batches
        all_latents = []
        for i in range(0, len(obs), batch_size):
            batch = obs[i:i + batch_size].astype(np.float32) / 255.0
            batch_t = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
            with torch.no_grad():
                mu, _ = model.encode(batch_t)
            all_latents.append(mu.cpu().numpy())

        latents = np.concatenate(all_latents, axis=0)

        out_name = f.stem.replace("rollout_", "encoded_") + ".npz"
        np.savez_compressed(
            encoded_path / out_name,
            latents=latents,
            actions=data["actions"],
            rewards=data["rewards"],
            dones=data["dones"],
        )

    log.info(f"Encoding complete: {len(files)} files")
