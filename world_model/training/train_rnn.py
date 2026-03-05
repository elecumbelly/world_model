"""RNN world model training loop."""

from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from world_model.models.rnn import WorldRNN, rnn_loss
from world_model.data.datasets import SequenceDataset
from world_model.utils.device import get_device
from world_model.utils.logging_utils import get_logger

log = get_logger(__name__)


def train_rnn(
    encoded_dir: str = "data/encoded",
    checkpoint_dir: str = "checkpoints/rnn",
    latent_dim: int = 32,
    action_dim: int = 4,
    hidden_dim: int = 256,
    lr: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 30,
    seq_len: int = 50,
    max_files: int | None = None,
    val_split: float = 0.1,
) -> dict:
    """Train the WorldRNN on encoded latent sequences.

    Returns dict with training history.
    """
    device = get_device()
    log.info(f"Training RNN on {device}")

    full_dataset = SequenceDataset(encoded_dir, seq_len=seq_len, action_dim=action_dim, max_files=max_files)
    if len(full_dataset) < batch_size:
        log.warning(f"Dataset ({len(full_dataset)} chunks) smaller than batch_size ({batch_size})")

    # Train/val split
    n_val = max(1, int(len(full_dataset) * val_split))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    log.info(f"Dataset: {n_train} train / {n_val} val sequence chunks")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = WorldRNN(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    history = {
        "z_loss": [], "r_loss": [], "d_loss": [], "total_loss": [],
        "val_z_loss": [], "val_r_loss": [], "val_d_loss": [], "val_total_loss": [],
    }
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        epoch_losses = {"z_loss": 0, "r_loss": 0, "d_loss": 0, "total_loss": 0}
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"RNN Epoch {epoch+1}/{epochs}", leave=False):
            z = batch["z"].to(device)
            z_next = batch["z_next"].to(device)
            actions = batch["actions"].to(device)
            rewards = batch["rewards"].to(device)
            dones = batch["dones"].to(device)

            next_z_pred, reward_pred, done_pred, _ = model(z, actions)
            loss, losses = rnn_loss(next_z_pred, z_next, reward_pred, rewards, done_pred, dones)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k, v in losses.items():
                epoch_losses[k] += v
            n_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= n_batches
            history[k].append(epoch_losses[k])

        # --- Validate ---
        model.eval()
        val_losses = {"val_z_loss": 0, "val_r_loss": 0, "val_d_loss": 0, "val_total_loss": 0}
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                z = batch["z"].to(device)
                z_next = batch["z_next"].to(device)
                actions = batch["actions"].to(device)
                rewards = batch["rewards"].to(device)
                dones = batch["dones"].to(device)

                next_z_pred, reward_pred, done_pred, _ = model(z, actions)
                _, losses = rnn_loss(next_z_pred, z_next, reward_pred, rewards, done_pred, dones)
                val_losses["val_z_loss"] += losses["z_loss"]
                val_losses["val_r_loss"] += losses["r_loss"]
                val_losses["val_d_loss"] += losses["d_loss"]
                val_losses["val_total_loss"] += losses["total_loss"]
                n_val_batches += 1

        for k in val_losses:
            val_losses[k] /= n_val_batches
            history[k].append(val_losses[k])

        log.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"train={epoch_losses['total_loss']:.6f} | "
            f"val={val_losses['val_total_loss']:.6f} | "
            f"z={epoch_losses['z_loss']:.6f} | "
            f"r={epoch_losses['r_loss']:.6f} | "
            f"d={epoch_losses['d_loss']:.6f}"
        )

        if val_losses["val_total_loss"] < best_val_loss:
            best_val_loss = val_losses["val_total_loss"]
            torch.save(model.state_dict(), ckpt_path / "best.pt")

    torch.save(model.state_dict(), ckpt_path / "final.pt")
    log.info(f"RNN training complete. Best val loss: {best_val_loss:.6f}")
    return history
