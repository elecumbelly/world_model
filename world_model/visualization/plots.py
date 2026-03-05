"""Loss curves, reconstruction grids, and latent space visualization."""

import numpy as np
import matplotlib.pyplot as plt
import torch

from world_model.models.vae import ConvVAE
from world_model.utils.device import get_device


def plot_loss_curves(history: dict, title: str = "Training Loss", save_path: str | None = None):
    """Plot training loss curves from a history dict."""
    fig, axes = plt.subplots(1, len(history), figsize=(5 * len(history), 4))
    if len(history) == 1:
        axes = [axes]

    for ax, (key, values) in zip(axes, history.items()):
        ax.plot(values)
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_reconstructions(
    vae: ConvVAE,
    observations: np.ndarray,
    num_images: int = 8,
    save_path: str | None = None,
):
    """Plot original vs reconstructed images in a 2-row grid.

    Args:
        vae: trained ConvVAE model
        observations: (N, H, W, 3) uint8 array
        num_images: how many pairs to show
    """
    device = get_device()
    vae.eval()

    indices = np.random.choice(len(observations), num_images, replace=False)
    originals = observations[indices]

    # Encode and decode
    batch = torch.from_numpy(originals.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
    with torch.no_grad():
        recons, _, _ = vae(batch)
    recons = recons.permute(0, 2, 3, 1).cpu().numpy()
    recons = (recons * 255).clip(0, 255).astype(np.uint8)

    fig, axes = plt.subplots(2, num_images, figsize=(2 * num_images, 4))
    for i in range(num_images):
        axes[0, i].imshow(originals[i])
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original")

        axes[1, i].imshow(recons[i])
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstructed")

    plt.suptitle("VAE Reconstructions")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_latent_space(
    vae: ConvVAE,
    observations: np.ndarray,
    rewards: np.ndarray | None = None,
    save_path: str | None = None,
    max_points: int = 5000,
):
    """2D t-SNE visualization of latent vectors, optionally colored by reward.

    Args:
        vae: trained ConvVAE model
        observations: (N, H, W, 3) uint8 array
        rewards: (N,) float array for coloring
        max_points: subsample if dataset is large
    """
    from sklearn.manifold import TSNE

    device = get_device()
    vae.eval()

    if len(observations) > max_points:
        idx = np.random.choice(len(observations), max_points, replace=False)
        observations = observations[idx]
        if rewards is not None:
            rewards = rewards[idx]

    # Encode all
    latents = []
    batch_size = 256
    for i in range(0, len(observations), batch_size):
        batch = observations[i:i + batch_size].astype(np.float32) / 255.0
        batch_t = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            mu, _ = vae.encode(batch_t)
        latents.append(mu.cpu().numpy())
    latents = np.concatenate(latents, axis=0)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embedded = tsne.fit_transform(latents)

    fig, ax = plt.subplots(figsize=(8, 8))
    if rewards is not None:
        scatter = ax.scatter(embedded[:, 0], embedded[:, 1], c=rewards, cmap="RdYlGn",
                             s=5, alpha=0.6)
        plt.colorbar(scatter, ax=ax, label="Reward")
    else:
        ax.scatter(embedded[:, 0], embedded[:, 1], s=5, alpha=0.6)

    ax.set_title("Latent Space (t-SNE)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
