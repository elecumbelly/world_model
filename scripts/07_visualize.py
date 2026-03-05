#!/usr/bin/env python3
"""Step 7: Generate all visualizations."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_model.utils.config import load_config
from world_model.utils.seeding import set_seed
from world_model.utils.device import get_device
from world_model.utils.logging_utils import get_logger
from world_model.models.vae import ConvVAE
from world_model.data.datasets import ObservationDataset
from world_model.visualization.plots import plot_reconstructions, plot_latent_space

log = get_logger("visualize")


def main(config_path: str = "world_model/configs/phase1_default.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.seed)
    device = get_device()

    # Load VAE
    vae = ConvVAE(latent_dim=cfg.vae.latent_dim).to(device)
    vae.load_state_dict(torch.load(
        f"{cfg.checkpoint_dir}/vae/best.pt", map_location=device, weights_only=True
    ))

    # Load observations
    dataset = ObservationDataset(f"{cfg.data_dir}/rollouts", max_files=100)
    observations = dataset.observations  # (N, H, W, 3) uint8

    log.info("Plotting reconstructions...")
    plot_reconstructions(vae, observations, num_images=8, save_path="vae_reconstructions.png")

    # Collect per-observation rewards for coloring
    log.info("Plotting latent space...")
    rollout_path = Path(f"{cfg.data_dir}/rollouts")
    all_rewards = []
    for f in sorted(rollout_path.glob("rollout_*.npz"))[:100]:
        data = np.load(f)
        rewards = data["rewards"]
        # Pad to match observations length (T+1 obs, T rewards)
        all_rewards.append(np.concatenate([[0.0], rewards]))
    all_rewards = np.concatenate(all_rewards)

    if len(all_rewards) >= len(observations):
        all_rewards = all_rewards[:len(observations)]
    else:
        all_rewards = None

    plot_latent_space(vae, observations, rewards=all_rewards, save_path="latent_space.png")

    log.info("All visualizations saved!")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "world_model/configs/phase1_default.yaml"
    main(config)
