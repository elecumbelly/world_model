#!/usr/bin/env python3
"""Step 2: Train the Convolutional VAE."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_model.utils.config import load_config
from world_model.utils.seeding import set_seed
from world_model.utils.logging_utils import get_logger
from world_model.training.train_vae import train_vae
from world_model.visualization.plots import plot_loss_curves

log = get_logger("train_vae")


def main(config_path: str = "world_model/configs/phase1_default.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.seed)

    history = train_vae(
        rollout_dir=f"{cfg.data_dir}/rollouts",
        checkpoint_dir=f"{cfg.checkpoint_dir}/vae",
        latent_dim=cfg.vae.latent_dim,
        beta=cfg.vae.beta,
        lr=cfg.vae.lr,
        batch_size=cfg.vae.batch_size,
        epochs=cfg.vae.epochs,
    )

    plot_loss_curves(history, title="VAE Training", save_path="vae_loss.png")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "world_model/configs/phase1_default.yaml"
    main(config)
