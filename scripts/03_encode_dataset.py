#!/usr/bin/env python3
"""Step 3: Encode all observations to latent vectors using trained VAE."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_model.utils.config import load_config
from world_model.utils.seeding import set_seed
from world_model.utils.logging_utils import get_logger
from world_model.training.train_vae import encode_dataset

log = get_logger("encode_dataset")


def main(config_path: str = "world_model/configs/phase1_default.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.seed)

    encode_dataset(
        rollout_dir=f"{cfg.data_dir}/rollouts",
        encoded_dir=f"{cfg.data_dir}/encoded",
        checkpoint_path=f"{cfg.checkpoint_dir}/vae/best.pt",
        latent_dim=cfg.vae.latent_dim,
    )

    log.info("Encoding complete!")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "world_model/configs/phase1_default.yaml"
    main(config)
