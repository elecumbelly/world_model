#!/usr/bin/env python3
"""Step 4: Train the RNN world model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_model.utils.config import load_config
from world_model.utils.seeding import set_seed
from world_model.utils.logging_utils import get_logger
from world_model.training.train_rnn import train_rnn
from world_model.visualization.plots import plot_loss_curves

log = get_logger("train_rnn")


def main(config_path: str = "world_model/configs/phase1_default.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.seed)

    history = train_rnn(
        encoded_dir=f"{cfg.data_dir}/encoded",
        checkpoint_dir=f"{cfg.checkpoint_dir}/rnn",
        latent_dim=cfg.rnn.latent_dim,
        action_dim=cfg.rnn.action_dim,
        hidden_dim=cfg.rnn.hidden_dim,
        lr=cfg.rnn.lr,
        batch_size=cfg.rnn.batch_size,
        epochs=cfg.rnn.epochs,
        seq_len=cfg.rnn.seq_len,
    )

    plot_loss_curves(history, title="RNN Training", save_path="rnn_loss.png")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "world_model/configs/phase1_default.yaml"
    main(config)
