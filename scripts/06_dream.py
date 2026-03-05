#!/usr/bin/env python3
"""Step 6: Generate dream trajectories."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_model.utils.config import load_config
from world_model.utils.seeding import set_seed
from world_model.utils.logging_utils import get_logger
from world_model.envs.grid_world import GridWorld
from world_model.dreaming.dream_engine import DreamEngine
from world_model.visualization.compare_view import save_comparison_gif, plot_comparison_grid

log = get_logger("dream")


def main(config_path: str = "world_model/configs/phase1_default.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.seed)

    engine = DreamEngine.from_checkpoints(
        vae_path=f"{cfg.checkpoint_dir}/vae/best.pt",
        rnn_path=f"{cfg.checkpoint_dir}/rnn/best.pt",
        controller_path=f"{cfg.checkpoint_dir}/controller/best.pt",
        latent_dim=cfg.vae.latent_dim,
        hidden_dim=cfg.rnn.hidden_dim,
    )

    env = GridWorld(
        grid_size=cfg.env.grid_size,
        render_size=cfg.env.render_size,
        max_steps=cfg.env.max_steps,
    )

    log.info("Generating real vs dream comparison...")
    result = engine.dream_and_compare(env, num_steps=50, seed=cfg.seed)

    plot_comparison_grid(
        result["real_observations"],
        result["dream_observations"],
        num_frames=10,
        save_path="real_vs_dream_grid.png",
    )

    save_comparison_gif(
        result["real_observations"],
        result["dream_observations"],
        save_path="real_vs_dream.gif",
        fps=5,
    )
    log.info("Saved real_vs_dream.gif and real_vs_dream_grid.png")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "world_model/configs/phase1_default.yaml"
    main(config)
