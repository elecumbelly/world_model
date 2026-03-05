#!/usr/bin/env python3
"""Step 5: Train the controller via CMA-ES."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_model.utils.config import load_config
from world_model.utils.seeding import set_seed
from world_model.utils.logging_utils import get_logger
from world_model.training.train_controller import train_controller
from world_model.visualization.plots import plot_loss_curves

log = get_logger("train_controller")


def main(config_path: str = "world_model/configs/phase1_default.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.seed)

    env_kwargs = {
        "grid_size": cfg.env.grid_size,
        "render_size": cfg.env.render_size,
        "max_steps": cfg.env.max_steps,
        "wall_density": cfg.env.wall_density,
        "num_food": cfg.env.num_food,
        "num_hazards": cfg.env.num_hazards,
        "food_respawn_steps": cfg.env.food_respawn_steps,
    }

    history = train_controller(
        vae_path=f"{cfg.checkpoint_dir}/vae/best.pt",
        rnn_path=f"{cfg.checkpoint_dir}/rnn/best.pt",
        checkpoint_dir=f"{cfg.checkpoint_dir}/controller",
        latent_dim=cfg.controller.latent_dim,
        action_dim=cfg.controller.action_dim,
        hidden_dim=cfg.controller.hidden_dim,
        population_size=cfg.controller.population_size,
        generations=cfg.controller.generations,
        num_rollouts=cfg.controller.num_rollouts,
        sigma_init=cfg.controller.sigma_init,
        env_kwargs=env_kwargs,
    )

    plot_loss_curves(
        {"mean_reward": history["mean_reward"], "best_reward": history["best_reward"]},
        title="Controller Training (CMA-ES)",
        save_path="controller_learning_curve.png",
    )


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "world_model/configs/phase1_default.yaml"
    main(config)
