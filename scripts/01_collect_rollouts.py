#!/usr/bin/env python3
"""Step 1: Collect random rollouts from the grid world."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_model.utils.config import load_config
from world_model.utils.seeding import set_seed
from world_model.utils.logging_utils import get_logger
from world_model.data.rollout_collector import collect_rollouts

log = get_logger("collect_rollouts")


def main(config_path: str = "world_model/configs/phase1_default.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.seed)

    log.info(f"Collecting {cfg.collect.num_rollouts} rollouts...")
    files = collect_rollouts(
        num_rollouts=cfg.collect.num_rollouts,
        save_dir=f"{cfg.data_dir}/rollouts",
        grid_size=cfg.env.grid_size,
        render_size=cfg.env.render_size,
        max_steps=cfg.env.max_steps,
        wall_density=cfg.env.wall_density,
        num_food=cfg.env.num_food,
        num_hazards=cfg.env.num_hazards,
        food_respawn_steps=cfg.env.food_respawn_steps,
        seed=cfg.seed,
    )
    log.info(f"Saved {len(files)} rollout files")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "world_model/configs/phase1_default.yaml"
    main(config)
