#!/usr/bin/env python3
"""Interactive play: control the agent with arrow keys in a pygame window."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_model.utils.config import load_config
from world_model.utils.logging_utils import get_logger
from world_model.envs.grid_world import GridWorld
from world_model.envs.rendering import PygameViewer

log = get_logger("play")

ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}


def main(config_path: str = "world_model/configs/phase1_default.yaml"):
    cfg = load_config(config_path)

    env = GridWorld(
        grid_size=cfg.env.grid_size,
        render_size=cfg.env.render_size,
        max_steps=cfg.env.max_steps,
        wall_density=cfg.env.wall_density,
        num_food=cfg.env.num_food,
        num_hazards=cfg.env.num_hazards,
        food_respawn_steps=cfg.env.food_respawn_steps,
    )
    viewer = PygameViewer(title="Grid World - Arrow Keys to Move, ESC to Quit")

    obs = env.reset(seed=42)
    total_reward = 0.0
    step = 0

    viewer.render_frame(obs, f"Step: {step} | Reward: {total_reward:.2f}")

    try:
        while True:
            action = viewer.get_action()
            if action is None:
                break

            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1

            event = info.get("event", "")
            info_text = f"Step: {step} | Reward: {total_reward:.2f} | {ACTION_NAMES[action]}"
            if event:
                info_text += f" | {event.upper()}"

            viewer.render_frame(obs, info_text)

            if done:
                log.info(f"Episode done! Steps: {step}, Total reward: {total_reward:.2f}")
                # Reset for a new episode
                obs = env.reset(seed=42 + step)
                total_reward = 0.0
                step = 0
                viewer.render_frame(obs, "NEW EPISODE - Press arrow key to start")
    finally:
        viewer.close()


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "world_model/configs/phase1_default.yaml"
    main(config)
