"""Pygame dual-pane interactive viewer: real world vs model prediction."""

import numpy as np
import torch

from world_model.envs.grid_world import GridWorld
from world_model.envs.rendering import DualPaneViewer
from world_model.dreaming.dream_engine import DreamEngine
from world_model.utils.logging_utils import get_logger

log = get_logger(__name__)


def run_interactive_comparison(
    dream_engine: DreamEngine,
    env_kwargs: dict | None = None,
    seed: int = 0,
):
    """Run an interactive dual-pane session.

    Left pane: real environment
    Right pane: model's prediction (re-encoded each step)

    Arrow keys control the agent. ESC or close window to quit.
    """
    if env_kwargs is None:
        env_kwargs = {}

    env = GridWorld(**env_kwargs)
    viewer = DualPaneViewer()

    real_obs = env.reset(seed=seed)
    z = dream_engine.encode_observation(real_obs)
    hidden = dream_engine.rnn.initial_hidden(1, dream_engine.device)

    dream_img = dream_engine.decode_latent(z)
    total_real_reward = 0.0
    total_dream_reward = 0.0
    step = 0

    viewer.render_pair(real_obs, dream_img, step, total_real_reward, total_dream_reward)

    try:
        while True:
            action = viewer.get_action()
            if action is None:
                break

            # Real step
            real_obs, real_r, done, _ = env.step(action)
            total_real_reward += real_r

            # Dream step
            action_oh = torch.zeros(1, 1, dream_engine.rnn.action_dim, device=dream_engine.device)
            action_oh[0, 0, action] = 1.0
            next_z, dream_r, done_prob, hidden = dream_engine.rnn.predict_step(
                z.unsqueeze(1), action_oh, hidden
            )
            z = next_z.squeeze(1)
            total_dream_reward += dream_r
            dream_img = dream_engine.decode_latent(z)

            step += 1
            viewer.render_pair(real_obs, dream_img, step, total_real_reward, total_dream_reward)

            if done:
                log.info(f"Episode done at step {step}. Real reward: {total_real_reward:.2f}")
                # Wait for user to close or press ESC
                while viewer.get_action() is not None:
                    pass
                break
    finally:
        viewer.close()
