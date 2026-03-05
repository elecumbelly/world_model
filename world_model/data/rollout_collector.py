"""Collect random or policy-driven rollouts from the grid world."""

from pathlib import Path
import numpy as np
from tqdm import tqdm

from world_model.envs.grid_world import GridWorld


def collect_rollouts(
    num_rollouts: int = 1000,
    save_dir: str = "data/rollouts",
    grid_size: int = 16,
    render_size: int = 64,
    max_steps: int = 200,
    seed: int = 42,
    policy=None,
    **env_kwargs,
) -> list[str]:
    """Collect rollouts and save each as an .npz file.

    Each .npz contains:
        observations: (T+1, H, W, 3) uint8
        actions: (T,) int
        rewards: (T,) float32
        dones: (T,) bool

    Args:
        policy: callable(obs, env) -> action. If None, uses random actions.
    Returns:
        List of saved file paths.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    env = GridWorld(
        grid_size=grid_size, render_size=render_size, max_steps=max_steps,
        **env_kwargs,
    )
    rng = np.random.RandomState(seed)
    saved_files = []

    for i in tqdm(range(num_rollouts), desc="Collecting rollouts"):
        obs = env.reset(seed=rng.randint(0, 2**31))
        observations = [obs]
        actions = []
        rewards = []
        dones = []

        done = False
        while not done:
            if policy is not None:
                action = policy(obs, env)
            else:
                action = rng.randint(0, env.action_space_n)

            obs, reward, done, info = env.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

        filepath = save_path / f"rollout_{i:05d}.npz"
        np.savez_compressed(
            filepath,
            observations=np.array(observations, dtype=np.uint8),
            actions=np.array(actions, dtype=np.int32),
            rewards=np.array(rewards, dtype=np.float32),
            dones=np.array(dones, dtype=bool),
        )
        saved_files.append(str(filepath))

    return saved_files
