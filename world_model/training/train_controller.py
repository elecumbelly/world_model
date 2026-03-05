"""CMA-ES optimization of the linear controller."""

from pathlib import Path
import numpy as np
import torch
import cma

from world_model.models.vae import ConvVAE
from world_model.models.rnn import WorldRNN
from world_model.models.controller import Controller
from world_model.envs.grid_world import GridWorld
from world_model.utils.device import get_device
from world_model.utils.logging_utils import get_logger

log = get_logger(__name__)


def evaluate_controller(
    controller: Controller,
    vae: ConvVAE,
    rnn: WorldRNN,
    env_kwargs: dict,
    num_rollouts: int = 16,
    device: torch.device | None = None,
) -> float:
    """Evaluate a controller by running rollouts in the real environment.

    Returns mean total reward across rollouts.
    """
    if device is None:
        device = get_device()

    env = GridWorld(**env_kwargs)
    total_rewards = []

    for i in range(num_rollouts):
        obs = env.reset(seed=i * 1000)
        hidden = rnn.initial_hidden(1, device)
        episode_reward = 0.0
        done = False

        while not done:
            # Encode observation
            obs_t = torch.from_numpy(obs.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                mu, _ = vae.encode(obs_t)
                z = mu  # Use mean (no sampling for evaluation)

            # Get action from controller
            h = hidden.squeeze(0)  # (1, hidden_dim)
            action = controller.act(z, h)

            # Update RNN hidden state
            action_oh = torch.zeros(1, 1, 4, device=device)
            action_oh[0, 0, action] = 1.0
            _, _, _, hidden = rnn.forward(z.unsqueeze(1), action_oh, hidden)

            obs, reward, done, _ = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)

    return float(np.mean(total_rewards))


def train_controller(
    vae_path: str = "checkpoints/vae/best.pt",
    rnn_path: str = "checkpoints/rnn/best.pt",
    checkpoint_dir: str = "checkpoints/controller",
    latent_dim: int = 32,
    action_dim: int = 4,
    hidden_dim: int = 256,
    population_size: int = 64,
    generations: int = 100,
    num_rollouts: int = 16,
    sigma_init: float = 0.1,
    env_kwargs: dict | None = None,
) -> dict:
    """Train controller using CMA-ES.

    Returns dict with training history (rewards per generation).
    """
    device = get_device()
    log.info(f"Training Controller via CMA-ES on {device}")

    if env_kwargs is None:
        env_kwargs = {}

    # Load frozen V and M models
    vae = ConvVAE(latent_dim=latent_dim).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
    vae.eval()

    rnn_model = WorldRNN(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    rnn_model.load_state_dict(torch.load(rnn_path, map_location=device, weights_only=True))
    rnn_model.eval()

    controller = Controller(latent_dim=latent_dim, hidden_dim=hidden_dim, action_dim=action_dim).to(device)
    log.info(f"Controller params: {controller.num_params}")

    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # CMA-ES setup
    init_params = controller.get_params()
    es = cma.CMAEvolutionStrategy(init_params, sigma_init, {
        "popsize": population_size,
        "seed": 42,
    })

    history = {"mean_reward": [], "best_reward": [], "generation": []}
    best_reward = -float("inf")

    for gen in range(generations):
        solutions = es.ask()
        rewards = []

        for params in solutions:
            controller.set_params(np.array(params))
            reward = evaluate_controller(controller, vae, rnn_model, env_kwargs, num_rollouts, device)
            rewards.append(reward)

        # CMA-ES minimizes, so negate rewards
        es.tell(solutions, [-r for r in rewards])

        mean_r = float(np.mean(rewards))
        best_r = float(np.max(rewards))
        history["mean_reward"].append(mean_r)
        history["best_reward"].append(best_r)
        history["generation"].append(gen)

        log.info(f"Gen {gen+1}/{generations} | mean={mean_r:.2f} | best={best_r:.2f}")

        if best_r > best_reward:
            best_reward = best_r
            best_idx = np.argmax(rewards)
            controller.set_params(np.array(solutions[best_idx]))
            torch.save(controller.state_dict(), ckpt_path / "best.pt")

        if es.stop():
            log.info(f"CMA-ES converged at generation {gen+1}")
            break

    # Save final best
    best_params = es.result.xbest
    controller.set_params(np.array(best_params))
    torch.save(controller.state_dict(), ckpt_path / "final.pt")
    log.info(f"Controller training complete. Best reward: {best_reward:.2f}")
    return history
