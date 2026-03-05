"""Dream engine: generates imagined trajectories entirely inside the learned model."""

import numpy as np
import torch

from world_model.models.vae import ConvVAE
from world_model.models.rnn import WorldRNN
from world_model.models.controller import Controller
from world_model.utils.device import get_device
from world_model.utils.logging_utils import get_logger

log = get_logger(__name__)


class DreamEngine:
    """Generates dream trajectories using the learned V, M, C models."""

    def __init__(
        self,
        vae: ConvVAE,
        rnn: WorldRNN,
        controller: Controller,
        device: torch.device | None = None,
    ):
        self.vae = vae
        self.rnn = rnn
        self.controller = controller
        self.device = device or get_device()

        self.vae.eval()
        self.rnn.eval()
        self.controller.eval()

    @classmethod
    def from_checkpoints(
        cls,
        vae_path: str = "checkpoints/vae/best.pt",
        rnn_path: str = "checkpoints/rnn/best.pt",
        controller_path: str = "checkpoints/controller/best.pt",
        latent_dim: int = 32,
        action_dim: int = 4,
        hidden_dim: int = 256,
    ) -> "DreamEngine":
        """Load all models from checkpoint files."""
        device = get_device()

        vae = ConvVAE(latent_dim=latent_dim).to(device)
        vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))

        rnn = WorldRNN(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
        rnn.load_state_dict(torch.load(rnn_path, map_location=device, weights_only=True))

        controller = Controller(latent_dim=latent_dim, hidden_dim=hidden_dim, action_dim=action_dim).to(device)
        controller.load_state_dict(torch.load(controller_path, map_location=device, weights_only=True))

        return cls(vae, rnn, controller, device)

    def encode_observation(self, obs: np.ndarray) -> torch.Tensor:
        """Encode a single observation to latent vector."""
        obs_t = torch.from_numpy(obs.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu, _ = self.vae.encode(obs_t)
        return mu

    def decode_latent(self, z: torch.Tensor) -> np.ndarray:
        """Decode a latent vector back to an RGB image."""
        with torch.no_grad():
            recon = self.vae.decode(z)
        img = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (img * 255).clip(0, 255).astype(np.uint8)

    def dream(
        self,
        initial_obs: np.ndarray,
        num_steps: int = 50,
        temperature: float = 0.0,
    ) -> dict:
        """Generate a dream trajectory starting from a real observation.

        Args:
            initial_obs: (H, W, 3) uint8 starting observation
            num_steps: number of dream steps to generate
            temperature: if > 0, sample from latent distribution; 0 = use mean

        Returns:
            dict with:
                observations: list of (H, W, 3) uint8 decoded dream images
                actions: list of int actions taken
                rewards: list of float predicted rewards
                dones: list of bool predicted done flags
                latents: list of (latent_dim,) numpy arrays
        """
        with torch.no_grad():
            z = self.encode_observation(initial_obs)
            hidden = self.rnn.initial_hidden(1, self.device)

            dream_obs = [self.decode_latent(z)]
            dream_actions = []
            dream_rewards = []
            dream_dones = []
            dream_latents = [z.squeeze(0).cpu().numpy()]

            for step in range(num_steps):
                # Controller selects action
                h = hidden.squeeze(0)
                action = self.controller.act(z, h)

                # One-hot encode action
                action_oh = torch.zeros(1, 1, self.rnn.action_dim, device=self.device)
                action_oh[0, 0, action] = 1.0

                # RNN predicts next state
                next_z, reward_val, done_prob, hidden = self.rnn.predict_step(
                    z.unsqueeze(1), action_oh, hidden
                )
                z = next_z.squeeze(1)

                # Add temperature-based noise
                if temperature > 0:
                    z = z + temperature * torch.randn_like(z)

                # Decode for visualization
                dream_img = self.decode_latent(z)

                dream_obs.append(dream_img)
                dream_actions.append(action)
                dream_rewards.append(reward_val)
                dream_dones.append(done_prob > 0.5)
                dream_latents.append(z.squeeze(0).cpu().numpy())

                if done_prob > 0.5:
                    log.info(f"Dream ended at step {step+1} (done predicted)")
                    break

        return {
            "observations": dream_obs,
            "actions": dream_actions,
            "rewards": dream_rewards,
            "dones": dream_dones,
            "latents": dream_latents,
        }

    def dream_and_compare(
        self,
        env,
        num_steps: int = 50,
        seed: int = 0,
    ) -> dict:
        """Run real environment and dream in parallel for comparison.

        Returns dict with real_* and dream_* trajectories.
        """
        with torch.no_grad():
            # Start real environment
            real_obs = env.reset(seed=seed)
            z = self.encode_observation(real_obs)
            hidden = self.rnn.initial_hidden(1, self.device)

            real_observations = [real_obs]
            dream_observations = [self.decode_latent(z)]
            real_rewards = []
            dream_rewards = []
            actions = []

            for step in range(num_steps):
                # Controller picks action (same for both)
                h = hidden.squeeze(0)
                action = self.controller.act(z, h)

                # One-hot action
                action_oh = torch.zeros(1, 1, self.rnn.action_dim, device=self.device)
                action_oh[0, 0, action] = 1.0

                # Dream: RNN predicts next
                next_z, dream_r, done_prob, hidden = self.rnn.predict_step(
                    z.unsqueeze(1), action_oh, hidden
                )
                z = next_z.squeeze(1)

                # Real: step environment
                real_obs, real_r, done, _ = env.step(action)

                dream_observations.append(self.decode_latent(z))
                real_observations.append(real_obs)
                dream_rewards.append(dream_r)
                real_rewards.append(real_r)
                actions.append(action)

                if done:
                    break

        return {
            "real_observations": real_observations,
            "dream_observations": dream_observations,
            "real_rewards": real_rewards,
            "dream_rewards": dream_rewards,
            "actions": actions,
        }
