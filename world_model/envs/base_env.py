from abc import ABC, abstractmethod
import numpy as np


class BaseEnv(ABC):
    """Abstract environment interface for all world model phases."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset environment, return initial observation (H, W, 3) uint8."""
        ...

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Take action, return (obs, reward, done, info)."""
        ...

    @abstractmethod
    def render(self) -> np.ndarray:
        """Return current RGB observation (H, W, 3) uint8."""
        ...

    @property
    @abstractmethod
    def action_space_n(self) -> int:
        """Number of discrete actions."""
        ...

    @property
    @abstractmethod
    def observation_shape(self) -> tuple[int, int, int]:
        """Shape of observations (H, W, C)."""
        ...
