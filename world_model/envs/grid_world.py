"""16x16 grid world: agent navigates walls, collects food, avoids hazards, reaches goal."""

import numpy as np
from .base_env import BaseEnv

# Cell types
EMPTY = 0
WALL = 1
AGENT = 2
FOOD = 3
HAZARD = 4
GOAL = 5

# Actions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

# Colors (RGB)
COLORS = {
    EMPTY: (0, 0, 0),        # Black
    WALL: (128, 128, 128),    # Gray
    AGENT: (66, 133, 244),    # Blue
    FOOD: (52, 168, 83),      # Green
    HAZARD: (234, 67, 53),    # Red
    GOAL: (251, 188, 4),      # Gold
}


class GridWorld(BaseEnv):
    def __init__(
        self,
        grid_size: int = 16,
        render_size: int = 64,
        max_steps: int = 200,
        wall_density: float = 0.1,
        num_food: int = 5,
        num_hazards: int = 3,
        food_respawn_steps: int = 10,
        reward_food: float = 1.0,
        reward_hazard: float = -1.0,
        reward_goal: float = 5.0,
        reward_step: float = -0.01,
    ):
        if render_size % grid_size != 0:
            raise ValueError(
                f"render_size ({render_size}) must be divisible by grid_size ({grid_size}). "
                f"Got remainder {render_size % grid_size}. Use render_size={grid_size * (render_size // grid_size)} "
                f"or render_size={grid_size * (render_size // grid_size + 1)}."
            )

        interior_cells = (grid_size - 2) ** 2
        required_entities = 1 + 1 + num_food + num_hazards  # agent + goal + food + hazards
        max_walls = int(interior_cells * wall_density)
        available = interior_cells - max_walls
        if available < required_entities:
            raise ValueError(
                f"Not enough interior cells for entities. "
                f"interior={interior_cells}, walls={max_walls} (density={wall_density}), "
                f"remaining={available}, required={required_entities} "
                f"(1 agent + 1 goal + {num_food} food + {num_hazards} hazards). "
                f"Reduce wall_density or num_food/num_hazards."
            )

        self.grid_size = grid_size
        self.render_size = render_size
        self.max_steps = max_steps
        self.wall_density = wall_density
        self.num_food = num_food
        self.num_hazards = num_hazards
        self.food_respawn_steps = food_respawn_steps
        self.reward_food = reward_food
        self.reward_hazard = reward_hazard
        self.reward_goal = reward_goal
        self.reward_step = reward_step

        self._rng = np.random.RandomState(42)
        self.grid = None
        self.agent_pos = None
        self.step_count = 0
        self._eaten_food = []  # (row, col, step_eaten) for respawn tracking

    @property
    def action_space_n(self) -> int:
        return 4

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (self.render_size, self.render_size, 3)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.step_count = 0
        self._eaten_food = []

        # Border walls
        self.grid[0, :] = WALL
        self.grid[-1, :] = WALL
        self.grid[:, 0] = WALL
        self.grid[:, -1] = WALL

        # Interior walls
        interior = []
        for r in range(1, self.grid_size - 1):
            for c in range(1, self.grid_size - 1):
                interior.append((r, c))

        num_walls = int(len(interior) * self.wall_density)
        self._rng.shuffle(interior)
        wall_cells = interior[:num_walls]
        remaining = interior[num_walls:]

        for r, c in wall_cells:
            self.grid[r, c] = WALL

        # Place entities from remaining empty cells
        self._rng.shuffle(remaining)
        idx = 0

        # Agent
        self.agent_pos = remaining[idx]
        self.grid[self.agent_pos[0], self.agent_pos[1]] = AGENT
        idx += 1

        # Goal
        r, c = remaining[idx]
        self.grid[r, c] = GOAL
        idx += 1

        # Food
        for _ in range(self.num_food):
            if idx >= len(remaining):
                break
            r, c = remaining[idx]
            self.grid[r, c] = FOOD
            idx += 1

        # Hazards
        for _ in range(self.num_hazards):
            if idx >= len(remaining):
                break
            r, c = remaining[idx]
            self.grid[r, c] = HAZARD
            idx += 1

        return self.render()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        assert self.grid is not None, "Call reset() first"
        self.step_count += 1

        dr, dc = ACTION_DELTAS[action]
        new_r = self.agent_pos[0] + dr
        new_c = self.agent_pos[1] + dc

        reward = self.reward_step
        done = False
        info = {}

        # Check bounds and wall collision
        if (0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size
                and self.grid[new_r, new_c] != WALL):
            target = self.grid[new_r, new_c]

            # Clear old position
            self.grid[self.agent_pos[0], self.agent_pos[1]] = EMPTY

            if target == FOOD:
                reward += self.reward_food
                self._eaten_food.append((new_r, new_c, self.step_count))
                info["event"] = "food"
            elif target == HAZARD:
                reward += self.reward_hazard
                info["event"] = "hazard"
            elif target == GOAL:
                reward += self.reward_goal
                done = True
                info["event"] = "goal"

            self.agent_pos = (new_r, new_c)
            self.grid[new_r, new_c] = AGENT

        # Respawn food
        still_eaten = []
        for fr, fc, step_eaten in self._eaten_food:
            if self.step_count - step_eaten >= self.food_respawn_steps:
                if self.grid[fr, fc] == EMPTY:
                    self.grid[fr, fc] = FOOD
            else:
                still_eaten.append((fr, fc, step_eaten))
        self._eaten_food = still_eaten

        # Max steps
        if self.step_count >= self.max_steps:
            done = True
            info["truncated"] = True

        return self.render(), reward, done, info

    def render(self) -> np.ndarray:
        """Render grid as 64x64 RGB image."""
        cell_size = self.render_size // self.grid_size  # 4px per cell
        img = np.zeros((self.render_size, self.render_size, 3), dtype=np.uint8)

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                color = COLORS[self.grid[r, c]]
                r_start = r * cell_size
                c_start = c * cell_size
                img[r_start:r_start + cell_size, c_start:c_start + cell_size] = color

        return img

    def get_state(self) -> dict:
        """Return serializable state for saving."""
        return {
            "grid": self.grid.copy(),
            "agent_pos": self.agent_pos,
            "step_count": self.step_count,
        }
