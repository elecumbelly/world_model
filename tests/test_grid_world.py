"""Tests for the grid world environment."""

import numpy as np
import pytest

from world_model.envs.grid_world import GridWorld, AGENT, WALL, FOOD, HAZARD, GOAL, EMPTY


class TestGridWorld:
    def setup_method(self):
        self.env = GridWorld(grid_size=16, render_size=64, max_steps=200)

    def test_reset_returns_observation(self):
        obs = self.env.reset(seed=42)
        assert obs.shape == (64, 64, 3)
        assert obs.dtype == np.uint8

    def test_grid_has_border_walls(self):
        self.env.reset(seed=42)
        grid = self.env.grid
        # Top and bottom rows
        assert np.all(grid[0, :] == WALL)
        assert np.all(grid[-1, :] == WALL)
        # Left and right columns
        assert np.all(grid[:, 0] == WALL)
        assert np.all(grid[:, -1] == WALL)

    def test_grid_contains_required_entities(self):
        self.env.reset(seed=42)
        grid = self.env.grid
        assert np.sum(grid == AGENT) == 1
        assert np.sum(grid == GOAL) >= 1
        assert np.sum(grid == FOOD) >= 1

    def test_step_returns_correct_format(self):
        self.env.reset(seed=42)
        obs, reward, done, info = self.env.step(0)
        assert obs.shape == (64, 64, 3)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_wall_collision(self):
        self.env.reset(seed=42)
        pos_before = self.env.agent_pos
        # Move up repeatedly - should hit wall eventually and not move
        for _ in range(20):
            self.env.step(0)  # UP
        # Agent can't go past row 1 (border wall at row 0)
        assert self.env.agent_pos[0] >= 1

    def test_max_steps_terminates(self):
        env = GridWorld(max_steps=5)
        env.reset(seed=42)
        done = False
        for _ in range(10):
            _, _, done, info = env.step(np.random.randint(4))
            if done:
                break
        assert done

    def test_action_space(self):
        assert self.env.action_space_n == 4

    def test_observation_shape(self):
        assert self.env.observation_shape == (64, 64, 3)

    def test_deterministic_with_same_seed(self):
        obs1 = self.env.reset(seed=123)
        obs2 = self.env.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)

    def test_different_seeds_different_maps(self):
        obs1 = self.env.reset(seed=1)
        obs2 = self.env.reset(seed=2)
        assert not np.array_equal(obs1, obs2)
