"""A Gymnasium wrapper around gym_super_mario_bros using composition and a custom compatibility adapter."""

from __future__ import annotations

from enum import Enum
from typing import Any

import gym as legacy_gym
import gymnasium as gym
import numpy as np

import yft_utils.nes_py_patch  # noqa: F401 (patches nes_py for NumPy 2.x compatibility)
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT



class RenderMode(str, Enum):
    """Supported render modes for MarioEnv."""

    RGB_ARRAY = "rgb_array"
    HUMAN = "human"
    NONE = "none"

    @classmethod
    def from_str(cls, value: str | RenderMode) -> RenderMode:
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError:
            valid = [e.value for e in cls]
            raise ValueError(f"Invalid render mode '{value}'. Valid choices: {valid}")


class MarioEnv(gym.Env):
    """Gymnasium environment wrapper around gym_super_mario_bros.

    Uses composition to wrap the JoypadSpace legacy emulator and adapt it to Gymnasium v1.0.0.
    """

    metadata = {"render_modes": [RenderMode.RGB_ARRAY.value, RenderMode.HUMAN.value, RenderMode.NONE.value]}

    def __init__(self, render_mode: str | RenderMode = RenderMode.RGB_ARRAY) -> None:
        super().__init__()
        # 1. Create the legacy Gym environment and unwrap it to bypass incompatible TimeLimit wrapper
        raw_env = legacy_gym.make("SuperMarioBros-v3").unwrapped

        # 2. Apply JoypadSpace wrapping to simplify action space
        self._env = JoypadSpace(raw_env, SIMPLE_MOVEMENT)

        # Expose spaces
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self.render_mode = RenderMode.from_str(render_mode)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        obs = self._env.reset()
        info = {}
        if isinstance(obs, tuple):
            obs, info = obs
        return np.array(obs, dtype=np.uint8), info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        step_result = self._env.step(int(action))

        # Unpack legacy 4-tuple (obs, reward, done, info) or modern 5-tuple
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False

        return np.array(obs, dtype=np.uint8), float(reward), bool(terminated), bool(truncated), info

    def render(self) -> np.ndarray | None:
        if self.render_mode == RenderMode.HUMAN:
            self._env.render(mode="human")
            return None
        if self.render_mode == RenderMode.NONE:
            return None
        return self._env.render(mode="rgb_array")

    def close(self) -> None:
        self._env.close()

