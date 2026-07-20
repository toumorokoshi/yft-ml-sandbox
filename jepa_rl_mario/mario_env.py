"""A pure-functional core representation of a simplified Mario RL environment,
wrapped in a Gymnasium Env interface via composition.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Environment Constants
INITIAL_POSITION = 0.0
INITIAL_HEIGHT = 0.0
STAGE_LENGTH = 10.0
WALK_SPEED = 0.5
JUMP_IMPULSE = 1.0
GRAVITY = 0.25
OBSTACLE_TOLERANCE = 0.2
OBSTACLES = (3.0, 7.0)

# Action constants
ACTION_IDLE = 0
ACTION_RIGHT = 1
ACTION_JUMP = 2

# Reward constants
REWARD_GOAL = 10.0
REWARD_DEATH = -10.0
REWARD_STEP = -0.1


@dataclasses.dataclass(frozen=True)
class MarioState:
    position: float
    height: float
    is_jumping: bool
    terminated: bool
    truncated: bool


def init_mario_state() -> MarioState:
    """Creates the initial MarioState. Minimal logic helper function."""
    return MarioState(
        position=INITIAL_POSITION,
        height=INITIAL_HEIGHT,
        is_jumping=False,
        terminated=False,
        truncated=False,
    )


def step_mario_state(state: MarioState, action: int) -> MarioState:
    """A pure function that computes the next state from current state and action.
    This function contains no IO and is completely deterministic.
    """
    if state.terminated or state.truncated:
        return state

    # Start with current state variables
    next_position = state.position
    next_height = state.height
    next_is_jumping = state.is_jumping

    # 1. Update horizontal movement
    if action == ACTION_RIGHT:
        next_position += WALK_SPEED

    # 2. Update vertical movement / jumping
    if action == ACTION_JUMP and not state.is_jumping:
        next_is_jumping = True
        next_height = JUMP_IMPULSE
    elif state.is_jumping:
        next_height -= GRAVITY
        if next_height <= 0.0:
            next_height = 0.0
            next_is_jumping = False

    # 3. Check obstacles and bounds
    next_terminated = False
    for obs in OBSTACLES:
        if next_height < 0.5 and abs(next_position - obs) < OBSTACLE_TOLERANCE:
            next_terminated = True

    if next_position >= STAGE_LENGTH:
        next_terminated = True

    return MarioState(
        position=next_position,
        height=next_height,
        is_jumping=next_is_jumping,
        terminated=next_terminated,
        truncated=False,
    )


def get_observation(state: MarioState) -> np.ndarray:
    """Computes the observation array from the state."""
    return np.array([state.position, state.height], dtype=np.float32)


def get_reward(prev_state: MarioState, next_state: MarioState, action: int) -> float:
    """Computes the reward for transitioning between states."""
    if next_state.terminated:
        if next_state.position >= STAGE_LENGTH:
            return REWARD_GOAL
        # If terminated but didn't reach the end, Mario hit an obstacle
        return REWARD_DEATH
    return REWARD_STEP


class MarioEnv(gym.Env):
    """Gymnasium environment wrapping the pure functional state transition functions.
    Uses composition to separate state management and environment logic.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self) -> None:
        super().__init__()
        # Define action space: Idle, Right, Jump
        self.action_space = spaces.Discrete(3)

        # Define observation space: [position, height]
        # Min position 0, max stage length + 1. Min height 0, max jump impulse.
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([STAGE_LENGTH + 1.0, JUMP_IMPULSE], dtype=np.float32),
            dtype=np.float32,
        )
        self._state: MarioState = init_mario_state()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._state = init_mario_state()
        obs = get_observation(self._state)
        return obs, {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        prev_state = self._state
        self._state = step_mario_state(prev_state, int(action))

        obs = get_observation(self._state)
        reward = get_reward(prev_state, self._state, int(action))

        return (
            obs,
            reward,
            self._state.terminated,
            self._state.truncated,
            {},
        )

    def render(self) -> str:
        """Render a text representation of the environment.
        This is the only IO-like function in the environment.
        """
        # Create a text line representing the track
        track_size = int(STAGE_LENGTH * 2)
        track = ["_"] * track_size

        # Place obstacles
        for obs in OBSTACLES:
            idx = int(obs * 2)
            if 0 <= idx < track_size:
                track[idx] = "X"

        # Place Mario
        mario_idx = int(self._state.position * 2)
        if 0 <= mario_idx < track_size:
            if self._state.height > 0.0:
                track[mario_idx] = "M"  # Jumping Mario
            else:
                track[mario_idx] = "m"  # Ground Mario
        elif mario_idx >= track_size:
            # Mario reached the end
            track[-1] = "G"

        render_str = "".join(track) + f" (Pos: {self._state.position:.1f}, Height: {self._state.height:.2f})"
        return render_str
