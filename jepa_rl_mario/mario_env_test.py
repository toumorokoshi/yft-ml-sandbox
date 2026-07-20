"""Tests for the Mario RL environment."""

from __future__ import annotations

import unittest
import numpy as np

from jepa_rl_mario.mario_env import (
    ACTION_IDLE,
    ACTION_JUMP,
    ACTION_RIGHT,
    MarioEnv,
    init_mario_state,
    step_mario_state,
    get_observation,
    get_reward,
    STAGE_LENGTH,
    MarioState,
)


class TestMarioEnvFunctional(unittest.TestCase):
    """Unit tests running on the data structures directly, with no environment instantiation."""

    def test_initial_state(self) -> None:
        state = init_mario_state()
        self.assertEqual(state.position, 0.0)
        self.assertEqual(state.height, 0.0)
        self.assertFalse(state.is_jumping)
        self.assertFalse(state.terminated)

    def test_move_right(self) -> None:
        state = init_mario_state()
        next_state = step_mario_state(state, ACTION_RIGHT)
        self.assertEqual(next_state.position, 0.5)
        self.assertEqual(next_state.height, 0.0)

    def test_jump_physics(self) -> None:
        state = init_mario_state()
        # Trigger jump
        state = step_mario_state(state, ACTION_JUMP)
        self.assertTrue(state.is_jumping)
        self.assertEqual(state.height, 1.0)

        # Let gravity bring Mario down
        state = step_mario_state(state, ACTION_IDLE)
        self.assertTrue(state.is_jumping)
        self.assertEqual(state.height, 0.75)

        state = step_mario_state(state, ACTION_IDLE)
        self.assertEqual(state.height, 0.5)

        state = step_mario_state(state, ACTION_IDLE)
        self.assertEqual(state.height, 0.25)

        state = step_mario_state(state, ACTION_IDLE)
        self.assertEqual(state.height, 0.0)
        self.assertFalse(state.is_jumping)

    def test_obstacle_collision(self) -> None:
        # Obstacle is at 3.0. Let's place Mario at 2.5 and walk right.
        # This will put him at 3.0 on the ground.
        state = init_mario_state()
        # Move right repeatedly until 2.5
        for _ in range(5):
            state = step_mario_state(state, ACTION_RIGHT)
        self.assertEqual(state.position, 2.5)
        self.assertFalse(state.terminated)

        # Walk right to 3.0 (obstacle position)
        state = step_mario_state(state, ACTION_RIGHT)
        self.assertEqual(state.position, 3.0)
        self.assertTrue(state.terminated)  # Hit obstacle

    def test_obstacle_clearance(self) -> None:
        # Mario jumps at 2.5, then moves right to 3.0.
        state = init_mario_state()
        for _ in range(5):
            state = step_mario_state(state, ACTION_RIGHT)
        self.assertEqual(state.position, 2.5)

        # Jump
        state = step_mario_state(state, ACTION_JUMP)
        self.assertTrue(state.is_jumping)
        self.assertEqual(state.height, 1.0)

        # Move right to 3.0 (overlapping obstacle) while in air
        state = step_mario_state(state, ACTION_RIGHT)
        self.assertEqual(state.position, 3.0)
        self.assertEqual(state.height, 0.75)
        self.assertFalse(state.terminated)  # Successfully cleared the obstacle!

    def test_rewards(self) -> None:
        state_init = init_mario_state()
        # Regular step reward
        state_next = step_mario_state(state_init, ACTION_RIGHT)
        self.assertEqual(get_reward(state_init, state_next, ACTION_RIGHT), -0.1)

        # Goal reward
        # Manually construct a state at/past goal
        state_goal_prev = MarioState(
            position=9.5, height=0.0, is_jumping=False, terminated=False, truncated=False
        )
        state_goal = step_mario_state(state_goal_prev, ACTION_RIGHT)
        self.assertEqual(get_reward(state_goal_prev, state_goal, ACTION_RIGHT), 10.0)


class TestMarioEnvIntegration(unittest.TestCase):
    """Integration test using the Gymnasium environment wrapper."""

    def test_gym_interface_integration(self) -> None:
        env = MarioEnv()
        obs, info = env.reset()
        self.assertEqual(obs.shape, (2,))
        self.assertEqual(obs[0], 0.0)
        self.assertEqual(obs[1], 0.0)

        # Take one step right
        next_obs, reward, terminated, truncated, info = env.step(ACTION_RIGHT)
        self.assertEqual(next_obs[0], 0.5)
        self.assertEqual(next_obs[1], 0.0)
        self.assertEqual(reward, -0.1)
        self.assertFalse(terminated)

        # Check render doesn't crash
        render_output = env.render()
        self.assertIsInstance(render_output, str)


if __name__ == "__main__":
    unittest.main()
