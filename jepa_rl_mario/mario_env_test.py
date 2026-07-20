"""Tests for the gym_super_mario_bros integration wrapper."""

from __future__ import annotations

import unittest
import numpy as np

from jepa_rl_mario.mario_env import MarioEnv


class TestMarioEnv(unittest.TestCase):
    """Integration tests verifying the Gymnasium wrapper for gym_super_mario_bros."""

    def test_gym_interface_integration(self) -> None:
        env = MarioEnv(render_mode="rgb_array")
        try:
            obs, info = env.reset()
            # Verify observation shape and types
            self.assertEqual(obs.shape, (240, 256, 3))
            self.assertEqual(obs.dtype, np.uint8)
            self.assertIsInstance(info, dict)

            # Test step execution
            action = int(env.action_space.sample())
            next_obs, reward, terminated, truncated, info = env.step(action)

            self.assertEqual(next_obs.shape, (240, 256, 3))
            self.assertEqual(next_obs.dtype, np.uint8)
            self.assertIsInstance(reward, float)
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIsInstance(info, dict)

            # Check render output format
            frame = env.render()
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape, (240, 256, 3))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
