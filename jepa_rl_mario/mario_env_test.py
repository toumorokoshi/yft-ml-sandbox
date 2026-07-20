"""Tests for the gym_super_mario_bros integration wrapper and DQN helper functions."""

from __future__ import annotations

import unittest
import numpy as np
import torch

from jepa_rl_mario.mario_env import MarioEnv
from jepa_rl_mario.train import preprocess_observation, select_action, compute_loss, QNetwork


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


class TestDQNAgentHelpers(unittest.TestCase):
    """Pure unit tests verifying the data structure transitions and computations for DQN training."""

    def test_preprocess_observation(self) -> None:
        # Mock a random RGB observation frame of shape (240, 256, 3)
        mock_obs = np.random.randint(0, 256, size=(240, 256, 3), dtype=np.uint8)
        processed = preprocess_observation(mock_obs)

        self.assertEqual(processed.shape, (80, 80))
        self.assertEqual(processed.dtype, np.float32)
        self.assertTrue((processed >= 0.0).all() and (processed <= 1.0).all())

    def test_select_action_greedy(self) -> None:
        # If epsilon is 0, selection must be greedy based on max value
        q_values = torch.tensor([[1.0, 5.0, 2.0, 0.5]])
        action = select_action(q_values, epsilon=0.0, num_actions=4)
        self.assertEqual(action, 1)

    def test_select_action_random(self) -> None:
        # If epsilon is 1, selection is random (check that it returns valid actions)
        q_values = torch.tensor([[1.0, 5.0, 2.0, 0.5]])
        actions = [select_action(q_values, epsilon=1.0, num_actions=4) for _ in range(100)]
        for action in actions:
            self.assertTrue(0 <= action < 4)

    def test_compute_loss(self) -> None:
        q_network = QNetwork(num_actions=2)
        target_network = QNetwork(num_actions=2)

        # Mock transition batch
        states = torch.randn(2, 1, 80, 80)
        actions = torch.tensor([0, 1], dtype=torch.long)
        rewards = torch.tensor([1.0, -1.0], dtype=torch.float32)
        next_states = torch.randn(2, 1, 80, 80)
        dones = torch.tensor([0.0, 1.0], dtype=torch.float32)

        loss = compute_loss(
            q_network,
            target_network,
            states,
            actions,
            rewards,
            next_states,
            dones,
            gamma=0.99,
        )

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Loss should be a scalar tensor


if __name__ == "__main__":
    unittest.main()
