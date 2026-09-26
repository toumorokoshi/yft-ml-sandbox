"""Unit tests for checkpointing, argument parsing, and helper functions in train.py."""

from __future__ import annotations

import signal
import tempfile
import unittest

import torch
import torch.optim as optim

from jepa_rl_mario.model import MarioModel
from jepa_rl_mario.train import (
    CHECKPOINT_KEY_EPISODES,
    CHECKPOINT_KEY_EPSILON,
    CHECKPOINT_KEY_MODEL,
    CHECKPOINT_KEY_OPTIMIZER,
    CHECKPOINT_KEY_TARGET_MODEL,
    GracefulInterruptHandler,
    apply_checkpoint_state,
    compute_total_episodes,
    create_checkpoint,
    extract_checkpoint,
    load_checkpoint,
    parse_args,
    resolve_checkpoint_save_path,
    save_checkpoint,
)

# Constants for testing (Rule 5)
TEST_EPSILON: float = 0.42
TEST_EPISODES: int = 15
DUMMY_TENSOR_VAL_1: float = 1.23
DUMMY_TENSOR_VAL_2: float = 4.56


class TestTrainCheckpointDataStructures(unittest.TestCase):
    """Pure unit tests operating directly on checkpoint data structures (Rules 1, 2, 3)."""

    def test_create_checkpoint(self) -> None:
        mock_model_state = {"layer.weight": torch.tensor([DUMMY_TENSOR_VAL_1])}
        mock_target_state = {"layer.weight": torch.tensor([DUMMY_TENSOR_VAL_2])}
        mock_optimizer_state = {"state": {}, "param_groups": []}

        checkpoint = create_checkpoint(
            model_state_dict=mock_model_state,
            target_state_dict=mock_target_state,
            optimizer_state_dict=mock_optimizer_state,
            epsilon=TEST_EPSILON,
            episodes=TEST_EPISODES,
        )

        self.assertIn(CHECKPOINT_KEY_MODEL, checkpoint)
        self.assertIn(CHECKPOINT_KEY_TARGET_MODEL, checkpoint)
        self.assertIn(CHECKPOINT_KEY_OPTIMIZER, checkpoint)
        self.assertEqual(checkpoint[CHECKPOINT_KEY_EPSILON], TEST_EPSILON)
        self.assertEqual(checkpoint[CHECKPOINT_KEY_EPISODES], TEST_EPISODES)
        self.assertTrue(torch.equal(checkpoint[CHECKPOINT_KEY_MODEL]["layer.weight"], mock_model_state["layer.weight"]))

    def test_create_checkpoint_without_optimizer(self) -> None:
        mock_model_state = {"layer.weight": torch.tensor([DUMMY_TENSOR_VAL_1])}
        mock_target_state = {"layer.weight": torch.tensor([DUMMY_TENSOR_VAL_2])}

        checkpoint = create_checkpoint(
            model_state_dict=mock_model_state,
            target_state_dict=mock_target_state,
            optimizer_state_dict=None,
            epsilon=TEST_EPSILON,
            episodes=TEST_EPISODES,
        )

        self.assertNotIn(CHECKPOINT_KEY_OPTIMIZER, checkpoint)
        self.assertEqual(checkpoint[CHECKPOINT_KEY_EPSILON], TEST_EPSILON)

    def test_extract_checkpoint_valid(self) -> None:
        mock_model_state = {"w": torch.tensor([1.0])}
        mock_target_state = {"w": torch.tensor([2.0])}
        mock_opt_state = {"state": {}}
        checkpoint = {
            CHECKPOINT_KEY_MODEL: mock_model_state,
            CHECKPOINT_KEY_TARGET_MODEL: mock_target_state,
            CHECKPOINT_KEY_OPTIMIZER: mock_opt_state,
            CHECKPOINT_KEY_EPSILON: TEST_EPSILON,
            CHECKPOINT_KEY_EPISODES: TEST_EPISODES,
        }

        m_state, t_state, o_state, eps, ep_count = extract_checkpoint(checkpoint)
        self.assertEqual(m_state, mock_model_state)
        self.assertEqual(t_state, mock_target_state)
        self.assertEqual(o_state, mock_opt_state)
        self.assertEqual(eps, TEST_EPSILON)
        self.assertEqual(ep_count, TEST_EPISODES)

    def test_extract_checkpoint_missing_model_key_raises(self) -> None:
        invalid_checkpoint = {CHECKPOINT_KEY_EPSILON: 0.5}
        with self.assertRaises(KeyError):
            extract_checkpoint(invalid_checkpoint)

    def test_apply_checkpoint_state(self) -> None:
        # Create models and an optimizer
        q_net = MarioModel(num_actions=2, height=80, width=80)
        target_net = MarioModel(num_actions=2, height=80, width=80)
        optimizer = optim.Adam(q_net.parameters(), lr=1e-3)

        # Create source model with distinct weights
        source_net = MarioModel(num_actions=2, height=80, width=80)
        for param in source_net.parameters():
            param.data.fill_(7.0)

        checkpoint = create_checkpoint(
            model_state_dict=source_net.state_dict(),
            target_state_dict=source_net.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            epsilon=0.25,
            episodes=10,
        )

        eps, ep_count = apply_checkpoint_state(
            q_network=q_net,
            checkpoint=checkpoint,
            target_network=target_net,
            optimizer=optimizer,
        )

        self.assertEqual(eps, 0.25)
        self.assertEqual(ep_count, 10)
        for param in q_net.parameters():
            self.assertTrue(torch.all(param == 7.0))
        for param in target_net.parameters():
            self.assertTrue(torch.all(param == 7.0))

    def test_parse_args_defaults_and_custom(self) -> None:
        # Test defaults
        default_args = parse_args([])
        self.assertIsNone(default_args.save_checkpoint)
        self.assertIsNone(default_args.load_checkpoint)
        self.assertFalse(default_args.eval)
        self.assertIsNone(default_args.epsilon)

        # Test custom arguments
        custom_args = parse_args([
            "--save-checkpoint",
            "/path/save.pt",
            "--load-checkpoint",
            "/path/load.pt",
            "--eval",
            "--epsilon",
            "0.05",
            "--episodes",
            "20",
            "--steps",
            "150",
            "--device",
            "cpu",
        ])
        self.assertEqual(custom_args.save_checkpoint, "/path/save.pt")
        self.assertEqual(custom_args.load_checkpoint, "/path/load.pt")
        self.assertTrue(custom_args.eval)
        self.assertEqual(custom_args.epsilon, 0.05)
        self.assertEqual(custom_args.episodes, 20)
        self.assertEqual(custom_args.steps, 150)
        self.assertEqual(custom_args.device, "cpu")

    def test_resolve_checkpoint_save_path(self) -> None:
        # If save_checkpoint is given, it is preferred
        self.assertEqual(
            resolve_checkpoint_save_path(
                save_checkpoint_arg="/tmp/save.pt",
                load_checkpoint_arg="/tmp/load.pt",
                is_eval=False,
            ),
            "/tmp/save.pt",
        )
        # If only load_checkpoint is given and not in eval mode, it saves back to load_checkpoint
        self.assertEqual(
            resolve_checkpoint_save_path(
                save_checkpoint_arg=None,
                load_checkpoint_arg="/tmp/load.pt",
                is_eval=False,
            ),
            "/tmp/load.pt",
        )
        # In eval mode without save_checkpoint, no saving path is resolved
        self.assertIsNone(
            resolve_checkpoint_save_path(
                save_checkpoint_arg=None,
                load_checkpoint_arg="/tmp/load.pt",
                is_eval=True,
            )
        )
        # Neither argument given
        self.assertIsNone(
            resolve_checkpoint_save_path(
                save_checkpoint_arg=None,
                load_checkpoint_arg=None,
                is_eval=False,
            )
        )

    def test_compute_total_episodes(self) -> None:
        self.assertEqual(compute_total_episodes(prior_episodes=None, episodes_completed=5), 5)
        self.assertEqual(compute_total_episodes(prior_episodes=10, episodes_completed=4), 14)
        self.assertEqual(compute_total_episodes(prior_episodes=0, episodes_completed=0), 0)

    def test_graceful_interrupt_handler_signal(self) -> None:
        handler = GracefulInterruptHandler()
        self.assertFalse(handler.interrupted)

        # First signal sets interrupted flag without raising
        handler._handle_signal(signal.SIGINT, None)
        self.assertTrue(handler.interrupted)

        # Second signal immediately raises KeyboardInterrupt for forced abort
        with self.assertRaises(KeyboardInterrupt):
            handler._handle_signal(signal.SIGINT, None)

    def test_graceful_interrupt_handler_context(self) -> None:
        with GracefulInterruptHandler() as handler:
            self.assertFalse(handler.interrupted)



class TestTrainCheckpointIntegration(unittest.TestCase):
    """Single IO integration test testing saving and loading from filesystem (Rule 3)."""

    def test_save_and_load_checkpoint_integration(self) -> None:
        model = MarioModel(num_actions=3, height=80, width=80)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        checkpoint_data = create_checkpoint(
            model_state_dict=model.state_dict(),
            target_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            epsilon=0.33,
            episodes=7,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = f"{tmpdir}/nested/subdir/checkpoint.pt"

            # Test save_checkpoint (creates parent dirs and writes)
            save_checkpoint(file_path, checkpoint_data)

            # Test load_checkpoint (reads file from filesystem)
            loaded = load_checkpoint(file_path, device=torch.device("cpu"))

            # Validate loaded checkpoint
            new_model = MarioModel(num_actions=3, height=80, width=80)
            eps, eps_count = apply_checkpoint_state(new_model, loaded)

            self.assertEqual(eps, 0.33)
            self.assertEqual(eps_count, 7)
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))

            # Test load_checkpoint with missing file raises FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                load_checkpoint(f"{tmpdir}/nonexistent.pt")


if __name__ == "__main__":
    unittest.main()
