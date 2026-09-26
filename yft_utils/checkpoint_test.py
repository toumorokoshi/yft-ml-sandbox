"""Unit tests for generic checkpoint utilities in yft_utils.checkpoint."""

from __future__ import annotations

import signal
import tempfile
import unittest
from typing import Final

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]
    HAS_TORCH = False

from yft_utils.checkpoint import (
    CHECKPOINT_KEY_EPISODES,
    CHECKPOINT_KEY_EPOCH,
    CHECKPOINT_KEY_EPSILON,
    CHECKPOINT_KEY_MODEL,
    CHECKPOINT_KEY_OPTIMIZER,
    CHECKPOINT_KEY_STEP,
    CHECKPOINT_KEY_TARGET_MODEL,
    GracefulInterruptHandler,
    apply_checkpoint_models,
    apply_checkpoint_state,
    compute_total_episodes,
    create_checkpoint,
    extract_checkpoint,
    extract_checkpoint_model_state,
    load_checkpoint,
    resolve_checkpoint_save_path,
    save_checkpoint,
)

# Constants for testing (Rule 5)
TEST_EPSILON: Final[float] = 0.42
TEST_EPISODES: Final[int] = 15
TEST_EPOCH: Final[int] = 3
TEST_STEP: Final[int] = 1200


class DummyModule(nn.Module if HAS_TORCH else object):  # type: ignore[misc]
    """Simple linear module for testing checkpoint restoration."""

    def __init__(self) -> None:
        super().__init__()
        if HAS_TORCH:
            self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestCheckpointDataStructures(unittest.TestCase):
    """Pure unit tests operating directly on checkpoint data structures (Rules 1, 2, 3)."""

    def test_create_checkpoint_minimal(self) -> None:
        mock_model_state = {"weight": [1.0, 2.0]}
        checkpoint = create_checkpoint(model_state_dict=mock_model_state)

        self.assertIn(CHECKPOINT_KEY_MODEL, checkpoint)
        self.assertEqual(checkpoint[CHECKPOINT_KEY_MODEL], mock_model_state)
        self.assertNotIn(CHECKPOINT_KEY_OPTIMIZER, checkpoint)

    def test_create_checkpoint_with_optimizer_and_metadata(self) -> None:
        mock_model_state = {"weight": [1.0, 2.0]}
        mock_optimizer_state = {"param_groups": []}

        checkpoint = create_checkpoint(
            model_state_dict=mock_model_state,
            optimizer_state_dict=mock_optimizer_state,
            epoch=TEST_EPOCH,
            step=TEST_STEP,
            epsilon=TEST_EPSILON,
            episodes=TEST_EPISODES,
        )

        self.assertIn(CHECKPOINT_KEY_MODEL, checkpoint)
        self.assertIn(CHECKPOINT_KEY_OPTIMIZER, checkpoint)
        self.assertEqual(checkpoint[CHECKPOINT_KEY_EPOCH], TEST_EPOCH)
        self.assertEqual(checkpoint[CHECKPOINT_KEY_STEP], TEST_STEP)
        self.assertEqual(checkpoint[CHECKPOINT_KEY_EPSILON], TEST_EPSILON)
        self.assertEqual(checkpoint[CHECKPOINT_KEY_EPISODES], TEST_EPISODES)

    def test_extract_checkpoint_valid(self) -> None:
        mock_model_state = {"w": [1.0]}
        mock_opt_state = {"state": {}}
        checkpoint = {
            CHECKPOINT_KEY_MODEL: mock_model_state,
            CHECKPOINT_KEY_OPTIMIZER: mock_opt_state,
            "custom_metric": 0.95,
        }

        model_state, opt_state, metadata = extract_checkpoint(checkpoint)
        self.assertEqual(model_state, mock_model_state)
        self.assertEqual(opt_state, mock_opt_state)
        self.assertEqual(metadata, {"custom_metric": 0.95})

    def test_extract_checkpoint_missing_model_key_raises(self) -> None:
        invalid_checkpoint = {"other_key": 123}
        with self.assertRaises(KeyError):
            extract_checkpoint(invalid_checkpoint)

    def test_extract_checkpoint_model_state(self) -> None:
        checkpoint = {CHECKPOINT_KEY_MODEL: {"layer.weight": [1]}}
        self.assertEqual(extract_checkpoint_model_state(checkpoint), {"layer.weight": [1]})

    @unittest.skipUnless(HAS_TORCH, "Requires PyTorch")
    def test_apply_checkpoint_state(self) -> None:
        model = DummyModule()
        target_model = DummyModule()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Set specific weight on target_model
        with torch.no_grad():
            for p in target_model.parameters():
                p.fill_(9.5)

        checkpoint = create_checkpoint(
            model_state_dict=target_model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            epsilon=TEST_EPSILON,
            episodes=TEST_EPISODES,
        )

        metadata = apply_checkpoint_state(checkpoint, model=model, optimizer=optimizer)

        self.assertEqual(metadata.get(CHECKPOINT_KEY_EPSILON), TEST_EPSILON)
        self.assertEqual(metadata.get(CHECKPOINT_KEY_EPISODES), TEST_EPISODES)

        # Verify weights were updated in model
        for p in model.parameters():
            self.assertTrue(torch.all(p == 9.5))

    @unittest.skipUnless(HAS_TORCH, "Requires PyTorch")
    def test_apply_checkpoint_models(self) -> None:
        m1 = DummyModule()
        m2 = DummyModule()

        with torch.no_grad():
            for p in m1.parameters():
                p.fill_(3.0)
            for p in m2.parameters():
                p.fill_(4.0)

        checkpoint = {
            CHECKPOINT_KEY_MODEL: m1.state_dict(),
            CHECKPOINT_KEY_TARGET_MODEL: m2.state_dict(),
        }

        new_m1 = DummyModule()
        new_m2 = DummyModule()

        apply_checkpoint_models(
            checkpoint,
            {
                CHECKPOINT_KEY_MODEL: new_m1,
                CHECKPOINT_KEY_TARGET_MODEL: new_m2,
            },
        )

        for p in new_m1.parameters():
            self.assertTrue(torch.all(p == 3.0))
        for p in new_m2.parameters():
            self.assertTrue(torch.all(p == 4.0))

    def test_resolve_checkpoint_save_path(self) -> None:
        self.assertEqual(
            resolve_checkpoint_save_path("/tmp/save.pt", "/tmp/load.pt", is_eval=False),
            "/tmp/save.pt",
        )
        self.assertEqual(
            resolve_checkpoint_save_path(None, "/tmp/load.pt", is_eval=False),
            "/tmp/load.pt",
        )
        self.assertIsNone(
            resolve_checkpoint_save_path(None, "/tmp/load.pt", is_eval=True),
        )
        self.assertIsNone(
            resolve_checkpoint_save_path(None, None, is_eval=False),
        )

    def test_compute_total_episodes(self) -> None:
        self.assertEqual(compute_total_episodes(None, 5), 5)
        self.assertEqual(compute_total_episodes(10, 4), 14)
        self.assertEqual(compute_total_episodes(0, 0), 0)

    def test_graceful_interrupt_handler_signal(self) -> None:
        handler = GracefulInterruptHandler()
        self.assertFalse(handler.interrupted)

        # First signal sets interrupted
        handler._handle_signal(signal.SIGINT, None)
        self.assertTrue(handler.interrupted)

        # Second signal escalates to KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            handler._handle_signal(signal.SIGINT, None)

    def test_graceful_interrupt_handler_context(self) -> None:
        with GracefulInterruptHandler() as handler:
            self.assertFalse(handler.interrupted)


class TestCheckpointIntegration(unittest.TestCase):
    """Single IO integration test testing saving and loading from filesystem (Rule 3)."""

    @unittest.skipUnless(HAS_TORCH, "Requires PyTorch")
    def test_save_and_load_checkpoint_integration(self) -> None:
        model = DummyModule()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        checkpoint_data = create_checkpoint(
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            episodes=10,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = f"{tmpdir}/nested/dir/checkpoint.pt"

            # Save checkpoint (creates dirs and writes file)
            save_checkpoint(file_path, checkpoint_data)

            # Load checkpoint
            loaded = load_checkpoint(file_path, device=torch.device("cpu"))

            new_model = DummyModule()
            metadata = apply_checkpoint_state(loaded, model=new_model)

            self.assertEqual(metadata.get(CHECKPOINT_KEY_EPISODES), 10)
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))

            # Non-existent file raises FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                load_checkpoint(f"{tmpdir}/nonexistent.pt")


if __name__ == "__main__":
    unittest.main()
