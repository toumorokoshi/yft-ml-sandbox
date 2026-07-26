"""Unit tests for jepa_rl_mario/main.py CLI arguments and simulation functions."""

from __future__ import annotations

import tempfile
import unittest
import numpy as np

from jepa_rl_mario.main import parse_args, save_frame, run_simulation
from jepa_rl_mario.mario_env import MarioEnv


class TestMainScaffolding(unittest.TestCase):
    """Unit tests for main script CLI arguments and helper functions."""

    def test_parse_args_defaults(self) -> None:
        args = parse_args([])
        self.assertEqual(args.steps, 20)
        self.assertEqual(args.episodes, 1)
        self.assertEqual(args.render_mode, "rgb_array")
        self.assertIsNone(args.seed)
        self.assertEqual(args.log_level, "INFO")

    def test_parse_args_custom(self) -> None:
        args = parse_args([
            "--steps", "50",
            "--episodes", "3",
            "--render-mode", "none",
            "--output", "test_out.png",
            "--seed", "42",
            "--log-level", "DEBUG",
        ])
        self.assertEqual(args.steps, 50)
        self.assertEqual(args.episodes, 3)
        self.assertEqual(args.render_mode, "none")
        self.assertEqual(args.output, "test_out.png")
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.log_level, "DEBUG")

    def test_save_frame_none(self) -> None:
        self.assertFalse(save_frame(None, "dummy.png"))

    def test_save_frame_valid(self) -> None:
        mock_frame = np.zeros((240, 256, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            self.assertTrue(save_frame(mock_frame, tmp.name))

    def test_run_simulation(self) -> None:
        env = MarioEnv(render_mode="rgb_array")
        try:
            total_reward, steps_taken, frame = run_simulation(env, steps=3, seed=123)
            self.assertEqual(steps_taken, 3)
            self.assertIsInstance(total_reward, float)
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape, (240, 256, 3))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
