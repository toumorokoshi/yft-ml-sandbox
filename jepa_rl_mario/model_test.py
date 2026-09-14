"""Unit tests for MarioModel and package imports."""

from __future__ import annotations

import unittest

import torch

from jepa_rl_mario.model import MarioModel


class TestMarioModel(unittest.TestCase):
    """Tests verifying architecture and forward pass of MarioModel."""

    def test_forward_output_shape(self) -> None:
        batch_size = 4
        num_actions = 7
        model = MarioModel(num_actions=num_actions)
        # Input shape: (batch, 1, 80, 80)
        inputs = torch.randn(batch_size, 1, 80, 80)
        outputs = model(inputs)
        self.assertEqual(outputs.shape, (batch_size, num_actions))

    def test_package_exports(self) -> None:
        from jepa_rl_mario import MarioEnv, MarioModel as PkgMarioModel, RenderMode

        self.assertIs(PkgMarioModel, MarioModel)
        self.assertTrue(issubclass(MarioEnv, object))
        self.assertTrue(issubclass(RenderMode, object))


if __name__ == "__main__":
    unittest.main()
