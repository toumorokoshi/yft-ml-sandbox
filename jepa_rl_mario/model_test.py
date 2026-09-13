"""Unit tests for MarioModel."""

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


if __name__ == "__main__":
    unittest.main()
