import unittest
import torch
import io
import tempfile
import os

from alexnet.main import (
    apply_state_dict,
    export_model_to_onnx_bytes,
    load_state_dict_from_file,
    write_bytes_to_file,
)
from alexnet_core.model import NeuralNetwork


class TestAlexNetExport(unittest.TestCase):
    def test_apply_state_dict_on_data_structures(self):
        """Test apply_state_dict directly on PyTorch model and dictionary data structures."""
        model = NeuralNetwork()
        state_dict = model.state_dict()

        # Modify one weight in the state dict to test that it gets applied
        with torch.no_grad():
            state_dict["linear_stack.2.weight"].fill_(0.5)

        updated_model = apply_state_dict(model, state_dict)
        self.assertTrue(torch.all(updated_model.linear_stack[2].weight == 0.5))

    def test_apply_state_dict_nested_checkpoint(self):
        """Test apply_state_dict with a checkpoint dictionary containing model_state_dict."""
        model = NeuralNetwork()
        state_dict = model.state_dict()
        with torch.no_grad():
            state_dict["linear_stack.2.weight"].fill_(0.7)

        checkpoint = {"model_state_dict": state_dict, "epoch": 10}
        updated_model = apply_state_dict(model, checkpoint)
        self.assertTrue(torch.all(updated_model.linear_stack[2].weight == 0.7))

    def test_apply_state_dict_full_module(self):
        """Test apply_state_dict with a full module object."""
        model = NeuralNetwork()
        source_model = NeuralNetwork()
        with torch.no_grad():
            source_model.linear_stack[2].weight.fill_(0.9)

        updated_model = apply_state_dict(model, source_model)
        self.assertTrue(torch.all(updated_model.linear_stack[2].weight == 0.9))

    def test_export_model_to_onnx_bytes_on_data_structures(self):
        """Test export_model_to_onnx_bytes directly on PyTorch model in-memory."""
        model = NeuralNetwork()
        input_size = (1, 3, 320, 400)
        onnx_bytes = export_model_to_onnx_bytes(model, input_size)

        self.assertGreater(len(onnx_bytes), 0)
        self.assertIsInstance(onnx_bytes, bytes)

    def test_integration_io(self):
        """Integration test demonstrating the filesystem IO wrappers functioning together."""
        model = NeuralNetwork()
        state_dict = model.state_dict()

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = os.path.join(tmpdir, "model.pth")
            onnx_path = os.path.join(tmpdir, "model.onnx")

            # Save state dict using PyTorch directly for the test setup
            torch.save(state_dict, weights_path)

            # Load using our IO wrapper
            loaded_state_dict = load_state_dict_from_file(weights_path)

            # Verify loaded data matches original
            self.assertEqual(list(loaded_state_dict.keys()), list(state_dict.keys()))

            # Apply to model
            model = apply_state_dict(model, loaded_state_dict)

            # Export bytes
            onnx_bytes = export_model_to_onnx_bytes(model, (1, 3, 320, 400))

            # Write using our IO wrapper
            write_bytes_to_file(onnx_bytes, onnx_path)

            # Verify file exists and is non-empty
            self.assertTrue(os.path.exists(onnx_path))
            self.assertGreater(os.path.getsize(onnx_path), 0)


if __name__ == "__main__":
    unittest.main()
