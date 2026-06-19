import os
import tempfile
import unittest
import onnx
import torch
from onnx_simple_linear.model import create_model, generate_dummy_input, export_to_onnx_bytes, INPUT_DIM, OUTPUT_DIM
from onnx_simple_linear.main import save_bytes_to_file

class TestSimpleLinearONNX(unittest.TestCase):
    def test_model_creation(self) -> None:
        """Test model architecture on data structures directly."""
        model = create_model(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
        self.assertIsInstance(model.linear, torch.nn.Linear)
        self.assertEqual(model.linear.in_features, INPUT_DIM)
        self.assertEqual(model.linear.out_features, OUTPUT_DIM)

    def test_dummy_input_generation(self) -> None:
        """Test dummy input shape and type on data structures directly."""
        dummy = generate_dummy_input(batch_size=2, input_dim=INPUT_DIM)
        self.assertEqual(dummy.shape, (2, INPUT_DIM))
        self.assertEqual(dummy.dtype, torch.float32)

    def test_export_to_onnx_bytes(self) -> None:
        """Test ONNX export directly in memory without file IO."""
        model = create_model()
        dummy = generate_dummy_input()
        onnx_bytes = export_to_onnx_bytes(model, dummy)
        
        # Verify it returns non-empty bytes
        self.assertGreater(len(onnx_bytes), 0)
        
        # Parse using onnx library to verify it is a valid ONNX graph
        onnx_model = onnx.load_from_string(onnx_bytes)
        onnx.checker.check_model(onnx_model)
        
        # Check inputs/outputs details in the graph structure
        graph = onnx_model.graph
        self.assertEqual(len(graph.input), 1)
        self.assertEqual(graph.input[0].name, "input")
        self.assertEqual(len(graph.output), 1)
        self.assertEqual(graph.output[0].name, "output")

    def test_export_integration_io(self) -> None:
        """Single integration test verifying file IO."""
        model = create_model()
        dummy = generate_dummy_input()
        onnx_bytes = export_to_onnx_bytes(model, dummy)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_file_path = os.path.join(tmpdir, "test_model.onnx")
            save_bytes_to_file(onnx_bytes, temp_file_path)
            
            self.assertTrue(os.path.exists(temp_file_path))
            self.assertGreater(os.path.getsize(temp_file_path), 0)
            
            # Load and verify it
            loaded_model = onnx.load(temp_file_path)
            onnx.checker.check_model(loaded_model)

if __name__ == "__main__":
    unittest.main()
