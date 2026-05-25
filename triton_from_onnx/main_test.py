import unittest
import torch
import onnx
from onnx import helper, TensorProto
import tempfile
import os

from triton_from_onnx.interpreter import (
    triton_add,
    triton_mul,
    triton_matmul,
    run_onnx_with_triton,
    load_onnx_model_from_file,
)
def create_gemm_onnx_model() -> onnx.ModelProto:
    """Helper: Constructs an ONNX ModelProto containing a Gemm node in-memory for testing."""
    a_val_info = helper.make_tensor_value_info("A", TensorProto.FLOAT, [4, 8])
    b_val_info = helper.make_tensor_value_info("B", TensorProto.FLOAT, [8, 6])
    c_val_info = helper.make_tensor_value_info("C", TensorProto.FLOAT, [6])
    y_val_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 6])
    
    gemm_node = helper.make_node(
        "Gemm",
        inputs=["A", "B", "C"],
        outputs=["Y"],
        alpha=1.5,
        beta=0.5,
        transA=0,
        transB=0,
    )
    
    graph = helper.make_graph(
        nodes=[gemm_node],
        name="gemm_graph",
        inputs=[a_val_info, b_val_info, c_val_info],
        outputs=[y_val_info],
    )
    
    model = helper.make_model(graph, producer_name="triton_from_onnx_test")
    model.opset_import[0].version = 13
    return model


def save_onnx_model_to_file(model: onnx.ModelProto, file_path: str) -> None:
    """Helper: Saves an ONNX model proto to a file for testing."""
    expanded_path = os.path.expanduser(file_path)
    os.makedirs(os.path.dirname(expanded_path), exist_ok=True)
    onnx.save(model, expanded_path)



class TestTritonFromOnnx(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = (
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available()
            else "cpu"
        )

    def test_triton_add_data_structures(self):
        """Test triton_add directly on PyTorch GPU data structures."""
        if self.device == "cpu":
            self.skipTest("Triton requires a GPU")
            
        x = torch.randn((4, 4), dtype=torch.float32, device=self.device)
        y = torch.randn((4, 4), dtype=torch.float32, device=self.device)
        
        result = triton_add(x, y)
        expected = x + y
        
        self.assertTrue(torch.allclose(result, expected, atol=1e-4, rtol=1e-4))

    def test_triton_mul_data_structures(self):
        """Test triton_mul directly on PyTorch GPU data structures."""
        if self.device == "cpu":
            self.skipTest("Triton requires a GPU")
            
        x = torch.randn((4, 4), dtype=torch.float32, device=self.device)
        y = torch.randn((4, 4), dtype=torch.float32, device=self.device)
        
        result = triton_mul(x, y)
        expected = x * y
        
        self.assertTrue(torch.allclose(result, expected, atol=1e-4, rtol=1e-4))

    def test_triton_matmul_data_structures(self):
        """Test triton_matmul directly on PyTorch GPU data structures."""
        if self.device == "cpu":
            self.skipTest("Triton requires a GPU")
            
        a = torch.randn((8, 16), dtype=torch.float32, device=self.device)
        b = torch.randn((16, 8), dtype=torch.float32, device=self.device)
        
        result = triton_matmul(a, b)
        expected = torch.matmul(a, b)
        
        self.assertTrue(torch.allclose(result, expected, atol=1e-4, rtol=1e-4))

    def test_run_onnx_with_triton_add_in_memory(self):
        """Test run_onnx_with_triton on an in-memory Add ONNX graph."""
        if self.device == "cpu":
            self.skipTest("Triton requires a GPU")
            
        # Create an Add node ONNX graph in-memory
        x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 3])
        y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 3])
        out_info = helper.make_tensor_value_info("Out", TensorProto.FLOAT, [3, 3])
        
        node = helper.make_node("Add", inputs=["X", "Y"], outputs=["Out"])
        graph = helper.make_graph([node], "add_graph", [x_info, y_info], [out_info])
        model = helper.make_model(graph)
        model.opset_import[0].version = 13
        
        inputs = {
            "X": torch.randn((3, 3), dtype=torch.float32, device=self.device),
            "Y": torch.randn((3, 3), dtype=torch.float32, device=self.device),
        }
        
        outputs = run_onnx_with_triton(model, inputs, self.device)
        result = outputs["Out"]
        expected = inputs["X"] + inputs["Y"]
        
        self.assertTrue(torch.allclose(result, expected, atol=1e-4, rtol=1e-4))

    def test_run_onnx_with_triton_gemm_in_memory(self):
        """Test run_onnx_with_triton on an in-memory Gemm ONNX graph."""
        if self.device == "cpu":
            self.skipTest("Triton requires a GPU")
            
        model = create_gemm_onnx_model()
        inputs = {
            "A": torch.randn((4, 8), dtype=torch.float32, device=self.device),
            "B": torch.randn((8, 6), dtype=torch.float32, device=self.device),
            "C": torch.randn((6,), dtype=torch.float32, device=self.device),
        }
        
        outputs = run_onnx_with_triton(model, inputs, self.device)
        result = outputs["Y"]
        expected = 1.5 * torch.matmul(inputs["A"], inputs["B"]) + 0.5 * inputs["C"]
        
        self.assertTrue(torch.allclose(result, expected, atol=1e-4, rtol=1e-4))

    def test_integration_io(self):
        """Integration test writing the ONNX model to disk, reading it, and running it."""
        if self.device == "cpu":
            self.skipTest("Triton requires a GPU")
            
        model = create_gemm_onnx_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_file_path = os.path.join(tmpdir, "gemm_temp.onnx")
            
            # Write to disk using IO wrapper
            save_onnx_model_to_file(model, model_file_path)
            
            # Read from disk using IO wrapper
            loaded_model = load_onnx_model_from_file(model_file_path)
            
            inputs = {
                "A": torch.randn((4, 8), dtype=torch.float32, device=self.device),
                "B": torch.randn((8, 6), dtype=torch.float32, device=self.device),
                "C": torch.randn((6,), dtype=torch.float32, device=self.device),
            }
            
            outputs = run_onnx_with_triton(loaded_model, inputs, self.device)
            result = outputs["Y"]
            expected = 1.5 * torch.matmul(inputs["A"], inputs["B"]) + 0.5 * inputs["C"]
            
            self.assertTrue(torch.allclose(result, expected, atol=1e-4, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
