import onnx
from onnx import helper, TensorProto
import os

# Constants
GEMM_OUTPUT_PATH = "gemm.onnx"
M, N, K = 1024, 1024, 1024

def create_large_gemm_onnx_model(m: int = M, n: int = N, k: int = K) -> onnx.ModelProto:
    """Inner function: Constructs a Gemm ONNX ModelProto on data structures."""
    a_val_info = helper.make_tensor_value_info("A", TensorProto.FLOAT, [m, k])
    b_val_info = helper.make_tensor_value_info("B", TensorProto.FLOAT, [k, n])
    c_val_info = helper.make_tensor_value_info("C", TensorProto.FLOAT, [n])
    y_val_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [m, n])
    
    gemm_node = helper.make_node(
        "Gemm",
        inputs=["A", "B", "C"],
        outputs=["Y"],
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=0,
    )
    
    graph = helper.make_graph(
        nodes=[gemm_node],
        name="gemm_graph",
        inputs=[a_val_info, b_val_info, c_val_info],
        outputs=[y_val_info],
    )
    
    model = helper.make_model(graph, producer_name="triton_from_onnx_gemm")
    model.opset_import[0].version = 13
    return model

def save_onnx_model_to_file(model: onnx.ModelProto, file_path: str) -> None:
    """IO wrapper for saving ONNX model to disk."""
    resolved_path = os.path.expanduser(file_path)
    if not os.path.isabs(resolved_path) and "BUILD_WORKSPACE_DIRECTORY" in os.environ:
        resolved_path = os.path.join(os.environ["BUILD_WORKSPACE_DIRECTORY"], resolved_path)
    onnx.save(model, resolved_path)

def main() -> None:
    model = create_large_gemm_onnx_model()
    save_onnx_model_to_file(model, GEMM_OUTPUT_PATH)
    print(f"Successfully generated Gemm ONNX model at {GEMM_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
