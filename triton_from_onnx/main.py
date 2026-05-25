import argparse
import sys
import os
import onnx
from onnx import numpy_helper
import torch

from triton_from_onnx.interpreter import (
    run_onnx_with_triton,
    load_onnx_model_from_file,
)
from yft_utils.timeit import timeit

# Constants
DEFAULT_MODEL_PATH = "model.onnx"


def run_onnx_with_pytorch(
    model: onnx.ModelProto,
    inputs: dict[str, torch.Tensor],
    device: str,
) -> dict[str, torch.Tensor]:
    """Inner function: Reference execution using PyTorch native operators on data structures."""
    env = {}
    for name, tensor in inputs.items():
        env[name] = tensor.to(device)

    for init in model.graph.initializer:
        np_arr = numpy_helper.to_array(init)
        env[init.name] = torch.from_numpy(np_arr).to(device)

    for node in model.graph.node:
        node_inputs = [env[name] for name in node.input]
        
        if node.op_type == "Add":
            env[node.output[0]] = node_inputs[0] + node_inputs[1]
        elif node.op_type == "Mul":
            env[node.output[0]] = node_inputs[0] * node_inputs[1]
        elif node.op_type == "Gemm":
            attrs = {attr.name: attr for attr in node.attribute}
            alpha = attrs["alpha"].f if "alpha" in attrs else 1.0
            beta = attrs["beta"].f if "beta" in attrs else 1.0
            transA = attrs["transA"].i if "transA" in attrs else 0
            transB = attrs["transB"].i if "transB" in attrs else 0
            
            A, B = node_inputs[0], node_inputs[1]
            if transA == 1:
                A = A.t()
            if transB == 1:
                B = B.t()
            out = alpha * torch.matmul(A, B)
            if len(node_inputs) > 2:
                C = node_inputs[2]
                out = out + beta * C
            env[node.output[0]] = out
        else:
            raise NotImplementedError(
                f"ONNX op {node.op_type} is not supported by the reference interpreter."
            )

    outputs = {}
    for out_val in model.graph.output:
        outputs[out_val.name] = env[out_val.name]
    return outputs


def generate_inputs_from_graph(
    model: onnx.ModelProto,
    device: str,
) -> dict[str, torch.Tensor]:
    """Inner function: Automatically generates inputs on the device from model metadata."""
    initializer_names = {init.name for init in model.graph.initializer}
    inputs = {}
    
    for inp in model.graph.input:
        if inp.name in initializer_names:
            continue
            
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            else:
                # Default dynamic dimensions to 1
                shape.append(1)
                
        # Generate random inputs for the model (assuming float32)
        inputs[inp.name] = torch.randn(shape, dtype=torch.float32, device=device)
        
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run arbitrary ONNX model using Triton on GPU."
    )
    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the ONNX model file (default: {DEFAULT_MODEL_PATH})",
    )
    args = parser.parse_args()

    # 1. Choose the device
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Detected device: {device}")
    if device == "cpu":
        print("Triton requires a GPU (CUDA or ROCm) to run. Cannot execute on CPU.")
        sys.exit(1)

    # 2. Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: ONNX model file not found at {args.model_path}")
        sys.exit(1)

    # 3. Read ONNX model from file path (using wrapper)
    print(f"Loading ONNX model from: {args.model_path}")
    loaded_model = load_onnx_model_from_file(args.model_path)
    print("Loaded ONNX model successfully.")

    # 4. Generate inputs automatically
    inputs = generate_inputs_from_graph(loaded_model, device)

    print("\n--- Inputs ---")
    for name, tensor in inputs.items():
        print(f"Input '{name}': shape={tensor.shape}, dtype={tensor.dtype}")

    # 5. Run the model using Triton-backed ONNX interpreter
    print("\nExecuting ONNX model using Triton...")
    try:
        timed_triton = timeit(run_onnx_with_triton)
        triton_outputs, triton_time = timed_triton(loaded_model, inputs, device)
        print(f"Triton execution time: {triton_time:.6f} seconds")
    except Exception as e:
        print(f"Error during Triton execution: {e}")
        sys.exit(1)

    # 6. Verify correctness against PyTorch reference interpreter
    print("Executing ONNX model using PyTorch reference interpreter...")
    try:
        timed_pytorch = timeit(run_onnx_with_pytorch)
        ref_outputs, pytorch_time = timed_pytorch(loaded_model, inputs, device)
        print(f"PyTorch reference execution time: {pytorch_time:.6f} seconds")
    except Exception as e:
        print(f"Error during reference execution: {e}")
        sys.exit(1)

    # 7. Compare results
    print("\n--- Outputs Comparison ---")
    all_correct = True
    for name in triton_outputs.keys():
        t_out = triton_outputs[name]
        r_out = ref_outputs[name]
        is_correct = torch.allclose(t_out, r_out, atol=1e-4, rtol=1e-4)
        print(f"Output '{name}': Triton shape={t_out.shape}, PyTorch shape={r_out.shape}")
        if is_correct:
            print(f"  Result match: PASS")
        else:
            print(f"  Result match: FAIL")
            all_correct = False

    if all_correct:
        print("\nSUCCESS: Triton output matches PyTorch reference execution!")
    else:
        print("\nFAILURE: Triton output does not match PyTorch reference execution!")
        sys.exit(1)


if __name__ == "__main__":
    main()
