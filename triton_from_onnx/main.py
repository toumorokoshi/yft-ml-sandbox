import argparse
import sys
import os
import onnx
from onnx import numpy_helper
import torch

from yft_utils.timeit import timeit


# Constants
DEFAULT_MODEL_PATH = "model.onnx"
NUM_TRIALS = 10
PROFILE_DIR = "/tmp"


def save_profile_trace(prof: torch.profiler.profile, path: str) -> None:
    """IO Wrapper: Saves the profiler trace to a file path."""
    prof.export_chrome_trace(path)


def profile_function(
    func,
    *args,
    **kwargs,
):
    """Inner function: Runs the function under torch.profiler and returns result and profiler."""
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        result = func(*args, **kwargs)
    return result, prof



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


def str_to_bool(val: str) -> bool:
    """Helper: Converts string to boolean value."""
    return val.lower() in ("true", "1", "yes", "t", "y")


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
    parser.add_argument(
        "--triton_print_autotuning",
        type=str_to_bool,
        default=False,
        help=f"If true, print triton autotuning",
    )
    args = parser.parse_args()
    if args.triton_print_autotuning:
        os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

    # Delay import of interpreter until environment variable is set
    from triton_from_onnx.interpreter import (
        run_onnx_with_triton,
        load_onnx_model_from_file,
    )

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

    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    triton_profile_path = os.path.join(PROFILE_DIR, f"triton_{model_name}.json")
    pytorch_profile_path = os.path.join(PROFILE_DIR, f"pytorch_{model_name}.json")

    for i in range(NUM_TRIALS):
        is_last_trial = (i == NUM_TRIALS - 1)

        # 5. Run the model using Triton-backed ONNX interpreter
        print(f"\nExecuting ONNX model using Triton... ({i} out of {NUM_TRIALS})")
        try:
            if is_last_trial:
                print(f"Profiling Triton execution (saving trace to {triton_profile_path})...")
                triton_outputs, prof = profile_function(
                    run_onnx_with_triton, loaded_model, inputs, device
                )
                save_profile_trace(prof, triton_profile_path)
            else:
                timed_triton = timeit(run_onnx_with_triton)
                triton_outputs, triton_time = timed_triton(loaded_model, inputs, device)
                print(f"Triton execution time: {triton_time:.6f} seconds")
        except Exception as e:
            print(f"Error during Triton execution: {e}")
            sys.exit(1)

        # 6. Verify correctness against PyTorch reference interpreter
        print("Executing ONNX model using PyTorch reference interpreter...")
        try:
            if is_last_trial:
                print(f"Profiling PyTorch reference execution (saving trace to {pytorch_profile_path})...")
                ref_outputs, prof = profile_function(
                    run_onnx_with_pytorch, loaded_model, inputs, device
                )
                save_profile_trace(prof, pytorch_profile_path)
            else:
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
