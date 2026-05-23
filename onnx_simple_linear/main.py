import os
import sys
import argparse
from typing import Sequence
from simple_linear_onnx.model import create_model, generate_dummy_input, export_to_onnx_bytes

DEFAULT_OUTPUT_PATH = "model.onnx"

def save_bytes_to_file(data: bytes, file_path: str) -> None:
    """IO wrapper for writing bytes to file."""
    resolved_path = os.path.expanduser(file_path)
    if not os.path.isabs(resolved_path) and "BUILD_WORKSPACE_DIRECTORY" in os.environ:
        resolved_path = os.path.join(os.environ["BUILD_WORKSPACE_DIRECTORY"], resolved_path)
    with open(resolved_path, "wb") as f:
        f.write(data)

def export_model_pipeline(output_path: str) -> None:
    """Inner function for the pipeline (composed of pure/data functions and wrapper IO)."""
    model = create_model()
    dummy_input = generate_dummy_input()
    onnx_bytes = export_to_onnx_bytes(model, dummy_input)
    save_bytes_to_file(onnx_bytes, output_path)

def parse_arguments(args: Sequence[str]) -> argparse.Namespace:
    """Helper to parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export a simple linear PyTorch model to ONNX.")
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to save the exported ONNX model (default: {DEFAULT_OUTPUT_PATH})"
    )
    return parser.parse_args(args)

def main(args: Sequence[str] = sys.argv[1:]) -> None:
    parsed_args = parse_arguments(args)
    export_model_pipeline(parsed_args.output)
    print(f"ONNX model successfully exported to: {parsed_args.output}")

if __name__ == "__main__":
    main()
