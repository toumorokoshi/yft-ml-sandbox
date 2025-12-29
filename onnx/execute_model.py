#!/usr/bin/env python3
import argparse
import numpy as np
import onnxruntime as ort
import logging

logger = logging.getLogger(__name__)


def get_execution_providers():
    available = ort.get_available_providers()
    if "MIGraphXExecutionProvider" in available:
        return ["MIGraphXExecutionProvider", "CPUExecutionProvider"]
    if "ROCMExecutionProvider" in available:
        return ["ROCMExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    logger.warning("no matching execution provider found; falling back to CPU")
    return ["CPUExecutionProvider"]


def create_session(model_path):
    providers = get_execution_providers()
    print(f"Using execution providers: {providers}")
    return ort.InferenceSession(model_path, providers=providers)


def generate_random_inputs(session):
    return {
        input_meta.name: np.random.randn(*input_meta.shape).astype(np.float32)
        for input_meta in session.get_inputs()
    }


def run_inference(session, inputs):
    output_names = [output.name for output in session.get_outputs()]
    return session.run(output_names, inputs)


def print_model_info(session):
    print("\nModel Inputs:")
    for input_meta in session.get_inputs():
        print(f"  {input_meta.name}: {input_meta.shape} ({input_meta.type})")

    print("\nModel Outputs:")
    for output_meta in session.get_outputs():
        print(f"  {output_meta.name}: {output_meta.shape} ({output_meta.type})")


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run ONNX model with MIGraphX or CPU")
    parser.add_argument("model_path", help="Path to ONNX model file")
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Only load model without running inference",
    )
    args = parser.parse_args()

    session = create_session(args.model_path)
    print_model_info(session)

    if not args.no_run:
        print("\nRunning inference with random inputs...")
        inputs = generate_random_inputs(session)
        outputs = run_inference(session, inputs)

        print("\nOutputs:")
        for i, output in enumerate(outputs):
            print(f"  Output {i}: shape={output.shape}, dtype={output.dtype}")
            print(f"    Sample values: {output.flat[:5]}")


if __name__ == "__main__":
    main()
