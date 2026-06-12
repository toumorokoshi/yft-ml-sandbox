# yft-ml-sandbox
yft's experiments with ml

## Setup

### ROCm GPU Compatibility (gfx1150)

If you have an AMD GPU with architecture gfx1150 (RX 7600/7700 series), you need to set the GPU architecture override:

```bash
source .env.rocm
```

Or manually set before running:
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

This is already configured in the `.venv/bin/activate` script.

## Updating Dependencies

`bazel run //:requirements.update`

## CUDA Benchmarks

For systems with NVIDIA GPUs, we have CUDA-based benchmarks:

### Running the GEMM Benchmark

To compile and run the GEMM benchmark to measure peak FP32, TF32, and FP16 TFLOPS:

```bash
# Build the benchmark using Bazel
bazel build //cuda/gemm:gemm_benchmark_bin

# Run the benchmark
bazel test //cuda/gemm:gemm_benchmark_test --test_output=all
```