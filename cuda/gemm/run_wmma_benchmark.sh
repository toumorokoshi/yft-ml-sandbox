#!/bin/bash
set -e

echo "Running WMMA & NVFP4 Tensor Core GEMM Benchmark..."

# Find the binary in local directory, bazel-bin, or runfiles
if [ -f "./cuda/gemm/wmma_nvfp4_benchmark" ]; then
    ./cuda/gemm/wmma_nvfp4_benchmark
elif [ -f "./wmma_nvfp4_benchmark" ]; then
    ./wmma_nvfp4_benchmark
elif [ -f "bazel-bin/cuda/gemm/wmma_nvfp4_benchmark" ]; then
    ./bazel-bin/cuda/gemm/wmma_nvfp4_benchmark
else
    # Search for the executable in the current directory tree
    BIN_PATH=$(find . -name wmma_nvfp4_benchmark -type f -executable | head -n 1)
    if [ -n "$BIN_PATH" ]; then
        $BIN_PATH
    else
        echo "Error: wmma_nvfp4_benchmark binary not found."
        exit 1
    fi
fi
