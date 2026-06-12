#!/bin/bash
set -e

echo "Running CUDA GEMM Performance Benchmark..."

# Find the binary in local directory, bazel-bin, or runfiles
if [ -f "./cuda/gemm/gemm_benchmark" ]; then
    ./cuda/gemm/gemm_benchmark
elif [ -f "./gemm_benchmark" ]; then
    ./gemm_benchmark
elif [ -f "bazel-bin/cuda/gemm/gemm_benchmark" ]; then
    ./bazel-bin/cuda/gemm/gemm_benchmark
else
    # Search for the executable in the current directory tree
    BIN_PATH=$(find . -name gemm_benchmark -type f -executable | head -n 1)
    if [ -n "$BIN_PATH" ]; then
        $BIN_PATH
    else
        echo "Error: gemm_benchmark binary not found."
        exit 1
    fi
fi
