#!/bin/bash
set -e

echo "Running CUDA raw PTX MMA Tensor Core Peak Benchmark..."

# Find the binary in local directory, bazel-bin, or runfiles
if [ -f "./cuda/gemm/peak_mma" ]; then
    ./cuda/gemm/peak_mma
elif [ -f "./peak_mma" ]; then
    ./peak_mma
elif [ -f "bazel-bin/cuda/gemm/peak_mma" ]; then
    ./bazel-bin/cuda/gemm/peak_mma
else
    # Search for the executable in the current directory tree
    BIN_PATH=$(find . -name peak_mma -type f -executable | head -n 1)
    if [ -n "$BIN_PATH" ]; then
        $BIN_PATH
    else
        echo "Error: peak_mma binary not found."
        exit 1
    fi
fi
