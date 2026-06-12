# Specification: CUDA GEMM Benchmark

This specification describes the design and implementation of the CUDA General Matrix Multiplication (GEMM) benchmark.

## Goal

To measure and compare the floating-point performance (FLOPS) of a custom CUDA implementation versus highly optimized vendor-provided cuBLAS implementations on the system's NVIDIA GPU.

## Design

The benchmark compares:
1. **Host-based validation**: A CPU-based verification on a small sub-matrix to ensure correctness of the GPU computations.
2. **Custom Tiled GEMM (FP32)**: A custom CUDA kernel using static shared memory tiling to optimize global memory access.
3. **cuBLAS SGEMM (FP32)**: Single-precision GEMM executing standard FP32 math.
4. **cuBLAS TF32 GEMM**: Single-precision GEMM configured to use TensorFloat-32 execution on Tensor Cores (if supported by hardware/driver).
5. **cuBLAS HGEMM (FP16)**: Mixed-precision GEMM utilizing FP16 inputs and FP32 accumulation on Tensor Cores.

## Mathematical Formulation

For matrix multiplication $C = A \times B$ where:
- $A$ is of size $M \times K$
- $B$ is of size $K \times N$
- $C$ is of size $M \times N$

The total number of floating-point operations performed is:
$$\text{FLOPs} = 2 \times M \times N \times K$$

The performance in Teraflops (TFLOPS) is computed as:
$$\text{TFLOPS} = \frac{2 \times M \times N \times K}{\text{time in seconds} \times 10^{12}}$$

## Execution Configuration

- **Warm-up Iterations**: 10 (to trigger driver cache initialization, device paging, and clocks ramp-up).
- **Benchmark Iterations**: 50 (timed using CUDA events to guarantee precision).
- **Matrix Sizes**: Square matrices with sizes $N \in \{1024, 2048, 4096, 8192\}$.
- **Thread Block Size**: $16 \times 16$ or $32 \times 32$ threads per block (tiled GEMM).

## Compilation and Build

Compiled using `nvcc` via a custom Bazel `genrule` to target the platform compiler directly without heavy toolchain setup.
- Optimizations: `-O3`
- Libraries linked: `lcublas`, `lcudart`
