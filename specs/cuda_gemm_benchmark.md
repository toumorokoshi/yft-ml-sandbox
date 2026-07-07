# Specification: CUDA GEMM Benchmark

This specification describes the design and implementation of the CUDA General Matrix Multiplication (GEMM) benchmark, including high-performance Tensor Core execution modes.

## Goal

To measure and compare the floating-point performance (FLOPS) of custom CUDA implementations (tiled and WMMA-based) versus highly optimized vendor-provided cuBLAS and cuBLASLt implementations on the system's NVIDIA GPU.

## Design

The benchmark compares:
1. **Host-based validation**: A CPU-based verification on a small sub-matrix to ensure correctness of the GPU computations.
2. **Custom Tiled GEMM (FP32)**: A custom CUDA kernel using static shared memory tiling to optimize global memory access.
3. **cuBLAS SGEMM (FP32)**: Single-precision GEMM executing standard FP32 math.
4. **cuBLAS TF32 GEMM**: Single-precision GEMM configured to use TensorFloat-32 execution on Tensor Cores.
5. **Custom WMMA GEMM (FP16)**: A warp-level C++ GEMM kernel utilizing `nvcuda::wmma` fragments ($16 \times 16 \times 16$ tile) to perform mixed-precision multiplication on Tensor Cores.
6. **cuBLAS HGEMM (FP16)**: Mixed-precision GEMM utilizing FP16 inputs and FP32 accumulation on Tensor Cores.
7. **cuBLASLt GEMM (FP8)**: Narrow-precision GEMM utilizing E4M3 inputs and FP32 accumulation.
8. **cuBLASLt GEMM (FP4/NVFP4)**: 4-bit floating-point GEMM utilizing E2M1 inputs, FP32 accumulation, and VEC16 block scaling.

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
- **Thread Block Size**:
  - **Custom Tiled GEMM**: $16 \times 16$ or $32 \times 32$ threads per block.
  - **Custom WMMA GEMM**: $32 \times 4$ threads per block (where X=32 is warp size, Y=4 is warps per block).

## Compilation and Build

Compiled using `nvcc` via a custom Bazel `genrule` to target the platform compiler directly without heavy toolchain setup.
- Optimizations: `-O3`
- Target Architecture: `-gencode arch=compute_121,code=sm_121` (required to JIT compile Blackwell-specific low-precision PTX instruction sequences).
- Libraries linked: `lcublas`, `lcublasLt`, `lcudart`

## Benchmark Results (NVIDIA GB10 Blackwell GPU)

Below is the observed performance for square matrices on the system's GB10 GPU (Compute Capability 12.1, 48 SMs):

### Square Matrix N = 8192

| Implementation | Time (ms) | TFLOPS | Status |
| :--- | :--- | :--- | :--- |
| **cuBLAS SGEMM (FP32)** | 54.05 | 20.34 | PASS |
| **Custom WMMA (FP16)** | 463.73 | 2.37 | PASS |
| **cuBLAS HGEMM (FP16)** | 12.19 | 90.23 | PASS |
| **cuBLASLt GEMM (FP8)** | 7.43 | 147.96 | PASS |
| **cuBLASLt GEMM (FP4/NVFP4)** | 3.37 | 326.30 | PASS |

*Note: The C++ `nvcuda::wmma` API does not natively support FP4 (`__nv_fp4_e2m1`) types at compile-time. High-throughput NVFP4 Tensor Core execution is instead enabled through `cuBLASLt` using block-scaled format qualifiers (`CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`).*
