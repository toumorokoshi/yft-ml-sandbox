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
6. **Custom WMMA V2 GEMM (FP16)**: An optimized warp-level GEMM using $32 \times 32$ register tiling per warp (holding 4 accumulator fragments) and $64 \times 64$ block tiling. This reduces global memory loads by 50% by reusing loaded fragments across the register tile.
7. **cuBLAS HGEMM (FP16)**: Mixed-precision GEMM utilizing FP16 inputs and FP32 accumulation on Tensor Cores.
8. **cuBLASLt GEMM (FP8)**: Narrow-precision GEMM utilizing E4M3 inputs and FP32 accumulation.
9. **cuBLASLt GEMM (FP4/NVFP4)**: 4-bit floating-point GEMM utilizing E2M1 inputs, FP32 accumulation, and VEC16 block scaling.

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
  - **Custom WMMA V2 GEMM**: $32 \times 4$ threads per block (arranged in a $2 \times 2$ warp grid computing a $64 \times 64$ tile of C).

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
| **cuBLAS SGEMM (FP32)** | 54.40 | 20.21 | PASS |
| **Custom WMMA (FP16)** | 391.23 | 2.81 | PASS |
| **Custom WMMA V2 (FP16, 32x32)** | 56.47 | **19.47** | PASS |
| **cuBLAS HGEMM (FP16)** | 13.19 | 83.37 | PASS |
| **cuBLASLt GEMM (FP8)** | 6.78 | 162.19 | PASS |
| **cuBLASLt FP4 (Tile 128x128)** | 3.39 | 324.73 | PASS |
| **cuBLASLt FP4 (FP4 Writeback)** | 2.93 | **375.92** | PASS |

### Blackwell NVFP4 Tile Size Sweep

When configuring different block tile layouts for cuBLASLt NVFP4 execution, the compatibility scan outputs:
```
Scanning cuBLASLt FP4 manual tile layout compatibility:
  Tile 8x8: Execution failed (Status: 7 - CUBLAS_STATUS_NOT_SUPPORTED)
  Tile 16x16: Execution failed (Status: 7)
  Tile 32x32: Execution failed (Status: 7)
  Tile 64x64: Execution failed (Status: 7)
  Tile 128x64: Execution failed (Status: 7)
  Tile 64x128: Execution failed (Status: 7)
  Tile 128x128: SUCCESS! (Time: 3.338 ms, TFLOPS: 329.43)
  Tile 256x64: Execution failed (Status: 7)
  Tile 64x256: Execution failed (Status: 7)
  Tile 256x128: Execution failed (Status: 7)
  Tile 128x256: Execution failed (Status: 7)
```
*Note: Due to the specialized register mapping and block scaling requirements of Blackwell's 4-bit Tensor Cores, only the `128x128` tile shape is compatible/supported. Other configurations return `CUBLAS_STATUS_NOT_SUPPORTED`.*

### Blackwell NVFP4 Register-Resident Peak (Direct PTX)

To measure the absolute limits of the GPU hardware and verify the 500 TFLOPS dense / 1 PFLOP sparse specs, the `peak_mma` benchmark executes raw inline PTX `mma.sync` and `mma.sp` instructions directly from register fragments (avoiding all memory latency and cache contention overheads).

| Mode | Reduction Tile | Achieved Peak | % of Spec |
| :--- | :--- | :--- | :--- |
| **mxf8f6f4 dense** | m16n8k32 | 235.6 TFLOPS | 47.1% |
| **mxf8f6f4 sparse (2:4)** | m16n8k64 | 471.1 TFLOPS | 47.1% |
| **mxf4nvf4 dense** | m16n8k64 | 471.5 TFLOPS | **94.3%** |
| **mxf4nvf4 sparse (2:4)** | m16n8k128 | **942.5 TFLOPS** | **94.3%** |

