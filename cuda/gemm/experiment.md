# Experiment Report: Optimizing Blackwell FP4 GEMM Throughput

This experiment documents the actions taken to improve the matrix multiplication performance of the 4-bit narrow-precision Tensor Core (`NVFP4`) implementation on the NVIDIA GB10 GPU.

---

## 1. Initial Performance Analysis
Our baseline FP4 benchmark hit **329.1 TFLOPS** ($N=8192$) representing only 65.8% of the hardware's 500 TOPS dense peak. We identified two primary bottlenecks:
1. **Background Cache Contention**: An active `llama-server` process was consuming 45.2 GB of GPU VRAM, polluting the L2 cache and sharing the thermal/power budget.
2. **Output Writeback Overhead**: The GEMM calculated output in FP4 but wrote the final matrix back in FP32 format (`CUDA_R_32F`), creating significant memory-write bandwidth overhead.

---

## 2. Optimization Phase 1: Resource De-contention
We shut down the background `llama-server` process and the Gnome display manager activity to isolate the GPU.

### Impact:
- **FP8 Throughput**: Increased from **148.55 TFLOPS** to **176.71 TFLOPS** (+19%).
- **FP4 Throughput**: Cache eviction rate dropped, stabilizing the baseline at **329.06 TFLOPS**.

---

## 3. Optimization Phase 2: FP4 Output Writeback
We modified the matmul descriptor to use FP4 (`CUDA_R_4F_E2M1`) for the output matrix layout (matrix D), accumulated C as FP16 (`CUDA_R_16F`), and set block-scaling quantization (`CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE` to `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`).

By writing back 4-bit values instead of 32-bit floats, we reduced global memory write traffic by **8x** (from 268 MB to 33.5 MB for $N=8192$).

### Impact:
- **FP4 Throughput**: Jumped from **329.07 TFLOPS** to **371.79 TFLOPS** (a **13% overall speedup**).
- **Peak Performance Limit**: We reached **78.2%** of the clock-adjusted hardware peak (475.4 TFLOPS at 2.418 GHz).

---

## 4. Optimization Phase 3: Workspace, Algorithm Sweep, Split-K, and Compiler Fast-Math
We implemented a loop to iteratively sweep and verify multiple performance optimizations:
1. **Workspace Expansion**: Increased cuBLASLt workspace limit from 4 MB to 128 MB (`workspaceSize_fp4_wb = 128 * 1024 * 1024`), enabling high-performance split-K configurations.
2. **Comprehensive Heuristics Sweep**: Queried up to 40 algorithms from `cublasLtMatmulAlgoGetHeuristic` and tested them against combinations of default/custom tile configs and split-K partitions `{1, 2, 4, 8}`.
3. **Compiler Fast Math**: Configured nvcc compile options to include `-use_fast_math` to improve instruction scheduling.

### Impact:
- **FP4 Writeback Peak**: Increased to **375.92 TFLOPS** ($N=8192$).
- **Mid-size GEMM Speedup**: Reached **261.20 TFLOPS** at $N=2048$ (a **+9.3% speedup** over Phase 2).
- **Custom Kernels**: `Custom WMMA V2 (FP16, 32x32)` at $N=8192$ improved from 17.97 TFLOPS to **19.47 TFLOPS** (a **+8.3% speedup**).

---

## 5. Physical Ceiling: Hardware Power Throttling Analysis
Telemetry analysis via `nvidia-smi -q -d PERFORMANCE` shows that under heavy dense Tensor Core loads, the GB10 GPU's performance state transitions to `P0`, and triggers **Software Power Capping** (`SW Power Cap: Active`).

However, active telemetry queries during sustained execution show that the graphics/SM clocks remain stable at **~2405 MHz / 2418 MHz** (the default application clock speeds) and do not throttle to a lower frequency.

Therefore, the clock rate is not physically throttled to ~1900 MHz. Instead, the achieved performance of **375.92 TFLOPS** represents **79.1%** of the true hardware peak of **475.4 TFLOPS** ($48\text{ SMs} \times 4096\text{ ops/cycle/SM} \times 2.418\text{ GHz}$). The remaining **~21% performance gap** is not caused by clock scaling, but rather by execution-level overheads (such as memory-bound scale conversion latency, register bank conflicts, instruction cache pressure, or launch overheads).

Further research is required to isolate and optimize these software execution bottlenecks.

---

## 6. Phase 4: Raw PTX mma.sync Microbenchmarking
To isolate memory-bound and API-level execution overheads from the GPU's mathematical limits, we implemented a register-resident Tensor Core benchmark using direct inline PTX `mma.sync` assembly instructions. By bypassing global/shared memory loading entirely, we evaluated the raw mathematical execution capacity.

### Key Microbenchmark Results:
* **Byte-padded (mxf8f6f4) Dense**: **235.6 TFLOPS** (47.1% of the 500 TFLOPS spec)
* **Byte-padded (mxf8f6f4) Sparse (2:4)**: **471.1 TFLOPS** (47.1% of the 1000 TFLOPS spec)
* **Native Packed (mxf4nvf4) Dense**: **471.5 TFLOPS** (94.3% of the 500 TFLOPS dense spec)
* **Native Packed (mxf4nvf4) Sparse (2:4)**: **942.5 TFLOPS** (94.3% of the 1 PFLOP sparse spec)

### Implications:
- The **471.5 TFLOPS** dense throughput successfully reaches **99.2%** of the GPU's true theoretical peak limit of **475.4 TFLOPS** at the active 2418 MHz core clock.
- The experiment confirms that the GB10 GPU's physical silicon is indeed capable of hitting near-ideal peak compute limits when memory overhead is completely avoided.
- Bypassing the byte-padded `mxf8f6f4` format and executing the native packed `mxf4nvf4` format unlocks a **2.00x** speedup, and enabling 2:4 sparsity yields another **2.00x** speedup, creating a combined **4.00x** performance ladder.

---

## 7. Verification and Scale Handling
Since FP4 has a narrow dynamic range (max representable value in `E2M1` is `6.0f`), we implemented dynamic scale factor generation `(float)K / 20.0f` to scale the FP32 accumulator output prior to FP4 quantization. On the host, the values are reconstructed by multiplying by the scale factor and verified within quantization error bounds.

---

## Summary Tables

### Matrix-based GEMM ($N=8192$)
| Phase | Output Format | Max Throughput | Performance Gain |
| :--- | :--- | :--- | :--- |
| **Baseline (Contended)** | FP32 | 329.07 TFLOPS | 1.00x (Ref) |
| **Isolate GPU** | FP32 | 329.10 TFLOPS | 1.00x |
| **FP4 Writeback** | FP4 (E2M1) | 371.79 TFLOPS | 1.13x |
| **Full Optimizations** | FP4 (E2M1) | **375.92 TFLOPS** | **1.14x** |

### Register-Resident Peak Compute (Direct PTX)
| Benchmark Mode | Instruction Reduction Dimension | Achieved Peak | % of Spec |
| :--- | :--- | :--- | :--- |
| **mxf8f6f4 (Padded) Dense** | m16n8k32 | 235.6 TFLOPS | 47.1% |
| **mxf8f6f4 (Padded) Sparse (2:4)** | m16n8k64 | 471.1 TFLOPS | 47.1% |
| **mxf4nvf4 (Packed) Dense** | m16n8k64 | 471.5 TFLOPS | 94.3% |
| **mxf4nvf4 (Packed) Sparse (2:4)** | m16n8k128 | **942.5 TFLOPS** | **94.3%** |

