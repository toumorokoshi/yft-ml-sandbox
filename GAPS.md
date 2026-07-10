# Performance Gaps & Future Optimizations

This document tracks identified performance gaps, pending features, and optimization opportunities in the sandbox.

## 1. Blackwell NVFP4 Dense Tensor Core Performance Gap (~21%)

- **Status**: Open
- **Description**: While the optimized Blackwell FP4 writeback GEMM achieves **375.92 TFLOPS** (representing ~79.1% of the peak theoretical performance of **475.4 TFLOPS** at 2.418 GHz), there is a remaining **~21% performance gap** to the physical ceiling of the GPU.
- **Hypothesized Bottlenecks**:
  1. **Memory-bound Scaling Conversions**: The scale conversion factors (`(float)K / 20.0f`) applied to matrix outputs before quantization could be adding global memory write/read cycles.
  2. **Register Pressures & Bank Conflicts**: Register reuse patterns in custom WMMA V2 kernel tiles may need further tuning.
  3. **Instruction Scheduling / Launch Latencies**: Instruction serialization or launch overheads in cuBLASLt descriptors.
- **Next Steps**:
  - Run profile sweeps via NVIDIA Nsight Compute (`ncu`) to identify cache hit rates, pipeline stalls, and register usage.
  - Implement fused scaling/quantization within a custom kernel to eliminate memory-write overheads.
