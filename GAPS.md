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

## 2. JEPA RL Mario Environment Baseline & Agent Training

- **Status**: In Progress
- **Description**: A Gymnasium RL environment scaffold and baseline DQN training loop have been created under `jepa_rl_mario` using `MarioModel`. Hardware acceleration via `yft_utils.detect_device` has been enabled across NVIDIA, AMD ROCm, and Apple Metal MPS. Checkpointing and graceful `KeyboardInterrupt` handling have been refactored into the shared `yft_utils.checkpoint` module to ensure model state persistence, resume capabilities, and episode completion across any training workflow.
- **Next Steps**:
  - Implement ViT attention layers and patch embeddings for visual state representation.
  - Design and train the JEPA model on the Mario track states to learn robust representation embeddings.
  - Implement planning/control loops based on the JEPA representation.


## 3. Dynamic Multi-GPU Backend Pip Resolution on Linux (CUDA vs ROCm)

- **Status**: Resolved
- **Description**: On Linux, NVIDIA and AMD share the `linux_x86_64` OS/Arch tuple, but require different PyTorch wheels (compiled against CUDA vs ROCm). `rules_python` platform mapping evaluates per-platform lockfiles by OS and architecture.
- **Resolution**:
  - Implemented dual pip hubs in `MODULE.bazel`: `@pypi` (standard PyPI / CUDA wheels using `requirements_lock_cuda.txt`) and `@pypi_rocm` (PyTorch ROCm 6.4 wheels using `requirements_lock_rocm.txt`).
  - Added build setting `--//:gpu_backend` in the root `BUILD.bazel` with `:is_rocm_backend` and `:is_cuda_backend` config settings.
  - Linked `.bazelrc` config flags `--config=rocm` and `--config=cuda` to `--//:gpu_backend=rocm` and `--//:gpu_backend=cuda`.
  - Exposed root aliases `//:torch` and `//:torchvision` with `actual = select({":is_rocm_backend": "@pypi_rocm//...", "//conditions:default": "@pypi//..."})`. Targets across the workspace consume `//:torch` and dynamically receive the matching accelerator package.

## 4. Triton Kernel Metal Backend for macOS

- **Status**: Open
- **Description**: OpenAI Triton currently compiles to PTX (NVIDIA) and AMD GCN/RDNA assembly (ROCm). On macOS Apple Silicon, Triton kernels are not natively compilable to Metal shading language. Scripts like `triton_from_onnx` gracefully fall back to the PyTorch native reference execution on `mps`.
- **Next Steps**:
  - Explore integration with Apple MLX or custom Metal compute shaders for direct kernel parity on Apple Silicon.
