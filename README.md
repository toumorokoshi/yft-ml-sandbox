# yft-ml-sandbox
yft's experiments with ml

## Hardware Accelerator & Platform Support

This project supports execution across **macOS (Apple Silicon Metal MPS)**, **AMD GPUs (ROCm)**, and **NVIDIA GPUs (CUDA)**, using **Bazel**.

### 1. macOS (Apple Silicon / Metal Performance Shaders)

On macOS, PyTorch natively leverages Apple Silicon GPUs via Metal Performance Shaders (`mps`).

```bash
# Run training using Metal (MPS) acceleration:
bazel run //jepa_rl_mario:train -- --render-mode none --episodes 5

# Or explicitly specify device:
bazel run //jepa_rl_mario:train -- --device mps
```

The build configuration automatically sets `PYTORCH_ENABLE_MPS_FALLBACK=1` on macOS to gracefully fall back to CPU for any operators not implemented on MPS.

### 2. AMD GPUs (ROCm / HIP)

For AMD GPUs (RDNA 2, RDNA 3, RDNA 3.5 APUs, CDNA):

```bash
# Run with ROCm configuration shortcut:
bazel run --config=rocm //jepa_rl_mario:train -- --render-mode none

# Or test alexnet:
bazel run --config=rocm //alexnet:main
```

#### Architecture Override (gfx1150 / Strix Point APU / RX 7600)
If using an APU or GPU requiring an architecture override (such as gfx1150), set:
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```
This override is automatically injected when using `--config=rocm`. Ensure your user belongs to the `render` and `video` groups to access `/dev/kfd` and `/dev/dri/renderD*`.

### 3. NVIDIA GPUs (CUDA)

For NVIDIA GPUs:

```bash
# Run with CUDA configuration:
bazel run --config=cuda //jepa_rl_mario:train -- --render-mode none

# Run CUDA GEMM benchmarks:
bazel build //cuda/gemm:gemm_benchmark_bin
bazel test //cuda/gemm:gemm_benchmark_test --test_output=all
```

---

## Hardware Discovery

Use `gpu_device_info` to inspect hardware across all platforms:

```bash
# Run multi-platform GPU discovery tests:
bazel test //gpu_device_info:all
```

---

## Updating Dependencies

- **Linux (AMD ROCm / NVIDIA CUDA)**: `bazel run //:requirements.update`
- **macOS (Apple Silicon)**: `bazel run //:requirements_darwin.update`

---

## Quality & Tests

```bash
# Run lint and unit tests:
just lint

# Run all Bazel unit tests:
bazel test //yft_utils:all //gpu_device_info:all
```