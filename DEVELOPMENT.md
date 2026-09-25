# Development Guide

## Environment Setup

### Multi-Platform Accelerator Support

This repository is configured to build and run across three hardware accelerator backends:
1. **macOS (Apple Silicon / Metal MPS)**: PyTorch wheels from standard PyPI with native `mps` support.
2. **Linux AMD ROCm**: PyTorch wheels built with ROCm HIP support (`https://download.pytorch.org/whl/rocm6.4`).
3. **Linux NVIDIA CUDA**: PyTorch wheels with CUDA support.

### Setting up VSCode

1. Use the `Python: Create Environment` command in VSCode.
2. Select `.venv` and install from `requirements.txt`.
3. If running on an AMD GPU with architecture `gfx1150`, add `export HSA_OVERRIDE_GFX_VERSION=11.0.0` to your shell profile or activate script.

---

## Running with Bazel

### macOS (Apple Silicon)
```bash
# Run targets using MPS
bazel run //jepa_rl_mario:train -- --render-mode none
bazel run //alexnet:main
```

### Linux AMD GPU (ROCm)
```bash
# Ensure user is in render and video groups
sudo usermod -aG render,video $USER

# Run targets using ROCm
bazel run --config=rocm //jepa_rl_mario:train -- --render-mode none
bazel run --config=rocm //alexnet:main
```

### Linux NVIDIA GPU (CUDA)
```bash
# Run targets using CUDA
bazel run --config=cuda //jepa_rl_mario:train -- --render-mode none
bazel run --config=cuda //alexnet:main
```

---

## Updating Locked Dependencies

- **Linux**: `bazel run //:requirements.update`
- **macOS (Darwin)**: `bazel run //:requirements_darwin.update`

---

## Code Quality

```bash
just fix   # Auto-formatting and lint fixing
just lint  # Run unit tests across all platform modules
```