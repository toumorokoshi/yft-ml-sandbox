# Multi-Accelerator Support Specification (NVIDIA CUDA, AMD ROCm, Apple Metal)

## 1. Objective

Provide first-class, cross-platform execution of PyTorch models and Python scripts across:
- **NVIDIA GPUs**: CUDA-accelerated via `cuda` device.
- **AMD GPUs**: ROCm/HIP-accelerated via `cuda` device (HIP translation layer).
- **Apple Silicon (macOS)**: Metal-accelerated via `mps` (Metal Performance Shaders) device.
- **CPU Fallback**: Graceful fallback when no hardware accelerator is present or when explicitly chosen.

---

## 2. Multi-Platform Bazel Architecture

### 2.1 Pip Dependency Resolution & Dual Pip Hubs
Due to conflicting binary wheels, differing index URLs, and mutually exclusive SHA256 checksums on Linux, `rules_python` manages two distinct pip hubs alongside macOS support:

- `requirements_lock_cuda.txt` (`linux_x86_64`): Pins Linux wheels for standard PyPI and NVIDIA CUDA (PyTorch 2.9.1 with CUDA 12.8 runtime).
- `requirements_lock_rocm.txt` (`linux_x86_64`): Pins Linux wheels for AMD ROCm 6.4 (PyTorch 2.9.1+rocm6.4 and pytorch-triton-rocm from `https://download.pytorch.org/whl/rocm6.4`).
- `requirements_lock_darwin.txt` (`osx_aarch64`, `osx_x86_64`): Pins macOS wheels with native Apple Silicon Metal/MPS support.

In `MODULE.bazel`:
```bzl
pip = use_extension("@rules_python//python/extensions:pip.bzl", "pip")

# Standard PyPI / CUDA Hub
pip.parse(
    hub_name = "pypi",
    python_version = PYTHON_VERSION,
    requirements_by_platform = {
        "//:requirements_lock_cuda.txt": "linux_x86_64",
        "//:requirements_lock_darwin.txt": "osx_aarch64,osx_x86_64",
    },
)
use_repo(pip, "pypi")

# AMD ROCm Hub
pip.parse(
    hub_name = "pypi_rocm",
    python_version = PYTHON_VERSION,
    requirements_by_platform = {
        "//:requirements_lock_rocm.txt": "linux_x86_64",
        "//:requirements_lock_darwin.txt": "osx_aarch64,osx_x86_64",
    },
    extra_pip_args = [
        "--extra-index-url=https://download.pytorch.org/whl/rocm6.4",
        "--find-links=https://download.pytorch.org/whl/rocm6.4",
    ],
)
use_repo(pip, "pypi_rocm")
```

In root `BUILD.bazel`, dynamic aliases select the appropriate hub based on `--//:gpu_backend`:
```bzl
alias(
    name = "torch",
    actual = select({
        ":is_rocm_backend": "@pypi_rocm//torch",
        "//conditions:default": "@pypi//torch",
    }),
    visibility = ["//visibility:public"],
)

alias(
    name = "torchvision",
    actual = select({
        ":is_rocm_backend": "@pypi_rocm//torchvision",
        "//conditions:default": "@pypi//torchvision",
    }),
    visibility = ["//visibility:public"],
)
```

All targets across the repository depend on `//:torch` and `//:torchvision`.

### 2.2 Environment Variables & Config Shortcuts
Configured in `.bazelrc`:
- **macOS (Apple Silicon)**:
  `build:macos --action_env=PYTORCH_ENABLE_MPS_FALLBACK=1`
  `test:macos --test_env=PYTORCH_ENABLE_MPS_FALLBACK=1`
- **AMD ROCm**:
  `build:rocm --action_env=HSA_OVERRIDE_GFX_VERSION=11.0.0`
  `build:rocm --//:gpu_backend=rocm`
  `test:rocm --test_env=HSA_OVERRIDE_GFX_VERSION=11.0.0`
  `test:rocm --//:gpu_backend=rocm`
- **NVIDIA CUDA**:
  `build:cuda --action_env=CUDA_VISIBLE_DEVICES=0`
  `build:cuda --//:gpu_backend=cuda`
  `test:cuda --test_env=CUDA_VISIBLE_DEVICES=0`
  `test:cuda --//:gpu_backend=cuda`

In build files (`jepa_rl_mario`, `alexnet`, `alexnet_dvgs`, `triton_from_onnx`), target `env` is configured via `select()`:
```bzl
env = select({
    "@platforms//os:osx": {
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    },
    "@platforms//os:linux": {
        "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
    },
    "//conditions:default": {},
})
```

Additionally, `yft_utils.device` proactively sets `HSA_OVERRIDE_GFX_VERSION="11.0.0"` before `import torch` on Linux systems to enable instant out-of-the-box hardware acceleration on RDNA 3 / 3.5 APUs (such as AMD Radeon 890M / gfx1150).

---

## 3. Unified Device Detection (`yft_utils.device`)

A pure functional module `yft_utils.device` exposes:
- `DeviceInfo`: Immutable dataclass containing `device`, `device_type`, `platform`, `is_nvidia`, `is_amd`, `is_apple`, and `device_name`.
- `resolve_device(cuda_available, is_hip, mps_available, preferred, cuda_device_name) -> DeviceInfo`: Pure resolver testable on data structures without hardware.
- `detect_device(preferred=None) -> DeviceInfo`: IO wrapper querying PyTorch runtime.
- `get_profiler_activities(device_type) -> list`: Safe profiler activity list avoiding CUDA crashes on macOS.

---

## 4. Hardware Discovery (`gpu_device_info`)

Composed of three modular readers implementing `GPUPlatformReader`:
- `gpu_device_info.nvidia`: Discovers NVIDIA GPUs via `nvidia-smi` CSV queries and extracts driver/CUDA versions.
- `gpu_device_info.amd`: Discovers AMD GPUs via Linux sysfs (`/sys/class/drm`, `/sys/devices/virtual/kfd`) and `rocm-smi`.
- `gpu_device_info.apple`: Discovers Apple Silicon GPUs and unified memory via `system_profiler SPDisplaysDataType -json`.

---

## 5. Script Adaptations

- **`jepa_rl_mario`**:
  - `train.py` accepts `--device {auto,cuda,mps,cpu}` and transfers networks (`q_network`, `target_network`) and observation/batch tensors to the detected accelerator.
  - `model.py` dynamically sizes linear layer inputs and fixes stride indexing.
- **`alexnet` & `alexnet_dvgs`**:
  - Integrated with `detect_device()` and `--device` argument.
- **`triton_from_onnx`**:
  - Automatically branches: executes Triton GPU kernels on NVIDIA CUDA and AMD ROCm, while running the PyTorch reference interpreter on macOS (MPS) and CPU without crashing on CUDA profiler activities.

