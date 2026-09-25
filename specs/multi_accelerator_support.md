# Multi-Accelerator Support Specification (NVIDIA CUDA, AMD ROCm, Apple Metal)

## 1. Objective

Provide first-class, cross-platform execution of PyTorch models and Python scripts across:
- **NVIDIA GPUs**: CUDA-accelerated via `cuda` device.
- **AMD GPUs**: ROCm/HIP-accelerated via `cuda` device (HIP translation layer).
- **Apple Silicon (macOS)**: Metal-accelerated via `mps` (Metal Performance Shaders) device.
- **CPU Fallback**: Graceful fallback when no hardware accelerator is present or when explicitly chosen.

---

## 2. Multi-Platform Bazel Architecture

### 2.1 Pip Dependency Resolution
Due to conflicting binary wheels, differing index URLs, and mutually exclusive SHA256 checksums, `rules_python` uses platform-specific lockfiles:

- `requirements_lock.txt` (`linux_x86_64`): Pins Linux wheels including ROCm/CUDA support.
- `requirements_lock_darwin.txt` (`osx_aarch64`, `osx_x86_64`): Pins macOS wheels with native Metal/MPS support.

In `MODULE.bazel`:
```bzl
pip = use_extension("@rules_python//python/extensions:pip.bzl", "pip")
pip.parse(
    hub_name = "pypi",
    python_version = "3.12",
    requirements_by_platform = {
        "//:requirements_lock.txt": "linux_x86_64",
        "//:requirements_lock_darwin.txt": "osx_aarch64,osx_x86_64",
    },
)
use_repo(pip, "pypi")
```

### 2.2 Environment Variables & Config Shortcuts
Configured in `.bazelrc`:
- **macOS (Apple Silicon)**:
  `build:macos --action_env=PYTORCH_ENABLE_MPS_FALLBACK=1`
  `test:macos --test_env=PYTORCH_ENABLE_MPS_FALLBACK=1`
- **AMD ROCm**:
  `build:rocm --action_env=HSA_OVERRIDE_GFX_VERSION=11.0.0`
  `test:rocm --test_env=HSA_OVERRIDE_GFX_VERSION=11.0.0`
- **NVIDIA CUDA**:
  `build:cuda --action_env=CUDA_VISIBLE_DEVICES=0`
  `test:cuda --test_env=CUDA_VISIBLE_DEVICES=0`

In build files (`alexnet`, `alexnet_dvgs`, `triton_from_onnx`), target `env` is configured via `select()`:
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

