"""Device resolution and accelerator utilities for NVIDIA, AMD ROCm, and Apple Metal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

# Constants (Rule 5)
DEVICE_CUDA: Final[str] = "cuda"
DEVICE_MPS: Final[str] = "mps"
DEVICE_CPU: Final[str] = "cpu"

PLATFORM_NVIDIA: Final[str] = "nvidia"
PLATFORM_AMD: Final[str] = "amd"
PLATFORM_APPLE: Final[str] = "apple"
PLATFORM_CPU: Final[str] = "cpu"


@dataclass(frozen=True)
class DeviceInfo:
    """Immutable representation of selected device and accelerator platform."""

    device: Any  # torch.device when torch is available, else str
    device_type: str
    platform: str
    is_nvidia: bool
    is_amd: bool
    is_apple: bool
    device_name: str


def _make_torch_device(name: str) -> Any:
    if HAS_TORCH and torch is not None:
        return torch.device(name)
    return name


def resolve_device(
    cuda_available: bool,
    is_hip: bool,
    mps_available: bool,
    preferred: Optional[str] = None,
    cuda_device_name: Optional[str] = None,
) -> DeviceInfo:
    """Pure function: resolves target device given hardware flags (Rule 1, Rule 6)."""
    clean_preferred = preferred.strip().lower() if preferred else None

    # Explicit user preference override
    if clean_preferred and clean_preferred != "auto":
        dev = _make_torch_device(clean_preferred)
        dev_type = getattr(dev, "type", clean_preferred)
        if dev_type == DEVICE_CUDA:
            plat = PLATFORM_AMD if is_hip else PLATFORM_NVIDIA
            return DeviceInfo(
                device=dev,
                device_type=DEVICE_CUDA,
                platform=plat,
                is_nvidia=(plat == PLATFORM_NVIDIA),
                is_amd=(plat == PLATFORM_AMD),
                is_apple=False,
                device_name=cuda_device_name or "CUDA Device",
            )
        elif dev_type == DEVICE_MPS:
            return DeviceInfo(
                device=dev,
                device_type=DEVICE_MPS,
                platform=PLATFORM_APPLE,
                is_nvidia=False,
                is_amd=False,
                is_apple=True,
                device_name="Apple Silicon Metal Performance Shaders",
            )
        else:
            return DeviceInfo(
                device=dev,
                device_type=DEVICE_CPU,
                platform=PLATFORM_CPU,
                is_nvidia=False,
                is_amd=False,
                is_apple=False,
                device_name="CPU",
            )

    # Automatic detection: 1. CUDA/ROCm, 2. Apple MPS, 3. CPU
    if cuda_available:
        plat = PLATFORM_AMD if is_hip else PLATFORM_NVIDIA
        return DeviceInfo(
            device=_make_torch_device(DEVICE_CUDA),
            device_type=DEVICE_CUDA,
            platform=plat,
            is_nvidia=(plat == PLATFORM_NVIDIA),
            is_amd=(plat == PLATFORM_AMD),
            is_apple=False,
            device_name=cuda_device_name or ("AMD ROCm GPU" if is_hip else "NVIDIA GPU"),
        )
    elif mps_available:
        return DeviceInfo(
            device=_make_torch_device(DEVICE_MPS),
            device_type=DEVICE_MPS,
            platform=PLATFORM_APPLE,
            is_nvidia=False,
            is_amd=False,
            is_apple=True,
            device_name="Apple Silicon Metal Performance Shaders",
        )
    else:
        return DeviceInfo(
            device=_make_torch_device(DEVICE_CPU),
            device_type=DEVICE_CPU,
            platform=PLATFORM_CPU,
            is_nvidia=False,
            is_amd=False,
            is_apple=False,
            device_name="CPU",
        )


def get_profiler_activities(device_type: str) -> list[Any]:
    """Pure function: returns safe profiler activities based on device type."""
    if not HAS_TORCH or torch is None:
        return []
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device_type == DEVICE_CUDA:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def detect_device(preferred: Optional[str] = None) -> DeviceInfo:
    """IO wrapper: queries PyTorch runtime and delegates to pure resolution (Rule 2)."""
    if not HAS_TORCH or torch is None:
        return resolve_device(
            cuda_available=False,
            is_hip=False,
            mps_available=False,
            preferred=preferred,
            cuda_device_name="CPU (PyTorch not installed)",
        )

    cuda_avail = torch.cuda.is_available()
    is_hip = bool(getattr(torch.version, "hip", None) is not None)
    mps_avail = bool(
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )
    cuda_name = None
    if cuda_avail:
        try:
            cuda_name = torch.cuda.get_device_name(0)
        except Exception:
            cuda_name = None

    return resolve_device(
        cuda_available=cuda_avail,
        is_hip=is_hip,
        mps_available=mps_avail,
        preferred=preferred,
        cuda_device_name=cuda_name,
    )
