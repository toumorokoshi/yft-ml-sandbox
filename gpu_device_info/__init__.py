from gpu_device_info.amd import (
    detect_amd_gpu_io,
    get_amd_gpu_info_io,
)
from gpu_device_info.interface import (
    PLATFORM_AMD,
    PLATFORM_UNKNOWN,
    GPUInfo,
    GPUPlatformReader,
    SystemGPUInfo,
    read_system_gpus,
)

# Composition of available platform readers
_READERS = [
    GPUPlatformReader(
        platform_name=PLATFORM_AMD,
        detect_fn=detect_amd_gpu_io,
        read_fn=get_amd_gpu_info_io,
    )
]


def get_gpu_info() -> SystemGPUInfo:
    """Retrieves system GPU device information.

    Uses composition of platform readers to identify the GPU vendor and gather
    relevant specifications (such as name, memory usage, GFX version, and driver).
    """
    return read_system_gpus(_READERS)


__all__ = [
    "get_gpu_info",
    "GPUInfo",
    "SystemGPUInfo",
    "PLATFORM_AMD",
    "PLATFORM_UNKNOWN",
]
