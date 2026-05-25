from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

# Constants
PLATFORM_AMD = "amd"
PLATFORM_UNKNOWN = "unknown"
NA_STRING = "N/A"


@dataclass(frozen=True)
class GPUClockInfo:
    sclk_current_mhz: Optional[int] = None
    sclk_min_mhz: Optional[int] = None
    sclk_max_mhz: Optional[int] = None
    mclk_current_mhz: Optional[int] = None
    mclk_min_mhz: Optional[int] = None
    mclk_max_mhz: Optional[int] = None


@dataclass(frozen=True)
class GPUCacheInfo:
    level: int
    size_kb: int
    type: str
    line_size_bytes: Optional[int] = None


@dataclass(frozen=True)
class GPUInfo:
    name: str
    vendor: str
    device_id: str
    vram_total_bytes: int
    vram_used_bytes: int
    driver_version: str
    driver_name: str
    temperature_c: Optional[float] = None
    utilization_pct: Optional[float] = None
    gfx_version: Optional[str] = None
    multiprocessor_count: Optional[int] = None
    warp_size: Optional[int] = None
    architecture_family: Optional[str] = None
    microarchitecture: Optional[str] = None
    vbios_version: Optional[str] = None
    card_series: Optional[str] = None
    card_model: Optional[str] = None
    clocks: Optional[GPUClockInfo] = None
    caches: list[GPUCacheInfo] = field(default_factory=list)


@dataclass(frozen=True)
class SystemGPUInfo:
    platform: str
    devices: list[GPUInfo]
    rocm_or_cuda_version: Optional[str] = None


@dataclass(frozen=True)
class GPUPlatformReader:
    platform_name: str
    detect_fn: Callable[[], bool]
    read_fn: Callable[[], SystemGPUInfo]


def read_system_gpus(readers: Sequence[GPUPlatformReader]) -> SystemGPUInfo:
    """Gets GPU info by iterating through detectors and running the first matching reader."""
    for reader in readers:
        if reader.detect_fn():
            try:
                return reader.read_fn()
            except Exception:
                continue
    return SystemGPUInfo(platform=PLATFORM_UNKNOWN, devices=[])
