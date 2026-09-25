"""Apple Silicon / Metal GPU discovery for macOS."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any, Final, Optional

from gpu_device_info.interface import (
    NA_STRING,
    PLATFORM_APPLE,
    GPUInfo,
    SystemGPUInfo,
)

# Constants (Rule 5)
APPLE_SYSTEM_PROFILER_CMD: Final[list[str]] = ["system_profiler", "SPDisplaysDataType", "-json"]


# Helper Functions (Rule 6)
def parse_vram_bytes(vram_entry: Any) -> int:
    """Helper to convert VRAM representation from system_profiler into bytes."""
    if isinstance(vram_entry, int):
        return vram_entry
    if isinstance(vram_entry, str):
        v = vram_entry.strip().upper()
        # e.g., "16 GB", "36 GB", "128 MB"
        parts = v.split()
        if len(parts) >= 2:
            try:
                num = float(parts[0])
                unit = parts[1]
                if "GB" in unit:
                    return int(num * 1024 * 1024 * 1024)
                elif "MB" in unit:
                    return int(num * 1024 * 1024)
            except ValueError:
                pass
    return 0


def parse_system_profiler_json(data: dict[str, Any]) -> SystemGPUInfo:
    """Pure function: transforms system_profiler JSON dict into SystemGPUInfo (Rule 1, Rule 2)."""
    devices: list[GPUInfo] = []
    displays = data.get("SPDisplaysDataType", [])

    for entry in displays:
        name = entry.get("_name", "Apple Silicon GPU")
        vendor = entry.get("spdisplays_vendor", "Apple")
        vram_entry = entry.get("spdisplays_vram", entry.get("spdisplays_vram_shared", "0 MB"))
        vram_bytes = parse_vram_bytes(vram_entry)
        metal_support = entry.get("spdisplays_metal", "Metal Supported")

        dev = GPUInfo(
            name=name,
            vendor=vendor,
            device_id=str(entry.get("spdisplays_device-id", NA_STRING)),
            vram_total_bytes=vram_bytes,
            vram_used_bytes=0,
            driver_version=metal_support,
            driver_name="Metal",
            architecture_family="Apple Silicon",
        )
        devices.append(dev)

    return SystemGPUInfo(
        platform=PLATFORM_APPLE,
        devices=devices,
        rocm_or_cuda_version=None,
    )


# IO Wrappers (Rule 2)
def detect_apple() -> bool:
    """IO wrapper: checks if the operating system is macOS."""
    return sys.platform == "darwin"


def execute_system_profiler() -> dict[str, Any]:
    """IO wrapper: runs system_profiler command and returns parsed JSON dict."""
    res = subprocess.run(APPLE_SYSTEM_PROFILER_CMD, capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


def read_apple_gpus() -> SystemGPUInfo:
    """IO wrapper: reads Apple GPU info and parses using pure function."""
    try:
        data = execute_system_profiler()
        return parse_system_profiler_json(data)
    except Exception:
        return SystemGPUInfo(platform=PLATFORM_APPLE, devices=[])
