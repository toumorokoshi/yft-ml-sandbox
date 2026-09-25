"""NVIDIA GPU hardware discovery via nvidia-smi."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Final, Optional

from gpu_device_info.interface import (
    NA_STRING,
    PLATFORM_NVIDIA,
    GPUClockInfo,
    GPUInfo,
    SystemGPUInfo,
)

# Constants (Rule 5)
NVIDIA_SMI_QUERY_FIELDS: Final[str] = (
    "name,pci.bus_id,memory.total,memory.used,driver_version,"
    "temperature.gpu,utilization.gpu,clocks.current.graphics,clocks.current.memory"
)
NVIDIA_SMI_QUERY_CMD: Final[list[str]] = [
    "nvidia-smi",
    f"--query-gpu={NVIDIA_SMI_QUERY_FIELDS}",
    "--format=csv,noheader,nounits",
]
NVIDIA_SMI_CUDA_CMD: Final[list[str]] = ["nvidia-smi"]
DEV_NVIDIA_PATH: Final[str] = "/dev/nvidia0"
PROC_NVIDIA_PATH: Final[str] = "/proc/driver/nvidia"


# Helper Functions (minimal logic, Rule 6)
def safe_parse_float(val: str) -> Optional[float]:
    """Helper to safely parse float string."""
    try:
        s = val.strip()
        return float(s) if s and s != NA_STRING and "[not supported]" not in s.lower() else None
    except ValueError:
        return None


def safe_parse_int(val: str) -> Optional[int]:
    """Helper to safely parse int string."""
    try:
        s = val.strip()
        return int(s) if s and s != NA_STRING and "[not supported]" not in s.lower() else None
    except ValueError:
        return None


def parse_cuda_version_from_text(smi_output: str) -> Optional[str]:
    """Pure function: extracts CUDA Version from standard nvidia-smi banner text."""
    for line in smi_output.splitlines():
        if "CUDA Version:" in line:
            parts = line.split("CUDA Version:")
            if len(parts) > 1:
                ver = parts[1].strip().split()[0].replace("|", "").strip()
                return ver if ver else None
    return None


def parse_nvidia_smi_csv(csv_text: str, cuda_version: Optional[str] = None) -> SystemGPUInfo:
    """Pure function: transforms nvidia-smi CSV string into SystemGPUInfo data structures (Rule 1, Rule 2)."""
    devices: list[GPUInfo] = []
    lines = [line.strip() for line in csv_text.strip().splitlines() if line.strip()]

    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue

        name = parts[0]
        bus_id = parts[1]
        mem_total_mib = safe_parse_float(parts[2]) or 0.0
        mem_used_mib = safe_parse_float(parts[3]) or 0.0
        driver_ver = parts[4]

        temp_c = safe_parse_float(parts[5]) if len(parts) > 5 else None
        util_pct = safe_parse_float(parts[6]) if len(parts) > 6 else None
        sclk = safe_parse_int(parts[7]) if len(parts) > 7 else None
        mclk = safe_parse_int(parts[8]) if len(parts) > 8 else None

        clocks = GPUClockInfo(
            sclk_current_mhz=sclk,
            mclk_current_mhz=mclk,
        ) if (sclk is not None or mclk is not None) else None

        device_info = GPUInfo(
            name=name,
            vendor="NVIDIA",
            device_id=bus_id,
            vram_total_bytes=int(mem_total_mib * 1024 * 1024),
            vram_used_bytes=int(mem_used_mib * 1024 * 1024),
            driver_version=driver_ver,
            driver_name="nvidia",
            temperature_c=temp_c,
            utilization_pct=util_pct,
            clocks=clocks,
        )
        devices.append(device_info)

    return SystemGPUInfo(
        platform=PLATFORM_NVIDIA,
        devices=devices,
        rocm_or_cuda_version=cuda_version,
    )


# IO Wrappers (Rule 2)
def detect_nvidia() -> bool:
    """IO wrapper: checks if NVIDIA GPU driver or CLI is present."""
    if os.path.exists(DEV_NVIDIA_PATH) or os.path.exists(PROC_NVIDIA_PATH):
        return True
    return shutil.which("nvidia-smi") is not None


def execute_command(cmd: list[str]) -> str:
    """IO wrapper: executes external process and captures stdout."""
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return res.stdout


def read_nvidia_gpus() -> SystemGPUInfo:
    """IO wrapper: reads NVIDIA GPU information using nvidia-smi and parses with pure functions."""
    try:
        csv_output = execute_command(NVIDIA_SMI_QUERY_CMD)
    except Exception:
        return SystemGPUInfo(platform=PLATFORM_NVIDIA, devices=[])

    cuda_ver = None
    try:
        banner_output = execute_command(NVIDIA_SMI_CUDA_CMD)
        cuda_ver = parse_cuda_version_from_text(banner_output)
    except Exception:
        cuda_ver = None

    return parse_nvidia_smi_csv(csv_output, cuda_ver)
