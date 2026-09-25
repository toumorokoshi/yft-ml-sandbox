"""Unit tests for NVIDIA GPU discovery parser."""

from __future__ import annotations

import unittest
from gpu_device_info.interface import PLATFORM_NVIDIA
from gpu_device_info.nvidia import (
    parse_cuda_version_from_text,
    parse_nvidia_smi_csv,
    safe_parse_float,
    safe_parse_int,
)

SAMPLE_NVIDIA_SMI_CSV = """
NVIDIA GeForce RTX 4090, 00000000:01:00.0, 24564, 1024, 550.54.14, 45, 12, 2520, 10501
NVIDIA GeForce RTX 3080, 00000000:02:00.0, 10240, 512, 550.54.14, 50, 20, 1710, 9500
"""

SAMPLE_NVIDIA_SMI_BANNER = """
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.14              Driver Version: 550.54.14      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
"""


class TestNvidiaParser(unittest.TestCase):
    """Unit tests on data structures without requiring physical NVIDIA hardware (Rule 3)."""

    def test_parse_nvidia_smi_csv(self) -> None:
        info = parse_nvidia_smi_csv(SAMPLE_NVIDIA_SMI_CSV, cuda_version="12.4")
        self.assertEqual(info.platform, PLATFORM_NVIDIA)
        self.assertEqual(info.rocm_or_cuda_version, "12.4")
        self.assertEqual(len(info.devices), 2)

        dev0 = info.devices[0]
        self.assertEqual(dev0.name, "NVIDIA GeForce RTX 4090")
        self.assertEqual(dev0.device_id, "00000000:01:00.0")
        self.assertEqual(dev0.vram_total_bytes, 24564 * 1024 * 1024)
        self.assertEqual(dev0.vram_used_bytes, 1024 * 1024 * 1024)
        self.assertEqual(dev0.driver_version, "550.54.14")
        self.assertEqual(dev0.temperature_c, 45.0)
        self.assertEqual(dev0.utilization_pct, 12.0)
        self.assertIsNotNone(dev0.clocks)
        self.assertEqual(dev0.clocks.sclk_current_mhz, 2520)
        self.assertEqual(dev0.clocks.mclk_current_mhz, 10501)

        dev1 = info.devices[1]
        self.assertEqual(dev1.name, "NVIDIA GeForce RTX 3080")
        self.assertEqual(dev1.temperature_c, 50.0)

    def test_parse_cuda_version(self) -> None:
        version = parse_cuda_version_from_text(SAMPLE_NVIDIA_SMI_BANNER)
        self.assertEqual(version, "12.4")

    def test_safe_parsers(self) -> None:
        self.assertEqual(safe_parse_float("45.5"), 45.5)
        self.assertIsNone(safe_parse_float("[Not Supported]"))
        self.assertIsNone(safe_parse_float("N/A"))
        self.assertEqual(safe_parse_int("1200"), 1200)
        self.assertIsNone(safe_parse_int("invalid"))


if __name__ == "__main__":
    unittest.main()
