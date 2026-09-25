"""Unit tests for Apple GPU parser."""

from __future__ import annotations

import unittest
from gpu_device_info.interface import PLATFORM_APPLE
from gpu_device_info.apple import parse_system_profiler_json, parse_vram_bytes

SAMPLE_APPLE_PROFILER_JSON = {
    "SPDisplaysDataType": [
        {
            "_name": "Apple M3 Max",
            "spdisplays_vendor": "Apple",
            "spdisplays_device-id": "0x5d00",
            "spdisplays_vram": "36 GB",
            "spdisplays_metal": "Metal 3",
        }
    ]
}


class TestAppleParser(unittest.TestCase):
    """Unit tests on data structures without requiring macOS host (Rule 3)."""

    def test_parse_system_profiler_json(self) -> None:
        info = parse_system_profiler_json(SAMPLE_APPLE_PROFILER_JSON)
        self.assertEqual(info.platform, PLATFORM_APPLE)
        self.assertEqual(len(info.devices), 1)

        dev0 = info.devices[0]
        self.assertEqual(dev0.name, "Apple M3 Max")
        self.assertEqual(dev0.vendor, "Apple")
        self.assertEqual(dev0.device_id, "0x5d00")
        self.assertEqual(dev0.vram_total_bytes, 36 * 1024 * 1024 * 1024)
        self.assertEqual(dev0.driver_version, "Metal 3")
        self.assertEqual(dev0.driver_name, "Metal")
        self.assertEqual(dev0.architecture_family, "Apple Silicon")

    def test_parse_vram_bytes(self) -> None:
        self.assertEqual(parse_vram_bytes("16 GB"), 16 * 1024 * 1024 * 1024)
        self.assertEqual(parse_vram_bytes("512 MB"), 512 * 1024 * 1024)
        self.assertEqual(parse_vram_bytes(1024), 1024)
        self.assertEqual(parse_vram_bytes("invalid"), 0)


if __name__ == "__main__":
    unittest.main()
