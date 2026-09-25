"""Unit tests for device detection and resolution."""

from __future__ import annotations

import unittest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

from yft_utils.device import (
    DEVICE_CUDA,
    DEVICE_MPS,
    DEVICE_CPU,
    PLATFORM_NVIDIA,
    PLATFORM_AMD,
    PLATFORM_APPLE,
    PLATFORM_CPU,
    resolve_device,
    get_profiler_activities,
    detect_device,
)


class TestDeviceResolution(unittest.TestCase):
    """Tests executing pure logic on data structures without requiring hardware (Rule 3)."""

    def test_resolve_nvidia(self) -> None:
        info = resolve_device(
            cuda_available=True,
            is_hip=False,
            mps_available=False,
            cuda_device_name="NVIDIA GeForce RTX 4090",
        )
        self.assertEqual(info.device_type, DEVICE_CUDA)
        self.assertEqual(info.platform, PLATFORM_NVIDIA)
        self.assertTrue(info.is_nvidia)
        self.assertFalse(info.is_amd)
        self.assertFalse(info.is_apple)
        self.assertEqual(info.device_name, "NVIDIA GeForce RTX 4090")

    def test_resolve_amd(self) -> None:
        info = resolve_device(
            cuda_available=True,
            is_hip=True,
            mps_available=False,
            cuda_device_name="AMD Radeon RX 7900 XTX",
        )
        self.assertEqual(info.device_type, DEVICE_CUDA)
        self.assertEqual(info.platform, PLATFORM_AMD)
        self.assertFalse(info.is_nvidia)
        self.assertTrue(info.is_amd)
        self.assertFalse(info.is_apple)
        self.assertEqual(info.device_name, "AMD Radeon RX 7900 XTX")

    def test_resolve_apple_silicon(self) -> None:
        info = resolve_device(
            cuda_available=False,
            is_hip=False,
            mps_available=True,
        )
        self.assertEqual(info.device_type, DEVICE_MPS)
        self.assertEqual(info.platform, PLATFORM_APPLE)
        self.assertFalse(info.is_nvidia)
        self.assertFalse(info.is_amd)
        self.assertTrue(info.is_apple)
        if HAS_TORCH:
            self.assertEqual(info.device.type, "mps")

    def test_resolve_cpu_fallback(self) -> None:
        info = resolve_device(
            cuda_available=False,
            is_hip=False,
            mps_available=False,
        )
        self.assertEqual(info.device_type, DEVICE_CPU)
        self.assertEqual(info.platform, PLATFORM_CPU)
        self.assertFalse(info.is_nvidia)
        self.assertFalse(info.is_amd)
        self.assertFalse(info.is_apple)
        if HAS_TORCH:
            self.assertEqual(info.device.type, "cpu")

    def test_explicit_preference_override(self) -> None:
        info = resolve_device(
            cuda_available=True,
            is_hip=False,
            mps_available=False,
            preferred="cpu",
        )
        self.assertEqual(info.device_type, DEVICE_CPU)
        if HAS_TORCH:
            self.assertEqual(info.device.type, "cpu")

    def test_profiler_activities(self) -> None:
        if not HAS_TORCH:
            self.skipTest("PyTorch not installed in this Python environment")
        cuda_activities = get_profiler_activities(DEVICE_CUDA)
        self.assertIn(torch.profiler.ProfilerActivity.CPU, cuda_activities)
        self.assertIn(torch.profiler.ProfilerActivity.CUDA, cuda_activities)

        mps_activities = get_profiler_activities(DEVICE_MPS)
        self.assertEqual(mps_activities, [torch.profiler.ProfilerActivity.CPU])

        cpu_activities = get_profiler_activities(DEVICE_CPU)
        self.assertEqual(cpu_activities, [torch.profiler.ProfilerActivity.CPU])

    def test_integration_detect_device(self) -> None:
        """Single integration test against system environment (Rule 3)."""
        info = detect_device()
        self.assertIn(info.device_type, [DEVICE_CUDA, DEVICE_MPS, DEVICE_CPU])


if __name__ == "__main__":
    unittest.main()
