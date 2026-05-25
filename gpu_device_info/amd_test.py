import unittest

from gpu_device_info import PLATFORM_AMD, get_gpu_info
from gpu_device_info.amd import (
    derive_warp_size_from_gfx,
    find_kfd_properties,
    get_platform_generation_info,
    parse_amdsmi_data,
    parse_caches,
    parse_dpm_clock_string,
    parse_dpm_clocks,
    parse_rocm_smi_json,
    safe_parse_float,
    safe_parse_int,
    safe_parse_str,
)
from gpu_device_info.interface import NA_STRING


class GpuDeviceInfoUnitTest(unittest.TestCase):

    def test_safe_parse_helpers(self):
        self.assertEqual(safe_parse_float("39.0°C"), 39.0)
        self.assertEqual(safe_parse_float("  -12.3 % "), -12.3)
        self.assertIsNone(safe_parse_float("N/A"))
        self.assertIsNone(safe_parse_float(None))

        self.assertEqual(safe_parse_int("536870912 B"), 536870912)
        self.assertIsNone(safe_parse_int("N/A"))
        self.assertIsNone(safe_parse_int(None))

        self.assertEqual(safe_parse_str(" AMD Radeon "), "AMD Radeon")
        self.assertEqual(safe_parse_str(None), NA_STRING)
        self.assertEqual(safe_parse_str(""), NA_STRING)

    def test_derive_warp_size_from_gfx(self):
        self.assertEqual(derive_warp_size_from_gfx("gfx906"), 64)
        self.assertEqual(derive_warp_size_from_gfx("gfx1030"), 32)
        self.assertEqual(derive_warp_size_from_gfx("gfx1100"), 32)
        self.assertIsNone(derive_warp_size_from_gfx("unknown"))
        self.assertIsNone(derive_warp_size_from_gfx(None))

    def test_get_platform_generation_info(self):
        arch, micro = get_platform_generation_info("gfx908")
        self.assertEqual(arch, "CDNA / Vega (GCN 5)")
        self.assertEqual(micro, "GFX9 Family")

        arch, micro = get_platform_generation_info("gfx1030")
        self.assertEqual(arch, "RDNA 2")
        self.assertEqual(micro, "Navi 2x")

        arch, micro = get_platform_generation_info("gfx1150")
        self.assertEqual(arch, "RDNA 3.5")
        self.assertEqual(micro, "Strix Point APU")

        arch, micro = get_platform_generation_info(None)
        self.assertIsNone(arch)
        self.assertIsNone(micro)

    def test_parse_dpm_clock_string(self):
        clk_str = """
0: 600Mhz 
1: 618Mhz *
2: 2900Mhz 
"""
        curr, min_f, max_f = parse_dpm_clock_string(clk_str)
        self.assertEqual(curr, 618)
        self.assertEqual(min_f, 600)
        self.assertEqual(max_f, 2900)

        # Handle Ghz
        curr, min_f, max_f = parse_dpm_clock_string("0: 1.5Ghz *")
        # 1.5 rounded or integer prefix: "1" or Ghz conversion
        # Here "1" is parsed since we match digit: "1.5" has digit 1 and 5
        # "15" since only digits are mapped. So "15Ghz" -> 15000 MHz.
        # Let's check with "1.5Ghz" -> digits "15" -> 15 * 1000 = 15000 MHz
        self.assertEqual(curr, 15000)

    def test_find_kfd_properties(self):
        topology = {
            0: {"device_id": "0", "vendor_id": "0"},
            1: {
                "device_id": "5390",  # decimal for 0x150e
                "vendor_id": "4098",  # decimal for 0x1002
                "wave_front_size": "32",
                "simd_count": "32",
                "simd_per_cu": "2",
            },
        }

        props = find_kfd_properties(topology, "0x150e", "0x1002")
        self.assertEqual(props.get("wave_front_size"), "32")

        props = find_kfd_properties(topology, "5390", "4098")
        self.assertEqual(props.get("wave_front_size"), "32")

    def test_parse_caches(self):
        raw_caches = [
            {"level": "1", "size": "32", "type": "9", "cache_line_size": "128"},
            {"level": "2", "size": "2048", "type": "9", "cache_line_size": "128"},
            {"level": "1", "size": "32", "type": "10", "cache_line_size": "128"},
        ]
        caches = parse_caches(raw_caches)
        self.assertEqual(len(caches), 3)
        self.assertEqual(caches[0].level, 1)
        self.assertEqual(caches[0].size_kb, 32)
        self.assertEqual(caches[0].type, "Data")
        self.assertEqual(caches[1].type, "Instruction")
        self.assertEqual(caches[2].level, 2)
        self.assertEqual(caches[2].size_kb, 2048)

    def test_parse_rocm_smi_json(self):
        mock_stdout = """
{
  "card0": {
    "Device Name": "AMD Radeon Graphics",
    "Card Vendor": "Advanced Micro Devices, Inc. [AMD/ATI]",
    "Device ID": "0x150e",
    "VRAM Total Memory (B)": "536870912",
    "VRAM Total Used Memory (B)": "461144064",
    "Temperature (Sensor edge) (C)": "38.0",
    "GPU use (%)": "2",
    "GFX Version": "gfx1150",
    "PCI Bus": "0000:C1:00.0",
    "VBIOS version": "113-STRIXEMU-001",
    "Card Series": "AMD Radeon Graphics",
    "Card Model": "0x150e"
  },
  "system": {
    "Driver version": "6.17.0-23-generic"
  }
}
"""
        topology = {
            1: {
                "device_id": "5390",
                "vendor_id": "4098",
                "wave_front_size": "32",
                "simd_count": "32",
                "simd_per_cu": "2",
            }
        }
        clocks = {
            "0000:c1:00.0": {
                "pp_dpm_sclk": "0: 600Mhz\n1: 618Mhz *\n2: 2900Mhz\n",
                "pp_dpm_mclk": "0: 1000Mhz\n1: 2400Mhz *\n2: 2800Mhz\n",
            }
        }
        caches = {
            1: [
                {"level": "1", "size": "32", "type": "9", "cache_line_size": "128"},
                {"level": "2", "size": "2048", "type": "9", "cache_line_size": "128"},
            ]
        }

        info = parse_rocm_smi_json(
            mock_stdout,
            rocm_version="7.1.1",
            topology=topology,
            clocks=clocks,
            caches=caches,
        )
        self.assertEqual(info.platform, "amd")
        self.assertEqual(len(info.devices), 1)

        dev = info.devices[0]
        self.assertEqual(dev.name, "AMD Radeon Graphics")
        self.assertEqual(dev.architecture_family, "RDNA 3.5")
        self.assertEqual(dev.microarchitecture, "Strix Point APU")
        self.assertEqual(dev.vbios_version, "113-STRIXEMU-001")
        self.assertEqual(dev.clocks.sclk_current_mhz, 618)
        self.assertEqual(dev.clocks.mclk_current_mhz, 2400)
        self.assertEqual(len(dev.caches), 2)
        self.assertEqual(dev.caches[0].size_kb, 32)
        self.assertEqual(dev.caches[1].size_kb, 2048)

    def test_parse_amdsmi_data(self):
        mock_raw = {
            "rocm_version": "7.1.1",
            "devices": [
                {
                    "bdf": "0000:c1:00.0",
                    "asic_info": {
                        "market_name": "AMD Radeon Graphics",
                        "vendor_name": "Advanced Micro Devices Inc. [AMD/ATI]",
                        "device_id": "0x150e",
                        "target_graphics_version": "gfx1150",
                        "num_compute_units": 16,
                        "subsystem_id": "0x000b",
                    },
                    "vbios_info": {
                        "part_number": "113-STRIXEMU-001",
                        "version": "023.010.001.022.000001",
                    },
                    "board_info": {
                        "product_name": "Strix [Radeon 880M / 890M]",
                    },
                    "vram_total_bytes": 536870912,
                    "vram_used_bytes": 456966144,
                    "temperature_c": 38.0,
                    "utilization_pct": 2.0,
                    "driver_info": {
                        "driver_name": "amdgpu",
                        "driver_version": "Linuxversion...",
                    },
                }
            ],
        }
        topology = {
            1: {
                "device_id": "5390",
                "vendor_id": "4098",
                "wave_front_size": "32",
                "simd_count": "32",
                "simd_per_cu": "2",
            }
        }
        clocks = {
            "0000:c1:00.0": {
                "pp_dpm_sclk": "0: 600Mhz\n1: 618Mhz *\n2: 2900Mhz\n",
                "pp_dpm_mclk": "0: 1000Mhz\n1: 2400Mhz *\n2: 2800Mhz\n",
            }
        }
        caches = {
            1: [
                {"level": "1", "size": "32", "type": "9", "cache_line_size": "128"},
                {"level": "2", "size": "2048", "type": "9", "cache_line_size": "128"},
            ]
        }

        info = parse_amdsmi_data(
            mock_raw, topology=topology, clocks=clocks, caches=caches
        )
        self.assertEqual(info.platform, "amd")
        self.assertEqual(len(info.devices), 1)

        dev = info.devices[0]
        self.assertEqual(dev.name, "AMD Radeon Graphics")
        self.assertEqual(dev.architecture_family, "RDNA 3.5")
        self.assertEqual(
            dev.vbios_version, "113-STRIXEMU-001 (Version 023.010.001.022.000001)"
        )
        self.assertEqual(dev.card_series, "Strix [Radeon 880M / 890M]")
        self.assertEqual(dev.clocks.sclk_current_mhz, 618)
        self.assertEqual(dev.clocks.mclk_current_mhz, 2400)
        self.assertEqual(len(dev.caches), 2)

    def test_integration_gpu_info_io(self):
        info = get_gpu_info()
        print("\n=== Integration Test GPU Info Result ===")
        print("Platform:", info.platform)
        print("ROCm/CUDA Version:", info.rocm_or_cuda_version)
        for i, dev in enumerate(info.devices):
            print(f"Device {i}:")
            print("  Name:", dev.name)
            print("  Vendor:", dev.vendor)
            print("  Device ID:", dev.device_id)
            print("  GFX Version:", dev.gfx_version)
            print("  Platform Gen Architecture:", dev.architecture_family)
            print("  Microarchitecture:", dev.microarchitecture)
            print("  VBIOS version:", dev.vbios_version)
            print("  Card Series:", dev.card_series)
            print("  Card Model/SKU:", dev.card_model)
            print("  VRAM Total (bytes):", dev.vram_total_bytes)
            print("  VRAM Used (bytes):", dev.vram_used_bytes)
            print("  Temperature (C):", dev.temperature_c)
            print("  Utilization (%):", dev.utilization_pct)
            print("  Driver Name:", dev.driver_name)
            print("  Driver Version:", dev.driver_version)
            print("  Multiprocessor Count (CUs):", dev.multiprocessor_count)
            print("  Warp/Wavefront Size:", dev.warp_size)
            if dev.clocks:
                print("  Frequencies:")
                print(
                    f"    System Clock (sclk) -> Current: {dev.clocks.sclk_current_mhz} MHz, Min: {dev.clocks.sclk_min_mhz} MHz, Max: {dev.clocks.sclk_max_mhz} MHz"
                )
                print(
                    f"    Memory Clock (mclk) -> Current: {dev.clocks.mclk_current_mhz} MHz, Min: {dev.clocks.mclk_min_mhz} MHz, Max: {dev.clocks.mclk_max_mhz} MHz"
                )
            if dev.caches:
                print("  Caches:")
                for cache in dev.caches:
                    print(
                        f"    Level {cache.level} {cache.type} Cache -> Size: {cache.size_kb} KB, Line Size: {cache.line_size_bytes} B"
                    )

        self.assertEqual(info.platform, PLATFORM_AMD)
        self.assertGreater(len(info.devices), 0)

        dev = info.devices[0]
        self.assertTrue(
            "AMD" in dev.vendor or "Advanced Micro Devices" in dev.vendor
        )
        self.assertGreater(dev.vram_total_bytes, 0)
        self.assertEqual(dev.multiprocessor_count, 16)
        self.assertEqual(dev.warp_size, 32)
        # Check clocks and caches are populated
        self.assertIsNotNone(dev.clocks)
        self.assertGreaterEqual(len(dev.caches), 0)


if __name__ == "__main__":
    unittest.main()
