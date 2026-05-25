import json
import os
import shutil
import subprocess
from typing import Any, Optional

from gpu_device_info.interface import (
    NA_STRING,
    PLATFORM_AMD,
    GPUCacheInfo,
    GPUClockInfo,
    GPUInfo,
    SystemGPUInfo,
)

# Constants
ROCM_VERSION_FILE = "/opt/rocm/.info/version"
ROCM_SMI_COMMAND = ["rocm-smi", "-a", "--showmeminfo", "vram", "--json"]
KFD_TOPOLOGY_PATH = "/sys/devices/virtual/kfd/kfd/topology/nodes"
DRM_CARDS_PATH = "/sys/class/drm"


# Helper Functions (minimal logic)
def safe_parse_float(val: Any) -> Optional[float]:
    if val is None or str(val).strip() == NA_STRING:
        return None
    try:
        cleaned = "".join(c for c in str(val) if c.isdigit() or c in (".", "-"))
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def safe_parse_int(val: Any) -> Optional[int]:
    if val is None or str(val).strip() == NA_STRING:
        return None
    try:
        cleaned = "".join(c for c in str(val) if c.isdigit() or c == "-")
        return int(cleaned) if cleaned else None
    except ValueError:
        return None


def safe_parse_str(val: Any) -> str:
    if val is None:
        return NA_STRING
    s = str(val).strip()
    return NA_STRING if s == "" else s


def derive_warp_size_from_gfx(gfx_version: Optional[str]) -> Optional[int]:
    if not gfx_version:
        return None
    v = gfx_version.strip().lower()
    if "gfx9" in v:
        return 64
    if "gfx10" in v or "gfx11" in v or "gfx12" in v:
        return 32
    return None


def get_platform_generation_info(
    gfx_version: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    if not gfx_version:
        return None, None

    v = gfx_version.strip().lower()
    if "gfx9" in v:
        return "CDNA / Vega (GCN 5)", "GFX9 Family"
    if "gfx101" in v:
        return "RDNA 1", "Navi 1x"
    if "gfx103" in v:
        return "RDNA 2", "Navi 2x"
    if "gfx110" in v or "gfx115" in v:
        if "gfx115" in v:
            return "RDNA 3.5", "Strix Point APU"
        return "RDNA 3", "Navi 3x"
    if "gfx12" in v:
        return "RDNA 4", "Navi 4x"
    return "Unknown AMD Architecture", None


def find_kfd_properties(
    topology: dict[int, dict[str, str]],
    device_id_str: Optional[str],
    vendor_id_str: Optional[str] = None,
) -> dict[str, str]:
    node_id = find_kfd_node_id(topology, device_id_str, vendor_id_str)
    return topology.get(node_id, {}) if node_id is not None else {}


def find_kfd_node_id(
    topology: dict[int, dict[str, str]],
    device_id_str: Optional[str],
    vendor_id_str: Optional[str] = None,
) -> Optional[int]:
    if not device_id_str:
        return None

    try:
        target_device_id = (
            int(device_id_str, 16)
            if device_id_str.startswith("0x")
            else int(device_id_str)
        )
    except ValueError:
        return None

    target_vendor_id = None
    if vendor_id_str:
        try:
            target_vendor_id = (
                int(vendor_id_str, 16)
                if vendor_id_str.startswith("0x")
                else int(vendor_id_str)
            )
        except ValueError:
            pass

    for node_id, props in topology.items():
        try:
            dev_id = int(props.get("device_id", ""))
            if dev_id != target_device_id:
                continue
            if target_vendor_id is not None:
                v_id = int(props.get("vendor_id", ""))
                if v_id != target_vendor_id:
                    continue
            return node_id
        except ValueError:
            continue
    return None


def parse_dpm_clock_string(
    clk_str: str,
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    if not clk_str or clk_str.strip() == NA_STRING:
        return None, None, None

    current_freq = None
    frequencies = []

    for line in clk_str.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(":", 1)
        val_part = parts[1].strip() if len(parts) > 1 else parts[0].strip()

        is_active = "*" in val_part
        cleaned_val = val_part.replace("*", "").strip()
        num_str = "".join(c for c in cleaned_val if c.isdigit())
        if num_str:
            try:
                freq = int(num_str)
                if "ghz" in cleaned_val.lower():
                    freq = freq * 1000
                frequencies.append(freq)
                if is_active:
                    current_freq = freq
            except ValueError:
                pass

    min_freq = min(frequencies) if frequencies else None
    max_freq = max(frequencies) if frequencies else None

    return current_freq, min_freq, max_freq


def parse_dpm_clocks(card_clocks: dict[str, str]) -> GPUClockInfo:
    sclk_str = card_clocks.get("pp_dpm_sclk", "")
    mclk_str = card_clocks.get("pp_dpm_mclk", "")

    s_current, s_min, s_max = parse_dpm_clock_string(sclk_str)
    m_current, m_min, m_max = parse_dpm_clock_string(mclk_str)

    return GPUClockInfo(
        sclk_current_mhz=s_current,
        sclk_min_mhz=s_min,
        sclk_max_mhz=s_max,
        mclk_current_mhz=m_current,
        mclk_min_mhz=m_min,
        mclk_max_mhz=m_max,
    )


def map_cache_type_str(type_val: Optional[int]) -> str:
    if type_val is None:
        return "Unknown"
    if type_val == 9:
        return "Data"
    if type_val == 10:
        return "Instruction"
    if type_val == 12 or type_val == 4:
        return "Unified"
    return f"Type {type_val}"


def parse_caches(raw_caches: list[dict[str, str]]) -> list[GPUCacheInfo]:
    caches = []
    for raw in raw_caches:
        level = safe_parse_int(raw.get("level"))
        size = safe_parse_int(raw.get("size"))
        type_val = safe_parse_int(raw.get("type"))
        line_size = safe_parse_int(raw.get("cache_line_size"))

        if level is not None and size is not None:
            caches.append(
                GPUCacheInfo(
                    level=level,
                    size_kb=size,
                    type=map_cache_type_str(type_val),
                    line_size_bytes=line_size,
                )
            )
    caches.sort(key=lambda c: (c.level, c.type, c.size_kb))
    return caches


# Pure Inner Parsing Functions (work directly on data structures, no IO)
def parse_rocm_smi_json(
    stdout_str: str,
    rocm_version: Optional[str] = None,
    topology: Optional[dict[int, dict[str, str]]] = None,
    clocks: Optional[dict[str, dict[str, str]]] = None,
    caches: Optional[dict[int, list[dict[str, str]]]] = None,
) -> SystemGPUInfo:
    """Parses JSON output from rocm-smi command line tool."""
    start_idx = stdout_str.find("{")
    if start_idx == -1:
        return SystemGPUInfo(
            platform=PLATFORM_AMD, devices=[], rocm_or_cuda_version=rocm_version
        )

    json_str = stdout_str[start_idx:]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return SystemGPUInfo(
            platform=PLATFORM_AMD, devices=[], rocm_or_cuda_version=rocm_version
        )

    devices = []
    system_info = data.get("system", {})
    driver_version = safe_parse_str(system_info.get("Driver version"))

    for key, card in data.items():
        if not key.startswith("card"):
            continue

        name = safe_parse_str(card.get("Device Name"))
        vendor = safe_parse_str(card.get("Card Vendor"))
        device_id = safe_parse_str(card.get("Device ID"))
        vram_total = safe_parse_int(card.get("VRAM Total Memory (B)")) or 0
        vram_used = safe_parse_int(card.get("VRAM Total Used Memory (B)")) or 0
        temp = safe_parse_float(card.get("Temperature (Sensor edge) (C)"))
        util = safe_parse_float(card.get("GPU use (%)"))
        gfx = safe_parse_str(card.get("GFX Version"))

        # Query KFD properties for multiprocessor and warp sizes
        props = find_kfd_properties(topology or {}, device_id, vendor)
        warp_size = safe_parse_int(props.get("wave_front_size"))
        if warp_size is None:
            warp_size = derive_warp_size_from_gfx(gfx)

        simd_count = safe_parse_int(props.get("simd_count"))
        simd_per_cu = safe_parse_int(props.get("simd_per_cu"))
        if (
            simd_count is not None
            and simd_per_cu is not None
            and simd_per_cu > 0
        ):
            multiprocessor_count = simd_count // simd_per_cu
        else:
            multiprocessor_count = None

        # Platform Generation mapping
        arch_fam, micro_arch = get_platform_generation_info(gfx)
        vbios = safe_parse_str(card.get("VBIOS version"))
        series = safe_parse_str(card.get("Card Series"))
        model = safe_parse_str(card.get("Card Model"))

        # Clock retrieval
        pci_bus = safe_parse_str(card.get("PCI Bus")).lower()
        card_clocks = (clocks or {}).get(pci_bus, {})
        clock_info = parse_dpm_clocks(card_clocks)

        # Cache retrieval
        node_id = find_kfd_node_id(topology or {}, device_id, vendor)
        card_caches = (caches or {}).get(node_id, []) if node_id is not None else []
        parsed_caches = parse_caches(card_caches)

        gpu_info = GPUInfo(
            name=name,
            vendor=vendor,
            device_id=device_id,
            vram_total_bytes=vram_total,
            vram_used_bytes=vram_used,
            driver_version=driver_version,
            driver_name="amdgpu",
            temperature_c=temp,
            utilization_pct=util,
            gfx_version=gfx,
            multiprocessor_count=multiprocessor_count,
            warp_size=warp_size,
            architecture_family=arch_fam,
            microarchitecture=micro_arch,
            vbios_version=vbios,
            card_series=series,
            card_model=model,
            clocks=clock_info,
            caches=parsed_caches,
        )
        devices.append(gpu_info)

    return SystemGPUInfo(
        platform=PLATFORM_AMD, devices=devices, rocm_or_cuda_version=rocm_version
    )


def parse_amdsmi_data(
    raw_data: dict[str, Any],
    rocm_version_fallback: Optional[str] = None,
    topology: Optional[dict[int, dict[str, str]]] = None,
    clocks: Optional[dict[str, dict[str, str]]] = None,
    caches: Optional[dict[int, list[dict[str, str]]]] = None,
) -> SystemGPUInfo:
    """Parses raw data collected from amdsmi python library."""
    rocm_version = raw_data.get("rocm_version") or rocm_version_fallback
    devices = []

    for device_data in raw_data.get("devices", []):
        asic_info = device_data.get("asic_info", {})
        driver_info = device_data.get("driver_info", {})
        vbios_info = device_data.get("vbios_info", {})
        board_info = device_data.get("board_info", {})

        name = safe_parse_str(asic_info.get("market_name"))
        vendor = safe_parse_str(asic_info.get("vendor_name"))
        device_id = safe_parse_str(asic_info.get("device_id"))
        vram_total = device_data.get("vram_total_bytes", 0)
        vram_used = device_data.get("vram_used_bytes", 0)
        temp = safe_parse_float(device_data.get("temperature_c"))
        util = safe_parse_float(device_data.get("utilization_pct"))
        gfx = safe_parse_str(asic_info.get("target_graphics_version"))
        driver_version = safe_parse_str(driver_info.get("driver_version"))
        driver_name = safe_parse_str(driver_info.get("driver_name", "amdgpu"))

        multiprocessor_count = safe_parse_int(
            asic_info.get("num_compute_units")
        )

        props = find_kfd_properties(topology or {}, device_id, vendor)
        warp_size = safe_parse_int(props.get("wave_front_size"))
        if warp_size is None:
            warp_size = derive_warp_size_from_gfx(gfx)

        if multiprocessor_count is None:
            simd_count = safe_parse_int(props.get("simd_count"))
            simd_per_cu = safe_parse_int(props.get("simd_per_cu"))
            if (
                simd_count is not None
                and simd_per_cu is not None
                and simd_per_cu > 0
            ):
                multiprocessor_count = simd_count // simd_per_cu

        # Platform Generation mapping
        arch_fam, micro_arch = get_platform_generation_info(gfx)
        vbios_part = safe_parse_str(vbios_info.get("part_number"))
        vbios_version_val = safe_parse_str(vbios_info.get("version"))
        vbios = (
            f"{vbios_part} (Version {vbios_version_val})"
            if vbios_part != NA_STRING and vbios_version_val != NA_STRING
            else (vbios_part if vbios_part != NA_STRING else vbios_version_val)
        )
        series = safe_parse_str(board_info.get("product_name"))
        model = safe_parse_str(asic_info.get("subsystem_id"))

        # Clock retrieval
        pci_bus = safe_parse_str(device_data.get("bdf")).lower()
        card_clocks = (clocks or {}).get(pci_bus, {})
        clock_info = parse_dpm_clocks(card_clocks)

        # Cache retrieval
        node_id = find_kfd_node_id(topology or {}, device_id, vendor)
        card_caches = (caches or {}).get(node_id, []) if node_id is not None else []
        parsed_caches = parse_caches(card_caches)

        gpu_info = GPUInfo(
            name=name,
            vendor=vendor,
            device_id=device_id,
            vram_total_bytes=vram_total,
            vram_used_bytes=vram_used,
            driver_version=driver_version,
            driver_name=driver_name,
            temperature_c=temp,
            utilization_pct=util,
            gfx_version=gfx,
            multiprocessor_count=multiprocessor_count,
            warp_size=warp_size,
            architecture_family=arch_fam,
            microarchitecture=micro_arch,
            vbios_version=vbios,
            card_series=series,
            card_model=model,
            clocks=clock_info,
            caches=parsed_caches,
        )
        devices.append(gpu_info)

    return SystemGPUInfo(
        platform=PLATFORM_AMD, devices=devices, rocm_or_cuda_version=rocm_version
    )


# IO Wrapper Functions
def detect_amd_gpu_io() -> bool:
    """Detects if an AMD GPU is present in the system (contains IO)."""
    if os.path.exists("/dev/kfd"):
        return True
    if shutil.which("rocm-smi") is not None:
        return True
    return False


def read_rocm_version_file_io() -> Optional[str]:
    """Reads ROCm version file from default location (contains IO)."""
    if os.path.exists(ROCM_VERSION_FILE):
        try:
            with open(ROCM_VERSION_FILE, "r") as f:
                return f.read().strip()
        except Exception:
            return None
    return None


def read_kfd_topology_io() -> dict[int, dict[str, str]]:
    """Reads the properties for all nodes in KFD topology (contains IO)."""
    if not os.path.isdir(KFD_TOPOLOGY_PATH):
        return {}

    topology = {}
    try:
        for node_dir in os.listdir(KFD_TOPOLOGY_PATH):
            if not node_dir.isdigit():
                continue
            node_id = int(node_dir)
            prop_path = os.path.join(KFD_TOPOLOGY_PATH, node_dir, "properties")
            if os.path.isfile(prop_path):
                with open(prop_path, "r") as f:
                    props = {}
                    for line in f:
                        line = line.strip()
                        if line and " " in line:
                            k, v = line.split(" ", 1)
                            props[k.strip()] = v.strip()
                    topology[node_id] = props
    except Exception:
        pass
    return topology


def read_kfd_caches_io() -> dict[int, list[dict[str, str]]]:
    """Reads all cache properties in the KFD topology (contains IO)."""
    if not os.path.isdir(KFD_TOPOLOGY_PATH):
        return {}

    node_caches = {}
    try:
        for node_dir in os.listdir(KFD_TOPOLOGY_PATH):
            if not node_dir.isdigit():
                continue
            node_id = int(node_dir)
            caches_dir = os.path.join(KFD_TOPOLOGY_PATH, node_dir, "caches")
            if not os.path.isdir(caches_dir):
                continue

            caches_list = []
            for cache_dir in os.listdir(caches_dir):
                if not cache_dir.isdigit():
                    continue
                prop_path = os.path.join(caches_dir, cache_dir, "properties")
                if os.path.isfile(prop_path):
                    try:
                        with open(prop_path, "r") as f:
                            props = {}
                            for line in f:
                                line = line.strip()
                                if line and " " in line:
                                    k, v = line.split(" ", 1)
                                    props[k.strip()] = v.strip()
                            caches_list.append(props)
                    except Exception:
                        pass
            if caches_list:
                node_caches[node_id] = caches_list
    except Exception:
        pass
    return node_caches


def read_dpm_clocks_io() -> dict[str, dict[str, str]]:
    """Reads DPM clock files from DRM card devices (contains IO)."""
    if not os.path.isdir(DRM_CARDS_PATH):
        return {}

    clocks = {}
    try:
        for card_dir in os.listdir(DRM_CARDS_PATH):
            if not card_dir.startswith("card"):
                continue
            device_path = os.path.join(DRM_CARDS_PATH, card_dir, "device")
            if not os.path.isdir(device_path):
                continue

            bdf = os.path.basename(os.path.realpath(device_path)).lower()

            card_clocks = {}
            for clk_name in ["pp_dpm_sclk", "pp_dpm_mclk"]:
                clk_file = os.path.join(device_path, clk_name)
                if os.path.isfile(clk_file):
                    try:
                        with open(clk_file, "r") as f:
                            card_clocks[clk_name] = f.read()
                    except Exception:
                        pass

            if card_clocks:
                clocks[bdf] = card_clocks
    except Exception:
        pass
    return clocks


def read_rocm_smi_cli_io() -> str:
    """Runs rocm-smi command-line tool to obtain GPU details (contains IO)."""
    clean_env = dict(os.environ)
    for k in list(clean_env.keys()):
        if k.startswith("PYTHON"):
            del clean_env[k]

    result = subprocess.run(
        ROCM_SMI_COMMAND,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=clean_env,
    )
    return result.stdout


def read_amdsmi_io() -> dict[str, Any]:
    """Retrieves raw GPU metric information using amdsmi python library (contains IO)."""
    import amdsmi

    amdsmi.amdsmi_init()
    try:
        try:
            rocm_ok, rocm_ver = amdsmi.amdsmi_get_rocm_version()
            rocm_version = rocm_ver if rocm_ok else None
        except Exception:
            rocm_version = None

        raw_devices = []
        handles = amdsmi.amdsmi_get_processor_handles()
        for handle in handles:
            proc_type = amdsmi.amdsmi_get_processor_type(handle)
            if (
                proc_type.get("processor_type")
                != "AMDSMI_PROCESSOR_TYPE_AMD_GPU"
            ):
                continue

            try:
                bdf = amdsmi.amdsmi_get_gpu_device_bdf(handle)
            except Exception:
                bdf = NA_STRING

            try:
                asic_info = amdsmi.amdsmi_get_gpu_asic_info(handle)
            except Exception:
                asic_info = {}

            try:
                vbios_info = amdsmi.amdsmi_get_gpu_vbios_info(handle)
            except Exception:
                vbios_info = {}

            try:
                board_info = amdsmi.amdsmi_get_gpu_board_info(handle)
            except Exception:
                board_info = {}

            try:
                vram_total = amdsmi.amdsmi_get_gpu_memory_total(
                    handle, amdsmi.AmdSmiMemoryType.VRAM
                )
                vram_used = amdsmi.amdsmi_get_gpu_memory_usage(
                    handle, amdsmi.AmdSmiMemoryType.VRAM
                )
            except Exception:
                vram_total = 0
                vram_used = 0

            try:
                temp = amdsmi.amdsmi_get_temp_metric(
                    handle,
                    amdsmi.AmdSmiTemperatureType.EDGE,
                    amdsmi.AmdSmiTemperatureMetric.CURRENT,
                )
            except Exception:
                temp = None

            try:
                activity = amdsmi.amdsmi_get_gpu_activity(handle)
                util = activity.get("gfx_activity")
            except Exception:
                util = None

            try:
                driver_info = amdsmi.amdsmi_get_gpu_driver_info(handle)
            except Exception:
                driver_info = {}

            raw_devices.append(
                {
                    "bdf": bdf,
                    "asic_info": asic_info,
                    "vbios_info": vbios_info,
                    "board_info": board_info,
                    "vram_total_bytes": vram_total,
                    "vram_used_bytes": vram_used,
                    "temperature_c": temp,
                    "utilization_pct": util,
                    "driver_info": driver_info,
                }
            )

        return {"rocm_version": rocm_version, "devices": raw_devices}
    finally:
        try:
            amdsmi.amdsmi_shut_down()
        except Exception:
            pass


def get_amd_gpu_info_io() -> SystemGPUInfo:
    """Reads system GPU information, trying amdsmi first, falling back to rocm-smi CLI."""
    rocm_ver_fallback = read_rocm_version_file_io()
    topology = read_kfd_topology_io()
    clocks = read_dpm_clocks_io()
    caches = read_kfd_caches_io()
    try:
        raw_data = read_amdsmi_io()
        return parse_amdsmi_data(raw_data, rocm_ver_fallback, topology, clocks, caches)
    except (ImportError, Exception):
        try:
            stdout_str = read_rocm_smi_cli_io()
            return parse_rocm_smi_json(stdout_str, rocm_ver_fallback, topology, clocks, caches)
        except Exception as e:
            raise RuntimeError(
                f"Failed to retrieve AMD GPU information: {e}"
            ) from e
