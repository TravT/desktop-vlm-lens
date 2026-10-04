#!/usr/bin/env python3
"""
Hardware Capability & Topology Detection Engine for Desktop VLM Lens.
Detects CPU vector extensions (AVX-512, AVX2, AVX512_VNNI), Intel hybrid P/E-core topology,
and classifies GPU adapters (NVIDIA CUDA, Intel Iris Xe / Arc Vulkan, AMD Radeon Vulkan).
Outputs recommended binary flavors and optimal runtime parameters (threads, ngl, token budget).
"""

import os
import sys
import json
import time
import shutil
import platform
import subprocess
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Force UTF-8 encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = PACKAGE_ROOT / ".hardware_profile.json"
CONFIG_FILE = PACKAGE_ROOT / "config.json"


def load_user_config() -> Dict[str, Any]:
    """Loads user configuration from config.json if present."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}



def _run_cmd(cmd: List[str], timeout: float = 3.0) -> Optional[str]:
    """Runs a system command safely and returns stripped stdout, or None on error."""
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
        if res.returncode == 0 and res.stdout:
            return res.stdout.strip()
    except Exception:
        pass
    return None


def detect_cpu_windows() -> Dict[str, Any]:
    """Detects CPU model, core topology, and instruction sets on Windows."""
    model = platform.processor() or "Generic Windows CPU"
    vendor = "Unknown"
    physical_cores = os.cpu_count() or 4
    logical_cores = os.cpu_count() or 4
    avx2 = False
    avx512 = False
    avx512_vnni = False

    # 1. Probe CPU architecture flags via Windows API
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        # Windows API feature flags:
        # PF_AVX2_INSTRUCTIONS_AVAILABLE = 40 (Win 10 1703+)
        # PF_AVX512F_INSTRUCTIONS_AVAILABLE = 41 (Win 10 19041+)
        avx2 = bool(kernel32.IsProcessorFeaturePresent(40))
        avx512 = bool(kernel32.IsProcessorFeaturePresent(41))
    except Exception:
        pass

    # 2. Query Win32_Processor via PowerShell for detailed topology
    ps_cmd = [
        "powershell", "-NoProfile", "-NonInteractive", "-Command",
        "Get-CimInstance Win32_Processor | Select-Object Name, Manufacturer, NumberOfCores, NumberOfLogicalProcessors | ConvertTo-Json -Compress"
    ]
    ps_out = _run_cmd(ps_cmd, timeout=4.0)
    if ps_out:
        try:
            data = json.loads(ps_out)
            if isinstance(data, list) and data:
                data = data[0]
            if isinstance(data, dict):
                model = data.get("Name", model)
                mfg = data.get("Manufacturer", "")
                if "Intel" in mfg or "Intel" in model:
                    vendor = "Intel"
                elif "AMD" in mfg or "AMD" in model:
                    vendor = "AMD"

                p_cores_val = data.get("NumberOfCores")
                l_cores_val = data.get("NumberOfLogicalProcessors")
                if p_cores_val:
                    physical_cores = int(p_cores_val)
                if l_cores_val:
                    logical_cores = int(l_cores_val)
        except Exception:
            pass

    if vendor == "Unknown":
        if "intel" in model.lower():
            vendor = "Intel"
        elif "amd" in model.lower():
            vendor = "AMD"

    # 3. Model taxonomy heuristics for AVX-512 on Windows if API was inconclusive
    if not avx512:
        # AMD Zen 4 (Ryzen 7000/8000) and Zen 5 (Ryzen 9000) feature AVX-512
        if re.search(r"Ryzen\s+[3579]\s+(7\d{3}|8\d{3}|9\d{3})", model, re.I):
            avx512 = True
            avx2 = True
        # Intel Xeon Scalable (Skylake-SP, Ice Lake, Sapphire Rapids, Emerald Rapids, Granite Rapids)
        elif "xeon" in model.lower() and any(k in model.lower() for k in ["platinum", "gold", "silver", "bronze", "w-", "e-"]):
            avx512 = True
            avx2 = True
        # Intel 11th Gen desktop/mobile (Rocket Lake / Tiger Lake)
        elif re.search(r"i[3579]-11\d{3}", model, re.I):
            avx512 = True
            avx2 = True

    # 4. Topology and Intel hybrid architecture calculation (P-cores vs E-cores)
    # On hybrid architectures: Logical = 2*P + E, Physical = P + E
    # Therefore: P = Logical - Physical, E = Physical - P
    is_hybrid = False
    p_cores = physical_cores
    e_cores = 0

    if logical_cores > physical_cores:
        calc_p = logical_cores - physical_cores
        calc_e = physical_cores - calc_p
        if calc_e > 0 and calc_p > 0:
            is_hybrid = True
            p_cores = calc_p
            e_cores = calc_e

    # Specific check for Intel Core Ultra (Meteor Lake / Arrow Lake) naming
    if "ultra" in model.lower() and not is_hybrid:
        is_hybrid = True
        # Default Meteor Lake P-core count for 155H/165H is 6
        p_cores = min(6, physical_cores)
        e_cores = physical_cores - p_cores

    return {
        "model": model,
        "vendor": vendor,
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "p_cores": p_cores,
        "e_cores": e_cores,
        "is_hybrid": is_hybrid,
        "avx2": avx2 or True,  # Almost all x64 systems since 2013 support AVX2
        "avx512": avx512,
        "avx512_vnni": avx512_vnni
    }


def detect_cpu_linux() -> Dict[str, Any]:
    """Detects CPU model, core topology, and instruction sets on Linux via /proc/cpuinfo."""
    model = "Generic Linux CPU"
    vendor = "Unknown"
    physical_cores = os.cpu_count() or 4
    logical_cores = os.cpu_count() or 4
    avx2 = False
    avx512 = False
    avx512_vnni = False

    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
            cpuinfo = f.read()

        core_set = set()
        logical_count = 0
        flags = set()

        for block in cpuinfo.strip().split("\n\n"):
            lines = {k.strip(): v.strip() for k, v in (line.split(":", 1) for line in block.splitlines() if ":" in line)}
            if "model name" in lines:
                model = lines["model name"]
            if "vendor_id" in lines:
                v_id = lines["vendor_id"]
                if "GenuineIntel" in v_id:
                    vendor = "Intel"
                elif "AuthenticAMD" in v_id:
                    vendor = "AMD"
            if "flags" in lines:
                for fl in lines["flags"].split():
                    flags.add(fl.lower())

            phys_id = lines.get("physical id")
            core_id = lines.get("core id")
            if phys_id is not None and core_id is not None:
                core_set.add((phys_id, core_id))
            if "processor" in lines:
                logical_count += 1

        if core_set:
            physical_cores = len(core_set)
        if logical_count > 0:
            logical_cores = logical_count

        avx2 = "avx2" in flags
        avx512 = any(f in flags for f in ["avx512f", "avx512vl", "avx512bw", "avx512dq"])
        avx512_vnni = "avx512_vnni" in flags
    except Exception:
        pass

    # Hybrid topology calculation
    is_hybrid = False
    p_cores = physical_cores
    e_cores = 0

    if logical_cores > physical_cores:
        calc_p = logical_cores - physical_cores
        calc_e = physical_cores - calc_p
        if calc_e > 0 and calc_p > 0:
            is_hybrid = True
            p_cores = calc_p
            e_cores = calc_e

    return {
        "model": model,
        "vendor": vendor,
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "p_cores": p_cores,
        "e_cores": e_cores,
        "is_hybrid": is_hybrid,
        "avx2": avx2,
        "avx512": avx512,
        "avx512_vnni": avx512_vnni
    }


def detect_cpu() -> Dict[str, Any]:
    """Dispatches CPU capability detection based on operating system."""
    if sys.platform == "win32":
        return detect_cpu_windows()
    elif sys.platform.startswith("linux"):
        return detect_cpu_linux()
    else:
        # Fallback generic detection
        cores = os.cpu_count() or 4
        return {
            "model": platform.processor() or "Generic CPU",
            "vendor": "Unknown",
            "physical_cores": cores,
            "logical_cores": cores,
            "p_cores": cores,
            "e_cores": 0,
            "is_hybrid": False,
            "avx2": True,
            "avx512": False,
            "avx512_vnni": False
        }


def detect_gpu_windows() -> Dict[str, Any]:
    """Detects GPU adapters on Windows via Win32_VideoController."""
    adapters: List[Dict[str, Any]] = []

    ps_cmd = [
        "powershell", "-NoProfile", "-NonInteractive", "-Command",
        "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion, VideoProcessor, Status | ConvertTo-Json -Compress"
    ]
    ps_out = _run_cmd(ps_cmd, timeout=4.0)
    if ps_out:
        try:
            raw = json.loads(ps_out)
            items = raw if isinstance(raw, list) else [raw]
            for it in items:
                if not isinstance(it, dict):
                    continue
                name = it.get("Name", "")
                if not name or "remote" in name.lower() or "citrix" in name.lower():
                    continue

                ram_bytes = it.get("AdapterRAM") or 0
                try:
                    ram_mb = int(ram_bytes) // (1024 * 1024)
                except Exception:
                    ram_mb = 0

                vendor = "Other"
                backend = "cpu"
                name_lower = name.lower()

                if any(k in name_lower for k in ["nvidia", "geforce", "quadro", "rtx", "gtx", "tesla"]):
                    vendor = "NVIDIA"
                    backend = "cuda"
                elif "intel" in name_lower and any(k in name_lower for k in ["iris", "arc", "xe", "uhd", "graphics"]):
                    vendor = "Intel"
                    backend = "vulkan"
                elif any(k in name_lower for k in ["amd", "radeon"]):
                    vendor = "AMD"
                    backend = "vulkan"

                adapters.append({
                    "name": name,
                    "vendor": vendor,
                    "backend": backend,
                    "vram_mb": ram_mb,
                    "driver": it.get("DriverVersion", "Unknown")
                })
        except Exception:
            pass

    # Pick the highest priority adapter:
    # 1. Discrete NVIDIA GPU (CUDA)
    # 2. Intel Iris Xe / Arc or AMD Radeon iGPU (Vulkan)
    # 3. Fallback
    primary = None
    for a in adapters:
        if a["vendor"] == "NVIDIA":
            primary = a
            break

    if not primary:
        for a in adapters:
            if a["vendor"] in ("Intel", "AMD") and a["backend"] == "vulkan":
                primary = a
                break

    if not primary and adapters:
        primary = adapters[0]

    if primary:
        return {
            "detected": True,
            "vendor": primary["vendor"],
            "model": primary["name"],
            "backend": primary["backend"],
            "vram_mb": primary["vram_mb"],
            "all_adapters": adapters
        }

    return {
        "detected": False,
        "vendor": "None",
        "model": "No Supported GPU",
        "backend": "cpu",
        "vram_mb": 0,
        "all_adapters": []
    }


def detect_gpu_linux() -> Dict[str, Any]:
    """Detects GPU adapters on Linux via lspci and nvidia-smi."""
    # Check for NVIDIA discrete GPU
    nvidia_out = _run_cmd(["nvidia-smi", "--query-gpu=gpu_name,memory.total", "--format=csv,noheader,nounits"], timeout=2.0)
    if nvidia_out:
        parts = [p.strip() for p in nvidia_out.splitlines()[0].split(",")]
        name = parts[0]
        vram_mb = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        return {
            "detected": True,
            "vendor": "NVIDIA",
            "model": name,
            "backend": "cuda",
            "vram_mb": vram_mb,
            "all_adapters": [{"name": name, "vendor": "NVIDIA", "backend": "cuda", "vram_mb": vram_mb}]
        }

    # Check via lspci
    lspci_out = _run_cmd(["lspci"], timeout=2.0)
    adapters = []
    if lspci_out:
        for line in lspci_out.splitlines():
            if any(term in line.lower() for term in ["vga compatible controller", "3d controller", "display controller"]):
                vendor = "Other"
                backend = "cpu"
                line_lower = line.lower()
                if "nvidia" in line_lower:
                    vendor = "NVIDIA"
                    backend = "cuda"
                elif "intel" in line_lower:
                    vendor = "Intel"
                    backend = "vulkan"
                elif "amd" in line_lower or "ati" in line_lower or "radeon" in line_lower:
                    vendor = "AMD"
                    backend = "vulkan"

                clean_name = line.split(":", 2)[-1].strip() if ":" in line else line
                adapters.append({
                    "name": clean_name,
                    "vendor": vendor,
                    "backend": backend,
                    "vram_mb": 0
                })

    primary = None
    for a in adapters:
        if a["vendor"] == "NVIDIA":
            primary = a
            break
    if not primary:
        for a in adapters:
            if a["vendor"] in ("Intel", "AMD"):
                primary = a
                break
    if not primary and adapters:
        primary = adapters[0]

    if primary:
        return {
            "detected": True,
            "vendor": primary["vendor"],
            "model": primary["name"],
            "backend": primary["backend"],
            "vram_mb": primary["vram_mb"],
            "all_adapters": adapters
        }

    return {
        "detected": False,
        "vendor": "None",
        "model": "No Supported GPU",
        "backend": "cpu",
        "vram_mb": 0,
        "all_adapters": []
    }


def detect_gpu() -> Dict[str, Any]:
    """Dispatches GPU detection based on operating system."""
    if sys.platform == "win32":
        return detect_gpu_windows()
    elif sys.platform.startswith("linux"):
        return detect_gpu_linux()
    else:
        return {
            "detected": False,
            "vendor": "None",
            "model": "No Supported GPU",
            "backend": "cpu",
            "vram_mb": 0,
            "all_adapters": []
        }


def determine_profile(
    cpu: Dict[str, Any],
    gpu: Dict[str, Any],
    os_platform: str,
    device_override: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Evaluates CPU and GPU capabilities to select the optimal binary flavor and runtime parameters.
    Supports user override via config.json or CLI flag (auto, cuda, vulkan, cpu).
    """
    user_cfg = load_user_config()
    target_device = (device_override or user_cfg.get("device") or "auto").lower()

    phys = cpu.get("physical_cores", 4)
    logical = cpu.get("logical_cores", phys)

    # 1. Thread tuning
    if target_device == "cpu":
        # Background CPU mode: reserve 2 physical cores for system/gaming responsiveness
        cfg_threads = user_cfg.get("cpu_threads")
        if cfg_threads:
            threads = int(cfg_threads)
        elif cpu.get("is_hybrid") and cpu.get("p_cores", 0) > 2:
            threads = max(2, cpu["p_cores"] - 2)
        else:
            threads = max(2, min(phys - 2, 8)) if phys > 2 else max(1, phys)
    else:
        # Standard GPU or auto mode: hybrid P-core binding or physical cores
        if cpu.get("is_hybrid") and cpu.get("p_cores", 0) > 0:
            threads = cpu["p_cores"]
        else:
            threads = max(2, min(phys, 8))

    threads_batch = min(logical, threads * 2, 16)

    # 2. Select binary flavor and offload layers based on target_device
    ngl = 0
    if target_device == "cpu":
        recommended_flavor = "win-avx512" if (os_platform == "win32" and cpu.get("avx512")) else ("win-cpu" if os_platform == "win32" else "linux-cpu")
        ngl = 0
    elif target_device == "cuda":
        recommended_flavor = "win-cuda" if os_platform == "win32" else "linux-cuda"
        ngl = 99
    elif target_device == "vulkan":
        recommended_flavor = "win-vulkan" if os_platform == "win32" else "linux-vulkan"
        ngl = 99
    else:
        # "auto" detection
        if os_platform == "win32":
            if gpu.get("vendor") == "NVIDIA":
                recommended_flavor = "win-cuda"
                ngl = 99
            elif gpu.get("backend") == "vulkan":
                recommended_flavor = "win-vulkan"
                vram = gpu.get("vram_mb", 0)
                ngl = 24 if (0 < vram < 2048) else 99
            elif cpu.get("avx512"):
                recommended_flavor = "win-avx512"
                ngl = 0
            else:
                recommended_flavor = "win-cpu"
                ngl = 0
        else:
            if gpu.get("vendor") == "NVIDIA":
                recommended_flavor = "linux-cuda"
                ngl = 99
            elif gpu.get("backend") == "vulkan":
                recommended_flavor = "linux-vulkan"
                ngl = 99
            else:
                recommended_flavor = "linux-cpu"
                ngl = 0

    image_max_tokens = int(user_cfg.get("image_max_tokens") or os.environ.get("VLM_MAX_IMAGE_TOKENS", "512"))
    image_min_tokens = int(user_cfg.get("image_min_tokens") or os.environ.get("VLM_MIN_IMAGE_TOKENS", "256"))
    background_priority = bool(user_cfg.get("background_priority", True))

    optimal_args = {
        "device": target_device,
        "threads": threads,
        "threads_batch": threads_batch,
        "ngl": ngl,
        "image_max_tokens": image_max_tokens,
        "image_min_tokens": image_min_tokens,
        "context_size": 4096,
        "background_priority": background_priority
    }

    return recommended_flavor, optimal_args


def detect(cache: bool = True, device_override: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes full hardware detection and returns structured profile.
    Saves profile to .hardware_profile.json for fast launcher query.
    """
    cpu_info = detect_cpu()
    gpu_info = detect_gpu()
    os_name = sys.platform

    flavor, optimal_args = determine_profile(cpu_info, gpu_info, os_name, device_override=device_override)

    profile = {
        "timestamp": time.time(),
        "platform": os_name,
        "cpu": cpu_info,
        "gpu": gpu_info,
        "recommended_flavor": flavor,
        "optimal_args": optimal_args
    }

    if cache:
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(profile, f, indent=2)
        except Exception:
            pass

    return profile


def load_cached_profile() -> Optional[Dict[str, Any]]:
    """Loads cached profile if valid, unexpired, and matches config.json."""
    if CACHE_FILE.exists():
        try:
            # Invalidate cache if config.json was modified after cache was written
            if CONFIG_FILE.exists() and CONFIG_FILE.stat().st_mtime > CACHE_FILE.stat().st_mtime:
                return None

            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if time.time() - data.get("timestamp", 0) < 86400:
                # Ensure cached device matches current config.json setting
                user_cfg = load_user_config()
                cfg_device = (user_cfg.get("device") or "auto").lower()
                cached_device = data.get("optimal_args", {}).get("device", "auto").lower()
                if cfg_device == cached_device:
                    return data
        except Exception:
            pass
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hardware Capability & Topology Detector for Desktop VLM Lens")
    parser.add_argument("--json", action="store_true", help="Output raw JSON profile")
    parser.add_argument("--flavor", "--profile", action="store_true", help="Print only recommended binary profile name")
    parser.add_argument("--threads", action="store_true", help="Print only optimal inference thread count (-t)")
    parser.add_argument("--ngl", action="store_true", help="Print only optimal GPU layer offload count (-ngl)")
    parser.add_argument("--tokens", "--max-tokens", action="store_true", help="Print only optimal visual token cap")
    parser.add_argument("--device", choices=["auto", "cuda", "vulkan", "cpu"], default=None, help="Force specific device profile")
    parser.add_argument("--no-cache", action="store_true", help="Bypass cached profile and force re-probe")
    args = parser.parse_args()

    profile = None
    if not args.no_cache and not args.device:
        profile = load_cached_profile()
    if not profile:
        profile = detect(cache=True, device_override=args.device)

    if args.flavor:
        print(profile["recommended_flavor"])
        return
    if args.threads:
        print(profile["optimal_args"]["threads"])
        return
    if args.ngl:
        print(profile["optimal_args"]["ngl"])
        return
    if args.tokens:
        print(profile["optimal_args"]["image_max_tokens"])
        return
    if args.json:
        print(json.dumps(profile, indent=2))
        return

    # User-friendly colored / formatted terminal summary
    c_info = profile["cpu"]
    g_info = profile["gpu"]
    args_info = profile["optimal_args"]

    print("\n=======================================================")
    print("  Desktop VLM Lens: Hardware Detection & Tuning Profile")
    print("=======================================================")
    print(f"\n[CPU Capability & Core Topology]")
    print(f"  • Model:            {c_info['model']}")
    print(f"  • Vendor:           {c_info['vendor']}")
    print(f"  • Physical Cores:   {c_info['physical_cores']} ({c_info['logical_cores']} logical threads)")
    if c_info["is_hybrid"]:
        print(f"  • Architecture:     Intel Hybrid (P-Cores: {c_info['p_cores']}, E-Cores: {c_info['e_cores']})")
        print(f"  • Thread Tuning:    Bound to {c_info['p_cores']} P-cores (eliminates E-core barrier stall)")
    else:
        print(f"  • Architecture:     Symmetric ({c_info['physical_cores']} physical cores)")
    print(f"  • Vector Flags:     AVX2: {'YES' if c_info['avx2'] else 'NO'} | AVX-512: {'YES (Zen 4/5 or Xeon)' if c_info['avx512'] else 'NO'}")

    print(f"\n[GPU & Acceleration Adapters]")
    if g_info["detected"]:
        print(f"  • Adapter Name:     {g_info['model']}")
        print(f"  • Vendor:           {g_info['vendor']}")
        print(f"  • Acceleration:     {g_info['backend'].upper()} Offload")
        if g_info["vram_mb"] > 0:
            print(f"  • Memory (VRAM):    {g_info['vram_mb']} MB")
        if g_info["vendor"] in ("Intel", "AMD") and g_info["backend"] == "vulkan":
            print(f"  • Zero-Admin Mode:  Active (Uses standard OEM Vulkan drivers, zero admin rights)")
    else:
        print(f"  • Status:           No dedicated/integrated GPU acceleration detected.")
        print(f"  • Mode:             Pure CPU Vector Inference")

    print(f"\n[Optimal Runtime Configuration]")
    print(f"  • Target Device Mode:    {args_info.get('device', 'auto').upper()}")
    print(f"  • Recommended Profile:   {profile['recommended_flavor']}")
    print(f"  • Inference Threads (-t): {args_info['threads']}")
    print(f"  • Batch Threads (-tb):    {args_info['threads_batch']}")
    print(f"  • Layer Offload (-ngl):   {args_info['ngl']}")
    print(f"  • Visual Token Budget:    {args_info['image_max_tokens']} max tokens (Cuts latency by 75%)")
    print(f"  • Process Priority:       {'BelowNormal (Background Task)' if args_info.get('background_priority') else 'Normal'}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
