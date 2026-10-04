#!/usr/bin/env python3
"""
Binary Acquisition Manager for Desktop VLM Lens.
Downloads official upstream llama-server prebuilt binaries for Windows or Linux.
Supports hardware-adaptive acceleration:
  - Windows Vulkan (Intel Iris Xe / Arc & AMD Radeon iGPU zero-admin offload)
  - Windows CUDA 12 (NVIDIA RTX/GTX discrete GPU)
  - Windows AVX-512 (AMD Zen 4/5 & Intel Xeon vector acceleration)
  - Windows CPU AVX2 (Standard corporate laptop fallback)
  - Linux Vulkan & CPU AVX2/AVX-512
Provides both automated download and copy-paste cURL commands.
"""

import os
import sys
import zipfile
import tarfile
import argparse
import urllib.request
from pathlib import Path

# Force UTF-8 encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = PACKAGE_ROOT / "bin"
SCRIPTS_DIR = Path(__file__).resolve().parent

# Pinned high-stability upstream llama.cpp release
LLAMA_CPP_RELEASE = "b11342"
GITHUB_BASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_RELEASE}"

BINARY_CATALOG = {
    "win-vulkan": {
        "name": "Windows Vulkan (Intel Iris Xe / Arc & AMD Radeon iGPU Offload)",
        "platform": "win32",
        "description": "Zero-admin GPU acceleration using native Windows OEM Vulkan drivers (vulkan-1.dll). 25-45 tok/s.",
        "archives": [
            {
                "url": f"{GITHUB_BASE}/llama-{LLAMA_CPP_RELEASE}-bin-win-vulkan-x64.zip",
                "filename": f"llama-{LLAMA_CPP_RELEASE}-bin-win-vulkan-x64.zip"
            }
        ]
    },
    "win-cuda": {
        "name": "Windows NVIDIA GPU (CUDA 12 Acceleration)",
        "platform": "win32",
        "description": "Offloads all layers to NVIDIA RTX/GTX discrete GPU for sub-second inference (<0.8s).",
        "archives": [
            {
                "url": f"{GITHUB_BASE}/llama-{LLAMA_CPP_RELEASE}-bin-win-cuda-12.4-x64.zip",
                "filename": f"llama-{LLAMA_CPP_RELEASE}-bin-win-cuda-12.4-x64.zip"
            },
            {
                "url": f"{GITHUB_BASE}/cudart-llama-bin-win-cuda-12.4-x64.zip",
                "filename": "cudart-llama-bin-win-cuda-12.4-x64.zip"
            }
        ]
    },
    "win-avx512": {
        "name": "Windows CPU AVX-512 (AMD Zen 4/5 & Intel Xeon)",
        "platform": "win32",
        "description": "Universal multi-architecture CPU runtime with dynamic dispatch for AMD Zen 4/5 and Xeon AVX-512 vector speedup.",
        "archives": [
            {
                "url": f"{GITHUB_BASE}/llama-{LLAMA_CPP_RELEASE}-bin-win-cpu-x64.zip",
                "filename": f"llama-{LLAMA_CPP_RELEASE}-bin-win-cpu-x64.zip"
            }
        ]
    },
    "win-cpu": {
        "name": "Windows CPU Universal AVX2 (Standard Corporate Laptop)",
        "platform": "win32",
        "description": "Runs on standard corporate Intel 8th-11th Gen laptops. No GPU or admin rights required.",
        "archives": [
            {
                "url": f"{GITHUB_BASE}/llama-{LLAMA_CPP_RELEASE}-bin-win-cpu-x64.zip",
                "filename": f"llama-{LLAMA_CPP_RELEASE}-bin-win-cpu-x64.zip"
            }
        ]
    },
    "linux-vulkan": {
        "name": "Linux Ubuntu x64 (Vulkan GPU Acceleration)",
        "platform": "linux",
        "description": "Zero-driver-install Vulkan GPU offload for Linux Intel/AMD/NVIDIA graphics.",
        "archives": [
            {
                "url": f"{GITHUB_BASE}/llama-{LLAMA_CPP_RELEASE}-bin-ubuntu-vulkan-x64.tar.gz",
                "filename": f"llama-{LLAMA_CPP_RELEASE}-bin-ubuntu-vulkan-x64.tar.gz"
            }
        ]
    },
    "linux-cpu": {
        "name": "Linux Ubuntu x64 (CPU AVX2/AVX-512)",
        "platform": "linux",
        "description": "Runs natively on Linux servers or desktop workstations.",
        "archives": [
            {
                "url": f"{GITHUB_BASE}/llama-{LLAMA_CPP_RELEASE}-bin-ubuntu-x64.tar.gz",
                "filename": f"llama-{LLAMA_CPP_RELEASE}-bin-ubuntu-x64.tar.gz"
            }
        ]
    }
}

PROFILE_ALIASES = {
    "win-vulkan-x64": "win-vulkan",
    "win-avx512-x64": "win-avx512",
    "win-cuda-x64": "win-cuda",
    "win-cuda-12.4-x64": "win-cuda",
    "win-cpu-x64": "win-cpu",
    "linux-cpu-x64": "linux-cpu",
    "linux-vulkan-x64": "linux-vulkan",
}


def resolve_profile_key(key: str) -> str:
    """Resolves aliases like win-vulkan-x64 to canonical catalog key win-vulkan."""
    return PROFILE_ALIASES.get(key.lower(), key.lower())


def get_hardware_recommendation() -> tuple[str, dict]:
    """Runs detect_hardware to discover the optimal local profile."""
    try:
        sys.path.insert(0, str(SCRIPTS_DIR))
        import detect_hardware
        profile = detect_hardware.detect(cache=True)
        recommended = profile.get("recommended_flavor", "win-cpu" if sys.platform == "win32" else "linux-cpu")
        return resolve_profile_key(recommended), profile
    except Exception:
        fallback = "win-cpu" if sys.platform == "win32" else "linux-cpu"
        return fallback, {}


def download_file(url: str, dest_path: Path):
    """Downloads a file with streaming progress indicator."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    print(f"Connecting to: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "desktop-vlm-lens/2.0"})

    with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as out_file:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = downloaded / total_size * 100
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  Progress: {percent:5.1f}% [{mb_down:6.1f} MB / {mb_total:6.1f} MB]")
                sys.stdout.flush()
            else:
                sys.stdout.write(f"\r  Downloaded: {downloaded / (1024*1024):6.1f} MB")
                sys.stdout.flush()

    print()
    temp_path.rename(dest_path)
    print(f"Saved: {dest_path.name}")


def extract_archive(archive_path: Path, target_dir: Path):
    """Extracts zip or tar.gz into target directory."""
    print(f"Extracting {archive_path.name} into {target_dir}...")
    if archive_path.name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(target_dir)
    elif archive_path.name.endswith(".tar.gz") or archive_path.name.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as t:
            t.extractall(target_dir)
    archive_path.unlink()  # Clean up archive

    # Ensure executable permissions on POSIX
    if sys.platform != "win32":
        for p in target_dir.glob("**/llama-*"):
            if p.is_file():
                p.chmod(p.stat().st_mode | 0o755)


def print_curl_commands(profile_key: str):
    """Prints copy-paste curl commands for manual binary setup."""
    key = resolve_profile_key(profile_key)
    entry = BINARY_CATALOG[key]
    print(f"\n=======================================================")
    print(f"Manual cURL Download for: {entry['name']}")
    print(f"Description: {entry['description']}")
    print(f"=======================================================\n")

    print("# Windows PowerShell:")
    for a in entry["archives"]:
        print(f"curl.exe -L -o bin\\{a['filename']} \"{a['url']}\"")
        if a["filename"].endswith(".zip"):
            print(f"tar.exe -xf bin\\{a['filename']} -C bin\\")
            print(f"del bin\\{a['filename']}")

    print("\n# Linux / Bash:")
    for a in entry["archives"]:
        print(f"curl -L -o bin/{a['filename']} \"{a['url']}\"")
        if a["filename"].endswith(".zip"):
            print(f"unzip -o bin/{a['filename']} -d bin/ && rm bin/{a['filename']}")
        elif a["filename"].endswith(".tar.gz"):
            print(f"tar -xzf bin/{a['filename']} -C bin/ && rm bin/{a['filename']}")
    print()


def interactive_select() -> tuple[str, bool]:
    """Prompts the user interactively with hardware-recommended default."""
    rec_key, hw_info = get_hardware_recommendation()

    print("\n=======================================================")
    print("  Desktop VLM Lens: Hardware-Adaptive Binary Selection")
    print("=======================================================")
    if hw_info:
        cpu_name = hw_info.get("cpu", {}).get("model", "Unknown CPU")
        gpu_name = hw_info.get("gpu", {}).get("model", "No GPU")
        backend = hw_info.get("gpu", {}).get("backend", "cpu").upper()
        print(f"Detected CPU: {cpu_name}")
        print(f"Detected GPU: {gpu_name} (Mode: {backend})")
        print(f"Recommended Profile: {BINARY_CATALOG.get(rec_key, {}).get('name', rec_key)}")
    print("-------------------------------------------------------\n")

    # Order keys so recommended profile is #1
    all_keys = list(BINARY_CATALOG.keys())
    keys = [rec_key] + [k for k in all_keys if k != rec_key]

    for i, k in enumerate(keys, 1):
        b = BINARY_CATALOG[k]
        is_rec = " [RECOMMENDED - HARDWARE DETECTED]" if k == rec_key else ""
        print(f"  [{i}] {b['name']}{is_rec}")
        print(f"      • {b['description']}\n")

    while True:
        choice = input(f"Select engine [1-{len(keys)}] (default: 1 - {BINARY_CATALOG[rec_key]['name']}): ").strip()
        if not choice:
            selected_key = rec_key
            break
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(keys):
                selected_key = keys[idx]
                break
        except ValueError:
            pass
        print("Invalid choice, enter a valid number.")

    print("\nDownload Method:")
    print("  [1] Automatic Download & Extraction (saves directly to bin/)")
    print("  [2] Print cURL Commands (for manual copy-paste / proxy environments)")

    while True:
        method = input("Select method [1-2] (default: 1): ").strip()
        if not method or method == "1":
            auto_download = True
            break
        elif method == "2":
            auto_download = False
            break
        print("Invalid choice, enter 1 or 2.")

    return selected_key, auto_download


def main():
    valid_profiles = list(BINARY_CATALOG.keys()) + list(PROFILE_ALIASES.keys())
    parser = argparse.ArgumentParser(description="Acquire llama-server prebuilt binaries for Desktop VLM Lens")
    parser.add_argument("--profile", choices=valid_profiles, help="Binary profile to fetch (supports aliases like win-vulkan-x64)")
    parser.add_argument("--device", choices=["auto", "cuda", "vulkan", "cpu"], default=None, help="Target execution device")
    parser.add_argument("--auto", action="store_true", help="Automatically download and extract hardware-recommended binary")
    parser.add_argument("--print-only", action="store_true", help="Print cURL commands without downloading")
    args = parser.parse_args()

    if args.device:
        target_dev = args.device.lower()
        if target_dev == "cpu":
            selected_key = "win-cpu" if sys.platform == "win32" else "linux-cpu"
        elif target_dev == "cuda":
            selected_key = "win-cuda" if sys.platform == "win32" else "linux-cuda"
        elif target_dev == "vulkan":
            selected_key = "win-vulkan" if sys.platform == "win32" else "linux-vulkan"
        else:
            selected_key, _ = get_hardware_recommendation()
        auto_download = not args.print_only
    elif args.auto:
        selected_key, _ = get_hardware_recommendation()
        auto_download = not args.print_only
    elif args.profile:
        selected_key = resolve_profile_key(args.profile)
        auto_download = not args.print_only
    else:
        selected_key, auto_download = interactive_select()

    entry = BINARY_CATALOG[selected_key]

    if not auto_download:
        print_curl_commands(selected_key)
        return

    print(f"\n--- Downloading Binaries: {entry['name']} ---")
    BIN_DIR.mkdir(parents=True, exist_ok=True)

    for a in entry["archives"]:
        archive_path = BIN_DIR / a["filename"]
        try:
            download_file(a["url"], archive_path)
            extract_archive(archive_path, BIN_DIR)
        except Exception as e:
            curl_cmd = "curl.exe" if sys.platform == "win32" else "curl"
            target_path = f"bin\\{a['filename']}" if sys.platform == "win32" else f"bin/{a['filename']}"
            print(f"\nDownload error: {e}")
            print("\nYou can download this binary manually using:")
            print(f"  {curl_cmd} -L -o {target_path} \"{a['url']}\"\n")
            sys.exit(1)

    print("\n[OK] Binary acquisition complete! Verified files in bin/\n")


if __name__ == "__main__":
    main()
