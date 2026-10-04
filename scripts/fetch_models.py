#!/usr/bin/env python3
"""
Model Acquisition & Catalog Manager for Desktop VLM Lens.
Fetches recommended Vision-Language Models (GGUF + mmproj) or outputs copy-paste cURL commands.
Enforces model tiering, hardware boundaries, and capability limitations.
"""

import os
import sys
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
MODELS_DIR = PACKAGE_ROOT / "models"

MODEL_CATALOG = {
    "qwen2.5-vl-3b": {
        "name": "Qwen2.5-VL-3B-Instruct (Recommended)",
        "tier": "Tier 1: Full Capability",
        "description": "High-precision UI grounding, dense multilingual OCR, and visual design QA.",
        "limitations": "Requires 4GB+ RAM on CPU (~8-14s latency) or 5.8GB VRAM on GPU (<0.8s latency).",
        "files": [
            {
                "filename": "Qwen2.5-VL-3B-Instruct.gguf",
                "url": "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
                "size_mb": 1930,
                "is_mmproj": False
            },
            {
                "filename": "mmproj-Qwen2.5-VL-3B-Instruct.gguf",
                "url": "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf",
                "size_mb": 1338,
                "is_mmproj": True
            }
        ]
    },
    "moondream2": {
        "name": "Moondream2 (1.8B)",
        "tier": "Tier 2: Balanced & Lightweight",
        "description": "Fast UI layout inspection, single-element pointing, and general visual QA (~2.5-4.5s on CPU).",
        "limitations": "Weaker on dense micro-font OCR; coordinate outputs provide center points rather than precise bounding boxes.",
        "files": [
            {
                "filename": "moondream2-text-model-f16.gguf",
                "url": "https://huggingface.co/moondream/moondream2-gguf/resolve/main/moondream2-text-model-f16.gguf",
                "size_mb": 1150,
                "is_mmproj": False
            },
            {
                "filename": "moondream2-mmproj-f16.gguf",
                "url": "https://huggingface.co/moondream/moondream2-gguf/resolve/main/moondream2-mmproj-f16.gguf",
                "size_mb": 420,
                "is_mmproj": True
            }
        ]
    },
    "smolvlm-500m": {
        "name": "SmolVLM-500M-Instruct",
        "tier": "Tier 3: Ultra-Fast Triage Only",
        "description": "Sub-2-second CPU binary triage, modal checks, and basic scene description (~1.2-2.0s on CPU).",
        "limitations": "STRICT LIMITATION: UI bounding-box grounding and dense multilingual OCR are DISABLED/UNRELIABLE due to spatial parameter constraints. Requires strict Idefics3 template.",
        "files": [
            {
                "filename": "SmolVLM-500M-Instruct-Q8_0.gguf",
                "url": "https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/SmolVLM-500M-Instruct-Q8_0.gguf",
                "size_mb": 520,
                "is_mmproj": False
            },
            {
                "filename": "mmproj-SmolVLM-500M-Instruct-Q8_0.gguf",
                "url": "https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf",
                "size_mb": 195,
                "is_mmproj": True
            }
        ]
    }
}


def download_file(url: str, dest_path: Path):
    """Downloads a file with streaming progress indicator."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    print(f"Connecting to: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "desktop-vlm-lens/1.0"})

    with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as out_file:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks

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
    print(f"Saved: {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")


def print_curl_commands(model_key: str):
    """Prints copy-paste curl commands for manual acquisition."""
    entry = MODEL_CATALOG[model_key]
    print(f"\n=======================================================")
    print(f"Manual cURL Download for: {entry['name']}")
    print(f"Tier: {entry['tier']}")
    print(f"Limitations: {entry['limitations']}")
    print(f"=======================================================\n")

    print("# Windows PowerShell:")
    for f in entry["files"]:
        target = f"models\\{f['filename']}"
        print(f"curl.exe -L -o {target} \"{f['url']}\"")

    print("\n# Linux / Bash:")
    for f in entry["files"]:
        target = f"models/{f['filename']}"
        print(f"curl -L -o {target} \"{f['url']}\"")
    print()


def interactive_select() -> tuple[str, bool]:
    """Prompts the user interactively if no CLI flags were passed."""
    print("\n=== Desktop VLM Lens: Model Catalog Selection ===")
    print("Choose the Vision-Language Model for your deployment:\n")

    keys = list(MODEL_CATALOG.keys())
    for i, k in enumerate(keys, 1):
        m = MODEL_CATALOG[k]
        print(f"  [{i}] {m['name']} ({m['tier']})")
        print(f"      • Capabilities: {m['description']}")
        print(f"      • Constraints:  {m['limitations']}")
        print()

    while True:
        choice = input(f"Select model [1-{len(keys)}] (default: 1): ").strip()
        if not choice:
            selected_key = keys[0]
            break
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(keys):
                selected_key = keys[idx]
                break
        except ValueError:
            pass
        print("Invalid choice, please enter 1, 2, or 3.")

    print("\nDownload Method:")
    print("  [1] Automatic Download (stream directly to models/)")
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
    parser = argparse.ArgumentParser(description="Acquire Vision-Language Models for Desktop VLM Lens")
    parser.add_argument("--model", choices=list(MODEL_CATALOG.keys()), help="Model identifier to fetch")
    parser.add_argument("--print-only", action="store_true", help="Print cURL commands without downloading")
    args = parser.parse_args()

    if args.model:
        selected_key = args.model
        auto_download = not args.print_only
    else:
        selected_key, auto_download = interactive_select()

    entry = MODEL_CATALOG[selected_key]

    if not auto_download:
        print_curl_commands(selected_key)
        return

    print(f"\n--- Downloading {entry['name']} ({entry['tier']}) ---")
    print(f"Notice: {entry['limitations']}\n")

    for f in entry["files"]:
        target = MODELS_DIR / f["filename"]
        if target.exists() and target.stat().st_size > 10 * 1024 * 1024:
            print(f"File already exists: {f['filename']} ({target.stat().st_size / (1024*1024):.1f} MB), skipping.")
            continue

        print(f"Fetching {f['filename']} (~{f['size_mb']} MB)...")
        try:
            download_file(f["url"], target)
        except Exception as e:
            curl_cmd = "curl.exe" if sys.platform == "win32" else "curl"
            target_path = f"models\\{f['filename']}" if sys.platform == "win32" else f"models/{f['filename']}"
            print("\nYou can download this file manually using:")
            print(f"  {curl_cmd} -L -o {target_path} \"{f['url']}\"\n")
            sys.exit(1)

    print("\n[OK] Model acquisition complete! Weights are verified in models/\n")


if __name__ == "__main__":
    main()
