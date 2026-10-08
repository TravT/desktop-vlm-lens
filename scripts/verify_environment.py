#!/usr/bin/env python3
"""
Environment & Capability Diagnostic Suite for Desktop VLM Lens.
Validates dependencies, checks models and binaries, inspects port 8085,
and reports active model capabilities and limitations.
"""

import sys
import os
import time
from pathlib import Path

# Force UTF-8 encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

import vlm_client
import preprocessor


def check_python():
    print(f"* Python Version: {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info < (3, 10):
        print("  [WARN] Python 3.10+ recommended.")
    else:
        print("  [OK] Python version OK.")


def check_dependencies():
    for pkg in ["PIL", "requests"]:
        try:
            __import__(pkg)
            print(f"  [OK] Package '{pkg}' is installed.")
        except ImportError:
            print(f"  [ERROR] Missing package: '{pkg}'. Run: pip install -r requirements.txt")


def check_models():
    model_file, mmproj_file = vlm_client.find_local_model()
    if not model_file:
        print("  [ERROR] No GGUF model found in models/.")
        print("     Run: python scripts/fetch_models.py")
        return None
    size_mb = model_file.stat().st_size / (1024 * 1024)
    print(f"  [OK] Detected Model: {model_file.name} ({size_mb:.1f} MB)")
    if mmproj_file and mmproj_file != model_file:
        proj_mb = mmproj_file.stat().st_size / (1024 * 1024)
        print(f"  [OK] Multimodal Projector: {mmproj_file.name} ({proj_mb:.1f} MB)")
    else:
        print("  [OK] Unified GGUF: Vision projector embedded in model weights.")

    name = model_file.name.lower()
    if "qwen" in name:
        return "qwen2.5-vl-3b"
    elif "moondream" in name:
        return "moondream2"
    elif "smol" in name:
        return "smolvlm-500m"
    return "custom"


def check_binaries():
    bin_file = vlm_client.find_local_binary()
    if not bin_file:
        print("  [ERROR] No llama-server binary found in bin/ or system PATH.")
        print("     Run: python scripts/fetch_binaries.py")
        return False
    print(f"  [OK] Detected llama-server: {bin_file}")
    return True


def check_server_health(model_profile):
    res = vlm_client.resolve_server()
    print(f"\n--- Checking Server Endpoint: {res.url or 'none (' + str(res.error) + ')'} ---")
    active = vlm_client.is_port_open(port=8085)
    if active:
        print("  [OK] Local port 8085 is ACTIVE and listening.")
    else:
        print("  [INFO] Port 8085 is offline. Testing auto-spawn capability...")
        t0 = time.time()
        ok = vlm_client.auto_spawn_llama_server()
        if ok:
            print(f"  [OK] Auto-spawn successful in {time.time() - t0:.1f}s!")
        else:
            print("  [ERROR] Auto-spawn failed. Ensure binary and model are configured.")
            return

    # Print model capabilities & limitations
    print("\n--- Model Capabilities & Limitations Profile ---")
    if model_profile == "qwen2.5-vl-3b":
        print("  * Profile: Qwen2.5-VL-3B (Tier 1: Full Power)")
        print("     - ground_ui_element: FULL SUPPORT (click targets from pixel boxes at 1024 px)")
        print("     - inspect_image_file: FULL SUPPORT (UI layout, contrast, design critique)")
        print("     - transcribe_screen_text: FULL SUPPORT (Dense multilingual OCR)")
    elif model_profile == "moondream2":
        print("  * Profile: Moondream2 (Tier 2: Balanced)")
        print("     - ground_ui_element: SUPPORTED (Center-point pointing)")
        print("     - inspect_image_file: FULL SUPPORT (Fast layout inspection)")
        print("     - transcribe_screen_text: PARTIAL (Headings only, fine-print limited)")
    elif model_profile == "smolvlm-500m":
        print("  * Profile: SmolVLM (Tier 3: Ultra-Fast Triage)")
        print("     - ground_ui_element: DISABLED (Parameter capacity insufficient for bounding boxes)")
        print("     - inspect_image_file: SUPPORTED (Simple scene description & modal triage)")
        print("     - transcribe_screen_text: DISABLED (Dense OCR not supported)")
    else:
        print(f"  * Profile: {model_profile} (Custom Model)")


def check_hardware():
    try:
        import detect_hardware
        hw = detect_hardware.load_cached_profile() or detect_hardware.detect(cache=True)
        c = hw.get("cpu", {})
        g = hw.get("gpu", {})
        opt = hw.get("optimal_args", {})
        print(f"  * CPU: {c.get('model', 'Unknown')} ({c.get('physical_cores', '?')} physical cores)")
        if c.get("is_hybrid"):
            print(f"    - Intel Hybrid: {c.get('p_cores')} P-cores, {c.get('e_cores')} E-cores (Bound to {opt.get('threads')} threads)")
        print(f"    - Vector Extensions: AVX2: {'YES' if c.get('avx2') else 'NO'} | AVX-512: {'YES' if c.get('avx512') else 'NO'}")
        if g.get("detected"):
            print(f"  * GPU: {g.get('model')} ({g.get('backend', 'cpu').upper()} offload, {g.get('vram_mb', 0)} MB)")
        else:
            print(f"  * GPU: None (Pure CPU mode)")
        if opt.get("device"):
            print(f"  * Configured Target Device: {opt.get('device').upper()}")
        print(f"  * Recommended Binary: {hw.get('recommended_flavor')}")
        print(f"  * Visual Token Budget: {opt.get('image_max_tokens', 512)} tokens (latency <3s)")
        if opt.get("background_priority"):
            print(f"  * Process Scheduling: BelowNormal (Background Mode)")
    except Exception as e:
        print(f"  [WARN] Hardware capability probe: {e}")


def main():
    print("=======================================================")
    print("Desktop VLM Lens: Diagnostic & Environment Check")
    print("=======================================================\n")

    print("[1/5] Python Environment:")
    check_python()
    check_dependencies()

    print("\n[2/5] Hardware & Acceleration Topology:")
    check_hardware()

    print("\n[3/5] Model Weights:")
    profile = check_models()

    print("\n[4/5] Inference Binaries:")
    has_bin = check_binaries()

    if profile and has_bin:
        print("\n[5/5] Server Connectivity & Auto-Spawn:")
        check_server_health(profile)
    else:
        print("\n[5/5] Skipping live server check: Missing model or binary.")

    print("\n=======================================================")
    print("Diagnostic Complete.")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
