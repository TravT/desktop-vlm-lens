import sys
import time
from pathlib import Path
from typing import Optional
from PIL import Image, ImageDraw

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

import subprocess
import vlm_client
import preprocessor

def get_vram_usage() -> Optional[int]:
    """Queries nvidia-smi for current VRAM usage in MB."""
    try:
        cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        if res.returncode == 0 and res.stdout.strip():
            return int(res.stdout.strip().split()[0])
    except Exception:
        pass
    return None

def run_benchmark():
    print("=======================================================")
    print("  Desktop VLM Lens: Live Hardware & Speedup Benchmark")
    print("=======================================================\n")

    vram_start = get_vram_usage()
    if vram_start is not None:
        print(f"[Telemetry] Initial GPU VRAM Usage: {vram_start} MB")

    # 1. Create a 1920x1080 mock UI canvas with a blue button
    test_img_path = PACKAGE_ROOT / "test_mock_ui.png"
    img = Image.new("RGB", (1920, 1080), color=(240, 244, 248))
    draw = ImageDraw.Draw(img)
    # Blue button at [xmin=800, ymin=500, xmax=1120, ymax=580]
    draw.rectangle([800, 500, 1120, 580], fill=(24, 119, 242))
    draw.text((880, 530), "Submit Order", fill=(255, 255, 255))
    img.save(test_img_path)
    print(f"[1/4] Generated test UI image: {test_img_path.name} (1920x1080)")

    # 2. Test Full-Screen Inference with 512 token budget
    t0 = time.time()
    img_proc, meta = preprocessor.preprocess_image(img, max_dim=1024)
    b64 = preprocessor.encode_image_base64(img_proc)
    prep_ms = (time.time() - t0) * 1000

    print(f"[2/4] Full-Screen Query (with --image-max-tokens 512 budget):")
    t1 = time.time()
    res1 = vlm_client.query_vlm(
        b64,
        "Locate the blue 'Submit Order' button. Return bounding box [ymin, xmin, ymax, xmax] normalized to 1000.",
        max_tokens=100
    )
    t_full = time.time() - t1

    print(f"      • Status:            {res1.get('status')}")
    print(f"      • Response:          {res1.get('content')}")
    print(f"      • Latency (TTFT):    {t_full:.3f}s")
    print(f"      • Prompt Tokens:     {res1.get('prompt_tokens')}")
    print(f"      • Completion Tokens: {res1.get('completion_tokens')}")

    boxes1 = preprocessor.parse_grounding_coordinates(res1.get("content", ""), 1920, 1080)
    if boxes1:
        p = boxes1[0]
        print(f"      • Click Target:      (x={p['click_x']}, y={p['click_y']})")
        print(f"      • Expected Target:   (x=960, y=540)")

    # 3. Test RoI Crop-on-Demand Zooming
    print(f"\n[3/4] RoI Crop-on-Demand Zooming (Target modal area [ymin=450, xmin=750, ymax=650, xmax=1150]):")
    t2 = time.time()
    crop_bbox = [450, 750, 650, 1150]
    img_cropped, crop_meta = preprocessor.preprocess_image(img, max_dim=1024, crop_bbox=crop_bbox)
    b64_crop = preprocessor.encode_image_base64(img_cropped)
    
    res2 = vlm_client.query_vlm(
        b64_crop,
        "Locate the button in this cropped region. Return bounding box [ymin, xmin, ymax, xmax] normalized to 1000.",
        max_tokens=100
    )
    t_crop = time.time() - t2

    print(f"      • Status:            {res2.get('status')}")
    print(f"      • Response:          {res2.get('content')}")
    print(f"      • Latency:           {t_crop:.3f}s")
    print(f"      • Prompt Tokens:     {res2.get('prompt_tokens')} (Conserves token budget!)")

    boxes2 = preprocessor.parse_grounding_coordinates(
        res2.get("content", ""),
        1920, 1080,
        crop_info=crop_meta.get("crop_info")
    )
    if boxes2:
        p2 = boxes2[0]
        print(f"      • Remapped Click:    (x={p2['click_x']}, y={p2['click_y']})")
        print(f"      • Remapped Flag:     {p2.get('remapped_from_crop')}")

    # 4. Cleanup
    if test_img_path.exists():
        test_img_path.unlink()

    print("\n=======================================================")
    print("  Benchmark Summary:")
    print(f"  • Full Canvas Latency: {t_full:.3f}s (Prompt tokens: {res1.get('prompt_tokens')})")
    print(f"  • RoI Crop Latency:    {t_crop:.3f}s (Prompt tokens: {res2.get('prompt_tokens')})")
    vram_end = get_vram_usage()
    if vram_end is not None:
        print(f"  • GPU VRAM Allocated:  {vram_end} MB")
    print("=======================================================\n")

if __name__ == "__main__":
    run_benchmark()
