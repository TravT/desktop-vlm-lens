#!/usr/bin/env python3
"""
Small-text reading benchmark with known content.

Renders a page whose text block has lines at 8-16 px, each holding three random codes (such as KX42-81) that
cannot be guessed, then runs the real `transcribe_screen_text` tool on the screenshot at several image
budgets and counts how many codes come back exactly, per font size. Shows where a downscaled screenshot
stops being readable and whether a native-resolution crop (`crop_bbox`) fixes it. Needs Playwright and a
local VLM server, like grounding_benchmark.py.

  python scripts/text_benchmark.py                      # 1920x1080 and 2560x1440, read / precise / crop
  python scripts/text_benchmark.py --canvas 1920x1080 --render-only /tmp/text.png
"""

import argparse
import json
import random
import re
import sys
import tempfile
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

SIZES = [8, 9, 10, 12, 14, 16]
LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"
DIGITS = "23456789"


def make_lines(seed=7):
    rnd = random.Random(seed)
    lines = []
    for size in SIZES:
        for _ in range(2):
            codes = [f"{rnd.choice(LETTERS)}{rnd.choice(LETTERS)}{rnd.choice(DIGITS)}{rnd.choice(DIGITS)}-{rnd.choice(DIGITS)}{rnd.choice(DIGITS)}"
                     for _ in range(3)]
            lines.append((size, codes))
    return lines


def render(canvas, path, seed=7):
    """Screenshot at `canvas` (w, h); returns (lines, block_rect) with the text block's box in pixels."""
    from playwright.sync_api import sync_playwright
    w, h = canvas
    lines = make_lines(seed)
    rows = "".join(f'<div style="font-size:{s}px;line-height:{s + 8}px;white-space:nowrap">{s}px&nbsp;&nbsp;{"&nbsp;&nbsp;".join(c)}</div>'
                   for s, c in lines)
    html = f'<body style="margin:0;background:#fff;color:#111;font-family:sans-serif"><div id="b" style="position:absolute;left:60px;top:50px;padding:6px">{rows}</div></body>'
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=1)
        page.set_content(html)
        r = page.eval_on_selector("#b", "e => { const r = e.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; }")
        page.screenshot(path=str(path))
        browser.close()
    return lines, r


def crop_for(rect, canvas, margin=12):
    w, h = canvas
    x1, y1, x2, y2 = rect
    return [max(0, int((y1 - margin) / h * 1000)), max(0, int((x1 - margin) / w * 1000)),
            min(1000, int((y2 + margin) / h * 1000) + 1), min(1000, int((x2 + margin) / w * 1000) + 1)]


def score(text, lines):
    """{size: (found, total)} counting codes found verbatim in the transcription."""
    flat = re.sub(r"\s+", "", text.upper())
    out = {}
    for size, codes in lines:
        f, t = out.get(size, (0, 0))
        out[size] = (f + sum(1 for c in codes if c in flat), t + len(codes))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--canvas", nargs="+", default=["1920x1080", "2560x1440"])
    ap.add_argument("--render-only", help="write the screenshot here and exit")
    ap.add_argument("--out", help="write raw rows as JSON")
    a = ap.parse_args()

    import os
    import server
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for cv in a.canvas:
            w, h = map(int, cv.split("x"))
            path = Path(a.render_only) if a.render_only else Path(tmp) / f"text_{cv}.png"
            lines, rect = render((w, h), path)
            if a.render_only:
                print(f"rendered {path} block={[round(v) for v in rect]} crop_bbox={crop_for(rect, (w, h))}")
                return
            crop = crop_for(rect, (w, h))
            print(f"\n== canvas {cv}: text block {round(rect[2] - rect[0])}x{round(rect[3] - rect[1])} px", flush=True)
            for name, args in (("read 768", {"detail": "read"}), ("precise 1024", {"detail": "precise"}),
                               ("crop native", {"detail": "precise", "crop_bbox": crop})):
                os.environ.pop("VLM_READ_PX", None)
                t0 = time.time()
                res = server.handle_tool_call("transcribe_screen_text", {"image_path": str(path), **args})
                dt = time.time() - t0
                text = res["content"][0]["text"]
                sc = score(text, lines)
                rows.append({"canvas": cv, "config": name, "sec": round(dt, 1), "scores": {str(k): v for k, v in sc.items()}})
                cells = "  ".join(f"{s}px {sc[s][0]}/{sc[s][1]}" for s in SIZES)
                print(f"  {name:13} {dt:5.1f}s  {cells}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
