#!/usr/bin/env python3
"""
Grounding accuracy benchmark with DOM ground truth.

Renders a few HTML pages with Playwright (Chromium), reads every target element's true box from the DOM,
runs the real `ground_ui_element` tool on the screenshot at several image budgets, and reports how far the
click point lands from the true centre. Needs `pip install playwright` and a Chromium; talks only to the
local VLM server (same rules as the lens itself).

  python scripts/grounding_benchmark.py                  # 512, 768, 1024 px
  python scripts/grounding_benchmark.py --px 1024 --pages login
  python scripts/grounding_benchmark.py --coarse         # also: 512 px locate, then crop and re-ground
"""

import argparse
import json
import math
import os
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

import server  # noqa: E402

LOGIN = """<body style="margin:0;font:16px sans-serif;background:#f4f5f7">
<div style="position:absolute;left:380px;top:180px;width:520px;padding:40px;background:#fff;border-radius:8px">
<h2 style="margin:0 0 24px">Welcome back</h2>
<label>Username</label><br><input data-t="Username text input field" style="width:480px;height:36px;margin:6px 0 18px"><br>
<label>Password</label><br><input data-t="Password text input field" type="password" style="width:480px;height:36px;margin:6px 0 18px"><br>
<button data-t="blue Sign in button" style="width:480px;height:44px;background:#1d64d8;color:#fff;border:0;border-radius:4px;font-size:17px">Sign in</button>
<p style="margin:18px 0 0"><a data-t="Forgot password? link" href="#">Forgot password?</a></p>
</div></body>"""

DASH = """<body style="margin:0;font:15px sans-serif;background:#fff">
<div style="height:56px;background:#1f2933;color:#fff;display:flex;align-items:center;padding:0 24px;gap:28px">
<b>Acme Cloud</b><a data-t="Billing link in the top bar" style="color:#fff">Billing</a><a style="color:#fff">Projects</a><a style="color:#fff">Support</a>
<span style="flex:1"></span>
<svg data-t="bell (notifications) icon in the top right" width="24" height="24" viewBox="0 0 24 24"><path d="M12 2a6 6 0 0 0-6 6v4l-2 3h16l-2-3V8a6 6 0 0 0-6-6zm-2 17a2 2 0 0 0 4 0z" fill="#fff"/></svg>
<svg data-t="gear (settings) icon in the top right" width="24" height="24" viewBox="0 0 24 24"><circle cx="12" cy="12" r="5" fill="none" stroke="#fff" stroke-width="3"/><circle cx="12" cy="12" r="10" fill="none" stroke="#fff" stroke-width="2" stroke-dasharray="3 3"/></svg>
</div>
<div style="padding:32px 48px"><h2>Servers</h2>
<input data-t="Search servers input" placeholder="Search servers" style="width:360px;height:34px">
<table style="margin-top:24px;width:1000px;border-collapse:collapse">
<tr style="background:#eef"><th align=left>Name</th><th align=left>Region</th><th align=left>Status</th><th></th></tr>
<tr style="height:48px"><td>web-01</td><td>eu-west</td><td>running</td><td><button data-t="Restart button in the web-01 row">Restart</button></td></tr>
<tr style="height:48px"><td>db-02</td><td>us-east</td><td>stopped</td><td><button data-t="red Delete button in the db-02 row" style="background:#c0392b;color:#fff;border:0;padding:6px 14px">Delete</button></td></tr>
</table>
<div style="position:absolute;right:48px;bottom:40px"><button data-t="green Create server button in the bottom right" style="background:#1e8e3e;color:#fff;border:0;padding:14px 26px;font-size:16px">Create server</button></div>
</div></body>"""

FORM = """<body style="margin:0;font:14px sans-serif;background:#fff;padding:30px 50px">
<h3>Account settings</h3>
<p><label><input data-t="Receive newsletter checkbox" type="checkbox"> Receive newsletter</label></p>
<p>Country <select data-t="Country dropdown" style="width:220px;height:30px"><option>Brazil</option><option>Portugal</option></select></p>
<p>Plan: <label><input data-t="Free plan radio button" type="radio" name="p"> Free</label> <label><input type="radio" name="p" checked> Pro</label></p>
<p style="font-size:12px">By saving you agree to the <a data-t="Terms of service link" href="#">Terms of service</a>.</p>
<p><button data-t="Save changes button" style="padding:8px 20px">Save changes</button> <button data-t="Cancel button" style="padding:8px 20px">Cancel</button></p>
</body>"""

PAGES = {
    "login": ("login 1280x800", LOGIN, (1280, 800)),
    "dash": ("dashboard 1920x1080", DASH, (1920, 1080)),
    "form": ("form 1366x768", FORM, (1366, 768)),
}


def render(page_key, tmpdir):
    """Screenshot plus true element boxes [(description, x1, y1, x2, y2)]."""
    from playwright.sync_api import sync_playwright
    _, html, (w, h) = PAGES[page_key]
    path = Path(tmpdir) / f"{page_key}.png"
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=1)
        page.set_content(html)
        boxes = page.eval_on_selector_all(
            "[data-t]", "els => els.map(e => {const r = e.getBoundingClientRect(); return [e.dataset.t, r.left, r.top, r.right, r.bottom]})")
        page.screenshot(path=str(path))
        browser.close()
    return path, boxes


def ground(image_path, desc, px, crop=None, frame=None):
    os.environ["VLM_PRECISE_PX"] = str(px)
    args = {"element_description": desc, "image_path": str(image_path)}
    if crop:
        args["crop_bbox"] = crop
    t0 = time.time()
    res = server.handle_tool_call("ground_ui_element", args)
    dt = time.time() - t0
    text = res["content"][0]["text"]
    m = re.search(r"```json\n(.*?)\n```", text, re.S)
    data = json.loads(m.group(1)) if m else {}
    return data, dt, res["isError"], text


def score(data, truth):
    click = data.get("primary_click")
    _, x1, y1, x2, y2 = truth
    if not click:
        return None, False
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return math.hypot(click["x"] - cx, click["y"] - cy), (x1 <= click["x"] <= x2 and y1 <= click["y"] <= y2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--px", type=int, nargs="+", default=[512, 768, 1024])
    ap.add_argument("--pages", nargs="+", default=list(PAGES), choices=list(PAGES))
    ap.add_argument("--coarse", action="store_true", help="also try 512 px locate -> crop -> re-ground")
    ap.add_argument("--out", help="write raw rows as JSON")
    a = ap.parse_args()

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for key in a.pages:
            path, boxes = render(key, tmp)
            label = PAGES[key][0]
            w, h = PAGES[key][2]
            print(f"\n== {label}: {len(boxes)} targets", flush=True)
            for px in a.px:
                for i, truth in enumerate(boxes):
                    data, dt, err, _ = ground(path, truth[0], px)
                    dist, inside = score(data, truth)
                    rows.append({"page": key, "px": px, "target": truth[0], "err_px": dist, "inside": inside,
                                 "sec": round(dt, 1), "first_on_image": i == 0, "mode": "direct"})
                    print(f"  {px:>4}px  {truth[0][:46]:<46} err={'none' if dist is None else f'{dist:5.1f}'}px"
                          f" inside={inside} {dt:5.1f}s", flush=True)
            if a.coarse:
                # all 512 px locates first (one image, so the server's prompt cache serves all but the first)
                coarse = [(truth, *ground(path, truth[0], 512)[:2]) for truth in boxes]
                for truth, data, dt1 in coarse:
                    click = data.get("primary_click")
                    if not click:
                        rows.append({"page": key, "px": "512>crop", "target": truth[0], "err_px": None, "inside": False,
                                     "sec": round(dt1, 1), "first_on_image": False, "mode": "coarse"})
                        continue
                    half_x, half_y = 0.22 * w, 0.22 * h
                    x0, y0 = max(0, click["x"] - half_x), max(0, click["y"] - half_y)
                    x1_, y1_ = min(w, click["x"] + half_x), min(h, click["y"] + half_y)
                    crop = [int(y0 / h * 1000), int(x0 / w * 1000), int(y1_ / h * 1000), int(x1_ / w * 1000)]
                    data2, dt2, _, _ = ground(path, truth[0], 1024, crop=crop)
                    dist, inside = score(data2, truth)
                    rows.append({"page": key, "px": "512>crop", "target": truth[0], "err_px": dist, "inside": inside,
                                 "sec": round(dt1 + dt2, 1), "first_on_image": False, "mode": "coarse"})
                    print(f"  512>crop {truth[0][:46]:<46} err={'none' if dist is None else f'{dist:5.1f}'}px"
                          f" inside={inside} {dt1 + dt2:5.1f}s", flush=True)

    print("\n== summary (click error vs DOM centre)")
    print(f"{'budget':>9} {'n':>3} {'found':>6} {'inside':>7} {'median':>8} {'p90':>8} {'max':>8} {'cold s':>7} {'warm s':>7}")
    for px in a.px + (["512>crop"] if a.coarse else []):
        sub = [r for r in rows if r["px"] == px]
        errs = sorted(r["err_px"] for r in sub if r["err_px"] is not None)
        if not sub:
            continue
        cold = [r["sec"] for r in sub if r["first_on_image"]]
        warm = [r["sec"] for r in sub if not r["first_on_image"]]
        p90 = errs[min(len(errs) - 1, int(len(errs) * 0.9))] if errs else float("nan")
        print(f"{str(px):>9} {len(sub):>3} {len(errs):>6} {sum(1 for r in sub if r['inside']):>7} "
              f"{(statistics.median(errs) if errs else float('nan')):>8.1f} {p90:>8.1f} {(errs[-1] if errs else float('nan')):>8.1f} "
              f"{(statistics.mean(cold) if cold else float('nan')):>7.1f} {(statistics.mean(warm) if warm else float('nan')):>7.1f}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
