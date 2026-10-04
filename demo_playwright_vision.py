"""
Playwright + Desktop VLM Lens End-to-End Verification Script.
Demonstrates text-only agent visual grounding:
1. Launches Playwright Chromium and navigates to a news portal.
2. Captures page screenshot.
3. Invokes desktop-vlm-lens MCP server to locate an element with bounding box & click coordinates.
4. Executes page.mouse.click() on the exact pixel center.
5. Verifies resulting navigation.
"""

import sys
import time
import json
from pathlib import Path

# Force UTF-8 on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from playwright.sync_api import sync_playwright

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import server


def run_demo():
    print("=" * 70)
    print("🚀 PLAYWRIGHT + DESKTOP VLM LENS END-TO-END DEMO")
    print("=" * 70)

    screenshot_path = Path(__file__).resolve().parent / "news_demo.png"

    with sync_playwright() as p:
        print("[1/4] Launching Playwright Chromium headless browser...")
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()

        url = "https://news.ycombinator.com"
        print(f"[2/4] Navigating to {url}...")
        page.goto(url, wait_until="networkidle", timeout=15000)
        time.sleep(1)

        # Capture initial screenshot
        page.screenshot(path=str(screenshot_path))
        print(f"      📸 Captured screenshot: {screenshot_path} ({screenshot_path.stat().st_size} bytes)")

        # Target: Locate the 'new' link in the top navigation bar
        target_description = "the 'new' navigation link in the orange top bar"
        print(f"\n[3/4] Calling 'ground_ui_element' MCP tool for: '{target_description}'...")
        t0 = time.time()
        res = server.handle_tool_call("ground_ui_element", {
            "image_path": str(screenshot_path),
            "element_description": target_description
        })
        dur = round(time.time() - t0, 2)
        print(f"      Done in {dur}s!")

        raw_text = res["content"][0]["text"]
        print("\n--- MCP Tool Response ---")
        print(raw_text)
        print("-------------------------")

        # Parse the JSON from the markdown code block
        click_x, click_y = None, None
        try:
            json_str = raw_text.split("```json\n")[1].split("\n```")[0]
            data = json.loads(json_str)
            if data.get("primary_click"):
                click_x = data["primary_click"]["x"]
                click_y = data["primary_click"]["y"]
        except Exception as e:
            print(f"Could not parse click coordinates: {e}")

        if click_x is not None and click_y is not None:
            print(f"\n[4/4] 🎯 Simulating exact mouse click at pixel coordinates: ({click_x}, {click_y})...")
            page.mouse.click(click_x, click_y)
            time.sleep(2)

            after_url = page.url
            print(f"      Navigated to URL: {after_url}")
            after_snap = Path(__file__).resolve().parent / "news_after_click.png"
            page.screenshot(path=str(after_snap))
            print(f"      📸 Captured post-click screenshot: {after_snap}")

            if "newest" in after_url:
                print("\n✅ SUCCESS: Successfully visually grounded and clicked navigation element via VLM coordinates!")
            else:
                print(f"\nℹ️ Click executed at ({click_x}, {click_y}), current page: {after_url}")
        else:
            print("\n⚠️ No click coordinates returned from VLM grounding.")

        browser.close()


if __name__ == "__main__":
    run_demo()
