# MiniMax 2.7 & Claude Code: Playwright Visual Automation Guide via Desktop VLM Lens

This guide documents how text-only LLMs (specifically **MiniMax 2.7**, Claude Code terminal sessions, or local reasoning models) use the `desktop-vlm-lens` MCP server to interact with web pages and graphical user interfaces visually.

---

## 1. System Prompt Recipe for MiniMax 2.7

Provide this system prompt when initializing your automation session:

```markdown
You are an autonomous web automation agent equipped with Playwright and the `desktop-vlm-lens` visual perception tool suite (`ground_ui_element`, `inspect_image_file`, `capture_and_inspect`, `transcribe_screen_text`).

Although your core model does not process image tokens natively, you have full visual perception by pairing Playwright screenshots with the MCP server:
1. Always take a screenshot after page navigation or significant state changes:
   await page.screenshot(path='current_view.png');
2. To click an element, locate it visually rather than relying on brittle CSS/XPath selectors:
   Call ground_ui_element(element_description='<Description>', image_path='current_view.png')
   Extract `primary_click.x` and `primary_click.y` from the JSON response.
   Execute: await page.mouse.click(x, y);
3. To verify design fidelity, margins, font contrast, or modal dialog centering:
   Call inspect_image_file(image_path='current_view.png', prompt='<Critique Prompt>')
4. To read error banners, toast notices, or capchas:
   Call transcribe_screen_text(image_path='current_view.png')
```

---

## 2. Playwright Python Pattern

```python
from playwright.sync_api import sync_playwright
import json
import requests

def click_visual_element(page, mcp_client, description, screenshot_path="step.png"):
    # 1. Capture current browser viewport
    page.screenshot(path=screenshot_path)
    
    # 2. Call desktop-vlm-lens tool
    # (Example using direct JSON-RPC or client wrapper)
    res = mcp_client.call_tool("ground_ui_element", {
        "element_description": description,
        "image_path": str(Path(screenshot_path).resolve())
    })
    
    data = json.loads(res["content"][0]["text"])
    click_x = data["primary_click"]["x"]
    click_y = data["primary_click"]["y"]
    
    # 3. Click precise pixel coordinates
    page.mouse.click(click_x, click_y)
    print(f"Clicked '{description}' at ({click_x}, {click_y})")

# Example Playwright Flow
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page(viewport={"width": 1280, "height": 720})
    page.goto("https://news.ycombinator.com")
    
    # Click visually without knowing the DOM structure
    click_visual_element(page, mcp, "login link in the top right header")
    page.wait_for_timeout(2000)
    browser.close()
```

---

## 3. Tool Reference & Input Contracts

### `ground_ui_element`
- **Arguments**:
  - `element_description` (string, required): Natural description of the UI target (e.g. *"Red Cancel button"*, *"Top left hamburger menu"*, *"Search input bar"*).
  - `image_path` (string, optional): Absolute path to the screenshot on disk.
  - `target` (string, optional): `"active_window"`, `"fullscreen"`, or `"clipboard"` if inspecting the live screen instead of a file.
- **Output**:
  ```json
  {
    "element_description": "Submit button",
    "primary_click": {"x": 640, "y": 480},
    "matches": [
      {
        "pixel_box": [620, 460, 660, 500],
        "normalized_box": [638, 480, 694, 520]
      }
    ]
  }
  ```

### `inspect_image_file`
- **Arguments**:
  - `image_path` (string, required): File path to analyze.
  - `prompt` (string, required): Design critique or layout question.
- **Use Cases**:
  - Validating responsive breakpoints.
  - Inspecting whether a modal is occluding content.
  - Checking contrast ratios and font readability.

### `transcribe_screen_text`
- **Arguments**:
  - `image_path` (string, optional): Path to screenshot.
  - `crop_bbox` (array, optional): `[ymin, xmin, ymax, xmax]` sub-region to read microscopic text at 100% optical fidelity.
  - `region_description` (string, optional): Focus region (e.g. *"the red error toast at the bottom"*).
- **Use Cases**:
  - Extracting transient alert messages before they fade out.
  - Parsing tabular data or status badges without DOM scraping.

---

## 4. High-Resolution RoI Zoom Recipe (`crop_bbox`)

When targeting microscopic elements (such as 8px modal close buttons, dense table cells, or subtle checkboxes), downsampling the full screen might obscure details. 

Pass `crop_bbox: [ymin, xmin, ymax, xmax]` (normalized `0-1000` or absolute pixels) to any tool:

```python
# Zoom directly into the modal dialog in the center (bounds: y: 200..800, x: 300..900)
res = mcp_client.call_tool("ground_ui_element", {
    "element_description": "Small gray close 'X' icon",
    "image_path": "current_view.png",
    "crop_bbox": [200, 300, 800, 900]
})

# Coordinates in res["primary_click"] are AUTOMATICALLY remapped back to full screen!
click_x = res["primary_click"]["x"]
click_y = res["primary_click"]["y"]
page.mouse.click(click_x, click_y)
```

**Benefits**:
1. **100% Native Optical Fidelity**: The target area is never downscaled.
2. **Ultra-Low Latency**: Consumes only 64–128 visual tokens instead of 1,620 (evaluates in <0.3s).
3. **Automated Remapping**: The returned `click_x` and `click_y` are pre-mapped to the full viewport canvas, so zero client-side coordinate math is needed.
