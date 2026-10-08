# 👁️ Desktop VLM Lens (`desktop-vlm-lens`)

**A portable, cross-platform Vision-Language Model (VLM) Model Context Protocol (MCP) server that gives "eyes" to text-only LLMs (MiniMax 2.7, Claude Code, DeepSeek-R1) and browser automation frameworks (Playwright).**

Runs 100% locally on CPU with **zero audio overhead**, **zero external network egress**, and **zero corporate data exfiltration**.

---

## 🚀 Key Capabilities

1. **Gives Vision to Text-Only Models**: Text-only LLMs like **MiniMax 2.7** or terminal-based agents can now inspect user interfaces, debug canvas elements, and read error dialogs.
2. **Playwright & UI Automation Grounding (`ground_ui_element`)**: Takes a natural language description (e.g. *"Blue Submit button"*), locates it via `Qwen2.5-VL-3B`, and returns the bounding box in pixels `[x1, y1, x2, y2]` (`box_xyxy_pixels`) plus a click point `(click_x, click_y)` in pixels of the original image for instant mouse clicks.
3. **Dual-Mode Screen Capture**:
   - **Active Window Capture**: Focuses exclusively on the foreground application to eliminate multi-monitor sprawl (e.g., 5120×1440 canvas shrinking buttons into unreadable dots).
   - **Clipboard Ingestion (`Win + Shift + S`)**: Directly inspects snipped screenshots copied to the system clipboard without saving them to disk first.
   - **Disk Files**: Analyzes Playwright screenshots (`page.screenshot(path=...)`) or mockups.
4. **Smart 1024px Lanczos Scaling**: Automatically downsamples screenshots to fit the 3B/5B VLM's optimal CPU budget (~300 tokens) while preserving font sharpness and contrast.
5. **Self-Healing Auto-Spawn**: Checks if `llama-server` is active on port 8085; if not, transparently launches the local Windows/Linux binary in the background.

---

## 🛠️ Exposed MCP Tools

| Tool Signature | Purpose | Parameters |
| :--- | :--- | :--- |
| `capture_and_inspect` | Captures active window, full screen, or clipboard and answers natural language visual questions. | `prompt` *(req)*, `target` (`"active_window"` \| `"fullscreen"` \| `"clipboard"`), `roi_crop`, `max_tokens` |
| `ground_ui_element` | Locates a specific UI button, icon, or link. Returns normalized boxes and pixel click coordinates `(click_x, click_y)`. | `element_description` *(req)*, `target`, `image_path` |
| `transcribe_screen_text` | High-precision OCR for error banners, dialogs, form fields, and logs. | `target`, `image_path`, `region_description`, `language` (`"auto"` \| `"en"` \| `"pt"`) |
| `inspect_image_file` | Inspects a local image file on disk (Playwright artifact, mock design, download). | `image_path` *(req)*, `prompt` *(req)*, `max_tokens` |

---

## 💻 Windows Setup (Office Laptop with MiniMax 2.7)

Designed to run completely offline on Windows 10/11 with zero admin privileges required.

### 1. Prerequisites
- Python 3.10+ installed and on PATH.

### 2. Automated Setup
Open PowerShell in the folder and execute:
```powershell
.\setup_windows.ps1
```

### 3. Place Model & Engine Binaries
1. **Download `llama-server.exe`** from [upstream llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases) (e.g., `llama-bXXXX-bin-win-avx2-x64.zip`) and place `llama-server.exe` into `.\bin\`.
2. **Download Qwen2.5-VL-3B GGUF** and vision projector into `.\models\`:
   - `models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf` (~2.1 GB)
   - `models/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf` (~600 MB)

### 4. Configure Your LLM Harness (MiniMax 2.7 / Claude Code)
In your harness MCP settings JSON:

```json
{
  "mcpServers": {
    "desktop-vlm-lens": {
      "command": "C:\\path\\to\\dev\\desktop-vlm-lens\\venv\\Scripts\\python.exe",
      "args": [
        "C:\\path\\to\\dev\\desktop-vlm-lens\\src\\server.py"
      ]
    }
  }
}
```

---

## 🐧 Linux Setup

The tool uses a `llama-server` on this machine, port 8085 (it starts one if it finds a binary and a model under the lens). It never looks for other machines. A server on another host can be used only by setting both `VLM_SERVER_URL` and `VLM_ALLOW_REMOTE=1`.

### 1. Run via Shell Launcher
```bash
./start_linux.sh
```

### 2. Existing local server
The script uses whatever is listening on `http://127.0.0.1:8085` (for example a `llama-server` you started, or a container publishing that port). If nothing is, the tool returns an error that says what was tried.

---

## 🎭 Playwright Automation Example

Here is how a text-only LLM uses `desktop-vlm-lens` to automate an intractable UI without fragile CSS selectors:

```python
# 1. Playwright captures current viewport
page.screenshot(path="checkout_step.png")

# 2. Text-only LLM invokes MCP tool
res = call_mcp_tool("desktop-vlm-lens", "ground_ui_element", {
    "image_path": "checkout_step.png",
    "element_description": "Green 'Place Order' button"
})

# Response returned:
# {
#   "matches": [{"box_xyxy_pixels": [450, 720, 590, 760], "click_x": 520, "click_y": 740}],
#   "primary_click": {"x": 520, "y": 740}
# }

# 3. LLM executes mouse click on the exact target pixel center
page.mouse.click(res["primary_click"]["x"], res["primary_click"]["y"])
```

---

## 🔒 Security & Air-Gap Compliance
- **Zero Cloud Egress**: All visual tokens remain strictly in RAM on CPU.
- **Zero Microphone / Audio Code**: No audio daemons, Whisper STT, or Pocket-TTS libraries are bundled or executed.
- **Memory Footprint**: Consumes < 3.2 GB RAM with Q4_K_M quantization.
