# 👁️ Desktop VLM Lens (`desktop-vlm-lens`)

**A portable, cross-platform Vision-Language Model (VLM) Model Context Protocol (MCP) server that gives "eyes" to text-only LLMs (MiniMax 2.7, Claude Code, DeepSeek-R1) and browser automation frameworks (Playwright).**

Runs 100% locally on standard office laptop CPUs (AVX2) with zero admin rights required, or on developer workstations with dedicated NVIDIA RTX GPUs (<0.8s latency). **Zero audio overhead**, **zero external network egress**, and **zero corporate data exfiltration**.

---

## 🚀 Key Capabilities

1. **Gives Eyes to Text-Only Models**: Text-only LLMs like **MiniMax 2.7** or terminal-based agents can now visually inspect user interfaces, debug rendered HTML, critique design aesthetics, and read error dialogs.
2. **Playwright & UI Automation Grounding (`ground_ui_element`)**: Takes a natural language description (e.g. *"First economy news headline link"*), locates it via VLM, and returns the element's bounding box in pixels `[x1, y1, x2, y2]` of the original image alongside a click point `(click_x, click_y)` for instant mouse clicks (`page.mouse.click(x, y)`). Qwen2.5-VL answers in absolute pixels of the resized image it saw; the lens converts that back to your image, and on a live window capture also returns screen coordinates (`primary_click_screen`).
3. **Visual Design Verification (`inspect_image_file`)**: Checks layouts, font hierarchy, contrast ratios, and centering on Playwright screenshots without fragile DOM selectors.
4. **Dual-Mode Screen Capture (`capture_and_inspect`)**:
   - **Active Window Capture**: Focuses exclusively on the foreground application to eliminate multi-monitor sprawl.
   - **Clipboard Ingestion (`Win + Shift + S`)**: Directly inspects snipped screenshots copied to the system clipboard without saving them to disk first.
5. **Image budget per question**: Screens are downsampled with Lanczos to a size that fits the question: grounding and reading both default to 1024 px (about 1000 visual tokens; 768 px is about 600), and every tool takes `detail` = `scene` (512 px) / `read` (768 px) / `precise` (1024 px). Cost grows with the pixel count; on a CPU a cold 1024 px image takes about 40-60 s. Measured click error (17 elements on 3 pages with DOM ground truth, 1280-1920 px canvases): median 33 px at 512 px, 4.6 px at 768 px, 2.2 px at 1024 px. Do not click from a 512 px grounding. **Small text needs a crop**: on a 1080p screen, unguessable 8-16 px codes were read 0/36 at 768 px, 12/36 at 1024 px and 36/36 from a native `crop_bbox` of the text block (also 36/36 on 1440p), so crop around fine print and fall back to full-screen reading only for large text.
5b. **Frame reuse**: A live capture returns a `frame_id`. Pass it to the next call (`frame_id`) to ask more questions about the *same* image: no new screenshot, identical pixels, and the server's prompt cache answers in a few seconds instead of re-reading the image. Frames are kept in memory only (3, five minutes) and never written to disk.
6. **Standalone by design**: The lens only talks to a `llama-server` on **this machine** (`127.0.0.1:8085`), started for you if a binary and a model are under the lens. There is no discovery of other machines and no fallback to one. A non-local `VLM_SERVER_URL` is refused unless you also set `VLM_ALLOW_REMOTE=1`. If no server is available the tool returns a clear error that lists what was tried.
7. **Self-Healing Auto-Spawn**: Checks if `llama-server` is active on port 8085; if not, transparently launches the local Windows/Linux binary in the background (stock llama.cpp: model plus a separate F16 `mmproj-*.gguf` projector; a single Ollama-format GGUF with the projector embedded also works).

---

## 📊 Supported Model Tiers & Capability Governance

| Tier & Model | Size (RAM) | Typical CPU Latency | Typical GPU Latency | `ground_ui_element` | `inspect_image_file` | `transcribe_screen_text` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Qwen2.5-VL-3B** *(Recommended)* | ~3.2 GB | **8s – 14s** | **0.14s – 0.78s** | ✅ **Full Support** (Pixel-accurate click targets) | ✅ **Full Support** (Deep UI critique & layout) | ✅ **Full Support** (Dense multilingual OCR) |
| **Tier 2: Moondream2 (1.8B)** *(Balanced)* | ~1.1 GB | **2.5s – 4.5s** | **~0.10s** | ⚡ **Center Point** (Points to elements) | ✅ **Full Support** (Fast visual inspection) | ⚠️ **Partial** (Headings only, fine-print limited) |
| **Tier 3: SmolVLM (500M)** *(Triage Only)* | ~500 MB | **1.2s – 2.0s** | **~0.05s** | ❌ **Disabled** (Spatial resolution limited) | ⚡ **Basic Triage** (Modal / error check) | ❌ **Disabled** (No dense OCR capacity) |

> [!NOTE]
> When **SmolVLM** is active, calling `ground_ui_element` or dense `transcribe_screen_text` returns an explicit, structured limitation notice guiding the agent to switch to Qwen2.5-VL-3B rather than generating hallucinated coordinates.

---

## 🛠️ Exposed MCP Tools

| Tool Signature | Purpose | Parameters |
| :--- | :--- | :--- |
| `ground_ui_element` | Locates a specific UI button, icon, or link. Returns normalized boxes and pixel click coordinates `(click_x, click_y)` remapped to full screen. | `element_description` *(req)*, `target`, `image_path`, `crop_bbox` |
| `inspect_image_file` | Inspects a local image file on disk (Playwright artifact, mock design, download) and answers design questions. | `image_path` *(req)*, `prompt` *(req)*, `crop_bbox`, `max_tokens` |
| `transcribe_screen_text` | High-precision OCR for error banners, dialogs, form fields, and logs at 100% native optical fidelity. | `target`, `image_path`, `crop_bbox`, `region_description`, `language` (`"auto"` \| `"en"` \| `"pt"`) |
| `capture_and_inspect` | Captures active window, full screen, or clipboard and answers natural language visual questions. | `prompt` *(req)*, `target` (`"active_window"` \| `"fullscreen"` \| `"clipboard"`), `crop_bbox`, `max_tokens` |

---

## 💻 1-Click Windows Setup (Office Laptop with MiniMax 2.7)

Designed to run completely offline on Windows 10/11 with zero admin privileges required.

### Method A: 1-Click Setup Wizard
Double-click `setup_windows.bat` or run in PowerShell:
```powershell
.\setup_windows.ps1
```
The wizard will:
1. Verify Python 3.10+ and build a local `venv` with lightweight dependencies.
2. Probe hardware capabilities via `scripts/detect_hardware.py` (CPU AVX-512/AVX2, Intel Iris Xe / AMD Radeon Vulkan, NVIDIA CUDA, and Intel hybrid P/E-cores).
3. Automatically recommend the optimal hardware execution profile:
   - **Vulkan iGPU (`win-vulkan-x64`)**: Zero-admin acceleration for Intel Iris Xe / Arc and AMD Radeon 780M/890M.
   - **NVIDIA GPU (`win-cuda-12.4-x64`)**: Sub-second CUDA 12 offload (<0.8s).
   - **AVX-512 CPU (`win-avx512-x64`)**: Vector speedup for AMD Zen 4/5 and Intel Xeon.
   - **Standard CPU (`win-cpu-x64`)**: Standard corporate fallback.
4. Prompt for model tier (**1. Qwen2.5-VL-3B**, **2. Moondream2**, **3. SmolVLM**).
5. Offer to **Download automatically** or **Print copy-paste cURL commands** (for restricted proxies).
6. Run an automated diagnostic smoke test (`scripts/verify_environment.py`).

---

### Method B: Manual cURL Acquisition (For Proxy / Metered Networks)

#### 1. Download `llama-server` Binary (Select Your Hardware Profile)
```powershell
# Option 1: Windows Vulkan (Intel Iris Xe / Arc & AMD Radeon iGPU - RECOMMENDED FOR LAPTOPS):
curl.exe -L -o bin\llama-b11342-bin-win-vulkan-x64.zip "https://github.com/ggml-org/llama.cpp/releases/download/b11342/llama-b11342-bin-win-vulkan-x64.zip"
tar.exe -xf bin\llama-b11342-bin-win-vulkan-x64.zip -C bin\
del bin\llama-b11342-bin-win-vulkan-x64.zip

# Option 2: Windows CPU Universal (Standard Corporate Laptop / AVX-512):
curl.exe -L -o bin\llama-b11342-bin-win-cpu-x64.zip "https://github.com/ggml-org/llama.cpp/releases/download/b11342/llama-b11342-bin-win-cpu-x64.zip"
tar.exe -xf bin\llama-b11342-bin-win-cpu-x64.zip -C bin\
del bin\llama-b11342-bin-win-cpu-x64.zip
```

#### 2. Download Recommended Qwen2.5-VL-3B Model Weights
```powershell
curl.exe -L -o models\Qwen2.5-VL-3B-Instruct.gguf "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
curl.exe -L -o models\mmproj-Qwen2.5-VL-3B-Instruct.gguf "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"
```

---

## ⚙️ Registering with MiniMax 2.7 / Claude Code / Agy

In your agent's MCP settings configuration file (e.g. `~/.gemini/config/mcp_config.json` or agent CLI settings):

```json
{
  "mcpServers": {
    "desktop-vlm-lens": {
      "command": "C:\\path\\to\\desktop-vlm-lens\\venv\\Scripts\\python.exe",
      "args": [
        "C:\\path\\to\\desktop-vlm-lens\\src\\server.py"
      ],
      "env": {
        "VLM_SERVER_URL": "http://127.0.0.1:8085"
      }
    }
  }
}
```

### Starting the Server
The MCP server automatically starts the local engine when a tool is called. You can also start the background engine manually anytime:
```powershell
.\launch_vlm_server.bat
```

---

## 🎭 Playwright Automation Example (MiniMax 2.7 Prompt)

Here is how a text-only LLM agent uses `desktop-vlm-lens` to automate an intractable UI without fragile CSS selectors:

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

- **Zero Cloud Egress**: All visual tokens remain strictly in RAM on local CPU or GPU.
- **Zero Microphone / Audio Code**: No audio daemons, Whisper STT, or Pocket-TTS libraries are bundled or executed.
- **Corporate Privacy**: Screenshots never leave `127.0.0.1`.

---

## 🐳 Containerization, GHCR Releases & Image Pinning

For headless Linux server or Nomad cluster deployments:
1. **Multi-Arch Dockerfile**: Codified at `Dockerfile` based on `python:3.12-slim`.
2. **Automated CI/CD**: `.github/workflows/docker-publish.yml` automatically builds and pushes multi-arch images on every push or release tag:
   `ghcr.io/travt/desktop-vlm-lens:latest`
3. **Immutable Image Pinning (ADR-39)**:
   When deployed in container environments, images are pinned by exact SHA256 digest (`image = "ghcr.io/travt/desktop-vlm-lens:latest@sha256:..."`) with `force_pull = false` to guarantee cold-boot zero-deadlock resilience.
4. **Renovate Ingestion**: Matches the `"own images"` package rule in `renovate.json`, generating automated pull requests when new release digests are published.

