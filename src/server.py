#!/usr/bin/env python3
"""
Desktop VLM Lens Model Context Protocol (MCP) Server.
Standalone, zero-audio visual perception oracle for text-only LLMs (MiniMax 2.7, Claude Code, Playwright).
Operates over standard stdio JSON-RPC 2.0 on Windows and Linux.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Force UTF-8 on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import capture
import preprocessor
import vlm_client

SERVER_NAME = "desktop-vlm-lens"
SERVER_VERSION = "2.0.0"
PROTOCOL_VERSION = "2024-11-05"


def log_debug(msg: str):
    """Logs strictly to stderr to keep stdout JSON-RPC clean."""
    print(f"[{SERVER_NAME}] {msg}", file=sys.stderr, flush=True)


TOOLS = [
    {
        "name": "capture_and_inspect",
        "description": (
            "Captures the desktop screen (active window, fullscreen, or clipboard snip), "
            "optimizes it to 1024px Lanczos aspect-ratio fit, and answers natural language visual questions. "
            "Supports optional RoI zoom cropping (crop_bbox) for high-resolution micro-region inspection."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Visual question or instruction (e.g. 'Is the modal dialog centered?', 'What error is shown?')."
                },
                "target": {
                    "type": "string",
                    "enum": ["active_window", "fullscreen", "clipboard"],
                    "default": "active_window",
                    "description": "Capture source: 'active_window' (focused app only), 'fullscreen' (all monitors), or 'clipboard' (Win+Shift+S snip)."
                },
                "crop_bbox": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional [ymin, xmin, ymax, xmax] RoI bounding box (0-1000 normalized or pixel coordinates) to zoom into a sub-region."
                },
                "roi_crop": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Alias for crop_bbox [ymin, xmin, ymax, xmax]."
                },
                "max_tokens": {
                    "type": "integer",
                    "default": 250,
                    "description": "Maximum tokens to generate."
                }
            },
            "required": ["prompt"]
        }
    },
    {
        "name": "ground_ui_element",
        "description": (
            "Locates a specific UI button, field, icon, or link on screen or in a screenshot file. "
            "Returns normalized bounding boxes [ymin, xmin, ymax, xmax] AND exact pixel click coordinates "
            "(click_x, click_y) ready for Playwright or OS mouse automation. "
            "Supports optional crop_bbox zooming: coordinates are automatically remapped back to full-canvas coordinates."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "element_description": {
                    "type": "string",
                    "description": "Natural language description of the element to click (e.g. 'Blue Submit button', 'User Avatar icon in top right')."
                },
                "target": {
                    "type": "string",
                    "enum": ["active_window", "fullscreen", "clipboard"],
                    "default": "active_window",
                    "description": "Capture target if inspecting live screen."
                },
                "image_path": {
                    "type": "string",
                    "description": "Optional absolute path to an image file (e.g. Playwright screenshot) instead of live screen capture."
                },
                "crop_bbox": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional [ymin, xmin, ymax, xmax] bounding box (0-1000 normalized or pixels) to zoom into a sub-region. Click coordinates are automatically remapped to full screen."
                },
                "roi_crop": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Alias for crop_bbox [ymin, xmin, ymax, xmax]."
                }
            },
            "required": ["element_description"]
        }
    },
    {
        "name": "transcribe_screen_text",
        "description": (
            "Performs high-precision OCR on the active window, full screen, or a screenshot file. "
            "Extracts dense text, error banners, toast notifications, dialog contents, and status logs. "
            "Supports crop_bbox to isolate and read microscopic text at 100% native optical fidelity."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": ["active_window", "fullscreen", "clipboard"],
                    "default": "active_window",
                    "description": "Capture target."
                },
                "image_path": {
                    "type": "string",
                    "description": "Optional path to a screenshot file on disk."
                },
                "region_description": {
                    "type": "string",
                    "description": "Optional focal region (e.g. 'the red alert banner at the top', 'the login modal')."
                },
                "crop_bbox": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional [ymin, xmin, ymax, xmax] RoI bounding box to crop and transcribe dense or microscopic text."
                },
                "roi_crop": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Alias for crop_bbox [ymin, xmin, ymax, xmax]."
                },
                "language": {
                    "type": "string",
                    "enum": ["auto", "en", "pt"],
                    "default": "auto",
                    "description": "Target language for OCR."
                }
            }
        }
    },
    {
        "name": "inspect_image_file",
        "description": (
            "Inspects a local image file on disk (e.g. Playwright test artifact, mock design, downloaded asset) "
            "and answers natural language questions. Supports crop_bbox for high-resolution sub-rectangle inspection."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Absolute path to the image file on disk."
                },
                "prompt": {
                    "type": "string",
                    "description": "Question or analysis prompt."
                },
                "crop_bbox": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional [ymin, xmin, ymax, xmax] bounding box to zoom into a specific sub-rectangle."
                },
                "roi_crop": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Alias for crop_bbox [ymin, xmin, ymax, xmax]."
                },
                "max_tokens": {
                    "type": "integer",
                    "default": 250,
                    "description": "Maximum tokens to generate."
                }
            },
            "required": ["image_path", "prompt"]
        }
    }
]


def resolve_image(args: Dict[str, Any]) -> tuple[Any, Dict[str, Any], Optional[str]]:
    """Resolves image from live capture, clipboard, or disk file."""
    image_path = args.get("image_path")
    target = args.get("target", "active_window")

    if image_path:
        img, meta = capture.load_image_file(image_path)
        if not img:
            return None, {}, meta.get("error", f"Could not load image at {image_path}")
        return img, meta, None

    if target == "clipboard":
        img, meta = capture.capture_clipboard()
        if not img:
            return None, {}, meta.get("error", "No image found in clipboard.")
        return img, meta, None
    elif target == "fullscreen":
        img, meta = capture.capture_fullscreen()
        return img, meta, None
    else:  # active_window
        img, meta = capture.capture_active_window()
        return img, meta, None


def handle_tool_call(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Executes requested tool and returns MCP content."""
    try:
        crop_param = args.get("crop_bbox") or args.get("roi_crop")

        if tool_name == "capture_and_inspect":
            prompt = args.get("prompt", "").strip()
            if not prompt:
                return {"content": [{"type": "text", "text": "Error: prompt parameter is required."}], "isError": True}

            img, cap_meta, err = resolve_image(args)
            if err:
                return {"content": [{"type": "text", "text": f"Capture Error: {err}"}], "isError": True}

            img_processed, prep_meta = preprocessor.preprocess_image(img, max_dim=1024, crop_bbox=crop_param)
            b64_uri = preprocessor.encode_image_base64(img_processed)

            res = vlm_client.query_vlm(b64_uri, prompt, max_tokens=args.get("max_tokens", 250))
            if res.get("status") != "success":
                return {"content": [{"type": "text", "text": f"VLM Error: {res.get('error')}"}], "isError": True}

            crop_note = ""
            if prep_meta.get("crop_applied"):
                ci = prep_meta.get("crop_info", {})
                crop_note = f"- **RoI Zoom Active**: {ci.get('crop_box')} (Evaluated at native unscaled pixel fidelity)\n"

            out_text = (
                f"### Desktop VLM Inspection\n"
                f"- **Source**: {cap_meta.get('source')} ({cap_meta.get('original_width')}x{cap_meta.get('original_height')})\n"
                f"{crop_note}"
                f"- **Processed Canvas**: {prep_meta.get('processed_width')}x{prep_meta.get('processed_height')}\n"
                f"- **Response**:\n{res.get('content')}\n\n"
                f"- **Latency**: {res.get('duration_sec')}s | **Tokens**: {res.get('prompt_tokens')} in, {res.get('completion_tokens')} out"
            )
            return {"content": [{"type": "text", "text": out_text}], "isError": False}

        elif tool_name == "ground_ui_element":
            desc = args.get("element_description", "").strip()
            if not desc:
                return {"content": [{"type": "text", "text": "Error: element_description is required."}], "isError": True}

            family = vlm_client.detect_model_family()
            if family == "smolvlm":
                msg = (
                    "⚠️ Model Limitation Notice: SmolVLM (Tier 3) is active. "
                    "SmolVLM is designed for ultra-fast binary triage (~1.5s on CPU) and lacks the "
                    "parameter resolution to regress 2D bounding boxes for UI grounding.\n\n"
                    "Recommended action: Switch to Qwen2.5-VL-3B for pixel-accurate grounding:\n"
                    "  python scripts/fetch_models.py --model qwen2.5-vl-3b"
                )
                return {"content": [{"type": "text", "text": msg}], "isError": False}

            img, cap_meta, err = resolve_image(args)
            if err:
                return {"content": [{"type": "text", "text": f"Capture Error: {err}"}], "isError": True}

            img_processed, prep_meta = preprocessor.preprocess_image(img, max_dim=1024, crop_bbox=crop_param)
            b64_uri = preprocessor.encode_image_base64(img_processed)

            if family == "moondream":
                prompt = (
                    f"Point to the '{desc}' in the image. "
                    f"Return ONLY the coordinate as [ymin, xmin, ymax, xmax] normalized to 1000."
                )
            else:
                prompt = (
                    f"Locate the '{desc}' in the image. "
                    f"Return ONLY the 2D bounding box in the format [ymin, xmin, ymax, xmax] normalized to 1000."
                )

            res = vlm_client.query_vlm(b64_uri, prompt, max_tokens=100, temperature=0.1)
            if res.get("status") != "success":
                return {"content": [{"type": "text", "text": f"VLM Error: {res.get('error')}"}], "isError": True}

            boxes = preprocessor.parse_grounding_coordinates(
                res.get("content", ""),
                cap_meta.get("original_width", 1000),
                cap_meta.get("original_height", 1000),
                crop_info=prep_meta.get("crop_info")
            )

            result_data = {
                "element_description": desc,
                "source": cap_meta.get("source"),
                "canvas_size": [cap_meta.get("original_width"), cap_meta.get("original_height")],
                "roi_crop_applied": prep_meta.get("crop_applied", False),
                "crop_info": prep_meta.get("crop_info"),
                "matches": boxes,
                "primary_click": {"x": boxes[0]["click_x"], "y": boxes[0]["click_y"]} if boxes else None,
                "raw_model_output": res.get("content"),
                "model_family": family
            }

            out_text = (
                f"### UI Element Grounding ({family.upper()})\n"
                f"```json\n{json.dumps(result_data, indent=2)}\n```\n\n"
            )
            if boxes:
                p = boxes[0]
                remap_note = " (remapped from RoI crop to full canvas)" if p.get("remapped_from_crop") else ""
                out_text += f"**Automation Action**: Click target at pixel coordinates `(x={p['click_x']}, y={p['click_y']})`{remap_note}."
            else:
                out_text += "⚠️ No bounding boxes identified. Try refining the element description."

            return {"content": [{"type": "text", "text": out_text}], "isError": False}

        elif tool_name == "transcribe_screen_text":
            family = vlm_client.detect_model_family()
            if family == "smolvlm":
                msg = (
                    "⚠️ Model Limitation Notice: SmolVLM (Tier 3) is active. "
                    "SmolVLM does not support dense multilingual OCR transcription.\n\n"
                    "Recommended action: Switch to Qwen2.5-VL-3B for dense OCR:\n"
                    "  python scripts/fetch_models.py --model qwen2.5-vl-3b"
                )
                return {"content": [{"type": "text", "text": msg}], "isError": False}

            img, cap_meta, err = resolve_image(args)
            if err:
                return {"content": [{"type": "text", "text": f"Capture Error: {err}"}], "isError": True}

            img_processed, prep_meta = preprocessor.preprocess_image(img, max_dim=1024, crop_bbox=crop_param)
            b64_uri = preprocessor.encode_image_base64(img_processed)

            region = args.get("region_description")
            lang = args.get("language", "auto")

            if region:
                prompt = f"Transcribe all visible text in {region}. Preserve layout and exact wording."
            else:
                prompt = "Transcribe all visible text on screen, including buttons, headers, error banners, and dialogs."

            if lang == "pt":
                prompt += " The text is in Portuguese."
            elif lang == "en":
                prompt += " The text is in English."

            res = vlm_client.query_vlm(b64_uri, prompt, max_tokens=400, temperature=0.1)
            if res.get("status") != "success":
                return {"content": [{"type": "text", "text": f"VLM Error: {res.get('error')}"}], "isError": True}

            crop_note = ""
            if prep_meta.get("crop_applied"):
                ci = prep_meta.get("crop_info", {})
                crop_note = f"- **RoI Zoom Active**: {ci.get('crop_box')} (Native Optical OCR)\n"

            out_text = (
                f"### Screen Text Transcription (OCR)\n"
                f"- **Source**: {cap_meta.get('source')} ({cap_meta.get('original_width')}x{cap_meta.get('original_height')})\n"
                f"{crop_note}"
                f"- **Transcribed Content**:\n{res.get('content')}\n\n"
                f"- **Latency**: {res.get('duration_sec')}s"
            )
            return {"content": [{"type": "text", "text": out_text}], "isError": False}

        elif tool_name == "inspect_image_file":
            img_path = args.get("image_path", "").strip()
            prompt = args.get("prompt", "").strip()
            if not img_path or not prompt:
                return {"content": [{"type": "text", "text": "Error: Both image_path and prompt are required."}], "isError": True}

            img, cap_meta, err = resolve_image({"image_path": img_path})
            if err:
                return {"content": [{"type": "text", "text": f"File Error: {err}"}], "isError": True}

            img_processed, prep_meta = preprocessor.preprocess_image(img, max_dim=1024, crop_bbox=crop_param)
            b64_uri = preprocessor.encode_image_base64(img_processed)

            res = vlm_client.query_vlm(b64_uri, prompt, max_tokens=args.get("max_tokens", 250))
            if res.get("status") != "success":
                return {"content": [{"type": "text", "text": f"VLM Error: {res.get('error')}"}], "isError": True}

            crop_note = ""
            if prep_meta.get("crop_applied"):
                ci = prep_meta.get("crop_info", {})
                crop_note = f"- **RoI Zoom Active**: {ci.get('crop_box')}\n"

            out_text = (
                f"### Image File Inspection\n"
                f"- **File**: `{img_path}`\n"
                f"- **Dimensions**: {cap_meta.get('original_width')}x{cap_meta.get('original_height')} -> {prep_meta.get('processed_width')}x{prep_meta.get('processed_height')}\n"
                f"{crop_note}"
                f"- **Response**:\n{res.get('content')}\n\n"
                f"- **Latency**: {res.get('duration_sec')}s"
            )
            return {"content": [{"type": "text", "text": out_text}], "isError": False}

        else:
            return {"content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}], "isError": True}

    except Exception as e:
        log_debug(f"Unhandled exception in tool {tool_name}: {e}")
        return {"content": [{"type": "text", "text": f"Server exception: {str(e)}"}], "isError": True}


def run_stdio_server():
    """Main JSON-RPC 2.0 loop reading from stdin and writing to stdout."""
    log_debug(f"Starting {SERVER_NAME} v{SERVER_VERSION} stdio JSON-RPC loop")

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as err:
            log_debug(f"JSON parse error: {err}")
            continue

        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if method == "initialize":
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION}
                }
            }
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()

        elif method == "tools/list":
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": TOOLS}
            }
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()

        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            result = handle_tool_call(tool_name, tool_args)
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": result
            }
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()

        elif method == "notifications/initialized":
            pass

        else:
            if req_id is not None:
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"}
                }
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h", "help"):
        print("Desktop VLM Lens MCP Server (Stdio JSON-RPC 2.0)")
        print("Backend:", vlm_client.resolve_best_server_url())
        print("Tools available: capture_and_inspect, ground_ui_element, transcribe_screen_text, inspect_image_file")
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] in ("--ping", "ping", "test"):
        url = vlm_client.resolve_best_server_url()
        print(f"VLM Server URL: {url}")
        print("Port 8085 active:", vlm_client.is_port_open(port=8085))
        sys.exit(0)
    run_stdio_server()
