#!/usr/bin/env python3
"""
Desktop VLM Lens Model Context Protocol (MCP) Server.
Standalone, zero-audio visual perception oracle for text-only LLMs (MiniMax 2.7, Claude Code, Playwright).
Operates over standard stdio JSON-RPC 2.0 on Windows and Linux.
"""

import os
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
import frames
import preprocessor
import vlm_client

SERVER_NAME = "desktop-vlm-lens"
SERVER_VERSION = "2.1.1"
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
            "Returns the element's bounding box and a click point (click_x, click_y) in pixels of the original "
            "image (primary_click; for a live window capture also primary_click_screen). The image is sent at 1024 px: "
            "smaller sizes make click targets unreliable. Supports optional crop_bbox zooming: coordinates are "
            "automatically remapped back to full-canvas coordinates."
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


# Image budgets: the cost of a call is linear in visual tokens, so each question gets the smallest
# image that answers it. Measured on 17 elements of 3 pages with DOM ground truth (Dell, stock Qwen2.5-VL-3B):
# click error median 33 px at 512 px (7/17 inside the element), 4.6 px at 768 px, 2.2 px at 1024 px; on the
# 1920 px page 768 px was 4x worse than 1024 px, so grounding stays at 1024 px. Reading showed no gain
# from 1024 px over 768 px, so transcription uses 768 px.
BUDGET_PX = {"scene": 512, "read": 768, "precise": 1024}
BUDGET_ENV = {"scene": "VLM_SCENE_PX", "read": "VLM_READ_PX", "precise": "VLM_PRECISE_PX"}
DEFAULT_DETAIL = {
    "capture_and_inspect": "precise",
    "inspect_image_file": "precise",
    "transcribe_screen_text": "read",
    "ground_ui_element": "precise",
}

FRAMES = frames.FrameStore()

_DETAIL_PROP = {
    "type": "string",
    "enum": ["scene", "read", "precise"],
    "description": ("Image size sent to the model: 'scene' 512 px (fast overview), 'read' 768 px (text), "
                    "'precise' 1024 px (small text, UI details). Cost grows with the pixel count."),
}
_FRAME_PROP = {
    "type": "string",
    "description": ("frame_id returned by an earlier capture. Reuses that exact image (no new screenshot), "
                    "so follow-up questions are much faster and refer to the same pixels. Kept in memory for a few minutes."),
}
for _tool in TOOLS:
    _props = _tool["parameters"]["properties"]
    if _tool["name"] in ("capture_and_inspect", "transcribe_screen_text", "inspect_image_file"):
        _props["detail"] = _DETAIL_PROP
    if _tool["name"] in ("capture_and_inspect", "transcribe_screen_text", "ground_ui_element"):
        _props["frame_id"] = _FRAME_PROP
    # MCP requires `inputSchema`; `parameters` is kept as an alias for harnesses that read the old key.
    _tool["inputSchema"] = _tool["parameters"]


def budget_px(detail: str) -> int:
    """Pixel budget for a detail level (env VLM_SCENE_PX / VLM_READ_PX / VLM_PRECISE_PX override)."""
    override = os.environ.get(BUDGET_ENV[detail])
    if override:
        try:
            return max(64, int(override))
        except ValueError:
            pass
    return BUDGET_PX[detail]


def resolve_image(args: Dict[str, Any]) -> tuple[Any, Dict[str, Any], Optional[str]]:
    """Resolves image from a remembered frame, live capture, clipboard, or disk file.

    Live captures are remembered in memory and get a frame_id in their metadata.
    """
    frame_id = args.get("frame_id")
    if frame_id:
        got = FRAMES.get(str(frame_id))
        if not got:
            return None, {}, (f"frame_id '{frame_id}' is unknown or expired (frames are kept in memory for "
                              f"{int(frames.TTL_SEC)} s); capture again.")
        img, meta = got
        meta["frame_id"] = frame_id
        return img, meta, None

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
    elif target == "fullscreen":
        img, meta = capture.capture_fullscreen()
    else:  # active_window
        img, meta = capture.capture_active_window()
    if img is not None:
        meta["frame_id"] = FRAMES.put(img, meta)
    return img, meta, None


def _error(text: str) -> Dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": True}


def _prepare(tool_name: str, args: Dict[str, Any], err_prefix: str = "Capture Error"):
    """Image for a tool call: (ctx, None) on success, (None, error_result) otherwise."""
    detail = args.get("detail") or DEFAULT_DETAIL[tool_name]
    if detail not in BUDGET_PX:
        return None, _error(f"Error: detail must be one of {sorted(BUDGET_PX)}, got '{detail}'.")
    img, cap_meta, err = resolve_image(args)
    if err:
        return None, _error(f"{err_prefix}: {err}")
    crop_param = args.get("crop_bbox") or args.get("roi_crop")
    img_processed, prep_meta = preprocessor.preprocess_image(img, max_dim=budget_px(detail), crop_bbox=crop_param)
    prep_meta["detail"] = detail
    return {
        "cap": cap_meta,
        "prep": prep_meta,
        "b64": preprocessor.encode_image_base64(img_processed),
    }, None


def _frame_line(cap_meta: Dict[str, Any]) -> str:
    fid = cap_meta.get("frame_id")
    return f"- **frame_id**: `{fid}` (pass it to reuse this exact capture)\n" if fid else ""


GROUND_PROMPT = ("Locate the '{desc}' in the image. "
                 "Return only the bounding box as [x1, y1, x2, y2] in pixel coordinates.")
GROUND_RETRY_PROMPT = ("Where is the '{desc}'? Answer with only the bounding box as four numbers in brackets, "
                       "[x1, y1, x2, y2], in pixel coordinates of the image. No other words.")
MOONDREAM_PROMPT = ("Point to the '{desc}' in the image. "
                    "Return ONLY the coordinate as [ymin, xmin, ymax, xmax] normalized to 1000.")


def _ground_element(ctx: Dict[str, Any], desc: str, family: str):
    """Queries the model for an element box. Returns (res, boxes, model_size, size_source).

    Qwen2.5-VL answers in absolute pixels of the resized image it saw; a prose answer gets one strict retry.
    """
    cap, prep = ctx["cap"], ctx["prep"]
    absolute = family != "moondream"
    prompt = (GROUND_PROMPT if absolute else MOONDREAM_PROMPT).format(desc=desc)
    res = vlm_client.query_vlm(ctx["b64"], prompt, max_tokens=100, temperature=0.1)
    if res.get("status") != "success":
        return res, [], None, None

    def parse(r):
        size, source = None, None
        if absolute:
            lo, hi = vlm_client.image_token_limits()
            size, source = preprocessor.resolve_model_size(
                prep["processed_width"], prep["processed_height"], r.get("prompt_tokens"), lo, hi)
        boxes = preprocessor.parse_grounding_coordinates(
            r.get("content", ""), cap.get("original_width", 1000), cap.get("original_height", 1000),
            crop_info=prep.get("crop_info"), model_size=size)
        return boxes, size, source

    boxes, size, source = parse(res)
    if absolute and not boxes:
        retry = vlm_client.query_vlm(ctx["b64"], GROUND_RETRY_PROMPT.format(desc=desc), max_tokens=100, temperature=0.0)
        if retry.get("status") == "success":
            res = retry
            boxes, size, source = parse(retry)
    return res, boxes, size, source


def _click_spaces(cap: Dict[str, Any], boxes: List[Dict[str, Any]]):
    """(space, primary_click, primary_click_screen): which coordinate space the click is in."""
    if not boxes:
        return None, None, None
    click = {"x": boxes[0]["click_x"], "y": boxes[0]["click_y"]}
    source = cap.get("source")
    if source == "active_window" and "left" in cap and "top" in cap:
        return "window", click, {"x": click["x"] + cap["left"], "y": click["y"] + cap["top"]}
    if source == "fullscreen":
        return "screen", click, dict(click)
    return "image", click, None


def handle_tool_call(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Executes requested tool and returns MCP content."""
    try:
        if tool_name == "capture_and_inspect":
            prompt = args.get("prompt", "").strip()
            if not prompt:
                return _error("Error: prompt parameter is required.")

            ctx, err = _prepare(tool_name, args)
            if err:
                return err
            cap_meta, prep_meta = ctx["cap"], ctx["prep"]

            res = vlm_client.query_vlm(ctx["b64"], prompt, max_tokens=args.get("max_tokens", 250))
            if res.get("status") != "success":
                return _error(f"VLM Error: {res.get('error')}")

            crop_note = ""
            if prep_meta.get("crop_applied"):
                ci = prep_meta.get("crop_info", {})
                crop_note = f"- **RoI Zoom Active**: {ci.get('crop_box')} (Evaluated at native unscaled pixel fidelity)\n"

            out_text = (
                f"### Desktop VLM Inspection\n"
                f"- **Source**: {cap_meta.get('source')} ({cap_meta.get('original_width')}x{cap_meta.get('original_height')})\n"
                f"{_frame_line(cap_meta)}"
                f"{crop_note}"
                f"- **Processed Canvas**: {prep_meta.get('processed_width')}x{prep_meta.get('processed_height')} ({prep_meta['detail']})\n"
                f"- **Response**:\n{res.get('content')}\n\n"
                f"- **Latency**: {res.get('duration_sec')}s | **Tokens**: {res.get('prompt_tokens')} in, {res.get('completion_tokens')} out"
            )
            return {"content": [{"type": "text", "text": out_text}], "isError": False}

        elif tool_name == "ground_ui_element":
            desc = args.get("element_description", "").strip()
            if not desc:
                return _error("Error: element_description is required.")

            family = vlm_client.detect_model_family()
            if family == "smolvlm":
                msg = (
                    "⚠️ Model Limitation Notice: SmolVLM (Tier 3) is active. "
                    "SmolVLM is designed for ultra-fast binary triage (~1.5s on CPU) and lacks the "
                    "parameter resolution to regress 2D bounding boxes for UI grounding.\n\n"
                    "Recommended action: Switch to Qwen2.5-VL-3B for accurate grounding:\n"
                    "  python scripts/fetch_models.py --model qwen2.5-vl-3b"
                )
                return {"content": [{"type": "text", "text": msg}], "isError": False}

            ctx, err = _prepare(tool_name, {**args, "detail": "precise"})
            if err:
                return err
            cap_meta, prep_meta = ctx["cap"], ctx["prep"]

            res, boxes, model_size, size_source = _ground_element(ctx, desc, family)
            if res.get("status") != "success":
                return _error(f"VLM Error: {res.get('error')}")

            space, primary, primary_screen = _click_spaces(cap_meta, boxes)
            crop_info = prep_meta.get("crop_info")
            ref_w = crop_info["crop_width"] if prep_meta.get("crop_applied") and crop_info else cap_meta.get("original_width")
            scale = round(ref_w / model_size[0], 3) if model_size and ref_w else None

            result_data = {
                "element_description": desc,
                "source": cap_meta.get("source"),
                "frame_id": cap_meta.get("frame_id"),
                "canvas_size": [cap_meta.get("original_width"), cap_meta.get("original_height")],
                "image_sent": [prep_meta.get("processed_width"), prep_meta.get("processed_height")],
                "model_input_size": list(model_size) if model_size else None,
                "size_source": size_source,
                "canvas_px_per_model_px": scale,
                "roi_crop_applied": prep_meta.get("crop_applied", False),
                "crop_info": crop_info,
                "matches": boxes,
                "click_space": space,
                "primary_click": primary,
                "primary_click_screen": primary_screen,
                "raw_model_output": res.get("content"),
                "model_family": family
            }
            if scale and scale > 1.5:
                result_data["scale_hint"] = (
                    f"The model saw this canvas at 1/{scale} scale; small targets (icons under about {int(24 * scale)} px) "
                    f"are unreliable. Pass crop_bbox around the area to ground it at native fidelity.")

            out_text = (
                f"### UI Element Grounding ({family.upper()})\n"
                f"```json\n{json.dumps(result_data, indent=2)}\n```\n\n"
            )
            if boxes:
                p = boxes[0]
                remap_note = " (remapped from RoI crop to full canvas)" if p.get("remapped_from_crop") else ""
                out_text += f"**Automation Action**: Click target at pixel coordinates `(x={p['click_x']}, y={p['click_y']})`{remap_note}, in {space} coordinates"
                if primary_screen and space == "window":
                    out_text += f"; on screen `(x={primary_screen['x']}, y={primary_screen['y']})`"
                out_text += "."
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

            ctx, err = _prepare(tool_name, args)
            if err:
                return err
            cap_meta, prep_meta = ctx["cap"], ctx["prep"]

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

            res = vlm_client.query_vlm(ctx["b64"], prompt, max_tokens=400, temperature=0.1)
            if res.get("status") != "success":
                return _error(f"VLM Error: {res.get('error')}")

            crop_note = ""
            if prep_meta.get("crop_applied"):
                ci = prep_meta.get("crop_info", {})
                crop_note = f"- **RoI Zoom Active**: {ci.get('crop_box')} (Native Optical OCR)\n"

            out_text = (
                f"### Screen Text Transcription (OCR)\n"
                f"- **Source**: {cap_meta.get('source')} ({cap_meta.get('original_width')}x{cap_meta.get('original_height')})\n"
                f"{_frame_line(cap_meta)}"
                f"{crop_note}"
                f"- **Transcribed Content**:\n{res.get('content')}\n\n"
                f"- **Latency**: {res.get('duration_sec')}s"
            )
            return {"content": [{"type": "text", "text": out_text}], "isError": False}

        elif tool_name == "inspect_image_file":
            img_path = args.get("image_path", "").strip()
            prompt = args.get("prompt", "").strip()
            if not img_path or not prompt:
                return _error("Error: Both image_path and prompt are required.")

            ctx, err = _prepare(tool_name, {**args, "image_path": img_path, "frame_id": None}, err_prefix="File Error")
            if err:
                return err
            cap_meta, prep_meta = ctx["cap"], ctx["prep"]

            res = vlm_client.query_vlm(ctx["b64"], prompt, max_tokens=args.get("max_tokens", 250))
            if res.get("status") != "success":
                return _error(f"VLM Error: {res.get('error')}")

            crop_note = ""
            if prep_meta.get("crop_applied"):
                ci = prep_meta.get("crop_info", {})
                crop_note = f"- **RoI Zoom Active**: {ci.get('crop_box')}\n"

            out_text = (
                f"### Image File Inspection\n"
                f"- **File**: `{img_path}`\n"
                f"- **Dimensions**: {cap_meta.get('original_width')}x{cap_meta.get('original_height')} -> {prep_meta.get('processed_width')}x{prep_meta.get('processed_height')} ({prep_meta['detail']})\n"
                f"{crop_note}"
                f"- **Response**:\n{res.get('content')}\n\n"
                f"- **Latency**: {res.get('duration_sec')}s"
            )
            return {"content": [{"type": "text", "text": out_text}], "isError": False}

        else:
            return _error(f"Unknown tool: {tool_name}")

    except Exception as e:
        log_debug(f"Unhandled exception in tool {tool_name}: {e}")
        return _error(f"Server exception: {str(e)}")


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
        res = vlm_client.resolve_server()
        print("Backend:", res.url or f"none ({res.error})")
        print("Tools available: capture_and_inspect, ground_ui_element, transcribe_screen_text, inspect_image_file")
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] in ("--ping", "ping", "test"):
        res = vlm_client.resolve_server()
        print(f"VLM Server URL: {res.url or 'none'} ({res.source or 'unavailable'})")
        if res.error:
            print(res.error)
        print("Port 8085 active:", vlm_client.is_port_open(port=8085))
        sys.exit(0 if res.url else 1)
    try:
        run_stdio_server()
    except BrokenPipeError:
        pass            # the client went away; exit quietly
