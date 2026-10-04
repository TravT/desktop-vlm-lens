"""
Cross-platform desktop screen, active window, and clipboard capture module.
Supports Windows 10/11 and Linux (X11 & Wayland) with zero heavyweight dependencies.
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from PIL import Image, ImageGrab, ImageOps


def capture_fullscreen() -> Tuple[Image.Image, Dict[str, Any]]:
    """Captures the primary monitor/desktop screen."""
    img = ImageGrab.grab()
    meta = {
        "source": "fullscreen",
        "original_width": img.width,
        "original_height": img.height
    }
    return img.convert("RGB"), meta


def capture_active_window() -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Captures only the currently focused/active foreground window.
    Prevents multi-monitor pixel sprawl (e.g. 5120x1440) from squashing UI elements.
    """
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            if hwnd:
                rect = wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                left, top, right, bottom = rect.left, rect.top, rect.right, rect.bottom
                width = right - left
                height = bottom - top

                # Get window title
                length = user32.GetWindowTextLengthW(hwnd)
                buff = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buff, length + 1)
                title = buff.value or "Active Window"

                if width > 10 and height > 10:
                    img = ImageGrab.grab(bbox=(left, top, right, bottom))
                    meta = {
                        "source": "active_window",
                        "title": title,
                        "left": left,
                        "top": top,
                        "original_width": img.width,
                        "original_height": img.height
                    }
                    return img.convert("RGB"), meta
        except Exception:
            pass

    elif sys.platform.startswith("linux"):
        # Attempt X11 active window detection via xdotool / xwininfo
        try:
            out = subprocess.check_output(["xdotool", "getactivewindow"], text=True).strip()
            win_id = int(out)
            geo = subprocess.check_output(["xwininfo", "-id", str(win_id)], text=True)
            
            x, y, w, h = None, None, None, None
            for line in geo.splitlines():
                line = line.strip()
                if line.startswith("Absolute upper-left X:"):
                    x = int(line.split(":")[1])
                elif line.startswith("Absolute upper-left Y:"):
                    y = int(line.split(":")[1])
                elif line.startswith("Width:"):
                    w = int(line.split(":")[1])
                elif line.startswith("Height:"):
                    h = int(line.split(":")[1])

            if x is not None and y is not None and w and h:
                img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
                meta = {
                    "source": "active_window",
                    "title": f"Window_{win_id}",
                    "left": x,
                    "top": y,
                    "original_width": img.width,
                    "original_height": img.height
                }
                return img.convert("RGB"), meta
        except Exception:
            pass

    # Fallback to full screen if active window lookup fails
    return capture_fullscreen()


def capture_clipboard() -> Tuple[Optional[Image.Image], Dict[str, Any]]:
    """
    Grabs an image from the system clipboard (e.g. snipped via Win+Shift+S or PrintScreen).
    """
    try:
        data = ImageGrab.grabclipboard()
        if isinstance(data, Image.Image):
            meta = {
                "source": "clipboard",
                "original_width": data.width,
                "original_height": data.height
            }
            return data.convert("RGB"), meta
        elif isinstance(data, list) and data and Path(data[0]).is_file():
            # Sometimes clipboard contains a file path to an image
            img = Image.open(data[0])
            img = ImageOps.exif_transpose(img).convert("RGB")
            meta = {
                "source": "clipboard_file",
                "path": str(data[0]),
                "original_width": img.width,
                "original_height": img.height
            }
            return img, meta
    except Exception as e:
        return None, {"error": f"Failed reading clipboard: {e}"}

    return None, {
        "error": "Clipboard does not contain an image. Use Win+Shift+S (Windows) or PrintScreen (Linux) to snip an image first."
    }


def load_image_file(path_str: str) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
    """Loads an image from local disk (e.g. Playwright output)."""
    p = Path(path_str).resolve()
    if not p.exists() or not p.is_file():
        return None, {"error": f"File '{path_str}' does not exist."}

    try:
        img = Image.open(p)
        img = ImageOps.exif_transpose(img).convert("RGB")
        meta = {
            "source": "file",
            "path": str(p),
            "original_width": img.width,
            "original_height": img.height
        }
        return img, meta
    except Exception as e:
        return None, {"error": f"Error opening image '{path_str}': {e}"}
