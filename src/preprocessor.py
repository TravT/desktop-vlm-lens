"""
Image preprocessing, aspect-ratio scaling, RoI zooming, and coordinate denormalization module.
Optimized for 3B/5B/7B Vision-Language Models on CPU and iGPU architectures.
Supports Region-of-Interest (RoI) crop-on-demand with automated coordinate re-mapping (ADR-43).
"""

import io
import re
import math
import base64
from typing import Tuple, Optional, Dict, Any, List
from PIL import Image


def crop_region_of_interest(
    img: Image.Image,
    bbox: List[int]
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Extracts a sub-region of interest from a PIL Image with 100% native unscaled optical fidelity.
    
    Args:
        img: Input PIL Image.
        bbox: [ymin, xmin, ymax, xmax] bounding box:
              - Normalized 0.0 - 1.0 (float)
              - Normalized 0 - 1000 (integer, Qwen-VL default)
              - Absolute pixel coordinates
              
    Returns:
        Tuple of (cropped_image, crop_metadata)
    """
    orig_w, orig_h = img.size
    if not bbox or len(bbox) != 4:
        return img, {
            "applied": False,
            "crop_box": (0, 0, orig_w, orig_h),
            "x_offset": 0,
            "y_offset": 0,
            "crop_width": orig_w,
            "crop_height": orig_h,
            "pre_crop_size": [orig_w, orig_h]
        }

    ymin, xmin, ymax, xmax = bbox
    max_val = max(abs(v) for v in bbox)

    if max_val <= 1.0:
        # 0.0 - 1.0 normalized float
        box = (
            int(round(xmin * orig_w)),
            int(round(ymin * orig_h)),
            int(round(xmax * orig_w)),
            int(round(ymax * orig_h))
        )
    elif max_val <= 1000 and (orig_w > 1000 or orig_h > 1000 or all(isinstance(v, int) for v in bbox)):
        # 0 - 1000 normalized integer (Qwen coordinate space)
        box = (
            int(round(xmin * orig_w / 1000.0)),
            int(round(ymin * orig_h / 1000.0)),
            int(round(xmax * orig_w / 1000.0)),
            int(round(ymax * orig_h / 1000.0))
        )
    else:
        # Absolute pixel coordinates
        box = (int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax)))

    # Clamp coordinates to ensure valid box within image boundaries
    clamped_box = (
        max(0, min(orig_w - 1, box[0])),
        max(0, min(orig_h - 1, box[1])),
        max(1, min(orig_w, box[2])),
        max(1, min(orig_h, box[3]))
    )

    # Ensure box has positive area
    if clamped_box[2] <= clamped_box[0] or clamped_box[3] <= clamped_box[1]:
        return img, {
            "applied": False,
            "crop_box": (0, 0, orig_w, orig_h),
            "x_offset": 0,
            "y_offset": 0,
            "crop_width": orig_w,
            "crop_height": orig_h,
            "pre_crop_size": [orig_w, orig_h]
        }

    cropped_img = img.crop(clamped_box)
    crop_w, crop_h = cropped_img.size

    meta = {
        "applied": True,
        "crop_box": clamped_box,
        "x_offset": clamped_box[0],
        "y_offset": clamped_box[1],
        "crop_width": crop_w,
        "crop_height": crop_h,
        "pre_crop_size": [orig_w, orig_h],
        "requested_bbox": bbox
    }
    return cropped_img, meta


def preprocess_image(
    img: Image.Image,
    max_dim: int = 1024,
    crop_bbox: Optional[List[int]] = None,
    roi_crop: Optional[List[int]] = None
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Applies optional RoI crop, preserves aspect ratio, and resamples using Lanczos.
    
    Args:
        img: Input PIL Image.
        max_dim: Maximum dimension for the longer edge (default 1024px).
        crop_bbox: Optional [ymin, xmin, ymax, xmax] RoI crop box.
        roi_crop: Alias for crop_bbox for backwards compatibility.
    """
    orig_w, orig_h = img.size
    target_crop = crop_bbox if crop_bbox is not None else roi_crop
    crop_info = None

    if target_crop and len(target_crop) == 4:
        img, crop_info = crop_region_of_interest(img, target_crop)

    cur_w, cur_h = img.size

    # Calculate aspect-ratio preserving downsampling
    if max(cur_w, cur_h) > max_dim:
        if cur_w >= cur_h:
            new_w = max_dim
            new_h = max(1, int(cur_h * (max_dim / cur_w)))
        else:
            new_h = max_dim
            new_w = max(1, int(cur_w * (max_dim / cur_h)))

        # Use high-fidelity Lanczos resampling to preserve font contrast
        resample_filter = getattr(Image.Resampling, "LANCZOS", Image.LANCZOS)
        img_scaled = img.resize((new_w, new_h), resample=resample_filter)
    else:
        img_scaled = img
        new_w, new_h = cur_w, cur_h

    meta = {
        "original_width": orig_w,
        "original_height": orig_h,
        "processed_width": new_w,
        "processed_height": new_h,
        "scale_x": new_w / float(orig_w),
        "scale_y": new_h / float(orig_h),
        "crop_applied": crop_info is not None and crop_info.get("applied", False)
    }
    if crop_info and crop_info.get("applied"):
        meta["crop_info"] = crop_info

    return img_scaled, meta


def encode_image_base64(img: Image.Image, format_str: str = "JPEG", quality: int = 90) -> str:
    """Encodes PIL Image to Base64 data URI string."""
    buf = io.BytesIO()
    if format_str.upper() in ("JPG", "JPEG"):
        img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG", optimize=True)
        mime = "image/png"

    raw_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{raw_b64}"


QWEN_FACTOR = 28            # 14 px patches, merged 2x2
QWEN_TEMPLATE_TOKENS = 31   # chat-template tokens around the image grid (measured: tokens = grid + 31)
TOKEN_TOLERANCE = 64        # slack for the prompt text when comparing expected and reported prompt tokens


def qwen_input_size(
    width: int,
    height: int,
    min_tokens: int = 256,
    max_tokens: Optional[int] = None,
    factor: int = QWEN_FACTOR,
) -> Tuple[int, int]:
    """Size (w, h) of an image as Qwen2.5-VL sees it after llama.cpp's resize.

    Both sides are rounded to multiples of 28; images under the --image-min-tokens floor are scaled up and,
    when --image-max-tokens is set, images over the cap are scaled down. Qwen2.5-VL answers grounding
    queries in absolute pixels of THIS image, not of the file that was sent.
    """
    w_bar = max(factor, int(math.floor(width / factor + 0.5)) * factor)
    h_bar = max(factor, int(math.floor(height / factor + 0.5)) * factor)
    min_pixels = min_tokens * factor * factor
    if max_tokens and w_bar * h_bar > max_tokens * factor * factor:
        beta = math.sqrt(width * height / float(max_tokens * factor * factor))
        w_bar = max(factor, int(math.floor(width / beta / factor)) * factor)
        h_bar = max(factor, int(math.floor(height / beta / factor)) * factor)
    elif w_bar * h_bar < min_pixels:
        beta = math.sqrt(min_pixels / float(width * height))
        w_bar = int(math.ceil(width * beta / factor)) * factor
        h_bar = int(math.ceil(height * beta / factor)) * factor
    return w_bar, h_bar


def expected_prompt_tokens(model_size: Tuple[int, int], factor: int = QWEN_FACTOR) -> int:
    """Image grid tokens plus the chat template, for a prompt of a few words."""
    return (model_size[0] // factor) * (model_size[1] // factor) + QWEN_TEMPLATE_TOKENS


def infer_model_size(prompt_tokens: int, width: int, height: int, factor: int = QWEN_FACTOR) -> Tuple[int, int]:
    """Model-seen size recovered from the server's prompt token count and the sent aspect ratio."""
    grid = max(1, prompt_tokens - QWEN_TEMPLATE_TOKENS)
    area = grid * factor * factor
    ratio = width / float(height)
    w_bar = max(factor, int(round(math.sqrt(area * ratio) / factor)) * factor)
    h_bar = max(factor, int(round(math.sqrt(area / ratio) / factor)) * factor)
    return w_bar, h_bar


def resolve_model_size(
    width: int,
    height: int,
    prompt_tokens: Optional[int] = None,
    min_tokens: int = 256,
    max_tokens: Optional[int] = None,
) -> Tuple[Tuple[int, int], str]:
    """The size the model saw, and where it came from: 'computed' or 'inferred'.

    The computed size is trusted when the server's prompt token count agrees with it. A disagreement
    means the server runs with different image-token limits than we assume (for example a lower
    --image-max-tokens), so the size is recovered from the token count instead.
    """
    size = qwen_input_size(width, height, min_tokens, max_tokens)
    if prompt_tokens and abs(prompt_tokens - expected_prompt_tokens(size)) > TOKEN_TOLERANCE:
        return infer_model_size(prompt_tokens, width, height), "inferred"
    return size, "computed"


_BOX_RE = re.compile(r"[\[\(]\s*([0-9\.]+)\s*,\s*([0-9\.]+)\s*,\s*([0-9\.]+)\s*,\s*([0-9\.]+)\s*[\]\)]")


def parse_grounding_coordinates(
    model_output: str,
    orig_width: int,
    orig_height: int,
    crop_info: Optional[Dict[str, Any]] = None,
    model_size: Optional[Tuple[int, int]] = None,
) -> List[Dict[str, Any]]:
    """
    Extracts 2D boxes from model output and computes pixel click coordinates on the full canvas.

    With `model_size` (the image size the model saw, see resolve_model_size) the four numbers are
    Qwen2.5-VL's native absolute pixels [x1, y1, x2, y2]. Without it they are read as normalized
    0-1000 or 0-1 [ymin, xmin, ymax, xmax] (other model families, older callers).
    If crop_info is provided, local crop coordinates are re-mapped to the full canvas.
    Result fields: pixel_box [ymin, xmin, ymax, xmax], box_xyxy_pixels [x1, y1, x2, y2],
    click_x, click_y on the canvas (the original capture, not the downscaled image).
    """
    results = []

    is_cropped = bool(crop_info and crop_info.get("applied"))
    if is_cropped:
        ref_w, ref_h = crop_info["crop_width"], crop_info["crop_height"]
        offset_x, offset_y = crop_info["x_offset"], crop_info["y_offset"]
    else:
        ref_w, ref_h = orig_width, orig_height
        offset_x = offset_y = 0

    for m in _BOX_RE.findall(model_output or ""):
        try:
            a, b, c, d = map(float, m)
        except ValueError:
            continue

        if model_size:
            mw, mh = model_size
            x1, x2 = sorted((max(0.0, min(float(mw), a)), max(0.0, min(float(mw), c))))
            y1, y2 = sorted((max(0.0, min(float(mh), b)), max(0.0, min(float(mh), d))))
            local_xmin = int(round(x1 / mw * ref_w))
            local_xmax = int(round(x2 / mw * ref_w))
            local_ymin = int(round(y1 / mh * ref_h))
            local_ymax = int(round(y2 / mh * ref_h))
            ymin, xmin, ymax, xmax = (round(y1 / mh * 1000), round(x1 / mw * 1000),
                                      round(y2 / mh * 1000), round(x2 / mw * 1000))
        else:
            ymin, xmin, ymax, xmax = a, b, c, d
            max_val = max(ymin, xmin, ymax, xmax)
            if max_val <= 1.0:
                scale_y, scale_x = ref_h, ref_w                          # 0.0 - 1.0 normalized
            elif max_val <= 1000.0:
                scale_y, scale_x = ref_h / 1000.0, ref_w / 1000.0        # 0 - 1000 normalized
            else:
                scale_y = scale_x = 1.0                                  # absolute pixels
            local_ymin, local_xmin = int(round(ymin * scale_y)), int(round(xmin * scale_x))
            local_ymax, local_xmax = int(round(ymax * scale_y)), int(round(xmax * scale_x))

        px_ymin, px_xmin = offset_y + local_ymin, offset_x + local_xmin
        px_ymax, px_xmax = offset_y + local_ymax, offset_x + local_xmax

        item = {
            "normalized_box": [ymin, xmin, ymax, xmax],
            "pixel_box": [px_ymin, px_xmin, px_ymax, px_xmax],
            "box_xyxy_pixels": [px_xmin, px_ymin, px_xmax, px_ymax],
            "click_x": int(round((px_xmin + px_xmax) / 2.0)),
            "click_y": int(round((px_ymin + px_ymax) / 2.0)),
        }
        if is_cropped:
            item["crop_relative_box"] = [local_ymin, local_xmin, local_ymax, local_xmax]
            item["remapped_from_crop"] = True
        results.append(item)

    return results
