"""
Image preprocessing, aspect-ratio scaling, RoI zooming, and coordinate denormalization module.
Optimized for 3B/5B/7B Vision-Language Models on CPU and iGPU architectures.
Supports Region-of-Interest (RoI) crop-on-demand with automated coordinate re-mapping (ADR-43).
"""

import io
import re
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


def parse_grounding_coordinates(
    model_output: str,
    orig_width: int,
    orig_height: int,
    crop_info: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Extracts 2D bounding boxes in format [ymin, xmin, ymax, xmax] from model output
    and calculates exact pixel click coordinates for UI automation.
    If crop_info is provided, re-maps local crop coordinates back to full canvas space.
    """
    matches = re.findall(r"[\[\(]\s*([0-9\.]+)\s*,\s*([0-9\.]+)\s*,\s*([0-9\.]+)\s*,\s*([0-9\.]+)\s*[\]\)]", model_output)
    results = []

    is_cropped = bool(crop_info and crop_info.get("applied"))
    if is_cropped:
        ref_w = crop_info["crop_width"]
        ref_h = crop_info["crop_height"]
        offset_x = crop_info["x_offset"]
        offset_y = crop_info["y_offset"]
    else:
        ref_w = orig_width
        ref_h = orig_height
        offset_x = 0
        offset_y = 0

    for m in matches:
        try:
            ymin, xmin, ymax, xmax = map(float, m)
        except ValueError:
            continue

        max_val = max(ymin, xmin, ymax, xmax)
        if max_val <= 1.0:
            # 0.0 - 1.0 normalized
            local_ymin = int(round(ymin * ref_h))
            local_xmin = int(round(xmin * ref_w))
            local_ymax = int(round(ymax * ref_h))
            local_xmax = int(round(xmax * ref_w))
        elif max_val <= 1000.0:
            # 0 - 1000 normalized (Qwen-VL default)
            local_ymin = int(round(ymin * ref_h / 1000.0))
            local_xmin = int(round(xmin * ref_w / 1000.0))
            local_ymax = int(round(ymax * ref_h / 1000.0))
            local_xmax = int(round(xmax * ref_w / 1000.0))
        else:
            # Absolute pixel coordinates
            local_ymin = int(round(ymin))
            local_xmin = int(round(xmin))
            local_ymax = int(round(ymax))
            local_xmax = int(round(xmax))

        # Remap to full canvas coordinate space
        px_ymin = offset_y + local_ymin
        px_xmin = offset_x + local_xmin
        px_ymax = offset_y + local_ymax
        px_xmax = offset_x + local_xmax

        # Exact center click point
        click_x = int(round((px_xmin + px_xmax) / 2.0))
        click_y = int(round((px_ymin + px_ymax) / 2.0))

        item = {
            "normalized_box": [ymin, xmin, ymax, xmax],
            "pixel_box": [px_ymin, px_xmin, px_ymax, px_xmax],
            "click_x": click_x,
            "click_y": click_y
        }
        if is_cropped:
            item["crop_relative_box"] = [local_ymin, local_xmin, local_ymax, local_xmax]
            item["remapped_from_crop"] = True

        results.append(item)

    return results
