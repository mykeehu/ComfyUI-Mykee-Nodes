"""
ComfyUI-Mykee-Nodes / Mykee Width/Height Resizer

Scales an incoming width / height pair up or down and returns the new pair.
The aspect ratio is kept (both sides get the same factor); only the optional
rounding to a multiple of 8 / 16 / 32 / 64 can shift it by a few pixels.

- image (optional IMAGE input): when connected, width / height are read from
  the image and the typed width / height are ignored (the frontend greys them
  out). Only the size of the image is used.
- mode:
    scale by  both sides are multiplied by scale_by (decimals allowed, 0.5 =
              half size, 1.5 = one and a half times).
    scale to  the reference side is set to scale_to pixels and the other side
              follows (e.g. longest side 1024 -> a 1920x1080 input gives
              1024x576).
- reference (what scale_to measures; scale_by ignores it): longest side
  (default), shortest side, width or height.
- divisible_by: Off / 8 / 16 / 32 / 64 - the result is rounded to the nearest
  multiple (half up), never below the divisor itself.

The node is an output node, so it also reports the result to the frontend
(web/mykee_wh_resizer.js), which shows it in a small box at the bottom of the
node. The box updates live from the typed numbers; when width / height come
through a connection their value is not known in the browser, so the box shows
the result of the last run instead.
"""

import math

MIN_SIZE = 1
MAX_SIZE = 65536  # sanity cap; a multiple of 16 / 32 / 64
MODES = ["scale by", "scale to"]
REFERENCES = ["longest side", "shortest side", "width", "height"]
DIVISORS = ["Off", "8", "16", "32", "64"]


def _divisor(divisible_by):
    try:
        return int(divisible_by)
    except (TypeError, ValueError):
        return 0  # "Off"


def _snap(value, divisor):
    """Round to the nearest multiple of divisor (half up); whole numbers only."""
    value = round(float(value), 6)  # swallow float noise like 511.99999999
    lowest = max(MIN_SIZE, divisor)
    if divisor:
        result = int(math.floor(value / divisor + 0.5)) * divisor
    else:
        result = int(math.floor(value + 0.5))
    return max(lowest, min(MAX_SIZE, result))


def _reference_value(width, height, reference):
    if reference == "width":
        return width
    if reference == "height":
        return height
    if reference == "shortest side":
        return min(width, height)
    return max(width, height)  # "longest side" and anything unknown


def _image_size(image):
    """(width, height) of an IMAGE tensor [B, H, W, C], or None."""
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) < 3:
        return None
    return int(shape[2]), int(shape[1])


def compute_size(width, height, mode, reference, scale_by, scale_to, divisible_by):
    """The final (width, height)."""
    width = max(MIN_SIZE, int(width))
    height = max(MIN_SIZE, int(height))
    if mode == "scale to":
        base = _reference_value(width, height, reference)
        # multiply first, divide last: fewer float rounding surprises
        new_w = width * float(scale_to) / base
        new_h = height * float(scale_to) / base
    else:
        new_w = width * float(scale_by)
        new_h = height * float(scale_by)
    divisor = _divisor(divisible_by)
    return _snap(new_w, divisor), _snap(new_h, divisor)


class MykeeWidthHeightResizer:
    CATEGORY = "Mykee/Image"
    FUNCTION = "run"
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    OUTPUT_TOOLTIPS = ("The scaled width.", "The scaled height.")
    # OUTPUT_NODE + the "ui" result below fill the preview box drawn by
    # web/mykee_wh_resizer.js (client-side only, never a node input).
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": MIN_SIZE,
                        "max": MAX_SIZE,
                        "step": 1,
                        "tooltip": "Incoming width. Type it, or connect it from another node.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": MIN_SIZE,
                        "max": MAX_SIZE,
                        "step": 1,
                        "tooltip": "Incoming height. Type it, or connect it from another node.",
                    },
                ),
                "mode": (
                    MODES,
                    {
                        "default": "scale to",
                        "tooltip": "scale by: multiply both sides by scale_by. "
                        "scale to: set the reference side to scale_to pixels, the other side follows.",
                    },
                ),
                "reference": (
                    REFERENCES,
                    {
                        "default": "longest side",
                        "tooltip": "What scale_to measures: the longest side, the shortest side, "
                        "the width or the height. (scale by does not use it.)",
                    },
                ),
                "scale_by": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 64.0,
                        "step": 0.05,
                        "round": 0.001,
                        "tooltip": "Multiplier for both sides (scale by mode), e.g. 0.5 or 1.5.",
                    },
                ),
                "scale_to": (
                    "INT",
                    {
                        "default": 1024,
                        "min": MIN_SIZE,
                        "max": 16384,
                        "step": 1,
                        "tooltip": "Target size in pixels of the reference side (scale to mode).",
                    },
                ),
                "divisible_by": (
                    DIVISORS,
                    {
                        "default": "Off",
                        "tooltip": "Round the result to the nearest multiple of this. "
                        "The aspect ratio can shift by a few pixels because of it.",
                    },
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Optional. When connected, the width / height are read from this image "
                        "(the width / height fields are then ignored). Only its size is used.",
                    },
                ),
            },
        }

    def run(self, width, height, mode, reference, scale_by, scale_to, divisible_by, image=None):
        image_size = _image_size(image)
        if image_size is not None:
            width, height = image_size
        new_w, new_h = compute_size(width, height, mode, reference, scale_by, scale_to, divisible_by)
        print(f"[Mykee Width/Height Resizer] {int(width)}x{int(height)} -> {new_w}x{new_h} ({mode})")
        return {"ui": {"size_preview": [f"{new_w} × {new_h}"]}, "result": (new_w, new_h)}


NODE_CLASS_MAPPINGS = {
    "MykeeWidthHeightResizer": MykeeWidthHeightResizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeWidthHeightResizer": "Mykee Width/Height Resizer",
}
