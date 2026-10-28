"""
ComfyUI-Mykee-Nodes / Mykee JSON BBOX Resizer

Re-maps the bounding boxes of a MOSS-Image layout JSON (the output of
"Mykee JSON Captioner", abs_centralized / yxyx integers) from the canvas it
was made for (original width/height) to another canvas (target width/height).
Nothing but the "bbox" of each item changes; captions, text and the rest of
the JSON are kept as they are.

Modes (what happens when the aspect ratio differs):
  stretch  x and y are scaled separately; the layout fills the whole target
           canvas (proportions of the boxes change with the canvas).
  contain  one uniform scale, the whole layout fits inside the target canvas
           and is centred (empty bands left/right or top/bottom).
  cover    one uniform scale, the target canvas is filled and the layout is
           centred and cropped; boxes are clamped to the canvas and items
           that end up completely outside are dropped.

The arithmetic mirrors the captioner / MOSS compile_prompt.py: centred
integers -> top-left pixels (add width // 2, height // 2), scale, enclose
fractional edges (floor min, ceil max), centre again on the target canvas.
"""
import json
import math
import re

_MODES = ["stretch", "contain", "cover"]


def _parse_layout(text):
    """JSON text (code fences and surrounding chatter tolerated) -> dict."""
    if isinstance(text, (list, tuple)):  # a list output wired straight in
        text = text[0] if text else ""
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", (text or "").strip()).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("Mykee JSON BBOX Resizer: no JSON object found in json_text")
    try:
        data, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError as e:
        raise ValueError(f"Mykee JSON BBOX Resizer: invalid JSON ({e})")
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise ValueError('Mykee JSON BBOX Resizer: the JSON needs an "items" list')
    space, fmt = data.get("bbox_coord_space"), data.get("bbox_format")
    if (space is not None and space != "abs_centralized") or (fmt is not None and fmt != "yxyx"):
        raise ValueError("Mykee JSON BBOX Resizer: only abs_centralized / yxyx boxes are supported "
                         f"(got {space!r} / {fmt!r})")
    return data


def _transform(ow, oh, tw, th, mode):
    """(sx, sy, off_x, off_y): target_px = original_px * scale + offset."""
    sx, sy = tw / ow, th / oh
    if mode == "stretch":
        return sx, sy, 0.0, 0.0
    s = min(sx, sy) if mode == "contain" else max(sx, sy)
    return s, s, (tw - ow * s) / 2.0, (th - oh * s) / 2.0


def resize_layout(data, ow, oh, tw, th, mode="stretch"):
    """Return a copy of `data` with every item's bbox re-mapped. Items whose
    box is missing or invalid keep what they have; items that end up empty /
    outside the target canvas are dropped (reported in the console)."""
    sx, sy, ox, oy = _transform(ow, oh, tw, th, mode)
    out = dict(data)
    items = []
    for n, item in enumerate(data["items"]):
        box = item.get("bbox") if isinstance(item, dict) else None
        if not (isinstance(box, list) and len(box) == 4
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box)):
            items.append(item)
            continue
        y0, x0, y1, x1 = box
        # centred integers -> top-left pixels of the original canvas -> target pixels
        px0, px1 = (x0 + ow // 2) * sx + ox, (x1 + ow // 2) * sx + ox
        py0, py1 = (y0 + oh // 2) * sy + oy, (y1 + oh // 2) * sy + oy
        # enclose fractional edges, stay inside the canvas
        nx0, nx1 = max(0, math.floor(round(px0, 9))), min(tw, math.ceil(round(px1, 9)))
        ny0, ny1 = max(0, math.floor(round(py0, 9))), min(th, math.ceil(round(py1, 9)))
        if nx0 >= nx1 or ny0 >= ny1:
            print(f"Mykee JSON BBOX Resizer: item {n} dropped - outside the target canvas")
            continue
        new_item = dict(item)
        new_item["bbox"] = [ny0 - th // 2, nx0 - tw // 2, ny1 - th // 2, nx1 - tw // 2]
        items.append(new_item)
    out["items"] = items
    return out


class MykeeJsonBboxResizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_text": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "MOSS layout JSON (e.g. the json output of "
                               "Mykee JSON Captioner): abs_centralized / "
                               "yxyx boxes made for the original size.",
                }),
                "original_width": ("INT", {
                    "default": 1024, "min": 1, "max": 16384, "step": 1,
                    "tooltip": "Width of the canvas the boxes were made for "
                               "(the captioner's width output).",
                }),
                "original_height": ("INT", {
                    "default": 1024, "min": 1, "max": 16384, "step": 1,
                    "tooltip": "Height of the canvas the boxes were made for "
                               "(the captioner's height output).",
                }),
                "target_width": ("INT", {
                    "default": 1024, "min": 16, "max": 16384, "step": 16,
                    "tooltip": "Width of the image to generate (MOSS-Image "
                               "needs multiples of 16).",
                }),
                "target_height": ("INT", {
                    "default": 1024, "min": 16, "max": 16384, "step": 16,
                    "tooltip": "Height of the image to generate (MOSS-Image "
                               "needs multiples of 16).",
                }),
                "mode": (_MODES, {
                    "default": "stretch",
                    "tooltip": "Only matters when the aspect ratio differs. "
                               "stretch: x and y scaled separately, fills "
                               "the target. contain: uniform scale, the "
                               "whole layout fits, centred. cover: uniform "
                               "scale, fills the target, centred and "
                               "cropped (outside boxes are clamped or "
                               "dropped).",
                }),
                "pretty_json": ("BOOLEAN", {
                    "default": True, "label_on": "indented", "label_off": "one line",
                    "tooltip": "Format the output indented instead of as one "
                               "compact line.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("json", "target_width", "target_height")
    FUNCTION = "run"
    CATEGORY = "Mykee/Prompt"
    # OUTPUT_NODE + the "ui" result below fill the read-only json_display box
    # drawn by web/mykee_json_bbox_resizer.js (a client-side widget, not a
    # declared input, so it never re-triggers the node).
    OUTPUT_NODE = True

    def run(self, json_text, original_width, original_height, target_width, target_height,
            mode, pretty_json):
        ow, oh, tw, th = int(original_width), int(original_height), int(target_width), int(target_height)
        if min(ow, oh, tw, th) < 1:
            raise ValueError("Mykee JSON BBOX Resizer: all sizes must be at least 1")
        if tw % 16 or th % 16:
            print(f"Mykee JSON BBOX Resizer: warning - target {tw}x{th} is not a multiple of 16 "
                  "(MOSS-Image needs that)")
        data = _parse_layout(json_text)
        result = resize_layout(data, ow, oh, tw, th, mode)
        text = json.dumps(result, ensure_ascii=False, indent=2 if pretty_json else None,
                          separators=None if pretty_json else (",", ":"))
        print(f"Mykee JSON BBOX Resizer: {ow}x{oh} -> {tw}x{th} ({mode}), "
              f"{len(result['items'])}/{len(data['items'])} items kept")
        return {"ui": {"json_display": [text]}, "result": (text, tw, th)}


NODE_CLASS_MAPPINGS = {"MykeeJsonBboxResizer": MykeeJsonBboxResizer}
NODE_DISPLAY_NAME_MAPPINGS = {"MykeeJsonBboxResizer": "Mykee JSON BBOX Resizer"}
