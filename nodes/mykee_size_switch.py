"""
ComfyUI-Mykee-Nodes / size switch-batch node

"Mykee Size Switch/Batch": 1..32 width/height pairs on one node, each pair
with a radio button (switch mode) or a checkbox (batch mode).

- switch mode: only the pair picked with the radio button goes out.
- batch mode: every ticked pair goes out, in order, as a list - so the nodes
  after it run once per pair (one image per size for an Empty Latent ->
  KSampler chain). width and height lists have the same length, so they stay
  aligned downstream.

image_N (optional IMAGE input): the size of the connected image is used as
the pair, and the typed width / height are ignored (the frontend greys them
out). It counts as an outside value, so adjust_external applies to it.

width_N / height_N are normal INT widgets, so each of them can also be fed
from outside by connecting a link to the widget; the connected value then
replaces the typed one. The inputs are lazy: a pair that is not selected is
never evaluated.

Rounding and aspect ratio:

- divisible_by (Off / 8 / 16 / 32 / 64): typed numbers are rounded to the
  nearest multiple of it.
- adjust_external: off = values that arrive through a connection pass
  through untouched; on = they are rounded / fitted to the ratio too.
- ratio_N (Custom, 1:1, 16:9, ...): one side of the pair is the base (the
  side edited last, or the connected one), the other side is calculated from
  it and rounded to the divisor, so only whole numbers come out.

The pair rows are shown / hidden by web/mykee_size_switch.js (pair_count) and
the selection is stored in the hidden pairs_config widget. For the prompt the
frontend sends the selected pairs only, and the values of the unselected pairs
are replaced by a constant, so editing an unselected pair does not make
anything re-run.
"""

import json
import math

try:
    from comfy_execution.graph import ExecutionBlocker
except ImportError:  # very old ComfyUI
    ExecutionBlocker = None

MAX_PAIRS = 32
MIN_SIZE = 16
MAX_SIZE = 16384
DEFAULT_SIZE = 1024
DIVISORS = ["Off", "8", "16", "32", "64"]
RATIOS = ["Custom", "1:1", "16:9", "9:16", "2:1", "1:2", "3:2", "2:3", "4:3", "3:4", "5:4", "4:5"]


def _selected_items(pairs_config):
    """Selected pairs from the frontend as dicts (index, driver, w_linked,
    h_linked), or None if the config carries no selection at all (e.g. a
    workflow sent through the API without the frontend)."""
    try:
        cfg = json.loads(pairs_config) if pairs_config else {}
    except (TypeError, ValueError):
        return None
    items = cfg.get("items") if isinstance(cfg, dict) else None
    if not isinstance(items, list):
        return None
    result, seen = [], set()
    for item in items:
        data = item if isinstance(item, dict) else {"index": item}
        try:
            index = int(data.get("index"))
        except (TypeError, ValueError):
            continue
        if 1 <= index <= MAX_PAIRS and index not in seen:
            seen.add(index)
            result.append(
                {
                    "index": index,
                    "driver": "h" if data.get("driver") == "h" else "w",
                    "w_linked": bool(data.get("w_linked")),
                    "h_linked": bool(data.get("h_linked")),
                }
            )
    return result


def _used_items(batch_mode, pair_count, pairs_config):
    """The pairs that actually go out, in order."""
    selected = _selected_items(pairs_config)
    if selected is None:
        # No frontend selection: batch = every pair, switch = the first one.
        indices = range(1, pair_count + 1) if batch_mode else [1]
        selected = [
            {"index": i, "driver": "w", "w_linked": False, "h_linked": False} for i in indices
        ]
    used = [item for item in selected if item["index"] <= pair_count]
    if not batch_mode:
        used = used[:1]  # switch: exactly one
    return used


def _used_indices(batch_mode, pair_count, pairs_config):
    return [item["index"] for item in _used_items(batch_mode, pair_count, pairs_config)]


def _divisor(divisible_by):
    try:
        return int(divisible_by)
    except (TypeError, ValueError):
        return 0  # "Off"


def _snap(value, divisor):
    """Round to the nearest multiple of divisor (half up); whole numbers only."""
    lowest = max(MIN_SIZE, divisor)
    if divisor:
        result = int(math.floor(value / divisor + 0.5)) * divisor
    else:
        result = int(math.floor(value + 0.5))
    return max(lowest, min(MAX_SIZE, result))


def _parse_ratio(ratio):
    """"16:9" -> (16, 9); "Custom" / anything else -> None."""
    try:
        a, b = str(ratio).split(":")
        a, b = int(a), int(b)
    except ValueError:
        return None
    return (a, b) if a > 0 and b > 0 else None


def _resolve_pair(width, height, ratio, divisor, item, adjust_external):
    """The final (width, height) of one pair.

    A connected value passes through untouched unless adjust_external is on.
    With a ratio, the base side (item["driver"], or the only unconnected /
    only connected side) is rounded and the other side is calculated from it.
    """
    w_raw = adjust_external is False and item["w_linked"]
    h_raw = adjust_external is False and item["h_linked"]
    ratio = _parse_ratio(ratio)
    if ratio is None:
        return (width if w_raw else _snap(width, divisor), height if h_raw else _snap(height, divisor))

    rw, rh = ratio
    driver = item["driver"]
    if item["w_linked"] != item["h_linked"]:
        driver = "w" if item["w_linked"] else "h"  # the connected side leads
    if driver == "w":
        base = width if w_raw else _snap(width, divisor)
        other = _snap(base * rh / rw, divisor)
        return (base, height if h_raw else other)
    base = height if h_raw else _snap(height, divisor)
    other = _snap(base * rw / rh, divisor)
    return (width if w_raw else other, base)


def _image_size(image):
    """(width, height) of an IMAGE tensor [B, H, W, C], or None."""
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) < 3:
        return None
    return int(shape[2]), int(shape[1])


def _to_size(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return DEFAULT_SIZE


class MykeeSizeSwitchBatch:
    CATEGORY = "Mykee/Image"
    FUNCTION = "main"
    RETURN_TYPES = ("INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "count", "label")
    OUTPUT_IS_LIST = (True, True, False, True)
    OUTPUT_TOOLTIPS = (
        "switch: the selected width. batch: the width of every ticked pair as a "
        "list - the nodes after it run once per pair.",
        "The matching heights (same order and length as width).",
        "How many pairs went out.",
        'Each pair as text, e.g. "1024x768" (handy for file names).',
    )

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, MAX_PAIRS + 1):
            optional[f"use_{i}"] = (
                "BOOLEAN",
                {
                    "default": i == 1,
                    "socketless": True,
                    "tooltip": f"Pair {i}: radio button (switch mode) / checkbox (batch mode).",
                },
            )
            optional[f"ratio_{i}"] = (
                RATIOS,
                {
                    "default": "Custom",
                    "socketless": True,
                    "tooltip": f"Aspect ratio (width:height) of pair {i}. With a ratio, one side is "
                    "calculated from the other (the one edited last, or the connected one).",
                },
            )
            optional[f"image_{i}"] = (
                "IMAGE",
                {
                    "lazy": True,
                    "tooltip": f"Optional. Takes the width / height of pair {i} from this image "
                    "(the typed values are then ignored).",
                },
            )
            for axis in ("width", "height"):
                optional[f"{axis}_{i}"] = (
                    "INT",
                    {
                        "default": DEFAULT_SIZE,
                        "min": MIN_SIZE,
                        "max": MAX_SIZE,
                        "step": 8,
                        "lazy": True,
                        "tooltip": f"{axis.capitalize()} of pair {i}. Can be fed from outside.",
                    },
                )
        return {
            "required": {
                "batch_mode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "batch",
                        "label_off": "switch",
                        "tooltip": "switch: only the pair picked with the radio button goes out. "
                        "batch: every ticked pair goes out, one after the other.",
                    },
                ),
                "pair_count": (
                    "INT",
                    {"default": 1, "min": 1, "max": MAX_PAIRS, "tooltip": "Number of width/height pairs."},
                ),
                "divisible_by": (
                    DIVISORS,
                    {
                        "default": "Off",
                        "tooltip": "Typed numbers are rounded (up or down) to the nearest multiple of this.",
                    },
                ),
                "adjust_external": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "adjust",
                        "label_off": "pass through",
                        "tooltip": "pass through: values that arrive through a connection are used as they are. "
                        "adjust: they are rounded to divisible_by and fitted to the ratio too.",
                    },
                ),
                "pairs_config": (
                    "STRING",
                    {"default": "{}", "hidden": True, "socketless": True},
                ),
            },
            "optional": optional,
        }

    def check_lazy_status(
        self, batch_mode=False, pair_count=1, divisible_by="Off", adjust_external=False, pairs_config="{}", **kwargs
    ):
        # A connected input that has not been evaluated yet shows up as None.
        needed = []
        for index in _used_indices(batch_mode, pair_count, pairs_config):
            image_key = f"image_{index}"
            if image_key in kwargs:
                # Connected image: it replaces the typed width / height.
                if kwargs[image_key] is None:
                    needed.append(image_key)
                continue
            for axis in ("width", "height"):
                key = f"{axis}_{index}"
                if key in kwargs and kwargs[key] is None:
                    needed.append(key)
        return needed

    def main(
        self, batch_mode=False, pair_count=1, divisible_by="Off", adjust_external=False, pairs_config="{}", **kwargs
    ):
        widths, heights, labels = [], [], []
        divisor = _divisor(divisible_by)
        for item in _used_items(batch_mode, pair_count, pairs_config):
            index = item["index"]
            width = _to_size(kwargs.get(f"width_{index}"))
            height = _to_size(kwargs.get(f"height_{index}"))
            image_size = _image_size(kwargs.get(f"image_{index}"))
            if image_size is not None:
                # An image counts as an outside value for both sides.
                width, height = image_size
                item = dict(item, w_linked=True, h_linked=True)
            width, height = _resolve_pair(
                width,
                height,
                kwargs.get(f"ratio_{index}", "Custom"),
                divisor,
                item,
                bool(adjust_external),
            )
            widths.append(width)
            heights.append(height)
            labels.append(f"{width}x{height}")

        if not widths:
            message = (
                "[Mykee Size Switch/Batch] No ticked pair - output blocked."
                if batch_mode
                else "[Mykee Size Switch/Batch] No pair is selected - output blocked."
            )
            print(message)
            if ExecutionBlocker is not None:
                blocked = ExecutionBlocker(None)
                return ([blocked], [blocked], blocked, [blocked])
            return ([], [], 0, [])

        return (widths, heights, len(widths), labels)


NODE_CLASS_MAPPINGS = {
    "MykeeSizeSwitchBatch": MykeeSizeSwitchBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeSizeSwitchBatch": "Mykee Size Switch/Batch",
}
