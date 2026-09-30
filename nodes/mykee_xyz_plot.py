"""
ComfyUI-Mykee-Nodes / XYZ plot nodes

Two nodes that together build an XYZ plot out of ANY widget parameter of
ANY node:

- "Mykee XYZ Plot" (the controller). Its x / y / z outputs are dragged onto
  the widgets that should be varied (cfg, steps, ckpt_name, seed, ...). The
  frontend (web/mykee_xyz_plot.js) then shows the connected parameter on the
  node: a value list for numbers / text, a searchable multi-select for
  combos. When the workflow is queued, the frontend splits the single run
  into one prompt per cell (Z outermost, X innermost) and writes the cell's
  value straight into the target widgets, the same way a Primitive node
  does. Because of that, only nodes whose value really changed re-run
  between cells - a checkpoint loader on the Z axis loads each model once,
  not once per cell.

- "Mykee XYZ Plot Assembler" collects the images of the cells (one per
  prompt) and, once the last cell has arrived, outputs the labelled grid.
  Until then it only shows a preview of the partial grid and blocks its
  outputs, so a Save Image node after it saves the finished table only.

The x / y / z outputs are only really used when a prompt reaches the server
without the frontend (e.g. an API call): then the node outputs the values
of cell 0.
"""

import json
import os
import time
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

try:
    from comfy_execution.graph import ExecutionBlocker
except ImportError:  # very old ComfyUI
    ExecutionBlocker = None


class AnyType(str):
    """Wildcard type: the axis outputs can be linked to any widget input."""

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


ANY = AnyType("*")
XYZ_INFO_TYPE = "MYKEE_XYZ_INFO"
AXIS_NAMES = ("X", "Y", "Z")
MODEL_EXTENSIONS = (
    ".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft", ".onnx", ".pkl",
)

# Finished / in-progress plots, keyed by run id. Only the most recent few are
# kept so an abandoned run cannot pile up in RAM.
_MAX_RUNS = 4
_RUNS = OrderedDict()


# ---------------------------------------------------------------------------
# Config / value helpers
# ---------------------------------------------------------------------------

def _parse_config(raw):
    try:
        cfg = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
    except (TypeError, ValueError):
        cfg = {}
    return cfg if isinstance(cfg, dict) else {}


def _active_axes(cfg):
    """Always returns 3 entries (X, Y, Z); an unused axis is None."""
    axes = cfg.get("axes")
    if not isinstance(axes, list):
        axes = []
    result = []
    for i in range(3):
        axis = axes[i] if i < len(axes) else None
        if isinstance(axis, dict) and isinstance(axis.get("values"), list) and axis["values"]:
            result.append(axis)
        else:
            result.append(None)
    return result


def _cast(value, value_type):
    if value_type == "INT":
        return int(round(float(value)))
    if value_type == "FLOAT":
        return float(value)
    if value_type == "BOOLEAN":
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    return str(value)


def _format_value(value, value_type):
    if value_type == "FLOAT":
        text = f"{float(value):.4f}".rstrip("0").rstrip(".")
        return text if text not in ("", "-0") else "0"
    if value_type == "BOOLEAN":
        return "true" if _cast(value, "BOOLEAN") else "false"
    text = str(value)
    # Model file names: the folder part and extension only add noise.
    if text.lower().endswith(MODEL_EXTENSIONS):
        text = os.path.splitext(text.replace("\\", "/").rsplit("/", 1)[-1])[0]
    return text


def _cell_indices(cell, dims):
    nx, ny, _nz = dims
    return cell % nx, (cell // nx) % ny, cell // (nx * ny)


# ---------------------------------------------------------------------------
# Controller node
# ---------------------------------------------------------------------------

class MykeeXYZPlot:
    CATEGORY = "Mykee/Utils"
    FUNCTION = "main"
    RETURN_TYPES = (ANY, ANY, ANY, "INT", XYZ_INFO_TYPE)
    RETURN_NAMES = ("x", "y", "z", "seed", "xyz_info")
    OUTPUT_TOOLTIPS = (
        "Drag onto any widget parameter to use it as the X axis (columns).",
        "Drag onto any widget parameter to use it as the Y axis (rows).",
        "Drag onto any widget parameter to use it as the Z axis (one grid per value).",
        "The optional seed input, passed through unchanged (0 if not connected).",
        "Connect to the Mykee XYZ Plot Assembler.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Holds the node's settings (JSON). Hidden: the frontend draws
                # its own panel, and fills in the resolved values at queue time.
                "xyz_config": (
                    "STRING",
                    {"default": "{}", "hidden": True, "socketless": True},
                ),
            },
            "optional": {
                "seed": (
                    "INT",
                    {
                        "forceInput": True,
                        "tooltip": (
                            "Optional. Passed through to the seed output and "
                            "printed in the plot's title, so the table documents "
                            "which seed it was made with."
                        ),
                    },
                ),
            },
        }

    def main(self, xyz_config="{}", seed=None):
        cfg = _parse_config(xyz_config)
        axes = _active_axes(cfg)
        dims = [len(a["values"]) if a else 1 for a in axes]
        total = dims[0] * dims[1] * dims[2]

        cell = int(cfg.get("cell", 0) or 0)
        run_id = str(cfg.get("run_id") or "")
        if not run_id:
            # Queued without the ComfyUI frontend: there is no fan-out, so
            # only cell 0 is rendered.
            run_id = f"single-{time.time_ns()}"
            print(
                "[Mykee XYZ Plot] No run id in the prompt (queued without the "
                "ComfyUI frontend?) - only the first cell is rendered."
            )
        if not 0 <= cell < total:
            raise ValueError(f"[Mykee XYZ Plot] Cell index {cell} is out of range (0..{total - 1}).")

        idx = _cell_indices(cell, dims)
        values = []
        info_axes = []
        for axis, i in zip(axes, idx):
            if axis is None:
                values.append(None)
                info_axes.append(None)
                continue
            value_type = str(axis.get("type", "STRING")).upper()
            values.append(_cast(axis["values"][i], value_type))
            info_axes.append(
                {
                    "label": str(axis.get("label") or axis.get("param") or "?"),
                    "value_labels": [_format_value(v, value_type) for v in axis["values"]],
                }
            )

        info = {
            "run_id": run_id,
            "cell": cell,
            "total": total,
            "dims": dims,
            "axes": info_axes,
            "seed": seed,
        }
        return (values[0], values[1], values[2], int(seed) if seed is not None else 0, info)


# ---------------------------------------------------------------------------
# Grid drawing
# ---------------------------------------------------------------------------

_FONT_CANDIDATES = ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf")


def _load_font(size):
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _text_width(font, text):
    try:
        return font.getlength(text)
    except AttributeError:
        return font.getsize(text)[0]


def _wrap(font, text, max_width):
    """Word wrap; words longer than a line (file names) are broken by character."""
    lines = []
    for paragraph in str(text).split("\n"):
        line = ""
        for word in paragraph.split(" "):
            candidate = f"{line} {word}" if line else word
            if _text_width(font, candidate) <= max_width:
                line = candidate
                continue
            if line:
                lines.append(line)
                line = ""
            while _text_width(font, word) > max_width and len(word) > 1:
                cut = len(word)
                while cut > 1 and _text_width(font, word[:cut]) > max_width:
                    cut -= 1
                lines.append(word[:cut])
                word = word[cut:]
            line = word
        lines.append(line)
    return lines or [""]


def _line_height(font):
    box = font.getbbox("Ágy")
    return int((box[3] - box[1]) * 1.35) + 1


def _draw_lines(draw, font, lines, box, align="center", fill=(0, 0, 0)):
    x0, y0, x1, y1 = box
    lh = _line_height(font)
    y = y0 + max(0, ((y1 - y0) - lh * len(lines)) // 2)
    for line in lines:
        w = _text_width(font, line)
        if align == "center":
            x = x0 + ((x1 - x0) - w) / 2
        else:
            x = x0
        draw.text((x, y), line, font=font, fill=fill)
        y += lh


def _fit_cell(img, size, bg):
    """Letterboxes img (PIL) into size, keeping its aspect ratio."""
    if img.size == size:
        return img
    scale = min(size[0] / img.width, size[1] / img.height)
    resized = img.resize(
        (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
        Image.LANCZOS,
    )
    canvas = Image.new("RGB", size, bg)
    canvas.paste(resized, ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2))
    return canvas


def _render_grids(run, font_size, cell_max_size):
    info = run["info"]
    cells = run["cells"]
    nx, ny, nz = info["dims"]
    ax_x, ax_y, ax_z = info["axes"]

    bg, fg, pending_bg = (255, 255, 255), (0, 0, 0), (60, 60, 60)
    gap = 6

    first = next(iter(cells.values()))
    cw, ch = first.width, first.height
    if cell_max_size > 0 and max(cw, ch) > cell_max_size:
        scale = cell_max_size / max(cw, ch)
        cw, ch = max(1, round(cw * scale)), max(1, round(ch * scale))

    if font_size <= 0:
        font_size = max(14, min(64, cw // 22))
    font = _load_font(font_size)
    title_font = _load_font(max(12, round(font_size * 0.8)))
    lh = _line_height(font)
    pad = max(4, font_size // 3)

    # Column headers (X values).
    col_lines = [_wrap(font, v, cw - 2 * pad) for v in ax_x["value_labels"]] if ax_x else []
    header_h = (max(len(l) for l in col_lines) * lh + 2 * pad) if col_lines else 0

    # Row headers (Y values).
    if ax_y:
        row_w_max = max(cw // 2, 120)
        widest = max(_text_width(font, v) for v in ax_y["value_labels"])
        row_w = int(min(row_w_max, widest + 2 * pad))
        row_lines = [_wrap(font, v, row_w - 2 * pad) for v in ax_y["value_labels"]]
        row_w = max(row_w, int(max(_text_width(font, l) for ls in row_lines for l in ls)) + 2 * pad)
    else:
        row_w, row_lines = 0, []

    grid_w = row_w + nx * cw + (nx - 1) * gap + pad
    # Title: which parameter is on which axis (+ seed).
    title_parts = []
    for name, axis in zip(AXIS_NAMES, (ax_x, ax_y, ax_z)):
        if axis:
            title_parts.append(f"{name}: {axis['label']}")
    if info.get("seed") is not None:
        title_parts.append(f"seed: {info['seed']}")

    blocks = []
    for zi in range(nz):
        parts = list(title_parts)
        if ax_z:
            parts = [f"{ax_z['label']} = {ax_z['value_labels'][zi]}"] + [
                p for p in parts if not p.startswith("Z:")
            ]
        title_lines = _wrap(title_font, "   |   ".join(parts), grid_w - 2 * pad) if parts else []
        title_h = len(title_lines) * _line_height(title_font) + 2 * pad if title_lines else 0

        height = title_h + header_h + ny * ch + (ny - 1) * gap + pad
        block = Image.new("RGB", (grid_w, height), bg)
        draw = ImageDraw.Draw(block)
        if title_lines:
            _draw_lines(draw, title_font, title_lines, (pad, 0, grid_w - pad, title_h), "left", fg)

        top = title_h + header_h
        for xi, lines in enumerate(col_lines):
            x0 = row_w + xi * (cw + gap)
            _draw_lines(draw, font, lines, (x0, title_h, x0 + cw, top), "center", fg)
        for yi, lines in enumerate(row_lines):
            y0 = top + yi * (ch + gap)
            _draw_lines(draw, font, lines, (pad, y0, row_w - pad, y0 + ch), "center", fg)

        for yi in range(ny):
            for xi in range(nx):
                cell = zi * nx * ny + yi * nx + xi
                pos = (row_w + xi * (cw + gap), top + yi * (ch + gap))
                if cell in cells:
                    block.paste(_fit_cell(cells[cell], (cw, ch), bg), pos)
                else:
                    draw.rectangle([pos, (pos[0] + cw - 1, pos[1] + ch - 1)], fill=pending_bg)
                    _draw_lines(
                        draw, font, ["pending"],
                        (pos[0], pos[1], pos[0] + cw, pos[1] + ch), "center", (170, 170, 170),
                    )
        blocks.append(block)

    # Blocks share their width; heights only differ if the Z titles wrap
    # differently - pad them to the same height so they fit in one batch.
    max_h = max(b.height for b in blocks)
    blocks = [b if b.height == max_h else _pad_bottom(b, max_h, bg) for b in blocks]

    combined = Image.new("RGB", (grid_w, max_h * nz + gap * 4 * (nz - 1)), bg)
    for zi, block in enumerate(blocks):
        combined.paste(block, (0, zi * (max_h + gap * 4)))
    return combined, blocks


def _pad_bottom(img, height, bg):
    canvas = Image.new("RGB", (img.width, height), bg)
    canvas.paste(img, (0, 0))
    return canvas


def _to_tensor(images):
    arrays = [np.asarray(img, dtype=np.float32) / 255.0 for img in images]
    return torch.from_numpy(np.stack(arrays, axis=0))


def _to_pil(image_tensor):
    array = np.clip(image_tensor.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    return Image.fromarray(array)


# ---------------------------------------------------------------------------
# Assembler node
# ---------------------------------------------------------------------------

class MykeeXYZPlotAssembler:
    CATEGORY = "Mykee/Utils"
    FUNCTION = "main"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("plot", "plot_per_z")
    OUTPUT_TOOLTIPS = (
        "The whole table in one image (Z grids stacked below each other). "
        "Only produced once the last cell has finished.",
        "One grid image per Z value, as a batch.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The image of the current cell (the first image of a batch is used)."}),
                "xyz_info": (XYZ_INFO_TYPE, {"tooltip": "From the Mykee XYZ Plot node."}),
                "font_size": (
                    "INT",
                    {"default": 0, "min": 0, "max": 256, "tooltip": "Label font size in pixels. 0 = automatic (from the cell size)."},
                ),
                "cell_max_size": (
                    "INT",
                    {
                        "default": 1024, "min": 0, "max": 8192, "step": 8,
                        "tooltip": "Cells are scaled down so their longer side is at most this many pixels. 0 = original size.",
                    },
                ),
            },
        }

    def main(self, images, xyz_info, font_size=0, cell_max_size=1024):
        run_id = xyz_info["run_id"]
        run = _RUNS.get(run_id)
        if run is None:
            run = {"info": xyz_info, "cells": {}}
            _RUNS[run_id] = run
            while len(_RUNS) > _MAX_RUNS:
                _RUNS.popitem(last=False)
        _RUNS.move_to_end(run_id)

        if images.shape[0] > 1:
            print(
                f"[Mykee XYZ Plot] Cell {xyz_info['cell']} got a batch of "
                f"{images.shape[0]} images - only the first one goes into the plot."
            )
        run["cells"][int(xyz_info["cell"])] = _to_pil(images[0])

        done, total = len(run["cells"]), int(xyz_info["total"])
        combined, blocks = _render_grids(run, int(font_size), int(cell_max_size))
        preview = self._preview(combined)
        print(f"[Mykee XYZ Plot] Cell {done}/{total} collected (run {run_id}).")

        if done < total:
            blocked = ExecutionBlocker(None) if ExecutionBlocker is not None else None
            return {"ui": preview, "result": (blocked, blocked)}

        _RUNS.pop(run_id, None)  # finished - free the cell images
        return {"ui": preview, "result": (_to_tensor([combined]), _to_tensor(blocks))}

    @staticmethod
    def _preview(image):
        from nodes import PreviewImage  # ComfyUI's own; imported lazily

        return PreviewImage().save_images(_to_tensor([image]), "mykee_xyz")["ui"]


NODE_CLASS_MAPPINGS = {
    "MykeeXYZPlot": MykeeXYZPlot,
    "MykeeXYZPlotAssembler": MykeeXYZPlotAssembler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeXYZPlot": "Mykee XYZ Plot",
    "MykeeXYZPlotAssembler": "Mykee XYZ Plot Assembler",
}
