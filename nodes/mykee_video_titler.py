"""
ComfyUI-Mykee-Nodes / Mykee Video Titler

The video version of the Mykee Meme Maker: top / bottom captions that appear
and disappear at set times, with a player on the node to find those times.

What it does
------------
- video in, video out (no image input). The video can be cut: trim_start /
  trim_end (seconds of the source video, trim_end 0 = until the end).
- A list of titles. Every title has a start and an end time (seconds of the
  source video) and its own captions and style - everything the Meme Maker
  offers for a caption (texts, font, sizes, colors, outline, wrapping, ...).
  Titles are kept in time order and may not overlap. With an empty list the
  captions typed on the node are shown over the whole video (Meme Maker
  behaviour).
- The frame settings (placement, frame color / width / height) belong to the
  whole video, not to a title: a video cannot change its size halfway. In
  frame placement the bands are as tall as the tallest title needs, for the
  whole video.
- text input (optional): an SRT / WebVTT subtitle file's text, or plain text
  (one title per paragraph or line, spread evenly over the video).
    text_mode "import to list": the node does not draw the text; the Titles
        tab's Import button turns it into titles that can be edited.
    text_mode "subtitle layer": the cues are drawn as they are, as a third
        layer above the titles, with the current style, subtitle_font_size
        and subtitle_position.
- The text input also takes the JSON this node saved: it is recognised by its
  content and can be loaded back with the Import button (titles with every
  setting, the cut, the frame settings). Whatever cannot be used here - a font
  that is not installed, an invalid value, a title that overlaps another - is
  left out and reported, the rest is loaded.
- The file name of the video is looked up like Mykee Get Filename does it,
  and filename_tag is appended. With save_output on, the node writes
  <name><tag>.mp4 and <name><tag>.json (the timing list with every caption
  setting) into ComfyUI's output folder - the same name for both. An existing
  file is never replaced unless overwrite is on; a counter is added instead.

Preview
-------
The browser plays the source video itself: the frontend finds the file of the
loader behind the video input and asks ComfyUI's own /view route for it. When
the video is not a plain file the browser can play (generated in memory,
trimmed before this node, a codec browsers do not know), the node writes a
small copy into ComfyUI's temp folder when it runs, and the player uses that.
The captions are drawn over the player by the same Python code as the result.

Security
--------
- Three HTTP routes. /mykee/titler/probe takes a file name, a subfolder and a
  folder type; the file must resolve (symlinks and ".." included) to a video
  file inside ComfyUI's input / output / temp folder - the same files the
  built-in /view route already serves - and only its frame rate, size and
  length are returned. /mykee/titler/parse and /mykee/titler/preview take no
  path at all; sizes, counts and every value are limited and validated.
- Saving: only into ComfyUI's output folder. The file name is reduced to a
  bare, safe name (no folders, no reserved names), whatever the workflow says.
"""

import asyncio
import base64
import io
import json
import math
import os
import re
import types
from collections import OrderedDict
from fractions import Fraction

import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    import av
except Exception:  # pragma: no cover - ComfyUI ships PyAV
    av = None

try:
    import folder_paths
except Exception:  # imported outside ComfyUI
    folder_paths = None

try:
    from server import PromptServer
    from aiohttp import web as _aiohttp_web
except Exception:  # imported outside ComfyUI
    PromptServer = None
    _aiohttp_web = None

from .mykee_get_filename import find_upstream_name
from .mykee_meme_maker import (
    DEFAULTS, MAX_TEXT, PREVIEW_MIN_SIDE, MykeeMemeMaker, _make_video, _number,
    build_layout, clean_params, compose_frames, font_names,
)
from .mykee_video_info import _base_name, _clean_tag, _file_source

NODE_NAME = "MykeeVideoTitler"
LOG = "[Mykee Video Titler]"

FALLBACK_NAME = "video"
MAX_TITLES = 2000
MAX_TITLES_JSON = 2 * 1024 * 1024       # characters of the titles field / bytes of a route body
MAX_SOURCE_TEXT = 2 * 1024 * 1024       # characters of the text input that are read
MAX_TIME = 1_000_000.0                  # seconds
MAX_NAME_LENGTH = 150
PREVIEW_SIDE = 640                      # longest side of the caption layer sent to the player
PLAIN_CUE_SECONDS = 3.0                 # length of a plain-text title when the video length is unknown
TEXT_MODES = ("import to list", "subtitle layer")
SUBTITLE_POSITIONS = ("bottom", "top")
VIDEO_EXTS = frozenset("mp4 m4v mov mkv webm avi wmv flv mpg mpeg ts mts m2ts 3gp ogv".split())
BROWSER_CODECS = frozenset(("h264", "vp8", "vp9", "av1"))
VIEW_TYPES = ("input", "output", "temp")

# Settings of the whole video (taken from the node when it runs) - everything
# else in DEFAULTS belongs to a title.
FRAME_KEYS = ("placement", "frame_color", "frame_width", "frame_width_mode", "frame_height_mode", "frame_height")
STYLE_KEYS = tuple(k for k in DEFAULTS if k not in FRAME_KEYS)
_COLOR_KEYS = ("text_color", "stroke_color", "frame_color")
_RESERVED = frozenset(
    ["con", "prn", "aux", "nul"] + [f"com{i}" for i in range(1, 10)] + [f"lpt{i}" for i in range(1, 10)]
)


def _own_message(exc):
    """A Meme Maker validation message under this node's name."""
    return str(exc).replace("[Mykee Meme Maker]", LOG)


def _hex(color):
    return "#{:02X}{:02X}{:02X}".format(*color) if isinstance(color, tuple) else str(color)


def _seconds(name, value):
    return round(_number(name, value, 0.0, MAX_TIME), 6)


# ---------------------------------------------------------------------------
# The text input: SRT / WebVTT / plain text
# ---------------------------------------------------------------------------

_TIME = r"(?:(\d{1,4}):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?"
_TIMING_RE = re.compile(rf"^\s*{_TIME}\s*-->\s*{_TIME}")
_TAG_RE = re.compile(r"<[^>\n]{0,200}>|\{\\[^}\n]{0,200}\}")


def _stamp(h, m, s, ms):
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + (int((ms or "0").ljust(3, "0")) / 1000.0)


def _cue_text(lines):
    text = _TAG_RE.sub("", "\n".join(lines)).strip()
    text = "".join(ch for ch in text if ch == "\n" or ch >= " " and ch != "\x7f")
    return text[:MAX_TEXT]


def parse_text(text, duration=0.0):
    """Cues [{start, end, text}] from SRT / WebVTT, or from plain text.

    Returns (cues, kind): kind is "timed" for a subtitle file and "plain" for
    text without time codes. Cues are in time order and never overlap (a cue
    that runs into the next one is cut there).
    """
    text = str(text or "")[:MAX_SOURCE_TEXT].replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    lines = text.split("\n")
    cues = []
    i = 0
    while i < len(lines) and len(cues) < MAX_TITLES:
        match = _TIMING_RE.match(lines[i])
        if not match:
            i += 1
            continue
        g = match.groups()
        start, end = _stamp(*g[0:4]), _stamp(*g[4:8])
        i += 1
        body = []
        while i < len(lines) and lines[i].strip() and not _TIMING_RE.match(lines[i]):
            body.append(lines[i])
            i += 1
        # the number line of the next SRT block is not part of this text
        if i < len(lines) and _TIMING_RE.match(lines[i]) and body and body[-1].strip().isdigit():
            body.pop()
        cue = _cue_text(body)
        if cue and end > start and start <= MAX_TIME:
            cues.append({"start": round(start, 3), "end": round(min(end, MAX_TIME), 3), "text": cue})

    kind = "timed"
    if not cues:
        kind = "plain"
        blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
        if len(blocks) == 1:
            blocks = [line for line in blocks[0].split("\n") if line.strip()]
        blocks = [t for t in (_cue_text([b]) for b in blocks[:MAX_TITLES]) if t]
        if blocks:
            try:
                duration = float(duration)
            except (TypeError, ValueError):
                duration = 0.0
            if not math.isfinite(duration) or duration <= 0:
                duration = PLAIN_CUE_SECONDS * len(blocks)
            step = min(duration, MAX_TIME) / len(blocks)
            cues = [
                {"start": round(n * step, 3), "end": round((n + 1) * step, 3), "text": t}
                for n, t in enumerate(blocks)
            ]

    cues.sort(key=lambda c: (c["start"], c["end"]))
    out = []
    for cue in cues:
        if out and cue["start"] < out[-1]["end"]:
            out[-1]["end"] = cue["start"]
            if out[-1]["end"] <= out[-1]["start"]:
                out.pop()
        if cue["end"] > cue["start"]:
            out.append(cue)
    return out, kind


# ---------------------------------------------------------------------------
# The text input: a JSON this node saved
# ---------------------------------------------------------------------------

def _usable(raw, keys, ignored):
    """The settings of raw that can be used here, one by one (colors as hex).
    What cannot be used is left out and named in the ignored set."""
    out = {}
    fonts = None
    for key in keys:
        if key not in raw:
            continue
        value = raw[key]
        if isinstance(value, (dict, list)) or value is None:
            ignored.add(key)
            continue
        try:
            cleaned = clean_params({key: value})[key]
        except ValueError:
            ignored.add(key)
            continue
        if key == "font" and cleaned:
            if fonts is None:
                try:
                    fonts = set(font_names())
                except Exception:
                    fonts = set()
            if cleaned not in fonts:
                ignored.add(f"font '{cleaned[:40]}' (not installed)")
                continue
        out[key] = _hex(cleaned) if key in _COLOR_KEYS else cleaned
    return out


def parse_project(text):
    """What can be loaded back from a JSON saved by this node, or None when
    the text is not such a JSON.

    Returns {titles: [{start, end, style}], trim: {start, end} | None,
    frame: {...}, form: {...} | None, cues: [...], ignored: [...]}; everything
    in it is validated, so it can be used as it is.
    """
    source = str(text or "")[:MAX_SOURCE_TEXT].lstrip("\ufeff \t\r\n")
    if not source.startswith("{"):
        return None
    try:
        data = json.loads(source)
    except (ValueError, RecursionError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("titles"), list):
        return None

    ignored = set()
    titles, bad = [], 0
    for item in data["titles"][:MAX_TITLES]:
        try:
            if not isinstance(item, dict):
                raise ValueError
            start, end = _seconds("start", item.get("start")), _seconds("end", item.get("end"))
            if end <= start:
                raise ValueError
        except ValueError:
            bad += 1
            continue
        titles.append({"start": start, "end": end, "style": _usable(item, STYLE_KEYS, ignored)})
    titles.sort(key=lambda t: (t["start"], t["end"]))
    kept, overlapping = [], 0
    for title in titles:
        if kept and title["start"] < kept[-1]["end"] - 1e-6:
            overlapping += 1
            continue
        kept.append(title)
    if len(data["titles"]) > MAX_TITLES:
        ignored.add(f"titles after the first {MAX_TITLES}")
    if bad:
        ignored.add(f"{bad} title(s) without a valid start / end")
    if overlapping:
        ignored.add(f"{overlapping} title(s) that overlap another one")

    trim = None
    raw_trim = data.get("trim")
    if isinstance(raw_trim, dict):
        try:
            start = _seconds("trim start", raw_trim.get("start", 0))
            end = 0.0 if raw_trim.get("until_end") is True else _seconds("trim end", raw_trim.get("end", 0))
            if end and end <= start:
                raise ValueError
            trim = {"start": start, "end": end}
        except ValueError:
            ignored.add("trim")

    frame = _usable(data["frame"], FRAME_KEYS, ignored) if isinstance(data.get("frame"), dict) else {}
    form = data.get("whole_video_caption")
    form = _usable(form, STYLE_KEYS, ignored) if isinstance(form, dict) else None

    cues = []
    subtitles = data.get("subtitles")
    if isinstance(subtitles, dict) and isinstance(subtitles.get("cues"), list):
        for cue in subtitles["cues"][:MAX_TITLES]:
            try:
                start, end = _seconds("start", cue.get("start")), _seconds("end", cue.get("end"))
                body = _cue_text([str(cue.get("text") or "")])
                if end > start and body and not (cues and start < cues[-1]["end"]):
                    cues.append({"start": start, "end": end, "text": body})
            except (ValueError, AttributeError):
                continue

    return {"titles": kept, "trim": trim, "frame": frame, "form": form, "cues": cues, "ignored": sorted(ignored)}


# ---------------------------------------------------------------------------
# The list of titles
# ---------------------------------------------------------------------------

def parse_titles(raw):
    """Validated titles [{start, end, style}] in time order (style = the
    title's own raw caption settings). Raises ValueError with a readable
    message on anything invalid, overlaps included."""
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        if len(raw) > MAX_TITLES_JSON:
            raise ValueError(f"{LOG} the list of titles is too large.")
        try:
            raw = json.loads(raw)
        except ValueError:
            raise ValueError(f"{LOG} the list of titles is not valid JSON.") from None
    if isinstance(raw, dict):
        raw = raw.get("titles")
    if not isinstance(raw, list):
        raise ValueError(f"{LOG} the list of titles must be a list.")
    if len(raw) > MAX_TITLES:
        raise ValueError(f"{LOG} too many titles (at most {MAX_TITLES}).")
    titles = []
    for n, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise ValueError(f"{LOG} title {n} is not an object.")
        try:
            start = _seconds(f"title {n} start", item.get("start"))
            end = _seconds(f"title {n} end", item.get("end"))
        except ValueError as e:
            raise ValueError(_own_message(e)) from None
        if end <= start:
            raise ValueError(f"{LOG} title {n}: the end ({end}) must be after the start ({start}).")
        titles.append({"start": start, "end": end, "style": {k: item[k] for k in STYLE_KEYS if k in item}})
    titles.sort(key=lambda t: (t["start"], t["end"]))
    for prev, cur in zip(titles, titles[1:]):
        if cur["start"] < prev["end"] - 1e-6:
            raise ValueError(
                f"{LOG} two titles overlap: {prev['start']} - {prev['end']} and {cur['start']} - {cur['end']}."
            )
    return titles


def _params(style, frame):
    """Cleaned Meme Maker parameters of one title: its own style + the video's frame settings."""
    try:
        return clean_params({**{k: style[k] for k in STYLE_KEYS if k in style}, **frame})
    except ValueError as e:
        raise ValueError(_own_message(e)) from None


# ---------------------------------------------------------------------------
# One canvas for the whole video
# ---------------------------------------------------------------------------

class _Canvas:
    """Size of the output picture and where the original sits in it."""

    def __init__(self, layouts, width, height):
        above = max([lay.img_y for lay in layouts] + [0])
        below = max([lay.total_h - lay.img_y - height for lay in layouts] + [0])
        left = max([lay.img_x for lay in layouts] + [0])
        self.width, self.height = width, height
        self.total_w = max([lay.total_w for lay in layouts] + [width])
        self.img_x, self.img_y = left, above
        self.total_h = above + height + below
        self.framed = self.total_h != height or self.total_w != width
        if self.framed and self.total_h % 2:  # video encoders want an even height
            self.total_h += 1


def _on_canvas(layout, canvas, frame_color):
    """The layout moved onto the common canvas (same attributes compose_frames reads)."""
    if not canvas.framed:
        return layout
    overlay = Image.new("RGBA", (canvas.total_w, canvas.total_h), tuple(frame_color) + (255,))
    ImageDraw.Draw(overlay).rectangle(
        [canvas.img_x, canvas.img_y, canvas.img_x + canvas.width - 1, canvas.img_y + canvas.height - 1],
        fill=(0, 0, 0, 0),
    )
    if layout is not None:
        # Same frame color everywhere, so the layout's own (smaller) bands simply
        # merge into the canvas; its picture area is transparent and stays a hole.
        moved = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
        moved.paste(layout.overlay, (canvas.img_x - layout.img_x, canvas.img_y - layout.img_y))
        overlay = Image.alpha_composite(overlay, moved)
    return types.SimpleNamespace(
        overlay=overlay, total_h=canvas.total_h, total_w=canvas.total_w, img_x=canvas.img_x, img_y=canvas.img_y,
        has_content=True,
    )


def _has_text(params):
    return bool(params["top_text"].strip() or params["bottom_text"].strip())


def _subtitle_params(style, text, position, font_size):
    raw = {k: style[k] for k in STYLE_KEYS if k in style}
    raw.update({"top_text": "", "bottom_text": "", "placement": "on image"})
    raw["top_text" if position == "top" else "bottom_text"] = text
    raw["top_font_size" if position == "top" else "bottom_font_size"] = font_size
    try:
        return clean_params(raw)
    except ValueError as e:
        raise ValueError(_own_message(e)) from None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _frame(seconds, fps, count):
    return max(0, min(count, int(math.floor(seconds * fps + 0.5))))


def render(images, fps, titles_params, cue_params, frame_color):
    """images (B, H, W, C) -> titled frames.

    titles_params / cue_params: [(first_frame, end_frame, params)] in output
    frames, in order, not overlapping within one list. Returns (frames, fonts)
    where fonts collects font warnings as (missing glyphs, font name, fallback).
    """
    count, height, width = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    warnings = set()

    def note(layout, wanted):
        if layout.fallback:
            warnings.add(f"Font '{wanted}' was not found on this system - using '{layout.font_name}'.")
        if layout.missing:
            warnings.add(f"The font '{layout.font_name}' has no glyph for: {''.join(layout.missing)}")

    layouts = []
    for _, _, params in titles_params:
        layout = build_layout(width, height, params)
        note(layout, params["font"])
        layouts.append(layout)
    canvas = _Canvas(layouts, width, height)
    placed = [_on_canvas(layout, canvas, frame_color) for layout in layouts]
    blank = _on_canvas(None, canvas, frame_color) if canvas.framed else None

    cue_layouts = []
    for _, _, params in cue_params:
        layout = build_layout(canvas.total_w, canvas.total_h, params)
        note(layout, params["font"])
        cue_layouts.append(layout)

    marks = {0, count}
    for first, end, _ in list(titles_params) + list(cue_params):
        marks.update((first, end))
    marks = sorted(m for m in marks if 0 <= m <= count)

    def active(spans, at):
        for index, (first, end, _) in enumerate(spans):
            if first <= at < end:
                return index
        return None

    parts = []
    for a, b in zip(marks, marks[1:]):
        if b <= a:
            continue
        frames = images[a:b]
        t = active(titles_params, a)
        layout = placed[t] if t is not None else blank
        if layout is not None and layout.has_content:
            frames = compose_frames(frames, layout)
        c = active(cue_params, a)
        if c is not None and cue_layouts[c].has_content:
            frames = compose_frames(frames, cue_layouts[c])
        parts.append(frames[..., :3] if frames.shape[-1] > 3 else frames)
    if not parts:
        return images, sorted(warnings)
    out = parts[0] if len(parts) == 1 else torch.cat([p.cpu() for p in parts], dim=0)
    return out, sorted(warnings)


def _trim_audio(audio, first, end, fps):
    try:
        if not audio or "waveform" not in audio:
            return audio
        rate = int(audio["sample_rate"])
        s0, s1 = int(round(first / fps * rate)), int(round(end / fps * rate))
        return {**audio, "waveform": audio["waveform"][..., s0:s1]}
    except Exception:
        return audio


# ---------------------------------------------------------------------------
# Files: the saved result and the player's source
# ---------------------------------------------------------------------------

def safe_name(name):
    """A bare file name (no folder, no reserved name), never empty."""
    text = "".join(ch for ch in str(name or "") if ch.isprintable())
    text = re.sub(r'[<>:"/\\|?*]+', "_", text).strip(" .")
    text = text[:MAX_NAME_LENGTH].strip(" .")
    if not text or text.lower().split(".")[0] in _RESERVED:
        text = f"{FALLBACK_NAME}_{text}".strip("_ .") if text else FALLBACK_NAME
    return text


def _free_name(folder, name, overwrite):
    """name, or name_0001 ... when a video / json of that name exists."""
    def taken(candidate):
        return any(os.path.exists(os.path.join(folder, candidate + ext)) for ext in (".mp4", ".json"))

    if overwrite or not taken(name):
        return name
    for n in range(1, 10000):
        candidate = f"{name}_{n:04d}"
        if not taken(candidate):
            return candidate
    raise RuntimeError(f"{LOG} no free file name left for '{name}' in the output folder.")


def _save_video(video, path):
    try:
        from comfy_api.latest import Types
        video.save_to(path, format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264)
    except ImportError:
        video.save_to(path)


def _inside(path, root):
    try:
        real, base = os.path.realpath(path), os.path.realpath(root)
        return os.path.commonpath([os.path.normcase(real), os.path.normcase(base)]) == os.path.normcase(base)
    except (ValueError, OSError):
        return False


def _view_ref(path):
    """{filename, subfolder, type} when path is a file ComfyUI's /view route serves, else None."""
    if folder_paths is None or not path or not os.path.isfile(path):
        return None
    for kind in VIEW_TYPES:
        root = folder_paths.get_directory_by_type(kind)
        if root and _inside(path, root):
            rel = os.path.relpath(os.path.realpath(path), os.path.realpath(root)).replace("\\", "/")
            return {"filename": os.path.basename(rel), "subfolder": os.path.dirname(rel), "type": kind}
    return None


def _probe_file(path):
    """Frame rate, size, length and codec of a video file (header only)."""
    with av.open(path, mode="r") as container:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            raise ValueError("no video stream")
        rate = stream.average_rate or stream.guessed_rate
        fps = float(Fraction(rate)) if rate else 0.0
        duration = float(container.duration / av.time_base) if container.duration else 0.0
        if not duration and stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        frames = int(stream.frames or 0) or (int(round(duration * fps)) if fps and duration else 0)
        return {
            "fps": round(fps, 6), "frames": frames, "duration": round(duration, 3),
            "width": int(stream.codec_context.width or 0), "height": int(stream.codec_context.height or 0),
            "codec": str(getattr(stream.codec_context, "name", "") or ""),
        }


_PREVIEW_SIGNATURES = {}


def _player_source(video, components, unique_id):
    """What the player on the node should load: the source file itself when
    the browser can play it as it is, otherwise a copy in the temp folder."""
    try:
        path = _file_source(video)
        if isinstance(path, str) and av is not None:
            window = getattr(video, "get_active_trim_window", None)
            trimmed = callable(window) and tuple(window()) != (0.0, 0.0)
            ref = None if trimmed else _view_ref(path)
            if ref is not None and _probe_file(path)["codec"] in BROWSER_CODECS:
                return ref
        if folder_paths is None:
            return None
        images = components.images
        safe_id = re.sub(r"[^0-9A-Za-z_-]", "_", str(unique_id or "node"))[:64]
        name = f"mykee_titler_{safe_id}.mp4"
        folder = folder_paths.get_temp_directory()
        os.makedirs(folder, exist_ok=True)
        target = os.path.join(folder, name)
        signature = (
            tuple(images.shape), float(components.frame_rate),
            float(images[0].float().mean()), float(images[-1].float().mean()),
        )
        if _PREVIEW_SIGNATURES.get(safe_id) != signature or not os.path.isfile(target):
            _save_video(_make_video(components, images), target)
            _PREVIEW_SIGNATURES[safe_id] = signature
        return {"filename": name, "subfolder": "", "type": "temp"}
    except Exception as e:
        print(f"{LOG} the player's preview copy could not be written: {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def _caption_inputs():
    """The Meme Maker's caption settings (kept in sync with it), without its
    preview-size fields, and with empty texts: here a text is a title."""
    types_ = MykeeMemeMaker.INPUT_TYPES()
    required = dict(types_["required"])
    for name in ("preview_width", "preview_height"):
        required.pop(name, None)
    # caption settings the Meme Maker declares as optional (added later) belong here too
    for name, spec in types_.get("optional", {}).items():
        if name in DEFAULTS:
            required[name] = spec
    for name, label in (("top_text", "top"), ("bottom_text", "bottom")):
        kind, options = required[name]
        required[name] = (kind, {**options, "default": "", "tooltip": f"Caption at the {label} of the title being edited."})
    return required


class MykeeVideoTitler:
    @classmethod
    def INPUT_TYPES(cls):
        required = {"video": ("VIDEO", {"tooltip": "The video to put titles on."})}
        required.update(_caption_inputs())
        required.update({
            "titles": ("STRING", {"default": "[]", "tooltip": "The list of titles (edited on the Titles tab)."}),
            "text_mode": (list(TEXT_MODES), {"default": TEXT_MODES[0], "tooltip": "What happens to the text input. import to list: nothing is drawn from it; use the Import button to turn it into titles. subtitle layer: its cues are drawn as they are, above the titles."}),
            "subtitle_font_size": ("FLOAT", {"default": 4.5, "min": 0.5, "max": 60.0, "step": 0.1, "tooltip": "Font size of the subtitle layer in % of the picture width."}),
            "subtitle_position": (list(SUBTITLE_POSITIONS), {"default": "bottom", "tooltip": "Where the subtitle layer is drawn."}),
            "trim_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": MAX_TIME, "step": 0.001, "tooltip": "Cut: the output starts here (seconds of the source video)."}),
            "trim_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": MAX_TIME, "step": 0.001, "tooltip": "Cut: the output ends here (seconds of the source video). 0 = until the end."}),
            "filename_tag": ("STRING", {"default": "", "multiline": False, "tooltip": "Appended to the video's file name: file \"video\" + tag \"_titled\" gives \"video_titled\"."}),
            "save_output": ("BOOLEAN", {"default": True, "tooltip": "Write the titled video and the timing list (JSON) into ComfyUI's output folder, with the same name."}),
            "overwrite": ("BOOLEAN", {"default": False, "tooltip": "Off: an existing file is kept and a counter is added to the new name. On: files of the same name are replaced."}),
        })
        return {
            "required": required,
            "optional": {
                "text": ("STRING", {"forceInput": True, "tooltip": "Optional: the text of an SRT / WebVTT subtitle file, or plain text (one title per paragraph or line)."}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "filename")
    OUTPUT_TOOLTIPS = (
        "The titled (and cut) video.",
        "File name of the video without folder and extension, with filename_tag appended - the name the node saves under.",
    )
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Mykee/Image"
    DESCRIPTION = (
        "Timed top / bottom titles on a video (Meme Maker captions with start and end times), "
        "with a player, cutting, subtitle import and a JSON timing list saved next to the video."
    )

    def run(self, video, titles="[]", text_mode=TEXT_MODES[0], subtitle_font_size=4.5, subtitle_position="bottom",
            trim_start=0.0, trim_end=0.0, filename_tag="", save_output=True, overwrite=False,
            text=None, prompt=None, unique_id=None, **caption):
        if video is None:
            raise ValueError(f"{LOG} Connect a video.")
        try:
            form = clean_params({k: caption[k] for k in DEFAULTS if k in caption})
        except ValueError as e:
            raise ValueError(_own_message(e)) from None
        frame = {k: caption[k] for k in FRAME_KEYS if k in caption}
        form_style = {k: caption[k] for k in STYLE_KEYS if k in caption}
        if text_mode not in TEXT_MODES:
            raise ValueError(f"{LOG} text_mode: must be one of {', '.join(TEXT_MODES)}.")
        if subtitle_position not in SUBTITLE_POSITIONS:
            raise ValueError(f"{LOG} subtitle_position: must be one of {', '.join(SUBTITLE_POSITIONS)}.")
        entries = parse_titles(titles)

        components = video.get_components()
        images = components.images
        fps = float(Fraction(components.frame_rate))
        if not fps > 0:
            raise ValueError(f"{LOG} the video has no usable frame rate.")
        source_count = int(images.shape[0])
        height, width = int(images.shape[1]), int(images.shape[2])
        source_duration = source_count / fps

        # ---- cut ------------------------------------------------------------
        first = _frame(_seconds("trim_start", trim_start), fps, source_count)
        end_s = _seconds("trim_end", trim_end)
        last = _frame(end_s, fps, source_count) if end_s > 0 else source_count
        if last <= first:
            raise ValueError(
                f"{LOG} nothing is left of the video: trim_start ({trim_start}) is not before "
                f"trim_end ({trim_end}) / the end of the video ({round(source_duration, 3)} s)."
            )
        cut = images[first:last]
        count = int(cut.shape[0])

        def span(start, end):
            a, b = _frame(start, fps, source_count) - first, _frame(end, fps, source_count) - first
            return max(0, min(count, a)), max(0, min(count, b))

        # ---- titles ---------------------------------------------------------
        title_spans, saved_titles = [], []
        if entries:
            for entry in entries:
                params = _params(entry["style"], frame)
                a, b = span(entry["start"], entry["end"])
                saved_titles.append((entry, params, a, b))
                if b > a and (_has_text(params) or params["placement"] == "frame"):
                    title_spans.append((a, b, params))
        elif _has_text(form) or form["placement"] == "frame":
            # no list: the captions on the node cover the whole video
            title_spans.append((0, count, form))
        if not title_spans and form["placement"] == "frame" and form["frame_width"] > 0:
            title_spans.append((0, count, _params({"top_text": "", "bottom_text": ""}, frame)))

        # ---- text input -----------------------------------------------------
        has_text = isinstance(text, str) and bool(text.strip())
        project = parse_project(text) if has_text else None
        if project is not None:
            # a JSON saved by this node: its titles are loaded with the Import button,
            # only its subtitle cues (if it has any) can be drawn directly
            cues, kind = project["cues"], "project"
        else:
            cues, kind = parse_text(text, source_duration) if has_text else ([], "")
        cue_spans = []
        if cues and text_mode == "subtitle layer":
            for cue in cues:
                a, b = span(cue["start"], cue["end"])
                if b > a:
                    cue_spans.append((a, b, _subtitle_params(form_style, cue["text"], subtitle_position, subtitle_font_size)))

        out, warnings = render(cut, fps, title_spans, cue_spans, form["frame_color"])
        for line in warnings:
            print(f"{LOG} {line}")

        parts = types.SimpleNamespace(
            images=out, audio=_trim_audio(components.audio, first, last, fps),
            frame_rate=components.frame_rate, metadata=getattr(components, "metadata", None),
        )
        video_out = _make_video(parts, out)

        # ---- name -----------------------------------------------------------
        base = None
        path = _file_source(video)
        if isinstance(path, str):
            base = _base_name(path) or None
        base = base or find_upstream_name(prompt, unique_id, "video") or FALLBACK_NAME
        filename = safe_name(base + _clean_tag(filename_tag))

        # ---- save -----------------------------------------------------------
        saved = None
        if save_output:
            if folder_paths is None:
                raise RuntimeError(f"{LOG} ComfyUI's output folder is not available.")
            folder = folder_paths.get_output_directory()
            os.makedirs(folder, exist_ok=True)
            name = _free_name(folder, filename, bool(overwrite))
            video_path = os.path.join(folder, name + ".mp4")
            json_path = os.path.join(folder, name + ".json")
            if not (_inside(video_path, folder) and _inside(json_path, folder)):
                raise RuntimeError(f"{LOG} refusing to write outside the output folder.")
            _save_video(video_out, video_path)
            document = {
                "version": 1,
                "generator": "Mykee Video Titler",
                "video": {
                    "file": name + ".mp4", "source": base,
                    "width": int(out.shape[2]), "height": int(out.shape[1]),
                    "frame_rate": round(fps, 6), "frame_count": count, "duration": round(count / fps, 3),
                },
                "trim": {
                    "start": round(first / fps, 6), "end": round(last / fps, 6),
                    "start_frame": first, "end_frame": last, "until_end": last == source_count,
                },
                "frame": {k: (_hex(form[k]) if k in _COLOR_KEYS else form[k]) for k in FRAME_KEYS},
                "titles": [
                    {
                        "start": entry["start"], "end": entry["end"],
                        "start_frame": _frame(entry["start"], fps, source_count),
                        "end_frame": _frame(entry["end"], fps, source_count),
                        "output_start": round(a / fps, 6), "output_end": round(b / fps, 6),
                        **{k: (_hex(params[k]) if k in _COLOR_KEYS else params[k]) for k in STYLE_KEYS},
                    }
                    for entry, params, a, b in saved_titles
                ],
            }
            if not entries and title_spans:
                document["whole_video_caption"] = {
                    k: (_hex(form[k]) if k in _COLOR_KEYS else form[k]) for k in STYLE_KEYS
                }
            if cue_spans:
                document["subtitles"] = {
                    "position": subtitle_position, "font_size": float(subtitle_font_size), "cues": cues,
                }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(document, f, ensure_ascii=False, indent=2)
            saved = {"video": name + ".mp4", "json": name + ".json"}
            print(f"{LOG} saved {name}.mp4 and {name}.json to the output folder.")

        ui = {"mykee_titler": [{
            "width": width, "height": height, "fps": round(fps, 6), "frames": source_count,
            "duration": round(source_duration, 3),
            "source": _player_source(video, components, unique_id),
            "cues": cues, "cue_kind": kind, "project": project,
            "filename": filename, "saved": saved, "warnings": warnings,
            "out_frames": count,
        }]}
        return {"ui": ui, "result": (video_out, filename)}


NODE_CLASS_MAPPINGS = {NODE_NAME: MykeeVideoTitler}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_NAME: "Mykee Video Titler"}


# ---------------------------------------------------------------------------
# Preview (the caption layer of one title on the video's common canvas)
# ---------------------------------------------------------------------------

_GEOMETRY_CACHE = OrderedDict()   # (params json, width, height) -> (img_x, img_y, total_w, total_h)
_GEOMETRY_CACHE_MAX = 4096


def _geometry(params, width, height):
    key = (json.dumps(params, sort_keys=True, default=str), width, height)
    hit = _GEOMETRY_CACHE.get(key)
    if hit is None:
        layout = build_layout(width, height, params)
        hit = types.SimpleNamespace(
            img_x=layout.img_x, img_y=layout.img_y, total_w=layout.total_w, total_h=layout.total_h)
        _GEOMETRY_CACHE[key] = hit
        while len(_GEOMETRY_CACHE) > _GEOMETRY_CACHE_MAX:
            _GEOMETRY_CACHE.popitem(last=False)
    else:
        _GEOMETRY_CACHE.move_to_end(key)
    return hit


def render_preview(data):
    """The caption layer the player lays over the video, for one title.

    data: {width, height, frame: {...}, titles: [style, ...], show: style | None,
           subtitle: {text, position, font_size, style} | None}
    Every title only decides the common canvas; "show" is the one drawn.
    """
    if not isinstance(data, dict):
        raise ValueError("invalid request")
    width = int(_number("width", data.get("width", 1024), 16, 16384, integer=True))
    height = int(_number("height", data.get("height", 576), 16, 16384, integer=True))
    scale = min(1.0, PREVIEW_SIDE / max(width, height))
    width = max(PREVIEW_MIN_SIDE, int(round(width * scale)))
    height = max(PREVIEW_MIN_SIDE, int(round(height * scale)))
    frame = data.get("frame") if isinstance(data.get("frame"), dict) else {}
    frame = {k: frame[k] for k in FRAME_KEYS if k in frame}
    styles = data.get("titles") if isinstance(data.get("titles"), list) else []
    if len(styles) > MAX_TITLES:
        raise ValueError(f"too many titles (at most {MAX_TITLES})")
    show = data.get("show") if isinstance(data.get("show"), dict) else None

    geometries = [_geometry(_params(s, frame), width, height) for s in styles if isinstance(s, dict)]
    layout = None
    missing, font, fallback, sizes = "", "", False, [None, None]
    if show is not None:
        params = _params(show, frame)
        layout = build_layout(width, height, params)
        geometries.append(layout)
        missing, font, fallback, sizes = "".join(layout.missing), layout.font_name, layout.fallback, list(layout.sizes)
    elif frame:
        geometries.append(_geometry(_params({"top_text": "", "bottom_text": ""}, frame), width, height))
    canvas = _Canvas(geometries, width, height)
    frame_color = _params({}, frame)["frame_color"]
    if canvas.framed:
        overlay = _on_canvas(layout, canvas, frame_color).overlay
    elif layout is not None:
        overlay = layout.overlay
    else:
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    subtitle = data.get("subtitle") if isinstance(data.get("subtitle"), dict) else None
    style = subtitle.get("style") if subtitle is not None else None
    if subtitle is not None and isinstance(style, dict):
        position = subtitle.get("position")
        if position not in SUBTITLE_POSITIONS:
            raise ValueError("subtitle position: must be bottom or top")
        size = _number("subtitle font size", subtitle.get("font_size", 4.5), 0.5, 60.0)
        cue = build_layout(canvas.total_w, canvas.total_h, _subtitle_params(style, subtitle.get("text"), position, size))
        if cue.has_content:
            overlay = Image.alpha_composite(overlay.convert("RGBA"), cue.overlay)

    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG", compress_level=3)
    return {
        "png": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "width": width, "height": height, "total_w": canvas.total_w, "total_h": canvas.total_h,
        "img_x": canvas.img_x, "img_y": canvas.img_y,
        "missing": missing, "font": font, "fallback": fallback, "sizes": sizes,
    }


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

def resolve_view_file(filename, subfolder, kind):
    """Path of a video inside ComfyUI's input / output / temp folder, or raises ValueError."""
    if folder_paths is None or kind not in VIEW_TYPES:
        raise ValueError("unknown folder type")
    if not isinstance(filename, str) or not isinstance(subfolder, str):
        raise ValueError("invalid file name")
    if not filename or len(filename) > 512 or len(subfolder) > 1024 or "\x00" in filename + subfolder:
        raise ValueError("invalid file name")
    if os.path.basename(filename.replace("\\", "/")) != filename:
        raise ValueError("invalid file name")
    if os.path.splitext(filename)[1].lstrip(".").lower() not in VIDEO_EXTS:
        raise ValueError("not a video file")
    root = folder_paths.get_directory_by_type(kind)
    if not root:
        raise ValueError("unknown folder type")
    path = os.path.join(root, subfolder, filename)
    if not _inside(path, root) or not os.path.isfile(path):
        raise ValueError("file not found")
    return os.path.realpath(path)


try:
    if PromptServer is None:
        raise RuntimeError("no server")
    _routes = PromptServer.instance.routes

    def _error(message, status):
        return _aiohttp_web.json_response({"error": message}, status=status)

    async def _json_body(request, limit):
        if request.content_length is not None and request.content_length > limit:
            return None, _error("the request is too large", 413)
        try:
            raw = await request.content.read(limit + 1)
            if len(raw) > limit:
                return None, _error("the request is too large", 413)
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            return None, _error("invalid JSON body", 400)
        if not isinstance(data, dict):
            return None, _error("invalid request", 400)
        return data, None

    @_routes.post("/mykee/titler/probe")
    async def _mykee_titler_probe(request):
        data, error = await _json_body(request, 8 * 1024)
        if error is not None:
            return error
        if av is None:
            return _error("PyAV is not available", 500)
        try:
            path = resolve_view_file(data.get("filename"), data.get("subfolder", ""), data.get("type", "input"))
        except ValueError as e:
            return _error(str(e), 400)
        loop = asyncio.get_running_loop()
        try:
            info = await loop.run_in_executor(None, _probe_file, path)
        except Exception:
            return _error("the video could not be read", 400)
        info["playable"] = info["codec"] in BROWSER_CODECS
        return _aiohttp_web.json_response(info)

    @_routes.post("/mykee/titler/parse")
    async def _mykee_titler_parse(request):
        data, error = await _json_body(request, MAX_SOURCE_TEXT * 4 + 1024)
        if error is not None:
            return error
        text = data.get("text")
        if not isinstance(text, str):
            return _error("invalid request", 400)
        loop = asyncio.get_running_loop()
        try:
            project = await loop.run_in_executor(None, parse_project, text)
            if project is not None:
                return _aiohttp_web.json_response({"cues": project["cues"], "kind": "project", "project": project})
            cues, kind = await loop.run_in_executor(None, parse_text, text, data.get("duration", 0))
        except Exception:
            return _error("the text could not be read", 400)
        return _aiohttp_web.json_response({"cues": cues, "kind": kind})

    @_routes.post("/mykee/titler/preview")
    async def _mykee_titler_preview(request):
        data, error = await _json_body(request, MAX_TITLES_JSON + 64 * 1024)
        if error is not None:
            return error
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, render_preview, data)
        except ValueError as e:
            return _error(str(e), 400)
        except Exception:
            return _error("the preview could not be rendered", 500)
        return _aiohttp_web.json_response(result)

except Exception:
    # No running PromptServer (module imported outside ComfyUI): the node
    # still works, the player just has nothing to call.
    pass
