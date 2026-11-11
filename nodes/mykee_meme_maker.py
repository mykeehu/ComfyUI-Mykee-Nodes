"""
ComfyUI-Mykee-Nodes / Mykee Meme Maker

Puts a top and / or bottom caption on an IMAGE, an image batch or a VIDEO -
either on the picture itself or in a frame (a colored band) outside of it.
The caption is rendered once (Pillow) into a transparent layer and that layer
is composited onto every frame, so a video costs hardly more than an image.

Everything is relative to the picture, so the same settings look the same at
any resolution:

  *_font_size    % of the image width (top and bottom caption separately)
  stroke_width   % of the font size
  padding        % of the image width
  frame_height   % of the image width (fixed frame mode)
  frame_width    % of the image width: a border all around the picture (frame mode, 0 = none)
  top_margin     % of the image HEIGHT: extra distance of the top caption from the top edge
  bottom_margin  % of the image HEIGHT: extra distance of the bottom caption from the bottom edge
                 (a vertical distance, so 0 - 100 spans the whole picture; in frame
                 placement it is empty room at the outer edge of the band)

Wrapping
--------
Lines are wrapped to the width inside the padding (a word that is wider than a
line is broken). Explicit line breaks in the text are kept.

  lines = 0   automatic: as many lines as needed, the font size is not touched
  lines = N   at most N lines: if the text needs more, the font is shrunk
              until it fits

In a frame of fixed height the text is shrunk until it fits the frame in both
modes (the frame can't grow). On the picture itself the text is never shrunk
for height - it is placed at the top / bottom padding and may run over the
edge if there is too much of it.

Fonts
-----
Only the fonts installed on the system are offered (Windows: the Windows
Fonts folders; macOS and Linux: the usual system font folders). The folders are
fixed - no folder or file name ever comes from the browser or the workflow;
the workflow only holds a font family name, which is looked up in the list
built from those folders. A missing font falls back to a default one (the
console says so). Bold / italic use the font family's own bold / italic file
when there is one and are synthesized otherwise.

Preview
-------
preview_width / preview_height (default 1024 x 1024, always shown above the tabs,
can be connected) are the shape of the node's preview until the node has run;
after a run the preview follows the real picture, unless one of the two fields
is edited. They never reach the result. The preview route returns just the
caption layer as a transparent PNG; the browser lays it over a gray picture, or
over a thumbnail of the first input frame that the node sends back after a run.

Security
--------
The two HTTP routes (the font list and the preview) take no path at all. The
preview request is validated like any untrusted input: text length, number
ranges, hex colors, picture size, are all limited, and an invalid value is
refused with a message.
"""

import asyncio
import base64
import functools
import io
import math
import os
import re
import sys
import threading

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

try:
    from comfy_execution.graph import ExecutionBlocker
except Exception:  # older ComfyUI
    ExecutionBlocker = None

try:
    from server import PromptServer
    from aiohttp import web as _aiohttp_web
except Exception:  # imported outside ComfyUI
    PromptServer = None
    _aiohttp_web = None


NODE_NAME = "MykeeMemeMaker"

MAX_TEXT = 1000
MAX_FONT_NAME = 120
MAX_LINES = 20
MIN_PX = 6                      # smallest font size the shrinking goes down to
MAX_FONT_PX = 4000
MAX_FONTS = 6000
MAX_FONT_FILE_BYTES = 200 * 1024 * 1024
MAX_FACES_PER_FILE = 32
PREVIEW_MAX_SIDE = 1024
PREVIEW_MIN_SIDE = 32
CHUNK = 16                      # frames composited at a time
THUMB_SIDE = 384                # longest side of the preview background sent back after a run

FONT_EXTS = (".ttf", ".otf", ".ttc", ".otc")
PLACEMENTS = ("on image", "frame")
FRAME_HEIGHT_MODES = ("fit text", "fixed")
FRAME_WIDTH_MODES = ("around image", "around meme")  # the border hugs the picture / the whole captioned meme
ALIGNS = ("left", "center", "right")
DEFAULT_FONTS = ("Impact", "Arial Black", "Arial", "DejaVu Sans", "Liberation Sans")

_LOCK = threading.RLock()       # FreeType faces are shared (cached) - render one at a time


# ---------------------------------------------------------------------------
# Parameters (shared by the node and the preview route)
# ---------------------------------------------------------------------------

DEFAULTS = {
    "top_text": "", "bottom_text": "", "font": "", "top_font_size": 9.0, "bottom_font_size": 9.0,
    "bold": False, "italic": False, "uppercase": True,
    "text_color": "#FFFFFF", "stroke_color": "#000000", "stroke_width": 8.0,
    "line_height": 1.0, "padding": 2.0,
    "placement": "on image", "frame_color": "#FFFFFF", "frame_width": 0.0, "frame_width_mode": "around image",
    "frame_height_mode": "fit text", "frame_height": 15.0,
    "top_align": "center", "top_lines": 0, "bottom_align": "center", "bottom_lines": 0,
    "top_margin": 0.0, "bottom_margin": 0.0,
}


def _hex_color(name, value):
    if isinstance(value, tuple) and len(value) == 3 and all(isinstance(c, int) and 0 <= c <= 255 for c in value):
        return value  # already cleaned (cleaning twice is harmless)
    s = str(value).strip().lstrip("#")
    if re.fullmatch(r"[0-9a-fA-F]{3}", s):
        s = "".join(c * 2 for c in s)
    if not re.fullmatch(r"[0-9a-fA-F]{6}", s):
        raise ValueError(f"[Mykee Meme Maker] {name}: '{str(value)[:20]}' is not a hex color (use #RRGGBB).")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def _number(name, value, lo, hi, integer=False):
    if isinstance(value, bool):
        raise ValueError(f"[Mykee Meme Maker] {name}: not a number.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"[Mykee Meme Maker] {name}: not a number.") from None
    if not math.isfinite(number):
        raise ValueError(f"[Mykee Meme Maker] {name}: not a number.")
    number = min(max(number, lo), hi)
    return int(round(number)) if integer else number


def _choice(name, value, options):
    value = str(value)
    if value not in options:
        raise ValueError(f"[Mykee Meme Maker] {name}: must be one of {', '.join(options)}.")
    return value


def _text(value):
    text = str(value if value is not None else "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch == "\n" or ch >= " " and ch != "\x7f")
    return text[:MAX_TEXT]


def clean_params(raw):
    """Validated, clamped parameters (a dict like DEFAULTS, colors as RGB
    tuples). Raises ValueError with a readable message on an invalid value."""
    raw = {**DEFAULTS, **(raw if isinstance(raw, dict) else {})}
    return {
        "top_text": _text(raw["top_text"]),
        "bottom_text": _text(raw["bottom_text"]),
        "font": str(raw["font"] or "")[:MAX_FONT_NAME],
        "top_font_size": _number("top_font_size", raw["top_font_size"], 0.5, 60.0),
        "bottom_font_size": _number("bottom_font_size", raw["bottom_font_size"], 0.5, 60.0),
        "bold": bool(raw["bold"]), "italic": bool(raw["italic"]), "uppercase": bool(raw["uppercase"]),
        "text_color": _hex_color("text_color", raw["text_color"]),
        "stroke_color": _hex_color("stroke_color", raw["stroke_color"]),
        "stroke_width": _number("stroke_width", raw["stroke_width"], 0.0, 40.0),
        "line_height": _number("line_height", raw["line_height"], 0.3, 4.0),
        "padding": _number("padding", raw["padding"], 0.0, 40.0),
        "placement": _choice("placement", raw["placement"], PLACEMENTS),
        "frame_color": _hex_color("frame_color", raw["frame_color"]),
        "frame_height_mode": _choice("frame_height_mode", raw["frame_height_mode"], FRAME_HEIGHT_MODES),
        "frame_height": _number("frame_height", raw["frame_height"], 1.0, 200.0),
        "frame_width": _number("frame_width", raw["frame_width"], 0.0, 40.0),
        "frame_width_mode": _choice("frame_width_mode", raw["frame_width_mode"], FRAME_WIDTH_MODES),
        "top_align": _choice("top_align", raw["top_align"], ALIGNS),
        "top_lines": _number("top_lines", raw["top_lines"], 0, MAX_LINES, integer=True),
        "bottom_align": _choice("bottom_align", raw["bottom_align"], ALIGNS),
        "bottom_lines": _number("bottom_lines", raw["bottom_lines"], 0, MAX_LINES, integer=True),
        "top_margin": _number("top_margin", raw["top_margin"], 0.0, 100.0),
        "bottom_margin": _number("bottom_margin", raw["bottom_margin"], 0.0, 100.0),
    }


# ---------------------------------------------------------------------------
# System fonts (fixed folders only)
# ---------------------------------------------------------------------------

def _font_roots():
    roots = []
    home = os.path.expanduser("~")
    if sys.platform.startswith("win"):
        windir = os.environ.get("WINDIR") or os.environ.get("SystemRoot") or r"C:\Windows"
        roots.append(os.path.join(windir, "Fonts"))
        local = os.environ.get("LOCALAPPDATA")
        if local:
            roots.append(os.path.join(local, "Microsoft", "Windows", "Fonts"))
    elif sys.platform == "darwin":
        roots += ["/System/Library/Fonts", "/Library/Fonts", os.path.join(home, "Library", "Fonts")]
    else:
        roots += ["/usr/share/fonts", "/usr/local/share/fonts", os.path.join(home, ".fonts"),
                  os.path.join(home, ".local", "share", "fonts")]
    found = []
    for root in roots:
        try:
            if os.path.isdir(root):
                found.append(os.path.realpath(root))
        except OSError:
            continue
    return found


def _inside(path, root):
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


_PLAIN_STYLE = {"regular", "normal", "book", "roman", "standard"}


def _split_style(family, style):
    """-> (family name, 'regular' | 'bold' | 'italic' | 'bolditalic').
    Other style words (Light, Black, Condensed ...) become part of the family
    name, so "Segoe UI Light" is its own entry."""
    bold = italic = False
    rest = []
    for token in str(style or "").replace("-", " ").split():
        low = token.lower()
        if low == "bold":
            bold = True
        elif low in ("italic", "oblique"):
            italic = True
        elif low in _PLAIN_STYLE:
            continue
        else:
            rest.append(token)
    name = " ".join([str(family or "").strip()] + rest).strip()
    return name, ("bold" if bold else "") + ("italic" if italic else "") or "regular"


_INDEX = None
_INDEX_LOCK = threading.Lock()


def _scan_fonts():
    index = {}
    count = 0
    for root in _font_roots():
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames.sort()
            for filename in sorted(filenames):
                if count >= MAX_FONTS:
                    return index
                if not filename.lower().endswith(FONT_EXTS):
                    continue
                path = os.path.join(dirpath, filename)
                try:
                    real = os.path.realpath(path)
                    if not _inside(real, root) or not os.path.isfile(real) or os.path.getsize(real) > MAX_FONT_FILE_BYTES:
                        continue
                except OSError:
                    continue
                for face in range(MAX_FACES_PER_FILE):
                    try:
                        font = ImageFont.truetype(real, 12, index=face)
                        family, style = font.getname()
                    except Exception:
                        break
                    name, key = _split_style(family, style)
                    if not name:
                        continue
                    entry = index.setdefault(name.lower(), {"name": name, "styles": {}})
                    entry["styles"].setdefault(key, (real, face))
                    count += 1
    return index


def font_index(refresh=False):
    global _INDEX
    with _INDEX_LOCK:
        if _INDEX is None or refresh:
            _INDEX = _scan_fonts()
            _font.cache_clear()
        return _INDEX


def font_names():
    return sorted((e["name"] for e in font_index().values()), key=str.lower)


def default_font_name():
    index = font_index()
    for name in DEFAULT_FONTS:
        if name.lower() in index:
            return index[name.lower()]["name"]
    names = font_names()
    return names[0] if names else ""


class FontRef:
    def __init__(self, name, path, face, synth_bold, synth_italic, fallback):
        self.name, self.path, self.face = name, path, face
        self.synth_bold, self.synth_italic, self.fallback = synth_bold, synth_italic, fallback


def resolve_font(name, bold, italic):
    """The font for a family name and bold / italic flags. Never takes a path:
    the name is only looked up in the index built from the system folders."""
    index = font_index()
    entry = index.get(str(name or "").strip().lower())
    fallback = False
    if entry is None:
        fallback = bool(str(name or "").strip())
        entry = index.get(default_font_name().lower()) if default_font_name() else None
    if entry is None:
        return FontRef("(built-in)", None, 0, bold, italic, fallback)
    styles = entry["styles"]
    want = ("bold" if bold else "") + ("italic" if italic else "") or "regular"
    order = [want]
    if bold and italic:
        order += ["bold", "italic"]
    order += ["regular", "bold", "italic", "bolditalic"]
    for key in order:
        if key in styles:
            path, face = styles[key]
            return FontRef(entry["name"], path, face, bold and "bold" not in key, italic and "italic" not in key, fallback)
    return FontRef("(built-in)", None, 0, bold, italic, fallback)


@functools.lru_cache(maxsize=256)
def _font(path, face, px):
    if path is None:
        try:
            return ImageFont.load_default(size=px)  # Pillow >= 10.1
        except TypeError:
            return ImageFont.load_default()
    return ImageFont.truetype(path, px, index=face)


METER_PX = 512  # text is measured at this size and scaled, so wrapping does not depend on the resolution


class _Meter:
    """Text widths scaled from a large reference size. Hinting rounds the glyph
    advances differently at every pixel size, which could wrap a line in the full
    size picture that fitted in the (smaller) preview - measuring at one fixed
    size and scaling makes the line breaks the same at every resolution."""
    def __init__(self, ref, px):
        self._ref, self._scale = _font(ref.path, ref.face, METER_PX), px / float(METER_PX)

    def getlength(self, text):
        return self._ref.getlength(text) * self._scale


def missing_chars(ref, text):
    """Characters of `text` the font has no glyph for (it draws its 'missing
    glyph' box or nothing). A heuristic, used only to warn."""
    chars = sorted({c for c in text if c.strip() and ord(c) > 127})
    if not chars:
        return []
    font = _font(ref.path, ref.face, 24)

    def bitmap(ch):
        im = Image.new("L", (56, 56), 0)
        ImageDraw.Draw(im).text((6, 6), ch, font=font, fill=255)
        return im.tobytes()

    reference = bitmap("\uffff")
    return [c for c in chars if bitmap(c) == reference]


# ---------------------------------------------------------------------------
# Layout and rendering
# ---------------------------------------------------------------------------

def _tokens(paragraph, font, max_w):
    out = []
    for word in paragraph.split():
        if font.getlength(word) <= max_w:
            out.append((word, True))
            continue
        chunk, first = "", True
        for ch in word:  # a word wider than a line is broken
            if chunk and font.getlength(chunk + ch) > max_w:
                out.append((chunk, first))
                first, chunk = False, ""
            chunk += ch
        if chunk:
            out.append((chunk, first))
    return out


def wrap_lines(text, font, max_w):
    """Greedy wrap; explicit line breaks are kept (an empty line stays empty)."""
    lines = []
    for paragraph in text.split("\n"):
        tokens = _tokens(paragraph, font, max_w)
        if not tokens:
            lines.append("")
            continue
        current = ""
        for token, spaced in tokens:
            trial = token if not current else current + (" " if spaced else "") + token
            if current and font.getlength(trial) > max_w:
                lines.append(current)
                current = token
            else:
                current = trial
        lines.append(current)
    while lines and lines[-1] == "":
        lines.pop()
    while lines and lines[0] == "":
        lines.pop(0)
    return lines


class Block:
    """One laid-out caption: its lines, font size and pixel height."""
    def __init__(self, lines, px, font, stroke, extra, pitch, height, ascent, meter=None):
        self.lines, self.px, self.font = lines, px, font
        self.meter = meter or font  # measures the lines (scale-independent)
        self.stroke, self.extra, self.pitch, self.height, self.ascent = stroke, extra, pitch, height, ascent


def layout_block(text, params, ref, width, pad, max_lines, max_h, size_pct, unit=None):
    """Wraps `text` and shrinks the font while it does not fit (more lines than
    allowed, or taller than max_h). Returns None for an empty text."""
    if not text.strip():
        return None
    size = min(size_pct / 100.0 * (unit or width), MAX_FONT_PX)  # % of the image width; the text wraps to `width`
    while True:
        px = int(max(MIN_PX, min(MAX_FONT_PX, round(size))))
        font = _font(ref.path, ref.face, px)
        stroke = int(round(px * params["stroke_width"] / 100.0))
        extra = max(1, int(round(px / 28))) if ref.synth_bold else 0
        edge = stroke + extra
        # the wrap decision uses unrounded sizes, so it is the same at every resolution
        exact = max(MIN_PX, min(MAX_FONT_PX, size))
        edge_exact = exact * params["stroke_width"] / 100.0 + (exact / 28.0 if ref.synth_bold else 0)
        avail = max(8, width - 2 * (params["padding"] / 100.0 * (unit or width)) - 2 * edge_exact)
        meter = _Meter(ref, exact)
        lines = wrap_lines(text, meter, avail)
        if not lines:
            return None
        ascent, descent = font.getmetrics()
        pitch = (ascent + descent) * params["line_height"]
        height = int(math.ceil((len(lines) - 1) * pitch + ascent + descent + 2 * edge))
        fits = (not max_lines or len(lines) <= max_lines) and (max_h is None or height <= max_h)
        if fits or px <= MIN_PX:
            return Block(lines, px, font, stroke, extra, pitch, height, ascent, meter)
        size = min(size * 0.96, px - 1)


def _render_block(block, params, ref, width, align, pad):
    """RGBA layer (width x block height) with the caption's lines."""
    edge = block.stroke + block.extra
    outline = Image.new("L", (width, block.height), 0)
    fill = Image.new("L", (width, block.height), 0)
    d_outline, d_fill = ImageDraw.Draw(outline), ImageDraw.Draw(fill)
    for i, line in enumerate(block.lines):
        if not line:
            continue
        line_w = block.meter.getlength(line)
        if align == "left":
            x = pad + edge
        elif align == "right":
            x = width - pad - edge - line_w
        else:
            x = (width - line_w) / 2.0
        y = edge + block.ascent + i * block.pitch
        d_outline.text((x, y), line, font=block.font, fill=255, anchor="ls", stroke_width=block.stroke + block.extra)
        d_fill.text((x, y), line, font=block.font, fill=255, anchor="ls", stroke_width=block.extra)
    if ref.synth_italic:
        shear = 0.2
        coeffs = (1, shear, -shear * block.height, 0, 1, 0)
        outline = outline.transform(outline.size, Image.AFFINE, coeffs, resample=Image.BICUBIC)
        fill = fill.transform(fill.size, Image.AFFINE, coeffs, resample=Image.BICUBIC)
    layer = Image.new("RGBA", (width, block.height), params["stroke_color"] + (0,))
    layer.putalpha(outline)
    top = Image.new("RGBA", (width, block.height), params["text_color"] + (0,))
    top.putalpha(fill)
    return Image.alpha_composite(layer, top)


class Layout:
    """The finished caption: a transparent RGBA layer the size of the output
    picture (with the opaque frame bands in frame mode)."""
    def __init__(self, overlay, top_band, total_h, missing, font_name, fallback, has_content, sizes=(None, None),
                 total_w=None, img_x=0, img_y=None):
        self.total_w = overlay.size[0] if total_w is None else total_w  # the output width (picture + side borders)
        self.img_x = img_x  # where the original picture sits in the output
        self.img_y = top_band if img_y is None else img_y
        self.sizes = sizes  # the font size really used (% of the image width) per caption - smaller than set when shrunk
        self.overlay, self.top_band, self.total_h = overlay, top_band, total_h
        self.missing, self.font_name, self.fallback, self.has_content = missing, font_name, fallback, has_content


def _prepared_text(text, params):
    return text.upper() if params["uppercase"] else text


def build_layout(width, height, params):
    """Lays out and renders both captions for a picture of width x height."""
    with _LOCK:
        ref = resolve_font(params["font"], params["bold"], params["italic"])
        pad = int(round(params["padding"] / 100.0 * width))
        top_text = _prepared_text(params["top_text"], params)
        bottom_text = _prepared_text(params["bottom_text"], params)
        frame = params["placement"] == "frame"
        border = int(round(params["frame_width"] / 100.0 * width)) if frame else 0
        around_meme = params["frame_width_mode"] == "around meme"
        b_img, b_out = (0, border) if around_meme else (border, 0)  # border next to the picture / around everything
        outer = width + 2 * b_img  # the text wraps to the (bordered) picture
        fixed = frame and params["frame_height_mode"] == "fixed"
        band_fixed = int(round(params["frame_height"] / 100.0 * width)) if fixed else None
        # extra distance from the top / bottom edge (in a frame: empty room at the band's outer edge)
        m_top = int(round(params["top_margin"] / 100.0 * height))
        m_bottom = int(round(params["bottom_margin"] / 100.0 * height))
        max_h_top = max(MIN_PX, band_fixed - 2 * pad - m_top) if fixed else None
        max_h_bottom = max(MIN_PX, band_fixed - 2 * pad - m_bottom) if fixed else None

        top = layout_block(top_text, params, ref, outer, pad, params["top_lines"], max_h_top, params["top_font_size"], width)
        bottom = layout_block(bottom_text, params, ref, outer, pad, params["bottom_lines"], max_h_bottom, params["bottom_font_size"], width)
        missing = sorted(set(missing_chars(ref, top_text + bottom_text)))
        if top is None and bottom is None and not border:
            empty = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            return Layout(empty, 0, height, missing, ref.name, ref.fallback, False)

        top_band = bottom_band = bottom_pad = 0
        if frame:
            if top is not None:
                top_band = band_fixed if fixed else top.height + 2 * pad + m_top
            if bottom is not None:
                bottom_band = band_fixed if fixed else bottom.height + 2 * pad + m_bottom
            if (height + 2 * border + top_band + bottom_band) % 2:  # video encoders want an even height
                if bottom_band:
                    bottom_band += 1
                elif top_band:
                    top_band += 1
                else:
                    bottom_pad = 1
        total = height + 2 * border + top_band + bottom_band + bottom_pad
        right_pad = 1 if border and (outer + 2 * b_out) % 2 else 0  # ... and an even width
        total_w = outer + 2 * b_out + right_pad
        img_x, img_y = b_out + b_img, b_out + top_band + b_img

        overlay = Image.new("RGBA", (total_w, total), (0, 0, 0, 0))
        if frame:
            # the frame color everywhere except the picture itself
            draw = ImageDraw.Draw(overlay)
            draw.rectangle([0, 0, total_w - 1, total - 1], fill=params["frame_color"] + (255,))
            draw.rectangle([img_x, img_y, img_x + width - 1, img_y + height - 1], fill=(0, 0, 0, 0))

        def place(block, align, y):
            nonlocal overlay
            tmp = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
            tmp.paste(_render_block(block, params, ref, outer, align, pad), (b_out, int(y)))
            overlay = Image.alpha_composite(overlay, tmp)

        if top is not None:
            y = b_out + m_top + (max(0.0, (top_band - m_top - top.height) / 2.0) if fixed else pad)
            place(top, params["top_align"], y)
        if bottom is not None:
            if frame:
                # the margin is at the outer (lower) edge of the band, the text sits in the rest
                y = total - bottom_pad - b_out - bottom_band + (max(0.0, (bottom_band - m_bottom - bottom.height) / 2.0) if fixed else pad)
            else:
                y = total - pad - bottom.height - m_bottom
            place(bottom, params["bottom_align"], y)
        sizes = tuple(None if b is None else round(b.px / width * 100.0, 1) for b in (top, bottom))
        return Layout(overlay, top_band, total, missing, ref.name, ref.fallback, True, sizes, total_w, img_x, img_y)


def compose_frames(images, layout):
    """images: IMAGE tensor (B, H, W, C) -> captioned tensor (B, H2, W2, 3).
    Frames are processed in chunks; only the caption's bounding box is blended."""
    batch, height, width = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    if not layout.has_content:
        return images
    total, top, left = layout.total_h, layout.img_y, layout.img_x
    out = np.zeros((batch, total, layout.total_w, 3), dtype=np.float32)
    box = layout.overlay.getchannel("A").getbbox()
    if box is not None:
        x0, y0, x1, y1 = box
        region = np.asarray(layout.overlay.crop(box), dtype=np.float32) / 255.0
        alpha, rgb = region[..., 3:4], region[..., :3]
    for start in range(0, batch, CHUNK):
        end = min(batch, start + CHUNK)
        chunk = np.asarray(images[start:end].cpu().numpy(), dtype=np.float32)[..., :3]
        out[start:end, top:top + height, left:left + width] = chunk
        if box is not None:
            part = out[start:end, y0:y1, x0:x1]
            out[start:end, y0:y1, x0:x1] = part * (1.0 - alpha) + rgb * alpha
    return torch.from_numpy(out)


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def render_preview(raw_params, width, height):
    """The caption layer for a picture of the given shape, as a transparent PNG
    (with the opaque frame bands in frame mode) - the browser lays it over a
    gray picture or over the last input frame. The size is scaled down to
    PREVIEW_MAX_SIDE. Returns a dict for the route's answer."""
    params = clean_params(raw_params)
    width = int(_number("width", width, PREVIEW_MIN_SIDE, 16384, integer=True))
    height = int(_number("height", height, PREVIEW_MIN_SIDE, 16384, integer=True))
    scale = min(1.0, PREVIEW_MAX_SIDE / max(width, height))
    width, height = max(PREVIEW_MIN_SIDE, int(round(width * scale))), max(PREVIEW_MIN_SIDE, int(round(height * scale)))
    layout = build_layout(width, height, params)
    buffer = io.BytesIO()
    layout.overlay.save(buffer, format="PNG", compress_level=3)
    return {
        "png": buffer.getvalue(), "layout": layout, "width": width, "height": height,
        "top_band": layout.top_band, "total_h": layout.total_h,
        "total_w": layout.total_w, "img_x": layout.img_x, "img_y": layout.img_y,
    }


def _thumbnail(frames):
    """The first frame as a small JPEG data URL (the node's preview background), or None."""
    try:
        first = np.asarray(frames[0:1].cpu().numpy(), dtype=np.float32)[0, :, :, :3]
        image = Image.fromarray((np.clip(first, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), "RGB")
        image.thumbnail((THUMB_SIDE, THUMB_SIDE))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=72)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def _make_video(components, images):
    """A VIDEO from the new frames with the original audio and frame rate."""
    try:
        from comfy_api.latest import InputImpl, Types
        make, parts = InputImpl.VideoFromComponents, Types.VideoComponents
    except Exception:
        try:
            from comfy_api.input_impl import VideoFromComponents as make
            from comfy_api.input import VideoComponents as parts
        except Exception:
            raise RuntimeError("[Mykee Meme Maker] this ComfyUI version has no VIDEO support - use the image output.") from None
    kwargs = {"images": images, "audio": components.audio, "frame_rate": components.frame_rate}
    metadata = getattr(components, "metadata", None)
    try:
        return make(parts(**kwargs, metadata=metadata))
    except TypeError:
        return make(parts(**kwargs))


class MykeeMemeMaker:
    @classmethod
    def INPUT_TYPES(cls):
        d = DEFAULTS
        return {
            "required": {
                "top_text": ("STRING", {"multiline": True, "default": "TOP TEXT", "tooltip": "Caption at the top. Can be connected from another node."}),
                "bottom_text": ("STRING", {"multiline": True, "default": "BOTTOM TEXT", "tooltip": "Caption at the bottom. Can be connected from another node."}),
                "font": ("STRING", {"default": d["font"], "tooltip": "Font family name (pick it from the list on the node)."}),
                "top_font_size": ("FLOAT", {"default": d["top_font_size"], "min": 0.5, "max": 60.0, "step": 0.1, "tooltip": "Top caption font size in % of the image width."}),
                "bottom_font_size": ("FLOAT", {"default": d["bottom_font_size"], "min": 0.5, "max": 60.0, "step": 0.1, "tooltip": "Bottom caption font size in % of the image width."}),
                "bold": ("BOOLEAN", {"default": d["bold"]}),
                "italic": ("BOOLEAN", {"default": d["italic"]}),
                "uppercase": ("BOOLEAN", {"default": d["uppercase"]}),
                "text_color": ("STRING", {"default": d["text_color"], "tooltip": "Hex color, e.g. #FFFFFF."}),
                "stroke_color": ("STRING", {"default": d["stroke_color"], "tooltip": "Hex color of the outline."}),
                "stroke_width": ("FLOAT", {"default": d["stroke_width"], "min": 0.0, "max": 40.0, "step": 0.5, "tooltip": "Outline thickness in % of the font size (0 = none)."}),
                "line_height": ("FLOAT", {"default": d["line_height"], "min": 0.3, "max": 4.0, "step": 0.05, "tooltip": "Line spacing (1.0 = the font's own)."}),
                "padding": ("FLOAT", {"default": d["padding"], "min": 0.0, "max": 40.0, "step": 0.5, "tooltip": "Space between the text and the edge of the picture / frame, in % of the image width."}),
                "placement": (list(PLACEMENTS), {"default": d["placement"], "tooltip": "on image: the text is drawn on the picture. frame: colored bands are added above / below the picture and the text goes there."}),
                "frame_color": ("STRING", {"default": d["frame_color"], "tooltip": "Hex color of the frame (the bands and the border)."}),
                "frame_width": ("FLOAT", {"default": d["frame_width"], "min": 0.0, "max": 40.0, "step": 0.5, "tooltip": "Border width all around the picture in % of the image width (0 = none). Frame placement only."}),
                "frame_width_mode": (list(FRAME_WIDTH_MODES), {"default": d["frame_width_mode"], "tooltip": "around image: the border hugs the picture (the text bands sit outside it). around meme: the border goes around the whole captioned meme (bands included). Only with frame_width > 0."}),
                "frame_height_mode": (list(FRAME_HEIGHT_MODES), {"default": d["frame_height_mode"], "tooltip": "fit text: the band is as tall as the text + padding. fixed: the band has frame_height (the text is shrunk to fit)."}),
                "frame_height": ("FLOAT", {"default": d["frame_height"], "min": 1.0, "max": 200.0, "step": 0.5, "tooltip": "Fixed band height in % of the image width."}),
                "top_align": (list(ALIGNS), {"default": d["top_align"]}),
                "top_lines": ("INT", {"default": 0, "min": 0, "max": MAX_LINES, "tooltip": "0 = automatic wrap. N = at most N lines; the font is shrunk if the text needs more."}),
                "bottom_align": (list(ALIGNS), {"default": d["bottom_align"]}),
                "bottom_lines": ("INT", {"default": 0, "min": 0, "max": MAX_LINES, "tooltip": "0 = automatic wrap. N = at most N lines; the font is shrunk if the text needs more."}),
                "preview_width": ("INT", {"default": 1024, "min": 16, "max": 16384, "tooltip": "Only for the preview on the node (the result always has the input's size): the width of the preview picture. Used until the node has run; after a run the preview follows the real picture (edit a field to override that until the next run). Can be connected."}),
                "preview_height": ("INT", {"default": 1024, "min": 16, "max": 16384, "tooltip": "Only for the preview on the node: the height of the preview picture. See preview_width. Can be connected."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "An image or an image batch."}),
                "video": ("VIDEO", {"tooltip": "A video (instead of image). The video output keeps its audio and frame rate."}),
                # optional (not required), so API workflows saved before these existed still validate
                "top_margin": ("FLOAT", {"default": d["top_margin"], "min": 0.0, "max": 100.0, "step": 0.5, "tooltip": "Extra distance of the top caption from the top edge, in % of the image height (0 = right at the padding). In frame placement: empty room at the top of the band."}),
                "bottom_margin": ("FLOAT", {"default": d["bottom_margin"], "min": 0.0, "max": 100.0, "step": 0.5, "tooltip": "Extra distance of the bottom caption from the bottom edge, in % of the image height (0 = right at the padding). In frame placement: empty room at the bottom of the band."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "VIDEO")
    RETURN_NAMES = ("image", "video")
    OUTPUT_TOOLTIPS = (
        "The captioned image / image batch (the frames, when the input is a video).",
        "The captioned video - only when the input is a video.",
    )
    FUNCTION = "run"
    CATEGORY = "Mykee/Image"
    DESCRIPTION = "Top / bottom caption (meme text) on an image, image batch or video: on the picture or in a frame, with any installed font."

    def run(self, top_text, bottom_text, font, top_font_size, bottom_font_size, bold, italic, uppercase, text_color, stroke_color,
            stroke_width, line_height, padding, placement, frame_color, frame_width, frame_width_mode, frame_height_mode, frame_height,
            top_align, top_lines, bottom_align, bottom_lines,
            preview_width=1024, preview_height=1024,
            image=None, video=None, top_margin=0.0, bottom_margin=0.0):  # preview_width / preview_height only steer the preview on the node
        if image is not None and video is not None:
            raise ValueError("[Mykee Meme Maker] Connect either an image or a video, not both.")
        if image is None and video is None:
            raise ValueError("[Mykee Meme Maker] Connect an image or a video.")
        params = clean_params({
            "top_text": top_text, "bottom_text": bottom_text, "font": font, "top_font_size": top_font_size, "bottom_font_size": bottom_font_size,
            "bold": bold, "italic": italic, "uppercase": uppercase, "text_color": text_color,
            "stroke_color": stroke_color, "stroke_width": stroke_width, "line_height": line_height,
            "padding": padding, "placement": placement, "frame_color": frame_color, "frame_width": frame_width, "frame_width_mode": frame_width_mode,
            "frame_height_mode": frame_height_mode, "frame_height": frame_height,
            "top_align": top_align, "top_lines": top_lines, "bottom_align": bottom_align, "bottom_lines": bottom_lines,
            "top_margin": top_margin, "bottom_margin": bottom_margin,
        })
        components = None
        if video is not None:
            components = video.get_components()
            frames = components.images
        else:
            frames = image
        height, width = int(frames.shape[1]), int(frames.shape[2])
        layout = build_layout(width, height, params)
        if layout.fallback:
            print(f"[Mykee Meme Maker] Font '{params['font']}' was not found on this system - using '{layout.font_name}'.")
        if layout.missing:
            print(f"[Mykee Meme Maker] The font '{layout.font_name}' has no glyph for: {''.join(layout.missing)}")
        out = compose_frames(frames, layout)

        if components is not None:
            video_out = _make_video(components, out)
        else:
            video_out = ExecutionBlocker(None) if ExecutionBlocker is not None else None
        ui = {"mykee_meme": [{"width": width, "height": height, "missing": "".join(layout.missing), "thumb": _thumbnail(frames)}]}
        return {"ui": ui, "result": (out, video_out)}


NODE_CLASS_MAPPINGS = {NODE_NAME: MykeeMemeMaker}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_NAME: "Mykee Meme Maker"}


# ---------------------------------------------------------------------------
# HTTP routes (no path in either of them)
# ---------------------------------------------------------------------------

try:
    if PromptServer is None:
        raise RuntimeError("no server")
    _routes = PromptServer.instance.routes

    def _error(message, status):
        return _aiohttp_web.json_response({"error": message}, status=status)

    @_routes.get("/mykee/meme/fonts")
    async def _mykee_meme_fonts(request):
        refresh = request.query.get("refresh") == "1"
        loop = asyncio.get_running_loop()
        names = await loop.run_in_executor(None, lambda: (font_index(refresh), font_names())[1])
        default = await loop.run_in_executor(None, default_font_name)
        return _aiohttp_web.json_response({"fonts": names, "default": default})

    @_routes.post("/mykee/meme/preview")
    async def _mykee_meme_preview(request):
        if request.content_length is not None and request.content_length > 64 * 1024:
            return _error("the request is too large", 413)
        try:
            data = await request.json()
        except Exception:
            return _error("invalid JSON body", 400)
        if not isinstance(data, dict) or not isinstance(data.get("params"), dict):
            return _error("invalid request", 400)
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None, render_preview, data["params"], data.get("width", 1024), data.get("height", 576))
        except ValueError as e:
            return _error(str(e), 400)
        except Exception:
            return _error("the preview could not be rendered", 500)
        layout = result["layout"]
        return _aiohttp_web.json_response({
            "png": base64.b64encode(result["png"]).decode("ascii"),
            "width": result["width"], "height": result["height"],
            "top_band": result["top_band"], "total_h": result["total_h"],
            "total_w": result["total_w"], "img_x": result["img_x"], "img_y": result["img_y"],
            "sizes": list(layout.sizes),
            "missing": "".join(layout.missing),
            "font": layout.font_name,
            "fallback": layout.fallback,
        })

except Exception:
    # No running PromptServer (module imported outside ComfyUI): the node
    # still works, the preview just has nothing to call.
    pass
