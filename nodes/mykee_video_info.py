"""
ComfyUI-Mykee-Nodes / Mykee Video Info

Connect a VIDEO (e.g. from ComfyUI's Load Video node) and the node shows its
width, height, frame rate, codec and container format in a small box on the
node, and passes the useful values on:

- width, height (INT), frame_rate (FLOAT)
- filename (STRING): the file name of the video without folder and extension,
  with filename_tag appended ("video" + "_edit" -> "video_edit").

Codec and format are only shown, not output: every node that takes them uses
its own list type, so a generic output could not be connected anywhere.

Where the values come from:

- A video that is a file (Load Video, or anything that keeps the file behind
  it, also after trimming) is only probed - the header is read, no frame is
  decoded.
- A video that only exists in memory (Create Video and similar) has no codec,
  format or file name yet; the box says so, and the filename output starts
  with FALLBACK_NAME.

The node is an output node, so it runs (and fills the box) even when none of
its outputs is connected. The frontend part is web/mykee_video_info.js.
"""

import io
import os
from fractions import Fraction

try:
    import av
except Exception:  # pragma: no cover - ComfyUI ships PyAV; only missing outside it
    av = None

LOG = "[Mykee Video Info]"
FALLBACK_NAME = "video"  # filename base of a video that has no file behind it
NOT_ENCODED = "not encoded yet"
UNKNOWN = "unknown"
MAX_TAG_LENGTH = 200


def _file_source(video):
    """The file path / byte buffer behind a file-based video, else None.

    Only classes that keep an already encoded file are asked: the generic
    get_stream_source() of other video types would encode the whole video just
    to answer, which this node must never trigger.
    """
    if type(video).__name__ == "VideoFromFile":
        try:
            return video.get_stream_source()
        except Exception:
            return None
    return None


def _first_part(video):
    """First clip of a concatenated video (it only holds file-based clips)."""
    parts = getattr(video, "videos", None)
    if isinstance(parts, (list, tuple)) and parts:
        return parts[0]
    return None


def _format_label(format_name, path):
    """A short container name: PyAV reports e.g. "mov,mp4,m4a,3gp,3g2,mj2"."""
    names = [n for n in str(format_name or "").split(",") if n]
    if not names:
        return UNKNOWN
    ext = os.path.splitext(path)[1].lstrip(".").lower() if path else ""
    if ext and ext in names:
        return ext
    if "mp4" in names:
        return "mp4"
    if "webm" in names and ext == "webm":
        return "webm"
    return names[0]


def _probe(source):
    """(codec, format_name) of a file path / buffer; (None, None) when unreadable."""
    if av is None or source is None:
        return None, None
    try:
        if isinstance(source, io.BytesIO):
            source.seek(0)
        with av.open(source, mode="r") as container:
            format_name = getattr(container.format, "name", None)
            codec = None
            for stream in container.streams:
                if stream.type == "video":
                    codec = getattr(stream.codec_context, "name", None)
                    break
            return codec, format_name
    except Exception as exc:
        print(f"{LOG} could not read the codec / format: {type(exc).__name__}: {exc}")
        return None, None
    finally:
        if isinstance(source, io.BytesIO):
            try:
                source.seek(0)
            except Exception:
                pass


def _frame_rate(video):
    getter = getattr(video, "get_frame_rate", None)
    rate = getter() if callable(getter) else video.get_components().frame_rate  # older ComfyUI
    return float(Fraction(rate))


def _rate_label(rate):
    text = f"{rate:.3f}".rstrip("0").rstrip(".")
    return f"{text} fps"


def _base_name(path):
    """File name without folder and extension - never a path."""
    name = os.path.basename(str(path).replace("\\", "/"))
    stem = os.path.splitext(name)[0]
    return stem or name


def read_info(video):
    width, height = video.get_dimensions()
    rate = _frame_rate(video)

    source = _file_source(video)
    path = source if isinstance(source, str) else None
    codec = container_format = None
    if source is not None:
        codec, format_name = _probe(source)
        container_format = _format_label(format_name, path) if format_name else None
        missing = UNKNOWN
    else:
        # A concatenated video is written when it is saved; its clips are files,
        # so the codec and the name of the first one are still worth showing.
        first = _first_part(video)
        first_source = _file_source(first) if first is not None else None
        if first_source is not None:
            codec, _ = _probe(first_source)
            if isinstance(first_source, str):
                path = first_source
        missing = NOT_ENCODED

    return {
        "width": int(width),
        "height": int(height),
        "frame_rate": rate,
        "codec": codec or missing,
        "format": container_format or missing,
        "base": _base_name(path) if path else FALLBACK_NAME,
        "has_file": path is not None,
    }


def _clean_tag(tag):
    """The tag as typed, without line breaks / control characters."""
    text = "".join(ch for ch in str(tag or "") if ch.isprintable())
    return text[:MAX_TAG_LENGTH]


class MykeeVideoInfo:
    CATEGORY = "Mykee/Utils"
    FUNCTION = "run"
    RETURN_TYPES = ("INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("width", "height", "frame_rate", "filename")
    OUTPUT_TOOLTIPS = (
        "Width of the video in pixels.",
        "Height of the video in pixels.",
        "Frame rate of the video (frames per second).",
        "File name of the video without folder and extension, with filename_tag appended.",
    )
    # OUTPUT_NODE + the "ui" result below fill the info box drawn by
    # web/mykee_video_info.js, also when no output is connected.
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Shows the width, height, frame rate, codec and format of a video and passes on the "
        "width, height, frame rate and the file name (with an optional tag appended)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "The video to inspect, e.g. from a Load Video node."}),
                "filename_tag": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Appended to the file name on the filename output: "
                        "file \"video\" + tag \"_edit\" gives \"video_edit\". Leave empty to pass the name on as it is.",
                    },
                ),
            },
        }

    def run(self, video, filename_tag=""):
        if video is None:
            raise ValueError(f"{LOG} Connect a video.")
        info = read_info(video)
        filename = info["base"] + _clean_tag(filename_tag)
        print(
            f"{LOG} {info['width']}x{info['height']}, {_rate_label(info['frame_rate'])}, "
            f"codec {info['codec']}, format {info['format']}, filename {filename}"
        )
        ui = {
            "video_info": [
                {
                    "width": info["width"],
                    "height": info["height"],
                    "frame_rate": _rate_label(info["frame_rate"]),
                    "codec": info["codec"],
                    "format": info["format"],
                    "base": info["base"],
                    "has_file": info["has_file"],
                }
            ]
        }
        return {"ui": ui, "result": (info["width"], info["height"], info["frame_rate"], filename)}


NODE_CLASS_MAPPINGS = {
    "MykeeVideoInfo": MykeeVideoInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeVideoInfo": "Mykee Video Info",
}
