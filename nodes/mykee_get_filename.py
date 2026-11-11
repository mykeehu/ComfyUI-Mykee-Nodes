"""
ComfyUI-Mykee-Nodes / Mykee Get Filename

Connect any media output (image, video, audio, ...) and the node passes it on
unchanged and outputs the name of the file it was loaded from - without folder
and extension, with filename_tag appended ("video" + "_edit" -> "video_edit").

Where the name comes from: an IMAGE or AUDIO does not carry its file name, so
the node reads the workflow instead. It walks back from its input, node by
node, until it finds a node with a media file in one of its fields (Load
Image, Load Video, Load Audio and the loaders of other node packs alike) and
takes that name. The nearest such node wins, so nodes between the loader and
this one (resize, crop, ...) do not matter. A VIDEO that is a file is asked
directly first. When nothing is found, the name is FALLBACK_NAME.

Only the workflow data is read - no file or folder is touched, and the output
is always a bare name, never a path.

The first output takes the type of whatever is connected to the input
(web/mykee_get_filename.js keeps the socket in sync).
"""

import os
import re
from collections import deque

from .mykee_video_info import _base_name, _clean_tag, _file_source

LOG = "[Mykee Get Filename]"
FALLBACK_NAME = "file"  # when no file name can be found behind the input
MAX_NODES = 200  # how many upstream nodes are looked at
MAX_VALUE_LENGTH = 1024

MEDIA_EXTENSIONS = frozenset(
    # images
    "png jpg jpeg jpe jfif webp gif bmp tif tiff tga exr hdr avif heic heif jxl psd ico svg "
    # video
    "mp4 m4v mov mkv webm avi wmv flv mpg mpeg ts mts m2ts 3gp ogv gifv "
    # audio
    "wav mp3 flac ogg oga opus m4a aac wma aiff aif "
    # other things a workflow loads from a file
    "latent safetensors npy npz json txt csv srt vtt obj glb gltf fbx ply stl".split()
)

# "name.png [input]" - the folder note ComfyUI adds to uploaded / pasted files
_ANNOTATION_RE = re.compile(r"\s*\[(?:input|output|temp)\]\s*$")


class AnyType(str):
    """Wildcard type: the input takes, and the first output gives, any type."""

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    __hash__ = str.__hash__


ANY = AnyType("*")


def _media_name(value):
    """The bare file name when value looks like a media file, else None."""
    if not isinstance(value, str) or not value or len(value) > MAX_VALUE_LENGTH:
        return None
    if "\n" in value or "\r" in value:
        return None  # a prompt, not a file field
    text = _ANNOTATION_RE.sub("", value.strip())
    ext = os.path.splitext(text)[1].lstrip(".").lower()
    if ext not in MEDIA_EXTENSIONS:
        return None
    name = _base_name(text)
    return name or None


def _is_link(value):
    return isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[1], int)


def find_upstream_name(prompt, unique_id, input_name="media"):
    """Name of the nearest media file in the nodes feeding the given input."""
    if not isinstance(prompt, dict):
        return None
    node = prompt.get(str(unique_id))
    if not isinstance(node, dict):
        return None
    start = (node.get("inputs") or {}).get(input_name)
    if not _is_link(start):
        return None

    queue = deque([str(start[0])])
    seen = set()
    while queue and len(seen) < MAX_NODES:
        node_id = queue.popleft()
        if node_id in seen:
            continue
        seen.add(node_id)
        inputs = (prompt.get(node_id) or {}).get("inputs")
        if not isinstance(inputs, dict):
            continue
        for value in inputs.values():
            name = _media_name(value)
            if name:
                return name
        for value in inputs.values():
            if _is_link(value):
                queue.append(str(value[0]))
    return None


class MykeeGetFilename:
    CATEGORY = "Mykee/Utils"
    FUNCTION = "run"
    RETURN_TYPES = (ANY, "STRING")
    RETURN_NAMES = ("media", "filename")
    OUTPUT_TOOLTIPS = (
        "The input, passed on unchanged.",
        "File name the input was loaded from, without folder and extension, with filename_tag appended.",
    )
    DESCRIPTION = (
        "Passes any media (image, video, audio, ...) through unchanged and outputs the name of the "
        "file it was loaded from, with an optional tag appended."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "media": (ANY, {"tooltip": "Any media output: image, video, audio, ... It is passed on unchanged."}),
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
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    def run(self, media, filename_tag="", prompt=None, unique_id=None):
        name = None
        source = _file_source(media)  # a video that is a file knows its own name
        if isinstance(source, str):
            name = _base_name(source) or None
        if not name:
            name = find_upstream_name(prompt, unique_id)
        if not name:
            name = FALLBACK_NAME
            print(f"{LOG} no media file found behind the input - using \"{FALLBACK_NAME}\" as the name.")
        filename = name + _clean_tag(filename_tag)
        print(f"{LOG} filename: {filename}")
        return (media, filename)


NODE_CLASS_MAPPINGS = {
    "MykeeGetFilename": MykeeGetFilename,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeGetFilename": "Mykee Get Filename",
}
