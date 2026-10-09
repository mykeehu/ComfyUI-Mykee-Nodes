"""
ComfyUI-Mykee-Nodes / Mykee Model Template

Remembers which CLIP / VAE files (and which CLIP type) belong to which
model, so you don't have to keep that in your head. The node has seven
outputs - model, clip_type, clip_1, clip_2, clip_3, vae and vae_2 - typed
"*" (vae_2 is for models that need two VAEs). You drag each one onto the combo
widget of a loader node (the ckpt_name of a checkpoint loader, the
unet_name of a diffusion-model loader, clip_name / clip_name1..3 of the
CLIP loaders, "type" of a CLIP loader for clip_type, vae_name of a VAE
loader). The frontend
(web/mykee_model_template.js) then shows that loader's file list on this
node, and whatever is selected here is what the loader receives - the
same idea as the Mykee XYZ Plot axes, but with one fixed value per output.

A "template" is one small JSON file with the current selection:

    {"version": 1, "model": "...", "clip_type": "...", "clip_1": "...", "clip_2": "...", "vae": "...", "notes": "..."}

"notes" is the optional free-text / Markdown "Template Notes" field shown at
the bottom of the node (a README-like description of the template). It is
stored as a normal JSON string, so newlines, quotes, backslashes and
non-ASCII characters (Hungarian accents, emoji, ...) are escaped / encoded
by the JSON layer; files are always written as UTF-8 and read as UTF-8
(UTF-8 with BOM and, as a last resort, cp1250 files are tolerated too).

Only the outputs that are connected are saved. When a template is loaded,
every entry is checked against the list of the widget that output is
connected to: an entry is applied only if the output is connected AND the
file name is really in that list. Anything else (output not connected,
file not installed on this machine, entry missing from the template) is
left exactly as it is. So loading a template with a model and a VAE onto a
node that has no CLIP connected just selects the model and the VAE.

Unlike the Prompt Template node, the "is it in the list" check needs the
connected widgets' option lists, so the loading itself happens in the
frontend; this module only stores / lists / reads the JSON files and
passes the selected names on as outputs.

Storage location, in order:
  1. custom_path, if that widget is filled in - used exactly as given
     (created if missing).
  2. otherwise ComfyUI/user/default/Model templates.

template_name is sanitized exactly like in the Prompt Template node, so
the file name is valid on both Windows and Linux.

Inputs:
  template_name - name of the template; also its file name once sanitized.
  custom_path   - optional folder to use instead of the default location.

Outputs:
  model, clip_type, clip_1, clip_2, clip_3, vae, vae_2 - the entries selected on
  the node, for the outputs that are connected (None for the others).
"""

import json
import os

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    from server import PromptServer
except ImportError:
    PromptServer = None

from .mykee_prompt_template import _sanitize_template_name


_HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_TEMPLATES_SUBPATH = os.path.join("user", "default", "Model templates")

# Output slots, in output order. The JS side uses the same names.
SLOTS = ("model", "clip_type", "clip_1", "clip_2", "clip_3", "vae", "vae_2")
TEMPLATE_VERSION = 1
MAX_NOTES_CHARS = 200_000


class AnyType(str):
    """Wildcard type: the outputs can be linked to any combo widget input."""

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


ANY = AnyType("*")


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _default_templates_dir():
    base = None
    if folder_paths is not None:
        get_user_dir = getattr(folder_paths, "get_user_directory", None)
        if callable(get_user_dir):
            try:
                user_dir = get_user_dir()
                if user_dir:
                    d = os.path.join(user_dir, "default", "Model templates")
                    os.makedirs(d, exist_ok=True)
                    return d
            except Exception:
                pass
        base = getattr(folder_paths, "base_path", None)
    if not base:
        # This file normally lives at
        # ComfyUI/custom_nodes/ComfyUI-Mykee-Nodes/nodes/mykee_model_template.py,
        # so three levels up is the ComfyUI root.
        base = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
    d = os.path.join(base, DEFAULT_TEMPLATES_SUBPATH)
    os.makedirs(d, exist_ok=True)
    return d


def _templates_dir(custom_path=None):
    custom_path = (custom_path or "").strip()
    if custom_path:
        os.makedirs(custom_path, exist_ok=True)
        return custom_path
    return _default_templates_dir()


def _template_path(directory, name):
    return os.path.join(directory, f"{name}.json")


def _list_template_names(directory):
    try:
        return sorted(
            os.path.splitext(f)[0] for f in os.listdir(directory) if f.lower().endswith(".json")
        )
    except FileNotFoundError:
        return []


def _clean_values(data):
    """Keeps only known slots with a non-empty string value."""
    values = {}
    if isinstance(data, dict):
        for slot in SLOTS:
            v = data.get(slot)
            if isinstance(v, str) and v.strip():
                values[slot] = v
    return values


def _clean_notes(data):
    """Returns the 'notes' string of a template dict ('' if missing).

    Line endings are normalised to LF, NUL characters are dropped and
    characters that can't be encoded as UTF-8 (lone surrogates from the
    browser) are replaced, so the file can always be written."""
    notes = data.get("notes") if isinstance(data, dict) else None
    if not isinstance(notes, str):
        return ""
    notes = notes.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    notes = notes.encode("utf-8", "replace").decode("utf-8")
    return notes[:MAX_NOTES_CHARS]


def _read_json_file(path):
    """Reads a JSON file as UTF-8 (BOM tolerated). A file that was edited and
    saved in a legacy Windows code page falls back to cp1250 (Central
    European), so accented text isn't lost."""
    with open(path, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1250", "replace")
    return json.loads(text)


def _load_template(directory, name):
    """Returns {"values": {slot: file name}, "notes": str} (only the slots
    present in the file) or None if the file doesn't exist / can't be read."""
    p = _template_path(directory, name)
    if not os.path.isfile(p):
        return None
    try:
        data = _read_json_file(p)
        if not isinstance(data, dict):
            return None
        return {"values": _clean_values(data), "notes": _clean_notes(data)}
    except Exception:
        return None


def _save_template(directory, name, values, notes=""):
    p = _template_path(directory, name)
    tmp = p + ".tmp"
    payload = {"version": TEMPLATE_VERSION}
    payload.update(_clean_values(values))
    notes = _clean_notes({"notes": notes})
    if notes.strip():
        payload["notes"] = notes
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)


def _parse_config(raw):
    try:
        cfg = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
    except (TypeError, ValueError):
        cfg = {}
    return cfg if isinstance(cfg, dict) else {}


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class MykeeModelTemplate:
    CATEGORY = "Mykee/Utils"
    FUNCTION = "run"
    RETURN_TYPES = (ANY, ANY, ANY, ANY, ANY, ANY, ANY)
    RETURN_NAMES = SLOTS
    OUTPUT_TOOLTIPS = (
        "Drag onto the model list of a checkpoint / diffusion model loader.",
        "Drag onto the \"type\" list of a CLIP loader (stable_diffusion, flux, ...).",
        "Drag onto the (first) CLIP list of a CLIP loader.",
        "Drag onto the second CLIP list of a dual / triple CLIP loader.",
        "Drag onto the third CLIP list of a triple CLIP loader.",
        "Drag onto the VAE list of a VAE loader.",
        "Drag onto the VAE list of a second VAE loader (for models that need two VAEs).",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template_name": ("STRING", {
                    "default": "",
                    "tooltip": "Name for this template - also its file name "
                               "once saved (sanitized so it's valid on both "
                               "Windows and Linux). Only needed for Save.",
                }),
                "custom_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional. If filled in, templates are loaded/"
                               "saved from this folder instead of the default "
                               "ComfyUI/user/default/Model templates.",
                }),
                # Holds the current selection (JSON). Hidden: the frontend
                # draws its own panel and fills in the resolved values of
                # the connected outputs at queue time.
                "model_config": ("STRING", {"default": "{}", "hidden": True, "socketless": True}),
            },
        }

    def run(self, template_name="", custom_path="", model_config="{}"):
        values = _parse_config(model_config).get("values")
        values = values if isinstance(values, dict) else {}
        out = []
        for slot in SLOTS:
            v = values.get(slot)
            out.append(v if isinstance(v, str) and v else None)
        return tuple(out)


# ---------------------------------------------------------------------------
# REST routes - list / load / save, called by this node's JS buttons.
# The "only apply what exists in the connected lists" logic lives in the JS,
# because only the frontend knows the connected widgets' option lists.
# ---------------------------------------------------------------------------

try:
    from aiohttp import web as _aiohttp_web

    _routes = PromptServer.instance.routes

    @_routes.get("/mykee/model_templates/list")
    async def _mykee_mt_list(request):
        directory = _templates_dir(request.query.get("path", ""))
        return _aiohttp_web.json_response({"templates": _list_template_names(directory)})

    @_routes.get("/mykee/model_templates/load")
    async def _mykee_mt_load(request):
        name = _sanitize_template_name(request.query.get("name", ""))
        if not name:
            return _aiohttp_web.json_response({"error": "name is required"}, status=400)
        directory = _templates_dir(request.query.get("path", ""))
        loaded = _load_template(directory, name)
        if loaded is None:
            return _aiohttp_web.json_response(
                {"error": f"No readable '{name}.json' found in {directory}"}, status=404
            )
        return _aiohttp_web.json_response({"name": name, "values": loaded["values"], "notes": loaded["notes"]})

    @_routes.post("/mykee/model_templates/save")
    async def _mykee_mt_save(request):
        try:
            data = await request.json()
        except Exception:
            return _aiohttp_web.json_response({"error": "invalid JSON body"}, status=400)
        name = _sanitize_template_name(data.get("name", ""))
        if not name:
            return _aiohttp_web.json_response({"error": "name is required"}, status=400)
        values = _clean_values(data.get("values"))
        notes = _clean_notes(data)
        if not values and not notes.strip():
            return _aiohttp_web.json_response(
                {"error": "nothing to save - no connected output has a value and the notes are empty"}, status=400
            )
        directory = _templates_dir(data.get("path", ""))
        try:
            _save_template(directory, name, values, notes)
        except Exception as e:
            return _aiohttp_web.json_response({"error": str(e)}, status=400)
        return _aiohttp_web.json_response({
            "name": name,
            "templates": _list_template_names(directory),
        })

except Exception:
    # No running PromptServer (e.g. module imported outside ComfyUI) - the
    # node still works, the buttons just won't have anything to call.
    pass


NODE_CLASS_MAPPINGS = {
    "MykeeModelTemplate": MykeeModelTemplate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeModelTemplate": "Mykee Model Template",
}
