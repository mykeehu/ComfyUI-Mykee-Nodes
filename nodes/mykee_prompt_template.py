"""
ComfyUI-Mykee-Nodes / Mykee Prompt Template

A small prompt-library node: "positive_prompt_text" and
"negative_prompt_text" are real multiline STRING widgets - they ARE the
node's two outputs (whatever's in them is always what actually runs
when you queue), and "Load" / "Save" / "Reload" just move both of them
to and from disk together - one plain JSON file per template, the same
one-file-per-item approach this pack already uses for StyleGAN vectors
(mykee_stylegan.py's MykeeStyleGANVectorSave).

Unlike the StyleGAN vector save, there's nothing here that needs the
graph to actually run (no "armed" gate, no OUTPUT_NODE) - the text
already lives in the widgets, so Save is a plain synchronous REST call
that writes both fields out immediately, whenever you click it.

Storage location, in order:
  1. custom_path, if that widget is filled in - used exactly as given
     (created if missing).
  2. otherwise ComfyUI/user/default/Prompt templates - mirrors where
     ComfyUI's own built-in prompt templates live, so this pack's
     templates show up in a familiar spot.

Each template is exactly one file: <template_name>.json, holding
{"prompt": "...", "negative_prompt": "..."}. Older files saved before
the negative-prompt field existed only have {"text": "..."} - those
still load fine, as a positive prompt with an empty negative prompt.

template_name is sanitized on save so the resulting file name is valid
on BOTH Windows and Linux - stripped of every character either OS
forbids, trimmed of trailing dots/spaces (Windows rejects those), and
renamed off of Windows' reserved device names (CON, PRN, AUX, NUL,
COM1-9, LPT1-9) if it happens to collide with one. The actually-used
name is sent back to the JS after a save, so the widget always
reflects what's really on disk.

Inputs:
  template_name        - name for this template; also its file name
                          once sanitized. Only needed for Load/Save -
                          the node still runs fine with it empty (Save
                          just refuses).
  positive_prompt_text - the positive prompt text. This is what the
                          node's first output is - Load overwrites it
                          from disk, Save writes its current value.
  negative_prompt_text - the negative prompt text, same deal, second
                          output.
  custom_path           - optional. Folder to use instead of the
                          default ComfyUI/user/default/Prompt
                          templates location.

Outputs:
  positive_prompt - positive_prompt_text, unchanged.
  negative_prompt - negative_prompt_text, unchanged.
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


_HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_TEMPLATES_SUBPATH = os.path.join("user", "default", "Prompt templates")

# Windows forbids these characters anywhere in a file name; Linux really
# only forbids "/" (and NUL), but stripping the full Windows set too means
# a template saved on Windows and copied to a Linux box (or vice versa)
# never trips either OS - which is the actual requirement here (a name
# valid on BOTH).
_INVALID_CHARS = '<>:"/\\|?*'
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
_MAX_NAME_LEN = 150  # plenty for a template name, well under any filesystem's path-length limit


def _sanitize_template_name(name):
    name = (name or "").strip()
    # Drop control characters and every char either OS forbids in a file name.
    name = "".join(c for c in name if c not in _INVALID_CHARS and ord(c) >= 32)
    # Windows also rejects a name ending in a dot or space.
    name = name.strip().rstrip(". ").strip()
    if name.upper() in _WINDOWS_RESERVED:
        name = "_" + name
    name = name[:_MAX_NAME_LEN].strip().rstrip(". ")
    return name


def _default_templates_dir():
    base = None
    if folder_paths is not None:
        get_user_dir = getattr(folder_paths, "get_user_directory", None)
        if callable(get_user_dir):
            try:
                user_dir = get_user_dir()
                if user_dir:
                    d = os.path.join(user_dir, "default", "Prompt templates")
                    os.makedirs(d, exist_ok=True)
                    return d
            except Exception:
                pass
        base = getattr(folder_paths, "base_path", None)
    if not base:
        # Fallback: this file normally lives at
        # ComfyUI/custom_nodes/ComfyUI-Mykee-Nodes/nodes/mykee_prompt_template.py,
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


def _load_template(directory, name):
    """Returns {"prompt": str, "negative_prompt": str} or None if the
    file doesn't exist / can't be read. Transparently upgrades the old
    single-field format ({"text": "..."}, from before negative prompts
    existed) into the new shape, with an empty negative prompt."""
    p = _template_path(directory, name)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        if "prompt" in data or "negative_prompt" in data:
            return {
                "prompt": str(data.get("prompt", "")),
                "negative_prompt": str(data.get("negative_prompt", "")),
            }
        if "text" in data:  # pre-negative-prompt file
            return {"prompt": str(data["text"]), "negative_prompt": ""}
    except Exception:
        pass
    return None


def _save_template(directory, name, prompt, negative_prompt):
    p = _template_path(directory, name)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"prompt": prompt, "negative_prompt": negative_prompt}, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)


class MykeePromptTemplate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template_name": ("STRING", {
                    "default": "",
                    "tooltip": "Name for this template - also its file name "
                               "once saved (sanitized so it's valid on both "
                               "Windows and Linux). Only needed for Load/Save.",
                }),
                "positive_prompt_text": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "The positive prompt text. This is the node's "
                               "first output. Load overwrites it from disk; "
                               "Save writes its current value to disk.",
                }),
                "negative_prompt_text": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "The negative prompt text. This is the node's "
                               "second output. Load overwrites it from disk; "
                               "Save writes its current value to disk.",
                }),
                "custom_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional. If filled in, templates are loaded/"
                               "saved from this folder instead of the default "
                               "ComfyUI/user/default/Prompt templates.",
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "run"
    CATEGORY = "Mykee/Prompt"

    def run(self, template_name, positive_prompt_text, negative_prompt_text, custom_path, unique_id=None):
        return (positive_prompt_text, negative_prompt_text)


# ---------------------------------------------------------------------------
# REST routes - list/load/save, called by this node's JS Load/Save/Reload
# buttons. Pure JSON read/write, no graph execution needed for any of them.
# ---------------------------------------------------------------------------

try:
    from aiohttp import web as _aiohttp_web

    _routes = PromptServer.instance.routes

    @_routes.get("/mykee/prompt_templates/list")
    async def _mykee_pt_list(request):
        custom_path = request.query.get("path", "")
        directory = _templates_dir(custom_path)
        return _aiohttp_web.json_response({"templates": _list_template_names(directory)})

    @_routes.get("/mykee/prompt_templates/load")
    async def _mykee_pt_load(request):
        name = _sanitize_template_name(request.query.get("name", ""))
        if not name:
            return _aiohttp_web.json_response({"error": "name is required"}, status=400)
        custom_path = request.query.get("path", "")
        directory = _templates_dir(custom_path)
        template = _load_template(directory, name)
        if template is None:
            return _aiohttp_web.json_response(
                {"error": f"No '{name}.json' found in {directory}"}, status=404
            )
        return _aiohttp_web.json_response({
            "name": name,
            "prompt": template["prompt"],
            "negative_prompt": template["negative_prompt"],
        })

    @_routes.post("/mykee/prompt_templates/save")
    async def _mykee_pt_save(request):
        try:
            data = await request.json()
        except Exception:
            return _aiohttp_web.json_response({"error": "invalid JSON body"}, status=400)
        name = _sanitize_template_name(data.get("name", ""))
        if not name:
            return _aiohttp_web.json_response({"error": "name is required"}, status=400)
        prompt = data.get("prompt", "")
        negative_prompt = data.get("negative_prompt", "")
        if not isinstance(prompt, str) or not isinstance(negative_prompt, str):
            return _aiohttp_web.json_response({"error": "prompt/negative_prompt must be strings"}, status=400)
        custom_path = data.get("path", "")
        directory = _templates_dir(custom_path)
        try:
            _save_template(directory, name, prompt, negative_prompt)
        except Exception as e:
            return _aiohttp_web.json_response({"error": str(e)}, status=400)
        return _aiohttp_web.json_response({
            "name": name,
            "templates": _list_template_names(directory),
        })

except Exception:
    # No running PromptServer (e.g. module imported outside ComfyUI) - the
    # node still works, the Load/Save/Reload buttons just won't have
    # anything to call.
    pass


NODE_CLASS_MAPPINGS = {
    "MykeePromptTemplate": MykeePromptTemplate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeePromptTemplate": "Mykee Prompt Template",
}
