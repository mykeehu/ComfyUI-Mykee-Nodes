"""
ComfyUI-Mykee-Nodes / Mykee Recipe

Builds "recipes" out of ANY widget parameter of ANY node: a recipe is a named
set of parameter values (cfg, sampler, scheduler, a model, a LoRA, a prompt
text, ...) that can be stored in a JSON file - several recipes per file - and
loaded back into the node.

The node has a "recipe_name" output and up to ten parameter outputs, p1 .. p10,
typed "*". Like the Mykee XYZ Plot axes and the Mykee Model Template outputs,
each parameter output is dragged onto a widget (cfg, sampler_name, ckpt_name,
a text box, a boolean, ...). The frontend (web/mykee_recipe.js) then takes over
that widget's type, shows an editor for the value on the node and gives the
output a key - the "alias" the recipe stores the value under (by default the
widget's name: "cfg", "sampler_name", ...). What is selected on the node is
what the connected widget receives.

How recipes find their targets
------------------------------
A recipe is NOT tied to a node class. It stores values under keys, and the key
of a parameter output is what links a recipe entry to a connected widget. So a
recipe made with the stock KSampler also loads onto a KSampler-like node from
somebody else, as long as the output carries the same key. The node class and
widget name are stored only as a hint (used as a fallback when no key matches
and exactly one entry fits the widget). Entries whose key matches no connected
output are skipped, outputs that the recipe has no entry for keep their value,
and values that do not fit the widget (wrong type, a file that is not in the
list, out of range) are not applied - the frontend reports each of these.

Recipe file (UTF-8 JSON):

    {"version": 1,
     "recipes": [
       {"name": "Portrait fast",
        "params": {
          "cfg":     {"type": "FLOAT", "value": 4.5, "hint": {"node": "KSampler", "widget": "cfg"}},
          "sampler": {"type": "COMBO", "value": "euler_ancestral"}}}
     ]}

Hand-written shortcuts are accepted and cleaned on load: a bare value
("cfg": 4.5), a single recipe object, or a plain list of recipes.

Security
--------
ComfyUI has no authentication, so the HTTP routes below treat every parameter
as hostile input:

  * the folder must lie inside an allowed root (mykee_path_guard);
  * a file name is reduced to a safe relative name ("a/b"), then the resolved
    path must still be inside the resolved folder, and a file or folder that
    is a symlink is refused, so a link can't lead out of the folder;
  * only "<name>.json" is ever read or written, files over MAX_FILE_BYTES are
    refused, and a save never overwrites a JSON file that is not a recipe file;
  * the content is validated against a strict schema (types, lengths, counts,
    finite numbers, no prototype-polluting keys) on load AND on save;
  * files are written atomically (temp file + rename), error texts never
    contain absolute paths.

Inputs:
  slots        - how many parameter outputs the node shows (1 - 10).
  recipe_name  - name of the current recipe; also the "recipe_name" output.
  json_name    - file name (without .json) used by Save / Load.
  custom_path  - optional folder instead of ComfyUI/user/default/Recipes.

Outputs:
  recipe_name, p1 .. p10 - the values selected on the node for the connected
  outputs (None for the others).
"""

import json
import math
import os
import re
import tempfile

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    from server import PromptServer
except ImportError:
    PromptServer = None

from .mykee_prompt_template import _sanitize_segment, _sanitize_template_name, _split_name, _sort_key
from .mykee_path_guard import check_custom_dir


_HERE = os.path.dirname(os.path.abspath(__file__))

MAX_SLOTS = 10
DEFAULT_SLOTS = 3
VALUE_TYPES = ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO")

DOC_VERSION = 1
MAX_RECIPES = 500
MAX_PARAMS = 64            # entries per recipe (more than the 10 outputs: hand-written extras are kept)
MAX_KEY = 64
MAX_NAME = 120
MAX_HINT = 100
MAX_STRING = 200_000       # one text value (a prompt)
MAX_COMBO = 1_000          # one combo value (a file name)
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_LIST_FILES = 2_000
SAFE_INT = 2 ** 53

_BAD_KEYS = frozenset({"__proto__", "constructor", "prototype"})
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class AnyType(str):
    """Wildcard type: the parameter outputs can be linked to any widget input."""

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


ANY = AnyType("*")
SLOT_NAMES = tuple(f"p{i}" for i in range(1, MAX_SLOTS + 1))


# ---------------------------------------------------------------------------
# Schema validation / cleaning (also used for the files)
# ---------------------------------------------------------------------------

def _text(value, limit, what):
    """A str without NUL / lone surrogates, at most `limit` characters long.
    Too long is an error, never a silent truncation (that would lose data)."""
    if not isinstance(value, str):
        raise ValueError(f"{what} must be text")
    value = value.replace("\x00", "").encode("utf-8", "replace").decode("utf-8")
    if len(value) > limit:
        raise ValueError(f"{what} is too long (over {limit} characters)")
    return value


def _clean_label(value, limit):
    """Single-line label: control characters dropped, trimmed, length-limited."""
    if not isinstance(value, str):
        return ""
    value = _CONTROL.sub("", value.encode("utf-8", "replace").decode("utf-8")).strip()
    return value[:limit].strip()


def _clean_key(value):
    key = _clean_label(value, MAX_KEY)
    if not key or key in _BAD_KEYS:
        return None
    return key


def _infer_type(value):
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, int):
        return "INT"
    if isinstance(value, float):
        return "FLOAT"
    if isinstance(value, str):
        return "STRING"
    raise ValueError("unsupported value (only numbers, text and true/false)")


def clean_value(value_type, value):
    """Returns (type, value) checked against the type, or raises ValueError."""
    value_type = _infer_type(value) if value_type is None else str(value_type).upper()
    if value_type not in VALUE_TYPES:
        raise ValueError(f'unknown type "{value_type[:20]}"')
    if value_type == "BOOLEAN":
        if not isinstance(value, bool):
            raise ValueError("a BOOLEAN value must be true or false")
        return value_type, value
    if value_type in ("INT", "FLOAT"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"a {value_type} value must be a number")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("the number is not finite")
        if value_type == "INT":
            if isinstance(value, float):
                if value != int(value):
                    raise ValueError("an INT value must be a whole number")
                value = int(value)
            if abs(value) > SAFE_INT:
                raise ValueError("the number is out of range")
            return value_type, value
        if abs(value) > 1e300:
            raise ValueError("the number is out of range")
        return value_type, float(value)
    limit = MAX_COMBO if value_type == "COMBO" else MAX_STRING
    return value_type, _text(value, limit, f"a {value_type} value")


def _clean_hint(hint):
    if not isinstance(hint, dict):
        return None
    out = {}
    for field in ("node", "widget"):
        text = _clean_label(hint.get(field), MAX_HINT)
        if text:
            out[field] = text
    return out or None


def _clean_param(entry):
    """One recipe entry -> {"type", "value"[, "hint"]}. A bare value is
    accepted as shorthand (the type is then inferred)."""
    hint = None
    if isinstance(entry, dict):
        declared = entry.get("type")
        value = entry.get("value")
        hint = _clean_hint(entry.get("hint"))
        if declared is not None and not isinstance(declared, str):
            raise ValueError("type must be text")
        value_type, value = clean_value(declared, value)
    else:
        value_type, value = clean_value(None, entry)
    out = {"type": value_type, "value": value}
    if hint:
        out["hint"] = hint
    return out


def _clean_recipe(raw, position, warnings):
    """Returns a clean recipe dict, or None (with a warning) if unusable."""
    if not isinstance(raw, dict):
        warnings.append(f"recipe {position}: not an object - skipped")
        return None
    name = _clean_label(raw.get("name"), MAX_NAME) or f"Recipe {position}"
    params_raw = raw.get("params")
    params = {}
    if params_raw is None:
        params_raw = {}
    if not isinstance(params_raw, dict):
        warnings.append(f"'{name}': params must be an object - skipped")
        return None
    for key_raw, entry in params_raw.items():
        if len(params) >= MAX_PARAMS:
            warnings.append(f"'{name}': more than {MAX_PARAMS} entries - the rest was dropped")
            break
        key = _clean_key(key_raw)
        if key is None:
            warnings.append(f"'{name}': invalid key {str(key_raw)[:30]!r} - skipped")
            continue
        try:
            params[key] = _clean_param(entry)
        except ValueError as e:
            warnings.append(f"'{name}': '{key}' skipped - {e}")
    return {"name": name, "params": params}


def clean_doc(data):
    """Validates a recipe document. Returns ({"version", "recipes"}, warnings);
    raises ValueError if there is no usable recipe list at all."""
    if isinstance(data, dict) and "recipes" not in data and "params" in data:
        data = {"recipes": [data]}  # a single recipe
    elif isinstance(data, list):
        data = {"recipes": data}
    if not isinstance(data, dict) or not isinstance(data.get("recipes"), list):
        raise ValueError('not a recipe file (expected {"version": 1, "recipes": [...]})')
    warnings = []
    raw = data["recipes"]
    if len(raw) > MAX_RECIPES:
        warnings.append(f"more than {MAX_RECIPES} recipes - the rest was dropped")
        raw = raw[:MAX_RECIPES]
    recipes = []
    for position, item in enumerate(raw, start=1):
        recipe = _clean_recipe(item, position, warnings)
        if recipe is not None:
            recipes.append(recipe)
    return {"version": DOC_VERSION, "recipes": recipes}, warnings


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _default_recipes_dir():
    base = None
    if folder_paths is not None:
        get_user_dir = getattr(folder_paths, "get_user_directory", None)
        if callable(get_user_dir):
            try:
                user_dir = get_user_dir()
                if user_dir:
                    return os.path.join(user_dir, "default", "Recipes")
            except Exception:
                pass
        base = getattr(folder_paths, "base_path", None)
    if not base:
        # This file normally lives at
        # ComfyUI/custom_nodes/ComfyUI-Mykee-Nodes/nodes/mykee_recipe.py,
        # so three levels up is the ComfyUI root.
        base = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
    return os.path.join(base, "user", "default", "Recipes")


def _recipes_dir(custom_path=None):
    """The folder to use, as a resolved path. Nothing is created here - the
    folder is only created by a real save."""
    custom_path = (custom_path or "").strip() if isinstance(custom_path, str) else ""
    if custom_path:
        return check_custom_dir(custom_path)  # PermissionError outside the allowed roots
    return os.path.realpath(_default_recipes_dir())


def _inside(root, path):
    try:
        return os.path.commonpath([os.path.normcase(path), os.path.normcase(root)]) == os.path.normcase(root)
    except ValueError:  # different drives
        return False


def _resolve_file(directory, name):
    """Safe path of "<name>.json" in `directory`. Raises ValueError for an
    unusable name and PermissionError for anything that would lead out of the
    folder (or goes through a symlink)."""
    clean = _sanitize_template_name(name)
    folders, base = _split_name(clean)
    if not base:
        raise ValueError("a file name is required")
    root = os.path.realpath(directory)
    path = os.path.join(root, *folders, base + ".json")
    real = os.path.realpath(path)
    if not _inside(root, real):
        raise PermissionError("that file is outside the recipe folder")
    # realpath() resolved every link; refuse links on the way, too, so a
    # symlinked folder or file can never be used (not even one that points to
    # another place inside the folder).
    probe = root
    for part in (*folders, base + ".json"):
        probe = os.path.join(probe, part)
        if os.path.islink(probe):
            raise PermissionError("symbolic links are not allowed in the recipe folder")
    return clean, real


def _list_files(directory):
    """Every .json file below `directory` as a relative name without the
    extension ("a/b"), sorted without regard to case. Links are skipped."""
    names = []
    try:
        for root, dirs, files in os.walk(directory, followlinks=False):
            dirs[:] = [d for d in dirs if not d.startswith(".") and not os.path.islink(os.path.join(root, d))]
            rel_root = os.path.relpath(root, directory)
            prefix = "" if rel_root == "." else rel_root.replace(os.sep, "/") + "/"
            for f in files:
                full = os.path.join(root, f)
                if f.lower().endswith(".json") and not f.startswith(".") and not os.path.islink(full):
                    names.append(prefix + os.path.splitext(f)[0])
                    if len(names) >= MAX_LIST_FILES:
                        return sorted(names, key=_sort_key)
    except OSError:
        return []
    return sorted(names, key=_sort_key)


def _read_json(path):
    """Reads a JSON file as UTF-8 (BOM tolerated; a legacy cp1250 file falls
    back to that code page). Refuses files over MAX_FILE_BYTES."""
    if os.path.getsize(path) > MAX_FILE_BYTES:
        raise ValueError("the file is too large")
    with open(path, "rb") as f:
        raw = f.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("the file is too large")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1250", "replace")
    return json.loads(text)


def _is_recipe_file(path):
    """True only for a JSON file that looks like one of ours, so a save can
    never clobber some other project's JSON file."""
    try:
        data = _read_json(path)
    except Exception:
        return False
    return isinstance(data, dict) and set(data) <= {"version", "recipes"} and isinstance(data.get("recipes"), list)


def load_file(directory, name):
    """Returns (clean name, doc, warnings). Raises FileNotFoundError, ValueError
    (bad content), PermissionError."""
    clean, path = _resolve_file(directory, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No '{clean}.json' found in the recipe folder")
    try:
        data = _read_json(path)
    except ValueError as e:
        raise ValueError(f"'{clean}.json' can't be read: {e}")
    doc, warnings = clean_doc(data)
    return clean, doc, warnings


def save_file(directory, name, doc_raw):
    """Validates and writes a recipe file. Returns the clean name."""
    doc, _warnings = clean_doc(doc_raw)
    clean, path = _resolve_file(directory, name)
    if os.path.exists(path) and not _is_recipe_file(path):
        raise ValueError(f"'{clean}.json' already exists and is not a recipe file - refusing to overwrite it")
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    if not _inside(os.path.realpath(directory), os.path.realpath(parent)):
        raise PermissionError("that file is outside the recipe folder")
    fd, tmp = tempfile.mkstemp(dir=parent, prefix=".mykee_recipe_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return clean


def _parse_config(raw):
    try:
        cfg = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
    except (TypeError, ValueError):
        cfg = {}
    return cfg if isinstance(cfg, dict) else {}


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class MykeeRecipe:
    CATEGORY = "Mykee/Utils"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",) + (ANY,) * MAX_SLOTS
    RETURN_NAMES = ("recipe_name",) + SLOT_NAMES
    OUTPUT_TOOLTIPS = (
        "Name of the current recipe (cleaned like a file name) - handy as a file name prefix.",
    ) + tuple(
        "Drag onto any widget (cfg, sampler, model, a text box, ...): the node then holds that "
        "parameter's value and stores it in the recipe."
        for _ in SLOT_NAMES
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "slots": ("INT", {
                    "default": DEFAULT_SLOTS, "min": 1, "max": MAX_SLOTS,
                    "tooltip": "How many parameter outputs (p1 ... p10) the node has.",
                }),
                "recipe_name": ("STRING", {
                    "default": "",
                    "tooltip": "Name of the recipe. 'Send to Recipe' stores the current parameters under "
                               "this name (an existing recipe with the same name is updated); "
                               "'Load Recipe' fills it in from the recipe that is loaded.",
                }),
                "json_name": ("STRING", {
                    "default": "",
                    "tooltip": "File name of the recipe JSON (without .json) used by Save / Load. "
                               "A sub-folder is allowed: 'portraits/sdxl'.",
                }),
                "custom_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional. If filled in, recipe files are loaded/saved from this folder "
                               "instead of the default ComfyUI/user/default/Recipes. The folder must be "
                               "inside ComfyUI's user / input / output folder or listed in "
                               "allowed_template_dirs.txt.",
                }),
                # Holds the node's state (JSON). Hidden: the frontend draws its
                # own panels, and fills in only the resolved values of the
                # connected outputs at queue time.
                "recipe_config": ("STRING", {"default": "{}", "hidden": True, "socketless": True}),
            },
        }

    def run(self, slots=DEFAULT_SLOTS, recipe_name="", json_name="", custom_path="", recipe_config="{}"):
        params = _parse_config(recipe_config).get("params")
        params = params if isinstance(params, dict) else {}
        try:
            slots = max(1, min(MAX_SLOTS, int(slots)))
        except (TypeError, ValueError):
            slots = DEFAULT_SLOTS

        # One file-name segment (a recipe name is not a path): '/' etc. are cleaned away.
        name_out = _sanitize_segment(recipe_name if isinstance(recipe_name, str) else "")
        out = [name_out]
        for index, slot_name in enumerate(SLOT_NAMES, start=1):
            entry = params.get(slot_name) if index <= slots else None
            if not isinstance(entry, dict) or "value" not in entry:
                out.append(None)
                continue
            try:
                _type, value = clean_value(entry.get("type"), entry.get("value"))
            except ValueError as e:
                raise ValueError(f"[Mykee Recipe] {slot_name}: {e}")
            out.append(value)
        return tuple(out)


# ---------------------------------------------------------------------------
# REST routes - list / load / save, called by this node's JS buttons.
# ---------------------------------------------------------------------------

try:
    from aiohttp import web as _aiohttp_web

    _routes = PromptServer.instance.routes

    def _error(message, status):
        return _aiohttp_web.json_response({"error": message}, status=status)

    @_routes.get("/mykee/recipes/list")
    async def _mykee_rc_list(request):
        try:
            directory = _recipes_dir(request.query.get("path", ""))
        except PermissionError as e:
            return _error(str(e), 403)
        return _aiohttp_web.json_response({"files": _list_files(directory)})

    @_routes.get("/mykee/recipes/load")
    async def _mykee_rc_load(request):
        try:
            directory = _recipes_dir(request.query.get("path", ""))
            name, doc, warnings = load_file(directory, request.query.get("name", ""))
        except PermissionError as e:
            return _error(str(e), 403)
        except FileNotFoundError as e:
            return _error(str(e), 404)
        except (ValueError, OSError) as e:
            return _error(str(e) if isinstance(e, ValueError) else "the file can't be read", 400)
        return _aiohttp_web.json_response({"name": name, "doc": doc, "warnings": warnings})

    @_routes.post("/mykee/recipes/save")
    async def _mykee_rc_save(request):
        if request.content_length is not None and request.content_length > MAX_FILE_BYTES * 2:
            return _error("the request is too large", 413)
        try:
            data = await request.json()
        except Exception:
            return _error("invalid JSON body", 400)
        if not isinstance(data, dict):
            return _error("invalid JSON body", 400)
        try:
            directory = _recipes_dir(data.get("path", ""))
            name = save_file(directory, data.get("name", ""), data.get("doc"))
        except PermissionError as e:
            return _error(str(e), 403)
        except (ValueError, OSError) as e:
            return _error(str(e) if isinstance(e, ValueError) else "the file can't be written", 400)
        return _aiohttp_web.json_response({"name": name, "files": _list_files(directory)})

except Exception:
    # No running PromptServer (e.g. module imported outside ComfyUI) - the
    # node still works, the buttons just won't have anything to call.
    pass


NODE_CLASS_MAPPINGS = {
    "MykeeRecipe": MykeeRecipe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeRecipe": "Mykee Recipe",
}
