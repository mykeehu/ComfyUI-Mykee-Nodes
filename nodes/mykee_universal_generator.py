"""
ComfyUI-Mykee-Nodes / Mykee Universal Generator

A universal prompt generator built from a library of attribute lists.

The library
-----------
Lives in  ComfyUI/user/default/Mykee Universal Generator/library  and is a
plain folder tree:

    library/<category>/<property>.json
    library/<category>/<sub-category>/<property>.json

Folders are categories (at most two levels deep), every JSON file is one
"property" - a list of values to pick from (eye colours, hair styles,
lighting setups, ... anything). A property file looks like this:

    {
      "version": 1,
      "label": "Eye color",          display name (default: the file name)
      "type": "multi",               "single" (one value) or "multi"
      "max": 2,                      most values in "multi" (0 = no limit)
      "order": 10,                   position in the prompt (lower = earlier)
      "items": [
        {"id": "k3f9a1", "prompt": "green eyes", "label": "green"}
      ]
    }

"prompt" is what goes into the prompt, "label" is only what the picker shows
(it defaults to the prompt). "id" keeps a recipe line attached to its item
when the list is edited or re-ordered.

The first time the node is used, the starter library shipped with the pack
(universal_generator_library/) is copied there. After that the folder is
yours: the node never overwrites it from the pack again.

A property may also carry "scope": "scene" (always goes to the Scene tab -
lighting, camera, style ...), "subject" (always goes to a person tab - body,
clothing, pose ...) or "" (goes to whichever tab is active).

Scene and subjects
------------------
A recipe has a Scene part (always there, the image as a whole) and any
number of subjects (people, creatures - anything that needs its own
attributes). Every subject has a free-text name ("old man", "Hank, a tall
man", ...) that is used in the prompt as given, so the attributes stay bound
to the right subject.

Recipes
-------
The node's recipe (what was picked) is stored in the workflow (hidden
"recipe_state" widget) and can be saved to / loaded from
  ComfyUI/user/default/Mykee Universal Generator/recipes/<name>.json
Every recipe line carries a COPY of the picked texts, so editing or deleting
a library file never breaks a saved recipe.

Recipe line kinds:
  lib     picked from a property. mode "fixed" uses the picked items;
          "random_checked" draws min..max of the picked items, "random_all"
          draws from every item of the property (a snapshot is stored).
          A fixed line edited by hand on the node keeps its text ("edited").
  manual  a free text line typed on the node.

Running
-------
tags   = every value in recipe order, comma separated (duplicates dropped);
         with subjects: the Scene block, then one block per subject that starts
         with the subject's name, the blocks joined with " BREAK ".
prompt = with use_llm on and a clip connected: the LLM writes a natural
         description from the structured list (built-in system prompt, or
         the system_prompt input); with use_llm off it is the tag list.
Random lines are drawn with the node's seed, so the same seed always gives
the same picks; every subject draws on its own, so changing one subject never
changes another one's picks.

Security
--------
ComfyUI has no authentication, so every HTTP route treats its input as
hostile: names are validated against a strict whitelist (no separators, no
"..", no Windows device names), category paths are at most two names deep,
every resolved path must stay inside the node's own folder (symlinks are
refused), only .json files are ever written, all JSON is schema-checked with
size limits, and deleting only MOVES files into the folder's _trash.
"""

import json
import os
import random
import re
import shutil
import threading
import time
import uuid

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    from server import PromptServer
except ImportError:
    PromptServer = None

from .mykee_llm_common import (
    empty_response_message, generate_text, generation_error_hint, split_thinking,
    tokenize_for, answer_after_unfinished_thinking, thinking_unfinished,
    with_thinking_limit, unload_clip as _unload_clip, TEMPLATE_MODES,
    TEMPLATE_MODE_TOOLTIP, THINKING_BUDGET_TOOLTIP, quote_text,
)

NODE_LABEL = "Mykee Universal Generator"
_HERE = os.path.dirname(os.path.abspath(__file__))
_PACK_ROOT = os.path.dirname(_HERE)
_DEFAULT_LIBRARY = os.path.join(_PACK_ROOT, "universal_generator_library")

ROOT_FOLDER_NAME = "Mykee Universal Generator"
LIBRARY_DIR = "library"
RECIPES_DIR = "recipes"
TRASH_DIR = "_trash"

# Limits (the JS uses the same numbers).
MAX_DEPTH = 2
MAX_NAME = 80
MAX_LABEL = 120
MAX_ITEM_TEXT = 1000
MAX_ITEMS = 2000
MAX_CATEGORIES = 1000
MAX_PROPERTIES = 5000
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_ENTRIES = 500
MAX_MANUAL_TEXT = 2000
MAX_EDIT_TEXT = 2000
MAX_SYSTEM_PROMPT = 20000
MAX_SUBJECTS = 16
MAX_SUBJECT_NAME = 120
SCENE = "scene"
SCOPES = ("", "scene", "subject")

_INVALID_CHARS = set('<>:"/\\|?*')
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
TYPES = ("single", "multi")
MODES = ("fixed", "random_checked", "random_all")

_lock = threading.Lock()


def _toast(node_id, severity, summary, detail):
    if PromptServer is None or getattr(PromptServer, "instance", None) is None:
        return
    try:
        PromptServer.instance.send_sync(
            "mykee.toast",
            {"node": node_id, "severity": severity, "summary": summary, "detail": detail, "life": 6000},
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Folders and safe paths
# ---------------------------------------------------------------------------

def _user_base():
    if folder_paths is not None:
        fn = getattr(folder_paths, "get_user_directory", None)
        if callable(fn):
            try:
                d = fn()
                if d:
                    return d
            except Exception:
                pass
        base = getattr(folder_paths, "base_path", None)
        if base:
            return os.path.join(base, "user")
    # ComfyUI/custom_nodes/ComfyUI-Mykee-Nodes/nodes/ -> ComfyUI/user
    return os.path.join(os.path.dirname(os.path.dirname(_PACK_ROOT)), "user")


def root_dir():
    """ComfyUI/user/default/Mykee Universal Generator (created on demand)."""
    d = os.path.join(_user_base(), "default", ROOT_FOLDER_NAME)
    os.makedirs(d, exist_ok=True)
    return d


def _real(p):
    return os.path.normcase(os.path.realpath(os.path.abspath(p)))


def valid_name(name):
    """A single folder / file-stem name: printable, no path separators or
    characters Windows forbids, no leading dot, no trailing dot/space, not a
    Windows device name, not the trash folder. Returns the name or None."""
    if not isinstance(name, str):
        return None
    if name != name.strip() or not (1 <= len(name) <= MAX_NAME):
        return None
    if any(c in _INVALID_CHARS or ord(c) < 32 or ord(c) == 127 for c in name):
        return None
    if name.startswith(".") or name.endswith(".") or name.endswith(" "):
        return None
    if name.split(".")[0].upper() in _WINDOWS_RESERVED:
        return None
    if name.casefold() == TRASH_DIR.casefold():
        return None
    return name


def split_category(path):
    """"Body/Face" -> ["Body", "Face"]; None if invalid (empty, too deep, a
    bad name)."""
    if not isinstance(path, str) or not path or len(path) > (MAX_NAME + 1) * MAX_DEPTH:
        return None
    parts = path.split("/")
    if len(parts) > MAX_DEPTH or any(valid_name(p) is None for p in parts):
        return None
    return parts


def _no_symlinks(base, parts):
    p = base
    for part in parts:
        p = os.path.join(p, part)
        if os.path.islink(p):
            return False
    return True


def safe_path(base, parts):
    """base + parts, refused (PermissionError) when it would leave base or go
    through a symlink. Every part must already be a valid_name()."""
    if any(valid_name(p) is None for p in parts if not p.endswith(".json")) or \
            any(p.endswith(".json") and valid_name(p[:-5]) is None for p in parts):
        raise PermissionError("invalid name")
    if not _no_symlinks(base, parts):
        raise PermissionError("symbolic links are not allowed here")
    p = os.path.join(base, *parts)
    rb = _real(base)
    rp = _real(p)
    try:
        if os.path.commonpath([rp, rb]) != rb or rp == rb:
            raise PermissionError("path is outside the node's folder")
    except ValueError:
        raise PermissionError("path is outside the node's folder")
    return p


def library_dir():
    d = os.path.join(root_dir(), LIBRARY_DIR)
    if os.path.islink(d):
        raise PermissionError("the library folder must not be a symbolic link")
    if not os.path.isdir(d):
        # First use: copy the starter library shipped with the pack.
        if os.path.isdir(_DEFAULT_LIBRARY):
            shutil.copytree(_DEFAULT_LIBRARY, d, symlinks=False)
        else:
            os.makedirs(d, exist_ok=True)
    return d


def recipes_dir():
    d = os.path.join(root_dir(), RECIPES_DIR)
    if os.path.islink(d):
        raise PermissionError("the recipes folder must not be a symbolic link")
    os.makedirs(d, exist_ok=True)
    return d


def _trash(path, rel_label):
    """Moves a file/folder into <root>/_trash instead of deleting it."""
    t = os.path.join(root_dir(), TRASH_DIR)
    os.makedirs(t, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe_label = re.sub(r"[^\w.-]+", "_", rel_label)[:120] or "item"
    target = os.path.join(t, f"{stamp}_{uuid.uuid4().hex[:6]}_{safe_label}")
    shutil.move(path, target)


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _read_json(path):
    if os.path.getsize(path) > MAX_FILE_BYTES:
        raise ValueError("file is too large")
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Schema checks
# ---------------------------------------------------------------------------

def _clip_str(v, limit):
    if not isinstance(v, str):
        return ""
    v = "".join(c for c in v if c in "\n\t" or ord(c) >= 32)
    return v[:limit]


def _int(v, lo, hi, default):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return default
    try:
        v = int(v)
    except (ValueError, OverflowError):
        return default
    return max(lo, min(hi, v))


def _new_id(taken):
    while True:
        i = uuid.uuid4().hex[:8]
        if i not in taken:
            return i


def clean_property(data, fallback_label):
    """Validates / normalises a property document. Raises ValueError."""
    if not isinstance(data, dict):
        raise ValueError("a property must be a JSON object")
    raw_items = data.get("items", [])
    if not isinstance(raw_items, list):
        raise ValueError("'items' must be a list")
    if len(raw_items) > MAX_ITEMS:
        raise ValueError(f"too many items (max {MAX_ITEMS})")
    items, taken, seen_prompts = [], set(), set()
    for it in raw_items:
        if isinstance(it, str):
            it = {"prompt": it}
        if not isinstance(it, dict):
            continue
        prompt = _clip_str(it.get("prompt", ""), MAX_ITEM_TEXT).strip()
        if not prompt:
            continue
        key = prompt.casefold()
        if key in seen_prompts:
            continue
        seen_prompts.add(key)
        label = _clip_str(it.get("label", ""), MAX_LABEL).strip() or prompt[:MAX_LABEL]
        iid = it.get("id")
        if not (isinstance(iid, str) and _ID_RE.match(iid)) or iid in taken:
            iid = _new_id(taken)
        taken.add(iid)
        items.append({"id": iid, "prompt": prompt, "label": label})
    ptype = data.get("type", "single")
    if ptype not in TYPES:
        ptype = "single"
    return {
        "version": 1,
        "label": _clip_str(data.get("label", ""), MAX_LABEL).strip() or fallback_label,
        "type": ptype,
        "max": _int(data.get("max", 0), 0, MAX_ITEMS, 0),
        "order": _int(data.get("order", 100), -9999, 9999, 100),
        "scope": data.get("scope") if data.get("scope") in SCOPES else "",
        "items": items,
    }


def _clean_item_snapshot(it):
    if not isinstance(it, dict):
        return None
    prompt = _clip_str(it.get("prompt", ""), MAX_ITEM_TEXT).strip()
    if not prompt:
        return None
    iid = it.get("id")
    iid = iid if isinstance(iid, str) and _ID_RE.match(iid) else ""
    label = _clip_str(it.get("label", ""), MAX_LABEL).strip() or prompt[:MAX_LABEL]
    return {"id": iid, "prompt": prompt, "label": label}


def clean_recipe(data):
    """Validates a recipe (from a file, the workflow or an HTTP body). Bad
    entries are dropped, never trusted."""
    if not isinstance(data, dict):
        data = {}
    subjects, sids = [], set()
    raw_subjects = data.get("subjects", [])
    for sub in (raw_subjects if isinstance(raw_subjects, list) else [])[:MAX_SUBJECTS]:
        if not isinstance(sub, dict):
            continue
        sid = sub.get("sid")
        if not (isinstance(sid, str) and _ID_RE.match(sid)) or sid in sids or sid == SCENE:
            continue
        sids.add(sid)
        name = " ".join(_clip_str(sub.get("name", ""), MAX_SUBJECT_NAME).split())
        subjects.append({"sid": sid, "name": name or f"person {len(subjects) + 1}"})
    entries = []
    uids = set()
    raw = data.get("entries", [])
    if not isinstance(raw, list):
        raw = []
    for e in raw[:MAX_ENTRIES]:
        if not isinstance(e, dict):
            continue
        uid = e.get("uid")
        if not (isinstance(uid, str) and _ID_RE.match(uid)) or uid in uids:
            uid = _new_id(uids)
        uids.add(uid)
        kind = e.get("kind")
        # Lines of an unknown (or no) subject belong to the scene - recipes
        # saved before subjects existed load as scene-only.
        subject = e.get("subject") if e.get("subject") in sids else SCENE
        if kind == "manual":
            text = _clip_str(e.get("text", ""), MAX_MANUAL_TEXT)
            entries.append({"uid": uid, "subject": subject, "kind": "manual", "text": text})
            continue
        if kind != "lib":
            continue
        cat = e.get("category", "")
        prop = e.get("property", "")
        if split_category(cat) is None or valid_name(prop) is None:
            continue
        items = [s for s in (_clean_item_snapshot(i) for i in (e.get("items") or [])[:MAX_ITEMS]) if s]
        mode = e.get("mode", "fixed")
        if mode not in MODES:
            mode = "fixed"
        ptype = e.get("ptype", "multi")
        if ptype not in TYPES:
            ptype = "multi"
        lo = _int(e.get("min", 1), 1, MAX_ITEMS, 1)
        hi = _int(e.get("max", lo), lo, MAX_ITEMS, lo)
        edited = bool(e.get("edited", False)) and mode == "fixed"
        entries.append({
            "uid": uid, "subject": subject, "kind": "lib", "category": cat, "property": prop,
            "propLabel": _clip_str(e.get("propLabel", ""), MAX_LABEL) or prop,
            "order": _int(e.get("order", 100), -9999, 9999, 100),
            "ptype": ptype, "mode": mode, "min": lo, "max": hi,
            "items": items, "edited": edited,
            "text": _clip_str(e.get("text", ""), MAX_EDIT_TEXT) if edited else "",
        })
    name = _clip_str(data.get("name", ""), MAX_NAME).strip()
    return {"version": 2, "name": name, "subjects": subjects, "entries": entries}


# ---------------------------------------------------------------------------
# Library access
# ---------------------------------------------------------------------------

def list_library():
    base = library_dir()
    categories, properties, errors = [], [], []

    def walk(folder, rel_parts):
        try:
            names = sorted(os.listdir(folder), key=lambda s: (s.casefold(), s))
        except OSError:
            return
        for n in names:
            full = os.path.join(folder, n)
            if os.path.islink(full):
                continue
            if os.path.isdir(full):
                if not rel_parts and n.casefold() == TRASH_DIR.casefold():
                    continue
                if valid_name(n) is None or len(rel_parts) >= MAX_DEPTH:
                    continue
                if len(categories) >= MAX_CATEGORIES:
                    continue
                parts = rel_parts + [n]
                categories.append({"path": "/".join(parts), "name": n, "depth": len(parts)})
                walk(full, parts)
            elif rel_parts and n.lower().endswith(".json") and not n.endswith(".tmp"):
                stem = n[:-5]
                if valid_name(stem) is None or len(properties) >= MAX_PROPERTIES:
                    continue
                cat = "/".join(rel_parts)
                try:
                    prop = clean_property(_read_json(full), stem)
                except Exception as e:
                    errors.append(f"{cat}/{n}: {type(e).__name__}: {e}")
                    continue
                prop["category"] = cat
                prop["file"] = stem
                properties.append(prop)

    walk(base, [])
    return {"categories": categories, "properties": properties, "errors": errors}


def create_category(parent, name):
    if valid_name(name) is None:
        raise ValueError("invalid category name")
    parts = [] if not parent else split_category(parent)
    if parts is None:
        raise ValueError("invalid parent category")
    if len(parts) + 1 > MAX_DEPTH:
        raise ValueError(f"categories can be at most {MAX_DEPTH} levels deep")
    base = library_dir()
    if parts:
        parent_dir = safe_path(base, parts)
        if not os.path.isdir(parent_dir):
            raise ValueError("parent category not found")
    target = safe_path(base, parts + [name])
    if os.path.exists(target):
        raise ValueError("a category with this name already exists")
    os.mkdir(target)
    return "/".join(parts + [name])


def rename_category(path, name):
    parts = split_category(path)
    if parts is None or valid_name(name) is None:
        raise ValueError("invalid category name")
    base = library_dir()
    src = safe_path(base, parts)
    if not os.path.isdir(src):
        raise ValueError("category not found")
    dst = safe_path(base, parts[:-1] + [name])
    if os.path.exists(dst) and os.path.normcase(src) != os.path.normcase(dst):
        raise ValueError("a category with this name already exists")
    os.rename(src, dst)
    return "/".join(parts[:-1] + [name])


def trash_category(path):
    parts = split_category(path)
    if parts is None:
        raise ValueError("invalid category")
    src = safe_path(library_dir(), parts)
    if not os.path.isdir(src):
        raise ValueError("category not found")
    _trash(src, "category_" + "__".join(parts))


def _property_path(category, file):
    parts = split_category(category)
    if parts is None or valid_name(file) is None:
        raise ValueError("invalid category or file name")
    base = library_dir()
    folder = safe_path(base, parts)
    if not os.path.isdir(folder):
        raise ValueError("category not found")
    return safe_path(base, parts + [file + ".json"])


def save_property(category, file, data, old_category=None, old_file=None):
    target = _property_path(category, file)
    clean = clean_property(data, file)
    if old_category is not None or old_file is not None:
        src = _property_path(old_category or "", old_file or "")
        if not os.path.isfile(src):
            raise ValueError("the property being edited no longer exists")
        same = os.path.normcase(_real(src)) == os.path.normcase(_real(target))
        if not same and os.path.exists(target):
            raise ValueError("a property with this name already exists in that category")
        _write_json(target, clean)
        if not same:
            os.remove(src)
    else:
        if os.path.exists(target):
            raise ValueError("a property with this name already exists in that category")
        _write_json(target, clean)
    clean["category"] = category
    clean["file"] = file
    return clean


def trash_property(category, file):
    src = _property_path(category, file)
    if not os.path.isfile(src):
        raise ValueError("property not found")
    _trash(src, f"{category.replace('/', '__')}__{file}.json")


def list_recipes():
    d = recipes_dir()
    out = []
    try:
        for n in os.listdir(d):
            full = os.path.join(d, n)
            if n.lower().endswith(".json") and os.path.isfile(full) and not os.path.islink(full):
                stem = n[:-5]
                if valid_name(stem) is not None:
                    out.append(stem)
    except OSError:
        pass
    return sorted(out, key=lambda s: (s.casefold(), s))


def load_recipe(name):
    if valid_name(name) is None:
        raise ValueError("invalid recipe name")
    p = safe_path(recipes_dir(), [name + ".json"])
    if not os.path.isfile(p):
        raise FileNotFoundError("recipe not found")
    r = clean_recipe(_read_json(p))
    r["name"] = name
    return r


def save_recipe(name, recipe, overwrite):
    if valid_name(name) is None:
        raise ValueError("invalid recipe name")
    p = safe_path(recipes_dir(), [name + ".json"])
    if os.path.exists(p) and not overwrite:
        raise FileExistsError("a recipe with this name already exists")
    r = clean_recipe(recipe)
    r["name"] = name
    _write_json(p, r)
    return r


def trash_recipe(name):
    if valid_name(name) is None:
        raise ValueError("invalid recipe name")
    p = safe_path(recipes_dir(), [name + ".json"])
    if not os.path.isfile(p):
        raise ValueError("recipe not found")
    _trash(p, f"recipe_{name}.json")


# ---------------------------------------------------------------------------
# Building the prompt
# ---------------------------------------------------------------------------

def resolve_recipe(recipe, seed):
    """Returns (blocks, picks):
      blocks - [{"name": subject name or None (scene), "tags": [...],
                 "lines": [(category, property label, [values])],
                 "extras": [manual lines]}], scene first, then the subjects
      picks  - {uid: [values]} of the random lines (shown on the node)
    Every block draws its random lines with its own generator, so editing
    one subject never changes the picks of another."""
    picks = {}
    blocks = []
    owners = [(SCENE, None)] + [(s["sid"], s["name"]) for s in recipe["subjects"]]
    for sid, name in owners:
        rng = random.Random(f"{int(seed)}/{sid}")
        tags, lines, extras, seen = [], [], [], set()

        def add_tag(t):
            k = t.casefold()
            if t and k not in seen:
                seen.add(k)
                tags.append(t)

        for e in recipe["entries"]:
            if e["subject"] != sid:
                continue
            if e["kind"] == "manual":
                t = e["text"].strip()
                if t:
                    extras.append(t)
                    add_tag(t)
                continue
            if e["mode"] == "fixed":
                values = [e["text"].strip()] if e["edited"] and e["text"].strip() else [i["prompt"] for i in e["items"]]
            else:
                pool = e["items"]
                if not pool:
                    values = []
                else:
                    if e["ptype"] == "single":
                        k = 1
                    else:
                        lo = min(e["min"], len(pool))
                        hi = min(max(e["max"], lo), len(pool))
                        k = rng.randint(lo, hi)
                    idx = sorted(rng.sample(range(len(pool)), k))
                    values = [pool[i]["prompt"] for i in idx]
                picks[e["uid"]] = values
            values = [v for v in values if v]
            if not values:
                continue
            for v in values:
                add_tag(v)
            lines.append((e["category"], e["propLabel"], values))
        blocks.append({"name": name, "tags": tags, "lines": lines, "extras": extras})
    return blocks, picks


def build_tags(blocks):
    """Scene tags, then one block per subject starting with its name,
    joined with BREAK. Empty blocks are left out (a subject with no lines
    still appears with its name - it is part of the picture)."""
    parts = []
    for b in blocks:
        if b["name"] is None:
            if b["tags"]:
                parts.append(", ".join(b["tags"]))
        else:
            parts.append(", ".join([b["name"]] + [t for t in b["tags"] if t.casefold() != b["name"].casefold()]))
    return " BREAK ".join(parts)


DEFAULT_SYSTEM_PROMPT = (
    "You turn a structured list of attributes into a single prompt for an "
    "image-generation model, written as natural, flowing English prose. The "
    "user message holds the attributes of the scene and, when the picture has "
    "subjects such as people or creatures, a separate block for each subject, "
    "introduced by the name or short description the subject is to be called "
    "by. Every block lists one attribute per line with its value or values and "
    "may end with free-form extra details. "
    "Use every attribute and every extra detail, keep their meaning exact, and "
    "do not invent subjects, objects, colors, clothing, places or any other "
    "detail the list does not state or clearly imply; add only the connecting "
    "words a natural sentence needs. When an attribute has several values, "
    "weave all of them into the description instead of choosing one. "
    "Translate any value that is not written in English into English. Follow "
    "the order of the list where it reads naturally, and keep details that "
    "belong to the same subject together. Describe explicit or mature content "
    "plainly and factually like any other detail, without censoring, refusing "
    "or moralizing.\n\n"
    "Every subject's attributes belong to that subject alone: describe each "
    "subject with exactly the attributes of its own block, never move an "
    "attribute to another subject, and make it unmistakable which subject "
    "each detail belongs to, also when several subjects share a kind such as "
    "two people of the same sex. Introduce each subject with the name or "
    "description its block gives, keep that wording recognizable, and use the "
    "position, pose and action attributes to place the subjects in the scene "
    "relative to each other and to the setting. The scene block describes the "
    "picture as a whole and applies to every subject. If there is no subject "
    "block, do not add people or creatures that the scene does not state.\n\n"
    "Answer with the prompt only: one or two paragraphs of plain prose, with "
    "no lists, headings, tags, quotation marks, explanations or comments "
    "before or after it. Keep the whole answer under {max_length} tokens and "
    "end it with a complete sentence."
)


def build_user_text(blocks):
    out = []
    n = 0
    for b in blocks:
        if b["name"] is None:
            if not b["lines"] and not b["extras"]:
                continue
            out.append("Scene:")
        else:
            n += 1
            out.append(f'Subject {n}, called "{b["name"]}":')
        for cat, label, values in b["lines"]:
            out.append(f"- {cat.replace('/', ' > ')} > {label}: {', '.join(values)}")
        if b["extras"]:
            out.append("- Extra details: " + "; ".join(b["extras"]))
        out.append("")
    return "\n".join(out).strip()


class MykeeUniversalGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_llm": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "On: the connected clip (a text-generation "
                               "capable model) writes a natural description "
                               "from the recipe - the 'prompt' output. Off: "
                               "nothing is generated, 'prompt' carries the tag "
                               "list too (no clip needed).",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": "fixed",
                    "tooltip": "Drives the random recipe lines AND the LLM's "
                               "sampling. Same seed = same picks and same "
                               "text. Set it to randomize for a new variation "
                               "on every run.",
                }),
                "max_length": ("INT", {
                    "default": 400, "min": 32, "max": 4096, "step": 8,
                    "tooltip": "Max tokens the LLM may write.",
                }),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.01, "max": 2.0, "step": 0.01}),
                "top_k": ("INT", {"default": 64, "min": 0, "max": 1000}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 5.0, "step": 0.01}),
                "thinking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Let a thinking-capable model reason first "
                               "(the reasoning is never part of the output).",
                }),
                "unload_after_run": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Free this clip from VRAM after generating "
                               "(other models are not touched).",
                }),
                # The whole recipe as JSON. Hidden; edited through the panel.
                "recipe_state": ("STRING", {"default": "{}", "hidden": True, "socketless": True}),
            },
            "optional": {
                "clip": ("CLIP", {
                    "tooltip": "A generation-capable text model (e.g. loaded "
                               "with Load CLIP). Only needed with use_llm on.",
                }),
                "system_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Empty = the built-in instruction. Fill in or "
                               "connect a text to override it. {max_length} "
                               "is replaced with the token limit.",
                }),
                "template_mode": (TEMPLATE_MODES, {"default": "auto", "tooltip": TEMPLATE_MODE_TOOLTIP}),
                "thinking_budget": ("INT", {
                    "default": 1024, "min": 0, "max": 16384, "step": 64,
                    "tooltip": THINKING_BUDGET_TOOLTIP,
                }),
            },
            "hidden": {"node_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("tags", "prompt")
    FUNCTION = "run"
    CATEGORY = "Mykee/Prompt"
    OUTPUT_NODE = True

    def run(self, use_llm, seed, max_length, temperature, top_k, top_p, min_p,
            repetition_penalty, thinking=False, unload_after_run=False, recipe_state="{}",
            clip=None, system_prompt="", template_mode="auto", thinking_budget=1024,
            node_id=None):
        try:
            recipe = clean_recipe(json.loads(recipe_state or "{}"))
        except Exception:
            recipe = clean_recipe({})
        blocks, picks = resolve_recipe(recipe, seed)
        tags_text = build_tags(blocks)

        def done(prompt_text):
            return {"ui": {"tags_display": [tags_text], "prompt_display": [prompt_text],
                           "picks": [json.dumps(picks, ensure_ascii=False)]},
                    "result": (tags_text, prompt_text)}

        if not use_llm or not tags_text:
            return done(tags_text)
        if clip is None:
            _toast(node_id, "warn", NODE_LABEL,
                   "use_llm is on but no clip is connected - the prompt output is the tag list.")
            return done(tags_text)

        custom = _clip_str(system_prompt or "", MAX_SYSTEM_PROMPT).strip()
        sys_text = custom or DEFAULT_SYSTEM_PROMPT
        sys_text = sys_text.replace("{max_length}", str(int(max_length)))
        user_text = quote_text(build_user_text(blocks))

        def make_prompts(kind):
            return sys_text, user_text

        plain_make_prompts = make_prompts
        make_prompts = with_thinking_limit(make_prompts, thinking, thinking_budget)
        answer = None
        try:
            tokens, chat_text, notes, used_kind = tokenize_for(
                clip, make_prompts, thinking=thinking, template_mode=template_mode)
            for n in notes:
                _toast(node_id, "warn", NODE_LABEL, n)
            raw = generate_text(
                clip, tokens, chat_text, max_length, True, seed, temperature, top_k,
                top_p, min_p, repetition_penalty, 0.0, thinking, thinking_budget)
            answer, thinking_text = split_thinking(raw)
            if thinking_unfinished(raw, answer, thinking_text, thinking):
                answer = answer_after_unfinished_thinking(
                    clip, plain_make_prompts, None, None, None, None, thinking_text,
                    max_length, True, seed, temperature, top_k, top_p, min_p,
                    repetition_penalty, 0.0, template_mode, NODE_LABEL)
            answer = (answer or "").strip().strip('"').strip()
            if not answer:
                _toast(node_id, "warn", NODE_LABEL,
                       f"{empty_response_message(raw)} - the prompt output is the tag list.")
                answer = tags_text
        except Exception as e:
            _toast(node_id, "error", NODE_LABEL,
                   f"Generation failed ({type(e).__name__}: {e}). {generation_error_hint(False, e)} "
                   "The prompt output is the tag list.")
            answer = tags_text
        finally:
            if unload_after_run:
                _unload_clip(clip)
        return done(answer)


# ---------------------------------------------------------------------------
# REST routes (all under /mykee/ugen/)
# ---------------------------------------------------------------------------

try:
    from aiohttp import web as _web

    _routes = PromptServer.instance.routes

    def _err(msg, status=400):
        return _web.json_response({"error": str(msg)}, status=status)

    async def _body(request):
        if request.content_length is not None and request.content_length > MAX_FILE_BYTES + 65536:
            raise ValueError("request is too large")
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("a JSON object is expected")
        return data

    def _str(data, key, optional=False):
        v = data.get(key)
        if v is None and optional:
            return None
        if not isinstance(v, str):
            raise ValueError(f"'{key}' must be a string")
        return v

    def _guard(fn):
        async def wrapper(request):
            try:
                return await fn(request)
            except PermissionError as e:
                return _err(e, 403)
            except FileNotFoundError as e:
                return _err(e, 404)
            except FileExistsError as e:
                return _err(e, 409)
            except (ValueError, json.JSONDecodeError) as e:
                return _err(e, 400)
            except Exception as e:
                print(f"{NODE_LABEL}: route error: {type(e).__name__}: {e}")
                return _err("internal error", 500)
        return wrapper

    @_routes.get("/mykee/ugen/library")
    @_guard
    async def _ugen_library(request):
        with _lock:
            return _web.json_response(list_library())

    @_routes.post("/mykee/ugen/category/create")
    @_guard
    async def _ugen_cat_create(request):
        d = await _body(request)
        with _lock:
            return _web.json_response({"path": create_category(_str(d, "parent"), _str(d, "name"))})

    @_routes.post("/mykee/ugen/category/rename")
    @_guard
    async def _ugen_cat_rename(request):
        d = await _body(request)
        with _lock:
            return _web.json_response({"path": rename_category(_str(d, "path"), _str(d, "name"))})

    @_routes.post("/mykee/ugen/category/delete")
    @_guard
    async def _ugen_cat_delete(request):
        d = await _body(request)
        with _lock:
            trash_category(_str(d, "path"))
        return _web.json_response({"ok": True})

    @_routes.post("/mykee/ugen/property/save")
    @_guard
    async def _ugen_prop_save(request):
        d = await _body(request)
        old_cat = _str(d, "old_category", optional=True)
        old_file = _str(d, "old_file", optional=True)
        with _lock:
            prop = save_property(_str(d, "category"), _str(d, "file"), d.get("data"), old_cat, old_file)
        return _web.json_response({"property": prop})

    @_routes.post("/mykee/ugen/property/delete")
    @_guard
    async def _ugen_prop_delete(request):
        d = await _body(request)
        with _lock:
            trash_property(_str(d, "category"), _str(d, "file"))
        return _web.json_response({"ok": True})

    @_routes.get("/mykee/ugen/recipes")
    @_guard
    async def _ugen_recipes(request):
        with _lock:
            return _web.json_response({"recipes": list_recipes()})

    @_routes.get("/mykee/ugen/recipe/load")
    @_guard
    async def _ugen_recipe_load(request):
        with _lock:
            return _web.json_response({"recipe": load_recipe(request.query.get("name", ""))})

    @_routes.post("/mykee/ugen/recipe/save")
    @_guard
    async def _ugen_recipe_save(request):
        d = await _body(request)
        with _lock:
            r = save_recipe(_str(d, "name"), d.get("recipe"), bool(d.get("overwrite", False)))
        return _web.json_response({"recipe": r})

    @_routes.post("/mykee/ugen/recipe/delete")
    @_guard
    async def _ugen_recipe_delete(request):
        d = await _body(request)
        with _lock:
            trash_recipe(_str(d, "name"))
        return _web.json_response({"ok": True})

except Exception as _e:  # no server (e.g. imported by a test)
    if PromptServer is not None:
        print(f"{NODE_LABEL}: could not register routes: {_e}")


NODE_CLASS_MAPPINGS = {"MykeeUniversalGenerator": MykeeUniversalGenerator}
NODE_DISPLAY_NAME_MAPPINGS = {"MykeeUniversalGenerator": "Mykee Universal Generator"}
