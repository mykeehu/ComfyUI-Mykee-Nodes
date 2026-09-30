"""
ComfyUI-Mykee-Nodes / StyleGAN2 image generator + editor + vector builder

Four nodes built around a StyleGAN2 generator (e.g. stylegan2-ffhq-1024x1024.pkl) -
displayed as "StyleGAN ..." but internally still named/keyed around "Face"
since that's what this pack's own vectors are built for; any StyleGAN2/3
generator (not necessarily a face model) works the same way:

  1) MykeeStyleGANFaceGenerator ("StyleGAN Image Generation")
     - Generates an image from a seed + truncation.
     - Reads every "direction vector" (.npy, e.g. gender/age/smile) it finds in
       models/stylegan/facegen_vectors/, together with its per-vector JSON
       parameters (which W layers it affects, its multiplier, its slider
       range), and builds one slider widget per vector - no restart needed,
       just hit the refresh button after adding/editing a vector.

  2) MykeeStyleGANEditor ("StyleGAN Editor")
     - Same vector sliders as Image Generation, but edits an existing photo
       instead of generating from a seed: projects the photo into W space
       (GAN inversion), applies the current vectors, re-synthesizes.
     - keep_cache (on by default) remembers the last projection per node
       instance, so tweaking sliders after the first run is fast - only a
       changed image/model/projection setting (or keep_cache off) re-runs
       the slow projection step.

  3) MykeeStyleGANVectorPreview
     - Builds w_a/w_b, the two W-space "extremes" of a new direction, either
       from two seeds (instant, no photos - the same trick the original
       generate2.py's seed-pair vector builder used) or by GAN-inverting two
       connected photos (NVIDIA's stylegan2-ada/stylegan3 projector
       algorithm - slower, but works from real faces).
     - Shows a side-by-side preview and outputs a w_pair connection - it
       never writes anything to disk, so you can re-roll seeds or try
       different photos freely.

  4) MykeeStyleGANVectorSave
     - Takes a w_pair from the Preview node. Only when THIS node actually
       runs does it save diff = w_b - w_a as <vector_name>.npy plus
       <vector_name>.json (layer range, multiplier, min/max) - bypass it
       (or just don't connect it yet) while you're still picking a good
       pair upstream.
     - "Load" / "Save" only touch that JSON (layer range, multiplier,
       min/max) without needing w_pair connected or the graph to run -
       pure JSON read/write over REST.

Model files live in ComfyUI/models/stylegan/*.pkl
Vectors live in       ComfyUI/models/stylegan/facegen_vectors/<name>.npy
Params live in        ComfyUI/models/stylegan/facegen_vectors/<name>.json

Needs the vendored NVIDIA runtime in ./stylegan_core (dnnlib + torch_utils)
to be importable, so a stylegan2/3 .pkl - which pickles its network
classes' *source code* alongside the weights (see
torch_utils/persistence.py) - can be unpickled and actually run.
"""

import contextlib
import copy
import json
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Make the vendored dnnlib / torch_utils importable (same trick as running
# generate2.py from inside the original project folder: its own directory
# needs to be on sys.path so the pickled StyleGAN classes resolve). The
# bias_act/upfirdn2d/filtered_lrelu ops each try to JIT-compile their own
# fast CUDA kernel on first use and fall back to a pure-PyTorch reference
# implementation if that fails (see the try/except added around each op's
# _init() in stylegan_core/torch_utils/ops/*.py) - no precompiled .pyd is
# shipped, since that's tied to one exact Python/torch build and isn't
# portable across ComfyUI installs.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # pack root (this file lives in nodes/)
_STYLEGAN_CORE_DIR = os.path.join(_HERE, "stylegan_core")
if _STYLEGAN_CORE_DIR not in sys.path:
    sys.path.insert(0, _STYLEGAN_CORE_DIR)

try:
    import dnnlib  # noqa: F401  (needed at unpickle time, keep imported)
except ImportError:
    dnnlib = None

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    from server import PromptServer
except ImportError:
    PromptServer = None

try:
    import comfy.model_management as _model_management
except ImportError:
    _model_management = None


def _check_interrupted():
    """Lets the ComfyUI Stop button actually interrupt a long-running loop
    (like the GAN-inversion optimization below) - without this, a custom
    node's Python loop has no idea Stop was pressed and just runs to
    completion regardless."""
    if _model_management is not None:
        _model_management.throw_exception_if_processing_interrupted()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

VECTORS_SUBDIR = "facegen_vectors"
VGG16_URL = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt"

_NUM_LAYERS_DEFAULT = 18  # stylegan2-ffhq-1024x1024 has 18 W+ layers (log2(1024)*2-2)

_DEFAULT_VECTOR_PARAMS = {
    "layer_start": 0,
    "layer_end": _NUM_LAYERS_DEFAULT - 1,
    "step": 0.25,
    "min": -2.0,
    "max": 2.0,
}

# Recorded alongside layer_start/layer_end/step/min/max in each vector's
# .json purely as provenance (how the vector was built) - not used to drive
# any UI, so a missing/garbage value here never breaks anything that reads
# _load_vector_params. Lets you reproduce a vector later (same model, same
# seeds/images) without having to remember what you set at the time.
_GENERATION_PARAM_KEYS = (
    "source_mode", "seed_a", "seed_b", "seed_truncation",
    "projection_steps", "projection_seed",
)


def _stylegan_models_dir():
    """ComfyUI/models/stylegan if available, else ./models/stylegan next to
    this package (mirrors mykee_character.py's _annotators_dir approach)."""
    if folder_paths is not None:
        try:
            base = os.path.join(folder_paths.models_dir, "stylegan")
            os.makedirs(base, exist_ok=True)
            return base
        except Exception:
            pass
    base = os.path.join(_HERE, "models", "stylegan")
    os.makedirs(base, exist_ok=True)
    return base


def _vectors_dir():
    d = os.path.join(_stylegan_models_dir(), VECTORS_SUBDIR)
    os.makedirs(d, exist_ok=True)
    return d


def _vgg16_path():
    return os.path.join(_stylegan_models_dir(), "vgg16.pt")


FFHQ_MODEL_FILENAME = "stylegan2-ffhq-1024x1024.pkl"
# Official NVIDIA hosting for the StyleGAN2(-ADA) FFHQ 1024x1024 (config-f)
# generator - same architecture/weights people commonly rename locally to
# "stylegan2-ffhq-1024x1024.pkl" (see NVlabs/stylegan2-ada-pytorch's README).
FFHQ_MODEL_URL = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"

# Full-body human generator from the StyleGAN-Human project (SHHQ dataset),
# built on the same NVIDIA stylegan2-ada/stylegan3 codebase this pack
# vendors - not faces, but every node here works on it identically (same
# G.mapping/G.synthesis API, same .pkl format). 1024x512 (2:1), so its
# num_ws layer count differs from the FFHQ model's 18 - see the README's
# "Layer reference" section.
HUMAN_MODEL_FILENAME = "stylegan-human-v2-1024x512.pkl"
HUMAN_MODEL_GDRIVE_ID = "1FlAb1rYa0r_--Zj_ML8e6shmaF28hQb5"
HUMAN_MODEL_GDRIVE_VIEW_URL = f"https://drive.google.com/file/d/{HUMAN_MODEL_GDRIVE_ID}/view"

# filename -> ("nvidia", direct_url) | ("gdrive", file_id). Offered in every
# model combo alongside whatever's actually on disk; picking one that isn't
# there yet triggers the matching auto-download in _ensure_model_on_disk.
_KNOWN_AUTO_DOWNLOAD_MODELS = {
    FFHQ_MODEL_FILENAME: ("nvidia", FFHQ_MODEL_URL),
    HUMAN_MODEL_FILENAME: ("gdrive", HUMAN_MODEL_GDRIVE_ID),
}


def _list_pkl_models():
    base = _stylegan_models_dir()
    try:
        files = set(f for f in os.listdir(base) if f.lower().endswith(".pkl"))
    except FileNotFoundError:
        files = set()
    # Always offer the known auto-downloadable models alongside whatever's
    # actually on disk (not just when the folder is empty) - otherwise,
    # once you have any real .pkl, there'd be no way to pick "download the
    # other one" from the combo at all.
    return sorted(files | set(_KNOWN_AUTO_DOWNLOAD_MODELS.keys()))


def _looks_like_html(path, peek_bytes=512):
    """A .pkl should be binary pickle data - if what we actually got starts
    with '<' (an HTML error/quota/interstitial page instead of the file),
    catch that here instead of writing it out as if it were the model and
    letting pickle.load blow up opaquely later, or worse, silently trying
    to run on garbage."""
    try:
        with open(path, "rb") as f:
            head = f.read(peek_bytes)
    except OSError:
        return False
    stripped = head.lstrip()
    return stripped[:1] in (b"<",) or b"<html" in stripped[:200].lower()


def _download_from_google_drive(file_id, dest_path, status_cb=None):
    """Google Drive serves an HTML "can't scan this file for viruses"
    interstitial for large files instead of the file itself on a plain
    GET, requiring a confirmation step replayed on a second request - and
    the exact mechanism (cookie name, token, endpoint) has changed more
    than once over the years, which is exactly what broke this before.
    Strategy, most to least reliable:
      1) The `gdown` package, if installed - purpose-built for this and
         actively maintained against Google's changes.
      2) The current (2024+) drive.usercontent.google.com endpoint with
         confirm=t, which bypasses the interstitial for most files.
      3) The older drive.google.com/uc flow, scraping a confirm token from
         a cookie or the page body.
    Whichever path is used, the result is validated (size + not HTML)
    before being accepted - never silently writes a bad file."""
    tmp = dest_path + ".tmp"

    def _report(msg):
        if status_cb:
            status_cb(msg)

    # 1) gdown, if available.
    try:
        import gdown
        _report("Downloading via gdown...")
        gdown.download(id=file_id, output=tmp, quiet=True)
        if os.path.isfile(tmp) and os.path.getsize(tmp) > 1_000_000 and not _looks_like_html(tmp):
            os.replace(tmp, dest_path)
            return
        if os.path.isfile(tmp):
            os.remove(tmp)
    except ImportError:
        pass
    except Exception:
        if os.path.isfile(tmp):
            os.remove(tmp)

    try:
        import requests
    except ImportError:
        raise RuntimeError(
            "Downloading from Google Drive needs either the 'gdown' or "
            "'requests' package, and neither is installed. Install one "
            "(pip install gdown - recommended, handles Google Drive's "
            f"quirks directly), or download the file by hand from "
            f"{HUMAN_MODEL_GDRIVE_VIEW_URL} and place it at: {dest_path}"
        )

    def _stream_to_tmp(response):
        downloaded = 0
        last_reported_mb = -20
        with open(tmp, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                mb = downloaded // (1024 * 1024)
                if mb >= last_reported_mb + 20:
                    _report(f"Downloading... {mb}MB")
                    last_reported_mb = mb

    session = requests.Session()

    # 2) Modern endpoint - confirm=t alone bypasses the interstitial for
    # most files without needing to scrape any token first.
    _report("Downloading...")
    try:
        response = session.get(
            "https://drive.usercontent.google.com/download",
            params={"id": file_id, "export": "download", "confirm": "t"},
            stream=True,
        )
        response.raise_for_status()
        _stream_to_tmp(response)
        if os.path.getsize(tmp) > 1_000_000 and not _looks_like_html(tmp):
            os.replace(tmp, dest_path)
            return
        if os.path.isfile(tmp):
            os.remove(tmp)
    except Exception:
        if os.path.isfile(tmp):
            os.remove(tmp)

    # 3) Legacy endpoint - scrape a confirmation token from a cookie or
    # the interstitial page body and replay it.
    url = "https://drive.google.com/uc?export=download"
    response = session.get(url, params={"id": file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break
    if token is None:
        import re
        m = re.search(r"confirm=([0-9A-Za-z_-]+)", response.text or "")
        if m:
            token = m.group(1)
    if token:
        response = session.get(url, params={"id": file_id, "confirm": token}, stream=True)
    response.raise_for_status()
    _stream_to_tmp(response)

    if not os.path.isfile(tmp) or os.path.getsize(tmp) < 1_000_000 or _looks_like_html(tmp):
        if os.path.isfile(tmp):
            os.remove(tmp)
        raise RuntimeError(
            "Google Drive returned a page instead of the actual file on "
            "every attempted method (this usually means its per-file "
            "download quota is temporarily exceeded, or Google changed "
            "the confirmation flow again) - installing 'gdown' "
            "(pip install gdown) is the most reliable fix, since it's "
            "maintained specifically to track these changes; otherwise "
            f"download the file by hand from {HUMAN_MODEL_GDRIVE_VIEW_URL} "
            f"and place it at: {dest_path}"
        )
    os.replace(tmp, dest_path)


def _ensure_model_on_disk(model_filename, unique_id=None):
    path = os.path.join(_stylegan_models_dir(), model_filename)
    if os.path.isfile(path):
        if _looks_like_html(path):
            # Self-heal: a previous auto-download attempt (before this
            # validation existed, or one that somehow still slipped
            # through) can leave an HTML error/quota page sitting where
            # the real .pkl should be - pickle.load then fails on it with
            # an opaque "invalid load key" error forever after, since this
            # function only downloads when the file is *missing*. Treat it
            # as missing instead of trusting its mere presence.
            os.remove(path)
        else:
            return path

    info = _KNOWN_AUTO_DOWNLOAD_MODELS.get(model_filename)
    if info is None:
        raise FileNotFoundError(
            f"StyleGAN model not found: {path}. Place the .pkl there, or pick "
            f"one of the known auto-download models from the combo "
            f"({', '.join(_KNOWN_AUTO_DOWNLOAD_MODELS.keys())})."
        )
    kind, ref = info

    _toast(unique_id, "info", "StyleGAN", f"'{model_filename}' not found - downloading it (one-time)...")
    _send_status(unique_id, f"Downloading {model_filename}...")
    try:
        if kind == "nvidia":
            torch.hub.download_url_to_file(ref, path)
            if _looks_like_html(path):
                raise RuntimeError("received an HTML page instead of the model file")
        else:
            _download_from_google_drive(ref, path, status_cb=lambda t: _send_status(unique_id, t))
    except Exception as e:
        if os.path.isfile(path):
            os.remove(path)  # don't leave a partial/corrupt file behind
        manual_url = ref if kind == "nvidia" else HUMAN_MODEL_GDRIVE_VIEW_URL
        raise RuntimeError(
            f"Couldn't auto-download {model_filename} ({e}). If this "
            f"machine has no internet access (or Google Drive's download "
            f"quota is temporarily exceeded, which happens on shared/heavily "
            f"downloaded files), download it by hand from {manual_url} and "
            f"place it at: {path}"
        )
    _send_status(unique_id, "Download complete.")
    return path


def _list_vector_npy_names():
    base = _vectors_dir()
    try:
        return sorted(
            os.path.splitext(f)[0] for f in os.listdir(base) if f.lower().endswith(".npy")
        )
    except FileNotFoundError:
        return []


def _vector_npy_path(name):
    return os.path.join(_vectors_dir(), f"{name}.npy")


def _vector_json_path(name):
    return os.path.join(_vectors_dir(), f"{name}.json")


def _sanitize_vector_name(name):
    name = (name or "").strip()
    keep = "-_. "
    name = "".join(c for c in name if c.isalnum() or c in keep).strip()
    return name


def _read_vector_json_raw(name):
    """Whatever's actually in <name>.json, unfiltered (layer params AND
    any recorded generation-provenance fields) - {} if missing/corrupt."""
    p = _vector_json_path(name)
    if os.path.isfile(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _write_vector_json_raw(name, data):
    p = _vector_json_path(name)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)


def _load_vector_params(name):
    """The layer_start/layer_end/step/min/max used to actually apply the
    vector. Missing/partial keys always fall back to the defaults, so old/
    hand-edited json files (or ones with only generation-provenance fields
    so far) never crash the node."""
    data = _read_vector_json_raw(name)
    params = dict(_DEFAULT_VECTOR_PARAMS)
    for k in _DEFAULT_VECTOR_PARAMS:
        if k in data:
            params[k] = data[k]
    return params


def _save_vector_params(name, params):
    """Updates just the layer_start/layer_end/step/min/max fields (the
    Save button) - preserves any generation-provenance fields already
    recorded in the file instead of wiping them out."""
    clean = {
        "layer_start": int(params.get("layer_start", 0)),
        "layer_end": int(params.get("layer_end", _NUM_LAYERS_DEFAULT - 1)),
        "step": float(params.get("step", 0.25)),
        "min": float(params.get("min", -2.0)),
        "max": float(params.get("max", 2.0)),
    }
    data = _read_vector_json_raw(name)
    data.update(clean)
    _write_vector_json_raw(name, data)
    return clean


def _save_vector_generation_params(name, generation):
    """Records provenance (source_mode/seeds/projection settings - not the
    images themselves, and not their filenames) alongside whatever layer/
    step/range params already exist for this vector - called once, right
    when the .npy itself is written. Never touches layer/step/range."""
    if not generation:
        return
    clean = {k: generation[k] for k in _GENERATION_PARAM_KEYS if k in generation}
    if not clean:
        return
    data = _read_vector_json_raw(name)
    data.update(clean)
    _write_vector_json_raw(name, data)


def _all_vectors_with_params():
    """{name: params} for every .npy currently in facegen_vectors/, read
    fresh from disk every call - this is what makes 'add a vector, hit
    refresh, no restart' work."""
    return {name: _load_vector_params(name) for name in _list_vector_npy_names()}


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


def _send_status(node_id, text):
    if PromptServer is None or getattr(PromptServer, "instance", None) is None:
        return
    try:
        PromptServer.instance.send_sync("mykee.stylegan_status", {"node": node_id, "text": text})
    except Exception:
        pass


def _disarm(node_id):
    """Tells the Vector Save node's JS to flip its 'armed' checkbox back
    off. Called as the very first thing save_vector does, before any
    saving logic - so even if this run then fails validation (bad name,
    overwrite refused, etc.), a later plain Run/Queue press never saves
    again by accident just because 'armed' was left on from this attempt."""
    if PromptServer is None or getattr(PromptServer, "instance", None) is None:
        return
    try:
        PromptServer.instance.send_sync("mykee.stylegan_disarm", {"node": node_id})
    except Exception:
        pass


def _notify_vector_saved(node_id):
    """Tells the Vector Save node's JS a save just completed, so it can
    refresh its "Existing vectors" list - the newly (over)written name
    should show up there without needing a manual refresh/Load click."""
    if PromptServer is None or getattr(PromptServer, "instance", None) is None:
        return
    try:
        PromptServer.instance.send_sync("mykee.stylegan_vector_saved", {"node": node_id})
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Generator loading (cached) + image <-> tensor helpers
# ---------------------------------------------------------------------------

_G_CACHE = {}


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_generator(model_filename, unique_id=None):
    """Loads G_ema straight out of the .pkl with plain pickle - same as the
    original Gradio app's load_model(): no legacy.load_network_pkl call, the
    network classes reconstruct themselves via torch_utils/persistence.py as
    long as dnnlib + torch_utils are importable (see sys.path setup above).
    Auto-downloads the default FFHQ model first if the folder is empty."""
    path = _ensure_model_on_disk(model_filename, unique_id=unique_id)

    key = os.path.abspath(path)
    cached = _G_CACHE.get(key)
    if cached is not None:
        return cached

    with open(path, "rb") as f:
        data = pickle.load(f)
    G = data["G_ema"].to(_device()).eval()
    _G_CACHE[key] = G
    return G


def _image_tensor_to_pil(img_tensor):
    """ComfyUI IMAGE (1,H,W,C float 0..1 RGB) -> PIL.Image RGB."""
    arr = img_tensor.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _pil_to_image_tensor(img):
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _synth_w_to_pil(G, w):
    """w: [1, num_ws, 512] -> PIL.Image RGB at the generator's native res."""
    with torch.no_grad():
        img = G.synthesis(w, noise_mode="const")
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return Image.fromarray(img[0].cpu().numpy(), "RGB")


# ---------------------------------------------------------------------------
# Vector application (Face Generator node)
# ---------------------------------------------------------------------------

def _get_vector_tensor(name, num_ws, device):
    p = _vector_npy_path(name)
    if not os.path.isfile(p):
        return None
    v = torch.from_numpy(np.load(p)).to(device).to(torch.float32)
    if v.numel() == num_ws * 512:
        return v.view(1, num_ws, 512)
    return v.view(1, 1, 512).repeat(1, num_ws, 1)


def _apply_vectors(w, vector_values):
    """vector_values: {name: slider_value}. Reads the vector .npy + its
    current json params fresh from disk, applies:
        w[:, start:end+1, :] += vec[:, start:end+1, :] * value * step
    for every named vector that both has a slider value != 0 and an .npy
    file on disk (silently skips names the UI sent that no longer exist -
    e.g. a vector the user deleted since the node was last refreshed)."""
    device = w.device
    num_ws = w.shape[1]
    for name, value in (vector_values or {}).items():
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value == 0.0:
            continue
        vec = _get_vector_tensor(name, num_ws, device)
        if vec is None:
            continue
        params = _load_vector_params(name)
        start = max(0, min(num_ws - 1, int(params["layer_start"])))
        end = max(start, min(num_ws - 1, int(params["layer_end"])))
        step = float(params["step"])
        w[:, start:end + 1, :] = w[:, start:end + 1, :] + (vec[:, start:end + 1, :] * value * step)
    return w


class MykeeStyleGANFaceGenerator:
    """Generates an image from scratch, from a StyleGAN2/3 generator, a
    seed and a set of direction-vector sliders that are discovered from
    models/stylegan/facegen_vectors/ at edit time (client-side refresh,
    see web/mykee_stylegan.js) and applied at run time (server-side,
    always re-read from disk so a vector saved a moment ago is honored
    even without clicking refresh first). Named "Facegenerator"
    internally/in facegen_vectors/ since faces are what this pack's
    vectors are built for, but any StyleGAN2/3 generator works the same
    way regardless of what it was trained to produce - display name is
    "StyleGAN Image Generation" for that reason. For editing an existing
    photo instead of generating from a seed, see MykeeStyleGANEditor."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_list_pkl_models(), {"tooltip": "A .pkl file from ComfyUI/models/stylegan. Click the refresh button on the node to rescan after adding a new one."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": True}),
                "truncation": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.5, "step": 0.01, "tooltip": "StyleGAN truncation_psi. Lower = safer/more average faces, higher = more variety/more extreme faces."}),
                # Hidden/synthetic: web/mykee_stylegan.js builds one slider per
                # discovered vector and keeps this JSON string in sync; it is
                # not meant to be hand-edited.
                "vector_values": ("STRING", {"default": "{}", "multiline": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Mykee/StyleGAN"

    def generate(self, model, seed, truncation, vector_values, unique_id=None):
        try:
            vector_values = json.loads(vector_values) if vector_values else {}
        except Exception:
            vector_values = {}

        G = _load_generator(model, unique_id=unique_id)
        device = _device()

        with torch.no_grad():
            rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
            z = torch.from_numpy(rng.randn(1, G.z_dim)).to(device)
            w = G.mapping(z, None, truncation_psi=truncation)
            w = _apply_vectors(w, vector_values)
            img = _synth_w_to_pil(G, w)

        return (_pil_to_image_tensor(img),)


# ---------------------------------------------------------------------------
# GAN inversion (projection) for the Vector Editor node
# ---------------------------------------------------------------------------

_VGG16_CACHE = {"model": None}


def _load_vgg16(device):
    if _VGG16_CACHE["model"] is not None:
        return _VGG16_CACHE["model"]

    path = _vgg16_path()
    if not os.path.isfile(path):
        try:
            torch.hub.download_url_to_file(VGG16_URL, path)
        except Exception as e:
            raise RuntimeError(
                "Couldn't download the VGG16 feature network needed for GAN "
                f"inversion ({VGG16_URL}) - {e}. If this machine has no "
                f"internet access, download the file manually and place it "
                f"at: {path}"
            )
    with open(path, "rb") as f:
        vgg16 = torch.jit.load(f).eval().to(device)
    _VGG16_CACHE["model"] = vgg16
    return vgg16


def _prep_target_for_vgg(pil_img, target_res, device):
    img = pil_img.convert("RGB").resize((target_res, target_res), Image.LANCZOS)
    arr = np.array(img, dtype=np.uint8).transpose([2, 0, 1])  # CHW, 0..255
    return torch.from_numpy(arr).unsqueeze(0).to(device).to(torch.float32)


@contextlib.contextmanager
def _deterministic_cuda():
    """Best-effort reproducibility for the GAN-inversion optimization:
    seeding the RNG (see `rng` below) fixes which random numbers get
    drawn, but cuDNN's convolution algorithms are *not* deterministic by
    default even with a fixed seed - GPU-parallel reduction order can
    still vary slightly step to step, and across 300+ optimization steps
    those tiny differences compound into a visibly different final
    vector. This forces deterministic algorithms where available for the
    duration of the block (warn_only=True: an op with no deterministic
    implementation just warns and falls back instead of raising), and
    restores the previous global settings afterward so this doesn't
    quietly slow down or change behavior for unrelated nodes running
    later in the same ComfyUI process."""
    prev_deterministic = torch.backends.cudnn.deterministic
    prev_benchmark = torch.backends.cudnn.benchmark
    prev_algorithms = torch.are_deterministic_algorithms_enabled()
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        yield
    finally:
        torch.backends.cudnn.deterministic = prev_deterministic
        torch.backends.cudnn.benchmark = prev_benchmark
        torch.use_deterministic_algorithms(prev_algorithms, warn_only=True)


def _project_to_w(G, target_pil, device, num_steps, status_cb=None, seed=0,
                   w_avg_samples=2000, initial_learning_rate=0.1,
                   initial_noise_factor=0.05, lr_rampdown_length=0.25,
                   lr_rampup_length=0.05, noise_ramp_length=0.75,
                   regularize_noise_weight=1e5):
    """GAN inversion via latent optimization - NVIDIA's stylegan2-ada/
    stylegan3 projector.py algorithm: optimize a single [1,1,512] w
    (broadcast across all layers), matching VGG16 ("LPIPS-ish") feature
    distance to the target image, with noise-buffer regularization.
    Returns the final w broadcast to [1, num_ws, 512].

    Deterministic given the same G/target_pil/num_steps/seed: the noise
    buffer initialization and the per-step latent noise injection below
    both draw from a Generator seeded with `seed`, instead of PyTorch's
    unseeded global RNG - without this, the exact same two input images
    and the exact same parameters would still produce a different vector
    on every run.

    ComfyUI's executor runs node FUNCTIONs inside torch.inference_mode(),
    which permanently strips autograd tracking from every tensor created
    under it - including our w_opt, no matter that it's built with
    requires_grad=True. torch.inference_mode(False) below re-enables
    normal (non-inference) tensors for this whole optimization, which is
    the standard fix for any training/optimization loop running as a
    ComfyUI node.

    G itself was loaded (and its buffers/params moved to device) back
    under the executor's default inference_mode, so its noise_const
    buffers are themselves inference tensors - in-place writes to them
    (needed every optimization step) would still fail even inside
    inference_mode(False), since "inference-ness" is fixed at a tensor's
    creation, not by the current context. NVIDIA's official projector.py
    sidesteps this the same way we do here: deepcopy G first, inside the
    now-enabled autograd context, so every buffer we touch is a fresh,
    ordinary (non-inference) tensor - which also protects the Face
    Generator/other calls' shared cached G from having its noise state
    permanently altered by this optimization."""
    with torch.inference_mode(False), torch.enable_grad(), _deterministic_cuda():
        G = copy.deepcopy(G).eval().requires_grad_(False).to(device)
        vgg16 = _load_vgg16(device)

        # W statistics (mean/std of the mapping network's output), used as
        # the optimization's starting point and to scale the per-step
        # latent noise.
        if status_cb:
            status_cb("Sampling W statistics...")
        z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
        with torch.no_grad():
            w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)
        w_avg = np.mean(w_samples, axis=0, keepdims=True)
        w_std = float(np.sqrt(np.sum((w_samples - w_avg) ** 2) / w_avg_samples))

        noise_bufs = {n: buf for n, buf in G.synthesis.named_buffers() if "noise_const" in n}

        target_res = G.img_resolution
        target_images = _prep_target_for_vgg(target_pil, target_res, device)
        if target_images.shape[2] > 256:
            target_images = F.interpolate(target_images, size=(256, 256), mode="area")
        with torch.no_grad():
            target_features = vgg16(target_images, resize_images=False, return_lpips=True)

        w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True)
        optimizer = torch.optim.Adam(
            [w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate
        )

        # A dedicated Generator (not PyTorch's unseeded global RNG) so the
        # exact same inputs/parameters always retrace the exact same
        # optimization path and land on the exact same final vector.
        rng = torch.Generator(device=device)
        rng.manual_seed(int(seed))

        for buf in noise_bufs.values():
            buf[:] = torch.randn(buf.shape, generator=rng, device=buf.device, dtype=buf.dtype)
            buf.requires_grad = True

        w_out = None
        for step in range(num_steps):
            _check_interrupted()
            t = step / max(1, num_steps)
            w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
            lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
            lr = initial_learning_rate * lr_ramp
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            w_noise = torch.randn(w_opt.shape, generator=rng, device=w_opt.device, dtype=w_opt.dtype) * w_noise_scale
            ws = (w_opt + w_noise).repeat(1, G.mapping.num_ws, 1)
            synth_images = G.synthesis(ws, noise_mode="const")
            synth_images = (synth_images + 1) * (255 / 2)
            if synth_images.shape[2] > 256:
                synth_images = F.interpolate(synth_images, size=(256, 256), mode="area")
            synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
            dist = (target_features - synth_features).square().sum()

            reg_loss = 0.0
            for v in noise_bufs.values():
                # noise_const buffers are 2D [H, W] - reshape to [1,1,H,W]
                # so roll(dims=2/3) and avg_pool2d (which expects a 4D
                # NCHW input) actually have those dimensions to operate on.
                noise = v[None, None, :, :]
                while True:
                    reg_loss = reg_loss + (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                    reg_loss = reg_loss + (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)
            loss = dist + reg_loss * regularize_noise_weight

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

            w_out = w_opt.detach().clone()

            if status_cb and (step % 25 == 0 or step == num_steps - 1):
                status_cb(f"Projecting... {step + 1}/{num_steps} (loss {float(loss):.3f})")

        return w_out.repeat(1, G.mapping.num_ws, 1)


def _image_fingerprint(image_tensor):
    """Cheap, approximate fingerprint of an IMAGE tensor - good enough to
    tell "probably the same picture" from "definitely a different one" for
    cache-invalidation purposes (not a security context), without the cost
    of hashing every pixel."""
    t = image_tensor.detach().float()
    return (tuple(t.shape), round(float(t.mean()), 6), round(float(t.std()), 6))


# Per-node-instance (keyed by unique_id) last projected W for
# MykeeStyleGANEditor's keep_cache option - {"key": (...), "w": tensor}.
# Unlike MykeeStyleGANVectorPreview's whole-result cache, only the
# (expensive) projection is cached here: vector sliders still apply fresh
# on every run, only the projection itself is skipped when the image/
# model/projection settings match the last run for this node.
_EDITOR_PROJECTION_CACHE = {}


class MykeeStyleGANEditor:
    """Edits an existing photo with the same direction vectors the Image
    Generation node uses - GAN inversion (project the photo into W space,
    the same algorithm as MykeeStyleGANVectorPreview's image mode) followed
    by applying the current vector sliders and re-synthesizing. There's no
    truncation setting here (unlike Image Generation): truncation_psi only
    means something for G.mapping()'s seed->W step, which this node never
    calls - it starts from the photo's own projected W instead.

    keep_cache (on by default) remembers the last projected W per node
    instance and reuses it - skipping the slow projection entirely -
    whenever the image/model/projection_steps/projection_seed all still
    match what produced it; only the vector sliders need to differ for a
    quick re-run. Change the image, model, or projection settings (or turn
    keep_cache off) to force a fresh projection."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The photo to edit."}),
                "model": (_list_pkl_models(), {"tooltip": "A .pkl file from ComfyUI/models/stylegan. Click the refresh button on the node to rescan after adding a new one."}),
                "keep_cache": ("BOOLEAN", {"default": True, "tooltip": "While on, re-running this node reuses the last projection of this image (skipping the slow part) as long as the image/model/projection_steps/projection_seed haven't changed - only the vector sliders need to differ for a fast re-run."}),
                "projection_steps": ("INT", {"default": 300, "min": 50, "max": 2000, "step": 50, "tooltip": "GAN-inversion iterations. More = closer match to the original photo, slower."}),
                "projection_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "tooltip": "Seeds the projection's internal noise so the exact same photo + settings always produce the exact same starting point."}),
                # Hidden/synthetic: web/mykee_stylegan.js builds one slider
                # per discovered vector and keeps this JSON string in sync;
                # it is not meant to be hand-edited. Same mechanism as the
                # Image Generation node.
                "vector_values": ("STRING", {"default": "{}", "multiline": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "edit_image"
    CATEGORY = "Mykee/StyleGAN"

    def edit_image(self, image, model, keep_cache, projection_steps, projection_seed, vector_values, unique_id=None):
        try:
            vector_values = json.loads(vector_values) if vector_values else {}
        except Exception:
            vector_values = {}

        def status(text):
            _send_status(unique_id, text)

        G = _load_generator(model, unique_id=unique_id)
        device = _device()

        cache_key = (model, int(projection_steps), int(projection_seed), _image_fingerprint(image))
        node_key = str(unique_id) if unique_id is not None else None
        cached = _EDITOR_PROJECTION_CACHE.get(node_key) if node_key is not None else None

        if keep_cache and cached is not None and cached["key"] == cache_key:
            w = cached["w"]
            status("Using cached projection (keep_cache is on) - applying current vectors.")
        else:
            status("Projecting image...")
            w = _project_to_w(G, _image_tensor_to_pil(image), device, int(projection_steps), status_cb=status, seed=projection_seed)
            if node_key is not None:
                _EDITOR_PROJECTION_CACHE[node_key] = {"key": cache_key, "w": w}

        with torch.no_grad():
            status("Applying vectors...")
            w_edit = _apply_vectors(w.clone(), vector_values)
            img = _synth_w_to_pil(G, w_edit)

        status("Done.")
        return (_pil_to_image_tensor(img),)


# Per-node-instance (keyed by unique_id) last result for MykeeStyleGANVectorPreview's
# keep_cache option - deliberately separate from ComfyUI's own graph cache,
# since some upstream image-source nodes defeat that by always reporting
# "changed" even when the image content is identical.
_PREVIEW_RESULT_CACHE = {}


class MykeeStyleGANVectorPreview:
    """Builds the two W-space extremes of a new direction vector and shows
    a preview - no filesystem writes happen here at all. Wire the "w_pair"
    output into a Mykee StyleGAN Vector Save node when you're happy with
    the preview; nothing gets saved until that node actually runs.

    source_mode picks how w_a/w_b are produced:
      - "image": image_a/image_b (two connected photos) get GAN-inverted
        into W space. Slow (a real optimization loop per image) but works
        from arbitrary real photos.
      - "seed": seed_a/seed_b are mapped straight to W - no photos, no
        projection, instant - the same trick the original generate2.py's
        seed-pair vector builder used. Only as good as how well-matched
        the two seeds happen to be; use the node's own "Randomize seed_a"/
        "Randomize seed_b" buttons to search for a good pair (like the old
        script's dice button did - NOT ComfyUI's built-in
        control_after_generate, which fights itself when two seed widgets
        on the same node both have it) and just re-queue to preview a new
        pair - nothing is written to disk by this node no matter how many
        times you re-roll.

    keep_cache (on by default) makes this node remember its own last
    result per-node-instance and just replay it instantly instead of
    recomputing, independent of ComfyUI's own graph caching - which some
    upstream image-source nodes defeat by always reporting "changed" even
    when the image content is identical, forcing an expensive
    re-projection on every single queue. Turn it off to force a fresh
    computation (new seeds, new photos), then back on once you're happy
    with the result and just want repeated queues (e.g. for the Save
    node) to reuse it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_mode": (["image", "seed"], {"default": "image", "tooltip": "image: invert image_a/image_b (real photos) into W space. seed: map seed_a/seed_b straight to W - instant, no photos/projection needed, but only as good as how well-matched the two seeds are."}),
                "model": (_list_pkl_models(), {"tooltip": "Generator used to build the vector - must match the one you'll apply the vector with."}),
                "keep_cache": ("BOOLEAN", {"default": True, "tooltip": "While on, re-running this node just replays its last result instantly instead of recomputing - independent of (and more reliable than) ComfyUI's own caching. Turn off to force a fresh computation (new seeds/photos), then back on once you're happy with the result."}),
                "seed_a": ("INT", {"default": 100, "min": 0, "max": 0xFFFFFFFF, "tooltip": "Only used when source_mode = seed. Extreme A. Use the node's own 'Randomize seed_a' button to reroll (not ComfyUI's built-in control_after_generate - two of those on one node ended up fighting each other)."}),
                "seed_b": ("INT", {"default": 200, "min": 0, "max": 0xFFFFFFFF, "tooltip": "Only used when source_mode = seed. Extreme B - the vector points from A to B. Use the node's own 'Randomize seed_b' button to reroll."}),
                "seed_truncation": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.5, "step": 0.01, "tooltip": "Only used when source_mode = seed. truncation_psi for seed_a/seed_b's mapping (matches the original generate2.py default)."}),
                "projection_steps": ("INT", {"default": 300, "min": 50, "max": 2000, "step": 50, "tooltip": "Only used when source_mode = image. GAN-inversion iterations per image. More = closer match, slower."}),
                "projection_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "tooltip": "Only used when source_mode = image. Seeds the optimization's internal noise so the exact same images + parameters always produce the exact same vector. Change it to explore a different optimization path for the same images."}),
            },
            "optional": {
                "image_a": ("IMAGE", {"tooltip": "Only used when source_mode = image. Vector extreme A (e.g. short hair)."}),
                "image_b": ("IMAGE", {"tooltip": "Only used when source_mode = image. Vector extreme B (e.g. long hair)."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "MYKEE_STYLEGAN_WPAIR")
    RETURN_NAMES = ("preview", "w_pair")
    FUNCTION = "preview_vector"
    CATEGORY = "Mykee/StyleGAN"

    def preview_vector(self, source_mode, model, keep_cache, seed_a, seed_b, seed_truncation, projection_steps,
                        projection_seed, image_a=None, image_b=None, unique_id=None):
        cache_key = str(unique_id) if unique_id is not None else None
        if keep_cache and cache_key is not None and cache_key in _PREVIEW_RESULT_CACHE:
            _send_status(unique_id, "Using cached result (keep_cache is on) - turn it off to recompute.")
            return _PREVIEW_RESULT_CACHE[cache_key]

        def status(text):
            _send_status(unique_id, text)

        G = _load_generator(model, unique_id=unique_id)
        device = _device()

        if source_mode == "seed":
            with torch.no_grad():
                status("Computing W from seeds...")
                rng_a = np.random.RandomState(int(seed_a) & 0xFFFFFFFF)
                z_a = torch.from_numpy(rng_a.randn(1, G.z_dim)).to(device)
                w_a = G.mapping(z_a, None, truncation_psi=seed_truncation)

                rng_b = np.random.RandomState(int(seed_b) & 0xFFFFFFFF)
                z_b = torch.from_numpy(rng_b.randn(1, G.z_dim)).to(device)
                w_b = G.mapping(z_b, None, truncation_psi=seed_truncation)
        else:
            if image_a is None or image_b is None:
                raise ValueError("source_mode = 'image' needs both image_a and image_b connected.")
            status("Projecting image A...")
            w_a = _project_to_w(G, _image_tensor_to_pil(image_a), device, int(projection_steps), status_cb=status, seed=projection_seed)
            status("Projecting image B...")
            w_b = _project_to_w(G, _image_tensor_to_pil(image_b), device, int(projection_steps), status_cb=status, seed=projection_seed)

        status("Rendering preview...")
        recon_a = _synth_w_to_pil(G, w_a)
        recon_b = _synth_w_to_pil(G, w_b)
        w = max(recon_a.width, recon_b.width)
        h = max(recon_a.height, recon_b.height)
        strip = Image.new("RGB", (w * 2, h))
        strip.paste(recon_a, (0, 0))
        strip.paste(recon_b, (w, 0))
        preview_tensor = _pil_to_image_tensor(strip)

        status("Done - not saved. Connect w_pair to a Vector Save node to save it.")
        generation_params = {
            "source_mode": source_mode,
            "seed_a": int(seed_a),
            "seed_b": int(seed_b),
            "seed_truncation": float(seed_truncation),
            "projection_steps": int(projection_steps),
            "projection_seed": int(projection_seed),
        }
        w_pair = {
            "w_a": w_a.detach(), "w_b": w_b.detach(), "preview": preview_tensor,
            "generation_params": generation_params,
        }
        result = (preview_tensor, w_pair)
        if cache_key is not None:
            _PREVIEW_RESULT_CACHE[cache_key] = result
        return result


class MykeeStyleGANVectorSave:
    """Takes a w_pair from a Mykee StyleGAN Vector Preview node and saves
    diff = w_b - w_a as <vector_name>.npy plus <vector_name>.json (the
    layer range / multiplier / slider range fields below) - but ONLY when
    'armed' is on. OUTPUT_NODE nodes run on ANY queue of the graph they're
    part of - including the plain toolbar Run/Queue button, not just this
    node's own 'Save vector' button - so without this gate, simply hitting
    Run while this node is wired up would silently save every time. The
    '✨ Save vector (queue prompt)' button turns 'armed' on and queues in
    one click; the node turns it back off again the instant it runs
    (before doing anything else), so a later plain Run never re-saves by
    itself. w_pair is an OPTIONAL input - leave it disconnected entirely
    if you're only using 'Load'/'Save' to browse/edit existing vectors'
    JSON, which don't need it, 'armed', or the graph to run at all, and
    can't overwrite an existing .npy. A save refuses to overwrite an
    existing <vector_name>.npy unless allow_overwrite is also on."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vector_name": ("STRING", {"default": "", "tooltip": "Required. File name (without extension) for the .npy/.json pair."}),
                "armed": ("BOOLEAN", {"default": False, "tooltip": "Safety gate - stays OFF normally so a plain Run/Queue press never saves by accident (OUTPUT_NODEs run on every queue, not just when you click this node's own button). The '✨ Save vector' button turns this on, queues, and the node turns it back off right after running. You can turn it on by hand if you specifically want one plain Run to also save."}),
                "allow_overwrite": ("BOOLEAN", {"default": False, "tooltip": "Must be on to save over a vector_name whose .npy already exists - protects an existing vector from being silently overwritten. Saving parameter changes with the Save button (JSON only) is unaffected by this."}),
                "layer_start": ("INT", {"default": 0, "min": 0, "max": 31}),
                "layer_end": ("INT", {"default": _NUM_LAYERS_DEFAULT - 1, "min": 0, "max": 31}),
                "step": ("FLOAT", {"default": 0.25, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "Multiplier applied on top of the Face Generator's slider value."}),
                "min_value": ("FLOAT", {"default": -2.0, "min": -100.0, "max": 100.0, "step": 0.1}),
                "max_value": ("FLOAT", {"default": 2.0, "min": -100.0, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                "w_pair": ("MYKEE_STYLEGAN_WPAIR", {"tooltip": "Only needed to actually save a new/updated vector (the '✨ Save vector' button). Connect this from a Mykee StyleGAN Vector Preview node's w_pair output. Not required for 📂 Load / 💾 Save, which only touch the JSON and don't need the graph to run at all."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_vector"
    CATEGORY = "Mykee/StyleGAN"
    OUTPUT_NODE = True

    def save_vector(self, vector_name, armed, allow_overwrite, layer_start, layer_end,
                     step, min_value, max_value, w_pair=None, unique_id=None):
        # Disarm immediately, before any validation/saving - see the class
        # docstring for why this has to happen first, not last.
        _disarm(unique_id)

        if not armed:
            _send_status(unique_id, "Not saved yet - use the '✨ Save vector' button below to save.")
            return ()

        if w_pair is None:
            raise ValueError(
                "w_pair isn't connected - connect a Mykee StyleGAN Vector "
                "Preview node's w_pair output to actually save a vector. "
                "(Not needed for 📂 Load / 💾 Save, which only touch the "
                "JSON.)"
            )

        name = _sanitize_vector_name(vector_name)
        if not name:
            raise ValueError("vector_name is required to save a vector.")

        npy_path = _vector_npy_path(name)
        if os.path.isfile(npy_path) and not allow_overwrite:
            raise ValueError(
                f"'{name}.npy' already exists - refusing to overwrite it. "
                f"Turn on 'allow_overwrite' to replace it, or pick a "
                f"different vector_name. (The Save button only touches "
                f"'{name}.json' and is unaffected by this.)"
            )

        diff = (w_pair["w_b"] - w_pair["w_a"]).detach().cpu().numpy()
        np.save(npy_path, diff)
        saved_params = _save_vector_params(name, {
            "layer_start": layer_start, "layer_end": layer_end,
            "step": step, "min": min_value, "max": max_value,
        })
        _save_vector_generation_params(name, w_pair.get("generation_params"))

        msg = f"Saved '{name}.npy' + '{name}.json' ({saved_params})"
        _send_status(unique_id, "Saved.")
        _toast(unique_id, "success", "StyleGAN Vector Save", msg)
        _notify_vector_saved(unique_id)
        return ()


# ---------------------------------------------------------------------------
# REST routes - vector discovery/params for both nodes' JS, model rescans
# ---------------------------------------------------------------------------

try:
    from aiohttp import web as _aiohttp_web

    _routes = PromptServer.instance.routes

    @_routes.get("/mykee/stylegan/models")
    async def _mykee_sg_models(request):
        return _aiohttp_web.json_response({"models": _list_pkl_models()})

    @_routes.get("/mykee/stylegan/vectors")
    async def _mykee_sg_vectors(request):
        return _aiohttp_web.json_response({"vectors": _all_vectors_with_params()})

    @_routes.post("/mykee/stylegan/save_vector_params")
    async def _mykee_sg_save_vector_params(request):
        try:
            data = await request.json()
        except Exception:
            return _aiohttp_web.json_response({"error": "invalid JSON body"}, status=400)
        name = _sanitize_vector_name(data.get("name", ""))
        if not name:
            return _aiohttp_web.json_response({"error": "name is required"}, status=400)
        if not os.path.isfile(_vector_npy_path(name)):
            return _aiohttp_web.json_response(
                {"error": f"No '{name}.npy' found - generate the vector first."}, status=400
            )
        try:
            saved = _save_vector_params(name, data)
        except Exception as e:
            return _aiohttp_web.json_response({"error": str(e)}, status=400)
        return _aiohttp_web.json_response({"saved": saved, "vectors": _all_vectors_with_params()})

except Exception:
    # No running PromptServer (e.g. module imported outside ComfyUI) - the
    # nodes still work, refresh buttons just won't have anything to call.
    pass


NODE_CLASS_MAPPINGS = {
    "MykeeStyleGANFaceGenerator": MykeeStyleGANFaceGenerator,
    "MykeeStyleGANEditor": MykeeStyleGANEditor,
    "MykeeStyleGANVectorPreview": MykeeStyleGANVectorPreview,
    "MykeeStyleGANVectorSave": MykeeStyleGANVectorSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeStyleGANFaceGenerator": "StyleGAN Image Generation",
    "MykeeStyleGANEditor": "StyleGAN Editor",
    "MykeeStyleGANVectorPreview": "StyleGAN Vector Preparation",
    "MykeeStyleGANVectorSave": "StyleGAN Vector Save",
}
