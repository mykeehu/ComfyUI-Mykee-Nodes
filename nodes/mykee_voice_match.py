"""
ComfyUI-Mykee-Nodes / voice & accent match node

Universal, model-independent post-processor: takes ANY generated audio plus
a reference sample and (optionally) reshapes the generated voice's timbre
and/or pitch-level/accent character to match the reference - regardless of
what produced the generated audio (Higgs, MiniMax, a local Bark checkpoint,
whatever). This is deliberately NOT wired into any specific TTS model.

Pipeline:
    1. Source separation (Demucs) on both inputs, so background music/SFX
       mixed into either track doesn't get treated as "voice".
    2. Voice conversion (Seed-VC, https://github.com/Plachtaa/seed-vc) on
       the isolated vocal stems - timbre is always transferred (Seed-VC has
       no toggle for this - see the docstring on MykeeVoiceAccentMatch);
       pitch-level/accent transfer is an additional, optional stage on top.
    3. Remix: the converted (or untouched) vocal stem is combined back with
       the generated audio's own separated background at a controllable
       level, reusing the same gain/duck machinery as Mykee Audio Merger.

Model weights are auto-downloaded, on first use, into shared, non-node-
specific folders under ComfyUI/models/ (whisper/, Seed-VC/, CAMpp/,
htdemucs/) so other node packs that expect models in those same
conventional locations can reuse them too - not squirreled away under this
node pack's own directory.

Everything in this file needs an NVIDIA GPU with a modern PyTorch/
transformers/demucs install to actually run - this is a fundamentally
heavier dependency than the pyworld-based nodes elsewhere in this pack.
"""

import os
import sys
import subprocess
import shutil
import stat

import numpy as np
import torch

import folder_paths
import comfy.model_management as mm
from comfy.model_patcher import ModelPatcher

try:
    from comfy.utils import ProgressBar as _ComfyProgressBar
except Exception:
    _ComfyProgressBar = None

from .mykee_audio import (
    _resample as _resample_torch,
    _match_channels,
    _fit_length,
    _db_to_lin,
    _block_rms,
    _envelope_follow,
    _upsample_curve,
)

# --- ComfyUI-managed model lifecycle -----------------------------------------
#
# The two models we have direct, plain-nn.Module access to in this pipeline -
# Seed-VC's DiT/CFM model + its CAM++ speaker encoder, and separately Demucs -
# are wrapped in ComfyUI's own comfy.model_patcher.ModelPatcher and registered
# via comfy.model_management.load_models_gpu(), the exact same VRAM
# (de)allocation system every built-in node (KSampler, CLIP, VAEs, ...) uses.
# This replaces an earlier hand-rolled cache + gc.collect()/torch.cuda.
# empty_cache() combo, for the same reasons this pack's MOSS-TTS nodes made
# the same switch: ComfyUI decides when to evict our models the same way it
# would for its own (instead of us guessing), reloads them transparently if
# they get evicted, and - the main win here - the model being resident via
# ComfyUI's own tracking is what lets the standard Cancel/Interrupt button
# and comfy.model_management.throw_exception_if_processing_interrupted()
# actually cooperate with a run in progress (see _check_interrupted() below).
#
# semantic_fn/f0_fn/vocoder_fn/mel_fn (closures returned by Seed-VC's own
# load_models(), wrapping further models - Whisper, RMVPE, BigVGAN - whose
# internals live in a git-cloned third-party repo not vendored into this
# file) stay on the previous manual lifecycle: dropped from the cache and
# left to Python's refcounting + gc.collect()/torch.cuda.empty_cache() to
# actually free the VRAM, same as MOSS-TTS's own "legacy path" for whatever
# doesn't fit the ModelPatcher shape. Their VRAM footprint is far smaller
# than the DiT/CAM++/Demucs models this now manages properly.

_patched_device_classes = {}


def _make_device_property_settable(model):
    """ComfyUI's ModelPatcher assumes the model it wraps has a plain,
    assignable `.device` attribute. Some model classes (mainly HuggingFace
    `transformers` ones) instead expose `.device` as a READ-ONLY property,
    which makes ModelPatcher's routine `self.model.device = device_to`
    bookkeeping raise an AttributeError. This is a no-op for plain
    nn.Module subclasses (Seed-VC's DiT/CAM++ models, Demucs's HTDemucs)
    that don't override `.device` at all - only kicks in for a class that
    actually needs it, and is cached per class so repeated loads reuse the
    same patched subclass instead of piling up a new one every time."""
    cls = type(model)
    if cls in _patched_device_classes:
        model.__class__ = _patched_device_classes[cls]
        return model
    device_attr = getattr(cls, "device", None)
    if not isinstance(device_attr, property) or device_attr.fset is not None:
        return model  # already assignable, or not a property at all - nothing to do
    patched_cls = type(
        cls.__name__ + "_ComfyDeviceSettable",
        (cls,),
        {"device": property(device_attr.fget, lambda self, value: None)},
    )
    _patched_device_classes[cls] = patched_cls
    model.__class__ = patched_cls
    return model


def _wrap_and_load_together(models, device):
    """Wraps each of `models` in its own ModelPatcher and registers ALL of
    them in a SINGLE load_models_gpu() call, so ComfyUI considers their
    COMBINED memory requirement atomically - either everything fits
    (evicting other, unrelated models as needed), or a clear out-of-memory
    error is raised, instead of one of ours getting silently, partially
    evicted later to make room for another (which happens if related
    models are registered via separate load_models_gpu() calls instead of
    one). Returns a list of ModelPatchers in the same order as `models`.
    Always force_full_load=True: these are plain nn.Module architectures,
    not built for ComfyUI's own layer-by-layer streaming low-vram path, so
    they must be loaded whole or not at all."""
    offload_device = mm.unet_offload_device()
    patchers = [
        ModelPatcher(_make_device_property_settable(m), load_device=device, offload_device=offload_device)
        for m in models
    ]
    mm.load_models_gpu(patchers, force_full_load=True)
    return patchers


def _wrap_seed_vc_model_container(model, campplus_model, device):
    """seed-vc's own load_models() returns `model` as a Munch NAMESPACE
    bundling several real nn.Module sub-components (model.cfm,
    model.length_regulator, and possibly others) as attributes, rather
    than being an nn.Module itself - confirmed by ComfyUI's ModelPatcher
    failing on model.state_dict() (Munch has no such method; it's a plain
    dict subclass with attribute-style access). So instead of wrapping
    `model` as a whole, this finds every actual nn.Module attribute on
    it, wraps EACH of those (together with campplus_model, in one
    load_models_gpu() call - see _wrap_and_load_together above) and
    points the namespace's attributes at the resulting patched module
    objects.

    Returns (model, campplus_model, patchers) - `model` is the SAME Munch
    container, with its nn.Module attributes swapped for their
    ComfyUI-managed equivalents; `campplus_model` is the patched CAM++
    module; `patchers` is every ModelPatcher created (for the caller to
    cache / reassert residency on / unload later).

    Also handles the case where `model` already IS a plain nn.Module
    directly (e.g. a future or different seed-vc version that doesn't use
    the Munch-namespace pattern) - wrapped directly, same as before.
    """
    if isinstance(model, torch.nn.Module):
        model_patcher, campplus_patcher = _wrap_and_load_together([model, campplus_model], device)
        return model_patcher.model, campplus_patcher.model, [model_patcher, campplus_patcher]

    items = model.items() if hasattr(model, "items") else vars(model).items()
    submodule_names = [name for name, value in items if isinstance(value, torch.nn.Module)]
    if not submodule_names:
        raise RuntimeError(
            "Seed-VC's loaded model container has no recognizable nn.Module sub-components to "
            "hand to ComfyUI's memory manager - the installed seed-vc version may have changed "
            "its internal structure since this node was written."
        )

    submodules = [getattr(model, name) for name in submodule_names]
    patchers = _wrap_and_load_together(submodules + [campplus_model], device)
    for name, patcher in zip(submodule_names, patchers[:-1]):
        setattr(model, name, patcher.model)  # Munch's __setattr__ also updates the underlying dict
    campplus_patcher = patchers[-1]
    return model, campplus_patcher.model, patchers


def _unload_patcher(patcher):
    """Explicitly moves a ComfyUI-managed patcher's weights back to its
    offload device right now, rather than waiting for ComfyUI's automatic
    eviction to reclaim the VRAM whenever something else happens to need
    it - used for the `unload_model` toggle, where the point is an
    immediate, guaranteed release."""
    if patcher is None:
        return
    try:
        patcher.unpatch_model(device_to=patcher.offload_device, unpatch_weights=True)
    except Exception as exc:
        print(
            f"[Mykee Voice/Accent Match] Note: ComfyUI-managed unload failed ({exc}); "
            "falling back to a manual .to('cpu')."
        )
        try:
            patcher.model.to("cpu")
        except Exception:
            pass


def _check_interrupted():
    """Raises if the user hit ComfyUI's Cancel/Interrupt button; a no-op
    otherwise. Call this periodically inside any loop that could run long
    (once per Seed-VC conversion chunk, once per pipeline stage) so an
    interrupt actually takes effect while this node is mid-run, instead of
    only being honored between whole node executions (ComfyUI's default,
    since a single call into this node is normally treated as one
    atomic/blocking step). Silently does nothing on a ComfyUI version too
    old to have this function."""
    check = getattr(mm, "throw_exception_if_processing_interrupted", None)
    if check is not None:
        check()


_SEED_VC_REPO_URL = "https://github.com/Plachtaa/seed-vc.git"
_SEED_VC_REPO_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "seed-vc"
)  # pack root/third_party/seed-vc (this file lives in nodes/)

_MODEL_VARIANTS = {
    "voice_only (whisper-small, lighter, timbre only)": {
        "f0_condition": False,
        "checkpoint_filename": "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
        "config_filename": "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
    },
    "voice_and_accent (whisper-base, heavier, timbre + pitch-level/accent)": {
        "f0_condition": True,
        "checkpoint_filename": "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
        "config_filename": "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
    },
}
_SEED_VC_HF_REPO = "Plachta/Seed-VC"
_CAMPPLUS_HF_REPO = "funasr/campplus"
_CAMPPLUS_FILENAME = "campplus_cn_common.bin"
_RMVPE_HF_REPO = "lj1995/VoiceConversionWebUI"
_RMVPE_FILENAME = "rmvpe.pt"


def _model_folder(name):
    """Registers (if needed) and returns ComfyUI/models/<name>, creating it
    if missing. Registered via folder_paths so other node packs that also
    expect e.g. a shared 'whisper' or 'htdemucs' models folder see the same
    directory, instead of every pack keeping its own private copy."""
    path = os.path.join(folder_paths.models_dir, name)
    os.makedirs(path, exist_ok=True)
    try:
        if name not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path(name, path)
    except Exception:
        pass
    return path


import re

_RENAME_MAP = {
    "modules": "mykee_seedvc_modules",
    "hf_utils": "mykee_seedvc_hf_utils",
    "inference": "mykee_seedvc_inference",
}


_NAMESPACING_PATCH_VERSION = 2

# All three names get their file/directory renamed (needed so our own
# imports land on a collision-free name). Only "modules" and "hf_utils"
# also get every in-file reference rewritten - "inference" turned out to
# also occur as an ordinary method name inside the vendored code itself
# (CFM.inference(), called as model.cfm.inference() elsewhere), and
# nothing else in this repo imports the top-level inference.py module by
# name, so there's nothing to fix up for it beyond the filename.
_TEXT_SUBSTITUTION_NAMES = ("modules", "hf_utils")


def _force_rmtree(path):
    """
    shutil.rmtree(path, ignore_errors=True) silently does nothing useful
    on Windows against a git checkout: git marks some files inside .git
    (particularly in .git/objects) read-only, rmtree can't delete those,
    and ignore_errors=True swallows that failure instead of surfacing it -
    leaving the directory (partially) in place, so a subsequent `git
    clone` into the same path fails with "destination path already exists
    and is not an empty directory". This clears the read-only bit on
    anything rmtree stumbles on and retries the removal, which is the
    standard fix for deleting git checkouts on Windows.
    """
    def _on_rm_error(func, bad_path, exc_info):
        os.chmod(bad_path, stat.S_IWRITE)
        func(bad_path)

    shutil.rmtree(path, onerror=_on_rm_error)


def _clean_clone_seed_vc_repo():
    """
    Clones _SEED_VC_REPO_URL into _SEED_VC_REPO_DIR, first force-removing
    anything already at that path. Used both for the very first clone and
    for the self-healing re-clone in _patch_seed_vc_namespacing - in
    either case, an interrupted previous attempt can leave the directory
    existing but incomplete/corrupt (e.g. a partially-succeeded rmtree
    that deleted the renamed modules/ folder but left undeletable .git
    internals behind), and `git clone` refuses to clone into a non-empty
    directory regardless of why it's non-empty. Always clearing first,
    rather than only checking "does it look done", is what actually makes
    this self-healing.
    """
    if os.path.exists(_SEED_VC_REPO_DIR):
        _force_rmtree(_SEED_VC_REPO_DIR)
    os.makedirs(os.path.dirname(_SEED_VC_REPO_DIR), exist_ok=True)
    print(f"[Mykee Voice/Accent Match] Cloning {_SEED_VC_REPO_URL} into {_SEED_VC_REPO_DIR} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", _SEED_VC_REPO_URL, _SEED_VC_REPO_DIR],
        check=True,
    )


def _patch_seed_vc_namespacing():
    """
    Seed-VC's own code uses a couple of very generic top-level names -
    'modules' (a whole package), 'hf_utils', 'inference' - that other
    ComfyUI custom node packs are quite likely to also define. Python
    caches imported modules by name process-wide, so a name collision
    would make our import silently resolve to someone else's module (or
    vice versa) instead of failing cleanly, breaking either us or them
    depending on load order. Rather than juggling sys.modules at import
    time, this permanently renames those files/folders in our vendored
    clone to unique, mykee-prefixed ones (see _RENAME_MAP), and rewrites
    references to "modules"/"hf_utils" throughout the repo to match - a
    one-time, one-line-per-file text patch that removes the collision
    risk for good.

    Side effect: any exact-word match of "modules" or "hf_utils" gets
    rewritten wherever it appears, including inside comments/docstrings/
    print strings that aren't actually a package reference (e.g. a code
    comment mentioning "load the modules" becomes "load the
    mykee_seedvc_modules") - cosmetically odd in a couple of spots but
    functionally harmless, since we never read those comments back out as
    code. A method/attribute access preceded by a "." (self.modules(), a
    real PyTorch nn.Module method unrelated to the "modules" package) is
    excluded from this, same reasoning as above.

    Guarded by a *versioned* marker file rather than a plain exists/not
    check: if this function's logic changes later (as it did once
    already, when renaming "inference" text turned out to break a method
    of the same name), a repo already patched by an older version of this
    function gets wiped and re-cloned from scratch automatically, rather
    than silently keeping stale, incorrectly-patched files around and
    needing the user to manually delete the folder.
    """
    marker = os.path.join(_SEED_VC_REPO_DIR, ".mykee_namespacing_patched")
    if os.path.exists(marker):
        with open(marker, "r", encoding="utf-8") as f:
            if f.read().strip() == str(_NAMESPACING_PATCH_VERSION):
                return
        print("[Mykee Voice/Accent Match] Vendored Seed-VC repo was patched by an older, "
              "since-corrected version of this node's namespacing fix - re-cloning it fresh.")
        _clean_clone_seed_vc_repo()

    old_modules_dir = os.path.join(_SEED_VC_REPO_DIR, "modules")
    new_modules_dir = os.path.join(_SEED_VC_REPO_DIR, _RENAME_MAP["modules"])
    if os.path.isdir(old_modules_dir) and not os.path.isdir(new_modules_dir):
        os.rename(old_modules_dir, new_modules_dir)

    old_hf_utils = os.path.join(_SEED_VC_REPO_DIR, "hf_utils.py")
    new_hf_utils = os.path.join(_SEED_VC_REPO_DIR, _RENAME_MAP["hf_utils"] + ".py")
    if os.path.isfile(old_hf_utils) and not os.path.isfile(new_hf_utils):
        os.rename(old_hf_utils, new_hf_utils)

    old_inference = os.path.join(_SEED_VC_REPO_DIR, "inference.py")
    new_inference = os.path.join(_SEED_VC_REPO_DIR, _RENAME_MAP["inference"] + ".py")
    if os.path.isfile(old_inference) and not os.path.isfile(new_inference):
        os.rename(old_inference, new_inference)

    # Negative lookbehind for a preceding "." excludes attribute/method
    # access such as self.modules() - only bare references to the name
    # itself (import statements, or "modules.something" used as a
    # namespace after import) get rewritten.
    patterns = [
        (re.compile(r"(?<!\.)\b" + re.escape(name) + r"\b"), _RENAME_MAP[name])
        for name in _TEXT_SUBSTITUTION_NAMES
    ]
    for root, _dirs, files in os.walk(_SEED_VC_REPO_DIR):
        if ".git" in root.split(os.sep):
            continue
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    text = f.read()
            except (UnicodeDecodeError, OSError):
                continue
            new_text = text
            for pattern, replacement in patterns:
                new_text = pattern.sub(replacement, new_text)
            if new_text != text:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(new_text)

    with open(marker, "w", encoding="utf-8") as f:
        f.write(str(_NAMESPACING_PATCH_VERSION))


def _ensure_seed_vc_repo():
    """Clones the Seed-VC code (not weights - just the Python modules
    implementing the DiT/CFM pipeline) into this node pack's third_party/
    folder on first use, and puts it on sys.path. This is a full
    application repo with its own modules.* package, not something
    pip-installable, so we vendor the code once via git and download only
    the model weights into the shared ComfyUI/models/ folders. The repo is
    archived upstream (read-only) as of late 2025, but still clones and
    runs fine - it just won't receive further updates."""
    already_cloned = os.path.isdir(os.path.join(_SEED_VC_REPO_DIR, "modules")) or os.path.isdir(
        os.path.join(_SEED_VC_REPO_DIR, _RENAME_MAP["modules"])
    )
    if not already_cloned:
        # _clean_clone_seed_vc_repo() force-removes whatever's at this path
        # first (if anything) rather than assuming "not already_cloned"
        # means the path is empty or absent - a previous interrupted
        # attempt can leave debris there (e.g. undeletable .git internals
        # on Windows) that would otherwise make `git clone` refuse to run.
        _clean_clone_seed_vc_repo()
    _patch_seed_vc_namespacing()
    if _SEED_VC_REPO_DIR not in sys.path:
        sys.path.insert(0, _SEED_VC_REPO_DIR)


import contextlib


@contextlib.contextmanager
def _forced_hf_offline(offline):
    """
    Directly overwrites huggingface_hub.constants.HF_HUB_OFFLINE (the
    Python attribute, not just the environment variable) for the duration
    of the block, restoring it afterward.

    Setting os.environ["HF_HUB_OFFLINE"] alone isn't enough: huggingface_hub
    reads that env var exactly once, the first time huggingface_hub.constants
    is imported anywhere in the process, and caches the boolean result in
    that module's HF_HUB_OFFLINE attribute - by the time this node runs,
    something else in ComfyUI's startup has very likely already imported it
    with the env var unset, so later env var changes have no effect. Any
    huggingface_hub call that doesn't get an explicit local_files_only
    argument (which is most calls we don't make ourselves - Demucs's own
    internals, or a HubMixin's preliminary config-resolution step that
    happens before it delegates to a subclass's _from_pretrained) falls
    back to checking this cached attribute, so it has to be corrected
    directly rather than through the environment.
    """
    try:
        import huggingface_hub.constants as _hf_constants
    except Exception:
        yield
        return
    prev = getattr(_hf_constants, "HF_HUB_OFFLINE", False)
    _hf_constants.HF_HUB_OFFLINE = bool(offline)
    try:
        yield
    finally:
        _hf_constants.HF_HUB_OFFLINE = prev


def _download_hf_file(repo_id, filename, cache_dir, local_files_only=False):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=repo_id, filename=filename, cache_dir=cache_dir,
        local_files_only=local_files_only,
    )


def _make_hf_router(campplus_dir, rmvpe_dir, local_files_only=False):
    """
    Builds a drop-in replacement for Seed-VC's own hf_utils.load_custom_model_from_hf,
    routing each known repo to the ComfyUI/models/ subfolder the user asked
    for instead of Seed-VC's own default (a local ./checkpoints/hf_cache
    folder inside whatever the current working directory happens to be).
    Matches the original function's calling convention as used throughout
    inference.py: returns (model_path, config_path) when a config filename
    is given, otherwise just the model path.
    """
    def _router(repo_id, model_filename, config_filename=None):
        if repo_id == _CAMPPLUS_HF_REPO:
            target_dir = campplus_dir
        elif repo_id == _RMVPE_HF_REPO:
            target_dir = rmvpe_dir
        else:
            # DiT checkpoints (Plachta/Seed-VC) and anything else fall back
            # to whatever HF_HUB_CACHE is set to for the duration of the
            # load - see _load_seed_vc_pipeline, which points that at the
            # Seed-VC models folder.
            target_dir = None
        model_path = _download_hf_file(repo_id, model_filename, target_dir, local_files_only)
        if config_filename is not None:
            config_path = _download_hf_file(repo_id, config_filename, target_dir, local_files_only)
            return model_path, config_path
        return model_path
    return _router


_SEED_VC_PIPELINE_CACHE = {}


def _load_seed_vc_pipeline(variant_name, device_str, fp16):
    """
    Loads (and caches) the Seed-VC (model, semantic_fn, f0_fn, vocoder_fn,
    campplus_model, mel_fn, mel_fn_args) tuple for the requested variant,
    downloading weights into the shared ComfyUI/models/ folders on first
    use. Cached per (variant, device, fp16) so repeated node runs in the
    same ComfyUI session don't reload from disk every time. Returns
    (pipeline, patchers) - patchers is [model_patcher, campplus_patcher],
    ComfyUI-managed handles for the two plain nn.Module pieces (see
    _wrap_and_load_together).

    Tries loading with HF_HUB_OFFLINE=1 first - if every file is already
    cached (the common case after the first run), this skips the
    "is there a newer version on the Hub?" HEAD request huggingface_hub
    and transformers otherwise make for every file on every load, which is
    all network round-trips for no benefit once nothing is actually going
    to change. Only falls back to a normal (network-allowed) load if the
    offline attempt fails - the first run ever, or if a file has gone
    missing from the cache since.
    """
    cache_key = (variant_name, device_str, fp16)
    cached = _SEED_VC_PIPELINE_CACHE.get(cache_key)
    if cached is not None:
        pipeline, patchers, _seed_vc_module = cached
        # Re-assert BOTH patchers together in one call, same reasoning as
        # the initial load: a cache hit only re-touching one of them would
        # leave the other "unguarded" against ComfyUI evicting it
        # separately in the meantime.
        mm.load_models_gpu(patchers, force_full_load=True)
        return cached

    try:
        result = _load_seed_vc_pipeline_impl(variant_name, device_str, fp16, offline=True)
    except Exception as exc:
        print(f"[Mykee Voice/Accent Match] Offline model load failed ({exc}); "
              f"retrying with network access allowed (first run, or a cached file is missing).")
        result = _load_seed_vc_pipeline_impl(variant_name, device_str, fp16, offline=False)

    _SEED_VC_PIPELINE_CACHE[cache_key] = result
    return result


def _load_seed_vc_pipeline_impl(variant_name, device_str, fp16, offline):
    variant = _MODEL_VARIANTS[variant_name]
    _ensure_seed_vc_repo()

    seed_vc_dir = _model_folder("Seed-VC")
    campplus_dir = _model_folder("CAMpp")
    whisper_dir = _model_folder("whisper")

    with _forced_hf_offline(offline):
        dit_checkpoint_path = _download_hf_file(
            _SEED_VC_HF_REPO, variant["checkpoint_filename"], seed_vc_dir, local_files_only=offline
        )
        dit_config_path = _download_hf_file(
            _SEED_VC_HF_REPO, variant["config_filename"], seed_vc_dir, local_files_only=offline
        )

        import mykee_seedvc_hf_utils as hf_utils  # from the vendored, namespaced seed-vc repo
        hf_utils.load_custom_model_from_hf = _make_hf_router(campplus_dir, rmvpe_dir=seed_vc_dir, local_files_only=offline)

        import transformers
        _orig_whisper_from_pretrained = transformers.WhisperModel.from_pretrained.__func__
        _orig_feat_extractor_from_pretrained = transformers.AutoFeatureExtractor.from_pretrained.__func__

        def _whisper_from_pretrained(cls, *args, **kwargs):
            kwargs.setdefault("cache_dir", whisper_dir)
            kwargs.setdefault("local_files_only", offline)
            return _orig_whisper_from_pretrained(cls, *args, **kwargs)

        def _feat_extractor_from_pretrained(cls, *args, **kwargs):
            kwargs.setdefault("cache_dir", whisper_dir)
            kwargs.setdefault("local_files_only", offline)
            return _orig_feat_extractor_from_pretrained(cls, *args, **kwargs)

        transformers.WhisperModel.from_pretrained = classmethod(_whisper_from_pretrained)
        transformers.AutoFeatureExtractor.from_pretrained = classmethod(_feat_extractor_from_pretrained)

        # BigVGAN's vendored _from_pretrained() declares 'proxies' and
        # 'resume_download' as required keyword-only arguments with no default
        # (matching the huggingface_hub API version it was written against).
        # Recent huggingface_hub releases dropped 'resume_download' as a
        # concept and no longer pass either of these down to _from_pretrained,
        # so the call fails with "missing required keyword-only arguments"
        # rather than the vendored code ever getting a chance to use them.
        # Giving them defaults here, before load_models() triggers the actual
        # import, papers over that API drift without needing to hand-edit the
        # vendored file (whose exact formatting isn't worth depending on).
        # local_files_only is forced the same way, for the same reason as
        # everywhere else here: relying on the HF_HUB_OFFLINE env var alone
        # doesn't reliably work once huggingface_hub/transformers may
        # already be imported elsewhere in the ComfyUI process with that
        # constant already resolved, so every call gets the flag directly.
        try:
            from mykee_seedvc_modules.bigvgan import bigvgan as _bigvgan_module
            _orig_bigvgan_from_pretrained = _bigvgan_module.BigVGAN._from_pretrained.__func__

            def _bigvgan_from_pretrained(cls, *args, **kwargs):
                kwargs.setdefault("proxies", None)
                kwargs.setdefault("resume_download", None)
                kwargs.setdefault("local_files_only", offline)
                return _orig_bigvgan_from_pretrained(cls, *args, **kwargs)

            _bigvgan_module.BigVGAN._from_pretrained = classmethod(_bigvgan_from_pretrained)

            # The public from_pretrained() (inherited from huggingface_hub's
            # ModelHubMixin) does its own preliminary config-resolution step
            # before ever delegating to _from_pretrained above - that step
            # happens too early for the _from_pretrained patch to reach, so
            # it needs its own local_files_only default too.
            _orig_bigvgan_public_from_pretrained = _bigvgan_module.BigVGAN.from_pretrained.__func__

            def _bigvgan_public_from_pretrained(cls, *args, **kwargs):
                kwargs.setdefault("local_files_only", offline)
                return _orig_bigvgan_public_from_pretrained(cls, *args, **kwargs)

            _bigvgan_module.BigVGAN.from_pretrained = classmethod(_bigvgan_public_from_pretrained)
        except Exception as exc:
            print(f"[Mykee Voice/Accent Match] Could not pre-patch BigVGAN.from_pretrained "
                  f"(huggingface_hub compatibility shim) - proceeding anyway: {exc}")

        prev_hf_cache = os.environ.get("HF_HUB_CACHE")
        os.environ["HF_HUB_CACHE"] = os.path.join(seed_vc_dir, "hf_cache")

        try:
            import mykee_seedvc_inference as seed_vc_inference  # the vendored, namespaced repo's inference.py

            class _Args:
                pass

            args = _Args()
            args.f0_condition = variant["f0_condition"]
            args.checkpoint = dit_checkpoint_path
            args.config = dit_config_path
            args.fp16 = fp16

            seed_vc_inference.device = torch.device(device_str)
            pipeline = seed_vc_inference.load_models(args)
        finally:
            if prev_hf_cache is None:
                os.environ.pop("HF_HUB_CACHE", None)
            else:
                os.environ["HF_HUB_CACHE"] = prev_hf_cache
            transformers.WhisperModel.from_pretrained = classmethod(_orig_whisper_from_pretrained)
            transformers.AutoFeatureExtractor.from_pretrained = classmethod(_orig_feat_extractor_from_pretrained)

    # Hand the DiT/CFM model container and the CAM++ speaker encoder to
    # ComfyUI's memory manager, together (see _wrap_and_load_together's
    # docstring for why "together" matters). `model` here is actually a
    # Munch namespace bundling several real nn.Module sub-components, not
    # an nn.Module itself - _wrap_seed_vc_model_container() handles that
    # (see its own docstring). semantic_fn/f0_fn/vocoder_fn/mel_fn are
    # left untouched - see the module-level comment near the top of this
    # file.
    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = pipeline
    model, campplus_model, patchers = _wrap_seed_vc_model_container(model, campplus_model, torch.device(device_str))
    pipeline = (model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args)

    return pipeline, patchers, seed_vc_inference


def _crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = chunk2[:overlap] * fade_in[:len(chunk2)] + (chunk1[-overlap:] * fade_out)[:len(chunk2)]
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


@torch.no_grad()
def _run_seed_vc(pipeline, seed_vc_module, source_np, ref_np, source_sr, ref_sr, f0_condition,
                  diffusion_steps, length_adjust, inference_cfg_rate,
                  timbre_mix, accent_mix, semi_tone_shift, device_str, fp16):
    """
    In-memory port of seed-vc's inference.py main() conversion loop (same
    chunking/crossfade/F0-shift logic) - reading/writing tensors directly
    instead of files on disk. source_np is at source_sr, ref_np is at
    ref_sr - these are resampled independently, since generated_audio and
    reference_audio are two unrelated inputs and very often won't share a
    sample rate. Returns (converted_np, out_sr).

    Unlike the original inference.py, timbre and pitch-level/accent
    matching are both continuous 0-1 mixes rather than hard on/off:
    - timbre_mix: the CAM++ speaker embedding used for voice conditioning
      is a linear blend between the source's own embedding (0.0) and the
      reference's (1.0), rather than always being the reference's. This
      is what actually lets accent_mix be used with timbre_mix at 0 - the
      model still needs *some* style vector, so at 0 it uses the source's
      own, leaving timbre essentially unchanged while still letting pitch
      be pulled toward the reference.
    - accent_mix: scales how much of the reference-vs-source median pitch
      difference gets applied, from none (0.0, source's own pitch level)
      to all of it (1.0, matching the reference's level exactly).
    """
    import torchaudio
    import librosa

    device = torch.device(device_str)
    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = pipeline

    sr = 22050 if not f0_condition else 44100
    hop_length = 256 if not f0_condition else 512
    max_context_window = sr // hop_length * 30
    overlap_frame_len = 16
    overlap_wave_len = overlap_frame_len * hop_length

    source_22 = librosa.resample(source_np, orig_sr=source_sr, target_sr=sr) if source_sr != sr else source_np
    ref_22 = librosa.resample(ref_np, orig_sr=ref_sr, target_sr=sr) if ref_sr != sr else ref_np

    source_audio = torch.tensor(source_22).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_22[: sr * 25]).unsqueeze(0).float().to(device)

    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
    if converted_waves_16k.size(-1) <= 16000 * 30:
        S_alt = semantic_fn(converted_waves_16k)
    else:
        overlapping_time = 5
        S_alt_list = []
        buffer = None
        traversed_time = 0
        while traversed_time < converted_waves_16k.size(-1):
            _check_interrupted()
            if buffer is None:
                chunk = converted_waves_16k[:, traversed_time: traversed_time + 16000 * 30]
            else:
                chunk = torch.cat(
                    [buffer, converted_waves_16k[:, traversed_time: traversed_time + 16000 * (30 - overlapping_time)]],
                    dim=-1,
                )
            S_alt_chunk = semantic_fn(chunk)
            if traversed_time == 0:
                S_alt_list.append(S_alt_chunk)
            else:
                S_alt_list.append(S_alt_chunk[:, 50 * overlapping_time:])
            buffer = chunk[:, -16000 * overlapping_time:]
            traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
        S_alt = torch.cat(S_alt_list, dim=1)

    ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    S_ori = semantic_fn(ori_waves_16k)

    mel = mel_fn(source_audio.to(device).float())
    mel2 = mel_fn(ref_audio.to(device).float())
    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(
        ori_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000
    )
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    if timbre_mix >= 1.0:
        style_blended = style2
    else:
        feat1 = torchaudio.compliance.kaldi.fbank(
            converted_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat1 = feat1 - feat1.mean(dim=0, keepdim=True)
        style1 = campplus_model(feat1.unsqueeze(0))
        style_blended = style1 * (1.0 - timbre_mix) + style2 * timbre_mix

    if f0_condition:
        F0_ori = f0_fn(ori_waves_16k[0], thred=0.03)
        F0_alt = f0_fn(converted_waves_16k[0], thred=0.03)
        F0_ori = torch.from_numpy(F0_ori).to(device)[None]
        F0_alt = torch.from_numpy(F0_alt).to(device)[None]

        voiced_F0_ori = F0_ori[F0_ori > 1]
        log_f0_alt = torch.log(F0_alt + 1e-5)
        voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
        voiced_log_f0_alt = torch.log(F0_alt[F0_alt > 1] + 1e-5)
        median_log_f0_ori = torch.median(voiced_log_f0_ori)
        median_log_f0_alt = torch.median(voiced_log_f0_alt)

        shifted_log_f0_alt = log_f0_alt.clone()
        if accent_mix > 0.0:
            shifted_log_f0_alt[F0_alt > 1] = (
                log_f0_alt[F0_alt > 1] + accent_mix * (median_log_f0_ori - median_log_f0_alt)
            )
        shifted_f0_alt = torch.exp(shifted_log_f0_alt)
        if semi_tone_shift != 0:
            factor = 2 ** (semi_tone_shift / 12)
            shifted_f0_alt[F0_alt > 1] = shifted_f0_alt[F0_alt > 1] * factor
    else:
        F0_ori = None
        shifted_f0_alt = None

    cond, _, _, _, _ = model.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt)
    prompt_condition, _, _, _, _ = model.length_regulator(S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori)

    max_source_window = max_context_window - mel2.size(2)
    processed_frames = 0
    generated_wave_chunks = []
    previous_chunk = None

    while processed_frames < cond.size(1):
        _check_interrupted()
        chunk_cond = cond[:, processed_frames: processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
            vc_target = model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                mel2, style_blended, None, diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
            )
            vc_target = vc_target[:, :, mel2.size(-1):]
            vc_wave = vocoder_fn(vc_target.float()).squeeze()
            vc_wave = vc_wave[None, :]

        if processed_frames == 0:
            if is_last_chunk:
                generated_wave_chunks.append(vc_wave[0].cpu().numpy())
                break
            generated_wave_chunks.append(vc_wave[0, :-overlap_wave_len].cpu().numpy())
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
        elif is_last_chunk:
            generated_wave_chunks.append(_crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len))
            processed_frames += vc_target.size(2) - overlap_frame_len
            break
        else:
            generated_wave_chunks.append(
                _crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len)
            )
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len

    converted = np.concatenate(generated_wave_chunks)
    return converted, sr


_DEMUCS_MODEL_CACHE = {}


def _get_demucs_model(model_name, device_str):
    """
    Loads (and caches) a Demucs separation model, wrapped in a ComfyUI
    ModelPatcher and registered with ComfyUI's memory manager (see the
    module-level comment near the top of this file). Tries with
    local-only loading first, same reasoning as the Seed-VC pipeline
    loader: Demucs fetches its pretrained weights via huggingface_hub too,
    and the HF_HUB_OFFLINE env var isn't reliable once huggingface_hub is
    already imported elsewhere in the ComfyUI process. Since we don't
    control Demucs's own source the way the vendored Seed-VC copy can be
    patched, this reaches into demucs.pretrained's own namespace and
    patches whatever name it bound hf_hub_download to there specifically -
    patching huggingface_hub.hf_hub_download itself doesn't help if Demucs
    already did `from huggingface_hub import hf_hub_download` at its own
    import time, since that binds a local name in its module that a later
    patch to the source module doesn't reach.

    Returns the ModelPatcher (not the raw model) - callers use
    `patcher.model`.
    """
    cache_key = (model_name, device_str)
    cached_patcher = _DEMUCS_MODEL_CACHE.get(cache_key)
    if cached_patcher is not None:
        mm.load_models_gpu([cached_patcher], force_full_load=True)
        return cached_patcher

    from demucs.pretrained import get_model
    import demucs.pretrained as _demucs_pretrained_module

    htdemucs_dir = _model_folder("htdemucs")
    prev_torch_home = os.environ.get("TORCH_HOME")
    os.environ["TORCH_HOME"] = htdemucs_dir

    def _attempt(offline):
        orig_fn = getattr(_demucs_pretrained_module, "hf_hub_download", None)
        if offline and orig_fn is not None:
            def _offline_hf_hub_download(*args, **kwargs):
                kwargs.setdefault("local_files_only", True)
                return orig_fn(*args, **kwargs)
            _demucs_pretrained_module.hf_hub_download = _offline_hf_hub_download
        try:
            with _forced_hf_offline(offline):
                return get_model(model_name)
        finally:
            if offline and orig_fn is not None:
                _demucs_pretrained_module.hf_hub_download = orig_fn

    try:
        try:
            model = _attempt(offline=True)
        except Exception as exc:
            print(f"[Mykee Voice/Accent Match] Offline Demucs model load failed ({exc}); "
                  f"retrying with network access allowed.")
            model = _attempt(offline=False)
    finally:
        if prev_torch_home is None:
            os.environ.pop("TORCH_HOME", None)
        else:
            os.environ["TORCH_HOME"] = prev_torch_home

    model.eval()
    patcher, = _wrap_and_load_together([model], torch.device(device_str))
    _DEMUCS_MODEL_CACHE[cache_key] = patcher
    return patcher


def _separate_vocals(waveform, sr, model_name, device_str):
    """
    Splits `waveform` (torch tensor, shape (channels, samples)) into
    (vocals, background) using Demucs, with weights stored in the shared
    ComfyUI/models/htdemucs folder. Returns two torch tensors at the same
    sample rate/channel count as the input.
    """
    from demucs.apply import apply_model

    _check_interrupted()
    patcher = _get_demucs_model(model_name, device_str)
    model = patcher.model  # ComfyUI's load_models_gpu() already put this on device_str

    wav = waveform
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)  # demucs expects stereo
    elif wav.shape[0] > 2:
        wav = wav[:2]

    model_sr = model.samplerate
    if sr != model_sr:
        wav = _resample_torch(wav.unsqueeze(0), sr, model_sr).squeeze(0)

    ref = wav.mean(0)
    wav_norm = (wav - ref.mean()) / (ref.std() + 1e-8)

    with torch.no_grad():
        sources = apply_model(model, wav_norm.unsqueeze(0).to(device_str), device=device_str, progress=False)[0]
    sources = sources * (ref.std() + 1e-8) + ref.mean()

    source_names = model.sources
    vocals_idx = source_names.index("vocals") if "vocals" in source_names else 0
    vocals = sources[vocals_idx].cpu()
    background = (sources.sum(dim=0) - sources[vocals_idx]).cpu()

    if model_sr != sr:
        vocals = _resample_torch(vocals.unsqueeze(0), model_sr, sr).squeeze(0)
        background = _resample_torch(background.unsqueeze(0), model_sr, sr).squeeze(0)

    return vocals, background



def _make_progress_bar():
    """Returns a comfy.utils.ProgressBar (0-100) if available, else None.
    This drives ComfyUI's native green progress bar overlay on the node -
    the same one KSampler uses - on top of (not instead of) this node's
    own text status widget."""
    if _ComfyProgressBar is None:
        return None
    try:
        return _ComfyProgressBar(100)
    except Exception:
        return None


def _set_progress(pbar, value):
    if pbar is None:
        return
    try:
        pbar.update_absolute(value, 100)
    except Exception:
        pass


def _send_status(unique_id, text):
    """Sends a short status string to this node instance's UI widget (see
    web/mykee_voice_match.js), in addition to whatever gets printed to the
    console. Silently does nothing if there's no PromptServer running (e.g.
    a bare Python test) or no unique_id was supplied."""
    if unique_id is None:
        return
    try:
        from server import PromptServer
        PromptServer.instance.send_sync(
            "mykee-voice-match-status", {"node": str(unique_id), "text": text}
        )
    except Exception:
        pass


def _unload_models():
    """Releases every cached Seed-VC pipeline and Demucs model this node
    has loaded. The DiT/CAM++ and Demucs models - registered with
    ComfyUI's own memory manager - are released the same way any built-in
    node's model would be, via ModelPatcher.unpatch_model(), an immediate
    and guaranteed release rather than a guess. The still-unmanaged pieces
    (semantic_fn/f0_fn/vocoder_fn/mel_fn closures - see the module-level
    comment near the top of this file) are dropped from the cache and left
    to gc.collect()/torch.cuda.empty_cache(), same as before."""
    for _pipeline, patchers, _seed_vc_module in _SEED_VC_PIPELINE_CACHE.values():
        for patcher in patchers:
            _unload_patcher(patcher)
    _SEED_VC_PIPELINE_CACHE.clear()

    for patcher in _DEMUCS_MODEL_CACHE.values():
        _unload_patcher(patcher)
    _DEMUCS_MODEL_CACHE.clear()

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class MykeeVoiceAccentMatch:
    """
    Universal, model-independent voice/accent matcher: reshapes any
    generated audio's voice to match a reference sample's timbre and/or
    pitch-level/accent character, regardless of what produced the
    generated audio in the first place.

    Pipeline: Demucs source separation on both inputs -> Seed-VC voice
    conversion on the isolated vocals -> remix with the generated audio's
    own separated background.

    timbre_mix and accent_mix are continuous 0-1 dials, not on/off
    switches: Seed-VC always needs *some* CAM++ speaker embedding to
    condition on, so rather than only ever using the reference's (with no
    way to dial that back), this linearly blends the source's own
    embedding and the reference's by timbre_mix. That's also what makes
    accent_mix > 0 meaningful even with timbre_mix at 0 - the pitch-level
    shift toward the reference still applies, while the style vector
    stays the source's own, keeping timbre close to unchanged.

    Needs a CUDA GPU. Everything (Whisper, the Seed-VC DiT/CFM model,
    CAM++, Demucs) is downloaded on first use into shared ComfyUI/models/
    subfolders (whisper/, Seed-VC/, CAMpp/, htdemucs/), not private to this
    node pack, so other tools that expect models in those conventional
    locations can reuse the same files.

    The Seed-VC and Demucs models are managed by ComfyUI's own memory
    manager (comfy.model_management), not a hand-rolled cache - so pressing
    ComfyUI's standard Cancel/Interrupt button actually stops a run in
    progress (checked once per conversion chunk, and between pipeline
    stages), and VRAM is (de)allocated the same way any built-in node's is.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_audio": ("AUDIO", {
                    "tooltip": "The audio to reshape - from any source (any TTS model, any generator). May contain background music/SFX; it will be separated out and remixed back afterward.",
                }),
                "reference_audio": ("AUDIO", {
                    "tooltip": "The 'DNA' sample - a speaker's voice to match. Does not need to say the same words as generated_audio. A few seconds to ~25 seconds is used; more doesn't help much beyond that.",
                }),
                "timbre_mix": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much of the reference speaker's voice/timbre to blend in - a continuous mix between the generated audio's own voice (0.0) and the reference's (1.0), via linearly blending their CAM++ speaker embeddings.",
                }),
                "accent_mix": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much of the reference-vs-source pitch-level difference to apply - 0.0 keeps the generated audio's own pitch level, 1.0 matches the reference's exactly. Above 0 uses the heavier whisper-base+F0 model. Works even with timbre_mix at 0 - the source's own voice is used as the style vector in that case, so timbre stays close to unchanged while pitch still shifts.",
                }),
                "model_variant": (list(_MODEL_VARIANTS.keys()), {
                    "default": list(_MODEL_VARIANTS.keys())[0],
                    "tooltip": "Which Seed-VC checkpoint pair to use - each bundles a specific Whisper size it was trained with; these can't be mixed and matched freely. Ignored (the F0 variant is used automatically) whenever accent_mix is above 0.",
                }),
                "demucs_model": (["htdemucs_ft", "htdemucs", "htdemucs_6s"], {
                    "default": "htdemucs_ft",
                    "tooltip": "Source-separation model used to split voice from background on both inputs. htdemucs_ft is the slower but higher-quality default.",
                }),
                "background_gain_db": ("FLOAT", {
                    "default": -1.0, "min": -60.0, "max": 12.0, "step": 0.5,
                    "tooltip": "Level of generated_audio's own background (music/SFX) relative to the converted voice when remixed back in. 0 = unchanged from the original mix level.",
                }),
                "duck_background": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Automatically lower the background further while the voice is active (sidechain ducking), same mechanism as Mykee Audio Merger.",
                }),
                "duck_amount_db": ("FLOAT", {
                    "default": 6.0, "min": 0.0, "max": 40.0, "step": 0.5,
                    "tooltip": "Extra attenuation applied to the background while the voice is active, on top of background_gain_db. Only used if duck_background is on.",
                }),
                "diffusion_steps": ("INT", {
                    "default": 30, "min": 4, "max": 100,
                    "tooltip": "Seed-VC quality/speed tradeoff - more steps is slower but can be cleaner.",
                }),
                "inference_cfg_rate": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Seed-VC parameter controlling how strongly the output sticks to the generated audio's original linguistic content.",
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "auto follows ComfyUI's current torch device. cpu will be extremely slow for this pipeline.",
                }),
                "unload_model": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Free the Seed-VC and Demucs models from VRAM once this run finishes, via ComfyUI's own model manager. Leave off to keep them loaded for faster repeat runs; turn on if you need the VRAM back for something else afterward.",
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("final_audio", "separated_background", "separated_voice")
    FUNCTION = "process"
    CATEGORY = "Mykee/Audio"

    def process(
        self,
        generated_audio,
        reference_audio,
        timbre_mix,
        accent_mix,
        model_variant,
        demucs_model,
        background_gain_db,
        duck_background,
        duck_amount_db,
        diffusion_steps,
        inference_cfg_rate,
        device,
        unload_model,
        unique_id=None,
    ):
        if device == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_str = device
        if device_str == "cpu":
            print("[Mykee Voice/Accent Match] WARNING: running on CPU - this pipeline "
                  "(Demucs + Whisper + a diffusion transformer) will be very slow.")

        try:
            pbar = _make_progress_bar()
            _set_progress(pbar, 0)
            _check_interrupted()
            _send_status(unique_id, "Separating vocals...")

            gen_wave = generated_audio["waveform"][0]  # (channels, samples)
            gen_sr = generated_audio["sample_rate"]
            ref_wave = reference_audio["waveform"][0]
            ref_sr = reference_audio["sample_rate"]

            gen_voice, gen_background = _separate_vocals(gen_wave, gen_sr, demucs_model, device_str)
            ref_voice, _ref_background = _separate_vocals(ref_wave, ref_sr, demucs_model, device_str)
            _set_progress(pbar, 20)

            gen_voice_mono = gen_voice.mean(0).cpu().numpy().astype(np.float32)
            ref_voice_mono = ref_voice.mean(0).cpu().numpy().astype(np.float32)

            if timbre_mix > 0.0 or accent_mix > 0.0:
                variant = _MODEL_VARIANTS[model_variant]
                f0_condition = accent_mix > 0.0
                # The chosen model_variant only matters for its Whisper
                # size when accent_mix is 0 - whenever accent_mix is above
                # 0 we need the F0-capable variant regardless of what was
                # selected, since that's the only one with pitch-extraction
                # wired up at all.
                effective_variant_name = model_variant
                if not f0_condition and variant["f0_condition"]:
                    effective_variant_name = [
                        name for name, v in _MODEL_VARIANTS.items() if not v["f0_condition"]
                    ][0]
                elif f0_condition and not variant["f0_condition"]:
                    effective_variant_name = [
                        name for name, v in _MODEL_VARIANTS.items() if v["f0_condition"]
                    ][0]

                _send_status(unique_id, "Loading Seed-VC model...")
                pipeline, seed_vc_patchers, seed_vc_module = _load_seed_vc_pipeline(effective_variant_name, device_str, fp16=(device_str == "cuda"))
                _set_progress(pbar, 40)

                _check_interrupted()
                _send_status(unique_id, "Converting voice...")
                converted_np, converted_sr = _run_seed_vc(
                    pipeline, seed_vc_module,
                    source_np=gen_voice_mono, ref_np=ref_voice_mono,
                    source_sr=gen_sr, ref_sr=ref_sr,
                    f0_condition=f0_condition,
                    diffusion_steps=diffusion_steps,
                    length_adjust=1.0,
                    inference_cfg_rate=inference_cfg_rate,
                    timbre_mix=timbre_mix,
                    accent_mix=accent_mix,
                    semi_tone_shift=0,
                    device_str=device_str,
                    fp16=(device_str == "cuda"),
                )
                voice_out = torch.from_numpy(converted_np).float().unsqueeze(0)  # mono, (1, samples)
                voice_out_sr = converted_sr
                _set_progress(pbar, 80)
            else:
                voice_out = gen_voice_mono[np.newaxis, :]
                voice_out = torch.from_numpy(voice_out).float()
                voice_out_sr = gen_sr
                _set_progress(pbar, 80)

            _check_interrupted()
            _send_status(unique_id, "Mixing...")

            # Resample the (possibly Seed-VC-native-rate) converted voice back
            # to the generated audio's original sample rate for a clean remix
            # and for compatibility with the rest of the graph downstream.
            if voice_out_sr != gen_sr:
                voice_out = _resample_torch(voice_out.unsqueeze(0), voice_out_sr, gen_sr).squeeze(0)

            voice_out = _match_channels(voice_out.unsqueeze(0), gen_wave.shape[0]).squeeze(0)
            target_len = gen_wave.shape[-1]
            voice_out = _fit_length(voice_out.unsqueeze(0), target_len, loop=False).squeeze(0)
            gen_background_fit = _fit_length(gen_background.unsqueeze(0), target_len, loop=False).squeeze(0)

            bg_gain_lin = float(_db_to_lin(background_gain_db))
            bg_gain_curve = np.full(target_len, bg_gain_lin, dtype=np.float64)

            if duck_background:
                voice_mono_for_env = voice_out.mean(0).cpu().numpy()
                hop = max(1, int(round(gen_sr / 100.0)))
                env = _block_rms(np.abs(voice_mono_for_env), hop)
                attack_coef = float(np.exp(-1.0 / max(100.0 * (30.0 / 1000.0), 1e-6)))
                release_coef = float(np.exp(-1.0 / max(100.0 * (150.0 / 1000.0), 1e-6)))
                env_follow = _envelope_follow(env, attack_coef, release_coef)
                peak = env_follow.max() if env_follow.size else 0.0
                norm = env_follow / peak if peak > 1e-9 else env_follow * 0.0
                duck_db_curve = -duck_amount_db * norm
                duck_lin_curve = _db_to_lin(duck_db_curve)
                duck_lin_full = _upsample_curve(duck_lin_curve, target_len)
                bg_gain_curve = bg_gain_curve * duck_lin_full

            gen_background_np = gen_background_fit.cpu().numpy().astype(np.float64)
            voice_out_np = voice_out.cpu().numpy().astype(np.float64)
            final_np = voice_out_np + gen_background_np * bg_gain_curve[np.newaxis, :]
            peak = np.max(np.abs(final_np)) if final_np.size else 0.0
            if peak > 0.98:
                final_np = final_np * (0.98 / peak)

            final_tensor = torch.from_numpy(final_np.astype(np.float32)).unsqueeze(0)
            voice_out_tensor = voice_out.unsqueeze(0).float()
            background_tensor = gen_background_fit.unsqueeze(0).float()

            _set_progress(pbar, 100)
            _send_status(unique_id, "Done")

            return (
                {"waveform": final_tensor, "sample_rate": gen_sr},
                {"waveform": background_tensor, "sample_rate": gen_sr},
                {"waveform": voice_out_tensor, "sample_rate": gen_sr},
            )
        except Exception:
            _send_status(unique_id, "Error")
            raise
        finally:
            if unload_model:
                _send_status(unique_id, "Unloading models...")
                _unload_models()
                _send_status(unique_id, "Idle (models unloaded)")



NODE_CLASS_MAPPINGS = {
    "MykeeVoiceAccentMatch": MykeeVoiceAccentMatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeVoiceAccentMatch": "Mykee Voice/Accent Match",
}
