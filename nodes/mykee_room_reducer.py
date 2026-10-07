"""
ComfyUI-Mykee-Nodes / Room Reducer

Reduces the "distant mic / roomy" quality of a speech recording - NOT
the same problem as Mykee Audio Dereverb (mykee_dereverb.py) solves.

WPE (that node) is deliberately built to leave a guard interval right
after the direct sound untouched, so it only cancels the LATE reverb
tail - it does not, and is not meant to, touch early reflections or
the room's own spectral coloration (the "small/boxy/far" tonal
signature a room stamps onto a distant or off-axis recording even
with zero audible echo). That coloration is exactly what makes a
clip sound "not into the mic" without there being any audible slap
or ring to point at.

This node targets that instead, via Resemble Enhance
(https://github.com/resemble-ai/resemble-enhance, `pip install
resemble-enhance`) - a neural speech-enhancement model trained
specifically to map noisy/distant/reverberant recordings onto
studio-quality, close-mic-sounding output (denoiser + a conditional-
flow-matching "enhancer" stage that also extends bandwidth). It is
complementary to WPE, not a replacement - chain them in either order;
WPE first tends to give the enhancer a cleaner signal to work with if
there IS an audible late tail as well.

Only supports mono at a fixed 44100 Hz internally (like VoiceFixer in
mykee_voice_cleanup.py) - stereo input is downmixed, and the output is
resampled back to the ORIGINAL input's sample rate so this node's
output rate matches its input rate.

On checkpoint location: unlike this pack's other AI nodes (VoiceFixer,
Seed-VC), this one's ~1GB checkpoint is left at Resemble Enhance's
OWN default download location (inside the installed `resemble_enhance`
package folder, under `model_repo/enhancer_stage2/`) rather than
redirected into ComfyUI/models/. The library resolves a second,
nested download (the denoiser submodule's weights) via a path baked
into the downloaded hparams.yaml itself - reimplementing that
redirection to a custom folder would mean also reverse-engineering
that inner path, with a real risk of silently loading an empty/
untrained denoiser if it's gotten wrong. Left at the library's own
default, this is guaranteed to match how the authors' own demo app
uses it.
"""

import contextlib
import os
import pathlib

import numpy as np
import torch

try:
    from comfy.utils import ProgressBar as _ComfyProgressBar
except Exception:
    _ComfyProgressBar = None

try:
    from server import PromptServer
except ImportError:
    PromptServer = None

import folder_paths
import comfy.model_management as mm
from comfy.model_patcher import ModelPatcher

from .mykee_audio import _resample as _resample_torch


def _check_interrupted():
    check = getattr(mm, "throw_exception_if_processing_interrupted", None)
    if check is not None:
        check()


def _send_status(unique_id, text):
    if PromptServer is None or getattr(PromptServer, "instance", None) is None:
        return
    try:
        PromptServer.instance.send_sync("mykee.voice_cleanup_status", {"node": unique_id, "text": text})
    except Exception:
        pass


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


def _make_progress_bar():
    return _ComfyProgressBar(100) if _ComfyProgressBar is not None else None


def _set_progress(pbar, value):
    if pbar is not None:
        pbar.update_absolute(value, 100)


# --- ComfyUI-managed model lifecycle (same pattern as mykee_voice_cleanup.py -
# see that file's module-level comments for the full rationale) ---------

_patched_device_classes = {}


def _make_device_property_settable(model):
    cls = type(model)
    if cls in _patched_device_classes:
        model.__class__ = _patched_device_classes[cls]
        return model
    device_attr = getattr(cls, "device", None)
    if not isinstance(device_attr, property) or device_attr.fset is not None:
        return model
    patched_cls = type(
        cls.__name__ + "_ComfyDeviceSettable",
        (cls,),
        {"device": property(device_attr.fget, lambda self, value: None)},
    )
    _patched_device_classes[cls] = patched_cls
    model.__class__ = patched_cls
    return model


def _unload_patcher(patcher):
    if patcher is None:
        return
    try:
        patcher.unpatch_model(device_to=patcher.offload_device, unpatch_weights=True)
    except Exception as exc:
        print(f"[Mykee Room Reducer] Note: ComfyUI-managed unload failed ({exc}); "
              "falling back to a manual .to('cpu').")
        try:
            patcher.model.to("cpu")
        except Exception:
            pass


def _model_folder(name):
    """Registers (if needed) and returns ComfyUI/models/<name>, creating it
    if missing. Not actually used for the checkpoint itself here (see this
    module's docstring on why) - kept for parity with the pack's other
    AI nodes in case a future version of Resemble Enhance exposes a
    proper custom-location download hook."""
    path = pathlib.Path(folder_paths.models_dir) / name
    path.mkdir(parents=True, exist_ok=True)
    try:
        if name not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path(name, str(path))
    except Exception:
        pass
    return path


@contextlib.contextmanager
def _posix_path_windows_shim():
    """
    Resemble Enhance's downloaded hparams.yaml was serialized on the
    authors' Linux training machine, so OmegaConf/PyYAML wrote its
    Path-typed fields (training-only dataset dirs like fg_dir/bg_dir/
    rir_dir - defaults like "data/fg", never read during inference)
    using a `!!python/object/apply:pathlib.PosixPath` YAML tag. Python
    3.12+ refuses to instantiate a PosixPath AT ALL on Windows (not
    just refuses to resolve it as a real path) - so parsing that YAML
    crashes with "cannot instantiate 'PosixPath' on your system" before
    inference ever starts, regardless of what those particular paths
    point to. This is a bug in how Resemble Enhance's checkpoint was
    packaged, not something fixable from this pack's side upstream -
    the standard workaround (used widely for this exact class of
    cross-platform-pickled-Path issue) is to alias PosixPath to
    WindowsPath for the duration of the parse; safe here specifically
    because none of the affected fields are ever read for inference.
    No-op on non-Windows (where PosixPath already works natively, and
    aliasing it to WindowsPath would itself raise the mirror-image
    error).
    """
    if os.name != "nt":
        yield
        return
    prev = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        yield
    finally:
        pathlib.PosixPath = prev


def _patch_resemble_enhance_numpy2_compat():
    """
    resemble_enhance.enhancer.lcfm.cfm.Solver.exponential_decay_mapping
    (used by the CFM ODE solver that "enhance" mode runs - NOT used by
    "denoise_only") does `float(scipy.optimize.fsolve(...))` to turn a
    1-element ndarray result into a Python float. That implicit
    conversion worked fine on the numpy version (1.26.2) pinned in
    resemble-enhance's own requirements.txt, but numpy 2.x - what a
    ComfyUI environment almost certainly already provides, since this
    pack deliberately installs resemble-enhance with `--no-deps`
    precisely to avoid clobbering ComfyUI's own numpy/torch versions
    (see requirements.txt) - raises "only 0-dimensional arrays can be
    converted to Python scalars" for that exact call instead.

    Patches the (static) method in place with an equivalent, numpy-
    2.x-safe version (`.item()` instead of a bare `float(...)` on the
    array). Idempotent - safe to call every run/from every node that
    might load this model, and a no-op if the package isn't installed
    at all (caller handles that separately).
    """
    from resemble_enhance.enhancer.lcfm.cfm import Solver

    def _exponential_decay_mapping_fixed(t, n=4):
        import scipy.optimize

        def h(t, a):
            return (a**t - 1) / (a - 1)

        a = np.asarray(scipy.optimize.fsolve(lambda a: h(1 / n, a) - 0.5, x0=0)).item()
        return h(t, a=a)

    Solver.exponential_decay_mapping = staticmethod(_exponential_decay_mapping_fixed)


# --- Progress reporting (KSampler-style bar on the node) -------------------

class _RoomProgress:
    """
    Maps Resemble Enhance's two nested loops (30 s audio chunks, and the
    CFM solver steps inside each chunk) onto ONE ComfyUI ProgressBar, so
    the node shows a live bar like KSampler does - across batch items and
    across the optional extra denoise pass for wet_mix blending.

    Within a phase, progress = (chunk + steps_done_in_chunk / steps_per_chunk)
    / n_chunks. The denoise-only passes have no solver steps, so they move
    per chunk. Resolution is 1000 units.
    """

    TOTAL = 1000

    def __init__(self, pbar, unique_id, batch):
        self.pbar = pbar
        self.unique_id = unique_id
        self.batch = batch
        self.item = 0
        self.lo, self.hi = 0.0, 1.0
        self.label = ""
        self.n_chunks, self.chunk = 1, 0
        self.steps_per_chunk, self.step = 0, 0

    def begin_item(self, item):
        self.item = item

    def begin_phase(self, lo, hi, label):
        """lo/hi: this phase's share (0-1) of the current batch item."""
        self.lo, self.hi, self.label = lo, hi, label
        self.n_chunks, self.chunk = 1, 0
        self.steps_per_chunk, self.step = 0, 0
        self._emit()

    def chunk_started(self, index, total):
        self.n_chunks = max(total, 1)
        self.chunk = index
        self.step = 0
        self.steps_per_chunk = 0
        self._emit()
        if index < total:
            _send_status(
                self.unique_id,
                f"{self.label} [{self.item + 1}/{self.batch}] chunk {index + 1}/{total}",
            )

    def step_done(self, steps_per_chunk):
        self.steps_per_chunk = max(int(steps_per_chunk), 1)
        self.step += 1
        self._emit()

    def _emit(self):
        within = min(self.step / self.steps_per_chunk, 1.0) if self.steps_per_chunk else 0.0
        phase_frac = min((self.chunk + within) / self.n_chunks, 1.0)
        item_frac = self.lo + (self.hi - self.lo) * phase_frac
        overall = (self.item + item_frac) / self.batch
        if self.pbar is not None:
            self.pbar.update_absolute(int(min(overall, 1.0) * self.TOTAL), self.TOTAL)


@contextlib.contextmanager
def _progress_hooks(progress):
    """
    Temporarily hooks Resemble Enhance's chunk loop (its `trange` in
    resemble_enhance.inference) and CFM solver step (Solver._step) so they
    report into `progress`. Both hooks also poll ComfyUI's interrupt flag,
    so Cancel now works mid-clip too, not only between batch items.
    Everything is restored on exit, even on error/interrupt.
    """
    import resemble_enhance.inference as re_inference
    from resemble_enhance.enhancer.lcfm.cfm import Solver

    orig_trange = re_inference.trange
    orig_step_prop = Solver.__dict__["_step"]

    def trange_with_progress(*args, **kwargs):
        total = len(range(*args))
        for i, value in enumerate(orig_trange(*args, **kwargs)):
            _check_interrupted()
            progress.chunk_started(i, total)
            yield value
        progress.chunk_started(total, total)

    def step_getter(self):
        fn = orig_step_prop.fget(self)

        def counted(*a, **k):
            _check_interrupted()
            out = fn(*a, **k)
            progress.step_done(self.n_steps)
            return out

        return counted

    re_inference.trange = trange_with_progress
    Solver._step = property(step_getter)
    try:
        yield
    finally:
        re_inference.trange = orig_trange
        Solver._step = orig_step_prop


# --- Resemble Enhance (optional AI room-coloration reducer) --------------

_ROOM_REDUCER_CACHE = {}


def _load_room_reducer(device_str):
    """
    Loads (and caches) the Resemble Enhance model, registered with
    ComfyUI's own memory manager like this pack's other model-backed
    nodes. Downloads its ~1GB checkpoint (from the ResembleAI/
    resemble-enhance HF repo, via plain HTTPS - no huggingface_hub
    dependency) the first time this runs - see this module's docstring
    for exactly where it lands and why it isn't redirected into
    ComfyUI/models/ like this pack's other AI nodes.

    The import is kept INSIDE this function (not at module top level)
    for the same reason as mykee_voice_cleanup.py's VoiceFixer loader:
    so ComfyUI doesn't try to import/trigger this on every startup for
    every user, only when this node actually runs.
    """
    cache_key = device_str
    cached_patcher = _ROOM_REDUCER_CACHE.get(cache_key)
    if cached_patcher is not None:
        mm.load_models_gpu([cached_patcher], force_full_load=True)
        return cached_patcher

    try:
        from resemble_enhance.enhancer.inference import load_enhancer
    except ImportError as exc:
        raise RuntimeError(
            "The 'resemble-enhance' package is required for this node. Install it with "
            "`pip install resemble-enhance` in ComfyUI's Python environment, then try again. "
            "(On first use afterward, it downloads its own ~1GB checkpoint from Hugging Face "
            "into the installed resemble_enhance package folder - this needs network access once.)"
        ) from exc

    _patch_resemble_enhance_numpy2_compat()

    with _posix_path_windows_shim():
        model = load_enhancer(None, device_str)  # run_dir=None -> library's own default download path
    model.eval()

    device = torch.device(device_str)
    offload_device = mm.unet_offload_device()
    patcher = ModelPatcher(_make_device_property_settable(model), load_device=device, offload_device=offload_device)
    mm.load_models_gpu([patcher], force_full_load=True)

    _ROOM_REDUCER_CACHE[cache_key] = patcher
    return patcher


def _run_room_reducer(mono, sample_rate, device_str, mode, nfe, solver, lambd, tau, blend_source, wet_mix, progress=None):
    """
    Runs Resemble Enhance on `mono` (1D torch tensor, any sample rate -
    the library resamples internally) and returns
    (wav, 44100, blend_wav_or_None, blend_sr_or_None); the library's own
    inference() handles chunking/crossfading long clips internally, so
    there's no manual chunking loop here (unlike MykeeAIVoiceRestore's
    VoiceFixer wrapper, which has to do that itself). Progress and Cancel
    inside a clip come from `progress` (see _progress_hooks), which hooks
    the library's own chunk loop and solver step.

    `mode` picks which stage runs:
    - "denoise_only": just the denoiser (removes background noise;
      lighter touch, does not directly address room coloration).
    - "enhance": the full enhancer stage (denoiser + CFM-based
      restoration) - this is the one that actually tackles the roomy/
      distant quality, at real GPU compute cost (nfe function
      evaluations of an ODE solver per chunk).

    `blend_source`/`wet_mix` only matter in "enhance" mode: if wet_mix
    is already 1.0 (no blending), or blend_source is "raw", the second
    return pair is (None, None) and the caller blends against the raw
    input itself (the original behavior). If blend_source is
    "denoise_only" and wet_mix < 1.0, this also runs the denoiser pass
    on the same input and returns it as the blend base instead - a
    cleaner (but still-roomy) reference than the raw signal, so mixing
    back doesn't reintroduce the original's room coloration/reverb the
    way blending with raw does. Costs one extra (cheap, non-ODE) model
    pass when it applies.
    """
    _load_room_reducer(device_str)  # ensures loaded/registered with ComfyUI's memory manager

    hooks = _progress_hooks(progress) if progress is not None else contextlib.nullcontext()

    with hooks:
        if mode == "denoise_only":
            from resemble_enhance.enhancer.inference import denoise
            if progress is not None:
                progress.begin_phase(0.0, 1.0, "Denoising")
            wav, out_sr = denoise(mono, sample_rate, device_str)
            return wav.to(dtype=mono.dtype).cpu(), out_sr, None, None

        from resemble_enhance.enhancer.inference import enhance
        needs_blend_pass = wet_mix < 1.0 and blend_source == "denoise_only"
        if progress is not None:
            progress.begin_phase(0.0, 0.9 if needs_blend_pass else 1.0, "Enhancing")
        wav, out_sr = enhance(mono, sample_rate, device_str, nfe=nfe, solver=solver, lambd=lambd, tau=tau)

        blend_wav, blend_sr = None, None
        if needs_blend_pass:
            from resemble_enhance.enhancer.inference import denoise
            if progress is not None:
                progress.begin_phase(0.9, 1.0, "Denoising (blend)")
            blend_wav, blend_sr = denoise(mono, sample_rate, device_str)
            blend_wav = blend_wav.to(dtype=mono.dtype).cpu()

    return wav.to(dtype=mono.dtype).cpu(), out_sr, blend_wav, blend_sr


class MykeeRoomReducer:
    """
    Reduces the "distant mic / roomy" character of a speech recording
    via Resemble Enhance - see this module's docstring for what that
    is, and how it differs from (and complements) Mykee Audio Dereverb.

    - mode - "enhance" is the one that actually addresses room
      coloration (denoiser + CFM restoration stage); "denoise_only"
      just strips background noise and won't do much for a roomy-but-
      quiet recording.
    - room_strength (the model's own "lambd" parameter, 0-1) - how hard
      the enhancer leans on the denoised signal while restoring. Higher
      treats the input as more degraded/farther from the mic and pushes
      harder toward a close, dry, studio-like target; lower is gentler
      and closer to the original tonal character. Only affects "enhance"
      mode.
    - solver / nfe - the enhancer's ODE solver and how many steps it
      takes. More nfe steps = better quality, diminishing returns past
      a point, more compute time. midpoint is a good default; rk4 is
      slower/more accurate, euler is faster/cruder. Only affects
      "enhance" mode.
    - tau - "prior temperature" (0-1) fed into the enhancer's
      restoration stage; higher can add detail/quality but risks
      instability/artifacts on already-decent input. Only affects
      "enhance" mode.
    - wet_mix (0-1) - blends the processed signal back with the
      original (post-downmix, at the original sample rate). 1.0 is
      fully processed; lower it if the full effect sounds over-
      smoothed/artificial for your material.
    - wet_mix_blend_source - what wet_mix blends *toward* when it's
      below 1.0. "raw" (default) blends toward the untouched input -
      simple, but since that still carries the original's room
      coloration/reverb, a low wet_mix can partly bring that back.
      "denoise_only" instead runs a quick denoiser-only pass of the
      same audio and blends toward that - still not as "dry" as the
      full enhance output, but doesn't reintroduce room coloration the
      way raw does, and is a good option if "enhance" alone sounds
      rougher/raspier than you'd like (a known trait of the CFM
      restoration stage inventing plausible-but-imperfect high-
      frequency detail) - blending some of it back toward the cleaner
      denoise_only signal tones that down. Only affects "enhance" mode;
      costs one extra (fast, non-ODE) model pass when it applies.

    Mono only: stereo input is downmixed before processing, and the
    result is always mono (chain Mykee Stereo To Mono / a later DSP
    node afterward if you need stereo back). Output is resampled back
    to the ORIGINAL input's sample rate.

    Needs a GPU for reasonable speed (a CPU run is very slow, especially
    at higher nfe) plus a one-time ~1GB checkpoint download - see
    _load_room_reducer's docstring for where.

    A batched AUDIO input is processed per-item; results are re-padded
    with trailing silence to a common length afterward.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Audio to process. Downmixed to mono regardless of input channel count.",
                }),
                "mode": (["enhance", "denoise_only"], {
                    "default": "enhance",
                    "tooltip": "'enhance': full denoiser + CFM restoration - this is what actually reduces room coloration/distant-mic quality. 'denoise_only': just strips background noise, lighter touch.",
                }),
                "room_strength": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "The model's 'lambd' - how hard the enhancer leans toward a close/dry/studio target vs. the original tonal character. Higher = treats input as more distant/degraded. Only affects 'enhance' mode.",
                }),
                "nfe": ("INT", {
                    "default": 64, "min": 1, "max": 128, "step": 1,
                    "tooltip": "ODE solver steps for the enhancer. More = better quality with diminishing returns, more compute time. Only affects 'enhance' mode.",
                }),
                "solver": (["midpoint", "rk4", "euler"], {
                    "default": "midpoint",
                    "tooltip": "ODE solver method. midpoint is a good default; rk4 is slower/more accurate; euler is faster/cruder. Only affects 'enhance' mode.",
                }),
                "tau": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Prior temperature fed into the enhancer's restoration stage. Higher can add detail but risks artifacts on already-decent input. Only affects 'enhance' mode.",
                }),
                "wet_mix": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blends the processed signal back with the original. 1.0 = fully processed; lower it if the full effect sounds over-smoothed for your material.",
                }),
                "wet_mix_blend_source": (["raw", "denoise_only"], {
                    "default": "raw",
                    "tooltip": "What wet_mix blends the 'enhance' output back toward when wet_mix < 1.0. 'raw': the original pre-processing signal (simple, but can reintroduce room coloration/reverb the enhancer removed). 'denoise_only': runs a denoiser-only pass of the same audio and blends toward that instead - cleaner base, avoids bringing back room coloration, at the cost of one extra (fast) model pass. Only affects 'enhance' mode - 'denoise_only' mode always blends with raw.",
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "auto follows ComfyUI's current torch device. cpu will be very slow, especially at higher nfe.",
                }),
                "unload_model_after_run": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Free Resemble Enhance from VRAM once this run finishes, via ComfyUI's own model manager. Leave off to keep it loaded for faster repeat runs.",
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "run"
    CATEGORY = "Mykee/Audio"

    def run(self, audio, mode, room_strength, nfe, solver, tau, wet_mix, wet_mix_blend_source, device, unload_model_after_run, unique_id=None):
        waveform = audio["waveform"]  # [B, C, S]
        original_sr = audio["sample_rate"]

        device_str = device
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        if device_str == "cuda" and not torch.cuda.is_available():
            _toast(unique_id, "warn", "Mykee Room Reducer",
                   "CUDA requested but not available - falling back to CPU (this will be slow).")
            device_str = "cpu"

        pbar = _ComfyProgressBar(_RoomProgress.TOTAL) if _ComfyProgressBar is not None else None
        batch_count = waveform.shape[0]
        progress = _RoomProgress(pbar, unique_id, batch_count)
        progress.begin_item(0)

        try:
            batch = waveform.shape[0]
            processed = []

            for b in range(batch):
                _check_interrupted()
                progress.begin_item(b)
                _send_status(unique_id, f"Reducing room character [{b + 1}/{batch}]...")
                clip = waveform[b]  # [C, S]
                mono = clip.mean(dim=0)

                out, out_sr, blend_wav, blend_sr = _run_room_reducer(
                    mono, original_sr, device_str, mode, nfe, solver, room_strength, tau, wet_mix_blend_source, wet_mix, progress=progress
                )

                if out_sr != original_sr:
                    out = _resample_torch(out.unsqueeze(0).unsqueeze(0), out_sr, original_sr).squeeze(0).squeeze(0)

                if wet_mix < 1.0:
                    if blend_wav is not None:
                        if blend_sr != original_sr:
                            blend_wav = _resample_torch(blend_wav.unsqueeze(0).unsqueeze(0), blend_sr, original_sr).squeeze(0).squeeze(0)
                        blend_base = blend_wav
                    else:
                        blend_base = mono
                    n = min(out.shape[-1], blend_base.shape[-1])
                    out = wet_mix * out[..., :n] + (1.0 - wet_mix) * blend_base[..., :n]

                processed.append(out.unsqueeze(0))  # [1, S] - mono channel dim

            max_len = max(c.shape[-1] for c in processed)
            padded = [
                c if c.shape[-1] == max_len else torch.nn.functional.pad(c, (0, max_len - c.shape[-1]))
                for c in processed
            ]
            out_waveform = torch.stack(padded, dim=0)

            if pbar is not None:
                pbar.update_absolute(_RoomProgress.TOTAL, _RoomProgress.TOTAL)
            _send_status(unique_id, "Done")
            return ({"waveform": out_waveform, "sample_rate": original_sr},)
        except Exception:
            _send_status(unique_id, "Error")
            raise
        finally:
            if unload_model_after_run:
                _send_status(unique_id, "Unloading model...")
                patcher = _ROOM_REDUCER_CACHE.pop(device_str, None)
                _unload_patcher(patcher)
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _send_status(unique_id, "Idle (model unloaded)")


NODE_CLASS_MAPPINGS = {
    "MykeeRoomReducer": MykeeRoomReducer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeRoomReducer": "Mykee Room Reducer",
}
