"""
ComfyUI-Mykee-Nodes / Voice Cleanup

Two independent audio-restoration nodes, meant to be chained in whichever
order suits the material:

- MykeeAudioDSPCleanup: fast, dependency-light DSP fixes (hum removal,
  gentle high-shelf "de-mufflening", spectral noise reduction). Pairs
  with MykeeAudioLeveling (in mykee_audio.py) for level drift/
  normalization.
- MykeeAIVoiceRestore: an AI restoration pass (VoiceFixer - denoise +
  de-reverb + bandwidth extension in one model, specifically built to
  restore a degraded speech recording toward studio quality while
  keeping the speaker's own voice/identity). Slower, needs a one-time
  model download.
"""

import contextlib
import os

import numpy as np
import torch

import folder_paths
import comfy.model_management as mm
from comfy.model_patcher import ModelPatcher

try:
    from comfy.utils import ProgressBar as _ComfyProgressBar
except Exception:
    _ComfyProgressBar = None

try:
    from server import PromptServer
except ImportError:
    PromptServer = None

try:
    from comfy_execution.graph import ExecutionBlocker
except ImportError:
    ExecutionBlocker = None


def _blocked_output():
    """An output that should NOT fire this run (e.g. one of
    MykeeAIVoiceRestore's three named-mode outputs when ai_mode isn't
    'all'). Returns ComfyUI's own ExecutionBlocker(None) - the documented
    way to 'disable an output' for one run: any node relying SOLELY on
    that output skips execution entirely, rather than running on a
    placeholder None value. Falls back to plain None on a ComfyUI old
    enough to lack this (degrades to a None value flowing downstream
    instead of true non-execution, but doesn't crash node loading)."""
    return ExecutionBlocker(None) if ExecutionBlocker is not None else None

from .mykee_audio import (
    _resample as _resample_torch,
    _spectral_cleanup,
)

# --- ComfyUI-managed model lifecycle (same pattern as mykee_voice_match.py -
# see that file's module-level comment for the full rationale) -----------

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
        print(f"[Mykee Voice Cleanup] Note: ComfyUI-managed unload failed ({exc}); "
              "falling back to a manual .to('cpu').")
        try:
            patcher.model.to("cpu")
        except Exception:
            pass


def _check_interrupted():
    check = getattr(mm, "throw_exception_if_processing_interrupted", None)
    if check is not None:
        check()


def _toast(node_id, severity, summary, detail):
    """Non-blocking ComfyUI toast notification - see mykee_character.py's
    _toast() for the full rationale; duplicated here (rather than
    cross-imported) so this file stays self-contained like this pack's
    other pipeline modules."""
    if PromptServer is None or getattr(PromptServer, "instance", None) is None:
        return
    try:
        PromptServer.instance.send_sync(
            "mykee.toast",
            {"node": node_id, "severity": severity, "summary": summary, "detail": detail, "life": 6000},
        )
    except Exception:
        pass


def _send_status(unique_id, text):
    if PromptServer is None or getattr(PromptServer, "instance", None) is None:
        return
    try:
        PromptServer.instance.send_sync("mykee.voice_cleanup_status", {"node": unique_id, "text": text})
    except Exception:
        pass


def _make_progress_bar():
    return _ComfyProgressBar(100) if _ComfyProgressBar is not None else None


def _set_progress(pbar, value):
    if pbar is not None:
        pbar.update_absolute(value, 100)


# --- VoiceFixer (optional AI restoration) --------------------------------

_VOICEFIXER_CACHE = {}


def _model_folder(name):
    """Registers (if needed) and returns ComfyUI/models/<name>, creating it
    if missing. Same helper as mykee_voice_match.py's (duplicated here so
    this file stays self-contained, matching this pack's convention)."""
    path = os.path.join(folder_paths.models_dir, name)
    os.makedirs(path, exist_ok=True)
    try:
        if name not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path(name, path)
    except Exception:
        pass
    return path


@contextlib.contextmanager
def _voicefixer_cache_redirect():
    """Redirects voicefixer's checkpoint downloads from its hardcoded
    ~/.cache/voicefixer/ into ComfyUI/models/VoiceFixer/, matching where
    this pack's other AI nodes keep their weights.

    voicefixer's own code has NO supported hook for this (no env var, no
    constructor argument, no config file) - the exact string
    os.path.join(os.path.expanduser("~"), ".cache/voicefixer/...") is
    hardcoded at the top of two of its submodules' __init__.py files, and
    that download-if-missing check runs the moment `import voicefixer`
    (or anything that imports it) executes - before any object exists to
    pass a path into even if the package did accept one. So the only way
    to redirect it at all is to make os.path.expanduser("~") itself
    resolve somewhere else for the moment those imports run: this
    temporarily overrides the HOME/USERPROFILE environment variables
    (what expanduser("~") actually reads) to point at
    ComfyUI/models/VoiceFixer/, and restores the real ones immediately
    after - the exact same technique already used elsewhere in this pack
    for Seed-VC (HF_HUB_CACHE) and Demucs (TORCH_HOME), just applied via
    HOME/USERPROFILE here since that's the only lever voicefixer exposes.

    Narrowly scoped to just the `from voicefixer import VoiceFixer` +
    `VoiceFixer()` call this wraps (both the module-level checkpoint
    checks AND VoiceFixer.__init__'s own re-resolution of the analysis-
    module checkpoint path need to be inside this scope - see
    _load_voicefixer()). One known side effect: any OTHER library
    voicefixer happens to import that also reads HOME/USERPROFILE for
    its own unrelated cache during this same brief window (e.g.
    matplotlib, a declared voicefixer dependency, defaults its config
    dir this way) will likewise write into ComfyUI/models/VoiceFixer/
    instead of the real home directory - harmless (just a stray
    subfolder there instead of in the user's actual home), not a
    functional problem.
    """
    target = _model_folder("VoiceFixer")
    prev_home = os.environ.get("HOME")
    prev_userprofile = os.environ.get("USERPROFILE")
    os.environ["HOME"] = target
    os.environ["USERPROFILE"] = target
    try:
        yield
    finally:
        if prev_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = prev_home
        if prev_userprofile is None:
            os.environ.pop("USERPROFILE", None)
        else:
            os.environ["USERPROFILE"] = prev_userprofile


def _load_voicefixer(device_str):
    """
    Loads (and caches) VoiceFixer, an AI speech-restoration model
    (denoise + de-reverb + bandwidth extension in one pass) - see
    https://github.com/haoheliu/voicefixer. Registered with ComfyUI's own
    memory manager like this pack's other model-backed nodes.

    Downloads its checkpoints (from Zenodo, a few hundred MB total) the
    first time `import voicefixer` runs anywhere in the process - which
    is exactly why this import is kept INSIDE this function (called only
    when ai_restore is actually used) rather than at the top of this
    file: importing it at module load time would make ComfyUI itself try
    to download these checkpoints on every startup, for every user,
    whether they use this node or not. The download destination is
    redirected into ComfyUI/models/VoiceFixer/ via
    _voicefixer_cache_redirect() - see that function's docstring for why
    that's the only way to do it.
    """
    cache_key = device_str
    cached_patcher = _VOICEFIXER_CACHE.get(cache_key)
    if cached_patcher is not None:
        mm.load_models_gpu([cached_patcher], force_full_load=True)
        return cached_patcher

    with _voicefixer_cache_redirect():
        try:
            from voicefixer import VoiceFixer
        except ImportError as exc:
            raise RuntimeError(
                "The 'voicefixer' package is required for ai_restore. Install it with "
                "`pip install voicefixer` in ComfyUI's Python environment, then try again. "
                "(On first use afterward, it downloads its own checkpoints from Zenodo into "
                "ComfyUI/models/VoiceFixer/ - this needs network access once.)"
            ) from exc

        vf = VoiceFixer()  # triggers its own checkpoint download on first-ever use
    vf.eval()

    device = torch.device(device_str)
    offload_device = mm.unet_offload_device()
    patcher = ModelPatcher(_make_device_property_settable(vf), load_device=device, offload_device=offload_device)
    mm.load_models_gpu([patcher], force_full_load=True)

    _VOICEFIXER_CACHE[cache_key] = patcher
    return patcher


def _run_voicefixer(mono_44k, sample_rate, device_str, mode):
    """
    Runs VoiceFixer on `mono_44k` (1D torch tensor, MUST already be at
    44100 Hz - the only rate its model supports) and returns a 1D torch
    tensor at 44100 Hz. Reimplements VoiceFixer's own restore_inmem()
    chunking loop (30s chunks) manually, rather than calling it directly,
    so a Cancel/Interrupt can take effect between chunks and so the
    'cuda' flag passed to its internal calls always matches wherever
    ComfyUI actually placed the model (restore_inmem() would otherwise
    move the model itself based on that flag, fighting ComfyUI's own
    placement).
    """
    assert sample_rate == 44100

    patcher = _load_voicefixer(device_str)
    vf = patcher.model
    device = patcher.load_device
    cuda = device.type == "cuda"

    from voicefixer.tools.pytorch_util import from_log, tensor2numpy

    wav_np = mono_44k.detach().cpu().numpy().astype(np.float32)

    seg_length = 44100 * 30
    pieces = []
    break_point = seg_length
    while break_point < wav_np.shape[0] + seg_length:
        _check_interrupted()
        segment = wav_np[break_point - seg_length: break_point]
        if mode == 1:
            segment = vf.remove_higher_frequency(segment)

        sp, mel_noisy = vf._pre(vf._model, segment, cuda)
        with torch.no_grad():
            out_model = vf._model(sp, mel_noisy)
            denoised_mel = from_log(out_model["mel"])
            out = vf._model.vocoder(denoised_mel, cuda=cuda)

        if torch.max(torch.abs(out)) > 1.0:
            out = out / torch.max(torch.abs(out))

        out, _ = vf._trim_center(out, segment)
        pieces.append(out)
        break_point += seg_length

    restored = torch.cat(pieces, dim=-1)
    restored_np = tensor2numpy(restored.squeeze(0))
    return torch.from_numpy(restored_np).to(dtype=mono_44k.dtype)


# --- Nodes ------------------------------------------------------------------

class MykeeAudioDSPCleanup:
    """
    Fast, dependency-light DSP fixes for a recording made without a
    dedicated/close mic - distant, off-axis, "in the background": low,
    wandering level; muffled tonal balance; audible mains hum;
    background room noise. No model download, runs on CPU fine.

    - remove_hum + hum_fundamental_hz + hum_harmonics - notches out AC
      mains hum (the fundamental + that many harmonics) - e.g. from an
      ungrounded cable or a nearby power supply the mic picked up.
    - low_cut_hz - a gentle high-pass to remove room rumble/handling
      noise.
    - high_shelf_gain_db + high_shelf_freq_hz - boosts frequencies above
      that point to restore some presence/air a distant or off-axis mic
      loses (the muffled/dull quality).
    - denoise_strength - reduces steady background noise via spectral
      subtraction against a profile learned from the quietest parts of
      the clip.

    Pairs well chained with MykeeAudioLeveling (level drift/normalize)
    and MykeeAIVoiceRestore - run this one first to clean up hum before
    the AI stage, after to touch up presence on the AI stage's output,
    or on its own for a fast fix that doesn't need a model download.

    A batched AUDIO input is processed per-item; results are re-padded
    (silence/channel-repeated) to a common shape afterward.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Audio to clean up.",
                }),
                "remove_hum": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Notch out AC mains hum (the fundamental + harmonics) - e.g. from an ungrounded cable or a nearby power supply picked up by the mic.",
                }),
                "hum_fundamental_hz": (["50", "60"], {
                    "default": "50",
                    "tooltip": "Mains frequency where you recorded - 50 Hz (Europe and most of the world) or 60 Hz (North America and parts of South America/Asia).",
                }),
                "hum_harmonics": ("INT", {
                    "default": 6, "min": 1, "max": 15, "step": 1,
                    "tooltip": "How many harmonics of the mains frequency to notch out (fundamental + this many multiples of it). Hum is rarely just the fundamental.",
                }),
                "low_cut_hz": ("FLOAT", {
                    "default": 70.0, "min": 0.0, "max": 300.0, "step": 5.0,
                    "tooltip": "Gently high-passes below this frequency to remove room rumble/handling noise. 0 disables. Keep this below a male voice's fundamental (~85 Hz+) so it doesn't thin out the voice.",
                }),
                "high_shelf_gain_db": ("FLOAT", {
                    "default": 6.0, "min": 0.0, "max": 18.0, "step": 0.5,
                    "tooltip": "Boosts frequencies above 'high_shelf_freq_hz' to restore some presence/air a distant or off-axis mic loses (the muffled/dull quality). 0 disables.",
                }),
                "high_shelf_freq_hz": ("FLOAT", {
                    "default": 4000.0, "min": 1000.0, "max": 12000.0, "step": 100.0,
                    "tooltip": "Only used when 'high_shelf_gain_db' > 0. Frequency the presence boost starts ramping in above.",
                }),
                "denoise_strength": ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Reduces steady background noise (room tone, air handling, etc.) via spectral subtraction against a profile learned from the quietest parts of the clip. 0 disables; higher is more aggressive but can start sounding processed/artifacted on already-clean audio.",
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

    def run(
        self,
        audio,
        remove_hum,
        hum_fundamental_hz,
        hum_harmonics,
        low_cut_hz,
        high_shelf_gain_db,
        high_shelf_freq_hz,
        denoise_strength,
        unique_id=None,
    ):
        waveform = audio["waveform"]  # [B, C, S]
        sample_rate = audio["sample_rate"]

        pbar = _make_progress_bar()
        _set_progress(pbar, 0)

        try:
            processed = []
            batch = waveform.shape[0]
            for b in range(batch):
                _check_interrupted()
                _send_status(unique_id, f"Cleaning up [{b + 1}/{batch}]...")
                clip = waveform[b]  # [C, S]

                if remove_hum or low_cut_hz > 0 or high_shelf_gain_db > 0 or denoise_strength > 0:
                    clip = _spectral_cleanup(
                        clip, sample_rate,
                        hum_fundamental_hz=float(hum_fundamental_hz) if remove_hum else 0.0,
                        hum_harmonics=hum_harmonics,
                        low_cut_hz=low_cut_hz,
                        high_shelf_freq_hz=high_shelf_freq_hz,
                        high_shelf_gain_db=high_shelf_gain_db,
                        denoise_strength=denoise_strength,
                    )

                processed.append(clip)
                _set_progress(pbar, int(90 * (b + 1) / batch))

            max_channels = max(c.shape[0] for c in processed)
            max_len = max(c.shape[-1] for c in processed)
            padded = []
            for c in processed:
                if c.shape[0] < max_channels:
                    c = c.repeat(max_channels, 1)
                if c.shape[-1] < max_len:
                    c = torch.nn.functional.pad(c, (0, max_len - c.shape[-1]))
                padded.append(c)
            out_waveform = torch.stack(padded, dim=0)

            _set_progress(pbar, 100)
            _send_status(unique_id, "Done")
            return ({"waveform": out_waveform, "sample_rate": sample_rate},)
        except Exception:
            _send_status(unique_id, "Error")
            raise


class MykeeAIVoiceRestore:
    """
    AI speech restoration via VoiceFixer (denoise + de-reverb + bandwidth
    extension in one pass), trained specifically to restore degraded
    speech while preserving the speaker's own identity/voice. Stronger
    than MykeeAudioDSPCleanup alone, especially for reverb/room tone DSP
    doesn't specifically address - but slower, and needs a GPU for
    reasonable speed plus a one-time model download (see
    _load_voicefixer's docstring for where).

    VoiceFixer only supports 44100 Hz mono, so stereo input is downmixed;
    the output is always resampled back to the ORIGINAL input's sample
    rate, so this node's output rate matches its input rate. The output
    is mono regardless of the input's channel count (chain
    MykeeAudioDSPCleanup after this if you specifically need a stereo
    result - it will just apply the same processing to a single channel).

    ai_mode is VoiceFixer's own mode 0/1/2 choice, plus a 4th 'all'
    option that runs all three and gives you a separate output PER mode
    (e.g. to A/B them with a Preview Audio each) instead of one blended
    result:
    - `balanced` (mode 0, VoiceFixer's default)
    - `denoise_first` (mode 1 - strips very harsh/high-frequency noise
      before restoring; try this if `balanced` leaves noise artifacts)
    - `aggressive` (mode 2 - VoiceFixer's own "more effective on
      seriously damaged speech" mode; can sound less natural on
      already-decent audio)
    - `all` - runs all three; the single `audio` output does NOT fire
      this run (see below), and instead `audio_balanced`/
      `audio_denoise_first`/`audio_aggressive` each carry that mode's
      result.

    Outputs: `audio` carries the result for modes balanced/
    denoise_first/aggressive; the three named outputs
    (`audio_balanced`/`audio_denoise_first`/`audio_aggressive`) carry
    nothing in that case. For `all`, it's the reverse: `audio` carries
    nothing, and the three named outputs each carry their own mode's
    result. "Carries nothing" means the output doesn't fire at all this
    run (via ComfyUI's ExecutionBlocker) - any node relying solely on
    that output simply doesn't execute, rather than receiving an empty/
    None value.

    A batched AUDIO input is processed per-item; results are re-padded
    with trailing silence to a common length afterward.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Audio to restore. Downmixed to mono for processing regardless of input channel count.",
                }),
                "ai_mode": (["balanced", "denoise_first", "aggressive", "all"], {
                    "default": "balanced",
                    "tooltip": "'balanced'/'denoise_first'/'aggressive': VoiceFixer's modes 0/1/2 - result goes to the 'audio' output. 'all': runs all three, result goes to 'audio_balanced'/'audio_denoise_first'/'audio_aggressive' instead (e.g. to compare them with a Preview Audio each) - 'audio' doesn't fire in that case.",
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "auto follows ComfyUI's current torch device. cpu will be very slow.",
                }),
                "unload_model_after_run": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Free VoiceFixer from VRAM once this run finishes, via ComfyUI's own model manager. Leave off to keep it loaded for faster repeat runs.",
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("audio", "audio_balanced", "audio_denoise_first", "audio_aggressive")
    FUNCTION = "run"
    CATEGORY = "Mykee/Audio"

    _MODE_MAP = {"balanced": 0, "denoise_first": 1, "aggressive": 2}

    def run(self, audio, ai_mode, device, unload_model_after_run, unique_id=None):
        waveform = audio["waveform"]  # [B, C, S]
        original_sr = audio["sample_rate"]

        device_str = device
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        if device_str == "cuda" and not torch.cuda.is_available():
            _toast(unique_id, "warn", "Mykee AI Voice Restore",
                   "CUDA requested but not available - falling back to CPU (this will be slow).")
            device_str = "cpu"

        modes_to_run = list(self._MODE_MAP.items()) if ai_mode == "all" else [(ai_mode, self._MODE_MAP[ai_mode])]

        pbar = _make_progress_bar()
        _set_progress(pbar, 0)

        try:
            batch = waveform.shape[0]
            results = {name: [] for name, _ in modes_to_run}
            total_steps = batch * len(modes_to_run)
            step = 0

            for b in range(batch):
                clip = waveform[b]  # [C, S]
                mono = clip.mean(dim=0)
                mono_44k = _resample_torch(mono.unsqueeze(0).unsqueeze(0), original_sr, 44100).squeeze(0).squeeze(0)

                for name, mode_int in modes_to_run:
                    _check_interrupted()
                    _send_status(unique_id, f"Restoring ({name}) [{b + 1}/{batch}]...")
                    restored = _run_voicefixer(mono_44k, 44100, device_str, mode_int)  # [1, S] @ 44100Hz
                    if 44100 != original_sr:
                        restored = _resample_torch(restored.unsqueeze(0), 44100, original_sr).squeeze(0)
                    results[name].append(restored)
                    step += 1
                    _set_progress(pbar, int(90 * step / total_steps))

            def _stack(items):
                max_len = max(c.shape[-1] for c in items)
                padded = [
                    c if c.shape[-1] == max_len else torch.nn.functional.pad(c, (0, max_len - c.shape[-1]))
                    for c in items
                ]
                return {"waveform": torch.stack(padded, dim=0), "sample_rate": original_sr}

            if ai_mode == "all":
                out = (
                    _blocked_output(),
                    _stack(results["balanced"]),
                    _stack(results["denoise_first"]),
                    _stack(results["aggressive"]),
                )
            else:
                out = (_stack(results[ai_mode]), _blocked_output(), _blocked_output(), _blocked_output())

            _set_progress(pbar, 100)
            _send_status(unique_id, "Done")
            return out
        except Exception:
            _send_status(unique_id, "Error")
            raise
        finally:
            if unload_model_after_run:
                _send_status(unique_id, "Unloading model...")
                patcher = _VOICEFIXER_CACHE.pop(device_str, None)
                _unload_patcher(patcher)
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _send_status(unique_id, "Idle (model unloaded)")


NODE_CLASS_MAPPINGS = {
    "MykeeAudioDSPCleanup": MykeeAudioDSPCleanup,
    "MykeeAIVoiceRestore": MykeeAIVoiceRestore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeAudioDSPCleanup": "Mykee Audio DSP Cleanup",
    "MykeeAIVoiceRestore": "Mykee AI Voice Restore",
}
