"""
ComfyUI-Mykee-Nodes / Audio Dereverb

Reduces room reverb/echo in an AUDIO clip using WPE (Weighted Prediction
Error) - a classical, non-neural late-reverberation suppression
algorithm (not a generative/voice-conversion model, so it can't alter
the speaker's timbre - it works by linear prediction in the STFT domain
to cancel out the delayed, predictable part of the signal, i.e. the
reverb tail, while leaving the direct sound/early reflections alone).
Well-established in speech processing (used e.g. in the REVERB Challenge
baseline, Kaldi, ESPnet) - see https://github.com/fgnt/nara_wpe, the
library this node wraps.

Deliberately kept as its own node (not folded into Mykee Audio DSP
Cleanup) so it can be placed anywhere in the chain relative to the other
Mykee audio nodes - e.g. before Mykee AI Voice Restore, so the AI stage
receives an already-dereverbed signal.
"""

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

import comfy.model_management as mm


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


def _wpe_dereverb(clip, sample_rate, taps, delay, iterations, fft_size, hop_size):
    """
    Runs WPE on `clip` (torch tensor, shape [C, S]) and returns a tensor
    of the same shape (trimmed/padded back to the original length - the
    STFT/ISTFT round trip can shift it by a few samples). Multi-channel
    input (e.g. stereo) gives WPE more to work with (it predicts each
    channel's reverb using cross-channel statistics too), but it works on
    mono (single-channel, prediction from that channel's own past only)
    as well, just less powerfully.
    """
    from nara_wpe.wpe import wpe
    from nara_wpe.utils import stft, istft

    x = clip.detach().cpu().numpy().astype(np.float64)  # [C, S]
    n_samples = x.shape[-1]

    Y = stft(x, size=fft_size, shift=hop_size)  # (C, T, F)
    Y_fdt = Y.transpose(2, 0, 1)  # (F, D=C, T) - the shape wpe() expects

    Z_fdt = wpe(Y_fdt, taps=taps, delay=delay, iterations=iterations)

    Z = Z_fdt.transpose(1, 2, 0)  # back to (C, T, F)
    z = istft(Z, size=fft_size, shift=hop_size)  # [C, S']

    if z.shape[-1] > n_samples:
        z = z[..., :n_samples]
    elif z.shape[-1] < n_samples:
        z = np.pad(z, ((0, 0), (0, n_samples - z.shape[-1])))

    return torch.from_numpy(z).to(dtype=clip.dtype)


def _has_duplicated_channels(clip, rtol=1e-6):
    """
    Detects whether a multi-channel clip's channels are (numerically)
    identical duplicates of each other - e.g. a mono source that was
    force-converted to "stereo" by duplicating its single channel into
    both (Mykee Stereo To Mono with convert_to_mono OFF does exactly
    this: "a mono input is duplicated to both channels").

    WPE's multi-channel mode relies on the channels being at least
    somewhat decorrelated - it builds its prediction filter from their
    cross-channel covariance, and perfectly linearly-dependent
    (duplicated) channels make that covariance matrix singular. Rather
    than raising a clean error, this pushes the underlying linear solve
    into a near-singular regime that can blow the output up into loud,
    harsh, "electronic"-sounding noise well outside the input's original
    level - reproduced and confirmed on real audio: a duplicated-stereo
    clip that peaked at 0.49 came out of WPE peaking at 1.93, while the
    same content run as true mono (or as stereo with even a tiny amount
    of independent per-channel noise) stayed stable at ~0.49.

    Genuinely independent stereo or multi-mic input is unaffected by
    this check (and unaffected by the fallback below) - only exact or
    near-exact channel duplicates trigger it.
    """
    if clip.shape[0] < 2:
        return False
    ref = clip[0]
    peak = ref.abs().max().clamp(min=1e-9)
    for c in range(1, clip.shape[0]):
        if (clip[c] - ref).abs().max() > rtol * peak:
            return False
    return True


class MykeeAudioDereverb:
    """
    Reduces room reverb/echo in an AUDIO clip via WPE (Weighted
    Prediction Error) - see this module's docstring for what that is and
    why it's a separate node.

    - taps - filter length (in STFT frames) used to predict/cancel the
      reverb tail. More taps can suppress a longer reverb tail but costs
      more compute and, past a point, can start coloring the sound.
    - delay - how many STFT frames right after the direct sound are left
      completely untouched (a guard interval), so WPE only targets the
      LATE reverb, not early reflections/the direct sound itself.
    - iterations - how many times WPE re-estimates its prediction filter
      against its own output; more can sharpen the result but with
      diminishing returns.
    - fft_size / hop_size - STFT frame size / hop, in samples. The
      defaults are tuned for typical speech sample rates (44.1/48kHz);
      lower them for a much lower sample rate input, or leave as-is
      otherwise.
    - wet_mix (0-1) - blends the dereverbed signal back with the
      original. 1.0 is fully dereverbed; lower it if the full effect
      sounds over-processed/artifacted for your material, or if you just
      want to take the edge off rather than eliminate the reverb.

    WPE's effectiveness varies a lot by recording - it's genuinely
    strongest with real multi-microphone-array diversity (its original
    use case), and more modest on a single mic or a plain stereo
    recording (not spaced mics). There's no substitute for listening and
    adjusting taps/delay/iterations/wet_mix by ear for your specific
    material.

    Guards against one specific failure mode automatically: if a clip's
    channels are exact/near-exact duplicates of each other (e.g. a mono
    source that was force-converted to "stereo" by duplicating its one
    channel - see Mykee Stereo To Mono's convert_to_mono=OFF), WPE's
    multi-channel prediction can become numerically unstable and produce
    loud, harsh, "electronic" noise well outside the input's level - see
    _has_duplicated_channels' docstring for why. This node detects that
    case and processes as mono internally instead, then restores the
    original channel count - no separate toggle needed, and genuinely
    independent stereo/multi-mic input is unaffected.

    This is a CPU-only, numpy-based algorithm (no GPU acceleration, no
    model download) - runs at roughly 0.3-0.5x realtime on a typical
    desktop CPU, so a long clip will take a while; there's a progress
    status while it runs.

    A batched AUDIO input is processed per-item.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Audio to dereverb.",
                }),
                "taps": ("INT", {
                    "default": 10, "min": 1, "max": 30, "step": 1,
                    "tooltip": "Filter length (STFT frames) used to predict/cancel the reverb tail. More can suppress a longer tail but costs more compute and can start coloring the sound past a point.",
                }),
                "delay": ("INT", {
                    "default": 3, "min": 1, "max": 10, "step": 1,
                    "tooltip": "STFT frames right after the direct sound left untouched (a guard interval) - WPE only targets reverb AFTER this, not early reflections/the direct sound.",
                }),
                "iterations": ("INT", {
                    "default": 3, "min": 1, "max": 10, "step": 1,
                    "tooltip": "How many times WPE re-estimates its prediction filter against its own output. More can sharpen the result, with diminishing returns and more compute time.",
                }),
                "fft_size": ("INT", {
                    "default": 2048, "min": 256, "max": 8192, "step": 256,
                    "tooltip": "STFT frame size in samples. The default suits 44.1/48kHz speech; lower it for a much lower sample rate input.",
                }),
                "hop_size": ("INT", {
                    "default": 512, "min": 64, "max": 4096, "step": 64,
                    "tooltip": "STFT hop size in samples - usually fft_size / 4.",
                }),
                "wet_mix": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blends the dereverbed signal back with the original. 1.0 = fully dereverbed; lower it if the full effect sounds over-processed for your material.",
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

    def run(self, audio, taps, delay, iterations, fft_size, hop_size, wet_mix, unique_id=None):
        try:
            from nara_wpe.wpe import wpe  # noqa: F401 - import check only
        except ImportError as exc:
            raise RuntimeError(
                "The 'nara_wpe' package is required for this node. Install it with "
                "`pip install nara_wpe` in ComfyUI's Python environment, then try again."
            ) from exc

        waveform = audio["waveform"]  # [B, C, S]
        sample_rate = audio["sample_rate"]

        pbar = _make_progress_bar()
        _set_progress(pbar, 0)

        try:
            processed = []
            batch = waveform.shape[0]
            for b in range(batch):
                _check_interrupted()
                _send_status(unique_id, f"Dereverbing [{b + 1}/{batch}]...")
                clip = waveform[b]  # [C, S]

                if _has_duplicated_channels(clip):
                    _toast(unique_id, "info", "Mykee Audio Dereverb",
                           "All channels are identical (e.g. a mono source forced to "
                           "stereo) - processing as mono to avoid WPE's instability on "
                           "duplicated channels, then restoring the channel count.")
                    dereverbed = _wpe_dereverb(clip[:1], sample_rate, taps, delay, iterations, fft_size, hop_size)
                    dereverbed = dereverbed.repeat(clip.shape[0], 1)
                else:
                    dereverbed = _wpe_dereverb(clip, sample_rate, taps, delay, iterations, fft_size, hop_size)

                if wet_mix < 1.0:
                    dereverbed = wet_mix * dereverbed + (1.0 - wet_mix) * clip

                processed.append(dereverbed)
                _set_progress(pbar, int(90 * (b + 1) / batch))

            out_waveform = torch.stack(processed, dim=0)

            _set_progress(pbar, 100)
            _send_status(unique_id, "Done")
            return ({"waveform": out_waveform, "sample_rate": sample_rate},)
        except Exception:
            _send_status(unique_id, "Error")
            raise


NODE_CLASS_MAPPINGS = {
    "MykeeAudioDereverb": MykeeAudioDereverb,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeAudioDereverb": "Mykee Audio Dereverb",
}
