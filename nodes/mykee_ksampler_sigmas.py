"""
ComfyUI-Mykee-Nodes / KSampler with Sigmas

"Mykee - KSampler with Sigmas" is the core KSampler with two extra, optional
ways to give it a ready-made noise schedule:

- `sigmas`      - SIGMAS input socket (from BasicScheduler, KarrasScheduler,
                  AlignYourStepsScheduler, SplitSigmas etc.),
- `sigmas_text` - a text field where the sigma values can be typed in
                  (e.g. "14.61, 7.49, 3.08, 0.0"); it can be connected too.

The `Switch_to_sigmas` toggle (bottom of the node, OFF by default) picks the
source, so the sigma settings can stay on the node while it is used with
another model:

- OFF -> steps / scheduler / denoise, exactly like the core KSampler; sigmas
         and sigmas_text are ignored.
- ON  -> connected `sigmas` if there is one, otherwise `sigmas_text`; steps /
         scheduler are ignored. ON with neither of them is an error.
         `denoise` stays active and is applied to the schedule, the way
         `denoise_mode` says:
           keep steps - same number of steps, spread over the lower part of
                        the schedule curve (like the core KSampler's denoise),
           cut        - the first steps are dropped (like SplitSigmasDenoise,
                        or KSamplerAdvanced's start_at_step).

The frontend greys out whatever is not used. sampler_name, seed and cfg are
always used.

The sampling itself is ComfyUI's own code path (comfy.sample.sample, which
already accepts a ready-made sigma schedule), so with BasicScheduler plugged
into `sigmas` the result is identical to the plain KSampler with the same
settings. Based on common_ksampler() in ComfyUI's nodes.py (GPL-3.0).
"""

import inspect

import torch

import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview

# A schedule longer than this is certainly a wiring mistake, not a real
# schedule; refuse it instead of starting an endless run.
MAX_SIGMA_STEPS = 10000
# Upper limit for the typed / connected sigma text (characters).
MAX_SIGMA_TEXT = 200000


def _accepted_kwargs(func, candidates):
    """Only the keyword arguments the installed ComfyUI version knows."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return {}
    return {k: v for k, v in candidates.items() if k in params and v is not None}


def _validate_sigmas(sigmas):
    if not isinstance(sigmas, torch.Tensor):
        raise ValueError("sigmas: expected a SIGMAS tensor from a scheduler node.")
    if sigmas.ndim != 1:
        raise ValueError(f"sigmas: expected a 1-D schedule, got shape {tuple(sigmas.shape)}.")
    if sigmas.numel() > MAX_SIGMA_STEPS + 1:
        raise ValueError(f"sigmas: schedule too long ({sigmas.numel() - 1} steps, max {MAX_SIGMA_STEPS}).")
    if not torch.is_floating_point(sigmas):
        sigmas = sigmas.float()
    if sigmas.numel() and (not torch.isfinite(sigmas).all() or (sigmas < 0).any()):
        raise ValueError("sigmas: the schedule contains negative, NaN or infinite values.")
    return sigmas


def _parse_sigma_text(text):
    """'14.6, 7.5 3.1; 0' (or a pasted '[14.6, 7.5, ...]') -> 1-D float tensor.
    Returns None for an empty / whitespace-only text."""
    if not isinstance(text, str):
        raise ValueError("sigmas_text: expected text.")
    if len(text) > MAX_SIGMA_TEXT:
        raise ValueError(f"sigmas_text: text too long (max {MAX_SIGMA_TEXT} characters).")
    cleaned = text.strip()
    if cleaned.startswith("tensor(") and cleaned.endswith(")"):
        cleaned = cleaned[len("tensor("):-1]
    cleaned = cleaned.replace("[", " ").replace("]", " ")
    for sep in (",", ";", "\n", "\r", "\t"):
        cleaned = cleaned.replace(sep, " ")
    parts = cleaned.split()
    if not parts:
        return None
    if len(parts) > MAX_SIGMA_STEPS + 1:
        raise ValueError(f"sigmas_text: too many values (max {MAX_SIGMA_STEPS + 1}).")
    values = []
    for part in parts:
        try:
            values.append(float(part))
        except ValueError:
            shown = part if len(part) <= 40 else part[:40] + "..."
            raise ValueError(f"sigmas_text: '{shown}' is not a number.") from None
    return torch.tensor(values, dtype=torch.float32)


DENOISE_MODES = ["keep steps", "cut"]


def _interp_curve(values, positions):
    """Sample a decreasing sigma curve at fractional indices (log-linear
    between neighbours when every value is positive, linear otherwise)."""
    n = values.numel()
    lo = positions.floor().long().clamp(0, n - 1)
    hi = (lo + 1).clamp(max=n - 1)
    frac = (positions - lo.to(positions.dtype)).clamp(0.0, 1.0)
    if bool((values > 0).all()):
        logs = values.log()
        return (logs[lo] + (logs[hi] - logs[lo]) * frac).exp()
    return values[lo] + (values[hi] - values[lo]) * frac


def _apply_denoise(sigmas, denoise, mode):
    """Apply a denoise < 1 to a full schedule. Returns the new schedule
    (fewer than 2 values = nothing to sample)."""
    if denoise >= 1.0:
        return sigmas
    steps = sigmas.numel() - 1
    if denoise <= 0.0 or steps < 1:
        return sigmas[:1]

    if mode == "cut":
        # Same maths as the core SplitSigmasDenoise (low_sigmas output).
        keep = round(steps * denoise)
        return sigmas[-(keep + 1):] if keep >= 1 else sigmas[:1]

    # "keep steps": what the core KSampler does with denoise - a longer schedule
    # (steps / denoise) of which only the last `steps` steps are run. Here the
    # longer schedule is made by resampling the given curve.
    ends_at_zero = float(sigmas[-1]) == 0.0
    curve = sigmas[:-1] if ends_at_zero else sigmas
    n = curve.numel()
    if n < 2:
        keep = round(steps * denoise)
        return sigmas[-(keep + 1):] if keep >= 1 else sigmas[:1]
    # Length of the imagined longer schedule, rounded down like the core
    # KSampler does (int(steps / denoise)).
    total = max(int(n / denoise), n)
    if total == n:
        return sigmas
    start = (total - n) / (total - 1)        # where its last n points begin (0..1)
    u = start + (1.0 - start) * torch.arange(n, dtype=torch.float64) / (n - 1)
    out = _interp_curve(curve.double(), u * (n - 1)).to(sigmas.dtype)
    if ends_at_zero:
        out = torch.cat([out, sigmas[-1:]])
    return out


class MykeeKSamplerSigmas:
    CATEGORY = "Mykee/Sampling"
    FUNCTION = "sample"
    RETURN_TYPES = ("LATENT",)
    DESCRIPTION = (
        "KSampler with an optional sigmas input. Without sigmas it behaves "
        "exactly like the core KSampler. With Switch_to_sigmas ON the schedule "
        "comes from the sigmas input (or the typed sigmas_text) and steps / "
        "scheduler / denoise are ignored."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "The random seed used for creating the noise.",
                }),
                "steps": ("INT", {
                    "default": 20, "min": 1, "max": 10000,
                    "tooltip": "Number of steps. Ignored when Switch_to_sigmas is ON.",
                }),
                "cfg": ("FLOAT", {
                    "default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
                    "tooltip": "Classifier-Free Guidance scale.",
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "tooltip": "The sampling algorithm. Always used, also with sigmas connected.",
                }),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                    "tooltip": "The noise schedule. Ignored when Switch_to_sigmas is ON.",
                }),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "Amount of denoising (lower = keeps more of the input "
                        "image). With Switch_to_sigmas ON it is applied to the "
                        "sigma schedule as denoise_mode says."
                    ),
                }),
            },
            "optional": {
                "sigmas": ("SIGMAS", {
                    "tooltip": (
                        "Optional ready-made noise schedule. Used when "
                        "Switch_to_sigmas is ON; takes priority over sigmas_text."
                    ),
                }),
                "sigmas_text": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Sigma values typed in, separated by commas or spaces "
                        "(e.g. 14.61, 7.49, 3.08, 0.0). Used when Switch_to_sigmas "
                        "is ON and the sigmas input is not connected."
                    ),
                }),
                "denoise_mode": (DENOISE_MODES, {
                    "default": "keep steps",
                    "tooltip": (
                        "How denoise < 1 is applied to the sigmas (Switch_to_sigmas "
                        "ON only). keep steps = same number of steps on the lower "
                        "part of the curve (like KSampler). cut = the first steps "
                        "are dropped (like SplitSigmasDenoise / KSamplerAdvanced "
                        "start_at_step)."
                    ),
                }),
                "Switch_to_sigmas": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "OFF = steps / scheduler / denoise (normal KSampler). "
                        "ON = the schedule comes from sigmas (or sigmas_text)."
                    ),
                }),
            },
        }

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive,
               negative, latent_image, denoise=1.0, sigmas=None, sigmas_text="",
               denoise_mode="keep steps", Switch_to_sigmas=False):
        latent = latent_image
        samples_in = latent["samples"]

        if not Switch_to_sigmas:
            sigmas = None  # the switch is OFF: plain KSampler
        elif sigmas is None:
            sigmas = _parse_sigma_text(sigmas_text) if sigmas_text else None
            if sigmas is None:
                raise ValueError(
                    "Switch_to_sigmas is ON, but there is no schedule: connect "
                    "the sigmas input or type values into sigmas_text (or turn "
                    "the switch OFF to use steps / scheduler / denoise)."
                )

        if sigmas is not None:
            sigmas = _validate_sigmas(sigmas)
            if denoise_mode not in DENOISE_MODES:
                raise ValueError(f"denoise_mode: unknown value {denoise_mode!r}.")
            sigmas = _apply_denoise(sigmas, float(denoise), denoise_mode)
            if sigmas.numel() < 2:
                # Nothing to do (e.g. the empty half of SplitSigmas):
                # pass the latent through unchanged, like SamplerCustom.
                return (latent.copy(),)
            steps = int(sigmas.numel()) - 1
            denoise = 1.0  # the schedule already contains any partial denoise

        samples_in = comfy.sample.fix_empty_latent_channels(
            model, samples_in,
            **_accepted_kwargs(comfy.sample.fix_empty_latent_channels, {
                "downscale_ratio_spacial": latent.get("downscale_ratio_spacial"),
                "downscale_ratio_temporal": latent.get("downscale_ratio_temporal"),
            }),
        )

        batch_inds = latent.get("batch_index")
        noise = comfy.sample.prepare_noise(samples_in, seed, batch_inds)
        noise_mask = latent.get("noise_mask")

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler, positive, negative,
            samples_in, denoise=denoise, noise_mask=noise_mask, sigmas=sigmas,
            callback=callback, disable_pbar=disable_pbar, seed=seed,
        )

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples
        return (out,)


NODE_CLASS_MAPPINGS = {
    "MykeeKSamplerSigmas": MykeeKSamplerSigmas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeKSamplerSigmas": "Mykee - KSampler with Sigmas",
}
