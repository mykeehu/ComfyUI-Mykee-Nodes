"""
ComfyUI-Mykee-Nodes / Sigmas Editor

"Mykee - Sigmas Editor" is a graphical editor for a sigma schedule. The curve
is drawn and edited in the browser (web/mykee_sigmas_editor.js): every point is
one sigma value (steps + 1 points), points can be dragged, typed in with up to
6 decimals, generated from the usual scheduler formulas, or loaded from a
connected scheduler. The values travel to this backend in the `sigma_values`
widget (a plain comma separated list, hidden on the node).

Outputs:
  sigmas      - SIGMAS, for SamplerCustom, Mykee - KSampler with Sigmas, ...
  sigmas_text - the same values as text (6 decimals), e.g. for sigmas_text of
                Mykee - KSampler with Sigmas.

The optional `sigmas` input is NOT passed through: its values are only sent to
the editor after a run, so the "Load from input" button can copy them into the
curve. The output always is the curve on the node, with one exception:

"flux (base/max shift)" curves follow the resolution. That generator computes
the shift the way ComfyUI's ModelSamplingFlux node does (base_shift at 256
image tokens, max_shift at 4096 tokens, linear in between and beyond,
shift = e^mu; tokens = width * height / 256). While such a curve has not been
edited since it was generated, the hidden `auto_flux` widget carries its
parameters, and this backend regenerates it from the actual `width` / `height`
(which may come through links, so the browser cannot know them in advance).
The regenerated values are sent back to the editor.
"""

import json
import math

import torch

from .mykee_ksampler_sigmas import MAX_SIGMA_STEPS, _parse_sigma_text, _validate_sigmas

# Kept in sync with web/mykee_sigmas_editor.js.
MAX_EDITOR_STEPS = 500
DEFAULT_VALUES = "1.000000, 0.750000, 0.500000, 0.250000, 0.000000"
MAX_AUTO_TEXT = 1000
EPS = 1e-6


def flux_shift(base_shift, max_shift, width, height):
    """ComfyUI ModelSamplingFlux: resolution dependent shift (returns e^mu)."""
    mm = (max_shift - base_shift) / (4096 - 256)
    b = base_shift - mm * 256
    mu = (width * height / 256.0) * mm + b
    return math.exp(min(mu, 50.0))


def flow_curve(steps, start, shift):
    """Flow schedule sigma = s*t / (1 + (s-1)*t), t evenly down to 0, first
    value = start. Same maths and rounding as web/mykee_sigmas_editor.js."""
    s = shift
    t0 = start / (s - (s - 1) * start)
    out = []
    for i in range(steps + 1):
        t = t0 * (1 - i / steps)
        out.append((s * t) / (1 + (s - 1) * t))
    out[-1] = 0.0
    values = [round(max(0.0, v), 6) for v in out]
    for i in range(1, len(values)):
        if values[i] > values[i - 1] - EPS:
            values[i] = round(max(0.0, values[i - 1] - EPS), 6)
    return values


def _parse_auto(text):
    """The hidden auto_flux widget: '' or {"base": .., "max": .., "start": ..}."""
    if not isinstance(text, str) or not text.strip():
        return None
    if len(text) > MAX_AUTO_TEXT:
        raise ValueError("Sigmas Editor: auto_flux data too long.")
    try:
        data = json.loads(text)
    except ValueError:
        raise ValueError("Sigmas Editor: auto_flux data is not valid.") from None
    if not isinstance(data, dict):
        raise ValueError("Sigmas Editor: auto_flux data is not valid.")
    try:
        base = float(data["base"])
        top = float(data["max"])
        start = float(data["start"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("Sigmas Editor: auto_flux data is not valid.") from None
    if not all(math.isfinite(v) for v in (base, top, start)):
        raise ValueError("Sigmas Editor: auto_flux data is not valid.")
    if not (0.0 <= base <= 100.0 and 0.0 <= top <= 100.0 and 0.0 < start <= 1.0):
        raise ValueError("Sigmas Editor: base / max shift must be 0..100 and start 0..1.")
    return base, top, start


def _format_sigmas(values):
    return ", ".join(f"{v:.6f}" for v in values)


class MykeeSigmasEditor:
    CATEGORY = "Mykee/Sampling"
    FUNCTION = "build"
    RETURN_TYPES = ("SIGMAS", "STRING")
    RETURN_NAMES = ("sigmas", "sigmas_text")
    DESCRIPTION = (
        "Graphical sigma schedule editor: drag the points, type the values, or "
        "generate a curve (flow shift, Karras, exponential, linear). The "
        "optional sigmas input can be loaded into the editor after a run."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {
                    "default": 4, "min": 1, "max": MAX_EDITOR_STEPS,
                    "tooltip": (
                        "Number of steps (points - 1). Changing it resamples the "
                        "current curve, so its shape is kept."
                    ),
                }),
                "sigma_max": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 1000.0, "step": 0.01,
                    "tooltip": (
                        "Top of the chart and of dragging. 1.0 for flow models "
                        "(Flux, Qwen Image, Wan, SD3...), about 14.61 for SD 1.5 / SDXL."
                    ),
                }),
                "log_view": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Logarithmic chart (useful for SD 1.5 / SDXL ranges). Display only.",
                }),
                "width": ("INT", {
                    "default": 1024, "min": 16, "max": 16384, "step": 8,
                    "tooltip": (
                        "Image width for the 'flux (base/max shift)' curve "
                        "(can be connected). Not used by the other curves."
                    ),
                }),
                "height": ("INT", {
                    "default": 1024, "min": 16, "max": 16384, "step": 8,
                    "tooltip": (
                        "Image height for the 'flux (base/max shift)' curve "
                        "(can be connected). Not used by the other curves."
                    ),
                }),
                "sigma_values": ("STRING", {
                    "default": DEFAULT_VALUES,
                    "multiline": False,
                    "tooltip": "The curve values (managed by the editor on the node).",
                }),
                "auto_flux": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Managed by the editor: parameters of an unedited flux curve.",
                }),
            },
            "optional": {
                "sigmas": ("SIGMAS", {
                    "tooltip": (
                        "Optional: a scheduler's sigmas. After a run the "
                        "'Load from input' button copies them into the editor."
                    ),
                }),
            },
        }

    def build(self, steps, sigma_max, log_view, sigma_values, width=1024, height=1024,
              auto_flux="", sigmas=None):
        try:
            values = _parse_sigma_text(sigma_values)
            if values is not None:
                values = _validate_sigmas(values)
        except ValueError as exc:
            message = str(exc).split(": ", 1)[-1]
            raise ValueError(f"Sigmas Editor: {message}") from None
        if values is None or values.numel() < 2:
            raise ValueError("Sigmas Editor: the curve needs at least 2 points.")
        if values.numel() > MAX_EDITOR_STEPS + 1:
            raise ValueError(f"Sigmas Editor: too many points (max {MAX_EDITOR_STEPS + 1}).")
        if bool((values[1:] > values[:-1]).any()):
            raise ValueError("Sigmas Editor: the values must not increase from one point to the next.")

        ui = {}
        auto = _parse_auto(auto_flux)
        if auto is not None:
            w = max(16, min(16384, int(width)))
            h = max(16, min(16384, int(height)))
            base, top, start = auto
            shift = flux_shift(base, top, w, h)
            regenerated = flow_curve(int(values.numel()) - 1, start, shift)
            values = torch.tensor(regenerated, dtype=torch.float32)
            ui["auto_values"] = regenerated
            ui["auto_info"] = [w, h, round(shift, 4)]

        out_sigmas = values.to(torch.float32)
        text = _format_sigmas(out_sigmas.tolist())

        if sigmas is not None:
            try:
                incoming = _validate_sigmas(sigmas)
            except ValueError as exc:
                print(f"[Mykee Sigmas Editor] input sigmas not usable for loading: {exc}")
                incoming = None
            if incoming is not None and 2 <= incoming.numel() <= min(MAX_EDITOR_STEPS, MAX_SIGMA_STEPS) + 1:
                ui["input_sigmas"] = [round(float(v), 6) for v in incoming.flatten().tolist()]

        return {"ui": ui, "result": (out_sigmas, text)}


NODE_CLASS_MAPPINGS = {
    "MykeeSigmasEditor": MykeeSigmasEditor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeSigmasEditor": "Mykee - Sigmas Editor",
}
