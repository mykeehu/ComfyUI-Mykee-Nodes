"""Mykee Latent Nyquist Notch.

Removes the component of a LATENT that alternates every single latent pixel
(the Nyquist frequency). In DiT models that patchify the latent in 2x2 blocks
(Flux / Chroma / Qwen Image ...) the patch grid can leave a faint 2-latent-pixel
periodic pattern behind. With an 8x VAE that is a 16 px stripe / grid pattern in
the decoded image, and it gets stronger with resolution.

The idea of a Nyquist notch for the 2 px VAE grid comes from the ComfyUI-DeGrid
node (https://github.com/lunaaispace-eng/ComfyUI-DeGrid, Apache-2.0). That node
works on decoded IMAGES at 2 px; this one works on LATENTS, before the VAE
decode, and uses a different, narrow-band method (see below). No code was copied.

Method (per image of the batch, per channel):
  * three carriers: rows (+1,-1,+1,... along H), columns (along W) and the
    checkerboard (rows x columns);
  * multiply the latent with a carrier and smooth it with a Gaussian: this is
    the local, signed amplitude of that Nyquist component. Slowly drifting
    phase is handled, because the amplitude is allowed to change sign;
  * cells whose raw amplitude is far above the typical one are real fine
    structure (fences, mesh, hard edges) and get ~0 weight BEFORE the
    smoothing, so they neither get removed nor leak into the neighbourhood;
  * the amplitude is clamped (auto: 3 x its median) and subtracted.
A component is only touched when it stands out of the neighbouring frequencies
(strength >= min_strength); otherwise the image passes through untouched.
"""

import math
import time

import torch
import torch.nn.functional as F

_PROBE_OFFSETS = (1.0 / 16.0, 1.0 / 10.0)   # cycles/px away from Nyquist
_NOISE_NORM = 0.81        # median|a| / median|probe| for pure noise
_PROTECT_SIGMA = 1.5      # latent px, small smoothing used to find real structure
_AUTO_LIMIT_MULT = 3.0
_MIN_SIZE = 32            # smaller latents are passed through

_COMPONENTS = (
    ("rows", "rows (horizontal stripes)"),
    ("columns", "columns (vertical stripes)"),
    ("checker", "checker (grid)"),
)


def _gauss_kernel(sigma, device):
    radius = max(1, int(math.ceil(3.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-0.5 * (xs / sigma) ** 2)
    return k / k.sum()


def _smooth(x, sigma):
    """Separable Gaussian smoothing of x: (N, C, H, W), replicate padding."""
    if sigma <= 0:
        return x
    k = _gauss_kernel(sigma, x.device).to(x.dtype)
    r = (k.numel() - 1) // 2
    c = x.shape[1]
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    y = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), ky, groups=c)
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="replicate"), kx, groups=c)
    return y


def _alternating(n, device, dtype):
    return 1.0 - 2.0 * (torch.arange(n, device=device) % 2).to(dtype)


def _carriers(h, w, device, dtype):
    sy = _alternating(h, device, dtype).view(1, 1, h, 1)
    sx = _alternating(w, device, dtype).view(1, 1, 1, w)
    return {"rows": sy, "columns": sx, "checker": sy * sx}, sx


def _median_all(t):
    return t.reshape(-1).median()


def _strength(x, name, sigma, sx):
    """How far the local Nyquist amplitude stands out of the same measurement
    at neighbouring frequencies (~1 for noise / no pattern)."""
    n, c, h, w = x.shape
    car, _ = _carriers(h, w, x.device, x.dtype)
    a = _smooth(x * car[name], sigma).abs()
    if name == "columns":
        base = x
        coord = torch.arange(w, device=x.device, dtype=torch.float64).view(1, 1, 1, w)
    else:
        base = x * sx if name == "checker" else x
        coord = torch.arange(h, device=x.device, dtype=torch.float64).view(1, 1, h, 1)
    probes = []
    for d in _PROBE_OFFSETS:
        ph = (2.0 * math.pi * (0.5 - d)) * coord
        re = _smooth(base * torch.cos(ph).to(x.dtype), sigma)
        im = _smooth(base * torch.sin(ph).to(x.dtype), sigma)
        probes.append(torch.sqrt(re * re + im * im))
    pm = _median_all(torch.stack(probes))
    return float(_median_all(a) / (pm + 1e-12) / _NOISE_NORM)


def _component(x, name, sigma, protect, max_amp, car):
    """Smoothed, signed Nyquist amplitude (N, C, H, W) of one carrier."""
    r = x * car[name]
    if protect > 0:
        a0 = _smooth(r, _PROTECT_SIGMA).abs()
        med = a0.flatten(2).median(dim=2).values.view(x.shape[0], x.shape[1], 1, 1)
        wgt = torch.exp(-(a0 / (protect * 1.4826 * med + 1e-9)) ** 4)
        r = r * wgt
    a = _smooth(r, sigma)
    if max_amp > 0:
        lim = torch.full_like(a[:, :, :1, :1], float(max_amp))
    else:
        lim = _AUTO_LIMIT_MULT * a.abs().flatten(2).median(dim=2).values.view(
            x.shape[0], x.shape[1], 1, 1)
    return torch.maximum(torch.minimum(a, lim), -lim)


def _process_one(x, use, strength, sigma, min_strength, protect, max_amp,
                 measure_only):
    """x: (1, C, H, W) float32. Returns (cleaned, lines)."""
    _, c, h, w = x.shape
    car, sx = _carriers(h, w, x.device, x.dtype)
    std = float(x.std()) + 1e-12
    corr = torch.zeros_like(x)
    lines = []
    for name, label in _COMPONENTS:
        if not use[name]:
            continue
        st = _strength(x, name, sigma, sx)
        lock = float(torch.sqrt(((x * car[name]).mean(dim=(0, 2, 3)) ** 2).mean()))
        if st < min_strength:
            lines.append("%s: strength %.1fx - below %.1fx, left untouched"
                         % (label, st, min_strength))
            continue
        a = _component(x, name, sigma, protect, max_amp, car)
        env = float(torch.sqrt((a * a).mean()))
        part = a * car[name]
        txt = ("%s: strength %.1fx, parity-locked %.4f, local envelope %.4f rms "
               "(%.2f%% of latent std)" % (label, st, lock, env, 100.0 * env / std))
        if not measure_only:
            corr = corr + float(strength) * part
            txt += " - removed"
        else:
            txt += " - measured only"
        lines.append(txt)
    return x - corr, lines


class MykeeLatentNyquistNotch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "enabled": ("BOOLEAN", {"default": True}),
                "mode": (["remove", "measure_only"], {"default": "remove"}),
                "rows": ("BOOLEAN", {"default": True}),
                "columns": ("BOOLEAN", {"default": True}),
                "checker": ("BOOLEAN", {"default": True}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                                       "step": 0.01}),
                "smoothing": ("FLOAT", {"default": 8.0, "min": 2.0, "max": 32.0,
                                        "step": 0.5}),
                "min_strength": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 50.0,
                                           "step": 0.5}),
                "protect_structure": ("FLOAT", {"default": 4.0, "min": 0.0,
                                                "max": 10.0, "step": 0.5}),
                "max_amplitude": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0,
                                            "step": 0.001}),
                "log_to_console": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "STRING")
    RETURN_NAMES = ("latent", "removed", "status")
    FUNCTION = "run"
    CATEGORY = "Mykee/Latent"

    def run(self, latent, enabled, mode, rows, columns, checker, strength,
            smoothing, min_strength, protect_structure, max_amplitude,
            log_to_console):
        samples = latent["samples"]
        stamp = time.strftime("%H:%M:%S")
        if not enabled:
            return (latent, self._zeros(latent, samples),
                    "[Mykee Latent Nyquist Notch] %s  disabled" % stamp)
        if samples.dim() not in (4, 5):
            return (latent, self._zeros(latent, samples),
                    "[Mykee Latent Nyquist Notch] %s  unsupported latent shape %s"
                    % (stamp, tuple(samples.shape)))

        five = samples.dim() == 5
        x_all = samples
        if five:                                   # (B, C, T, H, W) -> (B*T, C, H, W)
            b5, c5, t5, h5, w5 = samples.shape
            x_all = samples.permute(0, 2, 1, 3, 4).reshape(b5 * t5, c5, h5, w5)
        n, c, h, w = x_all.shape
        header = "[Mykee Latent Nyquist Notch] %s  latent %dx%dx%d, batch %d" % (
            stamp, c, h, w, n)
        if h < _MIN_SIZE or w < _MIN_SIZE:
            msg = header + "\nlatent smaller than %d px - passed through" % _MIN_SIZE
            return (latent, self._zeros(latent, samples), msg)

        use = {"rows": bool(rows), "columns": bool(columns), "checker": bool(checker)}
        orig_dtype = x_all.dtype
        out = torch.empty_like(x_all)
        text = [header]
        for i in range(n):
            xi = x_all[i:i + 1].float()
            ci, lines = _process_one(
                xi, use, strength, float(smoothing), float(min_strength),
                float(protect_structure), float(max_amplitude),
                mode == "measure_only")
            out[i:i + 1] = ci.to(orig_dtype)
            if n > 1:
                text.append("Image %d/%d:" % (i + 1, n))
            text.extend(lines)

        removed = x_all - out
        if five:
            out = out.reshape(b5, t5, c5, h5, w5).permute(0, 2, 1, 3, 4).contiguous()
            removed = removed.reshape(b5, t5, c5, h5, w5).permute(
                0, 2, 1, 3, 4).contiguous()
        status = "\n".join(text)
        if log_to_console:
            print(status)

        new_latent = latent.copy()
        new_latent["samples"] = out
        removed_latent = latent.copy()
        removed_latent["samples"] = removed
        return (new_latent, removed_latent, status)

    @staticmethod
    def _zeros(latent, samples):
        z = latent.copy()
        z["samples"] = torch.zeros_like(samples)
        return z


NODE_CLASS_MAPPINGS = {
    "MykeeLatentNyquistNotch": MykeeLatentNyquistNotch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeLatentNyquistNotch": "Mykee Latent Nyquist Notch",
}
