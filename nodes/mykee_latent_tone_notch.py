"""Mykee Latent Tone Notch.

Removes a faint, fixed-frequency stripe pattern ("tone") from a LATENT, before
the VAE decode. It is the general version of the Latent Nyquist Notch: that node
only knows the single frequency of 2 latent pixels per period, this one finds
(or is told) any frequency between 0.02 and 0.5 cycles per latent pixel, along
the width (vertical stripes) and/or the height (horizontal stripes).

Typical cases:
  * patch-grid stripes of DiT models (2 latent px = Nyquist, 0.5 cycles/px),
  * a fixed number of stripes across the image width, for example the 48
    chroma stripes of a fine-tuned model: the period grows with the resolution
    (48 cycles over a 128 px wide latent = 0.375 cycles/px), so no fixed pixel
    period describes it.

Method (per image of the batch):
  1. Auto mode: for every latent channel a power spectrum is taken along the
     axis (the latent is first averaged over a few bands of the other axis, so
     stripes that run across the whole picture add up while the noise does not).
     Each channel's spectrum is divided by its own local median (the "floor"
     around that frequency) and the channels are averaged. A sharp peak that
     stands `min_prominence` times above that floor is a stripe frequency. The
     frequency is refined between the spectrum bins and, if it is within 0.2 of
     a whole number of cycles across the picture, snapped to it.
  2. The stripe itself is a local, weighted least-squares fit of
         a(x, y) * cos(2*pi*f*t) + b(x, y) * sin(2*pi*f*t)
     with a Gaussian window (`smoothing`). Amplitude and phase may drift slowly
     over the picture. Cells whose amplitude is far above the typical one are
     real fine structure and get ~0 weight, so they are neither removed nor
     leak into the neighbourhood. The amplitude is limited (auto: 6 x median).
  3. The fitted stripe is subtracted. Up to `max_components` frequencies per
     axis are removed one after another (the spectrum is measured again on
     what is left).
With nothing above the threshold the latent passes through untouched.
"""

import math
import time

import torch
import torch.nn.functional as F

_PROTECT_SIGMA = 1.5        # latent px, small smoothing used to find real structure
_AUTO_LIMIT_MULT = 6.0      # amplitude limit = this x median amplitude
_MIN_SIZE = 32              # smaller latents are passed through
_PAD = 4                    # zero padding factor of the spectrum
_FLOOR_HALF = 12            # half width (in unpadded bins) of the spectrum floor window
_SNAP_TOL = 0.2             # snap to whole cycles across the picture within this
_EXCLUDE_BINS = 3           # unpadded bins around a handled frequency are skipped
_MAX_BANDS = 4              # bands of the other axis used for the spectrum

_AXES = (
    ("columns", "columns (vertical stripes)"),
    ("rows", "rows (horizontal stripes)"),
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _gauss_kernel(sigma, device):
    radius = max(1, int(math.ceil(3.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-0.5 * (xs / sigma) ** 2)
    return k / k.sum()


def _smooth(x, sigma, zero_pad=False):
    """Separable Gaussian smoothing of x: (N, C, H, W).

    Replicate padding by default. zero_pad=True is for the weighted fit: the
    numerators and the weight sums are smoothed the same way, so the ratio is
    exact at the borders too."""
    if sigma <= 0:
        return x
    k = _gauss_kernel(sigma, x.device).to(x.dtype)
    r = (k.numel() - 1) // 2
    c = x.shape[1]
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    mode = "constant" if zero_pad else "replicate"
    y = F.conv2d(F.pad(x, (0, 0, r, r), mode=mode), ky, groups=c)
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode=mode), kx, groups=c)
    return y


def _carrier_maps(h, w, axis, f, device):
    """cos / sin of 2*pi*f*t along the axis, shaped to broadcast over (1, C, H, W)."""
    n = w if axis == "columns" else h
    t = torch.arange(n, device=device, dtype=torch.float64)
    ph = 2.0 * math.pi * f * t
    cs, sn = torch.cos(ph).float(), torch.sin(ph).float()
    shape = (1, 1, 1, n) if axis == "columns" else (1, 1, n, 1)
    return cs.view(shape), sn.view(shape)


def _length(h, w, axis):
    return w if axis == "columns" else h


# ---------------------------------------------------------------------------
# spectrum / peak search
# ---------------------------------------------------------------------------

def _ratio_spectrum(x, axis):
    """x: (C, H, W). Returns (R, nfft): R[k] is the power at frequency
    k / nfft cycles per px divided by the local floor around it, averaged over
    the channels (about 1 where there is only noise)."""
    c, h, w = x.shape
    y = x if axis == "columns" else x.transpose(1, 2)        # (C, other, L)
    length = y.shape[2]
    other = y.shape[1]
    bands = max(1, min(_MAX_BANDS, other // 16))
    means = torch.stack([b.mean(dim=1) for b in y.chunk(bands, dim=1)], dim=1)
    means = means - means.mean(dim=-1, keepdim=True)          # (C, bands, L)
    win = torch.hann_window(length, periodic=False, device=x.device,
                            dtype=x.dtype)
    nfft = _PAD * length
    spec = torch.fft.rfft(means * win, n=nfft)                # (C, bands, F)
    p = (spec.real ** 2 + spec.imag ** 2).mean(dim=1)         # (C, F)
    nbins = p.shape[1]
    half = max(2, min(_PAD * _FLOOR_HALF, nbins // 2 - 1))
    padded = F.pad(p.unsqueeze(0), (half, half), mode="reflect")[0]
    floor = padded.unfold(-1, 2 * half + 1, 1).median(dim=-1).values
    ratio = p / (floor + 1e-20)
    return ratio.mean(dim=0), nfft


def _find_peak(r, nfft, fmin, skip):
    """Strongest peak of R in [fmin, 0.5], skipping the frequencies in `skip`.
    Returns (frequency, power ratio) or None."""
    lo = max(1, int(math.ceil(fmin * nfft)))
    hi = nfft // 2
    if hi <= lo:
        return None
    work = r.clone()
    work[:lo] = 0.0
    wide = _EXCLUDE_BINS * _PAD
    for fs in skip:
        k0 = int(round(fs * nfft))
        work[max(0, k0 - wide):k0 + wide + 1] = 0.0
    k = int(torch.argmax(work))
    if work[k] <= 0:
        return None
    delta = 0.0
    if 0 < k < r.numel() - 1:
        a, b, c = float(r[k - 1]), float(r[k]), float(r[k + 1])
        if a > 0 and b > 0 and c > 0:
            la, lb, lc = math.log(a), math.log(b), math.log(c)
            den = la - 2.0 * lb + lc
            if den < 0:
                delta = max(-0.5, min(0.5, 0.5 * (la - lc) / den))
    f = min(0.5, (k + delta) / nfft)
    return f, float(work[k])


# ---------------------------------------------------------------------------
# the stripe fit
# ---------------------------------------------------------------------------

def _fit_tone(x, axis, f, sigma, protect, max_amp):
    """x: (1, C, H, W). Returns the fitted stripe (1, C, H, W)."""
    _, c, h, w = x.shape
    cs, sn = _carrier_maps(h, w, axis, f, x.device)
    if protect > 0:
        a0 = _smooth(x * cs, _PROTECT_SIGMA)
        b0 = _smooth(x * sn, _PROTECT_SIGMA)
        m = torch.sqrt(a0 * a0 + b0 * b0)
        med = m.flatten(2).median(dim=2).values.view(1, c, 1, 1)
        wgt = torch.exp(-(m / (protect * 1.4826 * med + 1e-9)) ** 4)
    else:
        wgt = torch.ones_like(x)
    wx = wgt * x
    nc = _smooth(wx * cs, sigma, True)
    ns = _smooth(wx * sn, sigma, True)
    mcc = _smooth(wgt * cs * cs, sigma, True)
    mss = _smooth(wgt * sn * sn, sigma, True)
    mcs = _smooth(wgt * cs * sn, sigma, True)
    rho = 1e-3 * (mcc + mss) + 1e-8          # ridge: keeps the fit stable near Nyquist
    det = (mcc + rho) * (mss + rho) - mcs * mcs
    alpha = ((mss + rho) * nc - mcs * ns) / det
    beta = ((mcc + rho) * ns - mcs * nc) / det
    amp = torch.sqrt(alpha * alpha + beta * beta)
    if max_amp > 0:
        lim = torch.full_like(amp[:, :, :1, :1], float(max_amp))
    else:
        lim = _AUTO_LIMIT_MULT * amp.flatten(2).median(dim=2).values.view(1, c, 1, 1)
    scale = torch.clamp(lim / (amp + 1e-12), max=1.0)
    return (alpha * cs + beta * sn) * scale


def _snap(f, length):
    cyc = f * length
    near = round(cyc)
    if near >= 1 and abs(cyc - near) <= _SNAP_TOL:
        return min(0.5, near / length)
    return f


def _manual_frequency(value, unit, length):
    if unit == "period in latent px":
        f = 1.0 / max(value, 1e-6)
    elif unit == "cycles across the picture":
        f = value / float(length)
    else:
        f = value
    return max(0.0, min(0.5, f))


def _describe(f, length):
    return ("f=%.4f cyc/px (period %.3f latent px, %.2f cycles across %d px)"
            % (f, 1.0 / max(f, 1e-9), f * length, length))


def _process_one(x, use, strength, sigma, min_prom, protect, max_amp,
                 measure_only, max_components, fmin, auto, manual_value,
                 manual_unit, snap):
    """x: (1, C, H, W) float32. Returns (cleaned, lines)."""
    _, c, h, w = x.shape
    std = float(x.std()) + 1e-12
    res = x.clone()
    lines = []
    for axis, label in _AXES:
        if not use[axis]:
            continue
        length = _length(h, w, axis)
        if not auto:
            f = _manual_frequency(manual_value, manual_unit, length)
            if f < 0.01:
                lines.append("%s: manual frequency out of range - skipped" % label)
                continue
            stripe = _fit_tone(res, axis, f, sigma, protect, max_amp)
            env = float(torch.sqrt((stripe * stripe).mean()))
            txt = "%s: %s, manual, envelope %.4f rms (%.2f%% of latent std)" % (
                label, _describe(f, length), env, 100.0 * env / std)
            if measure_only:
                txt += " - measured only"
            else:
                res = res - float(strength) * stripe
                txt += " - removed"
            lines.append(txt)
            continue

        handled = []
        found = False
        shown = max_components + (2 if measure_only else 0)
        for _ in range(shown):
            ratio, nfft = _ratio_spectrum(res[0], axis)
            peak = _find_peak(ratio, nfft, fmin, handled)
            if peak is None:
                break
            found = True
            f, pr = peak
            prom = math.sqrt(max(pr, 0.0))
            if snap:
                f = _snap(f, length)
            above = prom >= min_prom
            handled.append(f)
            if not above and not measure_only:
                lines.append("%s: best candidate %s, prominence %.1fx - below "
                             "%.1fx, left untouched"
                             % (label, _describe(f, length), prom, min_prom))
                break
            stripe = _fit_tone(res, axis, f, sigma, protect, max_amp)
            env = float(torch.sqrt((stripe * stripe).mean()))
            txt = ("%s: %s, prominence %.1fx, envelope %.4f rms (%.2f%% of "
                   "latent std)" % (label, _describe(f, length), prom, env,
                                    100.0 * env / std))
            if measure_only:
                txt += (" - measured only%s"
                        % ("" if above else " (below threshold)"))
            else:
                res = res - float(strength) * stripe
                txt += " - removed"
            lines.append(txt)
        if not found:
            lines.append("%s: no spectral peak found" % label)
    return res, lines


# ---------------------------------------------------------------------------
# node
# ---------------------------------------------------------------------------

class MykeeLatentToneNotch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "enabled": ("BOOLEAN", {"default": True}),
                "mode": (["remove", "measure_only"], {"default": "remove"}),
                "columns": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Look for vertical stripes (the pattern changes "
                               "along the width)."}),
                "rows": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Look for horizontal stripes (the pattern "
                               "changes along the height)."}),
                "frequency_mode": (["auto", "manual"], {
                    "default": "auto",
                    "tooltip": "auto: the stripe frequencies are found in the "
                               "latent's spectrum. manual: use the value below."}),
                "min_prominence": ("FLOAT", {
                    "default": 2.5, "min": 1.0, "max": 50.0, "step": 0.1,
                    "tooltip": "Auto mode: a spectral peak is removed only when "
                               "it stands this many times above the spectrum "
                               "around it (about 1 for noise)."}),
                "max_components": ("INT", {
                    "default": 2, "min": 1, "max": 4,
                    "tooltip": "Auto mode: how many different frequencies are "
                               "removed per axis, one after another."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                                       "step": 0.01}),
                "smoothing": ("FLOAT", {
                    "default": 8.0, "min": 2.0, "max": 32.0, "step": 0.5,
                    "tooltip": "Latent pixels over which the stripe amplitude "
                               "and phase are fitted. Larger = narrower band, "
                               "follows a drifting pattern less closely."}),
                "protect_structure": ("FLOAT", {
                    "default": 4.0, "min": 0.0, "max": 10.0, "step": 0.5,
                    "tooltip": "Cells whose amplitude is this many times above "
                               "the typical one are treated as real fine "
                               "structure and excluded; 0 = off."}),
                "max_amplitude": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Largest correction per latent value; 0 = "
                               "automatic (6 x the median amplitude)."}),
                "min_frequency": ("FLOAT", {
                    "default": 0.06, "min": 0.02, "max": 0.45, "step": 0.005,
                    "tooltip": "Auto mode: lowest frequency searched, in cycles "
                               "per latent pixel. Lower frequencies mix with "
                               "real image content."}),
                "snap_to_whole_cycles": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto mode: when the frequency is within 0.2 of "
                               "a whole number of cycles across the picture "
                               "(like 48 stripes over the width), use exactly "
                               "that number."}),
                "manual_value": ("FLOAT", {
                    "default": 0.5, "min": 0.01, "max": 512.0, "step": 0.001,
                    "tooltip": "Manual mode: the stripe frequency, in the unit "
                               "chosen below."}),
                "manual_unit": ([
                    "cycles per latent px",
                    "period in latent px",
                    "cycles across the picture",
                ], {"default": "cycles per latent px"}),
                "log_to_console": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "STRING")
    RETURN_NAMES = ("latent", "removed", "status")
    FUNCTION = "run"
    CATEGORY = "Mykee/Latent"

    def run(self, latent, enabled, mode, columns, rows, frequency_mode,
            min_prominence, max_components, strength, smoothing,
            protect_structure, max_amplitude, min_frequency,
            snap_to_whole_cycles, manual_value, manual_unit, log_to_console):
        samples = latent["samples"]
        stamp = time.strftime("%H:%M:%S")
        if not enabled:
            return (latent, self._zeros(latent, samples),
                    "[Mykee Latent Tone Notch] %s  disabled" % stamp)
        if samples.dim() not in (4, 5):
            return (latent, self._zeros(latent, samples),
                    "[Mykee Latent Tone Notch] %s  unsupported latent shape %s"
                    % (stamp, tuple(samples.shape)))

        five = samples.dim() == 5
        x_all = samples
        if five:                                   # (B, C, T, H, W) -> (B*T, C, H, W)
            b5, c5, t5, h5, w5 = samples.shape
            x_all = samples.permute(0, 2, 1, 3, 4).reshape(b5 * t5, c5, h5, w5)
        n, c, h, w = x_all.shape
        header = "[Mykee Latent Tone Notch] %s  latent %dx%dx%d, batch %d" % (
            stamp, c, h, w, n)
        if h < _MIN_SIZE or w < _MIN_SIZE:
            msg = header + "\nlatent smaller than %d px - passed through" % _MIN_SIZE
            return (latent, self._zeros(latent, samples), msg)

        use = {"columns": bool(columns), "rows": bool(rows)}
        orig_dtype = x_all.dtype
        out = torch.empty_like(x_all)
        text = [header]
        for i in range(n):
            xi = x_all[i:i + 1].float()
            ci, lines = _process_one(
                xi, use, strength, float(smoothing), float(min_prominence),
                float(protect_structure), float(max_amplitude),
                mode == "measure_only", int(max_components),
                float(min_frequency), frequency_mode == "auto",
                float(manual_value), manual_unit, bool(snap_to_whole_cycles))
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
    "MykeeLatentToneNotch": MykeeLatentToneNotch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeLatentToneNotch": "Mykee Latent Tone Notch",
}
