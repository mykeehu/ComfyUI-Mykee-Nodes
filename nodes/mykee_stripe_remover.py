"""
ComfyUI-Mykee-Nodes / stripe remover

MykeeStripeRemover:
    Removes faint, periodic horizontal and/or vertical stripes from an IMAGE.

    Background: DiT image models (Chroma / Flux family, and others) work on
    "patches" - 8x VAE downsampling x 2x2 patches = a 16 px grid in image
    space. When such a model is run far above its training resolution, a
    faint banding with exactly that period (16 px, plus its harmonics 8 px,
    5.33 px ...) can appear, most visible in smooth areas such as sky.
    This is NOT a VAE artifact, so VAE-specific filters (for example the 2 px
    grid removers made for the Qwen / Wan VAEs) do not touch it.

    The node measures the stripes from the input image itself - nothing has
    to be tuned:

      1. Flat areas (sky, walls, smooth skin) are found automatically; only
         those are used for measuring, so real detail can not be mistaken
         for stripes.
      2. The row (or column) signal of the flat areas is searched for sharp
         periodic peaks. The period is refined to sub-pixel accuracy, so it
         also works on images that were resized after generation.
      3. The period is refined to a fraction of a pixel by a coherent fold
         (and snapped to an integer when it is that close - patch and VAE
         grids are integer periods). A 0.1 px period error at 16 px drifts
         ~13 px over 2048 rows and cancels the measurement completely.
      4. The stripe profile (one period, including all harmonics) is folded
         out of the image with a matched high-pass filter, separately for
         every colour channel and for a grid of zones (x and y): the strength
         changes across the picture and the phase of the stripes wanders
         slowly from top to bottom.
      5. The profile is subtracted. Because it is a tiny periodic pattern,
         no blur is applied to the image at all.

    Up to `max_components` different periods are removed one after another
    (for example 16 px first, then a weaker 32 px leftover).

Outputs: the cleaned image, a preview of exactly what was removed
(amplified around grey) and a status text.
"""

import math
import time

import torch
import torch.nn.functional as F

try:
    import comfy.model_management as _mm
except ImportError:  # running outside ComfyUI (tests)
    _mm = None


# ---------------------------------------------------------------------------
# run log
# ---------------------------------------------------------------------------
# The log of every run goes to the ComfyUI console (log_to_console, off by
# default) and to the `status` output. There is no text box on the node: the
# browser side kept losing runs, and for most users the log is just noise.
# Images are numbered only inside a batch ("Image 1/4:"), and that numbering
# starts again with every run - there is no running counter.


def _publish(image, label, status, to_console, node_id=None):
    """Log one run. Returns the status text with a one-line header."""
    h, w = int(image.shape[1]), int(image.shape[2])
    head = f"{time.strftime('%H:%M:%S')}  {w}x{h}"
    if node_id is not None:
        head += f", node {node_id}"
    if label and str(label).strip():
        head += f", {str(label).strip()}"
    full = head + "\n" + status
    if to_console:
        print("[Mykee Stripe Remover] " + full, flush=True)
    return full


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _work_device():
    """GPU if ComfyUI has one, else CPU."""
    if _mm is not None:
        try:
            return _mm.get_torch_device()
        except Exception:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _conv_y(x, kernel):
    """Convolve (C, H, W) along H with a 1D kernel, reflect padding."""
    c, h, w = x.shape
    k = kernel.numel()
    pad = k // 2
    # (C*W, 1, H): every column becomes one 1D signal
    t = x.permute(0, 2, 1).reshape(c * w, 1, h)
    if pad >= h:  # tiny image safety
        pad = h - 1
        kernel = kernel[(k // 2 - pad):(k // 2 + pad + 1)]
        kernel = kernel / kernel.sum()
    t = F.pad(t, (pad, pad), mode="reflect")
    t = F.conv1d(t, kernel.view(1, 1, -1).to(t.dtype))
    return t.reshape(c, w, h).permute(0, 2, 1).contiguous()


def _gauss_kernel(sigma, device):
    radius = max(1, int(math.ceil(sigma * 3.0)))
    xs = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (xs / sigma) ** 2)
    return k / k.sum()


def _box_kernel(period, device):
    """Box of width `period` px (real valued) as a sampled kernel.

    For an integer period this is [0.5, 1, ..., 1, 0.5] / P, which has an
    exact zero at every multiple of 1/P - so `x - box(x)` passes a P-periodic
    pattern with gain exactly 1 and suppresses slow content quadratically.
    """
    half = period / 2.0
    radius = int(math.ceil(half + 0.5))
    xs = torch.arange(-radius, radius + 1, device=device, dtype=torch.float64)
    lo = torch.clamp(xs - 0.5, min=-half)
    hi = torch.clamp(xs + 0.5, max=half)
    k = torch.clamp(hi - lo, min=0.0)
    k = k / k.sum()
    return k.to(torch.float32)


def _box_mean_2d(x, win):
    """(H, W) -> (H, W) local mean over a win x win window (reflect pad)."""
    t = x[None, None]
    pad = win // 2
    t = F.pad(t, (pad, pad, pad, pad), mode="reflect")
    t = F.avg_pool2d(t, win, stride=1)
    return t[0, 0]


def _hat_weights(width, n_blocks, device):
    """(n_blocks, width) linear 'hat' functions that sum to 1 along axis 0."""
    if n_blocks == 1:
        return torch.ones(1, width, device=device)
    centers = torch.linspace(0.0, width - 1.0, n_blocks, device=device)
    spacing = centers[1] - centers[0]
    xs = torch.arange(width, device=device, dtype=torch.float32)
    hats = torch.clamp(1.0 - (xs[None, :] - centers[:, None]).abs() / spacing, min=0.0)
    return hats / hats.sum(dim=0, keepdim=True).clamp(min=1e-8)


def _is_8bit_source(t):
    """True if an (H, W, C) float image in [0, 1] only holds k/255 values.

    Such an image comes from an 8-bit file (Load Image) and not straight
    from a VAE decode.
    """
    v = t.reshape(-1)
    v = v[:: max(1, v.numel() // 500000)] * 255.0
    frac = (v - v.round()).abs()
    return float((frac < 2e-3).float().mean()) > 0.98


# ---------------------------------------------------------------------------
# core algorithm (works on "horizontal stripes": pattern varies along H)
# ---------------------------------------------------------------------------

def _flat_mask(lum, flat_percent):
    """Boolean (H, W) mask of the smoothest `flat_percent` % of the picture.

    Smoothness is the local standard deviation of the luminance after a
    vertical high-pass, so slow sky gradients and the stripes themselves do
    not count as texture.
    """
    h, w = lum.shape
    hp = lum - _conv_y(lum[None], _gauss_kernel(6.0, lum.device))[0]
    m1 = _box_mean_2d(hp, 9)
    m2 = _box_mean_2d(hp * hp, 9)
    std = (m2 - m1 * m1).clamp(min=0).sqrt()
    # percentile on a subsample (torch.quantile has an element limit)
    sample = std.reshape(-1)
    if sample.numel() > 1_000_000:
        sample = sample[:: sample.numel() // 1_000_000 + 1]
    thr = torch.quantile(sample, flat_percent / 100.0)
    return std <= thr, float(thr)


def _row_signals(d, mask, hats, clip):
    """Weighted per-zone row means of the high-passed image.

    d: (C, H, W), mask: (H, W) float, hats: (Z, W).
    Returns s (Z, C, H) and wts (Z, H): mask weight per row and zone.
    """
    dc = d.clamp(-clip, clip) * mask[None]
    num = torch.einsum("chw,zw->zch", dc, hats)
    wts = torch.einsum("hw,zw->zh", mask, hats)
    s = num / wts[:, None, :].clamp(min=1e-6)
    return s, wts


# Periods the generators are known to produce: the VAE decoder grid (2, 4, 8 px)
# and the DiT patch grid (16 px) with their harmonics, i.e. 32/k px for
# k = 2..16. 32 px itself is NOT included: it sits too close to slow image
# content (a 31.75 / 32.38 px "peak" showed up in a real picture).
_GRID_PERIODS = tuple(32.0 / k for k in range(2, 17))
_GRID_THRESHOLD_FACTOR = 0.4   # these periods need only 40% of the threshold


def _near_grid(period):
    return any(abs(period - g) < 0.02 * g for g in _GRID_PERIODS)


def _snap_to_grid(period):
    """Nearest known grid period within 2%, else None."""
    best = min(_GRID_PERIODS, key=lambda g: abs(period - g))
    return best if abs(period - best) < 0.02 * best else None


def _detect_period(s_global, w_global, min_period, max_period, threshold,
                   exclude=(), grid_only=False,
                   grid_factor=_GRID_THRESHOLD_FACTOR):
    """Find the strongest sharp periodic peak in the row signal.

    s_global: (C, H) weighted row mean, w_global: (H,) row weight in [0, 1].
    Peaks within 3% of a period in `exclude` are ignored (periods that were
    already removed or rejected). A peak at a known grid period (see
    _GRID_PERIODS) only has to reach 40% of `threshold`: statistically such a
    peak is very unlikely by chance, and the shutter test showed a real
    16 px stripe at strength 37 that the plain threshold of 40 missed.
    Returns (period, prominence) or (None, best_prominence).
    """
    c, h = s_global.shape
    s = s_global.double()
    w = w_global.double()
    # remove the slow trend of the row signal (weighted), keep short periods
    k = _gauss_kernel(max(4.0, max_period / 2.0), s.device).double()
    trend_num = _conv_y((s * w[None])[:, :, None].float(), k.float())[:, :, 0].double()
    trend_den = _conv_y(w[None, :, None].float(), k.float())[0, :, 0].double()
    s = s - trend_num / trend_den[None].clamp(min=1e-6)
    s = s * w[None]

    n_pad = 1
    while n_pad < 8 * h:
        n_pad *= 2
    win = torch.hann_window(h, periodic=False, device=s.device, dtype=torch.float64)
    spec = torch.fft.rfft(s * win[None], n=n_pad, dim=1)
    energy = (spec.abs() ** 2).sum(dim=0)  # (n_pad/2+1,)
    freqs = torch.arange(energy.numel(), device=s.device, dtype=torch.float64) / n_pad

    f_lo, f_hi = 1.0 / max_period, min(0.5, 1.0 / min_period)
    in_band = (freqs >= f_lo) & (freqs <= f_hi)
    if in_band.sum() < 10:
        return None, 0.0

    # candidate peaks: local maxima inside the band
    e = energy.clone()
    e[~in_band] = 0
    is_peak = (e[1:-1] > e[:-2]) & (e[1:-1] >= e[2:])
    idx = torch.nonzero(is_peak).flatten() + 1
    if idx.numel() == 0:
        return None, 0.0
    order = torch.argsort(e[idx], descending=True)[:12]
    idx = idx[order]

    best_score, best_prom, best_idx, seen_prom = 0.0, 0.0, None, 0.0
    main_lobe = 16  # padded bins (hann main lobe at 8x zero padding)
    for i in idx.tolist():
        per_i = n_pad / float(i)
        if any(abs(per_i - p) < 0.03 * p for p in exclude):
            continue
        lo = max(1, int(i * 0.6))
        hi = min(energy.numel(), int(i * 1.6) + 2)
        seg = torch.cat([energy[lo:max(lo, i - main_lobe)], energy[i + main_lobe:hi]])
        if seg.numel() < 20:
            continue
        bg = float(seg.median())
        prom = float(energy[i]) / max(bg, 1e-30)
        seen_prom = max(seen_prom, prom)
        on_grid = _near_grid(per_i)
        if grid_only and not on_grid:
            continue
        eff = threshold * (grid_factor if on_grid else 1.0)
        score = prom / max(eff, 1e-9)
        if score >= 1.0 and score > best_score:
            best_score, best_prom, best_idx = score, prom, i
    if best_idx is None:
        return None, seen_prom

    # parabolic refinement of the peak position
    i = best_idx
    a, b, c_ = float(energy[i - 1]), float(energy[i]), float(energy[i + 1])
    denom = a - 2 * b + c_
    shift = 0.5 * (a - c_) / denom if abs(denom) > 1e-30 else 0.0
    f_peak = (i + max(-1.0, min(1.0, shift))) / n_pad
    return 1.0 / f_peak, best_prom


def _detect_tiles(d0, mask, clip, min_period, max_period, threshold,
                  seg=512, hop=256, min_cov=0.25, exclude=(),
                  grid_only=False, grid_factor=_GRID_THRESHOLD_FACTOR):
    """Search the stripe period in tiles (x-zones x row segments) of the flat
    area and keep the sharpest peak of any tile.

    A single row signal over the whole picture is diluted by everything else
    that is "flat" (sea ripples, plaster, ...): on a test image with a large
    sea area the 16 px stripes of the sky had a local peak strength of 381
    but only ~10 in the whole-image signal. In tiles the clean sky wins.

    Returns (period, strength, s_zone, w_zone): s_zone (C, H) / w_zone (H)
    are the full-height row signal and weights (flat fraction squared) of
    the winning x-zone, for the period refinement. period is None when no
    tile reaches `threshold`; strength is then the best value seen.

    A tile needs `min_cov` (25%) of flat pixels. The flat mask is a percentile
    (the smoothest 35% of the picture), so an all-texture picture has ~35% in
    every tile; with the old 50% rule such a picture was skipped completely
    ("no flat area") although it carried a coherent 4 px stripe of 2.3 levels.
    The stripes are phase-locked to the pixel grid, the texture is not, so the
    average over many texture pixels still shows them.
    """
    c, h, w = d0.shape
    dev = d0.device
    nz = int(max(1, min(6, w // 256)))
    edges = [int(round(i * w / nz)) for i in range(nz + 1)]
    hz = torch.zeros(nz, w, device=dev)
    for i in range(nz):
        hz[i, edges[i]:edges[i + 1]] = 1.0
    s_z, w_z = _row_signals(d0, mask, hz, clip)
    best_period, best_prom, best_zone = None, 0.0, None
    seen_max = 0.0
    scales = [min(seg, h)]
    if h >= 1536:
        scales.append(min(2 * seg, h))     # longer segments: more rows to average
    for z in range(nz):
        cov_row = (w_z[z] / float(max(1, edges[z + 1] - edges[z]))).clamp(0, 1)
        for sg in scales:
            for y0 in range(0, h - sg + 1, max(1, sg // 2)):
                if float(cov_row[y0:y0 + sg].mean()) < min_cov:
                    continue
                period, prom = _detect_period(
                    s_z[z][:, y0:y0 + sg], cov_row[y0:y0 + sg],
                    min_period, max_period, threshold, exclude,
                    grid_only, grid_factor)
                seen_max = max(seen_max, prom)
                if period is not None and prom > best_prom:
                    best_period, best_prom, best_zone = period, prom, z
    if best_period is None:
        return None, seen_max, None, None
    cov_row = (w_z[best_zone] / float(max(
        1, edges[best_zone + 1] - edges[best_zone]))).clamp(0, 1)
    return best_period, best_prom, s_z[best_zone], cov_row ** 2


def _fold_score(s, w, period):
    """Explained variance of the row signal when folded at `period`.

    s: (C, H), w: (H,). A coherent fold only scores high when the period is
    right to a fraction of a pixel over the WHOLE height (a 0.1 px error at
    a 16 px period drifts by ~13 px over 2048 rows and scores ~0).
    """
    c, h = s.shape
    nb = max(4, int(round(period)) if abs(period - round(period)) < 1e-6
             else int(round(period * 2)))
    ys = torch.arange(h, device=s.device, dtype=torch.float64)
    u = ((ys % period) / period * nb).float()
    i0 = torch.floor(u).long() % nb
    fr = u - torch.floor(u)
    i1 = (i0 + 1) % nb
    num = torch.zeros(c, nb, device=s.device)
    den = torch.zeros(nb, device=s.device)
    num.index_add_(1, i0, s * (w * (1 - fr))[None])
    num.index_add_(1, i1, s * (w * fr)[None])
    den.index_add_(0, i0, w * (1 - fr))
    den.index_add_(0, i1, w * fr)
    prof = num / den.clamp(min=1e-6)[None]
    prof = prof - (prof * den[None]).sum(dim=1, keepdim=True) / den.sum().clamp(min=1e-6)
    return float(((prof * prof) * den[None]).sum() / (den.sum().clamp(min=1e-6) * c))


def _refine_period(s, w, p0, snap_integer=True):
    """Sub-pixel period from the coherent fold score, then integer snap.

    The patch grid / VAE grids are integer periods (16, 8, 4 ...) unless the
    image was resized, so an estimate close to an integer is snapped to it
    as long as the integer still scores reasonably. (A real 16.00 px grid was
    measured as 16.07 on a 2048 px image because the fitted score is flat
    near its maximum; the stricter snap rule missed it.)

    Returns (period, fitted_period) - they differ only when snapped.
    """
    span = 0.02 * p0 + 0.02
    cands = torch.linspace(p0 - span, p0 + span, 161).tolist()
    scores = [_fold_score(s, w, p) for p in cands]
    best = max(range(len(cands)), key=lambda i: scores[i])
    p_best, sc_best = cands[best], scores[best]
    if snap_integer:
        pi = float(round(p_best))
        if pi >= 2 and abs(p_best - pi) <= 0.0095 * pi + 0.02:
            if _fold_score(s, w, pi) >= 0.15 * sc_best:
                return pi, p_best
    return p_best, p_best


def _fold_profiles(s, wts, period, hats_y, lam_rel=0.02):
    """Fold per-zone row signals into stripe profiles.

    s: (Z, C, H) row signal per horizontal zone, wts: (Z, H) pixel weights,
    hats_y: (Zy, H) partition of unity along the stripe axis. The profile is
    fitted separately in every (x-zone, y-zone) cell, because the stripes of
    these models are only quasi-periodic: measured phase wander of ~45 deg
    over the first 600 rows of a 2048 px image, with a constant period.
    Cells with little flat-area data are pulled towards the all-x-zones
    profile of the same y-zone.

    Returns profiles (Z, Zy, C, NB) and the interpolation indices.
    """
    z, c, h = s.shape
    nb = max(4, int(round(period)) if abs(period - round(period)) < 1e-6
             else int(round(period * 2)))
    ys = torch.arange(h, device=s.device, dtype=torch.float64)
    u = ((ys % period) / period * nb).float()
    i0 = torch.floor(u).long() % nb
    fr = u - torch.floor(u)
    i1 = (i0 + 1) % nb

    def accumulate(values):  # (Z, Zy, C or 1, H) -> (Z, Zy, C or 1, NB)
        out = torch.zeros(*values.shape[:3], nb, device=s.device)
        out.index_add_(3, i0, values * (1 - fr))
        out.index_add_(3, i1, values * fr)
        return out

    w = wts[:, None, None, :] * hats_y[None, :, None, :]  # (Z, Zy, 1, H)
    num = accumulate(s[:, None] * w)                       # (Z, Zy, C, NB)
    den = accumulate(w).expand(-1, -1, c, -1)
    prof_p = num.sum(dim=0, keepdim=True) / den.sum(dim=0, keepdim=True).clamp(min=1e-6)
    lam = lam_rel * float(den.sum(dim=0).mean())
    prof = (num + lam * prof_p) / (den + lam).clamp(min=1e-6)
    prof = prof - prof.mean(dim=3, keepdim=True)  # stripes have zero mean
    return prof, i0, i1, fr


def _synthesize(prof, i0, i1, fr, hats_x, hats_y):
    """Stripe pattern (C, H, W) from profiles (Z, Zy, C, NB)."""
    per_row = prof[..., i0] * (1 - fr) + prof[..., i1] * fr   # (Z, Zy, C, H)
    per_row = (per_row * hats_y[None, :, None, :]).sum(dim=1)  # (Z, C, H)
    return torch.einsum("zch,zw->chw", per_row, hats_x)


# ---------------------------------------------------------------------------
# adaptive (local lock-in) pass
# ---------------------------------------------------------------------------
# The zone fit above measures the stripes in the flat areas only (sky) and
# applies that pattern everywhere. On textured surfaces such as skin the real
# stripes were measured to be ~4x stronger than in the sky and slightly
# shifted in phase, so they stay. This pass follows the amplitude and phase
# of every stripe line locally: the image is multiplied with a complex
# carrier at the stripe frequency, averaged over ~64 rows x ~32 px (the
# stripes are coherent over that area, random texture is not), and what is
# left is the local complex amplitude of the stripe. A line is only removed
# to the extent it stands out of the neighbouring frequencies (a real stripe
# is a narrow spectral line, texture and edges are broadband), is capped at
# a small amplitude and is switched off near strong detail.

_AD_PY, _AD_PX = 16, 8          # pooling cell (rows, columns)
_AD_SY, _AD_SX = 4.0, 4.0       # smoothing sigma in pooled cells = 64 x 32 px
_AD_KAPPA = 4.0                 # line must exceed kappa x neighbour level
_AD_CAP = 1.5 / 255.0           # max amplitude of one removed line
_AD_TOTAL_CAP = 3.0 / 255.0     # max total adaptive correction per pixel
# Real periodic structure (brickwork, fences, blinds, hard edges) can have a
# line amplitude of many levels. Cells above ~3 levels are NOT stripe
# artifacts: they get ~0 weight BEFORE the spatial smoothing so they can not
# leak into the smooth areas next to them (v19 added ~1 level of fake 16 px
# stripes on a plain wall between two brick/edge areas), and the applied
# correction is never larger than what the cell itself measures.
_AD_REJECT = 3.0 / 255.0
_AD_LOCAL = 1.5
_AD_EDGE_E0 = 25.0              # default detail rms (levels) where the pass fades out
_AD_MAX_PERIOD = 24.0           # slower lines are too close to image content
_AD_DC_SIGMA = 12.0             # vertical high-pass before demodulation (rows)


def _smooth2d(t, sy, sx):
    """Gaussian smoothing of (N, 1, H, W) with replicate padding."""
    out = t
    if sy > 0:
        k = _gauss_kernel(sy, t.device)
        r = k.numel() // 2
        out = F.conv2d(F.pad(out, (0, 0, r, r), mode="replicate"),
                       k.view(1, 1, -1, 1).to(out.dtype))
    if sx > 0:
        k = _gauss_kernel(sx, t.device)
        r = k.numel() // 2
        out = F.conv2d(F.pad(out, (r, r, 0, 0), mode="replicate"),
                       k.view(1, 1, 1, -1).to(out.dtype))
    return out


def _lockin_raw(x, f, py, px):
    """Pooled (UNsmoothed) complex amplitude of frequency f (cycles/px along H).

    x: (C, H, W). Returns (re, im), each (C, 1, H/py, W/px).
    """
    c, h, w = x.shape
    hc, wc = (h // py) * py, (w // px) * px
    ys = torch.arange(h, device=x.device, dtype=torch.float64)
    ang = (-2.0 * math.pi * f) * ys
    cs = torch.cos(ang).to(x.dtype)[None, :, None]
    sn = torch.sin(ang).to(x.dtype)[None, :, None]
    xr = (x * cs)[:, None, :hc, :wc]
    xi = (x * sn)[:, None, :hc, :wc]
    return F.avg_pool2d(xr, (py, px)), F.avg_pool2d(xi, (py, px))


def _lockin(x, f, py, px, sy, sx):
    """Local complex amplitude of frequency f, smoothed on the pooled grid."""
    zr, zi = _lockin_raw(x, f, py, px)
    return _smooth2d(zr, sy, sx), _smooth2d(zi, sy, sx)


def _comb_frequencies(periods):
    """All harmonics k/P of the detected stripe periods that are short
    enough to be safe for the adaptive pass."""
    freqs = []
    for p in periods:
        for k in range(1, int(p / 2.0) + 1):
            f = k / p
            if f > 0.45 or 1.0 / f > _AD_MAX_PERIOD:
                continue
            if all(abs(f - g) > 0.0015 for g in freqs):
                freqs.append(f)
    return sorted(freqs)


def _adaptive_pass(res, freqs, edge_e0=_AD_EDGE_E0):
    """res: (C, H, W). Returns (cleaned, removed)."""
    c, h, w = res.shape
    dev = res.device
    py, px, sy, sx = _AD_PY, _AD_PX, _AD_SY, _AD_SX
    removed = torch.zeros_like(res)
    if h < 16 * py or w < 16 * px or not freqs:
        return res, removed

    # strong-detail gate on the pooled grid (1 on smooth areas and skin,
    # ~0 around strong edges)
    lum = res.mean(dim=0, keepdim=True) * 255.0
    k6 = _gauss_kernel(6.0, dev)
    blur = _conv_y(lum, k6)
    blur = _conv_y(blur.permute(0, 2, 1).contiguous(), k6).permute(0, 2, 1)
    hp = lum - blur
    hc, wc = (h // py) * py, (w // px) * px
    e2 = F.avg_pool2d((hp * hp)[:, None, :hc, :wc], (py, px))
    e = _smooth2d(e2, 24.0 / py, 24.0 / px).clamp(min=0).sqrt()
    gate = torch.exp(-(e / max(float(edge_e0), 1e-3)) ** 4)
    gate = _smooth2d(gate, sy, sx)

    # Work on a vertically high-passed copy. The image mean (~0.5) is
    # multiplied by the carrier too; the pooling box over py rows cancels it
    # only when the period is exactly py/k px (16, 8, 5.33 ...). For a
    # measured 16.07 px the leak was ~0.5 x 0.0044 = 0.6 levels of FAKE
    # stripe per line. Removing the slow content first removes the leak.
    work = res - _conv_y(res, _gauss_kernel(_AD_DC_SIGMA, dev))
    ys = torch.arange(h, device=dev, dtype=torch.float64)
    for f in freqs:
        zr0, zi0 = _lockin_raw(work, f, py, px)
        mag0 = (zr0 * zr0 + zi0 * zi0).sqrt()                    # (C,1,Hp,Wp)
        wgt = torch.exp(-((2.0 * mag0.amax(dim=0, keepdim=True))
                          / _AD_REJECT) ** 4)                    # (1,1,Hp,Wp)
        zr = _smooth2d(zr0 * wgt, sy, sx)
        zi = _smooth2d(zi0 * wgt, sy, sx)
        local = _smooth2d(mag0, 1.0, 1.0) * _AD_LOCAL
        n2, cnt = None, 0
        for sgn in (1.0, -1.0):
            fp = f + sgn / 64.0
            if abs(fp) < 0.03 or abs(fp) > 0.5:
                continue
            pr, pi = _lockin(work, fp, py, px, sy, sx)
            t = pr * pr + pi * pi
            n2 = t if n2 is None else n2 + t
            cnt += 1
        if cnt == 0:
            continue
        n2 = n2 / cnt
        p2 = zr * zr + zi * zi
        gain = (1.0 - _AD_KAPPA * n2 / p2.clamp(min=1e-18)).clamp(0.0, 1.0)
        gr, gi = zr * gain * gate, zi * gain * gate
        mag = (gr * gr + gi * gi).sqrt()
        lim = local.clamp(max=_AD_CAP / 2.0)
        scale = (lim / mag.clamp(min=1e-12)).clamp(max=1.0)
        gr, gi = gr * scale, gi * scale
        up_r = F.interpolate(gr, size=(h, w), mode="bilinear",
                             align_corners=False)[:, 0]
        up_i = F.interpolate(gi, size=(h, w), mode="bilinear",
                             align_corners=False)[:, 0]
        ang = (2.0 * math.pi * f) * ys
        cs = torch.cos(ang).to(res.dtype)[None, :, None]
        sn = torch.sin(ang).to(res.dtype)[None, :, None]
        line = 2.0 * (up_r * cs - up_i * sn)
        work = work - line
        removed = removed + line

    removed = removed.clamp(-_AD_TOTAL_CAP, _AD_TOTAL_CAP)
    return res - removed, removed


# ---------------------------------------------------------------------------
# finishing grain
# ---------------------------------------------------------------------------
_GRAIN_E0 = 8.0   # local detail rms (1/255 levels) where the grain fades out


def _finish_grain(x, amount, seed):
    """Add soft luminance grain of `amount` (rms, in 1/255 levels), only on
    smooth areas. x: (C, H, W) float in [0, 1].

    Measured on the 2048 px test image the weight is ~0.93-1.0 in the sky,
    ~0.2-0.35 on smooth skin and ~0 on cloth, hem and face detail. The grain
    hides stripe leftovers of a few hundredths of a level in the sky where
    nothing else masks them; it does not remove anything.
    """
    c, h, w = x.shape
    dev = x.device
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    n = torch.randn((1, 1, h, w), generator=gen)
    k = _gauss_kernel(0.6, torch.device("cpu"))     # tiny blur = soft grain
    r = k.numel() // 2
    n = F.conv2d(F.pad(n, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    n = F.conv2d(F.pad(n, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    n = (n / n.std().clamp(min=1e-6)).to(dev)

    lum = x.mean(dim=0, keepdim=True) * 255.0
    k6 = _gauss_kernel(6.0, dev)
    blur = _conv_y(lum, k6)
    blur = _conv_y(blur.permute(0, 2, 1).contiguous(), k6).permute(0, 2, 1)
    hp = lum - blur
    e2 = F.avg_pool2d((hp * hp)[None], (8, 8))
    e = _smooth2d(e2, 3.0, 3.0).clamp(min=0).sqrt()      # ~24 px scale
    wgt = torch.exp(-(e / _GRAIN_E0) ** 4)
    wgt = F.interpolate(wgt, size=(h, w), mode="bilinear",
                        align_corners=False)[0]          # (1, H, W)
    return x + (float(amount) / 255.0) * n[0] * wgt


def _remove_axis(img, cfg, log, axis_name):
    """img: (C, H, W) with the stripe pattern varying along H.

    Returns (cleaned, removed) both (C, H, W).
    """
    c, h, w = img.shape
    dev = img.device
    removed = torch.zeros_like(img)
    if h < 8 * cfg["min_period"] * 2 or h < 96:
        log.append(f"{axis_name}: image too small, skipped")
        return img, removed

    lum = img.mean(dim=0)
    mask_b, detail_thr = _flat_mask(lum, cfg["flat_percent"])
    detail_levels = detail_thr * 255.0      # detail rms of the 'flat' pixels
    # In a picture without any smooth area (detail of the flattest pixels well
    # above the ~5-6 levels of normal pictures) the relaxed grid threshold is
    # not used: the stripes are measured against real texture.
    grid_factor = _GRID_THRESHOLD_FACTOR if detail_levels < 10.0 else 1.0
    mask = mask_b.float()
    n_flat = float(mask.sum())
    if n_flat < 2000:
        log.append(f"{axis_name}: no flat area found, skipped")
        return img, removed

    n_zones = 1
    if cfg["spatial_adaptive"]:
        n_zones = int(max(1, min(12, round(w / 170.0))))
    hats = _hat_weights(w, n_zones, dev)
    n_zones_y = 1
    if cfg["spatial_adaptive"]:
        n_zones_y = int(max(1, min(8, round(h / 256.0))))
    hats_y = _hat_weights(h, n_zones_y, dev)

    res = img.clone()
    gauss_k = _gauss_kernel(max(4.0, cfg["max_period"] / 2.0), dev)
    found_any = False
    used = []
    used_periods = []

    skipped = []          # periods found but not usable (too weak / too strong)
    cum_amp_sq = 0.0
    first_prom = None
    # A weak or rejected period must not stop the search for the other ones
    # (a very clean sky: the strongest peak was a 2 px line far below
    # min_amplitude and the 16 / 8 px stripes behind it were never tried).
    for attempt in range(cfg["max_components"] + 6):
        if len(used_periods) >= cfg["max_components"]:
            break
        # --- detection on the Gaussian high-passed residual
        d0 = res - _conv_y(res, gauss_k)
        sel = d0[:, mask_b]
        if sel.numel() < 100:
            break
        sigma = float(sel.abs().median()) * 1.4826
        clip = max(6.0 * sigma, 1e-4)

        if cfg["mode"] == "manual":
            if attempt > 0:
                break
            period, prom = float(cfg["period"]), float("inf")
        else:
            period, prom, s_zone, w_zone = _detect_tiles(
                d0, mask, clip, cfg["min_period"], cfg["max_period"],
                cfg["threshold"], exclude=used_periods + skipped,
                grid_only=cfg["grid_only"], grid_factor=grid_factor)
            if first_prom is None and period is not None:
                first_prom = prom
        if period is None:
            if attempt == 0:
                if prom <= 0.0:
                    log.append(f"{axis_name}: no flat area large enough "
                               f"to measure the stripes")
                else:
                    log.append(f"{axis_name}: no periodic stripes found "
                               f"(peak strength {prom:.0f} < threshold "
                               f"{cfg['threshold']:.0f})")
            break
        fitted = period
        if cfg["mode"] == "auto":
            # Grid periods are exact (4 px, 16 px, 32/3 px ...). The coherent
            # fold score is a poor judge on a texture-heavy picture: it
            # compared a 4-bin fold (4.00 px) with an 8-bin fold (4.03 px),
            # the extra bins fitted more noise, 4.00 lost and 4.03 was used -
            # a 0.03 px error drifts 15 px over 2048 rows, and only ~10% of a
            # 2-3 level stripe was removed instead of ~30%.
            g = _snap_to_grid(period) if cfg["grid_only"] else None
            if g is not None:
                period = g
            else:
                period, fitted = _refine_period(s_zone, w_zone, period)
        if h < 8 * period:
            log.append(f"{axis_name}: period {period:.2f}px too long for this image")
            break
        if any(abs(period - p) < 0.02 * p for p in used_periods):
            skipped.append(period)
            continue  # same period again = only measurement leftovers

        # --- measurement with the matched high-pass (x - box_P(x))
        d_p = res - _conv_y(res, _box_kernel(period, dev))
        s_z, w_z = _row_signals(d_p, mask, hats, clip)
        prof, i0, i1, fr = _fold_profiles(s_z, w_z, period, hats_y)
        pattern = _synthesize(prof, i0, i1, fr, hats, hats_y)

        amp = float(pattern.pow(2).mean().sqrt()) * 255.0
        total_amp = math.sqrt(amp * amp + cum_amp_sq)   # all removed patterns together
        if total_amp > cfg["max_amplitude"]:
            log.append(f"{axis_name}: {period:.2f}px pattern rejected - amplitude "
                       f"{total_amp:.2f}/255 is above max_amplitude "
                       f"{cfg['max_amplitude']:.1f} (probably real texture)")
            skipped.append(period)
            continue
        if amp < cfg["min_amplitude"]:
            skipped.append(period)
            continue

        # Iterate on the FULLY cleaned residual so the next detection does not
        # see the part that `strength` < 1 would leave behind; `strength`
        # is applied once, at the very end.
        res = res - pattern
        removed = removed + pattern
        cum_amp_sq += amp * amp
        used_periods.append(period)
        found_any = True
        snapped = f", fitted {fitted:.2f}" if abs(fitted - period) > 1e-6 else ""
        used.append(
            f"{period:.2f}px (strength {prom:.0f}x, rms {amp:.2f}/255{snapped})"
            if math.isfinite(prom) else f"{period:.2f}px manual (rms {amp:.2f}/255)"
        )

    if found_any:
        log.append(f"{axis_name}: removed " + " + ".join(used)
                   + f" - flat area used: {100.0 * n_flat / (h * w):.0f}%")
    if cfg.get("adaptive") and used_periods:
        freqs = _comb_frequencies(used_periods)
        res, extra = _adaptive_pass(res, freqs, cfg["adaptive_edge"])
        removed = removed + extra
        rms = float(extra.pow(2).mean().sqrt()) * 255.0
        if rms > 0.0:
            log.append(f"{axis_name}: adaptive pass {rms:.2f}/255 rms "
                       f"on {len(freqs)} lines")
    if found_any and cfg["mode"] == "auto" and first_prom is not None:
        # Verification in float, on the cleaned residual: how sharp is the
        # strongest periodic peak that is still left? (An 8-bit saved image
        # can not show this in a very smooth sky: its rows carry almost no
        # information below ~0.1 level.)
        d0 = res - _conv_y(res, gauss_k)
        sel = d0[:, mask_b]
        if sel.numel() > 100:
            clip_r = max(6.0 * float(sel.abs().median()) * 1.4826, 1e-4)
            p_r, prom_r, _, _ = _detect_tiles(
                d0, mask, clip_r, cfg["min_period"], cfg["max_period"], 0.0)
            if p_r is not None:
                log.append(f"{axis_name}: strongest peak left {p_r:.2f}px "
                           f"{prom_r:.0f}x (first found {first_prom:.0f}x)")
    removed = removed * cfg["strength"]
    return img - removed, removed


# ---------------------------------------------------------------------------
# the node
# ---------------------------------------------------------------------------

class MykeeStripeRemover:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Off = the image passes through completely "
                               "untouched (quick A/B comparison).",
                }),
                "direction": (["both", "horizontal stripes", "vertical stripes"], {
                    "default": "both",
                    "tooltip": "Which stripes to look for. 'horizontal stripes' "
                               "= lines running left-right (the pattern changes "
                               "from row to row). 'both' measures each direction "
                               "separately and only removes what is really "
                               "there.",
                }),
                "mode": (["auto", "manual"], {
                    "default": "auto",
                    "tooltip": "auto: the period of the stripes is measured "
                               "from the image. manual: use the 'period' "
                               "widget instead (e.g. 16 for a Flux/Chroma "
                               "style patch grid).",
                }),
                "period": ("FLOAT", {
                    "default": 16.0, "min": 2.0, "max": 256.0, "step": 0.01,
                    "tooltip": "Manual mode only: stripe period in pixels "
                               "(also removes its harmonics).",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05,
                    "tooltip": "1.0 = subtract the measured stripes fully. "
                               "Lower = partial removal.",
                }),
                "max_components": ("INT", {
                    "default": 4, "min": 1, "max": 5,
                    "tooltip": "How many different stripe periods may be "
                               "removed one after another per direction "
                               "(typically 4 and 8 px from the VAE, 16 and "
                               "32 px from the DiT patch grid).",
                }),
                "spatial_adaptive": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Measure the stripes separately in a grid of "
                               "zones (strength changes across the picture, "
                               "and the phase of the stripes wanders slowly "
                               "from top to bottom). Off = one global "
                               "profile.",
                }),
                "detection_threshold": ("FLOAT", {
                    "default": 20.0, "min": 5.0, "max": 1000.0, "step": 1.0,
                    "tooltip": "Auto mode: how sharp a periodic peak must be "
                               "(compared with the noise around it) to count "
                               "as stripes. Lower = more sensitive, higher = "
                               "safer against false detections. The status "
                               "line shows the measured strength.",
                }),
                "max_amplitude": ("FLOAT", {
                    "default": 2.5, "min": 0.5, "max": 50.0, "step": 0.1,
                    "tooltip": "Safety limit (rms, in 1/255 units) for ALL removed "
                               "patterns together. Real stripe artifacts "
                               "measured 0.1-0.6 rms; a pattern that would "
                               "push the total above this is treated as real "
                               "image content (e.g. corrugated metal) and is "
                               "left alone.",
                }),
                "min_period": ("FLOAT", {
                    "default": 2.0, "min": 2.0, "max": 64.0, "step": 0.5,
                    "tooltip": "Auto mode: shortest period searched (px).",
                }),
                "max_period": ("FLOAT", {
                    "default": 48.0, "min": 4.0, "max": 256.0, "step": 1.0,
                    "tooltip": "Auto mode: longest period searched (px).",
                }),
                "flat_area_percent": ("FLOAT", {
                    "default": 35.0, "min": 5.0, "max": 90.0, "step": 1.0,
                    "tooltip": "The smoothest X% of the picture is used for "
                               "measuring the stripes. Raise it if there is "
                               "very little sky / smooth area.",
                }),
                "preview_gain": ("FLOAT", {
                    "default": 40.0, "min": 1.0, "max": 500.0, "step": 1.0,
                    "tooltip": "Amplification of the removed_stripes preview "
                               "ONLY - never affects the cleaned image.",
                }),
                "preview_view": (["4x zoom", "8x zoom", "full frame"], {
                    "default": "4x zoom",
                    "tooltip": "Framing of removed_stripes. The zoom options "
                               "show a magnified centre crop where the "
                               "stripes are clearly visible. Preview only; "
                               "the cleaned image is never cropped.",
                }),
                "dither": (["auto", "on", "off"], {
                    "default": "auto",
                    "tooltip": "Adds +-0.5/255 white noise to the cleaned "
                               "image. Needed when the INPUT is an 8-bit "
                               "picture (Load Image): its values are already "
                               "whole numbers, so subtracting a sub-1/255 "
                               "stripe pattern and saving to 8 bit turns it "
                               "into a NEW 1-level stripe pattern. auto = "
                               "on only if the input is detected as 8-bit "
                               "(a fresh VAE Decode output is float and "
                               "needs no dither).",
                }),
                "adaptive_pass": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "After the zone fit, follow the stripes "
                               "locally (amplitude and phase) so they are "
                               "also removed from textured areas such as "
                               "skin, where they are stronger than in the "
                               "sky. Only the stripe periods found by the "
                               "measurement are touched, and it fades out "
                               "around strong edges. Off = zone fit only.",
                }),
                "adaptive_detail_limit": ("FLOAT", {
                    "default": 25.0, "min": 5.0, "max": 80.0, "step": 1.0,
                    "tooltip": "Adaptive pass only: local detail strength "
                               "(rms, in 1/255 levels) at which the pass "
                               "fades out. Higher = removes more stripes "
                               "from textured areas such as skin, but also "
                               "acts closer to strong edges (risk of faint "
                               "ghost lines there). 15-25 is safe; above "
                               "~40 the correction concentrates on edges.",
                }),
                "finish_grain": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Soft luminance grain (rms, in 1/255 levels) "
                               "added AFTER the stripe removal, only on "
                               "smooth areas (sky, smooth skin); it fades "
                               "out on detail. It does not remove stripes, "
                               "it hides the faint leftovers. 0 = off. "
                               "Try 0.5-1.0.",
                }),
                "grid_periods_only": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto mode: only accept stripes at the periods "
                               "the generators really produce (VAE grid 2/4/8 "
                               "px, DiT grid 16 px and their harmonics, 32/k "
                               "px). Any other periodic pattern is treated as "
                               "real image content. Off = any period (needed "
                               "only for pictures that were resized after "
                               "decoding).",
                }),
                "log_to_console": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print every run's log (found periods, how "
                               "much was removed, what is left) to the "
                               "ComfyUI console / terminal window. Off by "
                               "default; meant for checking and testing.",
                }),
            },
            "optional": {
                "label": ("STRING", {
                    "default": "",
                    "tooltip": "Free text added to the header of this run's "
                               "console log - e.g. connect a seed or a file "
                               "name prefix to tell the runs apart.",
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "removed_stripes", "status")
    FUNCTION = "run"
    CATEGORY = "Mykee/Image"

    def run(self, image, enabled, direction, mode, period, strength,
            max_components, spatial_adaptive, detection_threshold,
            max_amplitude, min_period, max_period, flat_area_percent,
            preview_gain, preview_view, dither="auto", adaptive_pass=True,
            adaptive_detail_limit=25.0, finish_grain=0.0,
            grid_periods_only=True, log_to_console=True, label="",
            unique_id=None):
        if not enabled:
            gray = torch.full_like(image[..., :3], 0.5)
            msg = _publish(
                image, label, "disabled - image passed through untouched",
                log_to_console, unique_id)
            return (image, gray, msg)

        cfg = {
            "mode": mode,
            "period": period,
            "strength": strength,
            "max_components": int(max_components),
            "spatial_adaptive": bool(spatial_adaptive),
            "threshold": float(detection_threshold),
            "max_amplitude": float(max_amplitude),
            "min_amplitude": 0.02,
            "min_period": float(min_period),
            "max_period": float(max(max_period, min_period + 1.0)),
            "flat_percent": float(flat_area_percent),
            "adaptive": bool(adaptive_pass),
            "adaptive_edge": float(adaptive_detail_limit),
            "grid_only": bool(grid_periods_only),
        }
        dev = _work_device()
        out_imgs, prev_imgs, lines = [], [], []

        for b in range(image.shape[0]):
            src = image[b]
            n_ch = min(3, src.shape[-1])
            x = src[..., :n_ch].permute(2, 0, 1).to(dev, torch.float32)
            log = []
            removed_total = torch.zeros_like(x)
            cur = x
            if direction in ("both", "horizontal stripes"):
                cur, rem = _remove_axis(cur, cfg, log, "horizontal")
                removed_total = removed_total + rem
            if direction in ("both", "vertical stripes"):
                cur_t = cur.permute(0, 2, 1).contiguous()
                cur_t, rem_t = _remove_axis(cur_t, cfg, log, "vertical")
                cur = cur_t.permute(0, 2, 1).contiguous()
                removed_total = removed_total + rem_t.permute(0, 2, 1)

            if finish_grain > 0.0:
                cur = _finish_grain(cur, finish_grain, 7654321 + b)
                log.append(f"finish grain {finish_grain:.2f}/255 on smooth areas")
            use_dither = dither == "on" or (
                dither == "auto" and _is_8bit_source(src[..., :n_ch].cpu()))
            if use_dither and float(removed_total.abs().max()) > 0.0:
                gen = torch.Generator(device="cpu")
                gen.manual_seed(1234567 + b)  # deterministic: cache friendly
                noise = (torch.rand(cur.shape, generator=gen) - 0.5) / 255.0
                cur = cur + noise.to(cur.device)
                log.append("dither on" + (" (8-bit source detected)"
                                          if dither == "auto" else ""))
            cleaned = cur.clamp(0.0, 1.0).permute(1, 2, 0).cpu()
            if src.shape[-1] > n_ch:  # keep any extra (alpha) channel
                cleaned = torch.cat([cleaned, src[..., n_ch:]], dim=-1)
            out_imgs.append(cleaned)

            prev = (0.5 + preview_gain * removed_total).clamp(0, 1)
            prev = prev.permute(1, 2, 0).cpu()
            if prev.shape[-1] == 1:
                prev = prev.expand(-1, -1, 3)
            if preview_view != "full frame":
                zoom = 4 if preview_view == "4x zoom" else 8
                ph, pw = prev.shape[0], prev.shape[1]
                ch_, cw_ = max(8, ph // zoom), max(8, pw // zoom)
                y0, x0 = (ph - ch_) // 2, (pw - cw_) // 2
                crop = prev[y0:y0 + ch_, x0:x0 + cw_]
                # nearest-neighbour magnify back to the original size
                crop = crop.permute(2, 0, 1)[None]
                prev = F.interpolate(crop, size=(ph, pw), mode="nearest")[0]
                prev = prev.permute(1, 2, 0)
            prev_imgs.append(prev)

            if not log:
                log.append("nothing to do")
            # one line per message; a batch gets one block per image
            if image.shape[0] > 1:
                lines.append(f"Image {b + 1}/{image.shape[0]}:\n"
                             + "\n".join("  " + m for m in log))
            else:
                lines.append("\n".join(log))

        status = _publish(image, label, "\n\n".join(lines),
                          log_to_console, unique_id)
        return (torch.stack(out_imgs), torch.stack(prev_imgs), status)


NODE_CLASS_MAPPINGS = {
    "MykeeStripeRemover": MykeeStripeRemover,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeStripeRemover": "Mykee Stripe Remover",
}
