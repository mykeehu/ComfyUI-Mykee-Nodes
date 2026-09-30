"""
ComfyUI-Mykee-Nodes / emotion + voice-quality node

Mykee Emotion Timbre is built on the WORLD vocoder (via the `pyworld`
package): the input audio is decomposed into F0 (pitch contour), a
smoothed spectral envelope (SP) and an aperiodicity map (AP), those
parameters are reshaped, and the result is re-synthesized. This is
analysis-resynthesis, not generation - the output is built directly from
the same recording, so voice identity / accent is preserved far more
reliably than any generative "restyle" model could guarantee.

(This pack used to also ship a separate Mykee Prosody Shaper node for
tempo/pitch/energy only - its functionality, including micro_jitter_semitones,
is now fully covered by Mykee Emotion Timbre's tempo_scale/pitch_shift_semitones/
pitch_range_scale/micro_jitter_semitones/energy_range_scale, so it was removed.)

Requires: pip install pyworld
(pure-Python/Cython, MIT-licensed, no network calls, no GPU/model weights.
On Windows, if PyPI has no prebuilt wheel for your exact Python version, pip
will try to build it from source and needs the "Microsoft C++ Build Tools"
installed.)
"""

import json
import os
import sys
import types
import importlib.metadata as _importlib_metadata

import numpy as np
import torch

from .mykee_audio import _block_rms, _upsample_curve


def _ensure_pkg_resources_shim():
    """
    pyworld's __init__.py only does one thing with pkg_resources:
    `pkg_resources.get_distribution('pyworld').version`. Recent setuptools
    versions (81+) no longer bundle pkg_resources, which breaks pyworld's
    import - but pinning setuptools to an old version is fragile (torch,
    ComfyUI, or any other package can bump it right back up). Instead of
    relying on the environment's setuptools version at all, this stubs out
    just that one call using the stdlib's importlib.metadata - so pyworld
    imports cleanly regardless of what setuptools version is installed.
    If a real pkg_resources is already importable, this does nothing and
    gets out of the way.
    """
    if "pkg_resources" in sys.modules:
        return
    try:
        import pkg_resources  # noqa: F401
        return
    except ImportError:
        pass

    shim = types.ModuleType("pkg_resources")

    class _FakeDistribution:
        def __init__(self, version):
            self.version = version

    def get_distribution(name):
        try:
            version = _importlib_metadata.version(name)
        except Exception:
            version = "0.0.0"
        return _FakeDistribution(version)

    shim.get_distribution = get_distribution
    sys.modules["pkg_resources"] = shim


_ensure_pkg_resources_shim()

try:
    import pyworld as pw
    _HAS_PYWORLD = True
except ImportError:
    _HAS_PYWORLD = False

_FRAME_PERIOD_MS = 5.0
_SEMITONE = np.log(2.0) / 12.0  # 1 semitone in natural-log units
_FIXED_NOISE_SEED = 0  # internal seed for the smoothed-noise generators (micro_jitter, rate_variability, roughness) - not user-exposed, just keeps repeated runs deterministic

# mood_presets.json lives in the pack root (one level above nodes/), not next to this file.
_MOOD_PRESETS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mood_presets.json"
)


def _require_pyworld():
    if not _HAS_PYWORLD:
        raise RuntimeError(
            "This node needs the 'pyworld' package (WORLD vocoder). "
            "Install it with: pip install pyworld "
            "(on Windows, if no prebuilt wheel matches your Python version, "
            "you'll need the Microsoft C++ Build Tools to build it from source)."
        )


def _load_mood_presets_raw():
    try:
        with open(_MOOD_PRESETS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "neutral": {
                "tempo_scale": 1.0, "rate_variability": 0.0,
                "pitch_shift_semitones": 0.0, "pitch_range_scale": 1.0,
                "energy_range_scale": 1.0, "pause_scale": 1.0,
                "breathiness": 0.0, "brightness": 0.0, "roughness": 0.0,
                "vibrato_depth_semitones": 0.0, "vibrato_rate_hz": 5.0,
            }
        }


def _load_mood_presets():
    return {k: v for k, v in _load_mood_presets_raw().items() if not k.startswith("_")}


def _save_mood_presets_raw(data):
    tmp_path = _MOOD_PRESETS_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, _MOOD_PRESETS_PATH)


_PRESET_FIELDS = (
    "tempo_scale", "rate_variability", "pitch_shift_semitones", "pitch_range_scale",
    "energy_range_scale", "pause_scale", "breathiness", "brightness", "roughness",
    "vibrato_depth_semitones", "vibrato_rate_hz", "micro_jitter_semitones",
)


def _sanitize_preset_values(values):
    out = {}
    if not isinstance(values, dict):
        return out
    for k in _PRESET_FIELDS:
        if k in values:
            try:
                out[k] = float(values[k])
            except (TypeError, ValueError):
                pass
    return out


# Server-side API routes for the "Mykee Emotion Timbre" node's preset
# save/delete buttons (web/mykee_emotion_timbre.js). Guarded so this module
# still imports cleanly outside a running ComfyUI server (e.g. for testing).
try:
    from aiohttp import web as _aiohttp_web
    from server import PromptServer as _PromptServer

    _mykee_routes = _PromptServer.instance.routes

    @_mykee_routes.get("/mykee/emotion_timbre/presets")
    async def _mykee_get_presets(request):
        return _aiohttp_web.json_response({"presets": _load_mood_presets()})

    @_mykee_routes.post("/mykee/emotion_timbre/save_preset")
    async def _mykee_save_preset(request):
        try:
            data = await request.json()
        except Exception:
            return _aiohttp_web.json_response({"error": "invalid JSON body"}, status=400)
        name = str(data.get("name", "")).strip()
        if not name:
            return _aiohttp_web.json_response({"error": "name is required"}, status=400)
        if name.lower() == "custom":
            return _aiohttp_web.json_response(
                {"error": "'custom' is reserved and can't be saved as a preset"}, status=400
            )
        raw = _load_mood_presets_raw()
        raw[name] = _sanitize_preset_values(data.get("values"))
        _save_mood_presets_raw(raw)
        return _aiohttp_web.json_response({"presets": _load_mood_presets()})

    @_mykee_routes.post("/mykee/emotion_timbre/delete_preset")
    async def _mykee_delete_preset(request):
        try:
            data = await request.json()
        except Exception:
            return _aiohttp_web.json_response({"error": "invalid JSON body"}, status=400)
        name = str(data.get("name", "")).strip()
        if not name:
            return _aiohttp_web.json_response({"error": "name is required"}, status=400)
        if name.lower() == "custom":
            return _aiohttp_web.json_response({"error": "'custom' can't be deleted"}, status=400)
        raw = _load_mood_presets_raw()
        raw.pop(name, None)
        _save_mood_presets_raw(raw)
        return _aiohttp_web.json_response({"presets": _load_mood_presets()})

except Exception:
    pass


def _world_analyze(mono_f64, sr):
    f0, t = pw.dio(mono_f64, sr, frame_period=_FRAME_PERIOD_MS)
    f0 = pw.stonemask(mono_f64, f0, t, sr)
    sp = pw.cheaptrick(mono_f64, f0, t, sr)
    ap = pw.d4c(mono_f64, f0, t, sr)
    return f0, sp, ap


def _world_synthesize(f0, sp, ap, sr):
    y = pw.synthesize(
        np.ascontiguousarray(f0),
        np.ascontiguousarray(sp),
        np.ascontiguousarray(ap),
        sr,
        _FRAME_PERIOD_MS,
    )
    return y.astype(np.float32)


def _fit_np_length(x, target_len):
    n = x.shape[0]
    if n == target_len:
        return x
    if n > target_len:
        return x[:target_len]
    return np.pad(x, (0, target_len - n))


def _frame_presence(sp, gate_low_db=-45.0, gate_high_db=-25.0):
    """
    Per-frame weight in [0, 1], 0 in near-silent/pause frames and 1 in
    frames with real signal, based on each frame's energy relative to the
    loudest frame in the clip (from the WORLD spectral envelope, so it's
    already time-aligned with f0/sp/ap). Used to keep voice-quality
    effects (breathiness, brightness, roughness) from being audible during
    pauses, where they'd otherwise sound like a constant mic-noise/air
    layer rather than a property of the voice itself.
    """
    frame_energy = np.mean(sp, axis=1)
    energy_db = 10.0 * np.log10(frame_energy + 1e-12)
    peak_db = np.max(energy_db)
    rel_db = energy_db - peak_db
    return np.clip((rel_db - gate_low_db) / (gate_high_db - gate_low_db), 0.0, 1.0)



def _energy_expand_gain(y, sr, scale, control_hz=100.0, gate_low_db=-45.0, gate_high_db=-30.0):
    """
    Builds a gain curve that scales how much the loudness contour moves
    around its own median (>1 widens the dynamic range, <1 flattens it) -
    but leaves near-silent stretches (pauses, WORLD synthesis noise floor)
    at their original level regardless of `scale`. Without this gate,
    flattening the dynamics (scale < 1, e.g. for a "sad" mood) pulls those
    near-silent stretches UP toward the loudness median, which audibly
    raises the noise floor in every pause - exactly the artifact this
    guards against.

    gate_low_db/gate_high_db are relative to the loudest frame in the clip:
    frames quieter than gate_low_db are left untouched (gain 1.0), frames
    louder than gate_high_db get the full effect, and it fades linearly
    between the two so there's no audible seam at the gate boundary.
    """
    hop = max(1, int(round(sr / control_hz)))
    env = _block_rms(np.abs(y), hop)
    if env.size == 0:
        return np.ones_like(y, dtype=np.float64)

    baseline_e = np.median(env)
    target_env = np.maximum(baseline_e + (env - baseline_e) * scale, 1e-6)

    env_db = 20.0 * np.log10(env + 1e-9)
    peak_db = np.max(env_db)
    rel_db = env_db - peak_db
    presence = np.clip((rel_db - gate_low_db) / (gate_high_db - gate_low_db), 0.0, 1.0)

    blended_target = env * (1.0 - presence) + target_env * presence
    gain = np.clip(blended_target / np.maximum(env, 1e-6), 0.15, 4.0)
    return _upsample_curve(gain, y.shape[0])


def _smoothed_noise(n, seed, coef=0.85):
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 1.0, size=n)
    out = np.zeros(n, dtype=np.float64)
    prev = 0.0
    for i, v in enumerate(raw):
        prev = coef * prev + (1.0 - coef) * v
        out[i] = prev
    peak = np.max(np.abs(out)) if n else 0.0
    return out / peak if peak > 1e-9 else out


def _smooth_curve_symmetric(curve, coef=0.6):
    """Zero-phase-ish smoothing (forward pass + backward pass, averaged) -
    used for the pause-probability mask so pause boundaries don't cause
    abrupt jumps in the local playback rate."""
    n = curve.shape[0]
    fwd = np.zeros(n, dtype=np.float64)
    prev = 0.0
    for i in range(n):
        prev = coef * prev + (1.0 - coef) * curve[i]
        fwd[i] = prev
    bwd = np.zeros(n, dtype=np.float64)
    prev = 0.0
    for i in range(n - 1, -1, -1):
        prev = coef * prev + (1.0 - coef) * curve[i]
        bwd[i] = prev
    return (fwd + bwd) / 2.0


def _warp_resample(arr, source_index_for_output):
    """Resamples arr (1D or 2D, frame axis first) onto a new, possibly
    non-uniform frame grid, given for each output frame the (fractional)
    source frame index it should be read from."""
    n_frames = arr.shape[0]
    x_src = np.arange(n_frames, dtype=np.float64)
    if arr.ndim == 1:
        return np.interp(source_index_for_output, x_src, arr)
    out = np.empty((source_index_for_output.shape[0], arr.shape[1]), dtype=arr.dtype)
    for i in range(arr.shape[1]):
        out[:, i] = np.interp(source_index_for_output, x_src, arr[:, i])
    return out


def _min_run_filter(mask, min_len):
    """Zeroes out any run of consecutive 1s in `mask` shorter than
    `min_len` frames. Used so a brief low-energy dip (a stop-consonant
    closure, ~20-50ms) doesn't get misclassified as a pause - only a
    sustained quiet stretch (a real pause between words/phrases) does.
    Without this, brief closures were getting time-stretched along with
    real pauses, smearing fast consonant transitions into a buzzy,
    noise-like texture."""
    out = mask.copy()
    n = mask.shape[0]
    i = 0
    while i < n:
        if mask[i] > 0.5:
            j = i
            while j < n and mask[j] > 0.5:
                j += 1
            if (j - i) < min_len:
                out[i:j] = 0.0
            i = j
        else:
            i += 1
    return out


def _build_variable_warp(n_frames, sp, tempo_scale, rate_variability, pause_scale, seed):
    """
    Builds a local-playback-rate curve over the utterance and returns the
    (new_n_frames, source_index_for_output) pair needed by _warp_resample.

    - tempo_scale: overall speed multiplier.
    - rate_variability: adds smoothed random speed-up/slow-down on top of
      tempo_scale - an uneven, erratic pace (e.g. angry speech rushing then
      catching itself) rather than a perfectly steady rate change.
    - pause_scale: independently stretches/compresses just the low-energy
      (pause-like) stretches of the recording, detected from the WORLD
      spectral envelope's frame energy - this is what actually makes sad
      speech feel like it has longer silences, or angry speech clipped,
      short pauses, without changing the speech portions' rate. Only
      stretches of at least ~150ms count as a pause candidate (see
      _min_run_filter) - short dips are consonant closures, not pauses.
    """
    if n_frames < 2:
        return n_frames, np.arange(n_frames, dtype=np.float64)

    frame_energy = np.mean(sp, axis=1)
    thresh = np.percentile(frame_energy, 25) * 1.2
    raw_pause = (frame_energy < thresh).astype(np.float64)
    min_pause_frames = max(1, int(round(150.0 / _FRAME_PERIOD_MS)))
    raw_pause = _min_run_filter(raw_pause, min_pause_frames)
    pause_prob = _smooth_curve_symmetric(raw_pause, coef=0.6)

    if abs(rate_variability) > 1e-6:
        rate_noise = _smoothed_noise(n_frames, seed + 1000)
    else:
        rate_noise = np.zeros(n_frames)

    r_base = tempo_scale * (1.0 + rate_variability * 0.35 * rate_noise)
    r_base = np.maximum(r_base, 0.15)

    r_final = r_base * (1.0 - pause_prob + pause_prob / max(pause_scale, 1e-3))
    r_final = np.maximum(r_final, 0.1)

    inv_r = 1.0 / r_final
    cum_output = np.cumsum(inv_r)
    total_output = cum_output[-1]
    new_n_frames = max(2, int(round(total_output)))
    output_positions = np.arange(new_n_frames, dtype=np.float64)
    source_index_for_output = np.interp(
        output_positions, cum_output, np.arange(n_frames, dtype=np.float64)
    )
    return new_n_frames, source_index_for_output


def _peak_safe_scale(tensor, target_peak=0.98):
    """
    Linearly scales the whole tensor down (never up) so its peak doesn't
    exceed target_peak - unlike a tanh/soft-clip limiter, this can't add
    harmonic distortion, since it only ever changes overall level, never
    the waveform's shape. Needed because breathiness/brightness/roughness
    each add a bit of extra energy on top of the original signal, and a
    source that's already sitting at or near 0 dBFS (common for TTS
    output) has zero headroom to absorb that - a hard limiter would
    otherwise engage strongly enough to be audible as its own artifact.
    """
    peak = tensor.abs().max()
    if peak > target_peak:
        tensor = tensor * (target_peak / peak)
    return tensor


class MykeeEmotionTimbre:
    """
    Emotion shaper, built on the WORLD vocoder. The acoustic-emotion
    literature (e.g. Yildirim et al. 2004; Major & Chatterjee 2026's
    RAVDESS analysis) is consistent: the dominant carriers of vocal emotion
    are prosodic *dynamics* - mean pitch and pitch RANGE/variability,
    speaking rate and its variability, energy/loudness and its dynamic
    range, and pause length/frequency (sad speech pauses more and longer;
    angry/happy speech has short, clipped pauses) - not static voice
    timbre. Voice quality (breathiness, spectral brightness, roughness,
    vibrato) is a real but secondary cue.

    So this node applies both, in that order of importance: it warps a
    LOCAL, time-varying playback-rate curve (not a single global speed
    factor) built from tempo_scale + rate_variability (uneven pacing) +
    pause_scale (independently stretches/compresses just the low-energy,
    pause-like stretches, detected automatically), reshapes the pitch
    contour's register and range, reshapes the loudness contour's dynamic
    range, and only then layers the four voice-quality knobs on top.

    Named "mood" presets (from mood_presets.json in the pack root folder) are
    just saved combinations of all of these continuous parameters - to add
    a new mood, add a JSON entry, no code changes needed.

    intensity blends every parameter toward its neutral value (not the
    waveform toward the dry signal) before processing, since tempo/pause
    changes alter the output's duration - a real dry/wet crossfade isn't
    meaningful once the two aren't the same length.
    """

    @classmethod
    def INPUT_TYPES(cls):
        presets = list(_load_mood_presets().keys())
        preset_options = ["custom"] + presets
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "The narration to re-emote.",
                }),
                "preset": (preset_options, {
                    "default": "custom",
                    "tooltip": "If not 'custom', this overrides every slider below with the values stored in mood_presets.json for this mood. Pick 'custom' to control the sliders by hand. Add new moods by editing mood_presets.json (restart ComfyUI to see them here).",
                }),
                "tempo_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01,
                    "tooltip": "Overall speed multiplier, as in Mykee Prosody Shaper: >1 faster (angry/excited/fear), <1 slower (sad/tired). Ignored if preset != custom.",
                }),
                "rate_variability": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Adds uneven, erratic pacing on top of tempo_scale - rushing and catching itself, rather than a perfectly steady speed change. High for anger/excitement, near 0 for sadness/calm. Ignored if preset != custom.",
                }),
                "pitch_shift_semitones": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5,
                    "tooltip": "Shifts the whole pitch register up/down. Ignored if preset != custom.",
                }),
                "pitch_range_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.5, "step": 0.05,
                    "tooltip": "Scales how much the pitch moves around its own median. 1.0 = unchanged. <1.0 flattens toward monotone (sadness has the narrowest pitch range in the literature). >1.0 widens it (happiness/anger have the widest). Ignored if preset != custom.",
                }),
                "energy_range_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.5, "step": 0.05,
                    "tooltip": "Scales how much the loudness moves around its own median. 1.0 = unchanged. <1.0 flattens dynamics (subdued/sad). >1.0 widens it (quiet parts quieter, loud/shouted parts louder). Ignored if preset != custom.",
                }),
                "pause_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.3, "max": 3.0, "step": 0.05,
                    "tooltip": "Stretches (>1) or compresses (<1) just the low-energy, pause-like stretches of the recording (auto-detected) - speech portions keep their own rate. Sadness pauses more/longer; anger/excitement has short, clipped pauses. Ignored if preset != custom.",
                }),
                "breathiness": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Secondary voice-quality cue. Positive = airier/breathier voice. Negative = clearer/more resonant. Ignored if preset != custom.",
                }),
                "brightness": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Secondary voice-quality cue: spectral tilt. Positive = brighter/tenser (high-arousal emotions skew brighter). Negative = darker/softer. Ignored if preset != custom.",
                }),
                "roughness": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Secondary voice-quality cue: frame-level amplitude jitter (shimmer/rasp). Ignored if preset != custom.",
                }),
                "vibrato_depth_semitones": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Secondary voice-quality cue: depth of a slow sinusoidal pitch wobble. Ignored if preset != custom.",
                }),
                "vibrato_rate_hz": ("FLOAT", {
                    "default": 5.0, "min": 1.0, "max": 8.0, "step": 0.1,
                    "tooltip": "Speed of the vibrato wobble. Ignored if preset != custom (and irrelevant if vibrato_depth_semitones is 0).",
                }),
                "micro_jitter_semitones": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.5, "step": 0.02,
                    "tooltip": "Adds a small amount of natural-sounding random pitch flutter. Mainly useful when pitch_range_scale is pushed far from 1.0 and the result starts sounding too clean/robotic. Ignored if preset != custom.",
                }),
                "intensity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blends every parameter above toward its neutral value - 0.0 is the original audio essentially unchanged, 1.0 is the full effect. Use this to dial a preset back if it's too strong.",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "color"
    CATEGORY = "Mykee/Audio"

    def color(
        self,
        audio,
        preset,
        tempo_scale,
        rate_variability,
        pitch_shift_semitones,
        pitch_range_scale,
        energy_range_scale,
        pause_scale,
        breathiness,
        brightness,
        roughness,
        vibrato_depth_semitones,
        vibrato_rate_hz,
        micro_jitter_semitones,
        intensity,
    ):
        _require_pyworld()

        if preset != "custom":
            presets = _load_mood_presets()
            p = presets.get(preset)
            if p is not None:
                tempo_scale = p.get("tempo_scale", tempo_scale)
                rate_variability = p.get("rate_variability", rate_variability)
                pitch_shift_semitones = p.get("pitch_shift_semitones", pitch_shift_semitones)
                pitch_range_scale = p.get("pitch_range_scale", pitch_range_scale)
                energy_range_scale = p.get("energy_range_scale", energy_range_scale)
                pause_scale = p.get("pause_scale", pause_scale)
                breathiness = p.get("breathiness", breathiness)
                brightness = p.get("brightness", brightness)
                roughness = p.get("roughness", roughness)
                vibrato_depth_semitones = p.get("vibrato_depth_semitones", vibrato_depth_semitones)
                vibrato_rate_hz = p.get("vibrato_rate_hz", vibrato_rate_hz)
                micro_jitter_semitones = p.get("micro_jitter_semitones", micro_jitter_semitones)

        # intensity blends every control toward its neutral value up front,
        # rather than crossfading waveforms afterward - tempo/pause changes
        # alter duration, so a sample-domain dry/wet blend wouldn't line up.
        eff_tempo = 1.0 + (tempo_scale - 1.0) * intensity
        eff_rate_var = rate_variability * intensity
        eff_pitch_shift = pitch_shift_semitones * intensity
        eff_pitch_range = 1.0 + (pitch_range_scale - 1.0) * intensity
        eff_micro_jitter = micro_jitter_semitones * intensity
        eff_energy_range = 1.0 + (energy_range_scale - 1.0) * intensity
        eff_pause_scale = 1.0 + (pause_scale - 1.0) * intensity
        eff_breathiness = breathiness * intensity
        eff_brightness = brightness * intensity
        eff_roughness = roughness * intensity
        eff_vibrato_depth = vibrato_depth_semitones * intensity

        waveform = audio["waveform"]
        sr = audio["sample_rate"]
        batch, channels, _ = waveform.shape

        out_channels = []
        max_len = 0
        per_channel_results = []

        for b in range(batch):
            ch_results = []
            for c in range(channels):
                dry = waveform[b, c].detach().cpu().numpy().astype(np.float64)

                if np.max(np.abs(dry)) < 1e-9 or intensity <= 0.0:
                    ch_results.append(dry.astype(np.float32))
                    max_len = max(max_len, dry.shape[0])
                    continue

                f0, sp, ap = _world_analyze(dry, sr)
                n_frames, n_bins = sp.shape

                new_n_frames, src_idx = _build_variable_warp(
                    n_frames, sp, eff_tempo, eff_rate_var, eff_pause_scale, _FIXED_NOISE_SEED
                )
                f0 = _warp_resample(f0, src_idx)
                sp = _warp_resample(sp, src_idx)
                ap = _warp_resample(ap, src_idx)
                n_frames = new_n_frames

                voiced = f0 > 0.0
                if np.any(voiced):
                    logf0 = np.log(f0[voiced])
                    baseline = np.median(logf0)
                    new_logf0 = baseline + (logf0 - baseline) * eff_pitch_range
                    new_logf0 += eff_pitch_shift * _SEMITONE
                    if eff_micro_jitter > 0.0:
                        jitter = _smoothed_noise(int(voiced.sum()), _FIXED_NOISE_SEED) * eff_micro_jitter * _SEMITONE
                        new_logf0 += jitter
                    f0_new = np.zeros_like(f0)
                    f0_new[voiced] = np.exp(new_logf0)
                    f0 = f0_new

                if abs(eff_brightness) > 1e-6 or abs(eff_breathiness) > 1e-6 or eff_roughness > 1e-6:
                    presence = _frame_presence(sp)

                if abs(eff_brightness) > 1e-6:
                    bin_norm = np.linspace(-1.0, 1.0, num=n_bins)
                    tilt_db = eff_brightness * 8.0 * bin_norm
                    tilt_gain = np.power(10.0, tilt_db / 20.0)[np.newaxis, :]
                    gated_gain = 1.0 + (tilt_gain - 1.0) * presence[:, np.newaxis]
                    sp = sp * gated_gain

                if abs(eff_breathiness) > 1e-6:
                    weight = np.linspace(0.2, 1.0, num=n_bins)
                    delta = eff_breathiness * 0.35 * weight[np.newaxis, :] * presence[:, np.newaxis]
                    ap = np.clip(ap + delta, 1e-6, 1.0 - 1e-6)

                if eff_roughness > 1e-6:
                    shimmer = 1.0 + eff_roughness * 0.25 * _smoothed_noise(n_frames, _FIXED_NOISE_SEED)
                    shimmer = np.clip(shimmer, 0.5, 1.5)
                    gated_shimmer = 1.0 + (shimmer - 1.0) * presence
                    sp = sp * gated_shimmer[:, np.newaxis]

                if eff_vibrato_depth > 1e-6:
                    voiced = f0 > 0.0
                    if np.any(voiced):
                        t_frames = np.arange(n_frames) * (_FRAME_PERIOD_MS / 1000.0)
                        vib = eff_vibrato_depth * np.sin(2.0 * np.pi * vibrato_rate_hz * t_frames)
                        logf0 = np.log(f0[voiced])
                        logf0 += vib[voiced] * _SEMITONE
                        f0_new = f0.copy()
                        f0_new[voiced] = np.exp(logf0)
                        f0 = f0_new

                y = _world_synthesize(f0, sp, ap, sr)

                if abs(eff_energy_range - 1.0) > 1e-6 and y.size > 0:
                    gain_full = _energy_expand_gain(y, sr, eff_energy_range)
                    y = (y.astype(np.float64) * gain_full).astype(np.float32)

                ch_results.append(y)
                max_len = max(max_len, y.shape[0])

            per_channel_results.append(ch_results)

        for ch_results in per_channel_results:
            fitted = [_fit_np_length(y, max_len) for y in ch_results]
            out_channels.append(np.stack(fitted, axis=0))

        out_np = np.stack(out_channels, axis=0)
        out_tensor = torch.from_numpy(out_np).to(device=waveform.device, dtype=waveform.dtype)
        out_tensor = _peak_safe_scale(out_tensor)

        return ({"waveform": out_tensor, "sample_rate": sr},)


NODE_CLASS_MAPPINGS = {
    "MykeeEmotionTimbre": MykeeEmotionTimbre,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeEmotionTimbre": "Mykee Emotion Timbre",
}
