"""
ComfyUI-Mykee-Nodes / audio nodes
"""

import numpy as np
import torch

try:
    import torchaudio
    _HAS_TORCHAUDIO = True
except ImportError:
    _HAS_TORCHAUDIO = False


def _resample(waveform, orig_sr, target_sr):
    if orig_sr == target_sr:
        return waveform
    if _HAS_TORCHAUDIO:
        return torchaudio.functional.resample(waveform, orig_sr, target_sr)
    n_samples = waveform.shape[-1]
    new_n = max(1, int(round(n_samples * target_sr / orig_sr)))
    return torch.nn.functional.interpolate(
        waveform, size=new_n, mode="linear", align_corners=False
    )


def _match_channels(waveform, target_channels):
    channels = waveform.shape[1]
    if channels == target_channels:
        return waveform
    if channels == 1 and target_channels > 1:
        return waveform.repeat(1, target_channels, 1)
    if channels > 1 and target_channels == 1:
        return waveform.mean(dim=1, keepdim=True)
    if channels < target_channels:
        reps = (target_channels + channels - 1) // channels
        return waveform.repeat(1, reps, 1)[:, :target_channels, :]
    return waveform[:, :target_channels, :]


def _fit_length(waveform, target_len, loop):
    cur_len = waveform.shape[-1]
    if cur_len == target_len:
        return waveform
    if cur_len > target_len:
        return waveform[..., :target_len]
    if cur_len == 0:
        return torch.zeros(*waveform.shape[:-1], target_len, dtype=waveform.dtype, device=waveform.device)
    if loop:
        reps = (target_len + cur_len - 1) // cur_len
        tiled = waveform.repeat(1, 1, reps)
        return tiled[..., :target_len]
    pad = target_len - cur_len
    return torch.nn.functional.pad(waveform, (0, pad))


def _db_to_lin(db):
    return np.power(10.0, np.asarray(db) / 20.0)


def _block_rms(mono_np, hop):
    n = mono_np.shape[0]
    n_blocks = max(1, n // hop)
    trimmed = mono_np[: n_blocks * hop]
    if trimmed.size == 0:
        return np.zeros(1, dtype=np.float64)
    blocks = trimmed.reshape(n_blocks, hop)
    return np.sqrt(np.mean(blocks.astype(np.float64) ** 2, axis=1) + 1e-12)


def _envelope_follow(curve, attack_coef, release_coef):
    out = np.zeros_like(curve)
    prev = 0.0
    for i, v in enumerate(curve):
        coef = attack_coef if v > prev else release_coef
        prev = coef * prev + (1.0 - coef) * v
        out[i] = prev
    return out


def _upsample_curve(curve, n_target):
    n_src = curve.shape[0]
    if n_src == n_target:
        return curve
    if n_src == 1:
        return np.full(n_target, curve[0])
    x_src = np.linspace(0.0, 1.0, num=n_src)
    x_tgt = np.linspace(0.0, 1.0, num=n_target)
    return np.interp(x_tgt, x_src, curve)


def _block_peak(mono_np, hop):
    n = mono_np.shape[0]
    n_blocks = max(1, n // hop)
    trimmed = mono_np[: n_blocks * hop]
    if trimmed.size == 0:
        return np.zeros(1, dtype=np.float64)
    blocks = trimmed.reshape(n_blocks, hop)
    return np.max(np.abs(blocks), axis=1)


def _spectral_cleanup(waveform, sample_rate, hum_fundamental_hz=50.0, hum_harmonics=6,
                       hum_notch_width_hz=4.0, low_cut_hz=70.0,
                       high_shelf_freq_hz=4000.0, high_shelf_gain_db=6.0,
                       denoise_strength=0.6, denoise_floor=0.08,
                       n_fft=2048, hop_length=512):
    """
    One STFT-based pass over `waveform` (shape [C, S]) that:
      - notches out AC mains hum (hum_fundamental_hz + its harmonics) -
        e.g. a laptop power supply or an ungrounded cable picked up by a
        non-dedicated/room mic
      - gently high-passes below low_cut_hz (room rumble, handling noise)
      - applies a high-shelf boost above high_shelf_freq_hz to restore
        some of the "air"/presence a distant or off-axis mic loses
        (muffled/dull tonal balance)
      - reduces steady background noise via spectral subtraction: a noise
        magnitude profile is estimated from the quietest ~20% of frames
        (median, per frequency bin) and subtracted from every frame,
        floored so it never mutes audio entirely (avoids "musical noise"
        artifacts from over-aggressive subtraction)

    All four fixes are folded into ONE real-valued, non-negative
    frequency-domain gain curve per channel, which is multiplied directly
    into that channel's own complex STFT (magnitude scales, phase is
    untouched) - so stereo imaging/phase relationships between channels
    are preserved even though the hum/rumble/shelf shaping is derived
    from a channel-averaged analysis and the same for every channel, and
    the noise-reduction gain varies over time but is applied identically
    across channels too. Set any of hum_harmonics/low_cut_hz/
    high_shelf_gain_db/denoise_strength to 0 to skip that particular fix.
    """
    channels, n_samples = waveform.shape
    window = torch.hann_window(n_fft, dtype=waveform.dtype, device=waveform.device)
    specs = [
        torch.stft(waveform[c], n_fft=n_fft, hop_length=hop_length, window=window,
                   return_complex=True, center=True)
        for c in range(channels)
    ]
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate).to(waveform.device)
    n_bins, n_frames = specs[0].shape

    static_gain = torch.ones(n_bins, dtype=waveform.dtype, device=waveform.device)
    if hum_fundamental_hz > 0 and hum_harmonics > 0:
        for h in range(1, hum_harmonics + 1):
            f0 = hum_fundamental_hz * h
            notch = 1.0 - 0.93 * torch.exp(-0.5 * ((freqs - f0) / (hum_notch_width_hz / 2.355)) ** 2)
            static_gain = static_gain * notch
    if low_cut_hz > 0:
        hp = torch.sigmoid((freqs - low_cut_hz) / (low_cut_hz * 0.25 + 1.0))
        static_gain = static_gain * hp
    if high_shelf_gain_db != 0:
        shelf_lin = 10 ** (high_shelf_gain_db / 20.0)
        ramp = torch.sigmoid((freqs - high_shelf_freq_hz) / (high_shelf_freq_hz * 0.3 + 1.0))
        shelf = 1.0 + (shelf_lin - 1.0) * ramp
        static_gain = static_gain * shelf

    if denoise_strength > 0:
        mono_spec = torch.stack(specs, dim=0).abs().mean(dim=0)  # [bins, frames]
        frame_energy = mono_spec.pow(2).mean(dim=0)
        n_noise_frames = max(1, int(0.2 * n_frames))
        noise_frame_idx = torch.argsort(frame_energy)[:n_noise_frames]
        noise_profile = mono_spec[:, noise_frame_idx].median(dim=1).values  # [bins]
        eps = 1e-8
        dyn_gain = 1.0 - denoise_strength * (noise_profile.unsqueeze(1) / (mono_spec + eps))
        dyn_gain = dyn_gain.clamp(min=denoise_floor, max=1.0)
    else:
        dyn_gain = torch.ones((n_bins, n_frames), dtype=waveform.dtype, device=waveform.device)

    total_gain = static_gain.unsqueeze(1) * dyn_gain  # [bins, frames]

    out_channels = []
    for c in range(channels):
        cleaned_spec = specs[c] * total_gain
        out = torch.istft(cleaned_spec, n_fft=n_fft, hop_length=hop_length, window=window,
                           length=n_samples, center=True)
        out_channels.append(out)
    return torch.stack(out_channels, dim=0)


def _ceiling_follow(ceiling_curve, tighten_coef, release_coef):
    """Smooths a per-block gain CEILING (not a signal level) with fast
    attack toward a TIGHTER (lower) ceiling and slow release back toward
    a LOOSER (higher) one - the mirror image of _envelope_follow's
    up-fast/down-slow logic, needed because here a lower value means
    'clamp harder'. Used so a single loud transient's momentary peak-
    safety restriction doesn't snap back open again immediately after
    (which would otherwise show up as an audible gain 'dip and pop back'
    exactly on that transient) - the ceiling eases back open gradually
    instead."""
    out = np.zeros_like(ceiling_curve)
    prev = ceiling_curve[0] if len(ceiling_curve) else 0.0
    for i, v in enumerate(ceiling_curve):
        coef = tighten_coef if v < prev else release_coef
        prev = coef * prev + (1.0 - coef) * v
        out[i] = prev
    return out


def _auto_level(waveform, sample_rate, target_db=-20.0, max_gain_db=24.0,
                 attack_ms=50.0, release_ms=400.0, peak_ceiling_db=-1.0):
    """
    Rides the gain of `waveform` (shape [C, S]) to bring it toward
    target_db RMS, smoothing with attack/release so it doesn't pump on
    every syllable - fixes level drift/wandering (a mic at inconsistent
    distance, or a speaker who moved around) without squashing natural
    word-to-word dynamics the way a fast compressor would.

    The RMS-based gain target is computed from the channel-AVERAGED
    signal (so the leveling responds to overall loudness, not one loud
    channel), but is then hard-capped, sample-accurately, against the
    TRUE peak across all channels combined (not averaged - averaging
    would let one channel's peak slip through underestimated and clip),
    so the output never exceeds peak_ceiling_db regardless of how much
    gain the RMS target alone would otherwise call for. That safety
    ceiling itself is smoothed with a near-instant tighten and a slow
    release (see _ceiling_follow) - without this, a single loud
    transient in an otherwise-quiet recording would force the ceiling
    down for just that one block and then let it spring back open on the
    very next block, producing an audible momentary dip-and-recover right
    on the transient instead of a clean, steady level. A final
    safety-normalize catches any remaining rounding-level overshoot.
    """
    channels, n_samples = waveform.shape
    control_hz = 50.0
    hop = max(1, int(round(sample_rate / control_hz)))
    attack_coef = float(np.exp(-1.0 / max(control_hz * (attack_ms / 1000.0), 1e-6)))
    release_coef = float(np.exp(-1.0 / max(control_hz * (release_ms / 1000.0), 1e-6)))
    # Ceiling-tighten reacts fast but not instantaneously (~30ms) - fast
    # enough that a sudden loud transient is still caught well before it
    # could clip, but not so fast that the whole reduction lands within a
    # single ~20ms control block, which would be an audible "cliff" (a
    # sharp gain notch right on the transient) rather than a quick but
    # smooth duck.
    ceiling_tighten_coef = float(np.exp(-1.0 / max(control_hz * 0.015, 1e-6)))
    ceiling_release_coef = float(np.exp(-1.0 / max(control_hz * (release_ms / 1000.0) * 1.5, 1e-6)))

    mono = waveform.abs().mean(dim=0).detach().cpu().numpy()
    peak_track = waveform.abs().max(dim=0).values.detach().cpu().numpy()

    rms = _block_rms(mono, hop)
    env = _envelope_follow(rms, attack_coef, release_coef)
    env_db = 20 * np.log10(env + 1e-9)
    gain_db = np.clip(target_db - env_db, -max_gain_db, max_gain_db)
    gain_db_smooth = _envelope_follow(gain_db, attack_coef, release_coef)

    peaks = _block_peak(peak_track, hop)
    ceiling_lin = 10 ** (peak_ceiling_db / 20.0)
    max_allowed_gain_db = 20 * np.log10(ceiling_lin / (peaks + 1e-9) + 1e-9)
    max_allowed_gain_db = _ceiling_follow(max_allowed_gain_db, ceiling_tighten_coef, ceiling_release_coef)
    gain_db_safe = np.minimum(gain_db_smooth, max_allowed_gain_db)

    gain_lin = _upsample_curve(10 ** (gain_db_safe / 20.0), n_samples)
    gain_t = torch.from_numpy(gain_lin).to(dtype=waveform.dtype, device=waveform.device)
    out = waveform * gain_t.unsqueeze(0)

    peak_final = out.abs().max()
    if peak_final > ceiling_lin:
        out = out * (ceiling_lin / peak_final)
    return out


def _trim_long_silences(tensor, sample_rate, max_silence_seconds=0.5,
                         silence_threshold_db=-40.0, min_run_seconds=0.02):
    """Caps any silent gap in `tensor` (shape [C, S]) longer than
    `max_silence_seconds` down to exactly that length - shorter pauses
    are left untouched. Ported from this pack's MOSS-TTS nodes' own
    trim_long_silences() helper (same algorithm, same defaults).

    Detects silence on a channel-averaged RMS energy envelope over short
    frames (~`min_run_seconds` long); a run of consecutive silent frames
    longer than `max_silence_seconds` is truncated to that length. Trims
    ALL channels together at the same points, so stereo imaging is
    preserved. `silence_threshold_db` is in dBFS relative to full scale
    ±1.0 - lower it (more negative) to only catch near-total silence,
    raise it (less negative) to also catch quiet room tone/breath noise.
    """
    if max_silence_seconds <= 0:
        return tensor

    mono = tensor.mean(dim=0)
    frame_len = max(1, int(min_run_seconds * sample_rate))
    n_frames = mono.shape[-1] // frame_len
    if n_frames < 2:
        return tensor  # too short to bother

    usable_len = n_frames * frame_len
    frames = mono[:usable_len].reshape(n_frames, frame_len)
    frame_rms = frames.pow(2).mean(dim=-1).sqrt()
    threshold_amp = 10 ** (silence_threshold_db / 20.0)
    is_silent = (frame_rms < threshold_amp).tolist()

    max_silent_frames = max(1, int(max_silence_seconds / min_run_seconds))

    # Boolean keep-mask per frame: every non-silent frame is kept; for
    # each silent run, only its first max_silent_frames are kept.
    keep_mask = [True] * n_frames
    run_start = None
    for i, silent in enumerate(is_silent + [False]):  # sentinel flush at the end
        if silent and run_start is None:
            run_start = i
        elif not silent and run_start is not None:
            run_len = i - run_start
            if run_len > max_silent_frames:
                for j in range(run_start + max_silent_frames, i):
                    keep_mask[j] = False
            run_start = None

    if all(keep_mask):
        return tensor  # nothing to trim - avoid the reconstruction cost

    # Reconstruct at the full (not frame-truncated) sample resolution, on
    # the original (all-channel) tensor, so the trailing remainder past
    # usable_len is preserved untouched.
    kept_frame_slices = []
    run_start = None
    for i, keep in enumerate(keep_mask + [False]):
        if keep and run_start is None:
            run_start = i
        elif not keep and run_start is not None:
            kept_frame_slices.append((run_start, i))
            run_start = None

    parts = [tensor[..., start * frame_len:end * frame_len] for start, end in kept_frame_slices]
    if usable_len < tensor.shape[-1]:
        parts.append(tensor[..., usable_len:])
    if not parts:
        return tensor[..., :1] * 0  # degenerate case: entire clip was silence
    return torch.cat(parts, dim=-1)


def _apply_silence_handles(tensor, sample_rate, head_seconds, tail_seconds):
    """Pads silence before/after `tensor` (shape [C, S]). Ported from this
    pack's MOSS-TTS nodes' own apply_handles() helper."""
    head_samples = int(head_seconds * sample_rate)
    tail_samples = int(tail_seconds * sample_rate)
    if head_samples <= 0 and tail_samples <= 0:
        return tensor

    def _silence(n_samples):
        shape = list(tensor.shape)
        shape[-1] = n_samples
        return torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)

    parts = []
    if head_samples > 0:
        parts.append(_silence(head_samples))
    parts.append(tensor)
    if tail_samples > 0:
        parts.append(_silence(tail_samples))
    return torch.cat(parts, dim=-1)


class MykeeSilenceRemover:
    """
    Caps long silent gaps inside an AUDIO clip down to a set maximum
    length, and optionally pads fresh silence onto the start/end. Useful
    for cleaning up dead air from any source (not just this pack's own
    TTS nodes) - a generated take with an overlong pause, a recording
    with excess room tone between phrases, etc.

    trim_long_silences finds runs of near-silence (by RMS energy, on a
    channel-averaged envelope so stereo imaging is preserved) longer than
    max_silence_seconds and cuts each down to exactly that length -
    shorter pauses are left completely untouched, and actual audio
    content is never trimmed. head_seconds/tail_seconds then pad fresh
    silence onto the very start/end of the (possibly now-shorter) result,
    e.g. to leave room for a fade-in or to match a required clip length.

    A batched AUDIO input is processed per-item (each clip trimmed on its
    own silence pattern); since trimming can leave clips at different
    lengths, the batch is re-padded with trailing silence to a common
    length afterward so it's still a valid single AUDIO output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Audio clip(s) to process.",
                }),
                "trim_long_silences": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cap any silent gap longer than 'max_silence_seconds' down to exactly that length. Shorter pauses are left untouched.",
                }),
                "max_silence_seconds": ("FLOAT", {
                    "default": 0.5, "min": 0.05, "max": 30.0, "step": 0.05,
                    "tooltip": "Only used when 'trim_long_silences' is on. Any silent gap longer than this is cut down to this length.",
                }),
                "silence_threshold_db": ("FLOAT", {
                    "default": -40.0, "min": -80.0, "max": -10.0, "step": 1.0,
                    "tooltip": "Only used when 'trim_long_silences' is on. Audio quieter than this (dBFS) counts as silence - lower (more negative) to only catch near-total silence, raise to also catch quiet room tone/breath noise.",
                }),
                "min_run_seconds": ("FLOAT", {
                    "default": 0.02, "min": 0.005, "max": 0.5, "step": 0.005,
                    "tooltip": "Only used when 'trim_long_silences' is on. Frame size used to detect silence - smaller catches shorter gaps but is more sensitive to brief dips (consonant stops, etc.).",
                }),
                "head_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 30.0, "step": 0.05,
                    "tooltip": "Seconds of silence to pad onto the START of the output audio (added after trimming).",
                }),
                "tail_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 30.0, "step": 0.05,
                    "tooltip": "Seconds of silence to pad onto the END of the output audio (added after trimming).",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "run"
    CATEGORY = "Mykee/Audio"

    def run(
        self,
        audio,
        trim_long_silences,
        max_silence_seconds,
        silence_threshold_db,
        min_run_seconds,
        head_seconds,
        tail_seconds,
    ):
        waveform = audio["waveform"]  # [B, C, S]
        sample_rate = audio["sample_rate"]

        processed = []
        for b in range(waveform.shape[0]):
            clip = waveform[b]  # [C, S]
            if trim_long_silences:
                clip = _trim_long_silences(
                    clip, sample_rate,
                    max_silence_seconds=max_silence_seconds,
                    silence_threshold_db=silence_threshold_db,
                    min_run_seconds=min_run_seconds,
                )
            clip = _apply_silence_handles(clip, sample_rate, head_seconds, tail_seconds)
            processed.append(clip)

        max_len = max(clip.shape[-1] for clip in processed)
        padded = [
            clip if clip.shape[-1] == max_len
            else torch.nn.functional.pad(clip, (0, max_len - clip.shape[-1]))
            for clip in processed
        ]
        out_waveform = torch.stack(padded, dim=0)

        return ({"waveform": out_waveform, "sample_rate": sample_rate},)


class MykeeAudioMerger:
    """
    Two-input audio mixer: audio_narrator (foreground) + audio_background
    (ambience/noise bed) -> one merged AUDIO output.

    Optional sidechain ducking lowers the background while the narrator is
    speaking. Optional "reactive narrator boost" (Lombard-effect style) does
    the reverse: it raises the narrator's level a little when the background
    gets louder (e.g. a wind gust) - gain-only, no pitch/formant shifting, so
    it doesn't touch the narrator's voice character/accent.

    audio_narrator's sample rate and channel layout are used as the output
    reference; audio_background is resampled/channel-matched to it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_narrator": ("AUDIO", {
                    "tooltip": "Primary/foreground track (e.g. the TTS narration). Its sample rate and channel layout are used as the output reference.",
                }),
                "audio_background": ("AUDIO", {
                    "tooltip": "Ambience / background noise track to mix under the narrator.",
                }),
                "background_gain_db": ("FLOAT", {
                    "default": -18.0, "min": -60.0, "max": 12.0, "step": 0.5,
                    "tooltip": "Base level of the background track relative to the narrator, before ducking.",
                }),
                "length_mode": (["match_narrator_length", "match_background_length", "match_longer_length"], {
                    "default": "match_narrator_length",
                    "tooltip": "Which track's length the merged output follows. The other track is looped or silence-padded to fit (see loop_shorter_track).",
                }),
                "loop_shorter_track": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If a track needs to be extended to reach the target length, loop it instead of padding with silence.",
                }),
                "duck_background": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically lower the background level while the narrator is speaking (sidechain ducking).",
                }),
                "duck_amount_db": ("FLOAT", {
                    "default": 8.0, "min": 0.0, "max": 40.0, "step": 0.5,
                    "tooltip": "Extra attenuation applied to the background while the narrator is active (on top of background_gain_db).",
                }),
                "reactive_narrator_boost": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Lombard-effect style reaction: raise the narrator's level a bit when the background gets louder (e.g. wind gust). Gain-only - does not shift pitch/formants.",
                }),
                "reactive_boost_db": ("FLOAT", {
                    "default": 4.0, "min": 0.0, "max": 12.0, "step": 0.5,
                    "tooltip": "Maximum extra gain applied to the narrator at the background's loudest moment (relative to that track's own peak).",
                }),
                "attack_ms": ("FLOAT", {
                    "default": 150.0, "min": 5.0, "max": 2000.0, "step": 5.0,
                    "tooltip": "How fast the ducking/reactive boost reacts to a rise in level.",
                }),
                "release_ms": ("FLOAT", {
                    "default": 600.0, "min": 5.0, "max": 5000.0, "step": 5.0,
                    "tooltip": "How fast the ducking/reactive boost relaxes back after the level drops.",
                }),
                "output_gain_db": ("FLOAT", {
                    "default": 0.0, "min": -24.0, "max": 24.0, "step": 0.5,
                    "tooltip": "Final trim applied to the merged output.",
                }),
                "limiter": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Soft-clip (tanh) the merged output to prevent digital clipping. Only audibly colors the sound near/above 0 dBFS peaks.",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "merge"
    CATEGORY = "Mykee/Audio"

    def merge(
        self,
        audio_narrator,
        audio_background,
        background_gain_db,
        length_mode,
        loop_shorter_track,
        duck_background,
        duck_amount_db,
        reactive_narrator_boost,
        reactive_boost_db,
        attack_ms,
        release_ms,
        output_gain_db,
        limiter,
    ):
        narrator_wave = audio_narrator["waveform"]
        narrator_sr = audio_narrator["sample_rate"]
        bg_wave = audio_background["waveform"]
        bg_sr = audio_background["sample_rate"]

        target_sr = narrator_sr

        bg_wave = _resample(bg_wave, bg_sr, target_sr)
        bg_wave = _match_channels(bg_wave, narrator_wave.shape[1])

        if narrator_wave.shape[0] != bg_wave.shape[0]:
            if narrator_wave.shape[0] == 1:
                narrator_wave = narrator_wave.repeat(bg_wave.shape[0], 1, 1)
            elif bg_wave.shape[0] == 1:
                bg_wave = bg_wave.repeat(narrator_wave.shape[0], 1, 1)
            else:
                min_b = min(narrator_wave.shape[0], bg_wave.shape[0])
                narrator_wave = narrator_wave[:min_b]
                bg_wave = bg_wave[:min_b]

        narrator_len = narrator_wave.shape[-1]
        bg_len = bg_wave.shape[-1]

        if length_mode == "match_narrator_length":
            target_len = narrator_len
        elif length_mode == "match_background_length":
            target_len = bg_len
        else:
            target_len = max(narrator_len, bg_len)

        narrator_wave = _fit_length(narrator_wave, target_len, loop_shorter_track)
        bg_wave = _fit_length(bg_wave, target_len, loop_shorter_track)

        device = narrator_wave.device
        dtype = narrator_wave.dtype
        batch = narrator_wave.shape[0]

        bg_base_lin = float(_db_to_lin(background_gain_db))

        control_hz = 200.0
        hop = max(1, int(round(target_sr / control_hz)))
        attack_coef = float(np.exp(-1.0 / max(control_hz * (attack_ms / 1000.0), 1e-6)))
        release_coef = float(np.exp(-1.0 / max(control_hz * (release_ms / 1000.0), 1e-6)))

        out_batches = []
        for b in range(batch):
            narrator_np = narrator_wave[b].detach().cpu().numpy()
            bg_np = bg_wave[b].detach().cpu().numpy()

            bg_gain_full = np.full(target_len, bg_base_lin, dtype=np.float64)
            narrator_gain_full = np.ones(target_len, dtype=np.float64)

            if duck_background or reactive_narrator_boost:
                narrator_mono = np.abs(narrator_np).mean(axis=0)
                bg_mono = np.abs(bg_np).mean(axis=0)

                narrator_rms = _block_rms(narrator_mono, hop)
                bg_rms = _block_rms(bg_mono, hop)

                narrator_env = _envelope_follow(narrator_rms, attack_coef, release_coef)
                bg_env = _envelope_follow(bg_rms, attack_coef, release_coef)

                n_peak = float(narrator_env.max()) if narrator_env.size else 0.0
                b_peak = float(bg_env.max()) if bg_env.size else 0.0
                narrator_norm = narrator_env / n_peak if n_peak > 1e-9 else narrator_env * 0.0
                bg_norm = bg_env / b_peak if b_peak > 1e-9 else bg_env * 0.0

                if duck_background:
                    duck_db_curve = -duck_amount_db * narrator_norm
                    duck_lin_curve = _db_to_lin(duck_db_curve)
                    duck_lin_full = _upsample_curve(duck_lin_curve, target_len)
                    bg_gain_full = bg_gain_full * duck_lin_full

                if reactive_narrator_boost:
                    boost_db_curve = reactive_boost_db * bg_norm
                    boost_lin_curve = _db_to_lin(boost_db_curve)
                    boost_lin_full = _upsample_curve(boost_lin_curve, target_len)
                    narrator_gain_full = narrator_gain_full * boost_lin_full

            mixed = (
                narrator_np * narrator_gain_full[np.newaxis, :]
                + bg_np * bg_gain_full[np.newaxis, :]
            )
            out_batches.append(mixed)

        mixed_np = np.stack(out_batches, axis=0)
        mixed_tensor = torch.from_numpy(mixed_np).to(device=device, dtype=dtype)
        mixed_tensor = mixed_tensor * float(_db_to_lin(output_gain_db))

        if limiter:
            mixed_tensor = torch.tanh(mixed_tensor)

        return ({"waveform": mixed_tensor, "sample_rate": target_sr},)


class MykeeStereoToMono:
    """
    Simple channel-count normalizer for an AUDIO clip: forces the output
    to mono or stereo regardless of whether the input already is mono or
    stereo.

    - convert_to_mono OFF - output is stereo. A mono input is duplicated
      to both channels; a stereo input passes through unchanged.
    - convert_to_mono ON - output is mono. A stereo input is downmixed
      (channel-averaged); a mono input passes through unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Audio to normalize to mono or stereo.",
                }),
                "convert_to_mono": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Off: output is stereo (a mono input is duplicated to both channels; a stereo input passes through unchanged). On: output is mono (a stereo input is downmixed; a mono input passes through unchanged).",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "run"
    CATEGORY = "Mykee/Audio"

    def run(self, audio, convert_to_mono):
        target_channels = 1 if convert_to_mono else 2
        out_waveform = _match_channels(audio["waveform"], target_channels)
        return ({"waveform": out_waveform, "sample_rate": audio["sample_rate"]},)


class MykeeAudioLeveling:
    """
    Fixes level drift/wandering in an AUDIO clip - e.g. a speaker who
    moved closer to/further from the mic mid-recording - and/or
    normalizes its overall peak loudness. Two independent stages, either
    can be switched off on its own:

    1. auto_level - rides the gain toward a consistent target RMS
       loudness, smoothed with attack/release so it doesn't pump on
       every syllable, with a hard peak-safety ceiling so it never
       introduces clipping regardless of how much gain the level target
       alone would otherwise call for.
    2. normalize - a final overall gain trim so the clip's loudest peak
       sits at exactly normalize_peak_db, for consistent headroom across
       multiple clips.

    Runs auto_level first (if on), then normalize (if on), so normalize
    sets the final overall ceiling on top of whatever auto_level already
    evened out. Kept as its own node (not folded into Mykee Audio DSP
    Cleanup) so it can go anywhere in the chain relative to this pack's
    other audio nodes.

    A batched AUDIO input is processed per-item.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Audio to level/normalize.",
                }),
                "auto_level": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Rides the gain to smooth out level drift/wandering (e.g. the speaker moving relative to the mic) toward a consistent target loudness. Has a hard peak-safety ceiling, so it never introduces clipping.",
                }),
                "target_level_db": ("FLOAT", {
                    "default": -20.0, "min": -40.0, "max": -6.0, "step": 1.0,
                    "tooltip": "Only used when 'auto_level' is on. Target RMS loudness (dBFS) the gain rides toward.",
                }),
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Applies a final overall gain trim so the clip's loudest peak sits at exactly 'normalize_peak_db' - for consistent headroom across multiple clips. Runs after auto_level, if both are on.",
                }),
                "normalize_peak_db": ("FLOAT", {
                    "default": -1.0, "min": -12.0, "max": 0.0, "step": 0.5,
                    "tooltip": "Only used when 'normalize' is on. Target peak level (dBFS) for the clip's loudest sample.",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "run"
    CATEGORY = "Mykee/Audio"

    def run(self, audio, auto_level, target_level_db, normalize, normalize_peak_db):
        waveform = audio["waveform"]  # [B, C, S]
        sample_rate = audio["sample_rate"]

        processed = []
        for b in range(waveform.shape[0]):
            clip = waveform[b]  # [C, S]

            if auto_level:
                clip = _auto_level(clip, sample_rate, target_db=target_level_db)

            if normalize:
                peak = clip.abs().max()
                if peak > 1e-9:
                    target_lin = 10 ** (normalize_peak_db / 20.0)
                    clip = clip * (target_lin / peak)

            processed.append(clip)

        out_waveform = torch.stack(processed, dim=0)
        return ({"waveform": out_waveform, "sample_rate": sample_rate},)


NODE_CLASS_MAPPINGS = {
    "MykeeAudioMerger": MykeeAudioMerger,
    "MykeeSilenceRemover": MykeeSilenceRemover,
    "MykeeStereoToMono": MykeeStereoToMono,
    "MykeeAudioLeveling": MykeeAudioLeveling,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeAudioMerger": "Mykee Audio Merger",
    "MykeeSilenceRemover": "Mykee Silence Remover",
    "MykeeStereoToMono": "Mykee Stereo To Mono",
    "MykeeAudioLeveling": "Mykee Audio Leveling",
}
