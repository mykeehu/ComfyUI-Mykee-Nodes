# Mykee - KSampler with Sigmas

[← Back to the main README](../README.md)  ·  Category: `Mykee/Sampling`

The core ComfyUI **KSampler** with two extra, optional ways to give it a
noise schedule: the **sigmas** input socket and the **sigmas_text** field
where the values can simply be typed in. It lets you plug any sigma schedule (`BasicScheduler`, `KarrasScheduler`,
`ExponentialScheduler`, `AlignYourStepsScheduler`, `SplitSigmas`,
`SetFirstSigma`, ...) into a sampler that otherwise looks and works exactly
like the KSampler, with no need to rebuild the workflow around
`SamplerCustom`.

## How it works

The **`Switch_to_sigmas`** toggle at the bottom of the node (OFF by default)
picks where the noise schedule comes from. This way the sigma settings can
stay on the node while you use it with a model that needs the normal
scheduler: just flip the switch, nothing has to be deleted.

- **OFF**: the node *is* the KSampler. The schedule is built from `steps`,
  `scheduler` and `denoise`, the same way as in core. `sigmas` and
  `sigmas_text` are ignored (`sigmas_text` is greyed out).
- **ON**: `steps` and `scheduler` are **greyed out** and ignored. The
  schedule comes from:
  1. the **`sigmas`** input, if it is connected (then `sigmas_text` is greyed
     out too), otherwise
  2. the **`sigmas_text`** field: typed values separated by commas, spaces or
     semicolons, for example `14.61, 7.49, 3.08, 1.03, 0.0`. A pasted list in
     brackets, or a printed `tensor([...])`, works too. The field can be
     connected as well.

  ON with neither of them is an error, with a message saying what is missing.
  The number of steps is the length of the schedule minus one. A typed
  schedule should normally end with `0.0` for a fully denoised result.
- `sampler_name`, `seed` and `cfg` are always used.

## Denoise with sigmas (img2img refinement)

How much noise is added to the input image depends only on the **first
sigma** of the schedule: a lower start keeps more of the image. With
`Switch_to_sigmas` ON, `denoise` stays active and lowers the start of the
connected / typed schedule. At `1.0` the schedule is used unchanged. How
the rest of the schedule follows is set by **`denoise_mode`** (greyed out
while the switch is OFF):

- **keep steps** (default): the number of steps stays the same, spread over
  the lower part of the schedule curve. This is what the core KSampler's
  denoise does (it builds a longer schedule of `int(steps / denoise)` steps
  and runs only its last `steps` steps); here the longer schedule is made by
  resampling the given curve. For exponential schedules the result is the
  same as the KSampler's, for Karras within a fraction of a percent. Usually
  the better choice for refinement: even a small denoise gets enough steps
  for the details.
- **cut**: the first steps of the schedule are dropped, so fewer steps run
  (20 steps at denoise 0.3 = the last 6 steps). Same as the core
  `SplitSigmasDenoise` (its `low_sigmas` output), and the same idea as
  `start_at_step` of KSamplerAdvanced. Faster, but at a small denoise very
  few steps are left.

Denoise `0` means nothing is sampled: the latent is passed through unchanged.
If the connected schedule was already shortened (for example by
`BasicScheduler`'s own denoise or `SplitSigmasDenoise`), leave this
`denoise` at `1.0`, otherwise the start is lowered twice.

## Inputs

| Input | Type | Notes |
|---|---|---|
| model | MODEL | |
| seed | INT | With the usual "control after generate" combo. |
| steps | INT | Ignored (greyed out) while `Switch_to_sigmas` is ON. |
| cfg | FLOAT | |
| sampler_name | combo | Always used. |
| scheduler | combo | Ignored (greyed out) while `Switch_to_sigmas` is ON. |
| positive / negative | CONDITIONING | |
| latent_image | LATENT | |
| denoise | FLOAT | Switch OFF: as in the KSampler. Switch ON: applied to the sigma schedule (see `denoise_mode`). |
| sigmas | SIGMAS (optional) | A ready-made noise schedule. Used when the switch is ON; wins over `sigmas_text`. |
| sigmas_text | STRING (optional) | Typed sigma values. Used when the switch is ON and `sigmas` is not connected. Can be connected too. |
| denoise_mode | combo | `keep steps` (default) / `cut`; only with the switch ON. |
| Switch_to_sigmas | BOOLEAN | OFF (default) = steps / scheduler / denoise; ON = sigmas / sigmas_text. |

**Output:** `LATENT`, same as the KSampler.

## Notes

- With `BasicScheduler` (same model / scheduler / steps / denoise) connected
  to `sigmas`, the result is identical to the plain KSampler. The one
  exception is the `dpm_2`, `dpm_2_ancestral`, `uni_pc` and `uni_pc_bh2`
  samplers: for these, the KSampler quietly drops the second-to-last sigma
  from its own schedule, but a connected schedule is always used exactly as
  given (the same as in `SamplerCustom`).
- A schedule with fewer than two values (for example the empty half of a
  `SplitSigmas`) does no sampling: the latent is passed through unchanged.
- A schedule with negative, NaN or infinite values, a non-number in
  `sigmas_text`, or a schedule longer than 10000 steps is refused with an
  error message.
- The sampling runs through ComfyUI's own code (`comfy.sample.sample`), so
  latent previews, the progress bar, noise masks (inpainting) and batch
  indexes work as in the KSampler. Based on `common_ksampler()` from
  ComfyUI's `nodes.py` (GPL-3.0).
