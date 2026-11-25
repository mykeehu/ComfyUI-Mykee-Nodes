# Mykee - Sigmas Editor

[← Back to the main README](../README.md)  ·  Category: `Mykee/Sampling`

A graphical editor for a sigma (noise) schedule. Draw the curve with the
mouse, type the exact values, or generate it from the usual scheduler
formulas, then feed it to `SamplerCustom`, to
[Mykee - KSampler with Sigmas](ksampler-with-sigmas.md) or to anything else
that takes `SIGMAS`.

## What a sigma schedule is

Each value is the amount of noise at one point of the sampling. For flow
models (Flux, Qwen Image, Wan, SD3, ...) it is literally the noise share:
`x = (1 - σ) · image + σ · noise`, so `σ = 0.55` means 55 % noise. The
sampler walks from the first value down to the last one (normally `0`), one
step per gap. SD 1.5 / SDXL use a different scale (about 14.61 down to
0.03), but the idea is the same.

The **first value** decides how much of an input image changes (img2img /
refinement): lower = more of the image is kept. The **spacing** of the
following values decides where the steps are spent: dense at high noise =
composition, dense at low noise = details.

## The editor

- **Points are the steps.** There are `steps + 1` points, evenly spaced on
  the x axis (x = step index, y = sigma).
- **Drag** a point up or down. It always stays between its neighbours, so the
  curve keeps decreasing (each value at least 0.000001 below the previous
  one). Hovering shows the exact value.
- **Double-click** a point to delete it; double-click between two points to
  insert a new one there (in the middle, geometrically). `steps` follows.
- **Value field** under the chart: all values, comma separated, with 6
  decimals. Type or paste into it (commas, spaces or semicolons; a list in
  brackets or a printed `tensor([...])` also works). A valid list is taken
  over by the chart right away (the number of values sets `steps`); an
  invalid one is shown in red with the reason, and leaving the field puts
  the current curve back. The field's height can be dragged; it is kept.
- **steps** widget: changing it **resamples** the current curve with a
  monotone cubic spline (PCHIP), so the shape is kept, the first and last
  values stay, and the curve still decreases.
- **sigma_max**: the top of the chart and of dragging. `1.0` for flow models,
  about `14.61` for SD 1.5 / SDXL. It is raised automatically when the values
  (typed, generated or loaded) go above it.
- **log_view**: logarithmic y axis, handy for the SD 1.5 / SDXL range
  (display only).
- The colours follow the ComfyUI theme (dark or light), also when the theme
  is switched.

## Generator

The row under the chart replaces the curve with a formula, using the current
`steps`. Only the fields the chosen curve uses are shown.

| Curve | Uses | Formula |
|---|---|---|
| flow shift | shift, start | `t` runs evenly to 0, `σ = s·t / (1 + (s−1)·t)` - the flow models' own schedule. With `start` below 1 the curve begins lower and keeps the step count, the same as the KSampler's denoise. |
| flux base/max | base, max, start, **width, height** | The same flow curve, but the shift is computed from the resolution like ComfyUI's `ModelSamplingFlux` node: `base` at 256 image tokens, `max` at 4096 tokens (linear in between and beyond, tokens = width × height / 256), shift = e^mu. For example base 0.5 / max 0.9 gives shift 2.46 at 1024×1024 and 3.29 at 1328×1328. |
| Karras | rho, start, min | Evenly spaced in `σ^(1/ρ)` from `start` to `min`, then 0 (ρ = 7 is the usual value). |
| exponential | start, min | Evenly spaced in `log σ` from `start` to `min`, then 0. |
| linear | start | Straight line from `start` to 0. |

**start** always shows the first point of the current curve, so Generate
keeps where the curve starts: set the first point to 0.7 (img2img) and
Generate builds the rest of the steps from 0.7 down. Type another value into
`start` to begin elsewhere. For flow / flux curves `start` is limited to 1.

### width / height and the "auto" flux curve

`width` and `height` are node widgets, so they can be **connected** (e.g. to
the image size). They are greyed out unless the flux curve is selected.

A flux curve stays **auto** after Generate until you edit it (drag, type,
insert / delete, Load from input, or another Generate). While it is auto:

- changing `steps`, `width` or `height` on the node regenerates it at once
  (instead of resampling),
- if `width` / `height` are connected, their values are only known at run
  time: the backend then regenerates the curve with the real size, outputs
  that, and sends it back to the editor. The info line shows
  `auto: 1328×1328, shift 3.29`.

Any edit **freezes** the curve: from then on the output is exactly the values
on the node, whatever the size.

## Inputs and outputs

| | Type | Notes |
|---|---|---|
| sigmas (input, optional) | SIGMAS | A scheduler's sigmas. **Not passed through**: after a run, the **Load from input (N)** button copies them into the editor, so they can be edited further. |
| sigmas (output) | SIGMAS | The curve on the node. |
| sigmas_text (output) | STRING | The same values as text (6 decimals), e.g. for `sigmas_text` of Mykee - KSampler with Sigmas. |

**Copy** puts the values on the clipboard.

## Notes

- At least 1 step (2 values), at most 500 steps. Negative, NaN or increasing
  values are refused by the backend as well, with an error message.
- The output is always exactly the values in the field (rounded to 6
  decimals).
