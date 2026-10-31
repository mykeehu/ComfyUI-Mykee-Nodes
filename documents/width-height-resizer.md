# Mykee Width/Height Resizer

[← Back to the main README](../README.md)  ·  Category: `Mykee/Image`

Scales an incoming width / height pair up or down and returns the new pair.
Both sides get the same factor, so the aspect ratio is kept (only the optional
rounding to 8 / 16 / 32 / 64 can shift it by a few pixels).

Inputs:

- **width**, **height** - the incoming size. Type the numbers, or connect them
  from another node (e.g. an image-size node or
  [Mykee Size Switch/Batch](size-switch-batch.md)).
- **image** (optional) - when an image is connected, the width / height are
  read from it and the **width** / **height** fields are greyed out and
  ignored. Only the size of the image is used (for a batch, the size of the
  batch).
- **mode**:
  - **scale by** - both sides are multiplied by **scale_by** (decimals allowed:
    0.5 = half size, 1.5 = one and a half times).
  - **scale to** - the **reference** side is set to **scale_to** pixels and the
    other side follows. Example: reference `longest side`, scale_to 1024, a
    1920x1080 input gives 1024x576.
- **reference** (scale to only) - what scale_to measures: `longest side`
  (default), `shortest side`, `width` or `height`.
- **scale_by** / **scale_to** - only the one that belongs to the chosen mode is
  shown.
- **divisible_by** - `Off`, 8, 16, 32 or 64. The result is rounded to the nearest
  multiple (half up), and is never smaller than the divisor itself. The
  rounding is done per side, so the ratio can differ by a few pixels (e.g.
  1000x700 scaled to 512 on the longest side gives 512x358 with `Off`, but
  512x384 with 64).

Outputs:

- **width**, **height** - the scaled size (INT).

**Preview:** a small box at the bottom of the node shows the output size. It
updates live while you type or change a setting. When width / height arrive
through a connection their value is not known in the browser, so the box shows
the size of the last run instead (it is updated after every run). If the
inputs are a list (e.g. batch mode of Size Switch/Batch), the node runs once
per entry and the box lists every result.

The result is capped at 65536 pixels per side.
