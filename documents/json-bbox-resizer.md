# Mykee JSON BBOX Resizer

[← Back to the main README](../README.md)  ·  Category: `Mykee/Prompt`

Re-maps the bounding boxes of a MOSS-Image layout JSON (the `json` output of
[Mykee JSON Captioner](json-captioner.md): `abs_centralized` / `yxyx`
integers) from the canvas it was made for to another canvas. Use it when the
image you generate has a different size than the image that was captioned.
Only each item's `bbox` changes; captions, text and the rest of the JSON stay
as they are.

## Inputs

All fields can also be fed from a connected node.

- **json_text** - the layout JSON. Markdown fences and surrounding chatter
  are tolerated. Other box formats (not `abs_centralized` / `yxyx`) are
  rejected.
- **original_width, original_height** - the canvas the boxes were made for
  (the captioner's `width` / `height` outputs).
- **target_width, target_height** - the size of the image to generate
  (MOSS-Image needs multiples of 16; the node warns in the console otherwise).
- **mode** - matters only when the aspect ratio differs:
  - `stretch` (default): x and y are scaled separately, the layout fills the
    target canvas.
  - `contain`: one uniform scale, the whole layout fits and is centred.
  - `cover`: one uniform scale, the target is filled; the layout is centred
    and cropped, boxes are clamped to the canvas and items that end up
    completely outside are dropped.
- **pretty_json** - indented output instead of one compact line.

## Display

The node shows the resulting JSON in a read-only box at its bottom, with a
**Copy JSON** button under it (it stays after a workflow tab switch). With a
batch from the captioner the node runs once per JSON and the box lists them
all, separated by a blank line.

## Outputs

- **json** - the layout JSON with the re-mapped boxes.
- **target_width, target_height** - the target size, to wire into the
  generation (empty latent / MOSS-Image size).

## How it calculates

Centred integers are turned back into top-left pixels (adding
`width // 2`, `height // 2`), scaled, fractional edges are enclosed (floor
for the min edge, ceil for the max edge, the same as MOSS's
`compile_prompt.py`) and centred again on the target canvas. Boxes are kept
inside the canvas; an item whose box has no area left is dropped and
reported in the console.
