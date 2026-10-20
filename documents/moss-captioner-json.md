# Mykee MOSS Captioner (JSON)

[← Back to the main README](../README.md)  ·  Category: `Mykee/Prompt`

Captions images for [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)
(training datasets, or prompts for generation) and returns the result as
the **structured JSON of MOSS's own recaptioning pipeline**:

```json
{"caption": {"long": "...", "medium": "...", "short": "..."},
 "watermark": [{"box_2d": [y_min, x_min, y_max, x_max], "label": "..."}],
 "borders": ["none"]}
```

The built-in prompt is the "Caption Prompt" from the MOSS-Image user guide
(Post-Training / Image Captioning). `box_2d` is normalized 0-1000,
`[y_min, x_min, y_max, x_max]`, relative to the full image. MOSS recommends
Qwen3.5 (27B / 35B-A3B) or Gemma 4 for this and says that **thinking**
gives more accurate captions.

What it does on top of a plain Prompt Advanced call:

- **Validates the answer** against that schema (Markdown fences and trailing
  chatter are tolerated). A malformed answer is regenerated with a different
  seed, up to **retries** times.
- Splits the result into separate outputs (long / medium / short caption,
  watermark list, borders).
- **Batches:** every image of an IMAGE batch is captioned in turn with the
  clip loaded once. All outputs are lists, so they map onto a
  save-text-per-item node.
- **Downscales** big images to **image_megapixels** first. ComfyUI feeds a VL
  model the full-resolution image (up to ~12.8 MP), which is slow and costs a
  lot of tokens and VRAM. The `max_length` budget counts the remaining image
  tokens automatically.

## Inputs

- **clip, image** - a generation-capable VL clip (for example Qwen3.5-9B
  through the normal Load CLIP node) and the image(s).
- **reference** - optional original caption/tags. The prompt treats it as a
  hint and corrects whatever the image does not support. Empty = none.
- **caption_prompt** - empty = the built-in MOSS prompt. Fill it in to
  replace it entirely (keep the JSON schema, otherwise parsing fails);
  `{{reference}}` marks where the reference text goes.
- **image_megapixels** - downscale limit, 0 = send the image as it is.
- **thinking** - lets Qwen3.5 reason first. Raise **max_length**: the
  thinking text uses the same budget.
- **max_length, retries, sampling_mode, seed, temperature, top_k, top_p,
  min_p, repetition_penalty, presence_penalty** - generation controls. The
  defaults follow MOSS's own captioning setup (temperature 1.0, top_p 0.95).
- **pretty_json** - indented JSON instead of one compact line.
- **unload_after_run** - frees `clip` from VRAM when the batch is done.

## Outputs

All outputs are lists (one entry per image): **json**, **caption_long**,
**caption_medium**, **caption_short**, **watermark** (JSON list as text),
**borders** (JSON list as text). If an image cannot be captioned, `json`
holds `[MOSS captioning failed: reason]`, the other outputs are empty, a
toast appears and the console shows the last raw answer.

## Note on Qwen3.5 and "thinking"

Qwen3.5 thinks by default. ComfyUI only inserts the `<think>` control text
when it applies its own chat template, which the Prompt Advanced nodes and
this node skip (they build the chat text themselves). For Qwen3.5 clips they
therefore add the same prefix ComfyUI's template would: an empty
`<think></think>` block when `thinking` is off, an open `<think>` when it is
on. Other models are left untouched. If the model runs out of `max_length`
while still thinking, the response is reported as such instead of being
silently empty.
