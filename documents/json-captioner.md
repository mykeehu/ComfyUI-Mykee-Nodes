# Mykee JSON Captioner

[← Back to the main README](../README.md)  ·  Category: `Mykee/Prompt`

Captions images for [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)
(training datasets, or prompts for generation) and returns a ready-to-use
MOSS prompt. Two **modes**:

**layout** - the MOSS Layout JSON: one summary caption plus a list of items
with bounding boxes.

```json
{"summary":{"caption":"..."},
 "items":[{"label":"object","caption":"a woman","bbox":[-117,-295,724,82]},
          {"label":"text","caption":"white sans","text":"HELLO","bbox":[-432,-576,-388,0]}],
 "bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}
```

The model only reports each box in the convention chosen by **coord_mode**
and **box_order** (default: normalized 0-1000; `[x_min, y_min, x_max, y_max]`
for Qwen-style clips, `[y_min, x_min, y_max, x_max]` for Gemma 4 - see
*Inputs*). The node converts whatever it gets to normalized 0-1000 first,
then scales the boxes to the generation canvas (**width** /
**height** outputs, multiples of 16) and converts them to centered integer
`yxyx` with the same arithmetic as MOSS's `compile_prompt.py` - the model
never calculates the centered numbers. Boxes that are empty or outside the
image are dropped. Use the **width/height outputs as the generation size**,
otherwise the boxes do not line up.

By default (**separate_caption** = "2 calls") the summary caption is written
by a first call with the official MOSS caption prompt, which gives a more
accurate text, and a second call finds the items. "1 call" does both at once
(faster, usually a rougher caption).

**tag_list** - optional tags (comma or new-line separated) to find and box.
With a tag list the node uses a dedicated grounding prompt: the model must
answer for **every** listed tag, in order, with an empty list when the tag is
not visible (or is not a boxable thing, like "naked" or "smiling"), so it
cannot drift back to boxing whatever it finds interesting. Several instances
of a tag get separate items; item captions start with the tag; the label is
`character` for person-like tags and `object` otherwise. Tags that were not
found are listed in the console. With **tag_mode** `strict` only the listed
tags get items, with `hint` the model may add other important items too. If
none of the tags is found the run counts as failed. Quality depends on the
model: common, clearly visible things work well, tiny details and crowds of
identical objects are weaker.

**caption** - a plain caption. The built-in prompt is the "Caption Prompt"
from the MOSS-Image user guide; it also reports watermarks and borders,
which come out as separate outputs.

In both modes the **caption_length** switch (long / medium / short) picks
the single description that is used: `summary.caption` in layout mode, the
caption in caption mode. MOSS recommends Qwen3.5 (27B / 35B-A3B) or Gemma 4
and says that **thinking** gives more accurate captions.

**caption_length = detailed** is a different kind of description: instead of
one of the MOSS JSON captions, the node makes one extra free-text call with
its own built-in system prompt and gets a very detailed description in
paragraphs (type/style/camera and summary, people and clothing, nudity and
explicit actions, poses and spatial relations, background/light/mood). It
follows **explicit_detail**: with the switch ON the full prompt is used
(five paragraphs, including the adult-content paragraph and all the
body-part rules); with it OFF the same prompt is loaded without the explicit
parts (four paragraphs, no nudity / genitalia rules). Your **reference** and
**tag_list** are passed as the user message, which the prompt tells the model
to work in. The detailed prompt is self-sufficient, so the explicit scan call
is skipped. In layout mode the detailed text becomes `summary.caption` and
is also used as the caption the item/box call builds on; in caption mode it
is the caption (an additional MOSS call still supplies the watermark and
borders outputs). A custom **system_prompt** replaces the built-in detailed
prompt: it is what runs for the description (its `{max_length}` and
`{{reference}}`-style placeholders are filled in), and the JSON formatting
and processing around it stay the node's own.

What it does on top of a plain Prompt Advanced call:

- **Validates the answer** (Markdown fences and trailing chatter are
  tolerated). A malformed answer is regenerated with a different seed, up to
  **retries** times.
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
- **mode, caption_length** - see above (long / medium / short / detailed).
- **explicit_detail** - on (default): the node first makes an extra **scan
  call** that looks only for adult / explicit content and names it plainly
  in anatomical language - exposed body parts with their size, shape and
  state, how the subjects and objects physically relate to each other,
  the pose of each subject, fluids. The scan (printed in the console) is then
  given to the caption call(s) as accurate facts that must all appear in the
  description, with the same words. The model also gets a short system
  message and closing reminder that set the plain-language register, and your
  reference / tags are used for the scan too. Costs one more generation per
  image; if the scan finds no adult content, nothing is added. Off: the image
  is captioned in the usual way. The relative scale and proportions of the
  subjects are described either way.
- **reference, reference_mode** - your notes about the image: what is in it,
  corrections to what the model tends to get wrong ("the big object is a
  penis, not a mushroom"), camera angle, shot type, style. `hint` (default) is MOSS's
  original behaviour: the reference is only a hint, anything the image does
  not support is corrected. `balanced`: the notes refine the
  description - they add what the image cannot show by itself and decide
  ambiguous cases - and are ignored only where the image clearly contradicts
  them. `guide`: the notes are ground truth, the model follows them even
  against its own impression. Applies to every prompt of the node. Tags are
  always turned into natural prose (never quoted like "1girl"). Empty = none.
- **tag_list, tag_mode** - tags separated by commas or new lines (type them or
  connect a text output, e.g. a tagger). Layout mode: the tags to find and
  box, see above. Caption mode: the tags are given to the model as confirmed
  content of the image and are worked into the description together with the
  reference - so you can use only a tag list, only a reference, or both.
- **tags_to_prompt** - on: the tag list is appended to the end of the caption
  text, after a blank line, as `Tags: tag1, tag2, ...`. Applies to the
  **caption** output and to the caption inside the **json** (`summary.caption`
  in layout mode). Does nothing when **tag_list** is empty. Off by default.
- **separate_caption** - layout mode: 2 calls (accurate caption) or
  1 call (default).
- **system_prompt** - empty = the built-in prompt of the chosen mode. Fill
  it in to replace it entirely; it is sent as the instruction text in front
  of the image, like MOSS does. Keep that mode's JSON schema, otherwise
  parsing fails (the final JSON format itself is always built by the node).
  Placeholders: `{{reference}}`, `{{tags}}`, and in layout mode
  `{{caption_rule}}`, `{{max_items}}`, `{{tag_rule}}`, `{{box_schema}}`, `{{box_rule}}`. In layout mode a
  custom prompt always uses 1 call.
- **canvas_megapixels** - size of the generation canvas for the boxes. 0
  (default) = the image's own size, sides rounded to multiples of 16;
  otherwise the image's aspect ratio at about this many MP.
- **max_items** - layout mode: the most items (boxes) the result may have;
  extra items are cut, and with a tag list the tags come first. 0 = no limit
  (a ceiling of 60 applies).
- **image_megapixels** - downscale limit for the model's input, 0 = as is.
- **thinking** - lets Qwen3.5 / Gemma 4 reason first. The thinking text uses
  the same token budget as the answer; **thinking_budget** (optional, default
  1024) adds room for it on top of **max_length**.
- **template_mode** (optional) - `auto` (default): a Gemma 4 clip gets
  ComfyUI's native chat template (text first, image after it); Qwen-style
  clips keep the hand-built MOSS ChatML text (text first, then the image).
  `native`: ComfyUI's template for any clip (it puts the image *before* the
  text). `legacy_chatml`: always the hand-built Qwen text (not usable with
  Gemma 4).
- **max_length, retries, sampling_mode, seed, temperature, top_k, top_p,
  min_p, repetition_penalty, presence_penalty** - generation controls
  (defaults follow MOSS's captioning setup).
- **coord_mode** (optional) - layout mode: the scale the model is asked to
  report boxes in. `normalized_0_1000` (default): relative to the image,
  what Qwen3-VL / Qwen3.5 / Gemma 4 use. `absolute_pixels`: pixels of the
  image the model sees, for Qwen2.5-VL style models; the image is snapped
  to multiples of 32 first so the pixel size given to the model is exact.
  Adapted from
  [ComfyUI-MOSS-QwenVL-Layout](https://github.com/leroy989/ComfyUI-MOSS-QwenVL-Layout)
  by leroy989 (MIT, see `THIRD_PARTY_NOTICES.md`). It does not make boxes
  more precise on its own: it only has to match what the model actually
  outputs.
- **box_order** (optional) - layout mode: the axis order the model is asked
  to report boxes in. `xyxy` = `[x_min, y_min, x_max, y_max]` is Qwen's
  native grounding order; `yxyx` = `[y_min, x_min, y_max, x_max]` is the
  MOSS / Gemini order. `auto` (default): `xyxy` for Qwen-style clips, `yxyx`
  for Gemma 4; the tag grounding prompt stays `xyxy` and a custom
  **system_prompt** stays `yxyx` (the order it was written for, unless it
  uses the `{{box_schema}}` / `{{box_rule}}` placeholders). If boxes in
  `bbox_preview` look mirrored or swapped, switch to the other order.
- **item_detail** (optional) - layout mode, built-in prompts: how finely the
  image is split into items. `units` (default): whole subjects, one person =
  one item. `parts`: every person or animal also gets separate items for
  head, hair, torso, arms, hands, legs, feet and clothing, plus all objects
  and the main background regions, so the items cover most of the image.
  Raise **max_items** with `parts` (40+). Adapted from the smart mode of
  [ComfyUI-MOSS-QwenVL-Layout](https://github.com/leroy989/ComfyUI-MOSS-QwenVL-Layout)
  by leroy989 (MIT). Not used by the tag grounding prompt. A custom
  **system_prompt** can use the `{{item_scope}}` placeholder.
- **box_padding** (optional) - layout mode: grows every box by this percent of
  its own size on each side (clamped to the canvas), applied after the
  conversion, so `bbox_preview` shows the padded boxes. 0 = off, 2-5% is a
  good start. Idea from the same node.
- **pretty_json** - indented JSON instead of one compact line.
- **unload_after_run** - frees `clip` from VRAM when the batch is done.

## Outputs

All outputs are lists (one entry per image): **json** (the layout JSON, or
`{"caption": "..."}` in caption mode), **caption** (the plain chosen
caption), **watermark** and **borders** (JSON as text, caption mode only),
**width**, **height** (the canvas), **bbox_preview** (the image resized to
the canvas with the detected boxes drawn on it; in caption mode the
watermark boxes). If an image cannot be captioned, `json`
holds `[MOSS captioning failed: reason]`, the text outputs are empty, a
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
