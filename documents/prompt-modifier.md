# Mykee Prompt Advanced - Prompt Modifier

[← Back to the main README](../README.md)  ·  Category: `Mykee/Prompt`

<p align="center">
  <img src="../images/prompt-advanced-prompt-modifier.png" alt="Mykee Prompt Advanced - Prompt Modifier">
</p>

The "edit" sibling of **Mykee Prompt Advanced**. Instead of paraphrasing a
prompt or describing an image from scratch, it takes an **existing prompt
(or image)** plus a separate **edit_instruction**, and rewrites *only the
details the instruction addresses* - everything else (wording, structure,
language, length) stays as close to untouched as possible. It also
resolves contradictions the edit creates elsewhere: e.g. if the
instruction puts a piece of clothing back on, anything that only makes
sense with it off is fixed too, not left dangling.

**Two modes:**

- **Text-only** (no image): `prompt` is edited surgically according to
  `edit_instruction`, keeping its original form - tag list stays a tag
  list, prose stays prose, same language, same length - and only the parts
  the instruction touches change.
- **Image mode** (image + VL-capable clip): the model first works out
  internally exactly what the image shows, then writes a prose description
  of **that same scene with the edit applied**. The two-step approach is
  deliberate: a single "describe this image and also apply this edit"
  instruction proved unreliable at actually taking the edit into account.
  If `edit_instruction` is empty, the image is simply described as-is.
  Here `prompt` is optional extra context.

## Inputs

- **clip** - text/VL model (VL needed for image mode; must be
  generation-capable, see Mykee Prompt Advanced).
- **edit_instruction** - the change(s) to make, in plain language.
- **prompt** - the prompt to edit (text mode), or optional context (image
  mode).
- **prompt_rewrite** - OFF: no model call, no VRAM load; `prompt` passes
  through unchanged to both outputs.
- **system_prompt** - empty = built-in default (surgical edit / understand
  then re-describe with the edit applied). Fill in to override;
  `{max_length}` is replaced with the token limit.
- **max_length, sampling_mode, seed, temperature, top_k, top_p, min_p,
  repetition_penalty, presence_penalty, thinking, unload_after_run** - the
  same as in Mykee Prompt Advanced (same defaults).
- **image** (optional) - switches the node into image-edit mode.

## Outputs

- **prompt** - the original prompt, unchanged (so the node can sit inline
  in a chain).
- **edited_prompt** - the rewritten prompt / description. Also shown in a
  read-only text box on the node after each run.
