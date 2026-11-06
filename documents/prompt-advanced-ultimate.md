# Mykee Prompt Advanced Ultimate

[← Back to the main README](../README.md)  ·  Category: `Mykee/Prompt`

<p align="center">
  <img src="../images/prompt-advanced-ultimate.png" alt="Mykee Prompt Advanced Ultimate">
</p>

A merged, **two-stage** version of Mykee Prompt Advanced and the Prompt
Modifier, built around one idea: **first write out a detailed description,
then answer a question about it / edit it - from the text alone.**

Why two stages? On non-thinking VL models a single "describe the image
*and* answer this question about it" instruction is unreliable: the model
commits to a quick, statistically typical answer before it has actually
looked at the fine details, and there is no hidden reasoning to catch the
mistake afterwards - the generated tokens *are* the only reasoning. Splitting
the job forces stage 1 to write the visual evidence down as real, visible
text; stage 2 then works purely from that text (it never sees the image
again), where the answer can be checked against concrete, written evidence.

## Stage 2 default prompt (analyze)

The built-in analyze prompt answers strictly from RESPONSE 1, in the language
of the question (or the language the question asks for), without inventing
anything and without refusing NSFW content. It does not hunt for
contradictions on its own: two statements only conflict if they cannot both be
true at once (different hands or body parts doing different things at the same
time, or simply unmentioned details, are not contradictions). When the
question itself asks for contradictions, it reports only real ones, quotes
both parts, and says plainly that none were found if there are none. Lists
are used only when the question asks for one.

## Stage 1 - description

- **Image connected**: a full, highly detailed prose description. If
  `prompt` contains a specific question or focus, the default system prompt
  tells the model to explicitly confirm *or rule out* every detail relevant
  to it - including saying clearly when something is **not** present, which
  a plain "describe the image" instruction tends to skip.
- **No image**: paraphrases `prompt` back (like Mykee Prompt Advanced).

## Stage 2 - answer / edit (always text-only)

Runs only if **second_stage_enabled** is ON (default) **and**
`second_prompt` is not empty. Otherwise it is skipped and `response_2`
simply mirrors `response_1`.

The model receives stage 1's result as plain text plus `second_prompt`,
and works strictly from that. **second_mode** picks what that means:

- **analyze** (default) - `second_prompt` is a *question* about stage 1's
  result ("RESPONSE 1"); stage 2 answers it.
- **edit** - `second_prompt` is an *edit instruction*; stage 2 rewrites the
  stage-1 text accordingly, the same way as Prompt Modifier does (a short
  analysis/plan first, then only the part after the `===FINAL===` marker is
  used as `response_2`).

The built-in stage-2 prompt follows `second_mode`. If you write your own
`second_system_prompt`, refer to the labels the mode uses: "RESPONSE 1"
(analyze) or "PROMPT" / "EDIT INSTRUCTION" (edit).

Both stages share the same `clip`, loaded once. Each stage has its **own
independent controls**, because a long, varied description and a short,
deterministic answer usually want different settings.

## Tags

With **generate_tags** ON, stage 1 is additionally asked for an
e621-style comma-separated tag list below a `===TAGS===` marker line,
after the description. **max_tags** (2-100, default 20) limits the length;
the model is told not to pad the list with "absence" tags ("no bra",
"no jewelry", ...) just to reach that count. `response_1` always keeps the
full raw text (description + marker + tags); the tag list is also split out
into its own `tags_1` output. (If you use a custom `system_prompt` without
the marker, `tags_1` stays empty.)

## Inputs

- **clip, image, video, audio** - as in Mykee Prompt Advanced (Gemma 4 gets
  its native chat template, video and audio; see
  [Mykee Prompt Advanced](prompt-advanced.md#gemma-4-video-and-audio)). The
  media is used by stage 1 only.
- **prompt** - stage 1's user turn (a question/focus/context in image mode;
  the text to paraphrase in text-only mode). In text-only mode the text is
  sent wrapped in triple quotes (`"""..."""`), so it is treated as text to
  process and never as an instruction (no quotes of your own needed). In
  stage 2 the text taken over from stage 1 (`PROMPT` / `RESPONSE 1`) is
  quoted the same way; `second_prompt` is sent as written.
- **prompt_rewrite** - master switch. OFF: no model calls for either
  stage, everything passes through, no VRAM load.
- **system_prompt** - stage 1's system prompt (empty = built-in default).
- **generate_tags, max_tags** - see above.
- **stage1_passthrough** - text-only mode: stage 1 is skipped completely and
  `prompt` goes to stage 2 unchanged as RESPONSE 1 (no model call, no
  paraphrase, so also nothing that can refuse or shorten the text). Off
  (default): stage 1 reads and paraphrases the prompt as before. Ignored when
  an image, video or audio is connected, because stage 1 has to look at it.
  Use it when the prompt already is the text stage 2 should work with. It is
  the last widget of the node, so saved workflows keep their values.
- **max_length, sampling_mode, seed, temperature, top_k, top_p, min_p,
  repetition_penalty, presence_penalty, thinking** - stage 1's generation
  controls (same meaning as in Mykee Prompt Advanced).
- **second_stage_enabled, second_mode, second_prompt,
  second_system_prompt** - stage 2 settings, see above.
- **second_max_length, second_sampling_mode, second_seed,
  second_seed_control, second_temperature, second_top_k, second_top_p,
  second_min_p, second_repetition_penalty, second_presence_penalty,
  second_thinking** - stage 2's own generation controls.
  **second_seed_control** (fixed / increment / decrement / randomize) is
  the node's own equivalent of ComfyUI's *control after generate*: after
  each run it updates `second_seed` for the next one.
- **thinking_budget**, **second_thinking_budget**, **template_mode**
  (optional) - extra tokens for each stage's thinking (default 1024 / 512)
  and the chat-template mode; see
  [Mykee Prompt Advanced](prompt-advanced.md#gemma-4-video-and-audio).
- **unload_after_run** - frees the clip that was used *last* from VRAM once,
  after both stages have finished: `clip_2` if it is connected and stage 2
  ran, otherwise `clip`. Only that clip is unloaded; every other model
  (e.g. the diffusion model) stays loaded.
- **clip_2** (optional) - a second text model for stage 2. Connected (and not
  the same clip as `clip`): stage 1 runs with `clip`, then `clip` is taken out
  of memory and stage 2 runs with `clip_2` (it loads itself). At the start of
  the next run `clip_2` is taken out first and `clip` is loaded again, so only
  one of the two sits in memory at a time. Not connected, or the same clip as
  `clip`: both stages use `clip` as before, nothing is swapped.

## Stage 1 reuse

If nothing that feeds stage 1 changed since the previous run (prompt, system
prompt, image/video/audio content, stage-1 sampling settings and seed, thinking,
template mode, same `clip`), stage 1 is **not generated again**: its previous
result is reused and only stage 2 runs. This makes iterating on the stage-2
prompt fast. With `clip_2` connected no clip swapping happens in that case.
To get a fresh stage-1 result, change the stage-1 seed (or any other stage-1
input). Failed stage-1 runs are never reused. The cache lives in memory only
(last 8 results) and is cleared when ComfyUI restarts.

## Outputs

- **prompt_1** - stage 1's input prompt, unchanged.
- **prompt_2** - `second_prompt`, unchanged (whether or not stage 2 ran).
- **response_1** - stage 1's full, unsplit result (including the
  `===TAGS===` marker and tag list, if generated).
- **response_2** - stage 2's result, or a copy of `response_1` if stage 2
  was skipped.
- **tags_1** - only the tag list from `response_1`. Empty if
  `generate_tags` is off.
- **thinking_1**, **thinking_2** - the reasoning of stage 1 / stage 2 when
  their `thinking` toggle is on (empty otherwise). The reasoning never ends
  up in the responses.

`response_1`, `response_2`, `tags_1`, `thinking_1` and `thinking_2` are also shown in read-only text
boxes at the bottom of the node after each run, so you can check them
without wiring the outputs anywhere. Editing those boxes has no effect -
they are overwritten on the next run.
