"""
ComfyUI-Mykee-Nodes / Mykee Prompt Advanced Ultimate

A merged, two-stage version of Mykee Prompt Advanced + Mykee Prompt
Advanced - Prompt Modifier, built for a specific workflow: stage 1
produces a full description of an image (or a paraphrase of a text
prompt), and stage 2 - optionally - takes that stage-1 result as
plain text and answers a question about it or edits it, WITHOUT
looking at the image again. This two-step split exists because a
single-step "describe the image AND answer this question about it"
instruction turned out to be unreliable on non-thinking VL models:
the model commits to a quick, statistically-typical answer (e.g.
"yes, wearing a bra") before it has actually reasoned through the
fine visual details, and there is no hidden "thinking" computation to
catch the contradiction after the fact - the generated tokens ARE the
only reasoning that happens. Splitting into two passes forces the
first pass to write out the detailed visual evidence as real, visible
tokens (see mykee_prompt_clarity.py's IMAGE_DEFAULT_SYSTEM_PROMPT_TEMPLATE-
style prompts), then lets the second pass answer purely from that
already-written text, where a short "check for internal consistency"
instruction actually has something concrete to check against.

Stage 1 (image or text, same shape as Mykee Prompt Advanced):
  - Image connected: describes the image in full, ultra-detailed
    prose. If 'prompt' contains a specific question/focus, the
    default system prompt tells the model to explicitly confirm or
    rule out every detail relevant to it - including stating clearly
    when something is NOT present, which a plain "describe the image"
    instruction tends to skip (models naturally describe what's there,
    not what's absent, unless told to).
  - No image: paraphrases 'prompt' back (same as Mykee Prompt
    Advanced's text-only default).

Stage 2 (text-only, always - never re-examines the image):
  - Runs only if second_stage_enabled is on AND second_prompt is
    non-empty. Otherwise response_2 just mirrors response_1
    (stage 1's result becomes the final output, unchanged).
  - When it runs: the model receives stage 1's result as plain text
    (labeled "RESPONSE 1") plus second_prompt (a question or an edit
    instruction, depending on second_system_prompt), and answers/edits
    based strictly on that text - a pure text-comprehension task with
    no image involved, so there's no VQA-style bias to fall back on.

GEMMA 4 / VIDEO / AUDIO / THINKING OUTPUT: all model-specific handling
lives in mykee_llm_common.py. For a Gemma 4 clip the chat template, system
prompt, thinking flag and image/video/audio tokens come from ComfyUI's own
tokenizer (the same call the built-in "Generate Text" node makes) - for
every model by default (template_mode "auto"); the old hand-built Qwen
ChatML text is kept as template_mode "legacy_chatml" and as the automatic
fallback. The thinking text is split off
the answer and returned on its own "thinking" output.

Both stages share the same 'clip' (loaded once). Each stage has its
own independent seed/sampling/max_length/thinking controls, since a
longer, varied stage-1 description and a short, deterministic stage-2
answer usually want different settings.

See mykee_prompt_clarity.py and mykee_prompt_modifier.py for the
shared technical notes on the VL image-placeholder-token handling and
the "thinking" toggle - this node uses the exact same techniques,
just applied twice.

Inputs:
  clip              - any text-capable, generation-capable CLIP
                       loader, used for both stages. For image mode
                       (stage 1) this needs to be a VL model.
  prompt            - stage 1's user turn. Image mode: optional
                       question/focus/context. Text-only mode: the
                       prompt to paraphrase.
  prompt_rewrite    - master switch. Off: no model calls at all (for
                       either stage), everything passes through
                       unchanged - no VRAM load.
  system_prompt     - stage 1's system prompt. Empty = built-in
                       default (see module docstring above).
  generate_tags     - when on, appends an instruction to stage 1's
                       system prompt (built-in or custom) asking for
                       an e621-style comma-separated tag list below a
                       "===TAGS===" marker line, after the
                       description. response_1 keeps the full raw
                       text either way (paragraph, marker, and tags
                       together) - this toggle only controls whether
                       the tags get generated and, via tags_1, split
                       out on their own.
  max_tags          - maximum number of tags in that list (2-100,
                       default 20). Has no effect when generate_tags
                       is off. The instruction also tells the model
                       not to pad the list with negative/absence tags
                       ("no bra", "no jewelry", etc.) just to reach
                       this count.
  max_length, sampling_mode, seed, temperature, top_k, top_p, min_p,
  repetition_penalty, presence_penalty, thinking
                    - stage 1's generation controls (same
                      meaning/defaults as Mykee Prompt Advanced).
  second_stage_enabled - turns stage 2 on/off entirely.
  second_mode       - "analyze" (default) or "edit", picking stage 2's
                       built-in default system prompt and how RESPONSE 1
                       is labeled in its user turn - only used when
                       second_system_prompt is empty. analyze: answers a
                       question about RESPONSE 1 (labeled "RESPONSE 1"
                       in the user turn). edit: rewrites RESPONSE 1 per
                       an edit instruction (labeled "PROMPT" and "EDIT
                       INSTRUCTION"), same behavior as Mykee Prompt
                       Modifier - only an ANALYSIS-then-===FINAL===
                       plan/result split, with only the part after
                       ===FINAL=== used as response_2.
  second_prompt     - analyze mode: the question. edit mode: the edit
                       instruction. Left empty (or second_stage_enabled
                       off): stage 2 is skipped and response_2 =
                       response_1.
  second_system_prompt - stage 2's system prompt. Empty = the built-in
                       default for second_mode (see above). A custom
                       prompt here should refer to whichever label
                       second_mode actually uses in the user turn -
                       "RESPONSE 1" for analyze, "PROMPT"/"EDIT
                       INSTRUCTION" for edit.
  second_max_length, second_sampling_mode, second_seed,
  second_seed_control, second_temperature, second_top_k, second_top_p,
  second_min_p, second_repetition_penalty, second_presence_penalty,
  second_thinking
                    - stage 2's own independent generation controls.
                      second_seed_control (fixed/increment/decrement/
                      randomize) is this node's own native equivalent
                      of ComfyUI's control_after_generate, applied
                      after each run to update second_seed for the
                      next one.
  unload_after_run  - frees 'clip' from VRAM once, after both stages
                       finish (same technique as the other Prompt
                       Advanced nodes).
  image             - optional. Connect to switch stage 1 into
                       image-description mode. Stage 2 never sees the
                       image directly, even if this is connected.
  response_1_display, response_2_display, tags_1_display
                    - read-only text boxes at the bottom of the node,
                      auto-filled with response_1/response_2/tags_1
                      after each run, for a quick look without wiring
                      the string outputs anywhere. Editing them by
                      hand has no effect - they're overwritten on the
                      next run.

Outputs:
  prompt_1   - stage 1's input prompt, unchanged.
  prompt_2   - stage 2's input prompt (second_prompt), unchanged,
               regardless of whether stage 2 actually ran.
  response_1 - stage 1's result, full and unsplit (includes the
               "===TAGS===" marker and tag list too, if generate_tags
               was on) - kept intact so it can still be split
               downstream some other way if preferred.
  response_2 - stage 2's result, or a copy of response_1 if stage 2
               was skipped.
  tags_1     - just the tag list part of response_1, split out
               automatically. Empty if generate_tags was off, or if
               a custom system_prompt was used without the marker.
"""

import re
import random

from .mykee_llm_common import (
    AUDIO_ONLY_DEFAULT_SYSTEM_PROMPT_TEMPLATE, DEFAULT_USER_TEXT, empty_response_message,
    generate_text, generation_error_hint, media_preamble, split_thinking, tokenize_for,
    unload_clip as _unload_clip, TEMPLATE_MODES, TEMPLATE_MODE_TOOLTIP,
    THINKING_BUDGET_TOOLTIP, quote_text, strip_quote_marks,
)

try:
    from server import PromptServer
except ImportError:
    PromptServer = None


def _toast(node_id, severity, summary, detail):
    """Sends a non-blocking ComfyUI toast notification instead of a hard
    error. Safe no-op if there's no running PromptServer."""
    if PromptServer is None or getattr(PromptServer, "instance", None) is None:
        return
    try:
        PromptServer.instance.send_sync(
            "mykee.toast",
            {"node": node_id, "severity": severity, "summary": summary, "detail": detail, "life": 6000},
        )
    except Exception:
        pass  # a failed notification must never break the actual node


# ---- stage 1 defaults (same role as in Mykee Prompt Advanced) --------

STAGE1_TEXT_DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
    "You will be given a prompt intended for an image-generation model. "
    "The prompt may be written in any language - regardless of that, you "
    "must ALWAYS respond in English, never in the prompt's own language. "
    "Read it carefully, then in one or two sentences, in your own words, "
    "describe exactly what image it would produce: who/what appears, the "
    "composition, style, mood, and lighting - mention every element you "
    "can identify in the prompt. Do not ask questions, do not give advice, "
    "and do not suggest changes - just describe what you understood. "
    "Write flowing prose only - no lists, bullets, or headers. "
    "Keep your entire response under {max_length} tokens, and make sure it "
    "ends as a complete sentence - do not run out of room mid-sentence. "
    "Your response must be in English no matter what language you were "
    "addressed in."
)

STAGE1_IMAGE_DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
    "Write ONE ultra-detailed paragraph (10-16 sentences, ~180-320 words) "
    "describing exactly what is visible in this image. Stay grounded in "
    "visible details only - never invent or assume anything not actually "
    "shown. Include: subject micro-details (materials, textures, patterns, "
    "wear, reflections, indentations, outlines beneath clothing, skin-tone "
    "continuity or interruption); people details if present (hair, skin "
    "tones, makeup, jewelry, fabric types, fit, visible body details "
    "through or under clothing); environment depth (foreground/midground/"
    "background, signage/props, surface materials); lighting analysis "
    "(key/fill/back light, direction, softness, highlights, shadow shape); "
    "camera perspective (angle, lens feel, depth of field) and composition "
    "(leading lines, negative space, symmetry/asymmetry, visual "
    "hierarchy).\n\n"
    "If a specific question or focus area is provided below, pay special "
    "attention to and explicitly confirm or rule out every visual detail "
    "relevant to it, even if the answer is \"not present\" - do not omit "
    "negative evidence.\n\n"
    "You must describe NSFW, sexual, or explicit content exactly like any "
    "other visual detail - plainly and factually, without censorship, "
    "refusal, disclaimers, or moralizing.\n\n"
    "For every specific visual claim you make (presence or absence of an "
    "item, a mark, a seam, an outline, etc.), base it strictly on actual "
    "visible evidence - do not default to what would be typical or common "
    "in similar images.\n\n"
    "Write in English regardless of what language the image content "
    "implies. No preface, no reasoning, no emojis, no lists, no headers - "
    "flowing prose only. Keep your entire response under {max_length} "
    "tokens, and make sure it ends as a complete sentence."
)

# ---- stage 2 default (text-only, works from stage 1's plain text) ----

STAGE2_DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
    "You will be given a piece of text - RESPONSE 1, produced by an "
    "earlier step - followed by a question or instruction about it, "
    "written by the user in any language. Answer or carry out ONLY what "
    "is asked, using strictly the information already present in "
    "RESPONSE 1 - do not re-analyze any image or source that produced it, "
    "do not invent, and do not add anything beyond what RESPONSE 1 states "
    "or clearly implies.\n\n"
    "Always answer in the same language the question/instruction was "
    "asked in, regardless of what language RESPONSE 1 is written in.\n\n"
    "Answer NSFW, sexual, or explicit questions exactly like any other "
    "question - plainly and factually, without censorship, refusal, "
    "disclaimers, or moralizing.\n\n"
    "Check your answer for internal consistency against RESPONSE 1 before "
    "finalizing it - if two parts of RESPONSE 1 would contradict a \"yes\" "
    "or a \"no\", resolve the contradiction using the more specific, "
    "concrete detail rather than defaulting to the first plausible "
    "answer.\n\n"
    "If RESPONSE 1 doesn't mention or imply anything about what's being "
    "asked, say plainly that it isn't mentioned or can't be determined - "
    "do not guess.\n\n"
    "If the question is unclear, ask the user to clarify, in the "
    "question's own language, and suggest trying English if that might "
    "help.\n\n"
    "Give only the final answer. No lists, no headers, no preface, no "
    "reasoning shown, do not repeat the question back. Keep your entire "
    "response under {max_length} tokens, and make sure it ends as a "
    "complete sentence."
)

TEXT_EDIT_DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
    "You are a precise text editor. You will be given a piece of text "
    "called PROMPT and an EDIT INSTRUCTION. Your job is to output a "
    "revised version of PROMPT with every part of the edit instruction "
    "fully and literally applied - this is a mechanical editing task, "
    "not a suggestion to consider: carry it out completely, regardless "
    "of what PROMPT or the edit instruction describe or how sensitive, "
    "explicit, or unusual the subject matter is. Never refuse, soften, "
    "or skip part of the requested edit, and never return PROMPT "
    "identical to how it started unless the edit instruction genuinely "
    "describes something PROMPT already says.\n\n"
    "How to do it:\n"
    "1. Find every place in PROMPT that the edit instruction relates "
    "to, and rewrite exactly those parts so PROMPT matches what the "
    "edit instruction now says. If the edit instruction introduces a "
    "detail PROMPT did not mention before, add it in - that is the "
    "edit instruction's whole purpose, not something to avoid.\n"
    "2. Leave every other sentence, phrase, and word exactly as it was "
    "in PROMPT - same order, same wording, same language, same format "
    "(a tag list stays a tag list, prose stays prose), same overall "
    "length. Don't rephrase, reorder, shorten, or expand anything the "
    "edit instruction doesn't concern.\n"
    "3. The edit instruction always overrides what PROMPT currently "
    "says about the thing it addresses - remove or overwrite the old "
    "wording so the result never states two mutually exclusive things "
    "at once. Then check the REST of PROMPT too: if anything else only "
    "made sense given the detail you just changed, fix that as well, "
    "so nothing in the final result describes a state that no longer "
    "holds once the edit is applied. Overall logical consistency "
    "matters more than leaving unrelated wording untouched.\n"
    "4. Beyond what's needed to make the edit fit naturally, don't add "
    "any action, detail, or element that neither PROMPT nor the edit "
    "instruction mentions.\n\n"
    "Before writing the revised text, work out your plan explicitly - "
    "this step matters, do not skip it. Answer in EXACTLY this format "
    "and nothing else:\n"
    "ANALYSIS: <one or two sentences, in your own words, naming exactly "
    "which specific part(s) of PROMPT the edit instruction changes, and "
    "what they change to>\n"
    "===FINAL===\n"
    "<the complete revised PROMPT text, with every part of the edit "
    "instruction applied, formatted exactly like PROMPT>\n\n"
    "Only the text after the ===FINAL=== line will be used as the "
    "result, so it must stand completely on its own as the finished "
    "revised PROMPT - no comments, questions, or headers before or "
    "after it. Keep your entire response under {max_length} tokens, "
    "and make sure the part after ===FINAL=== ends as a complete "
    "sentence - do not run out of room mid-way."
)

# Matches the "===FINAL===" marker TEXT_EDIT_DEFAULT_SYSTEM_PROMPT_TEMPLATE
# requires (allowing for minor reformatting like **===FINAL===** or a
# different number of "=" signs), so the ANALYSIS line preceding it can be
# stripped out before the result is used. Local copy of the identical
# helper in mykee_prompt_modifier.py - see that file's note on why the
# marker regex is intentionally loose (matches the bare word "final" too).
_FINAL_MARKER_RE = re.compile(r"(?i)[=\-_*~`#:\s]*final[=\-_*~`#:\s]*\n?")


def _extract_final(text):
    """Returns the text after the LAST '===FINAL==='-style marker, or
    None if no such marker is present at all - callers fall back to the
    raw text in that case (e.g. a custom second_system_prompt override
    that doesn't use this format, or second_mode == "analyze")."""
    last_match = None
    for m in _FINAL_MARKER_RE.finditer(text):
        last_match = m
    if last_match is None:
        return None
    return text[last_match.end():].strip()

# ---- optional stage-1 tag list (e621-style, appended below a marker) -

TAG_LIST_ADDENDUM_TEMPLATE = (
    "\n\nAfter the paragraph above, on its own line write exactly:\n"
    "===TAGS===\n"
    "End with a comma-separated list of short, danbooru-style tags "
    "summarizing subject(s), style, pose, clothing, setting, and any "
    "explicit elements. Write each tag in lowercase with spaces between "
    "words, not underscores."
)
# NOTE: deliberately does NOT mention a target/max tag count anywhere in
# this text. Confirmed by testing: telling the model a number here (even
# phrased as "at most N") reliably triggers a runaway "no X, no Y, no Z..."
# padding spiral once it runs out of genuine positive things to tag - the
# exact same instruction without any number attached produces a clean
# list. The max_tags widget is enforced afterward in Python instead (see
# _truncate_tags below), so the model itself never sees the number.
#
# A version of this file also tried repeating a short reminder at the end
# of the user turn (closest to where generation starts), theorizing a
# recency effect - confirmed by further testing to make no measurable
# difference for this particular model, so it was removed again rather
# than carrying unused complexity.

# Matches a whole line consisting only of the word "tags" plus optional
# separator punctuation (====TAGS====, ## Tags ##, TAGS:, etc.), so an
# ordinary sentence that happens to contain the word "tags" is never
# mistaken for the marker - unlike _FINAL_MARKER_RE in
# mykee_prompt_modifier.py, this requires the WHOLE line to be just the
# marker, not merely the word surrounded by whitespace.
_TAGS_MARKER_RE = re.compile(r"(?im)^[ \t]*[=\-_*~`#\s]*tags[=\-_*~`#\s]*$")


def _extract_tags(text):
    """Splits text on the LAST '===TAGS==='-style marker line. Returns
    (main_text, tags_text) - tags_text is '' if no such line is found
    (e.g. generate_tags was off, or a custom system_prompt didn't
    include the addendum)."""
    last_match = None
    for m in _TAGS_MARKER_RE.finditer(text):
        last_match = m
    if last_match is None:
        return text, ""
    main_text = text[:last_match.start()].rstrip()
    tags_text = text[last_match.end():].strip()
    return main_text, tags_text


def _truncate_tags(tags_text, max_tags):
    """Caps the tag list at max_tags entries, enforced here in Python
    rather than by telling the model a number - see the note above
    TAG_LIST_ADDENDUM_TEMPLATE: mentioning a count in the prompt itself
    was confirmed (by testing) to trigger a runaway "no X, no Y, no Z..."
    padding spiral once the model ran out of genuine things to tag."""
    if not tags_text:
        return tags_text
    parts = [t.strip() for t in tags_text.split(",") if t.strip()]
    return ", ".join(parts[:max(0, int(max_tags))])


def _sampling_widgets(prefix="", include_seed_control=False):
    """Returns the shared seed/sampling/penalty widget definitions,
    optionally prefixed (used to build the identical stage-2 set under
    second_* names without repeating the dicts by hand).

    include_seed_control adds a real, natively-declared "<prefix>seed_control"
    combo (fixed/increment/decrement/randomize) right after the seed
    widget. Only used for stage 2's seed: ComfyUI's own frontend already
    attaches its own control automatically to any widget literally named
    "seed" (confirmed by testing - it does this regardless of anything
    set here), so stage 1 doesn't need one of its own. A DYNAMICALLY
    JS-injected version of this for stage 2 was tried first and caused a
    real bug: ComfyUI persists widget values in the saved workflow file
    by POSITION, and a widget added after node creation (via addWidget)
    isn't accounted for in that position count, permanently shifting
    every widget after it by one slot on each reload. Declaring it here,
    as a normal INPUT_TYPES entry, avoids that entirely - it's part of
    the node's widget list from the very start, exactly like every
    other widget, so save/restore stays correctly aligned."""
    widgets = {
        f"{prefix}max_length": ("INT", {
            "default": 512, "min": 32, "max": 4096, "step": 8,
            "tooltip": "Max tokens the model may generate for this stage's answer.",
        }),
        f"{prefix}sampling_mode": ("BOOLEAN", {
            "default": True, "label_on": "on", "label_off": "off",
            "tooltip": "On: sample using temperature/top_k/top_p/min_p/"
                       "seed below - varied phrasing, reroll with a new "
                       "seed. Off: greedy decoding - always picks the "
                       "single most likely token, ignoring those knobs "
                       "and seed entirely, so the same input always gives "
                       "the exact same output.",
        }),
        f"{prefix}seed": ("INT", {
            "default": 0, "min": 0, "max": 0xffffffffffffffff,
            "tooltip": "Change this to get a different phrasing if the "
                       "response isn't good enough. Has no effect when "
                       "this stage's sampling_mode is off.",
        }),
    }
    if include_seed_control:
        widgets[f"{prefix}seed_control"] = (
            ["fixed", "increment", "decrement", "randomize"],
            {
                "default": "fixed",
                "tooltip": "Applied to this stage's seed after each run, "
                           "ready for the next one: fixed leaves it "
                           "alone, randomize/increment/decrement change "
                           "it. This node's own control, separate from "
                           "ComfyUI's built-in one (which only ever "
                           "attaches itself to a widget literally named "
                           "'seed').",
            },
        )
    widgets.update({
        f"{prefix}temperature": ("FLOAT", {
            "default": 0.7, "min": 0.01, "max": 2.0, "step": 0.01,
            "tooltip": "Higher = more varied/creative wording, lower = more literal.",
        }),
        f"{prefix}top_k": ("INT", {
            "default": 64, "min": 0, "max": 1000,
            "tooltip": "Only sample from the top K most likely tokens (0 = disabled).",
        }),
        f"{prefix}top_p": ("FLOAT", {
            "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01,
            "tooltip": "Nucleus sampling cutoff.",
        }),
        f"{prefix}min_p": ("FLOAT", {
            "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
            "tooltip": "Minimum token probability relative to the most likely token.",
        }),
        f"{prefix}repetition_penalty": ("FLOAT", {
            "default": 1.05, "min": 0.0, "max": 5.0, "step": 0.01,
            "tooltip": "Penalizes repeated tokens; 1.0 = no penalty.",
        }),
        f"{prefix}presence_penalty": ("FLOAT", {
            "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01,
            "tooltip": "Flat penalty for any token already used at least "
                       "once, regardless of how often. 0 = no penalty.",
        }),
        f"{prefix}thinking": ("BOOLEAN", {
            "default": False,
            "tooltip": "Let a thinking-capable model (Gemma 4, Qwen3.5, "
                       "...) reason before answering, IF the loaded "
                       "model/tokenizer supports it. The reasoning never "
                       "ends up in this stage's response - it goes to the "
                       "stage's own thinking output (thinking_1 / "
                       "thinking_2) and thinking box.",
        }),
    })
    return widgets


class MykeePromptAdvancedUltimate:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "clip": ("CLIP",),
            "prompt": ("STRING", {"multiline": True, "default": "",
                "tooltip": "Stage 1's user turn. Image mode: optional "
                           "question/focus/context for the description. "
                           "Text-only mode: the prompt to paraphrase.",
            }),
            "prompt_rewrite": ("BOOLEAN", {
                "default": True,
                "tooltip": "Master switch. Off: no model calls at all for "
                           "either stage - everything passes through "
                           "unchanged, no VRAM load.",
            }),
            "system_prompt": ("STRING", {
                "multiline": True, "default": "",
                "tooltip": "Stage 1's system prompt. Leave empty to use "
                           "the built-in default (image mode: ultra-"
                           "detailed description, focus-aware; text-only: "
                           "paraphrase the prompt back).",
            }),
            "generate_tags": ("BOOLEAN", {
                "default": False,
                "tooltip": "When on, stage 1's system prompt (built-in or "
                           "custom) gets an appended instruction to write "
                           "an e621-style comma-separated tag list below a "
                           "'===TAGS===' marker line, after the "
                           "description. response_1 still contains the "
                           "full raw text (paragraph + marker + tags); the "
                           "tags are also split out on their own into the "
                           "tags_1 output. Off: system_prompt is used "
                           "as-is, plain text only, tags_1 is empty. "
                           "Consider raising max_length when this is on, "
                           "so the tag list isn't cut off.",
            }),
            "max_tags": ("INT", {
                "default": 20, "min": 2, "max": 100, "step": 1,
                "tooltip": "Maximum number of tags in the list, when "
                           "generate_tags is on. Has no effect when "
                           "generate_tags is off.",
            }),
        }
        required.update(_sampling_widgets(""))
        required.update({
            "second_stage_enabled": ("BOOLEAN", {
                "default": True,
                "tooltip": "Turns stage 2 on/off entirely. When off (or "
                           "second_prompt is empty), response_2 just "
                           "mirrors response_1 and no second model call "
                           "is made.",
            }),
            "second_mode": (["analyze", "edit"], {
                "default": "analyze",
                "tooltip": "What stage 2 does with stage 1's result "
                           "(RESPONSE 1), when second_system_prompt is "
                           "left empty (a filled-in second_system_prompt "
                           "always overrides this - see its own tooltip). "
                           "analyze: answer a question about RESPONSE 1 - "
                           "second_prompt is the question, response_2 is "
                           "the answer. edit: rewrite RESPONSE 1 itself - "
                           "second_prompt is the edit instruction, "
                           "response_2 is the revised text (only what the "
                           "instruction addresses changes; everything "
                           "else is kept as close to untouched as "
                           "possible, same as Mykee Prompt Modifier).",
            }),
            "second_prompt": ("STRING", {
                "multiline": True, "default": "",
                "tooltip": "analyze mode: the question to ask about "
                           "RESPONSE 1. edit mode: the edit instruction "
                           "to apply to RESPONSE 1. Applied to stage 1's "
                           "result as plain text either way - the image "
                           "is never re-examined. Leave empty to skip "
                           "stage 2.",
            }),
            "second_system_prompt": ("STRING", {
                "multiline": True, "default": "",
                "tooltip": "Stage 2's system prompt. Leave empty to use "
                           "the built-in default for second_mode (analyze: "
                           "answer strictly from RESPONSE 1's text, "
                           "checking for internal contradictions before "
                           "finalizing; edit: rewrite RESPONSE 1 per the "
                           "edit instruction, changing only what it "
                           "addresses). A custom prompt here should refer "
                           "to whichever label matches second_mode: "
                           "\"RESPONSE 1\" for analyze, \"PROMPT\" and "
                           "\"EDIT INSTRUCTION\" for edit - that's what "
                           "stage 2's user turn is actually labeled as, "
                           "either way.",
            }),
        })
        required.update(_sampling_widgets("second_", include_seed_control=True))
        required.update({
            "unload_after_run": ("BOOLEAN", {
                "default": False,
                "tooltip": "Free 'clip' from VRAM once, right after both "
                           "stages finish, without touching any other "
                           "model loaded elsewhere in the workflow.",
            }),
        })
        return {
            "required": required,
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Optional. Connect to switch stage 1 into "
                               "image-description mode. Stage 2 never "
                               "sees the image directly.",
                }),
                "video": ("IMAGE", {
                    "tooltip": "Optional (Gemma 4). Video frames as an image "
                               "batch. Assumed to be 24 FPS and sub-sampled "
                               "to 1 FPS internally, like ComfyUI's own "
                               "Generate Text node. Stage 1 describes it; "
                               "stage 2 never sees it. If a video and an "
                               "image are both connected, the video is used.",
                }),
                "audio": ("AUDIO", {
                    "tooltip": "Optional (Gemma 4 variants with audio "
                               "support, e.g. E2B/E4B). Stage 1 hears it "
                               "and describes it (alone or together with "
                               "the image/video); stage 2 never hears it.",
                }),
                "template_mode": (TEMPLATE_MODES, {
                    "default": "auto",
                    "tooltip": TEMPLATE_MODE_TOOLTIP,
                }),
                "thinking_budget": ("INT", {
                    "default": 1024, "min": 0, "max": 16384, "step": 64,
                    "tooltip": THINKING_BUDGET_TOOLTIP,
                }),
                "second_thinking_budget": ("INT", {
                    "default": 512, "min": 0, "max": 16384, "step": 64,
                    "tooltip": THINKING_BUDGET_TOOLTIP,
                }),
            },
            "hidden": {"node_id": "UNIQUE_ID"},
        }
    # NOTE: "response_1_display"/"response_2_display"/"tags_1_display" are
    # deliberately NOT declared inputs - see the matching note in
    # mykee_prompt_modifier.py. They're created purely client-side in
    # web/mykee_prompt_advanced_ultimate.js and filled in only through the
    # {"ui": {...}} channel below.

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt_1", "prompt_2", "response_1", "response_2", "tags_1", "thinking_1", "thinking_2")
    FUNCTION = "run"
    CATEGORY = "Mykee/Prompt"
    OUTPUT_NODE = True

    def _generate_stage(self, clip, make_prompts, image, video, audio, max_length, sampling_mode,
                         seed, temperature, top_k, top_p, min_p,
                         repetition_penalty, presence_penalty, thinking,
                         stage_label, node_id, template_mode="auto", thinking_budget=0):
        """Shared tokenize+generate+decode call for one stage. Returns
        (answer, thinking_text, media_kind_used). The answer is the
        decoded text with the thinking block split off (or a bracketed [...]
        error string on failure, matching the other Prompt Advanced nodes'
        convention)."""
        try:
            tokens, chat_text, notes, used_kind = tokenize_for(
                clip, make_prompts, image=image, video=video, audio=audio, thinking=thinking,
                template_mode=template_mode)
        except Exception as e:
            return f"[{stage_label}: tokenization failed: {type(e).__name__}: {e}]", "", None
        for note in notes:
            _toast(node_id, "warn", "Mykee Prompt Advanced Ultimate", f"{stage_label}: {note}")

        try:
            raw_output = generate_text(
                clip, tokens, chat_text, max_length, sampling_mode, seed, temperature,
                top_k, top_p, min_p, repetition_penalty, presence_penalty, thinking,
                thinking_budget)
            text, thinking_text = split_thinking(raw_output)
            if not text:
                text = f"[{stage_label}: {empty_response_message(raw_output)}]"
            return text, thinking_text, used_kind
        except Exception as e:
            return (
                f"[{stage_label}: generation failed: {type(e).__name__}: {e}]\n"
                + generation_error_hint(used_kind is not None),
                "", used_kind,
            )

    def run(self, clip, prompt, prompt_rewrite, system_prompt, generate_tags, max_tags,
            max_length, sampling_mode, seed, temperature, top_k, top_p, min_p,
            repetition_penalty, presence_penalty, thinking,
            second_stage_enabled, second_mode, second_prompt, second_system_prompt,
            second_max_length, second_sampling_mode, second_seed, second_seed_control,
            second_temperature, second_top_k, second_top_p, second_min_p,
            second_repetition_penalty, second_presence_penalty, second_thinking,
            unload_after_run=False, image=None, video=None, audio=None,
            template_mode="auto", thinking_budget=1024, second_thinking_budget=512, node_id=None):

        second_prompt_text = second_prompt.strip() if second_prompt and second_prompt.strip() else ""

        if not prompt_rewrite:
            # Master switch off: no model calls at all, either stage.
            print("Mykee Prompt Advanced Ultimate: prompt_rewrite is off - "
                  "passing prompt through unchanged, stage 2 skipped.")
            response_1 = prompt
            response_2 = response_1
            tags_1 = ""
            return {
                "ui": {
                    "response_1_display": [response_1],
                    "response_2_display": [response_2],
                    "tags_1_display": [tags_1],
                    "thinking_1_display": [""],
                    "thinking_2_display": [""],
                },
                "result": (prompt, second_prompt, response_1, response_2, tags_1, "", ""),
            }

        # ---- stage 1 ---------------------------------------------------
        custom_sys_1 = system_prompt.strip() if system_prompt and system_prompt.strip() else None
        if custom_sys_1 and "{max_length}" in custom_sys_1:
            custom_sys_1 = custom_sys_1.format(max_length=int(max_length))

        def make_prompts_1(kind):
            if kind is None:
                sys_text = custom_sys_1 or STAGE1_TEXT_DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(max_length=int(max_length))
                user_text = prompt
                # The prompt is material to describe, not an instruction to
                # carry out - delimit it so the model can tell the difference.
                if prompt and prompt.strip():
                    user_text = quote_text(prompt)
            else:
                if custom_sys_1:
                    sys_text = custom_sys_1
                elif kind == "audio":
                    sys_text = AUDIO_ONLY_DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(max_length=int(max_length))
                else:
                    sys_text = (media_preamble(image, video, audio)
                                + STAGE1_IMAGE_DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(max_length=int(max_length)))
                user_text = prompt.strip() if prompt and prompt.strip() else DEFAULT_USER_TEXT[kind]
            if generate_tags:
                sys_text = sys_text + TAG_LIST_ADDENDUM_TEMPLATE
            return sys_text, user_text

        response_1, thinking_1, used_kind_1 = self._generate_stage(
            clip, make_prompts_1, image, video, audio, max_length, sampling_mode, seed, temperature,
            top_k, top_p, min_p, repetition_penalty, presence_penalty, thinking,
            "Stage 1", node_id, template_mode, thinking_budget,
        )

        # ---- stage 2 (text-only; only runs if enabled and non-empty) ---
        thinking_2 = ""
        run_stage_2 = bool(second_stage_enabled) and bool(second_prompt_text)
        if run_stage_2:
            custom_sys_2 = second_system_prompt.strip() if second_system_prompt and second_system_prompt.strip() else None
            if custom_sys_2 and "{max_length}" in custom_sys_2:
                custom_sys_2 = custom_sys_2.format(max_length=int(second_max_length))

            if second_mode == "edit":
                sys_text_2 = custom_sys_2 or TEXT_EDIT_DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(max_length=int(second_max_length))
                user_text_2 = f"PROMPT:\n{quote_text(response_1)}\n\nEDIT INSTRUCTION:\n{second_prompt_text}"
            else:  # "analyze"
                sys_text_2 = custom_sys_2 or STAGE2_DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(max_length=int(second_max_length))
                user_text_2 = f"RESPONSE 1:\n{quote_text(response_1)}\n\n{second_prompt_text}"

            response_2, thinking_2, _ = self._generate_stage(
                clip, lambda kind: (sys_text_2, user_text_2), None, None, None,
                second_max_length, second_sampling_mode, second_seed,
                second_temperature, second_top_k, second_top_p, second_min_p,
                second_repetition_penalty, second_presence_penalty, second_thinking,
                "Stage 2", node_id, template_mode, second_thinking_budget,
            )
            # edit mode's default system prompt (and any custom one using
            # the same convention) forces the model to write an ANALYSIS
            # line before the actual result - only the text after the
            # last "===FINAL===" marker is the real output. Falls back to
            # the full response if no marker is found (analyze mode's
            # default doesn't use one, nor does a custom prompt that
            # doesn't follow this format).
            extracted = _extract_final(response_2)
            if extracted is not None:
                response_2 = extracted
            elif second_mode == "edit":
                print("Mykee Prompt Advanced Ultimate: no '===FINAL===' "
                      "marker found in stage 2's response - using the "
                      "full response as-is.")
            response_2 = strip_quote_marks(response_2)
        else:
            response_2 = response_1

        tags_1 = _truncate_tags(_extract_tags(response_1)[1], max_tags) if generate_tags else ""

        # ---- stage 2 seed control (native widget, see _sampling_widgets) -
        # Computed after generation so this run always used the seed the
        # widget actually showed; the result is pushed back via "ui" the
        # same way response_1_display etc. are, updating the widget for
        # the NEXT run. "fixed" leaves it untouched.
        _SEED_MIN, _SEED_MAX = 0, 0xffffffffffffffff
        if second_seed_control == "randomize":
            next_second_seed = random.randint(_SEED_MIN, _SEED_MAX)
        elif second_seed_control == "increment":
            next_second_seed = min(_SEED_MAX, int(second_seed) + 1)
        elif second_seed_control == "decrement":
            next_second_seed = max(_SEED_MIN, int(second_seed) - 1)
        else:  # "fixed"
            next_second_seed = int(second_seed)

        print("=" * 70)
        print("Mykee Prompt Advanced Ultimate")
        print("=" * 70)
        print(f"Prompt (stage 1): {prompt!r}")
        print(f"Media sent to stage 1: {used_kind_1 or 'none (text-only)'}")
        print(f"Second stage: {'ran' if run_stage_2 else 'skipped'}")
        if second_prompt_text:
            print(f"Second prompt: {second_prompt_text!r}")
        print("")
        if thinking_1:
            print("--- thinking_1 ---")
            print(thinking_1)
        print("--- response_1 ---")
        print(response_1)
        if generate_tags:
            print("--- tags_1 ---")
            print(tags_1 or "(no '===TAGS===' marker found in response_1)")
        if run_stage_2:
            if thinking_2:
                print("--- thinking_2 ---")
                print(thinking_2)
            print("--- response_2 ---")
            print(response_2)

        if unload_after_run:
            _unload_clip(clip)

        return {
            "ui": {
                "response_1_display": [response_1],
                "response_2_display": [response_2],
                "tags_1_display": [tags_1],
                "thinking_1_display": [thinking_1],
                "thinking_2_display": [thinking_2],
                "second_seed": [next_second_seed],
            },
            "result": (prompt, second_prompt, response_1, response_2, tags_1, thinking_1, thinking_2),
        }


NODE_CLASS_MAPPINGS = {
    "MykeePromptAdvancedUltimate": MykeePromptAdvancedUltimate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeePromptAdvancedUltimate": "Mykee Prompt Advanced Ultimate",
}
