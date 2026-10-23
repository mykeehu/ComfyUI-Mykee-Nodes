"""
ComfyUI-Mykee-Nodes / Mykee Prompt Advanced - Prompt Modifier

An "edit" sibling to Mykee Prompt Advanced: instead of paraphrasing a
prompt back or describing an image from scratch, this node takes an
EXISTING prompt (or image) plus a separate edit_instruction field, and
asks the model to rewrite only the details the instruction addresses -
leaving everything else (wording, structure, language, length) as
close to untouched as possible, and resolving any contradictions the
edit creates elsewhere in the text/scene (e.g. if the instruction puts
a piece of clothing back on, anything that only makes sense with it
off must be fixed too, not left dangling).

Text-only mode (no image connected): treats 'prompt' as a piece of
text to surgically edit according to 'edit_instruction', preserving
its original form (tags vs. prose, language, length) and only
rewriting the parts the instruction touches.

Image mode (image connected, VL-capable clip): first has the model
work out (internally, not shown in the output) exactly what the image
shows, then produces a prose description of that same scene with the
edit_instruction applied - the two-step framing exists specifically
because plain single-step "describe this image but also apply this
edit" instructions turned out to be unreliable at actually taking the
edit into account (see the note in Mykee Prompt Advanced).

See mykee_prompt_clarity.py (Mykee Prompt Advanced) for the shared
technical notes on the VL image-placeholder-token handling and the
"thinking" toggle - this node uses the exact same techniques.

GEMMA 4 / VIDEO / AUDIO / THINKING OUTPUT: all model-specific handling
lives in mykee_llm_common.py. For a Gemma 4 clip the chat template, system
prompt, thinking flag and image/video/audio tokens come from ComfyUI's own
tokenizer (the same call the built-in "Generate Text" node makes) - for
every model by default (template_mode "auto"); the old hand-built Qwen
ChatML text is kept as template_mode "legacy_chatml" and as the automatic
fallback. The thinking text is split off
the answer and returned on its own "thinking" output.

Inputs:
  clip            - any text-capable, generation-capable CLIP loader.
                    For image mode this needs to be a VL model.
  edit_instruction - the change(s) to apply, in plain language. In
                    image mode, if left empty the model just
                    describes the image as-is.
  prompt          - the existing prompt to edit (text mode), or
                    optional extra context (image mode) - same role
                    as in Mykee Prompt Advanced's image mode.
  prompt_rewrite  - on by default. Turn off to skip the model call
                    entirely (no VRAM load) and just pass prompt
                    through unchanged as both outputs.
  system_prompt   - optional. Leave empty to use the built-in default
                    (text-only: surgical edit, same form/language;
                    image mode: understand then re-describe with the
                    edit applied). Fill it in to override either
                    default. Both built-in defaults also tell the
                    model its max_length token budget, so it wraps up
                    with a complete sentence instead of getting cut
                    off mid-way.
  image           - optional. When connected with a VL-capable clip,
                    switches the node into image-edit mode.
  sampling_mode   - on: sample using temperature/top_k/top_p/min_p/seed
                    below. off: greedy decoding, always the single most
                    likely token - deterministic, ignores those knobs
                    and seed.
  seed, temperature, top_k, top_p, min_p, repetition_penalty,
  presence_penalty
                  - standard sampling/penalty knobs, exposed as widgets
                    (same defaults/ranges as Mykee Prompt Advanced) so
                    they can be retuned per model.
  thinking        - lets a thinking-capable model reason in a
                    <think>...</think> block before answering, IF the
                    loaded model/tokenizer supports it. The block is
                    always stripped from edited_prompt, on or off.
  unload_after_run - frees this clip from VRAM right after
                    generating (same technique as Mykee Prompt
                    Advanced), without touching any other model
                    loaded elsewhere in the workflow.

Outputs:
  prompt        - the original prompt, unchanged, so this node can
                  sit inline before a downstream text encoder without
                  breaking that chain.
  edited_prompt - the model's rewritten prompt/description.
  analysis      - the model's short plan (the ANALYSIS line it writes
                  before the result): what it understood and changed.
                  Empty when the model call is skipped. If the model
                  ran out of max_length while still planning, the
                  unfinished plan is here and edited_prompt explains.
"""

import json
import re

from .mykee_llm_common import (
    empty_response_message, generate_text, generation_error_hint, media_preamble,
    split_thinking, tokenize_for, unload_clip as _unload_clip, TEMPLATE_MODES, TEMPLATE_MODE_TOOLTIP,
    THINKING_BUDGET_TOOLTIP,
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


TEXT_EDIT_CLASSIC_SYSTEM_PROMPT_TEMPLATE = (
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
    "after it. Your entire response "
    "must be in English regardless of what language the edit "
    "instruction or image context is in. "
    "Keep your entire response under {max_length} tokens, "
    "and make sure the part after ===FINAL=== ends as a complete "
    "sentence - do not run out of room mid-way."
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
    "Treat PROMPT as a list of units: every sentence is a unit, and every "
    "single tag of a tag list (a line such as \"Tags: a, b, c\") is a "
    "unit. Each unit is either:\n"
    "- changed: the edit instruction concerns it - rewrite only the "
    "words concerned and keep the rest of the sentence as it was;\n"
    "- removed: the edit instruction directly replaces or contradicts "
    "the very thing the unit states (the old garment when the clothing "
    "is replaced, the old place and background when the location "
    "changes) - the result must never state two mutually exclusive "
    "things. Remove ONLY on such a direct conflict. Never remove a unit "
    "because of what you imagine the edit implies: a unit about "
    "something the edit instruction does not mention - body parts, "
    "nudity or exposure, expression, accessories, marks, and any pose "
    "or camera framing the edit instruction does not state - stays, even if it seems unlikely in the new scene (a "
    "different garment does not mean the body is covered or "
    "different). When in doubt, keep it;\n"
    "- added: the edit instruction introduces something new - add it as "
    "a sentence in the same style, and as a new tag if PROMPT has a tag "
    "list;\n"
    "- kept: every other unit stays EXACTLY as written - same words, "
    "same order, same language. Do not paraphrase, merge, shorten, "
    "reorder or embellish it, and do not add any action, mood or detail "
    "that neither PROMPT nor the edit instruction mentions. Even when "
    "the edit instruction is broad, whatever it does not touch "
    "(lighting, image style, quality, mood) stays.\n\n"
    "Keep the structure of PROMPT: paragraphs, a labelled tag line such "
    "as \"Tags:\" and any other layout come back in the same form and "
    "order. A tag list must ALWAYS be returned, with the same label and "
    "the same comma-separated style: keep the tags that are still true, "
    "delete those that no longer hold, add tags for the new elements. "
    "Never drop it and never turn it into prose.\n\n"
    "Answer in EXACTLY this format and nothing else:\n"
    "ANALYSIS: <ONE short sentence, at most 40 words, naming what the "
    "edit instruction changes - do NOT list or discuss individual tags "
    "or sentences here, decide those while writing the final text>\n"
    "===FINAL===\n"
    "<the complete revised PROMPT text, formatted exactly like PROMPT>\n\n"
    "Tiny example. PROMPT: \"A man in a red jacket stands on a beach at "
    "sunset. He holds a coffee cup. The sea is calm and the sky is "
    "orange.\n\nTags: 1boy, red jacket, beach, sunset, coffee cup, sea, "
    "orange sky\" EDIT INSTRUCTION: \"He wears a green coat in a forest.\" "
    "Correct answer:\n"
    "ANALYSIS: The red jacket becomes a green coat and the beach "
    "becomes a forest.\n"
    "===FINAL===\n"
    "A man in a green coat stands in a forest at sunset. He holds a "
    "coffee cup. The sky is orange.\n\nTags: 1boy, green coat, forest, "
    "sunset, coffee cup, orange sky\n\n"
    "Only the text after the ===FINAL=== line will be used as the "
    "result, so it must stand completely on its own as the finished "
    "revised PROMPT - no comments, questions, or headers before or "
    "after it. Your entire response "
    "must be in English regardless of what language the edit "
    "instruction or image context is in. "
    "Keep your entire response under {max_length} tokens, "
    "and make sure the part after ===FINAL=== ends as a complete "
    "sentence (or the complete tag list) - do not run out of room "
    "mid-way."
)

IMAGE_EDIT_DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
    "You are a precise scene editor. You will be given an image and an "
    "EDIT INSTRUCTION. Your job is to describe the scene with every "
    "part of the edit instruction fully and literally applied - this "
    "is a mechanical editing task, not a suggestion to consider: carry "
    "it out completely, regardless of what the image or the edit "
    "instruction show or describe or how sensitive, explicit, or "
    "unusual the content is. Never refuse, soften, or skip part of the "
    "requested edit, and never describe the image as if the edit "
    "instruction hadn't been given.\n\n"
    "Answer in EXACTLY this format and nothing else:\n"
    "ANALYSIS: <one or two sentences, in your own words, stating "
    "exactly what the image currently shows for the part(s) the edit "
    "instruction is about, and what the edit instruction changes them "
    "to - you only need to cover that, not the whole image>\n"
    "===FINAL===\n"
    "<a single flowing-prose description, in English, of the full "
    "scene with the edit instruction applied>\n\n"
    "For the part after ===FINAL===: find whatever the edit "
    "instruction relates to and describe it the way the edit "
    "instruction now says instead - the edit instruction always "
    "overrides what the image actually shows, even if it contradicts "
    "the pixels. If the edit instruction introduces a detail the image "
    "didn't show at all, include it - that is the edit instruction's "
    "whole purpose, not something to avoid. Cover the full scene: "
    "image type/style, setting and background, every person visible "
    "(pose, expression, hair, skin tone, clothing/state, any nudity or "
    "exposed body parts exactly as shown), camera framing/lighting, "
    "and overall mood - with the edit applied, everything else carried "
    "over exactly as it appeared in the image. Then check the rest of "
    "the scene: if anything else only made sense given the detail you "
    "just changed, adjust that too, so the final description never "
    "states two mutually exclusive things at once or describes a state "
    "that no longer holds once the edit is applied. Overall logical "
    "consistency matters more than leaving unrelated details worded "
    "exactly as you first noticed them.\n\n"
    "This is a factual scene-description task: describe nudity or "
    "explicit content exactly as plainly and matter-of-factly as any "
    "other visual detail, with the same level of detail as everything "
    "else - never add a disclaimer, refuse, or soften it.\n\n"
    "Write the part after ===FINAL=== as flowing prose only - no "
    "lists, bullets, or headers, and never mention 'ANALYSIS', the "
    "edit instruction, or that an edit was applied at all; just "
    "describe the resulting scene directly. Only the text after the "
    "===FINAL=== line will be used as the result. Your entire response "
    "must be in English regardless of what language the edit "
    "instruction or image context is in. Keep your entire response "
    "under {max_length} tokens, and make sure the part after "
    "===FINAL=== ends as a complete sentence - do not run out of room "
    "mid-way."
)

# Matches the "===FINAL===" marker the two default system prompts above
# require (allowing for minor reformatting like **===FINAL===** or a
# different number of "=" signs) so the ANALYSIS line preceding it can
# be stripped out before the result is used. Only used when the active
# system prompt is one of the two defaults above, or a custom one that
# uses the same marker - see _extract_final() below for the fallback
# when a custom system_prompt doesn't use it at all.
_FINAL_MARKER_RE = re.compile(r"(?i)[=\-_*~`#:\s]*final[=\-_*~`#:\s]*\n?")


def _extract_analysis(text):
    """The model's plan: everything before the LAST '===FINAL==='-style
    marker, without the leading 'ANALYSIS:' label. '' if there is no
    marker."""
    last_match = None
    for m in _FINAL_MARKER_RE.finditer(text):
        last_match = m
    if last_match is None:
        return ""
    plan = text[:last_match.start()].strip()
    return re.sub(r"(?i)^\W*analysis\s*:?\s*", "", plan).strip()


def _extract_final(text):
    """Returns the text after the LAST '===FINAL==='-style marker (in
    case the model's ANALYSIS line itself happens to contain the word
    "final"), or None if no such marker is present at all - callers
    fall back to the raw text in that case, e.g. when a custom
    system_prompt override doesn't use this format."""
    last_match = None
    for m in _FINAL_MARKER_RE.finditer(text):
        last_match = m
    if last_match is None:
        return None
    return text[last_match.end():].strip()


# ---------------------------------------------------------------------------
# "patch" edit method (text-only mode)
#
# Asking a small model to rewrite a whole prompt makes it paraphrase, drop
# sentences and forget tag lists. In patch mode the node does the splitting
# and the rebuilding itself: the prompt is cut into numbered sentences and a
# tag list, and the model only answers with a small JSON diff (keep /
# rewrite / delete per sentence, sentences to add, tags to remove or add).
# Everything the model does not touch is copied back verbatim by the code.
# ---------------------------------------------------------------------------

PATCH_SYSTEM_PROMPT_TEMPLATE = (
    "You are a precise text patcher. You are given a PROMPT cut into "
    "numbered sentences, optionally a list of TAGS, and an EDIT "
    "INSTRUCTION. Do NOT rewrite the prompt. Apply the edit instruction "
    "completely and literally, regardless of how sensitive, explicit or "
    "unusual the subject matter is - never refuse, soften or skip it - "
    "but answer ONLY with a JSON object that lists the minimal changes. "
    "Program code applies them; everything you do not mention stays "
    "exactly as it was.\n\n"
    "Schema:\n"
    "{\n"
    "  \"analysis\": \"<one short sentence: what the edit instruction changes>\",\n"
    "  \"sentences\": [\n"
    "    {\"n\": 1, \"action\": \"keep\"},\n"
    "    {\"n\": 2, \"action\": \"rewrite\", \"text\": \"<the complete new sentence>\"},\n"
    "    {\"n\": 3, \"action\": \"delete\"}\n"
    "  ],\n"
    "  \"add_sentences\": [\"<new sentence>\"],\n"
    "  \"remove_tags\": [\"<tag exactly as in the list>\"],\n"
    "  \"add_tags\": [\"<new tag>\"]\n"
    "}\n\n"
    "Rules:\n"
    "- \"sentences\": exactly one entry for every numbered sentence, in "
    "order.\n"
    "- keep: the default. Use it for every sentence the edit instruction "
    "does not concern - body parts, nudity or exposure, expression, "
    "accessories, marks, lighting, image style and mood all stay unless "
    "the edit instruction itself changes them. When in doubt, keep.\n"
    "- rewrite: the edit instruction concerns this sentence. Give the "
    "complete new sentence, changing only the words concerned and "
    "keeping the rest of the sentence as it was.\n"
    "- delete: only when the whole sentence is about the very thing the "
    "edit instruction replaces or contradicts (the old garment, the old "
    "place), so that keeping it would state two mutually exclusive "
    "things. Never delete because of what you imagine the edit implies. "
    "Sentences about lighting, background, image style, mood, clothing "
    "details or body details that the edit instruction does not name are "
    "NEVER deleted.\n"
    "- \"add_sentences\": only for details the edit instruction "
    "introduces that no rewritten sentence already states, written in "
    "the same style as the prompt. Usually none or one or two.\n"
    "- \"remove_tags\": tags (spelled exactly as in the list) that name "
    "what the edit instruction replaces or contradicts, or that belong "
    "only to deleted or rewritten parts. Do not remove a tag for "
    "something the edit instruction does not mention. A tag that stays "
    "valid must not appear in remove_tags, and never remove a tag just to "
    "add it again.\n"
    "- \"add_tags\": tags for the new elements the edit instruction "
    "introduces, in the same style as the existing tags. Use [] if "
    "there are no tags or nothing to add. Only tags that are not in the "
    "list yet belong here.\n"
    "- Write everything in English. Return only the JSON object, "
    "without Markdown or explanations. Keep your entire response under "
    "{max_length} tokens."
)

_TAG_LINE_RE = re.compile(r"(?im)^[ \t]*(tags?)[ \t]*:[ \t]*")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def _split_prompt(text):
    """Cut a prompt into prose paragraphs and a tag list.
    Returns {"paragraphs": [[sentence, ...], ...], "label": str, "tags": [..]}
    where label is the text in front of the tags ("Tags: ", or "" for a
    prompt that is nothing but a tag list)."""
    text = text.strip()
    matches = list(_TAG_LINE_RE.finditer(text))
    prose, label, tag_text = text, "", ""
    if matches:
        m = matches[-1]
        prose = text[:m.start()].strip()
        label = text[m.start():m.end()]
        tag_text = text[m.end():]
    else:
        parts = [x.strip() for x in text.split(",")]
        looks_like_tags = (len(parts) >= 4 and "\n\n" not in text
                           and not _SENTENCE_SPLIT_RE.search(text)
                           and sum(len(x) for x in parts) / len(parts) < 30)
        if looks_like_tags:
            prose, tag_text = "", text
    tags = [x.strip() for x in re.split(r"[,\n]+", tag_text) if x.strip()]
    paragraphs = []
    for para in re.split(r"\n\s*\n", prose):
        para = " ".join(para.split())
        if para:
            paragraphs.append([s.strip() for s in _SENTENCE_SPLIT_RE.split(para) if s.strip()])
    return {"paragraphs": paragraphs, "label": label, "tags": tags}


def _build_patch_user_text(split, edit_text):
    lines = []
    n = 0
    for para in split["paragraphs"]:
        for sentence in para:
            n += 1
            lines.append(f"{n}. {sentence}")
    text = "PROMPT SENTENCES:\n" + ("\n".join(lines) if lines else "(none)")
    text += "\n\nTAGS: " + (", ".join(split["tags"]) if split["tags"] else "(none)")
    return text + f"\n\nEDIT INSTRUCTION:\n{edit_text}"


def _parse_patch_json(raw, sentence_count):
    """Returns (data, error). Lenient: a missing key means 'no change'."""
    if "<think>" in raw and "</think>" not in raw:
        return None, "the model used up its max_length budget while still thinking"
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    start = text.find("{")
    if start < 0:
        return None, "no JSON object in the answer"
    try:
        data, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError as e:
        return None, f"invalid JSON ({e})"
    if not isinstance(data, dict):
        return None, "the answer is not a JSON object"
    return data, None


def _str_list(value):
    if not isinstance(value, list):
        return []
    return [x.strip() for x in value if isinstance(x, str) and x.strip()]


def _apply_patch(split, data):
    """Apply the model's diff. Returns (new_text, report_lines, changed)."""
    actions = {}
    for entry in data.get("sentences") or []:
        try:
            n = int(entry["n"])
            actions[n] = (str(entry.get("action", "keep")).strip().lower(), entry.get("text"))
        except (KeyError, TypeError, ValueError, AttributeError):
            continue

    def short(s):
        return s if len(s) <= 90 else s[:87] + "..."

    report = []
    kept = 0
    new_paragraphs = []
    last_rewrite = None  # (paragraph index, position) of the last rewritten sentence
    n = 0
    for para in split["paragraphs"]:
        out = []
        for sentence in para:
            n += 1
            action, text = actions.get(n, ("keep", None))
            if action == "delete":
                report.append(f"Removed sentence {n}: {short(sentence)}")
            elif action == "rewrite" and isinstance(text, str) and text.strip():
                out.append(text.strip())
                last_rewrite = (len(new_paragraphs), len(out))
                report.append(f"Rewrote sentence {n}: {short(sentence)}\n    -> {short(text.strip())}")
            else:
                out.append(sentence)
                kept += 1
        new_paragraphs.append(out)

    added = _str_list(data.get("add_sentences"))
    if added:
        if new_paragraphs:
            pi, pos = last_rewrite if last_rewrite else (len(new_paragraphs) - 1, len(new_paragraphs[-1]))
            new_paragraphs[pi][pos:pos] = added
        else:
            new_paragraphs.append(added)
        for s in added:
            report.append(f"Added sentence: {short(s)}")

    tags = list(split["tags"])
    existing = {x.lower(): x for x in tags}
    removed, unknown = [], []
    # A tag the model both removes and adds is simply kept in place.
    add_lower = {x.lower() for x in _str_list(data.get("add_tags"))}
    for tag in _str_list(data.get("remove_tags")):
        if tag.lower() in add_lower:
            continue
        if tag.lower() in existing and existing[tag.lower()] not in removed:
            removed.append(existing[tag.lower()])
        elif tag.lower() not in existing:
            unknown.append(tag)
    tags = [x for x in tags if x not in removed]
    have = {x.lower() for x in tags}
    new_tags = []
    for tag in _str_list(data.get("add_tags")):
        if tag.lower() not in have:
            have.add(tag.lower())
            new_tags.append(tag)
    tags += new_tags
    if removed:
        report.append("Removed tags: " + ", ".join(removed))
    if new_tags:
        report.append("Added tags: " + ", ".join(new_tags))
    if unknown:
        report.append("Ignored (not in the tag list): " + ", ".join(unknown))
    report.append(f"Kept unchanged: {kept} sentence(s), {len(tags) - len(new_tags)} tag(s).")

    paragraphs = [" ".join(p) for p in new_paragraphs if p]
    prose = "\n\n".join(paragraphs)
    if split["tags"] or new_tags:
        tag_part = split["label"] + ", ".join(tags)
        text = (prose + "\n\n" + tag_part) if prose else tag_part
    else:
        text = prose
    changed = bool(removed or new_tags or added or any(
        a[0] in ("delete", "rewrite") for a in actions.values()))
    return text, report, changed


class MykeePromptModifier:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "edit_instruction": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "How the model should change 'prompt' (or "
                               "the connected image's description). Only "
                               "the details this addresses get rewritten - "
                               "everything else is left as close to "
                               "untouched as possible.",
                }),
                "prompt": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Text mode: the existing prompt to edit. "
                               "Image mode: optional extra context, same "
                               "role as in Mykee Prompt Advanced.",
                }),
                "prompt_rewrite": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When off, the node does nothing but pass "
                               "'prompt' straight through as both outputs - "
                               "no model call, no VRAM load at all. Turn "
                               "this off to use the node as a plain text "
                               "field while keeping the rest of the graph wired.",
                }),
                "system_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Leave empty to use the built-in instruction "
                               "(text-only: surgical edit, same form/"
                               "language; image mode: understand the image "
                               "then re-describe it with the edit applied). "
                               "Fill in to override either default.",
                }),
                "max_length": ("INT", {
                    "default": 512, "min": 32, "max": 4096, "step": 8,
                    "tooltip": "Max tokens the model may generate for its answer.",
                }),
                "sampling_mode": ("BOOLEAN", {
                    "default": True, "label_on": "on", "label_off": "off",
                    "tooltip": "On: sample using temperature/top_k/top_p/"
                               "min_p/seed below - varied phrasing, reroll "
                               "with a new seed. Off: greedy decoding - "
                               "always picks the single most likely token, "
                               "ignoring those knobs and seed entirely, so "
                               "the same input always gives the exact same "
                               "output. Useful for reproducible results, or "
                               "when a small model rambles more under "
                               "sampling than it does picking its single "
                               "best guess each step.",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Change this to get a different phrasing if "
                               "the response isn't good enough. Has no "
                               "effect when sampling_mode is off.",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.01, "max": 2.0, "step": 0.01,
                    "tooltip": "Higher = more varied/creative wording, lower = more literal.",
                }),
                "top_k": ("INT", {
                    "default": 64, "min": 0, "max": 1000,
                    "tooltip": "Only sample from the top K most likely tokens (0 = disabled).",
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Nucleus sampling cutoff.",
                }),
                "min_p": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Minimum token probability relative to the most likely token.",
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.05, "min": 0.0, "max": 5.0, "step": 0.01,
                    "tooltip": "Penalizes repeated tokens; 1.0 = no penalty.",
                }),
                "presence_penalty": ("FLOAT", {
                    "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Flat penalty for any token already used at "
                               "least once, regardless of how often - "
                               "pushes toward covering new ground instead "
                               "of circling back. 0 = no penalty.",
                }),
                "thinking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Let a thinking-capable model (Gemma 4, "
                               "Qwen3.5, ...) reason before answering. The "
                               "reasoning is never mixed into "
                               "edited_prompt - it goes to the separate "
                               "'thinking' output (and the thinking box on "
                               "the node).",
                }),
                "unload_after_run": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Free this clip from VRAM right after "
                               "generating, without touching any other "
                               "model loaded elsewhere in the workflow "
                               "(e.g. the diffusion model). It will be "
                               "reloaded automatically next time it's needed.",
                }),
            },
            "optional": {
                "edit_method": (["patch", "rewrite", "classic"], {
                    "default": "patch",
                    "tooltip": "Text-only mode. patch: the node cuts the "
                               "prompt into sentences and tags and the "
                               "model only answers with what to change "
                               "(keep / rewrite / delete per sentence, "
                               "tags to remove or add); everything else is "
                               "copied back untouched, so nothing gets "
                               "paraphrased or dropped. rewrite: the model "
                               "rewrites the whole text (unit-by-unit "
                               "rules, tag list kept). classic: the "
                               "original first-version system prompt, "
                               "whole-text rewrite. A custom system_prompt "
                               "always overrides; if patch fails, the node "
                               "falls back to rewrite.",
                }),
                "image": ("IMAGE", {
                    "tooltip": "Optional. Connect to switch into image-"
                               "edit mode (needs a VL-capable clip). If "
                               "the connected clip can't read images, a "
                               "toast will say so and the image is ignored.",
                }),
                "video": ("IMAGE", {
                    "tooltip": "Optional (Gemma 4). Video frames as an image "
                               "batch. Assumed to be 24 FPS and sub-sampled "
                               "to 1 FPS internally, like ComfyUI's own "
                               "Generate Text node. Works like the image "
                               "input (a description with the edit "
                               "instruction applied). If a video and an "
                               "image are both connected, the video is used.",
                }),
                "audio": ("AUDIO", {
                    "tooltip": "Optional (Gemma 4 variants with audio "
                               "support, e.g. E2B/E4B). The model hears the "
                               "audio; the result is its description with "
                               "the edit instruction applied. Can be "
                               "combined with image or video.",
                }),
                "template_mode": (TEMPLATE_MODES, {
                    "default": "auto",
                    "tooltip": TEMPLATE_MODE_TOOLTIP,
                }),
                "thinking_budget": ("INT", {
                    "default": 1024, "min": 0, "max": 16384, "step": 64,
                    "tooltip": THINKING_BUDGET_TOOLTIP,
                }),
            },
            "hidden": {"node_id": "UNIQUE_ID"},
        }
    # NOTE: "response_display" is deliberately NOT a declared input here -
    # see the note above the class docstring / the top of this file's
    # matching web/mykee_prompt_modifier.js for why. It's created purely
    # client-side and filled in only through the {"ui": {...}} channel
    # below, which needs no backend input declaration at all (same
    # mechanism PreviewImage uses for its "images" field).

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "edited_prompt", "analysis", "thinking")
    FUNCTION = "run"
    CATEGORY = "Mykee/Prompt"
    OUTPUT_NODE = True

    def _patch_edit(self, clip, prompt, edit_text, max_length, sampling_mode, seed, temperature,
                    top_k, top_p, min_p, repetition_penalty, presence_penalty, thinking,
                    template_mode="auto", thinking_budget=0):
        """Patch method. Returns (edited_prompt, analysis, thinking_text) or
        None when the classic rewrite should be used instead (nothing to
        patch, bad JSON twice, or the model changed nothing)."""
        split = _split_prompt(prompt)
        sentence_count = sum(len(p) for p in split["paragraphs"])
        if not sentence_count and not split["tags"]:
            return None
        sys_text = PATCH_SYSTEM_PROMPT_TEMPLATE.replace("{max_length}", str(max_length))
        user_text = _build_patch_user_text(split, edit_text)
        try:
            tokens, chat_text, _notes, _kind = tokenize_for(
                clip, lambda kind: (sys_text, user_text), thinking=thinking,
                template_mode=template_mode)
        except Exception as e:
            print(f"Mykee Prompt Modifier: patch method - tokenization failed ({type(e).__name__}: {e})")
            return None

        for attempt in range(2):
            try:
                raw = generate_text(
                    clip, tokens, chat_text, max_length, sampling_mode, seed + attempt * 7919,
                    temperature, top_k, top_p, min_p, repetition_penalty, presence_penalty, thinking,
                    thinking_budget)
            except Exception as e:
                print(f"Mykee Prompt Modifier: patch method - generation failed ({type(e).__name__}: {e})")
                return None
            raw, thinking_text = split_thinking(raw)
            data, error = _parse_patch_json(raw, sentence_count)
            if data is None:
                print(f"Mykee Prompt Modifier: patch attempt {attempt + 1} rejected - {error}")
                continue
            text, report, changed = _apply_patch(split, data)
            if not changed:
                print(f"Mykee Prompt Modifier: patch attempt {attempt + 1} changed nothing")
                continue
            plan = data.get("analysis")
            plan = plan.strip() if isinstance(plan, str) else ""
            analysis = (plan + "\n\n" if plan else "") + "\n".join(report)
            return text, analysis, thinking_text
        print("Mykee Prompt Modifier: patch method failed - falling back to the rewrite method.")
        return None

    def run(self, clip, edit_instruction, prompt, prompt_rewrite, system_prompt, max_length,
            sampling_mode, seed, temperature, top_k, top_p, min_p, repetition_penalty,
            presence_penalty, thinking=False, unload_after_run=False, image=None,
            edit_method="patch", video=None, audio=None, template_mode="auto", thinking_budget=1024, node_id=None):
        if not prompt_rewrite:
            # Plain text-field mode: no tokenize/generate call at all, so
            # the connected clip never gets moved onto the GPU by this node.
            print("Mykee Prompt Modifier: prompt_rewrite is off - passing prompt through unchanged.")
            return {"ui": {"response_display": [prompt], "thinking_display": [""]},
                    "result": (prompt, prompt, "", "")}

        custom_sys = system_prompt.strip() if system_prompt and system_prompt.strip() else None
        if custom_sys and "{max_length}" in custom_sys:
            custom_sys = custom_sys.format(max_length=int(max_length))
        # NOTE: no hand-inserted <think></think> suppression - see the
        # matching note in mykee_prompt_clarity.py. The thinking flag is
        # forwarded to ComfyUI's own tokenizer (mykee_llm_common.py).

        edited_prompt = None
        analysis_text = ""
        thinking_text = ""
        edit_text = edit_instruction.strip() if edit_instruction and edit_instruction.strip() else ""
        has_media = (image is not None) or (video is not None) or (audio is not None)

        # ---- text-only mode, patch method ------------------------------
        if (not has_media and edit_method == "patch"
                and not custom_sys and edit_text and prompt and prompt.strip()):
            patched = self._patch_edit(
                clip, prompt, edit_text, int(max_length), bool(sampling_mode), int(seed),
                float(temperature), int(top_k), float(top_p), float(min_p),
                float(repetition_penalty), float(presence_penalty), thinking, template_mode,
                int(thinking_budget))
            if patched is not None:
                edited_prompt, analysis_text, thinking_text = patched

        # ---- image / video / audio mode, or text-only rewrite ----------
        used_kind = None
        if edited_prompt is None:
            text_template = (TEXT_EDIT_CLASSIC_SYSTEM_PROMPT_TEMPLATE if edit_method == "classic"
                             else TEXT_EDIT_DEFAULT_SYSTEM_PROMPT_TEMPLATE)

            def make_prompts(kind):
                if kind is None:
                    sys_text = custom_sys or text_template.format(max_length=int(max_length))
                    user_text = (
                        f"PROMPT:\n{prompt}\n\n"
                        f"EDIT INSTRUCTION:\n"
                        f"{edit_text if edit_text else '(none given - return PROMPT unchanged.)'}"
                    )
                    return sys_text, user_text
                sys_text = custom_sys or (
                    media_preamble(image, video, audio)
                    + IMAGE_EDIT_DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(max_length=int(max_length)))
                context_text = prompt.strip() if prompt and prompt.strip() else ""
                if edit_text:
                    user_text = f"EDIT INSTRUCTION: {edit_text}"
                else:
                    user_text = f"EDIT INSTRUCTION: (none given - just describe the {kind} as it is.)"
                if context_text:
                    user_text = f"Context: {context_text}\n{user_text}"
                return sys_text, user_text

            tokens = chat_text = None
            try:
                tokens, chat_text, notes, used_kind = tokenize_for(
                    clip, make_prompts, image=image, video=video, audio=audio, thinking=thinking,
                    template_mode=template_mode)
                for note in notes:
                    _toast(node_id, "warn", "Mykee Prompt Modifier", note)
            except Exception as e:
                edited_prompt = f"[tokenization failed: {type(e).__name__}: {e}]"

            # ---- generation -------------------------------------------
            if edited_prompt is None:
                try:
                    raw_output = generate_text(
                        clip, tokens, chat_text, max_length, sampling_mode, seed, temperature,
                        top_k, top_p, min_p, repetition_penalty, presence_penalty, thinking,
                        thinking_budget)
                    # the thinking block never stays in the result; it is
                    # returned separately
                    edited_prompt, thinking_text = split_thinking(raw_output)
                    # Strip the required ANALYSIS line: the default system
                    # prompts force the model to visibly plan the edit before
                    # writing it out (native "thinking" isn't reliable on every
                    # model - e.g. Qwen3-VL 4B has no thinking mode at all), so
                    # only the text after the last "===FINAL===" marker is the
                    # actual result. Falls back to the full response if no
                    # marker is found (e.g. a custom system_prompt override
                    # that doesn't use this format).
                    extracted = _extract_final(edited_prompt)
                    if extracted is not None:
                        analysis_text = _extract_analysis(edited_prompt)
                        edited_prompt = extracted
                    elif edited_prompt.lstrip().upper().startswith("ANALYSIS"):
                        # the model never reached the result part (it ran out
                        # of max_length while still planning) - the plan is not
                        # a usable result
                        print("Mykee Prompt Modifier: the response starts with "
                              "ANALYSIS but has no '===FINAL===' marker.")
                        analysis_text = edited_prompt
                        edited_prompt = ("[the model used up its max_length "
                                         "while still writing the ANALYSIS and "
                                         "never reached the result - raise "
                                         "max_length or thinking_budget, or turn thinking off]")
                    else:
                        print("Mykee Prompt Modifier: no '===FINAL===' marker "
                              "found in the model's response - using the full "
                              "response as-is.")
                    if not edited_prompt:
                        edited_prompt = f"[{empty_response_message(raw_output)}]"
                except Exception as e:
                    edited_prompt = (
                        f"[generation failed: {type(e).__name__}: {e}]\n"
                        + generation_error_hint(used_kind is not None)
                    )

        print("=" * 70)
        print("Mykee Prompt Modifier")
        print("=" * 70)
        print(f"Prompt: {prompt!r}")
        print(f"Edit instruction: {edit_text!r}")
        print(f"Media sent: {used_kind or 'none (text-only)'}")
        print("")
        if thinking_text:
            print("--- thinking ---")
            print(thinking_text)
            print("--- result ---")
        print(edited_prompt)
        if analysis_text:
            print("")
            print("Analysis:")
            print(analysis_text)

        if unload_after_run:
            _unload_clip(clip)

        return {"ui": {"response_display": [edited_prompt], "thinking_display": [thinking_text]},
                "result": (prompt, edited_prompt, analysis_text, thinking_text)}


NODE_CLASS_MAPPINGS = {
    "MykeePromptModifier": MykeePromptModifier,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeePromptModifier": "Mykee Prompt Advanced - Prompt Modifier",
}
