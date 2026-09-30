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
"""

import re

try:
    from server import PromptServer
except ImportError:
    PromptServer = None

try:
    import comfy.model_management as model_management
except ImportError:
    model_management = None


def _unload_clip(clip):
    """Best-effort VRAM cleanup for just this node's clip, without
    touching any other model loaded elsewhere in the workflow. See the
    matching helper in mykee_prompt_clarity.py for the full rationale -
    kept as a local copy here since none of the pack's node files share
    a common utils module."""
    if model_management is None:
        return
    patcher = getattr(clip, "patcher", None)
    if patcher is None:
        return
    try:
        loaded = model_management.loaded_models()
        if patcher in loaded:
            loaded.remove(patcher)
        model_management.free_memory(1e30, model_management.get_torch_device(), loaded)
        model_management.soft_empty_cache(True)
    except Exception:
        pass
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass


def _count_prompt_tokens(tokens):
    """Best-effort count of how many tokens `tokens` (as returned by
    clip.tokenize()) actually holds.

    This matters because ComfyUI's CLIP.generate() forwards `max_length`
    straight through to the underlying model's HF-style generate() call,
    where max_length is - per HF's own docs - the length of the FULL
    sequence (prompt + newly generated tokens combined), not a budget for
    the new tokens alone. Without correcting for this, a long prompt (the
    system prompt plus the existing 'prompt' text being edited, which can
    easily be several hundred tokens on its own) eats directly into the
    same budget as the actual rewritten output, so raising the max_length
    widget mostly just buys room for the prompt that was already going to
    be there - the completion barely grows, and the result is truncated
    mid-sentence exactly where the combined budget runs out. Returns None
    if the tokens structure isn't the expected
    {encoder_name: [[(token_id, weight), ...], ...]} shape (older/newer
    ComfyUI versions may differ), so callers can fall back to a rough
    character-based estimate instead of silently guessing wrong.
    """
    try:
        total = 0
        for chunks in tokens.values():
            for chunk in chunks:
                total += len(chunk)
        return total
    except Exception:
        return None


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
                    "tooltip": "Let a thinking-capable model reason in a "
                               "<think>...</think> block before answering. "
                               "The block is stripped from edited_prompt "
                               "either way; this only controls whether the "
                               "model is allowed to use it at all.",
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
                "image": ("IMAGE", {
                    "tooltip": "Optional. Connect to switch into image-"
                               "edit mode (needs a VL-capable clip). If "
                               "the connected clip can't read images, a "
                               "toast will say so and the image is ignored.",
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

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "edited_prompt")
    FUNCTION = "run"
    CATEGORY = "Mykee/Prompt"
    OUTPUT_NODE = True

    def run(self, clip, edit_instruction, prompt, prompt_rewrite, system_prompt, max_length,
            sampling_mode, seed, temperature, top_k, top_p, min_p, repetition_penalty,
            presence_penalty, thinking=False, unload_after_run=False, image=None,
            node_id=None):
        if not prompt_rewrite:
            # Plain text-field mode: no tokenize/generate call at all, so
            # the connected clip never gets moved onto the GPU by this node.
            print("Mykee Prompt Modifier: prompt_rewrite is off - passing prompt through unchanged.")
            return {"ui": {"response_display": [prompt]}, "result": (prompt, prompt)}

        custom_sys = system_prompt.strip() if system_prompt and system_prompt.strip() else None
        if custom_sys and "{max_length}" in custom_sys:
            custom_sys = custom_sys.format(max_length=int(max_length))
        # NOTE: no hand-inserted <think></think> suppression - see the
        # matching note in mykee_prompt_clarity.py. We just forward
        # thinking=thinking to clip.tokenize() below.

        tokens = None
        edited_prompt = None
        edit_text = edit_instruction.strip() if edit_instruction and edit_instruction.strip() else ""

        # ---- image mode: try it first if an image is connected -----------
        if image is not None:
            sys_text = custom_sys or IMAGE_EDIT_DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(max_length=int(max_length))
            context_text = prompt.strip() if prompt and prompt.strip() else ""
            if edit_text:
                user_text = f"EDIT INSTRUCTION: {edit_text}"
            else:
                user_text = "EDIT INSTRUCTION: (none given - just describe the image as it is.)"
            if context_text:
                user_text = f"Context: {context_text}\n{user_text}"
            # Same VL image-placeholder handling as Mykee Prompt Advanced -
            # see the technical note at the top of mykee_prompt_clarity.py.
            chat_text = (
                f"<|im_start|>system\n{sys_text}<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{user_text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            try:
                tokens = clip.tokenize(chat_text, image=image, skip_template=True, min_length=1, thinking=thinking)
            except TypeError:
                _toast(
                    node_id, "warn", "Mykee Prompt Modifier",
                    "The connected clip doesn't support image input (not a "
                    "VL model) - the image was ignored.",
                )
                tokens = None
            except Exception as e:
                edited_prompt = f"[image tokenization failed: {type(e).__name__}: {e}]"

        # ---- text-only mode: no image, or image mode above didn't run ----
        if tokens is None and edited_prompt is None:
            sys_text = custom_sys or TEXT_EDIT_DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(max_length=int(max_length))
            user_text = (
                f"PROMPT:\n{prompt}\n\n"
                f"EDIT INSTRUCTION:\n"
                f"{edit_text if edit_text else '(none given - return PROMPT unchanged.)'}"
            )
            chat_text = (
                f"<|im_start|>system\n{sys_text}<|im_end|>\n"
                f"<|im_start|>user\n{user_text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            try:
                # this text also starts with "<|im_start|>", so it's used
                # verbatim - see the comment in the image branch above.
                tokens = clip.tokenize(chat_text, skip_template=True, min_length=1, thinking=thinking)
            except Exception as e:
                edited_prompt = f"[tokenization failed: {type(e).__name__}: {e}]"

        # ---- generation -----------------------------------------------
        if edited_prompt is None:
            # max_length here is a TOTAL sequence budget for clip.generate()
            # (prompt + completion), not a completion-only budget - see
            # _count_prompt_tokens() above. Inflate it by however many
            # tokens the prompt itself just used, so the `max_length`
            # widget keeps meaning what it visually promises: room for the
            # generated response, on top of however long the prompt is.
            prompt_token_count = _count_prompt_tokens(tokens)
            if prompt_token_count is None:
                # Structure wasn't recognized - fall back to a generous
                # ~3 chars/token estimate from the actual chat text sent in,
                # rather than risk under-shooting and truncating again.
                prompt_token_count = len(chat_text) // 3
            effective_max_length = prompt_token_count + int(max_length)
            try:
                generated_ids = clip.generate(
                    tokens,
                    do_sample=bool(sampling_mode),
                    max_length=effective_max_length,
                    temperature=float(temperature),
                    top_k=int(top_k),
                    top_p=float(top_p),
                    min_p=float(min_p),
                    repetition_penalty=float(repetition_penalty),
                    presence_penalty=float(presence_penalty),
                    seed=int(seed),
                )
                edited_prompt = clip.decode(generated_ids)
                # safety net in case a thinking-capable model still emits a
                # <think> block despite no thinking request
                edited_prompt = re.sub(r"<think>.*?(?:</think>|$)", "", edited_prompt, flags=re.DOTALL).strip()
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
                    edited_prompt = extracted
                else:
                    print("Mykee Prompt Modifier: no '===FINAL===' marker "
                          "found in the model's response - using the full "
                          "response as-is.")
                if not edited_prompt:
                    edited_prompt = "[the model returned an empty response]"
            except Exception as e:
                edited_prompt = (
                    f"[generation failed: {type(e).__name__}: {e}]\n"
                    "Make sure 'clip' here is a generation-capable chat "
                    "model, not an encoder with no LM head wired up for "
                    "generation in ComfyUI (e.g. Krea2's own Qwen3-VL "
                    "conditioning encoder)."
                )

        print("=" * 70)
        print("Mykee Prompt Modifier")
        print("=" * 70)
        print(f"Prompt: {prompt!r}")
        print(f"Edit instruction: {edit_text!r}")
        print(f"Image connected: {image is not None}")
        print("")
        print(edited_prompt)

        if unload_after_run:
            _unload_clip(clip)

        return {"ui": {"response_display": [edited_prompt]}, "result": (prompt, edited_prompt)}


NODE_CLASS_MAPPINGS = {
    "MykeePromptModifier": MykeePromptModifier,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeePromptModifier": "Mykee Prompt Advanced - Prompt Modifier",
}
