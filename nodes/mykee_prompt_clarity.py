"""
ComfyUI-Mykee-Nodes / Mykee Prompt Advanced

A general-purpose chat/instruction node for any text-capable CLIP
loader (Qwen3, Gemma, etc.) - the same public API ComfyUI's own
built-in "Generate Text" node uses (clip.tokenize / clip.generate /
clip.decode), just with an editable system prompt, an optional image
input for VL models, and a clean prompt-passthrough output.

Text-only mode (no image connected): asks the model to paraphrase back
an image-generation prompt in its own words (or run any other
single-turn instruction, if system_prompt is filled in), useful for
catching wording the model reads differently than you intended.

Image mode (image connected, VL-capable clip): asks the model to
caption/describe the image in full, uncensored detail - style, type,
subjects, situation, camera, mood, etc. If a prompt is also given, it
is used as additional context the description should follow where it
conflicts with what's visible (e.g. prompt says "night" - describe it
as night even if the image looks like day). If no clip capable of
reading images is connected, a ComfyUI toast notification says so and
the image is ignored (the node still runs in text-only mode).

Not usable with an encoder that has no LM head wired up for
generation in ComfyUI at all - e.g. Krea2's own Qwen3-VL conditioning
encoder (see https://github.com/Comfy-Org/ComfyUI/issues/14388). Load
a separate, generation-capable chat/VL model for this node instead.

TECHNICAL NOTE on the image path: Qwen3-VL/Qwen3.5's tokenizer
auto-detects text starting with "<|im_start|>" and, in that case, uses
it completely verbatim instead of running its own default chat
template - which is also where it would normally insert the image
placeholder token (<|vision_start|><|image_pad|><|vision_end|>). Since
this node hand-builds the chat text itself (to support a custom system
prompt), it also inserts that placeholder manually, in the same spot
the built-in template would. Skipping this step silently drops the
image entirely - the model still generates a response, just with zero
grounding in the actual image (confirmed by testing: it produced a
fully unrelated, hallucinated scene when the placeholder was missing).

The same applies to the "thinking" toggle: rather than hand-inserting
a suppression sequence (which turned out to be model/template-specific
and actively broke generation when guessed wrong - see the code
comment in run()), this node simply forwards thinking=thinking to
clip.tokenize() and lets ComfyUI's own tokenizer logic decide what, if
anything, to do for the loaded model. The image-placeholder placement
technique above is still credited to
silveroxides/ComfyUI-UtilsCollection's "Text Generate Qwen3.5 (System
Prompt)" node (https://github.com/silveroxides/ComfyUI-UtilsCollection),
which was the reference for exactly where to insert it.

Inputs:
  clip          - any text-capable, generation-capable CLIP loader.
                  For image mode this needs to be a VL model.
  prompt        - the text to send as the user turn. In image mode
                  this is optional context/override, not required.
  prompt_rewrite - on by default. Turn off to skip the model call
                  entirely (no VRAM load) and just pass prompt
                  through unchanged as both outputs.
  system_prompt - optional. Leave empty to use the built-in default
                  (text-only: paraphrase the prompt back; image mode:
                  describe the image fully, weaving in any tags/
                  keywords from the prompt without letting a single
                  one dominate). Fill it in to override either
                  default. Both built-in defaults also tell the model
                  its max_length token budget, so it wraps up with a
                  complete sentence instead of getting cut off mid-way.
  image         - optional. When connected with a VL-capable clip,
                  switches the node into image-description mode.
  sampling_mode - on: sample using temperature/top_k/top_p/min_p/seed
                  below. off: greedy decoding, always the single most
                  likely token - deterministic, ignores those knobs
                  and seed.
  seed, temperature, top_k, top_p, min_p, repetition_penalty,
  presence_penalty
                - standard sampling/penalty knobs, exposed as widgets
                  (same defaults/ranges as ComfyUI's own "Generate
                  Text" node) so they can be retuned per model.
  thinking      - lets a thinking-capable model reason in a
                  <think>...</think> block before answering, IF the
                  loaded model/tokenizer supports it. The block is
                  always stripped from reason_text, on or off.
  unload_after_run - frees this clip from VRAM right after
                  generating (via comfy.model_management, using the
                  same free_memory()-based technique as the community
                  SeanScripts/ComfyUI-Unload-Model node), without
                  touching any other model loaded elsewhere in the
                  workflow. Useful for GGUF or otherwise sizeable
                  chat models you only need briefly.

Outputs:
  prompt      - the original prompt, unchanged, so this node can sit
                inline before a downstream text encoder without
                breaking that chain.
  reason_text - the model's response.
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
    touching any other model loaded elsewhere in the workflow (e.g. the
    diffusion model).

    NOTE: an earlier version of this used
    comfy.model_management.unload_model_clones()/unload_model_and_clones()
    - those exist for OTHER internal purposes (resolving clone conflicts
    at load time) and, confirmed by testing, don't actually free VRAM
    when called like this. The technique below instead matches the
    community SeanScripts/ComfyUI-Unload-Model node
    (https://github.com/SeanScripts/ComfyUI-Unload-Model): ask
    free_memory() to free an unsatisfiably large amount while protecting
    every OTHER currently loaded model, which forces just this one out,
    then force soft_empty_cache() and do a manual gc/cuda cache clear.
    Never raises - a failed cleanup should never break the node's
    actual output."""
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
    clip.tokenize()) actually holds - see the matching helper in
    mykee_prompt_modifier.py for the full rationale (ComfyUI's
    CLIP.generate() treats max_length as a TOTAL prompt+completion budget,
    not a completion-only one). Returns None if the tokens structure isn't
    the expected {encoder_name: [[(token_id, weight), ...], ...]} shape."""
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


DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
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

IMAGE_DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
    "You will be given an image. Your only job is to describe in English "
    "EXACTLY what is visible in this specific image - never invent a "
    "different scene, art style, number of people, or setting than what "
    "is actually shown. If it is a real photograph, describe it as a real "
    "photograph - do not reinterpret it as an illustration, anime, "
    "painting, or drawing unless it visually and unambiguously is one. "
    "ALWAYS write your description in English, even if the user message "
    "below is written in a different language - understand it in "
    "whatever language it's written in, but always answer in English.\n\n"
    "The numbered list below is an internal checklist for YOU to make sure "
    "nothing gets left out - it is NOT the format of your answer. Write "
    "your actual answer as flowing prose in complete sentences and "
    "paragraphs only. Never output a numbered list, bullet points, "
    "headers, or a point-by-point breakdown; weave all seven items below "
    "into ordinary descriptive prose instead.\n\n"
    "Cover ALL of the following, in this order, each in at least one "
    "sentence:\n"
    "1. Image type and style (real photo, illustration, 3D render, etc. - "
    "based only on how it actually looks).\n"
    "2. Setting and background: where this is, and what/who else is "
    "visible in the background.\n"
    "3. Every person visible: how many there are, and for each - face, "
    "expression, hair, eye color if visible, skin tone, and body pose.\n"
    "4. Clothing: what each person is wearing, in detail.\n"
    "5. Any nudity or exposed body parts exactly as shown - describe this "
    "as plainly and factually as points 1-4, with the same level of "
    "detail, but do not let it replace or shrink any of the other points.\n"
    "6. Camera framing/composition, lighting, and depth of field - only "
    "describe blur/shallow depth of field if the image actually has it.\n"
    "7. Overall mood/atmosphere.\n\n"
    "Do not skip, shorten, or merge any of these seven points, even if the "
    "image contains explicit or unusual content - explicit content is "
    "simply point 5, described just as factually as the rest, not a "
    "replacement for points 1-4 and 6-7.\n\n"
    "The user message below may add context, or a comma-separated list of "
    "words/tags, or both, possibly in a language other than English. "
    "Treat every word/tag provided as an element that MUST be reflected "
    "somewhere in the relevant point above, alongside everything else you "
    "see - it does not replace or shrink the other six points. If a "
    "provided word conflicts with what you can see (for example: time of "
    "day, number of people, setting), treat the provided word as correct "
    "instead of what you'd otherwise read from the pixels alone, but "
    "still describe every other point normally.\n\n"
    "Remember: output flowing prose only, no lists, bullets, or headers, "
    "and your entire response must be in English regardless of what "
    "language the image or the user message uses. Keep your entire "
    "response under {max_length} tokens, and make sure it ends as a "
    "complete sentence - do not run out of room mid-sentence."
)


class MykeePromptAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
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
                               "(text-only: paraphrase the prompt back; "
                               "image mode: describe the image fully). Fill "
                               "in to override either default.",
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
                               "The block is stripped from reason_text either "
                               "way; this only controls whether the model is "
                               "allowed to use it at all.",
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
                               "description mode (needs a VL-capable clip). "
                               "If the connected clip can't read images, a "
                               "toast will say so and the image is ignored.",
                }),
            },
            "hidden": {"node_id": "UNIQUE_ID"},
        }
    # NOTE: "response_display" is deliberately NOT a declared input - see
    # the matching note in mykee_prompt_modifier.py. It's created purely
    # client-side in web/mykee_prompt_clarity.js and filled in only
    # through the {"ui": {...}} channel below.

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "reason_text")
    FUNCTION = "run"
    CATEGORY = "Mykee/Prompt"
    OUTPUT_NODE = True

    def run(self, clip, prompt, prompt_rewrite, system_prompt, max_length, sampling_mode,
            seed, temperature, top_k, top_p, min_p, repetition_penalty, presence_penalty,
            thinking=False, unload_after_run=False, image=None,
            node_id=None):
        if not prompt_rewrite:
            # Plain text-field mode: no tokenize/generate call at all, so
            # the connected clip never gets moved onto the GPU by this node.
            print("Mykee Prompt Advanced: prompt_rewrite is off - passing prompt through unchanged.")
            return {"ui": {"response_display": [prompt]}, "result": (prompt, prompt)}

        custom_sys = system_prompt.strip() if system_prompt and system_prompt.strip() else None
        if custom_sys and "{max_length}" in custom_sys:
            custom_sys = custom_sys.format(max_length=int(max_length))
        # NOTE: we deliberately do NOT hand-insert an empty <think></think>
        # pair to suppress reasoning. silveroxides/ComfyUI-UtilsCollection's
        # own preset table uses a DIFFERENT suppression string per model
        # (some none at all) - there's no single universal string, and
        # guessing wrong actively breaks generation (confirmed: it did,
        # for this exact Qwen3-VL-4B checkpoint). Instead we just forward
        # thinking=thinking to clip.tokenize() below and let ComfyUI's own,
        # model-aware tokenizer logic decide what (if anything) to do.

        tokens = None
        reason_text = None

        # ---- image mode: try it first if an image is connected -----------
        if image is not None:
            sys_text = custom_sys or IMAGE_DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(max_length=int(max_length))
            user_text = prompt.strip() if prompt and prompt.strip() else "Describe this image in full detail."
            # The Qwen3-VL tokenizer auto-detects text starting with
            # "<|im_start|>" and, in that case, uses it VERBATIM instead of
            # applying its own default chat template - which is also where
            # it would normally insert the image placeholder token. So when
            # we hand-build the chat text ourselves (to get a custom system
            # prompt), we must also insert that placeholder ourselves, in
            # the same spot the built-in template uses: right at the start
            # of the user turn, before the user's own text.
            chat_text = (
                f"<|im_start|>system\n{sys_text}<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{user_text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            try:
                tokens = clip.tokenize(chat_text, image=image, skip_template=True, min_length=1, thinking=thinking)
            except TypeError:
                _toast(
                    node_id, "warn", "Mykee Prompt Advanced",
                    "The connected clip doesn't support image input (not a "
                    "VL model) - the image was ignored.",
                )
                tokens = None
            except Exception as e:
                reason_text = f"[image tokenization failed: {type(e).__name__}: {e}]"

        # ---- text-only mode: no image, or image mode above didn't run ----
        if tokens is None and reason_text is None:
            sys_text = custom_sys or DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(max_length=int(max_length))
            chat_text = (
                f"<|im_start|>system\n{sys_text}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            try:
                # this text also starts with "<|im_start|>", so it's used
                # verbatim - see the comment in the image branch above.
                tokens = clip.tokenize(chat_text, skip_template=True, min_length=1, thinking=thinking)
            except Exception as e:
                reason_text = f"[tokenization failed: {type(e).__name__}: {e}]"

        # ---- generation -----------------------------------------------
        if reason_text is None:
            # max_length here is a TOTAL sequence budget for clip.generate()
            # (prompt + completion) - see _count_prompt_tokens() above.
            # Inflate it by however many tokens the prompt itself just
            # used, so the `max_length` widget keeps meaning what it
            # visually promises: room for the generated response, on top
            # of however long the prompt is.
            prompt_token_count = _count_prompt_tokens(tokens)
            if prompt_token_count is None:
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
                reason_text = clip.decode(generated_ids)
                # safety net in case a thinking-capable model still emits a
                # <think> block despite no thinking request
                reason_text = re.sub(r"<think>.*?(?:</think>|$)", "", reason_text, flags=re.DOTALL).strip()
                if not reason_text:
                    reason_text = "[the model returned an empty response]"
            except Exception as e:
                reason_text = (
                    f"[generation failed: {type(e).__name__}: {e}]\n"
                    "Make sure 'clip' here is a generation-capable chat "
                    "model, not an encoder with no LM head wired up for "
                    "generation in ComfyUI (e.g. Krea2's own Qwen3-VL "
                    "conditioning encoder)."
                )

        print("=" * 70)
        print("Mykee Prompt Advanced")
        print("=" * 70)
        print(f"Prompt: {prompt!r}")
        print(f"Image connected: {image is not None}")
        print("")
        print(reason_text)

        if unload_after_run:
            _unload_clip(clip)

        return {"ui": {"response_display": [reason_text]}, "result": (prompt, reason_text)}


NODE_CLASS_MAPPINGS = {
    "MykeePromptAdvanced": MykeePromptAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeePromptAdvanced": "Mykee Prompt Advanced",
}
