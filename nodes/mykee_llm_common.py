"""
ComfyUI-Mykee-Nodes / shared LLM helpers for the Prompt Advanced family

Used by Mykee Prompt Advanced, Mykee Prompt Advanced - Prompt Modifier and
Mykee Prompt Advanced Ultimate. This file registers no nodes - it only holds
the code those three nodes share, so the model-specific handling lives in ONE
place:

  * Model family detection (only Gemma 4 has video/audio support).
  * Building the tokens for one chat turn:
      - Gemma 4 uses ComfyUI's OWN chat template (the same call the built-in
        "Generate Text" node makes): system prompt, thinking flag and the
        image / video / audio placeholders are all produced by ComfyUI's
        Gemma4 tokenizer. Hand-building Qwen-style "<|im_start|>" text for a
        Gemma 4 model (what these nodes used to do for every model) makes
        Gemma see garbage and never receive the image - hence the nonsense
        answers.
      - Every other model uses the same native call by default
        (template_mode "auto"/"native"; image only); the previous,
        hand-built Qwen ChatML text stays available as "legacy_chatml" and
        as the automatic fallback.
  * Splitting the decoded output into the answer and the thinking text.
    ComfyUI's Gemma4 tokenizer already rewrites Gemma's
    "<|channel>thought ... <channel|>" block into "<think> ... </think>",
    so one splitter serves both families.

Repository of the ComfyUI sources this follows (comfy_extras/nodes_textgen.py
= the built-in "Generate Text" node, comfy/text_encoders/gemma4.py = the
Gemma 4 tokenizer / template): https://github.com/comfy-org/comfyui
"""

import re

try:
    import comfy.model_management as model_management
except ImportError:
    model_management = None


# --------------------------------------------------------------------------
# Model family detection
# --------------------------------------------------------------------------

def is_gemma4(clip):
    """True for a Gemma 4 text encoder / chat model. Same check the built-in
    TextGenerateLTX2Prompt node uses (tokenizer.clip_name), plus a class-name
    fallback."""
    tok = getattr(clip, "tokenizer", None)
    if "gemma4" in str(getattr(tok, "clip_name", "")).lower():
        return True
    for obj in (tok, getattr(clip, "cond_stage_model", None)):
        if "gemma4" in type(obj).__name__.lower():
            return True
    return False


def family_name(clip):
    return "gemma4" if is_gemma4(clip) else "default"


def qwen35_think_suffix(clip, thinking):
    """Assistant-turn suffix for Qwen3.5-family clips (default path only).

    Qwen3.5 thinks by default. ComfyUI's tokenizer only adds the <think>
    control text when it applies its own chat template, and that is skipped
    on the default path because the chat text is built by hand (it starts
    with "<|im_start|>"). So add exactly what ComfyUI's own template would
    add - for Qwen3.5 clips only; every other model is left untouched."""
    for obj in (getattr(clip, "tokenizer", None), getattr(clip, "cond_stage_model", None)):
        if "qwen35" in type(obj).__name__.lower():
            return "<think>\n" if thinking else "<think>\n\n</think>\n\n"
    return ""


def _prompt_ends_with_think_open(clip, tokens):
    """True if the tokenized prompt itself ends with an opened "<think>"
    (thinking on: the template already opened the block, so the model's
    output starts INSIDE it). Same check the built-in Generate Text node
    does."""
    try:
        if isinstance(tokens, dict):
            tokens = next(iter(tokens.values()))
        ids = [t[0] for t in tokens[-1] if isinstance(t, tuple) and isinstance(t[0], int)]
        return clip.decode(ids[-4:]).rstrip().endswith("<think>")
    except Exception:
        return False


def restore_think_open(clip, tokens, raw_output):
    """When the prompt already ended with the opening "<think>", the decoded
    output has no opening tag. Put it back so split_thinking() treats the
    reasoning as one block (and an unfinished block is recognised as one,
    not mistaken for the answer). A no-op otherwise."""
    if "</think>" in raw_output or raw_output.lstrip().startswith("<think>"):
        return raw_output
    if _prompt_ends_with_think_open(clip, tokens):
        return "<think>\n" + raw_output
    return raw_output


# --------------------------------------------------------------------------
# Tokenizing one chat turn
# --------------------------------------------------------------------------

class MediaNotSupported(Exception):
    """The connected clip can't take the connected media (handled by
    tokenize_for(), which falls back to a text-only turn)."""


class LegacyChatText(str):
    """Marks chat text that was built by hand (Qwen ChatML)."""


TEMPLATE_MODES = ["auto", "native", "legacy_chatml"]
TEMPLATE_MODE_TOOLTIP = (
    "How the chat template is built. auto (default): ComfyUI's own "
    "tokenizer builds it for whatever model is loaded - exactly what the "
    "built-in Generate Text node does, so it works with any model ComfyUI "
    "has a chat template for (Gemma 4, Qwen3-VL, Qwen3.5, ...); if that "
    "fails on the connected clip, the node falls back to the old Qwen "
    "ChatML text. native: ComfyUI's template only, errors are shown. "
    "legacy_chatml: the old hand-built Qwen-style '<|im_start|>' text "
    "(image only) - for a model where the native template misbehaves. "
    "Gemma 4 always uses the native template."
)


def _qwen_chat_text(sys_text, user_text, with_image):
    vision = "<|vision_start|><|image_pad|><|vision_end|>" if with_image else ""
    return LegacyChatText(
        f"<|im_start|>system\n{sys_text}<|im_end|>\n"
        f"<|im_start|>user\n{vision}{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _has_embeds(tokens):
    """True if the tokens carry at least one media embed (a dict where the
    token id would be) - i.e. the tokenizer really took the media."""
    try:
        if isinstance(tokens, dict):
            tokens = next(iter(tokens.values()))
        return any(isinstance(t, tuple) and isinstance(t[0], dict) for chunk in tokens for t in chunk)
    except Exception:
        return True  # unknown structure - don't second-guess


def _tokenize_native(clip, sys_text, user_text, image, video, audio, thinking):
    notes = []
    gemma = is_gemma4(clip)
    media = {}
    if gemma:
        if video is not None:
            media["video"] = video
            if image is not None:
                notes.append("Gemma 4 takes either a video or an image per turn - the image was ignored, the video is used.")
        elif image is not None:
            media["image"] = image
        if audio is not None:
            media["audio"] = audio
    else:
        # only Gemma 4 has video/audio support in ComfyUI
        if image is None and (video is not None or audio is not None):
            raise MediaNotSupported(
                "The connected clip can't take video or audio input (that needs "
                "Gemma 4) - the media was ignored, running text-only.")
        if video is not None:
            notes.append("The connected clip has no video input - the video was ignored (video needs Gemma 4).")
        if audio is not None:
            notes.append("The connected clip has no audio input - the audio was ignored (audio needs Gemma 4).")
        if image is not None:
            media["image"] = image
    # skip_template=False: ComfyUI's tokenizer builds the model's own chat
    # template (system turn, thinking flag, media placeholders) itself.
    try:
        tokens = clip.tokenize(
            user_text, skip_template=False, min_length=1, thinking=bool(thinking),
            system_prompt=sys_text or "", **media,
        )
    except TypeError:
        if media:
            raise MediaNotSupported(
                "The connected clip doesn't support this media input (not a "
                "VL model) - it was ignored, running text-only.")
        raise
    if media and not _has_embeds(tokens):
        raise MediaNotSupported(
            "The connected clip's tokenizer did not take the media (not a "
            "VL model?) - it was ignored, running text-only.")
    return tokens, f"{sys_text}\n{user_text}", notes


def _tokenize_legacy(clip, sys_text, user_text, image, video, audio, thinking):
    """Qwen-style ChatML built by hand, image only."""
    notes = []
    if image is None and (video is not None or audio is not None):
        raise MediaNotSupported(
            "The connected clip can't take video or audio input (that needs "
            "Gemma 4) - the media was ignored, running text-only.")
    if video is not None:
        notes.append("The connected clip has no video input - the video was ignored (video needs Gemma 4).")
    if audio is not None:
        notes.append("The connected clip has no audio input - the audio was ignored (audio needs Gemma 4).")

    suffix = qwen35_think_suffix(clip, thinking)
    if image is not None:
        chat_text = LegacyChatText(_qwen_chat_text(sys_text, user_text, True) + suffix)
        try:
            tokens = clip.tokenize(chat_text, image=image, skip_template=True, min_length=1, thinking=thinking)
            return tokens, chat_text, notes
        except TypeError:
            raise MediaNotSupported(
                "The connected clip doesn't support image input (not a VL "
                "model) - the image was ignored, running text-only.")
    chat_text = LegacyChatText(_qwen_chat_text(sys_text, user_text, False) + suffix)
    tokens = clip.tokenize(chat_text, skip_template=True, min_length=1, thinking=thinking)
    return tokens, chat_text, notes


def tokenize_chat(clip, sys_text, user_text, image=None, video=None, audio=None,
                  thinking=False, template_mode="auto"):
    """Tokenize one system+user turn for generation.

    Returns (tokens, chat_text, notes):
      tokens    - what clip.generate() takes.
      chat_text - the text that was tokenized (only used for a length
                  estimate; a LegacyChatText when built by hand).
      notes     - list of short human-readable warnings (shown as toasts).

    template_mode: see TEMPLATE_MODES / TEMPLATE_MODE_TOOLTIP.

    Raises MediaNotSupported when the clip can't take the media, and
    whatever clip.tokenize() raises for a real failure."""
    if is_gemma4(clip):
        tokens, chat_text, notes = _tokenize_native(clip, sys_text, user_text, image, video, audio, thinking)
        if template_mode == "legacy_chatml":
            notes.append("Gemma 4 can't use the legacy ChatML text - the native template was used.")
        return tokens, chat_text, notes
    if template_mode == "legacy_chatml":
        return _tokenize_legacy(clip, sys_text, user_text, image, video, audio, thinking)
    if template_mode == "native":
        return _tokenize_native(clip, sys_text, user_text, image, video, audio, thinking)
    # auto: native first, old ChatML text if the native call blows up
    try:
        return _tokenize_native(clip, sys_text, user_text, image, video, audio, thinking)
    except MediaNotSupported:
        raise
    except Exception as e:
        tokens, chat_text, notes = _tokenize_legacy(clip, sys_text, user_text, image, video, audio, thinking)
        notes.append(f"ComfyUI's native template failed for this clip ({type(e).__name__}: {e}) - used the legacy ChatML text.")
        return tokens, chat_text, notes


def tokenize_for(clip, make_prompts, image=None, video=None, audio=None, thinking=False,
                 template_mode="auto"):
    """Tokenize with media if any is connected and the clip can take it,
    otherwise fall back to a text-only turn.

    make_prompts(kind) -> (sys_text, user_text), where kind is 'image',
    'video', 'audio' or None (= text-only).

    Returns (tokens, chat_text, notes, kind_used). notes are warnings for a
    toast; kind_used is what was really sent (None = text-only)."""
    kind = media_kind(image, video, audio)
    notes = []
    if kind is not None:
        sys_text, user_text = make_prompts(kind)
        try:
            tokens, chat_text, n = tokenize_chat(clip, sys_text, user_text, image, video, audio, thinking, template_mode)
            return tokens, chat_text, n, kind
        except MediaNotSupported as e:
            notes.append(str(e))
    sys_text, user_text = make_prompts(None)
    tokens, chat_text, n = tokenize_chat(clip, sys_text, user_text, thinking=thinking, template_mode=template_mode)
    return tokens, chat_text, notes + n, None


def completion_budget(clip, tokens, max_length, chat_text):
    """The max_length to hand to clip.generate().

    Native template (Gemma 4 and everything built by ComfyUI): exactly the
    widget value, like the built-in node (current ComfyUI counts it as
    completion tokens only, and media soft-tokens are not part of the text
    length anyway).

    Legacy ChatML text: unchanged from before - the prompt's own token count
    is added on top, because the ComfyUI build that path was tested on
    treated max_length as a TOTAL prompt+completion budget."""
    max_length = int(max_length)
    if not isinstance(chat_text, LegacyChatText):
        return max_length
    count = None
    try:
        count = sum(len(chunk) for chunks in tokens.values() for chunk in chunks)
    except Exception:
        pass
    if count is None:
        count = len(chat_text) // 3
    return count + max_length


def generate_text(clip, tokens, chat_text, max_length, sampling_mode, seed, temperature,
                  top_k, top_p, min_p, repetition_penalty, presence_penalty, thinking,
                  thinking_budget=0):
    """clip.generate() + decode. Returns the raw decoded text (still
    containing any <think>...</think> block - use split_thinking()).

    thinking_budget: extra tokens added on top of max_length when thinking
    is on. The reasoning is generated BEFORE the answer and eats from the
    same token budget, so without extra room a long thought leaves nothing
    (or too little) for the answer itself."""
    if thinking:
        max_length = int(max_length) + max(0, int(thinking_budget))
    generated_ids = clip.generate(
        tokens,
        do_sample=bool(sampling_mode),
        max_length=completion_budget(clip, tokens, max_length, chat_text),
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        min_p=float(min_p),
        repetition_penalty=float(repetition_penalty),
        presence_penalty=float(presence_penalty),
        seed=int(seed),
    )
    return restore_think_open(clip, tokens, clip.decode(generated_ids))


# --------------------------------------------------------------------------
# Default system prompt pieces for video / audio input
# --------------------------------------------------------------------------

def media_kind(image, video, audio):
    """'video', 'image', 'audio' or None - which media decides the mode.
    A video wins over an image (Gemma 4 can only take one of the two)."""
    if video is not None:
        return "video"
    if image is not None:
        return "image"
    if audio is not None:
        return "audio"
    return None


VIDEO_PREAMBLE = (
    "The visual input is a VIDEO, not a still image: you receive frames "
    "sampled once per second, each labeled with its timestamp. Wherever the "
    "instructions below say \"image\", apply them to the video as a whole - "
    "describe what is visible and how it changes over time, in chronological "
    "order, and mention the timestamps where something happens or changes.\n\n"
)

AUDIO_ADDENDUM = (
    "An AUDIO recording is also provided together with the visual input. "
    "After describing what is visible, add what is HEARD: any speech "
    "(quoted word for word in its original language, plus who seems to say "
    "it and in what tone), music (genre, mood, instruments), and other "
    "sounds or ambience. Describe only what is actually audible.\n\n"
)

AUDIO_ONLY_PREAMBLE = (
    "The input is an AUDIO recording only - there is no image. Wherever the "
    "instructions below refer to what is visible, apply them to what is "
    "HEARD instead: speech (quoted word for word in its original language), "
    "music, sounds and ambience.\n\n"
)

AUDIO_ONLY_DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
    "You will be given an audio recording. Describe in English EXACTLY what "
    "is audible in it - never invent sounds, words or speakers that are not "
    "there. Cover: any speech, quoted word for word in its original "
    "language (with who seems to be speaking, their tone and emotion, and "
    "the language), music (genre, mood, instruments, tempo), other sounds, "
    "and the acoustic setting/ambience. Describe explicit or sexual audio "
    "plainly and factually, exactly like any other sound - without "
    "censorship, refusal, disclaimers, or moralizing.\n\n"
    "If the user message below contains a question or focus, explicitly "
    "confirm or rule out every audible detail relevant to it. Write flowing "
    "prose only - no lists, bullets, or headers. Keep your entire response "
    "under {max_length} tokens, and make sure it ends as a complete sentence."
)

DEFAULT_USER_TEXT = {
    "image": "Describe this image in full detail.",
    "video": "Describe this video in full detail.",
    "audio": "Describe this audio in full detail.",
}


def media_preamble(image, video, audio):
    """Text put in front of an image-oriented system prompt when video
    and/or audio is connected. Image-only input gets nothing (unchanged
    behavior)."""
    kind = media_kind(image, video, audio)
    if kind == "video":
        return VIDEO_PREAMBLE + (AUDIO_ADDENDUM if audio is not None else "")
    if kind == "image":
        return AUDIO_ADDENDUM if audio is not None else ""
    if kind == "audio":
        return AUDIO_ONLY_PREAMBLE
    return ""


# --------------------------------------------------------------------------
# Splitting the output
# --------------------------------------------------------------------------

_THINK_BLOCK_RE = re.compile(r"<think>(.*?)(?:</think>|$)", re.DOTALL)


def split_thinking(text):
    """Split decoded model output into (answer, thinking).

    Handles complete "<think>...</think>" blocks, a block the model never
    closed (it ran out of max_length while thinking - thinking is kept, the
    answer is then empty) and a lone "</think>" with no opening tag."""
    blocks = [m.group(1).strip() for m in _THINK_BLOCK_RE.finditer(text)]
    answer = _THINK_BLOCK_RE.sub("", text)
    if "<think>" not in text and "</think>" in answer:
        before, _, after = answer.rpartition("</think>")
        blocks.append(before.strip())
        answer = after
    answer = answer.replace("</think>", "").strip()
    thinking = "\n\n".join(b for b in blocks if b)
    return answer, thinking


def empty_response_message(raw_output):
    """Explain WHY the stripped response is empty."""
    if "<think>" in raw_output and "</think>" not in raw_output:
        return ("the model used up its token budget while still thinking - "
                "raise thinking_budget (or max_length) or turn thinking off")
    return "the model returned an empty response"


THINKING_BUDGET_TOOLTIP = (
    "Extra tokens added on top of max_length while thinking is on. The "
    "model's reasoning is generated before the answer and uses the same "
    "token budget, so a long thought can leave no room for the answer - "
    "this gives the thinking its own room. Has no effect when thinking is "
    "off. Raise it if the answer comes back empty or cut off while thinking."
)


def generation_error_hint(media_used):
    hint = ("Make sure 'clip' here is a generation-capable chat model, not an "
            "encoder with no LM head wired up for generation in ComfyUI (e.g. "
            "Krea2's own Qwen3-VL conditioning encoder).")
    if media_used:
        hint += (" A video/audio input also needs a Gemma 4 variant that has "
                 "audio/vision weights loaded (e.g. E2B/E4B for audio).")
    return hint


# --------------------------------------------------------------------------
# VRAM cleanup (shared copy of the helper the three nodes used to duplicate)
# --------------------------------------------------------------------------

def unload_clip(clip):
    """Best-effort VRAM cleanup for just this clip, without touching any
    other model loaded elsewhere in the workflow."""
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
