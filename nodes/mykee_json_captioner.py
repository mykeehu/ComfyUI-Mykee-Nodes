"""
ComfyUI-Mykee-Nodes / Mykee JSON Captioner

Captions an image with a generation-capable VL clip (e.g. Qwen3.5-9B loaded
through the normal Load CLIP node) and returns a ready-to-use MOSS-Image
prompt. Two modes:

  layout   The MOSS-Image Layout JSON (summary + items with bounding boxes):

             {"summary":{"caption":"..."},
              "items":[{"label":"object","caption":"a woman",
                        "bbox":[-117,-295,724,82]}, ...],
              "bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}

           The model only reports each box as normalized 0-1000
           [y_min, x_min, y_max, x_max] (the convention MOSS's own
           captioning prompt uses). The node converts them to pixels on the
           canvas (width/height outputs, multiples of 16) and then to
           centered yxyx integers with the same arithmetic as MOSS's
           compile_prompt.py - the model never calculates those numbers.
           Use the width/height outputs as the generation size.

  caption  Plain caption of the chosen length. The built-in prompt is the
           "Caption Prompt" published in the MOSS-Image user guide
           (Post-Training / Image Captioning); it also reports watermarks
           (box_2d, normalized 0-1000) and borders, which come out as
           separate outputs.

`caption_length` (long / medium / short) picks which single description is
used - as summary.caption in layout mode, as the caption in caption mode.

On top of a plain "Prompt Advanced" call the node:
  * parses and VALIDATES the model's JSON (code fences and trailing chatter
    are tolerated); on a malformed answer it regenerates with a different
    seed, up to `retries` times
  * captions an IMAGE batch one by one with the clip loaded once (all
    outputs are lists, so they map onto a save-per-item node)
  * downscales large images to `image_megapixels` first - ComfyUI feeds VL
    models the full-resolution image (up to ~12.8 MP), which costs a lot of
    tokens and VRAM; the max_length budget accounts for the image tokens

Qwen3.5 thinks by default; the node adds the same <think> prefix ComfyUI's
own chat template would (see _think_suffix). MOSS recommends thinking for
more accurate captions - turn `thinking` on and raise max_length (the
thinking text counts against it).

Outputs (all lists, one entry per image):
  json, caption, watermark, borders, width, height, bbox_preview
  (bbox_preview: the image resized to the canvas with the detected boxes
  drawn on it - layout items, or the watermark boxes in caption mode)
  On failure `json` holds "[MOSS captioning failed: reason]" and the text
  outputs are empty strings.
"""
import json
import math
import re

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - ComfyUI always has torch
    torch = None
    F = None

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - ComfyUI always has these
    np = Image = ImageDraw = ImageFont = None

try:
    import comfy.model_management as model_management
except ImportError:
    model_management = None

try:
    import comfy.utils as comfy_utils
except ImportError:
    comfy_utils = None

try:
    from server import PromptServer
except ImportError:
    PromptServer = None


# Verbatim from the MOSS-Image user guide (Apache-2.0), "Caption Prompt".
# {{reference}} is replaced with the optional reference caption.
MOSS_CAPTION_PROMPT = r'''You are an image recaptioner. Return exactly one JSON object with this schema:
{
  "caption": {
    "long": "a detailed image description, including all non-watermark visible text",
    "medium": "a one-paragraph image description",
    "short": "a one-sentence image description"
  },
  "watermark": [
    {"box_2d": [y_min, x_min, y_max, x_max], "label": "watermark text"}
  ],
  "borders": ["none"]
}

Rules:
- caption.long:
    - Describe the composition, key subjects, actions, setting, colors, lighting, and artistic style in detail. Include only information visible in the image.
    - Preserve visible text exactly, excluding watermarks. Do not translate, paraphrase, romanize, or normalize image text.
    - Describe text layout as much as possible: line breaks, ordering, relative position, alignment, size hierarchy, font/style, color, spacing, and how text blocks are arranged.
- caption.medium: compress long into one paragraph.
- caption.short: summarize the main subject, action, or relationship in one sentence.
- Do not describe watermarks in any caption field. Record them only in watermark.
- watermark:
    - Use normalized 0–1000 [y_min, x_min, y_max, x_max] coordinates relative to the full image.
    - Set label to the visible watermark text, or a short description if unreadable.
    - Use [] if there are no watermarks.
- borders:
    - Record evident borders or margins using one or more of "left", "right", "up", and "down".
    - Use ["none"] if there are no evident borders or margins. Do not combine "none" with other directions.
- All three caption fields must be non-empty strings.
- Treat the original caption as a reference. Correct any claims that are not supported by the image.
- Return only the JSON object, without Markdown or explanations.

Here is the original caption reference:
{{reference}}'''

# Layout annotation prompts (original wording; the item vocabulary follows
# MOSS's layout reference). Placeholders: {{caption_rule}}, {{caption}},
# {{max_items}}, {{reference}}, {{tags}}, {{ref_rule}}; the "- {{tag_rule}}" line is
# dropped when no tag list is given.
_ITEM_RULES = r'''- items[].label: EXACTLY one of "object", "character" or "text" - never a tag or any other word.
- items: a flat list of the visually coherent units that matter for the composition, at most {{max_items}}. Use "character" for each person or character, "object" for every other distinct subject, animal, prop or notable background element, and "text" for each block of visible text. Group parts that belong together (one person, not separate items for hair and hands). Natural overlap between boxes is allowed. Do not list tiny details.
- items[].caption: the visible appearance of that unit (clothing, colors, material, pose, expression). For a text item: its typography and color. Do not describe its position. {{item_rule}}
- items[].text: only for "text" items. The exact visible characters, preserved verbatim - do not translate, paraphrase or normalize; write line breaks as \n. Do not include watermarks.
- items[].box_2d: a tight box around the unit, as normalized 0-1000 [y_min, x_min, y_max, x_max] relative to the full image, where [0, 0] is the top-left corner.
- {{tag_rule}}
- {{ref_rule}}
- Return only the JSON object, without Markdown or explanations.'''

# One call: summary caption + items.
LAYOUT_PROMPT = r'''You are an image layout annotator for an image-generation model. Look at the image and return exactly one JSON object with this schema:
{
  "caption": "a description of the whole image",
  "items": [
    {"label": "object", "caption": "visible appearance of this unit", "box_2d": [y_min, x_min, y_max, x_max]},
    {"label": "text", "caption": "typography and color of this text", "text": "the exact visible characters", "box_2d": [y_min, x_min, y_max, x_max]}
  ]
}

Rules:
- caption: {{caption_rule}} Describe the whole image: composition, key subjects, actions, setting, colors, lighting and artistic style. Include only information visible in the image. {{scene_rule}} Do not mention boxes, coordinates or JSON.
''' + _ITEM_RULES + r'''

Reference (may be empty): {{reference}}'''

# Second call of the two-call flow: the caption is already written by the
# official MOSS caption prompt, only the items are requested.
LAYOUT_ITEMS_PROMPT = r'''You are an image layout annotator for an image-generation model. Look at the image and return exactly one JSON object with this schema:
{
  "items": [
    {"label": "object", "caption": "visible appearance of this unit", "box_2d": [y_min, x_min, y_max, x_max]},
    {"label": "text", "caption": "typography and color of this text", "text": "the exact visible characters", "box_2d": [y_min, x_min, y_max, x_max]}
  ]
}

Rules:
- The description of the whole image is already written (below). Use it as context; do not repeat it and do not add a "caption" field.
''' + _ITEM_RULES + r'''

Image description: {{caption}}
Reference (may be empty): {{reference}}'''

_PALETTE = ((255, 64, 64), (64, 200, 64), (64, 140, 255), (255, 200, 0),
            (255, 0, 255), (0, 220, 220), (255, 128, 0), (170, 110, 255))

_LENGTH_RULES = {
    "short": "One sentence that summarizes the main subject, action or relationship.",
    "medium": "One paragraph.",
    "long": "A detailed description; several paragraphs are allowed.",
}
_LABELS = ("object", "character", "text")
_BORDER_VALUES = ("left", "right", "up", "down", "none")
_THINK_BUDGET_ERROR = ("the model used up its max_length budget while still "
                       "thinking - raise max_length or turn thinking off")


def _unload_clip(clip):
    """Best-effort VRAM cleanup for just this node's clip, without
    touching any other model loaded elsewhere in the workflow (same
    technique as the Mykee Prompt Advanced nodes)."""
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
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass


def _toast(node_id, severity, summary, detail):
    """Non-blocking ComfyUI toast; safe no-op without a PromptServer."""
    if PromptServer is None or getattr(PromptServer, "instance", None) is None:
        return
    try:
        PromptServer.instance.send_sync(
            "mykee.toast",
            {"node": node_id, "severity": severity, "summary": summary, "detail": detail, "life": 6000},
        )
    except Exception:
        pass  # a failed notification must never break the actual node


def _think_suffix(clip, thinking):
    """Assistant-turn suffix for Qwen3.5-family clips.

    Qwen3.5 thinks by default. ComfyUI's tokenizer only adds the <think>
    control text when it applies its own chat template, and that is
    skipped here because this node builds the chat text itself (it starts
    with "<|im_start|>"). Add exactly what ComfyUI's template would add -
    for Qwen3.5 clips only; every other model is left untouched.
    """
    for obj in (getattr(clip, "tokenizer", None), getattr(clip, "cond_stage_model", None)):
        if "qwen35" in type(obj).__name__.lower():
            return "<think>\n" if thinking else "<think>\n\n</think>\n\n"
    return ""


def _restore_think_open(clip, thinking, raw_output):
    """When the prompt itself already ended with the opening "<think>\n"
    (thinking on, see _think_suffix), the model's answer starts INSIDE the
    block, so the decoded text has no opening tag. Put it back so the
    <think>...</think> stripping treats the reasoning as one block (and an
    unfinished block is recognised as one, not mistaken for the answer)."""
    if _think_suffix(clip, thinking) == "<think>\n":
        return "<think>\n" + raw_output
    return raw_output


def _count_prompt_tokens(tokens):
    """Best-effort count of the tokens in clip.tokenize() output
    ({encoder_name: [[(token_id, weight), ...], ...]}); None if the
    structure is not the expected one."""
    try:
        return sum(len(chunk) for chunks in tokens.values() for chunk in chunks)
    except Exception:
        return None


def _fit_image(img, megapixels):
    """Downscale an NHWC float image so it has at most `megapixels` MP,
    keeping the aspect ratio and snapping the sides to multiples of 32
    (the Qwen3.5 vision patch grid). Returns the (possibly new) image."""
    if not megapixels or megapixels <= 0 or F is None:
        return img
    _, h, w, _ = img.shape
    limit = megapixels * 1_000_000
    if h * w <= limit:
        return img
    scale = math.sqrt(limit / (h * w))
    new_h = max(32, int(h * scale) // 32 * 32)
    new_w = max(32, int(w * scale) // 32 * 32)
    nchw = img.permute(0, 3, 1, 2)
    nchw = F.interpolate(nchw, size=(new_h, new_w), mode="area")
    return nchw.permute(0, 2, 3, 1).contiguous()


def _estimate_image_tokens(img):
    """How many tokens the single <|image_pad|> placeholder expands to:
    ComfyUI rounds the sides to multiples of 32 and uses one token per
    32x32 block. Plus a little slack for the vision start/end markers."""
    _, h, w, _ = img.shape
    return (max(1, round(h / 32)) * max(1, round(w / 32))) + 16


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _extract_json_object(raw):
    """Pull the first JSON object out of a model answer. Returns
    (dict, None) or (None, error)."""
    if "<think>" in raw and "</think>" not in raw:
        return None, _THINK_BUDGET_ERROR
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    start = text.find("{")
    if start < 0:
        return None, "no JSON object found in the answer"
    try:
        data, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError as e:
        return None, f"invalid JSON ({e})"
    if not isinstance(data, dict):
        return None, "the answer is not a JSON object"
    return data, None


def parse_caption_json(raw):
    """Validate the MOSS recaption answer. Returns (obj, error): exactly
    one of them is None. `obj` is the normalized dict in schema order."""
    data, error = _extract_json_object(raw)
    if error:
        return None, error

    caption = data.get("caption")
    if not isinstance(caption, dict):
        return None, "'caption' must be an object with long/medium/short"
    out_caption = {}
    for key in ("long", "medium", "short"):
        value = caption.get(key)
        if not isinstance(value, str) or not value.strip():
            return None, f"caption.{key} must be a non-empty string"
        out_caption[key] = value.strip()

    watermark = data.get("watermark", [])
    if not isinstance(watermark, list):
        return None, "'watermark' must be an array"
    out_watermark = []
    for i, item in enumerate(watermark):
        box = item.get("box_2d") if isinstance(item, dict) else None
        if not (isinstance(box, list) and len(box) == 4 and all(_is_number(v) for v in box)):
            return None, f"watermark[{i}].box_2d must be four numbers"
        if not all(0 <= v <= 1000 for v in box):
            return None, f"watermark[{i}].box_2d is outside the 0-1000 range"
        label = item.get("label", "")
        out_watermark.append({"box_2d": box, "label": label if isinstance(label, str) else str(label)})

    borders = data.get("borders", ["none"])
    if not isinstance(borders, list) or not borders or not all(b in _BORDER_VALUES for b in borders):
        return None, "'borders' must be a non-empty list of left/right/up/down/none"
    if "none" in borders and len(borders) > 1:
        borders = [b for b in borders if b != "none"]  # "none" must not be combined
    return {"caption": out_caption, "watermark": out_watermark, "borders": list(dict.fromkeys(borders))}, None


_PERSON_WORDS = ("girl", "boy", "woman", "man", "person", "people", "character", "human",
                 "lady", "guy", "male", "female", "child", "kid")


def _normalize_label(label):
    """Models sometimes put the tag itself ("penis", "1girl") into `label`.
    MOSS only knows object / character / text, so map anything else:
    person-like words -> character, everything else -> object."""
    name = label.strip().lower() if isinstance(label, str) else ""
    if name in _LABELS:
        return name
    if any(w in name for w in _PERSON_WORDS):
        return "character"
    return "object"


def parse_layout_json(raw, max_items, need_caption=True):
    """Validate the layout answer. Returns ({"caption", "items"}, None) with
    each item carrying its normalized box as "_box" [y0, x0, y1, x1], or
    (None, error)."""
    data, error = _extract_json_object(raw)
    if error:
        return None, error

    caption = data.get("caption")
    if isinstance(caption, dict):  # tolerate the recaption shape
        caption = caption.get("long") or caption.get("medium") or caption.get("short")
    if need_caption and (not isinstance(caption, str) or not caption.strip()):
        return None, "'caption' must be a non-empty string"
    items = data.get("items")
    if not isinstance(items, list) or not items:
        return None, "'items' must be a non-empty array"

    out_items = []
    for i, item in enumerate(items[:max(1, int(max_items))]):
        where = f"items[{i}]"
        if not isinstance(item, dict):
            return None, f"{where} must be an object"
        label = _normalize_label(item.get("label"))
        cap = item.get("caption")
        if not isinstance(cap, str) or not cap.strip():
            return None, f"{where}.caption must be a non-empty string"
        box = item.get("box_2d", item.get("bbox_2d", item.get("bbox")))
        if not (isinstance(box, list) and len(box) == 4 and all(_is_number(v) for v in box)):
            return None, f"{where}.box_2d must be four numbers [y_min, x_min, y_max, x_max]"
        y0, x0, y1, x1 = (min(1000.0, max(0.0, float(v))) for v in box)
        y0, y1 = sorted((y0, y1))
        x0, x1 = sorted((x0, x1))
        if y1 - y0 < 1 or x1 - x0 < 1:
            # a degenerate box is dropped, it does not sink the whole answer
            print(f"Mykee JSON Captioner: {where} dropped - empty or out-of-image box")
            continue
        entry = {"label": label, "caption": cap.strip()}
        text = item.get("text")
        if label == "text" and not (isinstance(text, str) and text.strip()):
            label = "object"  # a text item without literal text is not valid MOSS layout
            entry["label"] = label
        if isinstance(text, str) and text.strip():
            entry["text"] = text
        entry["_box"] = [y0, x0, y1, x1]
        out_items.append(entry)
    if not out_items:
        return None, "no item had a usable box"
    return {"caption": caption.strip() if isinstance(caption, str) else "", "items": out_items}, None


def _centered_box(x1, y1, x2, y2, width, height):
    """Pixel box (top-left origin) -> [y1, x1, y2, x2] centered integers.
    Same arithmetic as MOSS-Image's compile_prompt.py: fractional edges are
    enclosed (floor left/top, ceil right/bottom) before subtracting the
    canvas center. Raises ValueError for an invalid or out-of-bounds box."""
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("box outside the canvas")
    result = [
        math.floor(round(y1, 9)) - height // 2,
        math.floor(round(x1, 9)) - width // 2,
        math.ceil(round(y2, 9)) - height // 2,
        math.ceil(round(x2, 9)) - width // 2,
    ]
    if not (result[0] < result[2] and result[1] < result[3]):
        raise ValueError("box too small at export precision")
    return result


def compile_layout(draft, width, height):
    """Draft (normalized boxes) -> MOSS Layout dict with abs_centralized /
    yxyx boxes. Returns (obj, error)."""
    items = []
    for n, item in enumerate(draft["items"]):
        y0, x0, y1, x1 = item["_box"]
        try:
            bbox = _centered_box(x0 / 1000.0 * width, y0 / 1000.0 * height,
                                 x1 / 1000.0 * width, y1 / 1000.0 * height, width, height)
        except ValueError as e:
            print(f"Mykee JSON Captioner: item {n} ({item['caption'][:40]!r}) dropped - {e}")
            continue
        out = {"label": item["label"], "caption": item["caption"]}
        if "text" in item:
            out["text"] = item["text"]
        out["bbox"] = bbox
        items.append(out)
    if not items:
        return None, "no item had a usable bounding box"
    return {
        "summary": {"caption": draft["caption"]},
        "items": items,
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }, None


def _canvas_size(img_h, img_w, megapixels):
    """Generation canvas: the image's aspect ratio at ~`megapixels` MP, sides
    snapped to multiples of 16 (a MOSS-Image requirement)."""
    if not megapixels or megapixels <= 0:  # 0 = the image's own size
        w, h = img_w, img_h
    else:
        area = max(0.05, float(megapixels)) * 1_000_000
        w = math.sqrt(area * img_w / img_h)
        h = w * img_h / img_w
    snap = lambda v: max(64, int(round(v / 16.0)) * 16)
    return snap(w), snap(h)


def _parse_tags(text):
    """Comma / semicolon / newline separated tag text -> unique tag list."""
    seen, tags = set(), []
    for part in re.split(r"[,;\n]+", text or ""):
        tag = part.strip()
        if tag and tag.lower() not in seen:
            seen.add(tag.lower())
            tags.append(tag)
    return tags


def _tag_rule(tags, tag_mode):
    """The prompt rule that tells the model which tags to locate."""
    if not tags:
        return ""
    joined = ", ".join(tags)
    base = ("Create one item for every separate instance of a tag (several people = several "
            "items) and start its caption with the tag (or its English equivalent). Skip tags "
            "that are not visible or cannot be boxed (moods, styles, abstract concepts) - "
            "never invent a box. Write the captions in English.")
    if tag_mode == "strict":
        return f"Annotate ONLY these tags: {joined}. " + base
    return (f"Make sure these tags are covered whenever they are visible: {joined}. " + base
            + " You may add other important units as well.")


_MOSS_REF_RULE = ("- Treat the original caption as a reference. Correct any claims that are not "
                 "supported by the image.")
_MOSS_REF_TAIL = "Here is the original caption reference:\n{{reference}}"
_REF_RULE_HINT = "Treat the reference below as a hint only; correct any claim that the image does not support."
_TAG_SENTENCE = ("A tag list, if given, consists of booru-style labels of things confirmed to be in the "
                 "image: express them as natural prose, never quote or list the tags themselves "
                 "(\"1girl\" is simply \"a woman\"), and do not force a tag in where it does not fit "
                 "the description. Name a confirmed thing plainly and with certainty - no hedging "
                 "such as \"X or Y\".")
_KEEP_DETAIL = ("The notes only add to the description: keep it exactly as complete and detailed as it "
                "would be without them (full setting, background, lighting, colors, textures). If a "
                "single note cannot be reconciled with the image, leave only that detail out and "
                "keep describing everything else.")
_REF_RULE_BALANCED = ("The notes (reference) below were written by a person who can see the image. Use them "
                      "to refine the description: add what the image cannot show by itself (camera angle, "
                      "shot type, names of things, style, setting hints) and prefer their identification "
                      "when the image is ambiguous or you are unsure what something is. Ignore a note only "
                      "if the image clearly contradicts it. " + _TAG_SENTENCE + " " + _KEEP_DETAIL +
                      " Never mention the notes or tags themselves.")
_REF_RULE_GUIDE = ("The notes (reference) below were written by a person who can see the image. Treat them "
                   "as ground truth: when they identify or name something, or state the camera angle, shot "
                   "type, pose, setting or style, use that instead of your own guess - preferably in their "
                   "wording - and work it into the description naturally. " + _TAG_SENTENCE + " " + _KEEP_DETAIL +
                   " Do not invent details beyond what is visible or noted, and never mention the notes "
                   "or tags themselves.")


_SCALE_RULE = ("Describe the relative scale and proportions of the subjects and objects to each other and "
               "to the frame (for example a tiny figure beside a giant object), and the body proportions "
               "of people.")
_EXPLICIT_RULE = ("Describe adult and explicit content plainly and in the same detail as everything else, "
                  "without euphemisms or omissions: nudity, size and shape of breasts and other body parts, "
                  "genitals (type, size, shape, state), their position and contact with other subjects or "
                  "objects, fluids, and the pose or action. If any genitalia are visible, name body parts "
                  "directly and correctly (breasts, nipples, areolae, vulva, labia, clitoris, penis, "
                  "testicles, anus, buttocks, and so on), without euphemisms. If a nipple outline, vulva "
                  "shape or penis bulge is visible through clothing, say it plainly.")


_EXPLICIT_FINAL = ("IMPORTANT - uncensored captioning task: explicit content must be described, never "
                   "softened, skipped or summarized. Use the direct anatomical words. When a tag or note "
                   "names a body part or an act, write that same word in the description - never replace "
                   "it with a vaguer or hedged term (such as \"phallic object\" or \"mushroom-like "
                   "structure\" for a penis).")


_SCAN_PROMPT = r"""You are an image analyst for an adult dataset. Look at the image and report ONLY its adult / explicit content, in plain anatomical language, without euphemisms, hedging or disclaimers.
Cover, whenever visible:
- nudity, and which body parts are exposed (for example breasts, nipples, areolae, vulva, labia, clitoris, penis, testicles, anus, buttocks), with their size, shape and state;
- how the subjects and objects physically relate to each other: contact, where body parts are in relation to each other, and any sexual act, named plainly;
- the body position and pose of each subject;
- fluids and where they are.
Name everything by its correct anatomical word. Use the person's notes and tags below when they name something. Do not describe anything that is not visible. If no adult content is visible, answer exactly: none
Answer as a short plain-text list of 3-10 lines.

Notes: {{reference}}
Tags: {{tags}}"""


def _build_scan_prompt(reference, tags):
    return (_SCAN_PROMPT.replace("{{reference}}", reference or "(none)")
                        .replace("{{tags}}", ", ".join(tags) or "(none)"))


def parse_scan_text(raw):
    """The explicit-content scan is free text; only thinking leftovers and
    code fences are stripped. Returns (text, None) or (None, error)."""
    if "<think>" in raw and "</think>" not in raw:
        return None, _THINK_BUDGET_ERROR
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    if not text:
        return None, "the scan was empty"
    return text[:3000], None


def _scene_rule(explicit):
    """Extra caption rule: relative scale, and (optionally) explicit detail."""
    return _SCALE_RULE + (" " + _EXPLICIT_RULE if explicit else "")


def _item_rule(explicit):
    """Extra rule for the caption of a single layout item."""
    return ("Mention the size and shape of notable body parts and the item's scale relative to the other "
            "subjects." + (" Describe adult content plainly and name body parts directly and correctly, without euphemisms." if explicit else ""))


_EXPLICIT_SYSTEM = ("You are an uncensored image annotator for adult datasets. You describe everything "
                    "that is visible, including nudity and sexual content, accurately and in plain "
                    "anatomical language. You never use euphemisms and never soften, hedge or omit "
                    "explicit details.")
_EXPLICIT_WORDS = ("Anatomical and explicit words in the notes or tags (penis, vulva, nipples, ...) are used "
                   "exactly as given - never replaced by a euphemism or a vaguer term, and never hedged.")


def _ref_rule(reference_mode, reference, tags, explicit=False):
    """Rule line for the reference / notes. hint = MOSS's original behaviour,
    balanced = the notes refine the description unless the image clearly
    contradicts them, guide = the notes are ground truth."""
    rule = _REF_RULE_HINT
    if reference_mode != "hint" and (reference.strip() or tags):
        rule = _REF_RULE_GUIDE if reference_mode == "guide" else _REF_RULE_BALANCED
    if explicit and (reference.strip() or tags):
        rule += " " + _EXPLICIT_WORDS
    return rule


def _build_caption_prompt(base, reference, tags, reference_mode, explicit, scan=""):
    """The MOSS caption prompt with the person's notes and tag list worked in
    (see _ref_rule for the three modes)."""
    notes = reference.strip() or "(none)"
    out = base
    rule = _ref_rule(reference_mode, reference, tags, explicit)
    if reference_mode != "hint" and (reference.strip() or tags):
        out = out.replace(_MOSS_REF_RULE, "- " + rule)
        out = out.replace(_MOSS_REF_TAIL, "Here are the notes:\n{{reference}}")
    elif rule != _REF_RULE_HINT:  # hint mode: MOSS's own rule plus the word rule
        out = out.replace(_MOSS_REF_RULE, _MOSS_REF_RULE + " " + _EXPLICIT_WORDS)
    out = out.replace("{{reference}}", notes)
    extra = "    - " + _SCALE_RULE + ("\n    - " + _EXPLICIT_RULE if explicit else "") + "\n"
    if scan:
        extra += ("    - The scan of the adult content at the end is accurate: include every point of it "
                  "in the description, with its exact anatomical words, relationships and poses.\n")
    out = out.replace("- caption.medium: compress long", extra + "- caption.medium: compress long", 1)
    if tags:
        out += "\nTags (booru-style labels of things in the image): " + ", ".join(tags)
    if scan:
        out += "\nScan of the adult content in the image (accurate):\n" + scan
    return out


def _build_tag_prompt(tags, tag_mode, caption_text, caption_rule, reference, max_items, ref_rule,
                      scene_rule, item_rule):
    """Dedicated grounding prompt for a tag list. The model must answer for
    EVERY listed tag (empty `instances` when it is not visible), which keeps
    it from drifting back to its own idea of what to box. Boxes use the
    native Qwen grounding order [x_min, y_min, x_max, y_max], 0-1000."""
    joined = ", ".join(tags)
    caption_field = '"caption": "a description of the whole image",\n  ' if caption_text is None else ""
    extra_schema = ""
    extra_rule = ""
    if tag_mode == "hint":
        extra_schema = (',\n  "extra_items": [{"label": "object", "caption": "visible appearance", '
                        '"bbox_2d": [x_min, y_min, x_max, y_max]}]')
        extra_rule = (f'- extra_items: other visually important units that are NOT in the tag list, '
                      f'at most {max_items}; "label" is EXACTLY "object", "character" or "text" '
                      '(a "text" item also needs "text" with the exact visible characters). '
                      'May be empty.\n')
    caption_rule_line = ""
    context = ""
    if caption_text is None:
        caption_rule_line = (f"- caption: {caption_rule} Describe the whole image: composition, key "
                             "subjects, actions, setting, colors, lighting and artistic style. Include "
                             "only information visible in the image. " + scene_rule + "\n")
    else:
        context = f"\nImage description (already written, for context): {caption_text}"
    return (
        f"You are an object grounding assistant. Find these tags in the image: {joined}\n\n"
        "For EACH tag, in the given order, decide whether it is clearly visible and can be boxed. "
        "Return exactly one JSON object:\n"
        "{\n  " + caption_field + '"tags": [\n'
        '    {"tag": "<the tag exactly as given>", "instances": [\n'
        '      {"caption": "visible appearance of this instance", "bbox_2d": [x_min, y_min, x_max, y_max]}\n'
        "    ]}\n  ]" + extra_schema + "\n}\n\n"
        "Rules:\n"
        '- Exactly one entry in "tags" per given tag, with the same spelling and in the same order. '
        'Use an empty "instances" list when the tag is not visible, or is not a thing that can be boxed '
        '(a mood, a style, or an attribute such as "naked" or "smiling"). Never guess and never invent a box.\n'
        '- Several separate instances of one tag (for example two people) are several entries in "instances".\n'
        "- bbox_2d: a tight box around that instance, normalized 0-1000 relative to the full image, in the "
        "order x_min, y_min, x_max, y_max, where [0, 0] is the top-left corner.\n"
        "- instance caption: the visible appearance of that instance (colors, material, pose). "
        "Do not describe its position. " + item_rule + "\n"
        + caption_rule_line + extra_rule +
        f"- {ref_rule}\n"
        "- Return only the JSON object, without Markdown or explanations.\n"
        f"{context}\nReference (may be empty): {reference}"
    )


def _norm_box_xyxy(box):
    """[x_min, y_min, x_max, y_max] (0-1000) -> [y_min, x_min, y_max, x_max]
    clamped and ordered, or None if it is invalid / empty."""
    if not (isinstance(box, list) and len(box) == 4 and all(_is_number(v) for v in box)):
        return None
    x0, y0, x1, y1 = (min(1000.0, max(0.0, float(v))) for v in box)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return [y0, x0, y1, x1]


def parse_tag_json(raw, tags, max_items, need_caption, allow_extra):
    """Validate the tag grounding answer. Returns ({"caption", "items"}, None)
    like parse_layout_json, or (None, error). Tags the model reports as not
    visible are only logged."""
    data, error = _extract_json_object(raw)
    if error:
        return None, error
    caption = data.get("caption")
    if isinstance(caption, dict):
        caption = caption.get("long") or caption.get("medium") or caption.get("short")
    if need_caption and (not isinstance(caption, str) or not caption.strip()):
        return None, "'caption' must be a non-empty string"
    entries = data.get("tags")
    if not isinstance(entries, list):
        return None, "'tags' must be an array"

    wanted = {tag.lower(): tag for tag in tags}
    seen, items = set(), []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("tag", "")).strip().lower()
        if key not in wanted:
            continue
        tag = wanted[key]
        for inst in entry.get("instances") or []:
            if not isinstance(inst, dict):
                continue
            box = _norm_box_xyxy(inst.get("bbox_2d", inst.get("box_2d")))
            if box is None:
                continue
            cap = inst.get("caption")
            cap = cap.strip() if isinstance(cap, str) and cap.strip() else tag
            if not cap.lower().startswith(tag.lower()):
                cap = f"{tag}, {cap}"
            items.append({"label": _normalize_label(tag), "caption": cap, "_box": box})
            seen.add(key)
    missing = [tag for tag in tags if tag.lower() not in seen]
    if missing:
        print(f"Mykee JSON Captioner: tags not found in the image: {', '.join(missing)}")

    # max_items caps the total; the listed tags come first, so the model's
    # own extra items are the ones that get cut.
    if allow_extra and isinstance(data.get("extra_items"), list):
        for extra in data["extra_items"][:max(1, int(max_items))]:
            if not isinstance(extra, dict):
                continue
            box = _norm_box_xyxy(extra.get("bbox_2d", extra.get("box_2d")))
            cap = extra.get("caption")
            if box is None or not isinstance(cap, str) or not cap.strip():
                continue
            label = _normalize_label(extra.get("label"))
            entry = {"label": label, "caption": cap.strip(), "_box": box}
            text = extra.get("text")
            if label == "text" and isinstance(text, str) and text.strip():
                entry["text"] = text
            elif label == "text":
                entry["label"] = "object"
            items.append(entry)
    if not items:
        return None, f"none of the tags was found in the image ({', '.join(tags)})"
    return {"caption": caption.strip() if isinstance(caption, str) else "",
            "items": items[:max(1, int(max_items))]}, None


def _load_font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _draw_preview(img, canvas_w, canvas_h, boxes):
    """First image of the NHWC batch, resized to the canvas, with labelled
    boxes drawn on it. `boxes` = [(label, (x0, y0, x1, y1)), ...] in canvas
    pixels. Returns a 1xHxWx3 float tensor; falls back to the plain image
    if drawing fails (a preview must never break the node)."""
    try:
        arr = (img[0, :, :, :3].clamp(0, 1) * 255).round().byte().cpu().numpy()
        pil = Image.fromarray(arr).convert("RGB").resize((canvas_w, canvas_h), Image.LANCZOS)
        draw = ImageDraw.Draw(pil)
        font = _load_font(max(12, canvas_h // 50))
        for n, (label, box) in enumerate(boxes):
            color = _PALETTE[n % len(_PALETTE)]
            draw.rectangle(box, outline=color, width=max(2, canvas_h // 300))
            if label:
                tb = draw.textbbox((0, 0), label, font=font)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
                y = box[1] - th - 6 if box[1] - th - 6 >= 0 else box[1]
                draw.rectangle((box[0], y, box[0] + tw + 8, y + th + 6), fill=color)
                draw.text((box[0] + 4, y + 2 - tb[1]), label, fill=(0, 0, 0), font=font)
        return torch.from_numpy(np.asarray(pil).astype(np.float32) / 255.0).unsqueeze(0)
    except Exception as e:
        print(f"Mykee JSON Captioner: preview drawing failed ({type(e).__name__}: {e})")
        return img[:1, :, :, :3].contiguous()


def _layout_boxes(items, canvas_w, canvas_h):
    """Centered yxyx items -> preview boxes in canvas pixels."""
    boxes = []
    for n, item in enumerate(items):
        y0, x0, y1, x1 = item["bbox"]
        px = (x0 + canvas_w // 2, y0 + canvas_h // 2, x1 + canvas_w // 2, y1 + canvas_h // 2)
        boxes.append((f"{n + 1} {item['label']}: {item['caption'][:36]}", px))
    return boxes


def _watermark_boxes(watermarks, canvas_w, canvas_h):
    """Normalized 0-1000 watermark boxes -> preview boxes in canvas pixels."""
    boxes = []
    for w in watermarks:
        y0, x0, y1, x1 = w["box_2d"]
        boxes.append(("watermark", (x0 / 1000 * canvas_w, y0 / 1000 * canvas_h,
                                    x1 / 1000 * canvas_w, y1 / 1000 * canvas_h)))
    return boxes


class MykeeJsonCaptioner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {
                    "tooltip": "A generation-capable VL clip (e.g. Qwen3.5-9B "
                               "through Load CLIP). MOSS recommends Qwen3.5 "
                               "27B / 35B-A3B or Gemma 4 for captioning.",
                }),
                "image": ("IMAGE", {
                    "tooltip": "Image(s) to caption. A batch is captioned "
                               "one image at a time.",
                }),
                "mode": (["layout", "caption"], {
                    "default": "layout",
                    "tooltip": "layout: MOSS Layout JSON (summary + items "
                               "with centered yxyx bounding boxes). "
                               "caption: plain caption text of the chosen "
                               "length (plus watermark/borders outputs).",
                }),
                "caption_length": (["long", "medium", "short"], {
                    "default": "long",
                    "tooltip": "Which single description is used: "
                               "summary.caption in layout mode, the "
                               "caption in caption mode.",
                }),
                "explicit_detail": ("BOOLEAN", {
                    "default": True, "label_on": "on", "label_off": "off",
                    "tooltip": "Also describe adult / explicit content in "
                               "detail (nudity, body parts and their size "
                               "and shape, genitals, position, contact, "
                               "fluids) without euphemisms. Relative scale "
                               "and proportions of the subjects are always "
                               "described.",
                }),
                "reference": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Optional notes about the image: what is "
                               "in it, corrections to what the model tends "
                               "to get wrong, camera angle, style ... See "
                               "reference_mode for how strictly the model "
                               "follows it. Empty = none.",
                }),
                "reference_mode": (["hint", "balanced", "guide"], {
                    "default": "hint",
                    "tooltip": "How strictly the model follows your reference "
                               "text and tag list. hint: MOSS's original - "
                               "only a hint, anything the image does not "
                               "support is corrected. balanced: your notes "
                               "refine the description (camera angle, what "
                               "an object really is, ...) unless the image "
                               "clearly contradicts them. guide: your notes "
                               "are ground truth, the model follows them "
                               "even against its own impression. Tags are "
                               "always written as natural prose.",
                }),
                "tag_list": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Tags, separated by commas or new lines (type "
                               "them or connect a text output; e.g. from a "
                               "tagger). Layout mode: the tags to find and "
                               "box (empty = the model picks the items "
                               "itself). Caption mode: confirmed content "
                               "that is worked into the description.",
                }),
                "tag_mode": (["strict", "hint"], {
                    "default": "hint",
                    "tooltip": "strict: only the tags in tag_list get items. "
                               "hint: the tags are covered when visible and "
                               "the model may add other important items. "
                               "Tags that are not visible are skipped.",
                }),
                "tags_to_prompt": ("BOOLEAN", {
                    "default": False, "label_on": "on", "label_off": "off",
                    "tooltip": "Add the tag list to the end of the caption "
                               "text: a blank line, then \"Tags: \" and the "
                               "tags separated by commas. Applies to the "
                               "caption output and to the caption inside the "
                               "json. Does nothing when tag_list is empty.",
                }),
                "separate_caption": ("BOOLEAN", {
                    "default": False, "label_on": "2 calls", "label_off": "1 call",
                    "tooltip": "Layout mode. 2 calls: the summary caption is "
                               "written with the official MOSS caption prompt "
                               "(more accurate), then a second call finds "
                               "the boxes. 1 call: both at once (faster, "
                               "usually a rougher caption). A custom "
                               "system_prompt always uses 1 call.",
                }),
                "system_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Empty = the built-in prompt of the chosen "
                               "mode. Fill in to replace it entirely (it is "
                               "sent as the instruction text in front of the "
                               "image, like MOSS does). Keep that mode's "
                               "JSON schema or parsing fails; the final "
                               "JSON format itself is built by the node. "
                               "Placeholders: {{reference}}, {{tags}}, and "
                               "in layout mode {{caption_rule}}, "
                               "{{max_items}}, {{tag_rule}}.",
                }),
                "canvas_megapixels": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 4.0, "step": 0.05,
                    "tooltip": "Size of the generation canvas for the "
                               "bounding boxes. 0 = the image's own size "
                               "(sides rounded to multiples of 16). Other "
                               "values: the image's aspect ratio at about "
                               "this many megapixels. Use the width/height "
                               "outputs as the MOSS-Image generation size.",
                }),
                "max_items": ("INT", {
                    "default": 0, "min": 0, "max": 100,
                    "tooltip": "Layout mode: the most items (boxes) the "
                               "result may have; extra items are cut. With a "
                               "tag list the tags come first. 0 = no limit "
                               "(a ceiling of 60 applies).",
                }),
                "image_megapixels": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 16.0, "step": 0.1,
                    "tooltip": "Downscale larger images to at most this many "
                               "megapixels before the model sees them (0 = "
                               "send the image as it is). ComfyUI otherwise "
                               "feeds the full resolution, which is slow and "
                               "costs many tokens.",
                }),
                "thinking": ("BOOLEAN", {
                    "default": False, "label_on": "on", "label_off": "off",
                    "tooltip": "Let a thinking model (Qwen3.5) reason first. "
                               "MOSS recommends it for more accurate "
                               "captions; raise max_length, the thinking "
                               "text uses the same budget.",
                }),
                "max_length": ("INT", {
                    "default": 4096, "min": 256, "max": 16384, "step": 64,
                    "tooltip": "Max tokens the model may generate per image "
                               "(thinking included). The image's own tokens "
                               "are added on top automatically.",
                }),
                "retries": ("INT", {
                    "default": 1, "min": 0, "max": 5,
                    "tooltip": "If the answer is not valid JSON for the "
                               "chosen mode, regenerate (different seed) up "
                               "to this many more times.",
                }),
                "sampling_mode": ("BOOLEAN", {
                    "default": True, "label_on": "on", "label_off": "off",
                    "tooltip": "On: sample with the settings below. Off: "
                               "greedy decoding (deterministic).",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Sampling seed (each image in a batch and "
                               "each retry uses a derived seed).",
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 2.0, "step": 0.01,
                    "tooltip": "MOSS's own captioning setup uses 1.0.",
                }),
                "top_k": ("INT", {
                    "default": 20, "min": 0, "max": 1000,
                    "tooltip": "Only sample from the top K tokens (0 = off).",
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Nucleus sampling cutoff (MOSS uses 0.95).",
                }),
                "min_p": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Minimum token probability relative to the best.",
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01,
                    "tooltip": "1.0 = off. Keep it low: captions must copy "
                               "visible text exactly.",
                }),
                "presence_penalty": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "0 = off. Keep it low for long captions.",
                }),
                "pretty_json": ("BOOLEAN", {
                    "default": True, "label_on": "indented", "label_off": "one line",
                    "tooltip": "Format the json output indented instead of "
                               "as one compact line.",
                }),
                "unload_after_run": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Free 'clip' from VRAM once the whole batch "
                               "is done, without touching other models.",
                }),
            },
            "hidden": {"node_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT", "INT", "IMAGE")
    RETURN_NAMES = ("json", "caption", "watermark", "borders", "width", "height", "bbox_preview")
    OUTPUT_IS_LIST = (True, True, True, True, True, True, True)
    FUNCTION = "run"
    CATEGORY = "Mykee/Prompt"

    def _generate_json(self, clip, img, prompt_text, parse_fn, thinking, max_length,
                       retries, sampling_mode, seed, temperature, top_k, top_p,
                       min_p, repetition_penalty, presence_penalty, system_text=""):
        """Run the model on one NHWC image until parse_fn(raw) accepts the
        answer. Returns (obj, error, last_raw)."""
        # MOSS sends the text first, then the image, in one user turn.
        chat_text = (
            (f"<|im_start|>system\n{system_text}<|im_end|>\n" if system_text else "")
            + f"<|im_start|>user\n{prompt_text}"
            "<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n"
            "<|im_start|>assistant\n"
            + _think_suffix(clip, thinking)
        )
        try:
            tokens = clip.tokenize(chat_text, image=img, skip_template=True, min_length=1, thinking=thinking)
        except TypeError:
            return None, "the connected clip doesn't support image input (not a VL model)", ""
        except Exception as e:
            return None, f"tokenization failed: {type(e).__name__}: {e}", ""

        prompt_tokens = _count_prompt_tokens(tokens)
        if prompt_tokens is None:
            prompt_tokens = len(chat_text) // 3
        # max_length is a TOTAL budget (prompt + completion), and the single
        # <|image_pad|> placeholder expands into the image's tokens.
        effective_max_length = prompt_tokens + _estimate_image_tokens(img) + int(max_length)

        error, raw = "no attempt was made", ""
        for attempt in range(int(retries) + 1):
            attempt_seed = (int(seed) + attempt * 7919) % (2 ** 63)
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
                    seed=attempt_seed,
                )
                raw = _restore_think_open(clip, thinking, clip.decode(generated_ids))
            except Exception as e:
                return None, (f"generation failed: {type(e).__name__}: {e}. Make sure "
                              "'clip' is a generation-capable VL chat model."), raw
            obj, error = parse_fn(raw)
            if obj is not None:
                return obj, None, raw
            print(f"Mykee JSON Captioner: attempt {attempt + 1} rejected - {error}")
        return None, error, raw

    def run(self, clip, image, mode, caption_length, explicit_detail, reference, reference_mode, tag_list,
            tag_mode, tags_to_prompt, separate_caption, system_prompt, canvas_megapixels, max_items,
            image_megapixels, thinking, max_length, retries, sampling_mode, seed,
            temperature, top_k, top_p, min_p, repetition_penalty, presence_penalty,
            pretty_json, unload_after_run, node_id=None):
        max_items = int(max_items) if max_items and int(max_items) > 0 else 60
        ref = reference.strip() if reference and reference.strip() else "(none)"
        custom = system_prompt if system_prompt and system_prompt.strip() else None
        tags = _parse_tags(tag_list)
        tag_rule = _tag_rule(tags, tag_mode)
        tags_suffix = ("\n\nTags: " + ", ".join(tags)) if (tags_to_prompt and tags) else ""
        scene_rule, item_rule = _scene_rule(explicit_detail), _item_rule(explicit_detail)
        ctx = {"scan": ""}  # the explicit-content scan of the current image

        def cur_ref():
            """The reference text of the current image, scan included."""
            if not ctx["scan"]:
                return ref
            return ((ref + "\n") if ref != "(none)" else "") + "Scan of the adult content (accurate): " + ctx["scan"]

        def cur_rule():
            return _ref_rule(reference_mode, (reference or "") + ctx["scan"], tags, explicit_detail)

        def fill(template, **extra):
            template = template.replace("- {{tag_rule}}\n", f"- {tag_rule}\n" if tag_rule else "")
            ref_rule = cur_rule()
            values = {"reference": cur_ref(), "tags": ", ".join(tags) or "(none)", "tag_rule": tag_rule,
                      "caption_rule": _LENGTH_RULES[caption_length], "max_items": str(int(max_items)),
                      "ref_rule": ref_rule, "scene_rule": scene_rule, "item_rule": item_rule}
            values.update(extra)
            for key, value in values.items():
                template = template.replace("{{%s}}" % key, value)
            return template

        two_calls = mode == "layout" and bool(separate_caption) and custom is None
        use_tags = mode == "layout" and bool(tags) and custom is None
        if tags and mode == "layout":
            print(f"Mykee JSON Captioner: tag list ({tag_mode}, "
                  f"{'dedicated grounding prompt' if use_tags else 'custom system_prompt'}): {tags}")

        def finish(prompt):
            """Built-in prompts end with the explicit-content reminder (the
            last instruction the model reads is the one it follows best)."""
            return prompt + "\n\n" + _EXPLICIT_FINAL if explicit_detail else prompt

        def tag_prompt(caption_text):
            return finish(_build_tag_prompt(tags, tag_mode, caption_text, _LENGTH_RULES[caption_length],
                                            cur_ref(), int(max_items), cur_rule(), scene_rule, item_rule))

        def caption_prompt():
            if custom:
                return fill(custom)
            return finish(_build_caption_prompt(MOSS_CAPTION_PROMPT, reference or "", tags,
                                                reference_mode, explicit_detail, ctx["scan"]))
        gen = dict(thinking=thinking, max_length=max_length, retries=retries,
                   sampling_mode=sampling_mode, temperature=temperature, top_k=top_k,
                   top_p=top_p, min_p=min_p, repetition_penalty=repetition_penalty,
                   presence_penalty=presence_penalty,
                   system_text=_EXPLICIT_SYSTEM if explicit_detail else "")
        dump = lambda v: json.dumps(v, ensure_ascii=False, indent=2 if pretty_json else None,
                                    separators=None if pretty_json else (",", ":"))
        count = image.shape[0]
        progress = comfy_utils.ProgressBar(count) if comfy_utils is not None else None
        results = {k: [] for k in RETURN_KEYS}
        failures = 0
        for i in range(count):
            canvas_w, canvas_h = _canvas_size(image.shape[1], image.shape[2], canvas_megapixels)
            img = _fit_image(image[i:i + 1], image_megapixels)
            seed_i = int(seed) + i
            ctx["scan"] = ""
            if explicit_detail:
                # Extra call first: find and name the adult content, how the
                # bodies relate and the poses; the captions then build on it.
                scan, scan_err, _ = self._generate_json(
                    clip, img, _build_scan_prompt(ref if ref != "(none)" else "", tags),
                    parse_scan_text, seed=seed_i + 31337, **gen)
                if scan and scan.strip(" .\n").lower() != "none":
                    ctx["scan"] = scan
                    print(f"Mykee JSON Captioner: explicit scan ({len(scan)} chars):\n{scan}")
                elif scan_err:
                    print(f"Mykee JSON Captioner: explicit scan failed ({scan_err}), continuing without it")
                else:
                    print("Mykee JSON Captioner: explicit scan found no adult content")
            obj, error, raw, boxes = None, None, "", []

            if mode == "layout":
                caption_text = None
                if two_calls:
                    cap_obj, error, raw = self._generate_json(
                        clip, img, caption_prompt(), parse_caption_json, seed=seed_i, **gen)
                    if cap_obj is not None:
                        caption_text = cap_obj["caption"][caption_length]
                        prompt_text = (tag_prompt(caption_text) if use_tags
                                       else finish(fill(LAYOUT_ITEMS_PROMPT, caption=caption_text)))
                else:
                    prompt_text = (tag_prompt(None) if use_tags else
                                   fill(custom) if custom else finish(fill(LAYOUT_PROMPT)))

                if error is None:
                    def parse_fn(raw_, cw=canvas_w, ch=canvas_h, ct=caption_text):
                        if use_tags:
                            draft, err = parse_tag_json(raw_, tags, max_items, ct is None,
                                                        tag_mode == "hint")
                        else:
                            draft, err = parse_layout_json(raw_, max_items, need_caption=ct is None)
                        if err:
                            return None, err
                        if ct is not None:
                            draft["caption"] = ct
                        return compile_layout(draft, cw, ch)
                    obj, error, raw = self._generate_json(
                        clip, img, prompt_text, parse_fn, seed=seed_i + (104729 if two_calls else 0), **gen)
                if obj is not None:
                    boxes = _layout_boxes(obj["items"], canvas_w, canvas_h)
            else:
                obj, error, raw = self._generate_json(
                    clip, img, caption_prompt(), parse_caption_json, seed=seed_i, **gen)
                if obj is not None:
                    boxes = _watermark_boxes(obj["watermark"], canvas_w, canvas_h)

            if obj is None:
                failures += 1
                print(f"Mykee JSON Captioner: image {i + 1}/{count} failed - {error}")
                if raw:
                    print(f"  last raw answer (first 600 chars): {raw[:600]!r}")
                values = [f"[MOSS captioning failed: {error}]", "", "", "", canvas_w, canvas_h]
            elif mode == "layout":
                obj["summary"]["caption"] += tags_suffix
                values = [dump(obj), obj["summary"]["caption"], "", "", canvas_w, canvas_h]
                print(f"Mykee JSON Captioner: image {i + 1}/{count} OK "
                      f"({len(obj['items'])} items, canvas {canvas_w}x{canvas_h})")
            else:
                chosen = obj["caption"][caption_length] + tags_suffix
                values = [dump({"caption": chosen}), chosen,
                          json.dumps(obj["watermark"], ensure_ascii=False),
                          json.dumps(obj["borders"], ensure_ascii=False), canvas_w, canvas_h]
                print(f"Mykee JSON Captioner: image {i + 1}/{count} OK "
                      f"({len(chosen)} chars {caption_length} caption)")
            values.append(_draw_preview(image[i:i + 1], canvas_w, canvas_h, boxes))
            for key, value in zip(RETURN_KEYS, values):
                results[key].append(value)
            if progress is not None:
                progress.update(1)

        if failures:
            _toast(node_id, "warn", "Mykee JSON Captioner",
                   f"{failures} of {count} image(s) could not be captioned - see the console.")
        if unload_after_run:
            _unload_clip(clip)
        return tuple(results[k] for k in RETURN_KEYS)


RETURN_KEYS = ("json", "caption", "watermark", "borders", "width", "height", "bbox_preview")

NODE_CLASS_MAPPINGS = {
    "MykeeJsonCaptioner": MykeeJsonCaptioner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeJsonCaptioner": "Mykee JSON Captioner",
}
