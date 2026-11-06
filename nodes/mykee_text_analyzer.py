"""
ComfyUI-Mykee-Nodes / text analyzer node

"Mykee Text Analyzer": answers a question about a long text that does not fit
into the model in one piece. Feed it the chunks of the Mykee Text Splitter
(or a text that is short enough to go in on its own).

Two rounds, like Mykee Prompt Advanced Ultimate:

- Round 1 (response_1): the model reads the chunks one by one WITH the
  question in front of it and writes notes - direct answers and also indirect
  evidence (hints, circumstances, changes), so that something the text never
  says outright can still be worked out. note_mode "rolling": every chunk is
  read together with the notes of the earlier ones (the notes are kept short
  by condensing them when they grow past notes_max_tokens). note_mode
  "independent": every chunk is read on its own, and the notes are only put
  together afterwards.
- Round 2 (response_2): the final answer, from the notes only. Notes that are
  too long for one call are condensed in groups first.

The notes depend on the question, so they are cached together with the
question (and the round-1 settings, the chunks and the model): a new question
reads the text again, while changing only round 2's settings reuses the notes.
An interrupted or failed run continues at the chunk where it stopped when
queued again.

Inputs: question, and the text either typed into the text box, connected to
it, or connected to the text_in socket (then the box is hidden). The text may
be a list (the splitter's chunks) or a single text. A chunk longer than
input_token_limit is refused with a hint to use the splitter - the node never
cuts text itself.
"""

import hashlib
import random
import re
from collections import OrderedDict

try:
    from comfy_execution.graph import ExecutionBlocker  # noqa: F401
except ImportError:  # very old ComfyUI
    ExecutionBlocker = None

try:
    import comfy.model_management as model_management
except Exception:  # pragma: no cover - outside ComfyUI
    model_management = None

try:
    from comfy.utils import ProgressBar
except Exception:  # pragma: no cover - outside ComfyUI
    ProgressBar = None

from .mykee_llm_common import (
    TEMPLATE_MODES, TEMPLATE_MODE_TOOLTIP, THINKING_BUDGET_TOOLTIP,
    answer_after_unfinished_thinking, empty_response_message, generate_text,
    generation_error_hint, split_thinking, strip_quote_marks, thinking_unfinished,
    tokenize_for, unload_clip as _unload_clip, with_thinking_limit,
)
from .mykee_prompt_advanced_ultimate import _clip_signature, _same_clip, _sampling_widgets
from .mykee_text_splitter import _Counter

LOG = "Mykee Text Analyzer:"


# --------------------------------------------------------------------------
# Built-in system prompts
# --------------------------------------------------------------------------

NOTES_SYSTEM_ROLLING = (
    "You are reading a long text one part at a time so that a question about it can be "
    "answered afterwards. Each turn gives you the QUESTION, the NOTES SO FAR taken from the "
    "earlier parts (missing for the first part) and the next TEXT PART. Read the text part "
    "carefully and write down, as short plain notes, everything in it that helps to answer "
    "the question: statements that answer it directly, and also indirect evidence such as "
    "hints, circumstances, changes, things people say or do, and details that point toward an "
    "answer when taken together with the earlier notes, even when the text never states the "
    "answer outright. Keep what the text says apart from what you conclude from it, and label "
    "your own conclusions as conclusions. Use the earlier notes only as context: do not repeat "
    "them, write only what this part adds, and say so when this part corrects or contradicts "
    "an earlier note. Never invent anything the text does not support. Write the notes in the "
    "language of the text. If the text part holds nothing that bears on the question, reply "
    "with the single word NONE and nothing else."
)

NOTES_SYSTEM_INDEPENDENT = (
    "You are reading one part of a long text so that a question about the whole text can be "
    "answered afterwards. Each turn gives you the QUESTION and one TEXT PART; the other parts "
    "are not visible to you, so note everything that could matter even when it is only a "
    "fragment of the picture. Read the text part carefully and write down, as short plain "
    "notes, everything in it that helps to answer the question: statements that answer it "
    "directly, and also indirect evidence such as hints, circumstances, changes, things people "
    "say or do, even when the text never states the answer outright. Keep what the text says "
    "apart from what you conclude from it, and label your own conclusions as conclusions. "
    "Never invent anything the text does not support. Write the notes in the language of the "
    "text. If the text part holds nothing that bears on the question, reply with the single "
    "word NONE and nothing else."
)

COMPACT_SYSTEM = (
    "You will receive a QUESTION and a set of NOTES that were taken while reading a long text "
    "part by part. Rewrite the notes more compactly so that they stay as useful as possible "
    "for answering the question: keep every fact and every piece of indirect evidence that "
    "bears on it together with the part numbers it comes from, merge duplicates, drop what "
    "does not matter for the question, and keep conclusions marked as conclusions. Do not add "
    "anything new. Write in the language of the notes and output only the rewritten notes."
)

FINAL_SYSTEM = (
    "You will receive a QUESTION and NOTES that were collected part by part from a long text. "
    "Answer the question using only the notes. When the notes state the answer, give it "
    "plainly. When they do not state it but contain evidence that points toward it, give the "
    "most likely answer, say clearly that it is an inference and name the evidence it rests "
    "on. When the notes do not give enough to answer, say that the text does not tell it "
    "instead of guessing. When two notes disagree, say so and prefer the better supported "
    "one, explaining why. Answer in the language of the question, concisely, without "
    "repeating the notes."
)

_NONE_RE = re.compile(r"^\W*NONE\W*$", re.IGNORECASE)

# Tells the model how much room it has (the model cannot see max_length), so it
# plans a reply that fits instead of running into the limit mid-sentence.
LENGTH_ADDENDUM_TEMPLATE = (
    " Keep your whole reply within about {words} words. If there is more to say than "
    "fits, keep what matters most for the question and leave out the rest, and always "
    "finish your last sentence."
)
TOKENS_PER_WORD = 2.4   # rough, for Hungarian; the addendum asks for ~70% of the room

_SENTENCE_END_RE = re.compile(r"[.!?…][\"'”’»)\]]*(?=\s|$)")


def _trim_to_sentence(text):
    """Cut a reply that ran into max_length back to its last full sentence."""
    last = None
    for m in _SENTENCE_END_RE.finditer(text):
        last = m
    if last is not None and last.end() > len(text) * 0.5:
        return text[:last.end()].rstrip()
    return text


def _ends_cleanly(text):
    """True when the reply ends with a full sentence (it most likely finished by itself)."""
    return bool(re.search(r"[.!?…][\"'”’»)\]]*\s*$", text))


def _hit_limit(text, counter, max_length):
    return counter(text) >= int(max_length * 0.95)


# --------------------------------------------------------------------------
# Caches
# --------------------------------------------------------------------------

_NOTES_CACHE = OrderedDict()   # key -> dict(notes, thinking, parts)
_NOTES_CACHE_MAX = 6
_PARTIAL = OrderedDict()       # key -> dict(next, rolling, entries, thinks)
_PARTIAL_MAX = 4


def _remember(store, key, value, limit):
    store[key] = value
    store.move_to_end(key)
    while len(store) > limit:
        store.popitem(last=False)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _u(value):
    """INPUT_IS_LIST hands every input over as a list: the single value."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _as_texts(value):
    if value is None:
        return None
    if isinstance(value, list):
        return [str(x) for x in value if x is not None]
    return [str(value)]


def _toast(node_id, severity, detail):
    try:
        from server import PromptServer
        if getattr(PromptServer, "instance", None) is not None:
            PromptServer.instance.send_sync(
                "mykee.toast",
                {"node": node_id, "severity": severity, "summary": "Mykee Text Analyzer",
                 "detail": detail, "life": 6000},
            )
    except Exception:
        pass


def _push_progress(node_id, notes, thinking, status):
    """Live view while the node runs: the frontend writes these into the
    notes / thinking / status boxes (the normal result only arrives at the end)."""
    try:
        from server import PromptServer
        if node_id is not None and getattr(PromptServer, "instance", None) is not None:
            PromptServer.instance.send_sync(
                "mykee.text_analyzer.progress",
                {"node": str(node_id), "response_1": notes, "thinking_1": thinking, "status": status},
            )
    except Exception:
        pass


def _interrupt_check():
    if model_management is not None:
        try:
            model_management.throw_exception_if_processing_interrupted()
        except AttributeError:
            pass


def _generate(clip, sys_text, user_text, cfg, label):
    """One tokenize + generate call. Returns (answer, thinking_text); raises on a
    failure (the caller turns it into a message)."""
    words = max(40, int(cfg["max_length"] / TOKENS_PER_WORD * 0.7))
    sys_text = sys_text + LENGTH_ADDENDUM_TEMPLATE.format(words=words)

    def make_prompts(kind):
        return sys_text, user_text

    wrapped = with_thinking_limit(make_prompts, cfg["thinking"], cfg["thinking_budget"])
    tokens, chat_text, _notes, _kind = tokenize_for(
        clip, wrapped, thinking=cfg["thinking"], template_mode=cfg["template_mode"])
    raw = generate_text(
        clip, tokens, chat_text, cfg["max_length"], cfg["sampling_mode"], cfg["seed"],
        cfg["temperature"], cfg["top_k"], cfg["top_p"], cfg["min_p"],
        cfg["repetition_penalty"], cfg["presence_penalty"], cfg["thinking"], cfg["thinking_budget"])
    text, thinking_text = split_thinking(raw)
    if thinking_unfinished(raw, text, thinking_text, cfg["thinking"]):
        text = answer_after_unfinished_thinking(
            clip, make_prompts, None, None, None, None, thinking_text, cfg["max_length"],
            cfg["sampling_mode"], cfg["seed"], cfg["temperature"], cfg["top_k"], cfg["top_p"],
            cfg["min_p"], cfg["repetition_penalty"], cfg["presence_penalty"],
            cfg["template_mode"], f"{LOG} {label}")
    if not text:
        raise RuntimeError(empty_response_message(raw))
    return text, thinking_text


def _stage_cfg(p, prefix):
    g = lambda name: p[f"{prefix}{name}"]
    return {
        "max_length": int(g("max_length")), "sampling_mode": bool(g("sampling_mode")),
        "seed": int(g("seed")), "temperature": float(g("temperature")), "top_k": int(g("top_k")),
        "top_p": float(g("top_p")), "min_p": float(g("min_p")),
        "repetition_penalty": float(g("repetition_penalty")),
        "presence_penalty": float(g("presence_penalty")), "thinking": bool(g("thinking")),
        "thinking_budget": int(p.get("second_thinking_budget" if prefix else "thinking_budget") or 0),
        "template_mode": p.get("template_mode") or "auto",
    }


class _StageError(Exception):
    pass


def _compact(clip, question, notes_text, cfg, notes_max_tokens):
    cfg = dict(cfg, thinking=False, max_length=max(cfg["max_length"], min(4096, int(notes_max_tokens))))
    user = f"QUESTION:\n{question}\n\nNOTES:\n{notes_text}"
    text, _ = _generate(clip, COMPACT_SYSTEM, user, cfg, "condensing the notes")
    return text.strip()


def _reduce_to_fit(clip, question, notes_text, counter, target, cfg, notes_max_tokens, log):
    """Condenses the notes in groups until they fit into `target` tokens."""
    for _ in range(4):
        if counter(notes_text) <= target:
            return notes_text
        blocks = [b for b in re.split(r"\n\s*\n", notes_text) if b.strip()]
        groups, cur, cur_tokens = [], [], 0
        for block in blocks:
            t = counter(block)
            if cur and cur_tokens + t > target * 0.6:
                groups.append(cur)
                cur, cur_tokens = [], 0
            cur.append(block)
            cur_tokens += t
        if cur:
            groups.append(cur)
        log(f"notes too long for one call ({counter(notes_text)} tokens) - condensing {len(groups)} group(s)")
        parts = []
        for group in groups:
            _interrupt_check()
            parts.append(_compact(clip, question, "\n\n".join(group), cfg, notes_max_tokens))
        new_text = "\n\n".join(parts)
        if counter(new_text) >= counter(notes_text):
            break
        notes_text = new_text
    return notes_text


# --------------------------------------------------------------------------
# The node
# --------------------------------------------------------------------------

class MykeeTextAnalyzer:
    CATEGORY = "Mykee/Prompt"
    FUNCTION = "run"
    OUTPUT_NODE = True
    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("response_1", "response_2", "status", "thinking_1", "thinking_2")
    OUTPUT_TOOLTIPS = (
        "The notes collected from the chunks (round 1).",
        "The final answer (round 2), from the notes.",
        "How the run went: parts, parts with notes, what was reused.",
        "The thinking of round 1, per part (when thinking is on).",
        "The thinking of round 2 (when thinking is on).",
    )
    DESCRIPTION = (
        "Answers a question about a long text: reads the chunks one by one and takes "
        "notes with the question in mind, then answers from the notes."
    )

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "clip": ("CLIP", {"tooltip": "The model that reads the chunks and takes the notes "
                                         "(and gives the final answer, unless clip_2 is connected)."}),
            "question": ("STRING", {
                "multiline": True, "default": "",
                "tooltip": "What you want to know about the text. The notes are written for "
                           "THIS question: a new question reads the text again.",
            }),
            "text": ("STRING", {
                "multiline": True, "default": "",
                "tooltip": "The text to analyse (one piece - it must be short enough for the "
                           "model; for a long text use the Mykee Text Splitter and connect its "
                           "chunks to text_in). Can also be connected here directly.",
            }),
            "mode": (["understanding"], {
                "default": "understanding",
                "tooltip": "understanding: the notes collect direct answers AND indirect "
                           "evidence, so a conclusion can be drawn even when the text never "
                           "says it.",
            }),
            "note_mode": (["rolling", "independent"], {
                "default": "rolling",
                "tooltip": "rolling: every chunk is read together with the notes of the "
                           "earlier ones, so clues in different places can be connected; the "
                           "notes are condensed when they grow past notes_max_tokens. "
                           "independent: every chunk is read on its own, the notes are put "
                           "together only for the final answer (simpler, but connections "
                           "between chunks are left to round 2).",
            }),
            "notes_system_prompt": ("STRING", {
                "multiline": True, "default": "",
                "tooltip": "Round 1's system prompt. Empty = built-in default for the "
                           "note_mode. Round 1's user turn is labelled QUESTION, NOTES SO FAR "
                           "(rolling only) and TEXT PART. A chunk without anything relevant "
                           "should be answered with the single word NONE.",
            }),
        }
        notes_widgets = _sampling_widgets("")
        notes_widgets["max_length"][1].update({"default": 768, "max": 8192,
            "tooltip": "Max tokens of the notes written for ONE chunk."})
        notes_widgets["temperature"][1]["default"] = 0.3
        notes_widgets["top_k"][1]["default"] = 40
        notes_widgets["thinking_budget"] = ("INT", {
            "default": 1024, "min": 0, "max": 16384, "step": 64,
            "tooltip": THINKING_BUDGET_TOOLTIP,
        })
        required.update(notes_widgets)
        required["second_system_prompt"] = ("STRING", {
            "multiline": True, "default": "",
            "tooltip": "Round 2's system prompt. Empty = built-in default (answer from the "
                       "notes only; say when it is an inference or when the text does not tell). "
                       "Its user turn is labelled QUESTION and NOTES.",
        })
        second_widgets = _sampling_widgets("second_", include_seed_control=True)
        second_widgets["second_max_length"][1].update({"default": 1024, "max": 8192,
            "tooltip": "Max tokens of the final answer."})
        second_widgets["second_temperature"][1]["default"] = 0.4
        second_widgets["second_thinking_budget"] = ("INT", {
            "default": 512, "min": 0, "max": 16384, "step": 64,
            "tooltip": THINKING_BUDGET_TOOLTIP,
        })
        required.update(second_widgets)
        required.update({
            "notes_max_tokens": ("INT", {
                "default": 2048, "min": 256, "max": 20000, "step": 64,
                "tooltip": "rolling mode: when the notes grow past this many tokens they are "
                           "condensed by the model (keeping what bears on the question). Also "
                           "the size notes are condensed to when they are too long for round 2.",
            }),
            "input_token_limit": ("INT", {
                "default": 16384, "min": 0, "max": 1000000, "step": 256,
                "tooltip": "A chunk (or the typed text) longer than this many tokens is refused "
                           "- use the Mykee Text Splitter. The node never cuts text itself. "
                           "0 = no check.",
            }),
            "unload_after_run": ("BOOLEAN", {
                "default": False,
                "tooltip": "Free the clip that was used last from VRAM when the run is done.",
            }),
        })
        # The inputs one wires or changes all the time come first (they stay
        # visible above the tabs drawn by web/mykee_text_analyzer.js); the
        # rest follows in the order of its tab.
        head = ["clip", "question", "text", "note_mode", "seed", "second_seed", "second_seed_control"]
        ordered = {k: required[k] for k in head}
        ordered.update({k: v for k, v in required.items() if k not in head})
        required = ordered
        return {
            "required": required,
            "optional": {
                "text_in": ("STRING", {
                    "forceInput": True,
                    "tooltip": "The text to analyse: a single text or the chunks list of the "
                               "Mykee Text Splitter. Connected: the text box is hidden and "
                               "ignored.",
                }),
                "chunk_count": ("INT", {
                    "forceInput": True,
                    "tooltip": "Optional: the splitter's chunk_count. Only used to check that no "
                               "chunk got lost on the way.",
                }),
                "clip_2": ("CLIP", {
                    "tooltip": "Optional: a different model for round 2 (the final answer). "
                               "'clip' is taken out of memory first.",
                }),
                "template_mode": (TEMPLATE_MODES, {"default": "auto", "tooltip": TEMPLATE_MODE_TOOLTIP}),
            },
            "hidden": {"node_id": "UNIQUE_ID"},
        }

    # ------------------------------------------------------------------
    def _result(self, answer, notes, status, think1, think2, second_seed=None):
        ui = {
            "response_1_display": [notes], "response_2_display": [answer],
            "thinking_1_display": [think1], "thinking_2_display": [think2],
            "status_display": [status],
        }
        if second_seed is not None:
            ui["second_seed"] = [second_seed]
        return {"ui": ui, "result": (notes, answer, status, think1, think2)}

    def run(self, **kw):
        node_id = _u(kw.get("node_id"))
        p = {k: _u(v) for k, v in kw.items() if k not in ("text", "text_in")}
        clip = p["clip"]
        clip_2 = p.get("clip_2")
        two_clips = clip_2 is not None and not _same_clip(clip, clip_2)
        question = str(p.get("question") or "").strip()
        note_mode = p.get("note_mode") or "rolling"

        text_in = kw.get("text_in")
        chunks = _as_texts(text_in) if text_in is not None else _as_texts(kw.get("text"))
        chunks = [c for c in (chunks or []) if c.strip()]

        def fail(message):
            print(f"{LOG} {message}")
            return self._result(f"[{message}]", "", message, "", "")

        if not question:
            return fail("no question - write what you want to know about the text")
        if not chunks:
            return fail("no text to analyse")

        counter = _Counter(clip)
        limit = int(p.get("input_token_limit") or 0)
        sizes = [counter(c) for c in chunks]
        if limit > 0:
            for i, n in enumerate(sizes, 1):
                if n > limit:
                    where = "The text" if len(chunks) == 1 else f"Part {i}"
                    return fail(f"{where} is too long: {n} tokens, the limit is {limit}. Cut it with "
                                f"the Mykee Text Splitter first and connect its chunks to text_in "
                                f"(or raise input_token_limit if the model can take more).")
        total = len(chunks)
        status_lines = [f"{total} part(s), {sum(sizes)} tokens in all"]
        expected = _u(kw.get("chunk_count")) if kw.get("chunk_count") is not None else None
        if expected is not None and int(expected) != total:
            status_lines.append(f"WARNING: chunk_count says {expected} but {total} part(s) arrived")
            _toast(node_id, "warn", f"chunk_count is {expected}, but {total} part(s) arrived.")

        notes_cfg = _stage_cfg(p, "")
        final_cfg = _stage_cfg(p, "second_")
        notes_max_tokens = int(p.get("notes_max_tokens") or 2048)
        custom_notes_prompt = (p.get("notes_system_prompt") or "").strip()
        notes_system = custom_notes_prompt or (
            NOTES_SYSTEM_ROLLING if note_mode == "rolling" else NOTES_SYSTEM_INDEPENDENT)
        final_system = (p.get("second_system_prompt") or "").strip() or FINAL_SYSTEM

        sig = _clip_signature(clip) or ("object", id(clip))
        key = (
            tuple(hashlib.sha1(c.encode("utf-8", "replace")).hexdigest() for c in chunks),
            question, note_mode, p.get("mode"), notes_system,
            tuple(sorted((k, str(v)) for k, v in notes_cfg.items())), notes_max_tokens, sig,
        )

        def log(line):
            print(f"{LOG} {line}")

        # ---------------- round 1: the notes -------------------------------
        notes_text = ""
        think1 = ""
        error = None
        cached = _NOTES_CACHE.get(key)
        notes_ran = False
        if cached is not None:
            _NOTES_CACHE.move_to_end(key)
            notes_text, think1 = cached["notes"], cached["thinking"]
            status_lines.append("round 1: notes reused (same text, question and round 1 settings)")
            status_lines.extend(cached["parts"])
        else:
            notes_ran = True
            if two_clips:
                _unload_clip(clip_2)
            state = _PARTIAL.get(key) or {"next": 0, "rolling": "", "entries": [], "thinks": [], "parts": []}
            if state["next"]:
                log(f"continuing at part {state['next'] + 1} of {total}")
                status_lines.append(f"round 1: continued at part {state['next'] + 1}")
            _remember(_PARTIAL, key, state, _PARTIAL_MAX)
            pbar = ProgressBar(total) if ProgressBar is not None else None
            if pbar is not None and state["next"]:
                pbar.update(state["next"])
            try:
                for i in range(state["next"], total):
                    _interrupt_check()
                    chunk = chunks[i]
                    _push_progress(
                        node_id,
                        state["rolling"] if note_mode == "rolling" else "\n\n".join(state["entries"]),
                        "\n\n".join(state["thinks"]),
                        "\n".join(status_lines + state["parts"] + [f"reading part {i + 1} of {total} ..."]))
                    user = f"QUESTION:\n{question}\n\n"
                    if note_mode == "rolling" and state["rolling"]:
                        user += f"NOTES SO FAR:\n{state['rolling']}\n\n"
                    user += f"TEXT PART {i + 1} OF {total}:\n{chunk}"
                    try:
                        out, think = _generate(clip, notes_system, user, notes_cfg, f"part {i + 1}/{total}")
                    except Exception as e:
                        if type(e).__name__ == "InterruptProcessingException":
                            raise
                        error = (f"part {i + 1} of {total} failed: {type(e).__name__}: {e}\n"
                                 + generation_error_hint(False, e))
                        break
                    out = strip_quote_marks(out).strip()
                    cut_note = ""
                    if not _NONE_RE.match(out) and _hit_limit(out, counter, notes_cfg["max_length"]):
                        if _ends_cleanly(out):
                            # long, but it ended with a full sentence: most likely complete
                            cut_note = " - close to max_length, but it ends with a full sentence"
                            log(f"part {i + 1}/{total}: the notes are close to max_length")
                        else:
                            trimmed = _trim_to_sentence(out)
                            cut_note = (" - hit max_length, cut at the last full sentence (raise max_length)"
                                        if len(trimmed) < len(out) else " - hit max_length (raise max_length)")
                            out = trimmed
                            log(f"part {i + 1}/{total}: the notes hit max_length")
                            _toast(node_id, "warn", f"Part {i + 1}: the notes hit max_length and were cut. "
                                                    f"Raise max_length if notes get lost.")
                    if think:
                        state["thinks"].append(f"[Part {i + 1}]\n{think}")
                    if _NONE_RE.match(out):
                        state["parts"].append(f"part {i + 1}: nothing relevant")
                        log(f"part {i + 1}/{total}: nothing relevant")
                    else:
                        entry = f"[Part {i + 1}] {out}"
                        state["entries"].append(entry)
                        state["parts"].append(f"part {i + 1}: notes ({counter(out)} tokens){cut_note}")
                        log(f"part {i + 1}/{total}: {counter(out)} tokens of notes")
                        if note_mode == "rolling":
                            state["rolling"] = (state["rolling"] + "\n\n" + entry) if state["rolling"] else entry
                            if counter(state["rolling"]) > notes_max_tokens:
                                try:
                                    state["rolling"] = _compact(
                                        clip, question, state["rolling"], notes_cfg, notes_max_tokens)
                                    log(f"notes condensed to {counter(state['rolling'])} tokens")
                                except Exception as e:
                                    if type(e).__name__ == "InterruptProcessingException":
                                        raise
                                    error = (f"condensing the notes failed: {type(e).__name__}: {e}\n"
                                             + generation_error_hint(False, e))
                                    break
                    state["next"] = i + 1
                    if pbar is not None:
                        pbar.update(1)
            finally:
                _remember(_PARTIAL, key, state, _PARTIAL_MAX)
            think1 = "\n\n".join(state["thinks"])
            notes_text = state["rolling"] if note_mode == "rolling" else "\n\n".join(state["entries"])
            status_lines.extend(state["parts"])
            if error is None:
                _PARTIAL.pop(key, None)
                _remember(_NOTES_CACHE, key, {"notes": notes_text, "thinking": think1,
                                              "parts": list(state["parts"])}, _NOTES_CACHE_MAX)

        with_notes = sum(1 for line in status_lines if "notes (" in line)
        if error is not None:
            status_lines.append(f"STOPPED: {error}")
            status_lines.append("Queue again to continue at the part where it stopped.")
            status = "\n".join(status_lines)
            print(f"{LOG} {error}")
            return self._result(f"[{error}]", notes_text, status, think1, "")

        # ---------------- round 2: the answer ------------------------------
        answer_clip = clip_2 if two_clips else clip
        _push_progress(node_id, notes_text, think1,
                       "\n".join(status_lines + ["writing the final answer from the notes ..."]))
        if two_clips and notes_ran:
            _unload_clip(clip)
        think2 = ""
        try:
            if not notes_text.strip():
                notes_for_answer = "(none of the parts contained anything relevant to the question)"
            else:
                target = max(1000, (limit if limit > 0 else 16384) - 1024)
                notes_for_answer = _reduce_to_fit(
                    answer_clip, question, notes_text, counter, target, notes_cfg, notes_max_tokens,
                    lambda line: (log(line), status_lines.append(line)))
            user = (f"QUESTION:\n{question}\n\nNOTES (collected part by part from a long text of "
                    f"{total} part(s)):\n{notes_for_answer}")
            answer, think2 = _generate(answer_clip, final_system, user, final_cfg, "final answer")
            answer = strip_quote_marks(answer)
            if _hit_limit(answer, counter, final_cfg["max_length"]) and not _ends_cleanly(answer):
                answer = _trim_to_sentence(answer)
                status_lines.append("final answer hit second_max_length and was cut at the last full "
                                    "sentence (raise second_max_length)")
                _toast(node_id, "warn", "The final answer hit second_max_length and was cut.")
        except Exception as e:
            if type(e).__name__ == "InterruptProcessingException":
                raise
            msg = f"final answer failed: {type(e).__name__}: {e}\n" + generation_error_hint(False, e)
            status_lines.append(f"STOPPED: {msg}")
            print(f"{LOG} {msg}")
            return self._result(f"[{msg}]", notes_text, "\n".join(status_lines), think1, "")

        seed = int(p.get("second_seed") or 0)
        control = p.get("second_seed_control") or "fixed"
        if control == "randomize":
            next_seed = random.randint(0, 0xffffffffffffffff)
        elif control == "increment":
            next_seed = min(0xffffffffffffffff, seed + 1)
        elif control == "decrement":
            next_seed = max(0, seed - 1)
        else:
            next_seed = seed

        status_lines.insert(1, f"notes found in {with_notes} of {total} part(s), note_mode {note_mode}")
        status = "\n".join(status_lines)
        print("=" * 70)
        print(f"{LOG} {status_lines[0]}; {status_lines[1]}")
        print("--- response_1 (notes) ---")
        print(notes_text)
        print("--- response_2 (answer) ---")
        print(answer)

        if p.get("unload_after_run"):
            _unload_clip(answer_clip)
        return self._result(answer, notes_text, status, think1, think2, next_seed)


NODE_CLASS_MAPPINGS = {
    "MykeeTextAnalyzer": MykeeTextAnalyzer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeTextAnalyzer": "Mykee Text Analyzer",
}
