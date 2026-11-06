"""
ComfyUI-Mykee-Nodes / text splitter node

"Mykee Text Splitter": cuts a long text into chunks that a chat model can
digest, without ever cutting inside a sentence.

- The size of a chunk is measured in TOKENS, with the tokenizer of the
  connected clip (so it is exact for that model).
- limit_mode "manual": chunk_tokens is the size.
  limit_mode "auto": the node probes the connected model - it runs real
  prompt passes of growing size, catches the out-of-memory error, and
  settles on a size that fits (minus a reserve for the question and the
  answer, and a safety margin). The result is remembered per model/GPU.
- Cuts are made at paragraph ends when the chunk is already well filled,
  otherwise at the last sentence end that still fits. Abbreviations
  (dr., pl., stb.), initials (J. K.), ordinal numbers (5. június) and the
  dialogue dashes of Hungarian text are not sentence ends. A sentence that
  alone is longer than the limit is the only thing ever cut smaller (at
  clause marks, then at words).
- The chunks go out as a list (the nodes after it run once per chunk), plus
  their count, the clip (pass-through) and an info text.
- Optionally the chunks are also written to numbered .txt files. The split
  itself is cached, so switching saving on after a run does not split again.
"""

import gc
import hashlib
import os
import re
import time
from collections import OrderedDict

try:
    from comfy_execution.graph import ExecutionBlocker
except ImportError:  # very old ComfyUI
    ExecutionBlocker = None

try:
    import comfy.model_management as model_management
except Exception:  # pragma: no cover - outside ComfyUI
    model_management = None

from .mykee_llm_common import (
    _is_out_of_memory,
    completion_budget,
    tokenize_chat,
)

LOG = "Mykee Text Splitter:"
MIN_CHUNK_TOKENS = 256


# --------------------------------------------------------------------------
# Sentence / paragraph units
# --------------------------------------------------------------------------

# lower-case, no dot. A dot after one of these is never a sentence end.
_ABBREVIATIONS = {
    "dr", "pl", "stb", "ill", "ld", "vö", "ún", "kb", "sz", "tel", "ifj", "id",
    "özv", "prof", "szt", "st", "mr", "mrs", "ms", "jr", "sr", "nr", "fsz", "em",
    "ajt", "hrsz", "ker", "krt", "ny", "du", "db", "kft", "zrt", "bt", "rt",
    "nyrt", "évf", "old", "ábr", "jan", "febr", "márc", "ápr", "máj", "jún",
    "júl", "aug", "szept", "okt", "nov", "dec", "köv", "kk", "ua", "uo", "vm",
    "hu", "min", "mp", "óra", "ü", "ált", "alk", "ang", "ném", "fr", "ol", "lat",
    "gör", "stílszerűen", "ev", "ea", "gy", "ty", "ny",
}

_UPPER = "A-ZÁÉÍÓÖŐÚÜŰ"
_CLOSERS = "\"'”’»)\\]"
_OPENERS = "\"'„“‘«(\\["
# a sentence end candidate: . ! ? … (several allowed), closing quotes/brackets,
# then whitespace
_END_RE = re.compile(r"([.!?…]+)([" + _CLOSERS + r"]*)(\s+)")
_UPPER_RE = re.compile("[" + _UPPER + "]")
_WORD_BEFORE_RE = re.compile(r"([^\W\d_]+)$", re.UNICODE)


def _is_sentence_end(line, match):
    """Is the end candidate `match` (in `line`) a real sentence end?"""
    marks = match.group(1)
    after = line[match.end():]
    if not after:
        return False
    nxt = after[0]
    # Dialogue dashes never start a new unit inside a line, quote-tag text
    # in lower case ('... ?" - kérdezte') does not either.
    if nxt in "—–-":
        return False
    probe = after.lstrip(_OPENERS) or after
    first = probe[0]
    if not (_UPPER_RE.match(first) or first.isdigit()):
        return False
    if marks.endswith(".") and not marks.endswith("…") and len(marks) == 1:
        before = line[:match.start()]
        m = _WORD_BEFORE_RE.search(before)
        if m:
            word = m.group(1)
            if word.lower() in _ABBREVIATIONS:
                return False
            if len(word) == 1 and word.isupper():
                return False  # an initial: J. K. Rowling
    return True


def _split_line(line):
    """Sentences of one line (paragraph): [(text, whitespace_after), ...]."""
    out = []
    start = 0
    for m in _END_RE.finditer(line):
        if _is_sentence_end(line, m):
            out.append((line[start:m.end(2)], m.group(3)))
            start = m.end()
    tail = line[start:]
    if tail.strip():
        out.append((tail.rstrip(), ""))
    elif out:
        out[-1] = (out[-1][0], out[-1][1] + tail)
    return out


def _make_units(text):
    """The whole text as units: [[text, whitespace_after, paragraph_end], ...]."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"(\n+)", text)
    units = []
    i = 0
    while i < len(parts):
        line = parts[i]
        sep = parts[i + 1] if i + 1 < len(parts) else ""
        i += 2
        if not line.strip():
            if units and sep:
                units[-1][1] += line + sep
            continue
        sentences = _split_line(line)
        for k, (sentence, ws) in enumerate(sentences):
            last = k == len(sentences) - 1
            units.append([sentence, (sep if last else ws) or ("" if last else " "), last])
    return units


# --------------------------------------------------------------------------
# Token counting (tokenizer of the connected clip)
# --------------------------------------------------------------------------

class _Counter:
    """count(text) -> number of tokens, with the clip's own tokenizer."""

    def __init__(self, clip):
        self.method = "estimate (3 characters per token)"
        self._fn = lambda text: max(1, len(text) // 3) if text else 0
        try:
            tokenizer = getattr(clip, "tokenizer", None)
            inner = getattr(tokenizer, getattr(tokenizer, "clip", ""), None)
            hf = getattr(inner, "tokenizer", None)
            if hf is not None and self._works(hf):
                self._fn = self._hf_count(hf)
                self.method = "the clip's tokenizer"
                return
        except Exception:
            pass
        try:
            def via_clip(text):
                tokens = clip.tokenize(text, skip_template=True, min_length=1)
                return sum(len(chunk) for chunks in tokens.values() for chunk in chunks)
            if via_clip("test") > 0:
                self._fn = via_clip
                self.method = "the clip's tokenize()"
        except Exception:
            pass

    @staticmethod
    def _ids(hf, text):
        try:
            out = hf(text, add_special_tokens=False)
        except TypeError:
            out = hf(text)
        if isinstance(out, dict):
            out = out.get("input_ids")
        elif hasattr(out, "input_ids"):
            out = out.input_ids
        return out

    def _works(self, hf):
        ids = self._ids(hf, "Teszt mondat egy kis szöveggel.")
        return ids is not None and 0 < len(ids) < 40

    def _hf_count(self, hf):
        def count(text):
            if not text:
                return 0
            return len(self._ids(hf, text))
        return count

    def __call__(self, text):
        return self._fn(text)


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

def _break_long(text, limit, counter):
    """Pieces of one over-long sentence: clause marks first, then words."""
    pieces = [text]
    for pattern in (r"(?<=[;:])\s+", r"(?<=,)\s+", r"\s+"):
        nxt = []
        for piece in pieces:
            if counter(piece) <= limit:
                nxt.append(piece)
                continue
            parts = re.split(pattern, piece)
            acc = ""
            for part in parts:
                cand = (acc + " " + part) if acc else part
                if acc and counter(cand) > limit:
                    nxt.append(acc)
                    acc = part
                else:
                    acc = cand
            if acc:
                nxt.append(acc)
        pieces = nxt
        if all(counter(p) <= limit for p in pieces):
            break
    return pieces


def split_text(text, limit, counter, paragraph_fill=70, overlap=0):
    """Returns (chunks, notes). Every chunk has at most `limit` tokens."""
    limit = max(MIN_CHUNK_TOKENS, int(limit))
    notes = []
    units = _make_units(text)
    if not units:
        return [], notes

    counts = [counter(u[0]) for u in units]
    broken = 0
    i = 0
    while i < len(units):
        if counts[i] > limit:
            pieces = _break_long(units[i][0], limit, counter)
            broken += 1
            new_units = []
            for k, piece in enumerate(pieces):
                last = k == len(pieces) - 1
                new_units.append([piece, units[i][1] if last else " ", units[i][2] if last else False])
            units[i:i + 1] = new_units
            counts[i:i + 1] = [counter(u[0]) for u in new_units]
            i += len(new_units)
        else:
            i += 1
    if broken:
        notes.append(f"{broken} sentence(s) were longer than the limit alone and had to be cut at clause marks/words.")

    def joined(a, b):
        parts = []
        for k in range(a, b):
            parts.append(units[k][0])
            if k < b - 1:
                parts.append(units[k][1])
        return "".join(parts).strip()

    chunks = []
    n = len(units)
    start = 0
    prev_end = 0
    overlap = max(0, int(overlap))
    while start < n:
        used = 0
        j = start
        para_cut = None
        while j < n and used + counts[j] <= limit:
            used += counts[j]
            j += 1
            if units[j - 1][2]:
                para_cut = (j, used)
        if j >= n:
            end = n
        else:
            end = max(j, start + 1)
            if para_cut and para_cut[0] != j and para_cut[0] > prev_end \
                    and para_cut[1] >= limit * max(0, min(100, paragraph_fill)) / 100.0:
                end = para_cut[0]
        text_chunk = joined(start, end)
        # verify with the real count of the joined text (tokens at the joints can
        # differ from the sum of the parts): step back to the last end that fits
        while end - start > 1 and counter(text_chunk) > limit:
            end -= 1
            text_chunk = joined(start, end)
        if text_chunk:
            chunks.append(text_chunk)
        if end >= n:
            break
        prev_end = end
        next_start = end - overlap if overlap else end
        start = next_start if next_start > start else end
    return chunks, notes


# --------------------------------------------------------------------------
# Memory probe (auto mode)
# --------------------------------------------------------------------------

_PROBE_CACHE = {}


def _clip_signature(clip):
    try:
        from .mykee_prompt_advanced_ultimate import _clip_signature as sig
        return sig(clip)
    except Exception:
        return None


def _device_info(clip):
    """(label, total_bytes) of the device the clip runs on, or None for CPU."""
    try:
        import torch
        device = getattr(getattr(clip, "patcher", None), "load_device", None)
        if device is None and model_management is not None:
            device = model_management.get_torch_device()
        if device is None or getattr(device, "type", "cpu") != "cuda":
            return None
        props = torch.cuda.get_device_properties(device)
        return (f"{props.name}:{device}", int(props.total_memory))
    except Exception:
        return None


def _free_cuda_cache():
    try:
        gc.collect()
        if model_management is not None:
            model_management.soft_empty_cache(True)
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _run_probe_pass(clip, base_text, tokens_per_char, n_tokens):
    """One real prompt pass of about n_tokens. True = fits, False = out of memory."""
    chars = max(64, int(n_tokens / max(tokens_per_char, 0.05)))
    reps = chars // max(1, len(base_text)) + 1
    text = (base_text + "\n") * reps
    text = text[:chars]
    tokens, chat_text, _ = tokenize_chat(clip, "", text)
    try:
        clip.generate(
            tokens,
            do_sample=False,
            max_length=completion_budget(clip, tokens, 1, chat_text),
            temperature=1.0, top_k=0, top_p=1.0, min_p=0.0,
            repetition_penalty=1.0, presence_penalty=0.0, seed=0,
        )
        return True
    except Exception as e:
        if _is_out_of_memory(e):
            _free_cuda_cache()
            return False
        raise
    finally:
        del tokens


def _interrupt_check():
    if model_management is not None:
        try:
            model_management.throw_exception_if_processing_interrupted()
        except AttributeError:
            pass


def probe_prompt_limit(clip, base_text, counter, upper, reprobe=False):
    """Largest prompt (in tokens) that the clip can take without running out
    of memory, at most `upper` (the search stops there). Returns
    (tokens_ok, tested_upper_reached, note)."""
    device = _device_info(clip)
    if device is None:
        return None, False, "no CUDA device - the memory probe was skipped"
    key = (_clip_signature(clip), device)
    cached = None if reprobe or key[0] is None else _PROBE_CACHE.get(key)
    lo = cached["lo"] if cached else 0
    hi = cached["hi"] if cached else None
    if cached and (hi is not None or lo >= upper):
        return lo, hi is None, f"remembered probe result ({lo} tokens ok)"

    sample = base_text[:20000]
    tpc = max(counter(sample), 1) / max(len(sample), 1)

    def attempt(n):
        _interrupt_check()
        t0 = time.time()
        ok = _run_probe_pass(clip, base_text, tpc, n)
        print(f"{LOG} probe {n} tokens: {'fits' if ok else 'out of memory'} ({time.time() - t0:.1f}s)")
        return ok

    if lo == 0 and hi is None:
        n = min(2048, upper)
        while True:
            if attempt(n):
                lo = n
                break
            hi = n
            n //= 2
            if n < 256:
                break
        if lo == 0:
            _PROBE_CACHE[key] = {"lo": 0, "hi": hi}
            return 0, False, "even 256 tokens did not fit"
    n = lo * 2
    while hi is None and lo < upper:
        n = min(n, upper)
        if attempt(n):
            lo = n
            n *= 2
        else:
            hi = n
    if hi is not None:
        for _ in range(5):
            if hi - lo <= max(256, int(lo * 0.05)):
                break
            mid = ((lo + hi) // 2 // 128) * 128
            if mid <= lo:
                break
            if attempt(mid):
                lo = mid
            else:
                hi = mid
    _PROBE_CACHE[key] = {"lo": lo, "hi": hi}
    return lo, hi is None, f"probe finished ({lo} tokens fit)"


# --------------------------------------------------------------------------
# Result cache, saving
# --------------------------------------------------------------------------

_SPLIT_CACHE = OrderedDict()
_SPLIT_CACHE_MAX = 6


def _safe_base_name(name):
    name = (name or "").strip()
    name = os.path.basename(name.replace("\\", "/"))
    if name.lower().endswith(".txt"):
        name = name[:-4]
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return name or "text"


def save_chunks_to_folder(chunks, folder, file_name):
    folder = (folder or "").strip().strip('"')
    if not folder:
        return None, "save_chunks is on, but save_folder is empty - nothing was saved."
    try:
        os.makedirs(folder, exist_ok=True)
        base = _safe_base_name(file_name)
        width = max(3, len(str(len(chunks))))
        for i, chunk in enumerate(chunks, 1):
            path = os.path.join(folder, f"{base}_{i:0{width}d}.txt")
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(chunk)
        return folder, f"saved {len(chunks)} file(s) to {folder} ({base}_{1:0{width}d}.txt ...)"
    except Exception as e:
        return None, f"saving failed: {type(e).__name__}: {e}"


# --------------------------------------------------------------------------
# The node
# --------------------------------------------------------------------------

class MykeeTextSplitter:
    CATEGORY = "Mykee/Prompt"
    FUNCTION = "main"
    RETURN_TYPES = ("STRING", "INT", "CLIP", "STRING")
    RETURN_NAMES = ("chunks", "chunk_count", "clip", "info")
    OUTPUT_IS_LIST = (True, False, False, False)
    OUTPUT_TOOLTIPS = (
        "The chunks, as a list: the nodes after it run once per chunk.",
        "How many chunks there are.",
        "The connected clip, passed through - wire it on to the analyser if "
        "that should use the same model.",
        "What was measured and how the text was split (also printed to the console).",
    )
    DESCRIPTION = (
        "Cuts a long text into chunks of a token size that fits the model, "
        "only at sentence ends (paragraph ends when possible)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {
                    "tooltip": "The model whose tokenizer measures the chunks, and "
                               "(auto mode) whose memory limit is probed. Connect "
                               "the same clip the analyser will use.",
                }),
                "text": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "The text to split. Connect a text here to use "
                               "that instead of typing (the box is then greyed out).",
                }),
                "limit_mode": (["auto", "manual"], {
                    "default": "auto",
                    "tooltip": "auto: probe the model with real prompt passes of "
                               "growing size, catch the out-of-memory error and use "
                               "the biggest size that fits (see reserve_tokens, "
                               "safety_percent, max_chunk_tokens). The result is "
                               "remembered per model and GPU. manual: chunk_tokens is "
                               "the chunk size.",
                }),
                "chunk_tokens": ("INT", {
                    "default": 8192, "min": MIN_CHUNK_TOKENS, "max": 1000000, "step": 64,
                    "tooltip": "manual mode: the size of a chunk in tokens. "
                               "Ignored in auto mode.",
                }),
                "reserve_tokens": ("INT", {
                    "default": 2048, "min": 0, "max": 100000, "step": 128,
                    "tooltip": "auto mode: room kept free in the model's window for "
                               "the question, the system prompt and the answer "
                               "(including thinking). Subtracted from what the probe "
                               "found.",
                }),
                "safety_percent": ("INT", {
                    "default": 85, "min": 30, "max": 100, "step": 5,
                    "tooltip": "auto mode: only this percentage of the probed size "
                               "is used, because the real run needs a little more "
                               "memory than the probe.",
                }),
                "max_chunk_tokens": ("INT", {
                    "default": 16384, "min": 0, "max": 1000000, "step": 256,
                    "tooltip": "auto mode: upper limit of a chunk. Small models read "
                               "very long texts worse even when the memory would "
                               "allow it. 0 = no limit (the probe then goes as far as "
                               "the text or the memory reaches).",
                }),
                "paragraph_fill_percent": ("INT", {
                    "default": 70, "min": 0, "max": 100, "step": 5,
                    "tooltip": "When the next paragraph does not fit whole: if the "
                               "chunk already holds at least this percentage of the "
                               "limit, it is closed at the paragraph end. Otherwise it "
                               "is filled up to the last sentence end that fits. 100 = "
                               "always fill to the limit, 0 = always close at "
                               "paragraph ends.",
                }),
                "overlap_sentences": ("INT", {
                    "default": 0, "min": 0, "max": 20, "step": 1,
                    "tooltip": "Number of sentences repeated at the start of the next "
                               "chunk, so a scene is not lost at the cut. 0 = off.",
                }),
                "save_chunks": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Also write the chunks to numbered .txt files "
                               "(file_name_001.txt, ...) in save_folder. Switching "
                               "it on after a run does not split again - the "
                               "remembered chunks are written.",
                }),
                "save_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Folder for the saved files (created if missing).",
                }),
                "file_name": ("STRING", {
                    "default": "",
                    "tooltip": "Base name of the saved files, e.g. 'book' gives "
                               "book_001.txt. Empty = 'text'.",
                }),
                "reprobe": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "auto mode: forget the remembered probe result and "
                               "measure again (e.g. after freeing VRAM). Switch it "
                               "off again afterwards - while it is on the text is "
                               "split anew every run.",
                }),
            },
        }

    def main(self, clip, text, limit_mode, chunk_tokens, reserve_tokens, safety_percent,
             max_chunk_tokens, paragraph_fill_percent, overlap_sentences, save_chunks,
             save_folder, file_name, reprobe=False):
        text = text or ""
        if not text.strip():
            msg = "no text to split"
            print(f"{LOG} {msg}.")
            if ExecutionBlocker is not None:
                return ([ExecutionBlocker(None)], 0, clip, msg)
            return ([], 0, clip, msg)

        sig = _clip_signature(clip)
        text_hash = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
        key = (text_hash, sig, limit_mode, int(chunk_tokens), int(reserve_tokens),
               int(safety_percent), int(max_chunk_tokens), int(paragraph_fill_percent),
               int(overlap_sentences))
        cached = None
        if sig is not None and not (reprobe and limit_mode == "auto"):
            cached = _SPLIT_CACHE.get(key)
            if cached is not None:
                _SPLIT_CACHE.move_to_end(key)

        if cached is not None:
            chunks, info_lines = cached
            info_lines = list(info_lines) + ["(split taken from the cache)"]
        else:
            counter = _Counter(clip)
            info_lines = [f"tokens counted with {counter.method}"]
            total = counter(text)
            info_lines.append(f"text: {len(text)} characters, {total} tokens")

            if limit_mode == "auto":
                cap = int(max_chunk_tokens)
                safety = max(30, min(100, int(safety_percent))) / 100.0
                reserve = max(0, int(reserve_tokens))
                if cap > 0:
                    upper = int((cap + reserve) / safety) + 1
                else:
                    upper = total + reserve + 1
                upper = min(upper, total + reserve + 1)
                limit = None
                note = ""
                try:
                    ok_tokens, _, note = probe_prompt_limit(clip, text, counter, upper, reprobe)
                except Exception as e:
                    ok_tokens = None
                    note = f"probe failed ({type(e).__name__}: {e})"
                if ok_tokens:
                    limit = int(ok_tokens * safety) - reserve
                    if cap > 0:
                        limit = min(limit, cap)
                    if ok_tokens >= upper:
                        limit = max(limit, min(cap if cap > 0 else total, total))
                    info_lines.append(f"memory probe: {note}; chunk size {limit} tokens "
                                      f"(reserve {reserve}, safety {int(safety * 100)}%)")
                else:
                    fallback = int(chunk_tokens)
                    limit = fallback
                    info_lines.append(f"memory probe unavailable ({note}); using chunk_tokens = {fallback}")
                limit = max(MIN_CHUNK_TOKENS, limit)
            else:
                limit = max(MIN_CHUNK_TOKENS, int(chunk_tokens))
                info_lines.append(f"manual chunk size {limit} tokens")

            chunks, notes = split_text(text, limit, counter, paragraph_fill_percent, overlap_sentences)
            info_lines.extend(notes)
            sizes = [counter(c) for c in chunks]
            if sizes:
                info_lines.append(f"{len(chunks)} chunk(s), tokens per chunk: min {min(sizes)}, "
                                  f"max {max(sizes)}, average {sum(sizes) // len(sizes)}")
            if sig is not None:
                _SPLIT_CACHE[key] = (chunks, list(info_lines))
                while len(_SPLIT_CACHE) > _SPLIT_CACHE_MAX:
                    _SPLIT_CACHE.popitem(last=False)

        if save_chunks:
            _, saved = save_chunks_to_folder(chunks, save_folder, file_name)
            info_lines.append(saved)

        info = "\n".join(info_lines)
        for line in info_lines:
            print(f"{LOG} {line}")
        return (chunks, len(chunks), clip, info)


NODE_CLASS_MAPPINGS = {
    "MykeeTextSplitter": MykeeTextSplitter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeTextSplitter": "Mykee Text Splitter",
}
