"""
ComfyUI-Mykee-Nodes / Before/After Text Injection node

"Mykee Before/After Text Injection": the same "before" and "after" text is
put around every connected text. text_N in -> text_N out, 1..50 pairs (the
visible count is managed by web/mykee_text_injection.js).

    output = before + before_separator + text + after_separator + after

A separator is only used when its text is not empty: an empty "before"
drops before_separator, an empty "after" drops after_separator. An empty
(or not connected) text gives before + before_separator + after.

Bypass: when the node is bypassed, ComfyUI connects output N to input N
by slot index. So text_1..text_N must be the node's only input slots - the
widgets are socketless (see the frontend for older workflows).

Lists are kept per input: a list input (e.g. the batch output of Mykee Text
Switch/Batch) gives a list of the same length on its own output, without
affecting the other pairs.
"""

MAX_TEXTS = 50


def _unescape(separator):
    return str(separator).replace("\\n", "\n").replace("\\t", "\t")


def _first(value, default):
    """INPUT_IS_LIST wraps the widget values in lists too."""
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


def inject(text, before, before_separator, after, after_separator):
    result = str(text)
    if not result:
        # Nothing to wrap: no doubled separator between before and after.
        return before + (before_separator if before and after else "") + after
    if before:
        result = before + before_separator + result
    if after:
        result = result + after_separator + after
    return result


class MykeeBeforeAfterTextInjection:
    CATEGORY = "Mykee/Prompt"
    FUNCTION = "main"
    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",) * MAX_TEXTS
    RETURN_NAMES = tuple(f"text_{i}" for i in range(1, MAX_TEXTS + 1))
    OUTPUT_IS_LIST = (True,) * MAX_TEXTS

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            f"text_{i}": (
                "STRING",
                {"forceInput": True, "tooltip": f"Comes out on text_{i}, with the before / after text around it."},
            )
            for i in range(1, MAX_TEXTS + 1)
        }
        return {
            "required": {
                "text_count": (
                    "INT",
                    {"socketless": True, "default": 2, "min": 1, "max": MAX_TEXTS, "tooltip": "Number of text input / output pairs."},
                ),
                "before_text": (
                    "STRING",
                    {"socketless": True, "default": "", "multiline": True, "tooltip": "Put in front of every text. Empty = nothing (and no separator)."},
                ),
                "before_separator": (
                    "STRING",
                    {"socketless": True, "default": "\\n", "tooltip": "Between the before text and the text (\\n = new line, \\t = tab). Only used if before_text is not empty."},
                ),
                "after_text": (
                    "STRING",
                    {"socketless": True, "default": "", "multiline": True, "tooltip": "Put after every text. Empty = nothing (and no separator)."},
                ),
                "after_separator": (
                    "STRING",
                    {"socketless": True, "default": "\\n", "tooltip": "Between the text and the after text (\\n = new line, \\t = tab). Only used if after_text is not empty."},
                ),
            },
            "optional": optional,
        }

    def main(self, text_count, before_text, before_separator, after_text, after_separator, **kwargs):
        count = max(1, min(MAX_TEXTS, int(_first(text_count, 2))))
        before = str(_first(before_text, ""))
        after = str(_first(after_text, ""))
        before_sep = _unescape(_first(before_separator, "\\n"))
        after_sep = _unescape(_first(after_separator, "\\n"))

        outputs = []
        for i in range(1, MAX_TEXTS + 1):
            texts = kwargs.get(f"text_{i}")
            if i > count:
                outputs.append([])
                continue
            if texts is None:
                texts = [""]  # not connected: only the before / after text
            elif not isinstance(texts, list):
                texts = [texts]
            outputs.append([inject(t, before, before_sep, after, after_sep) for t in texts])
        return tuple(outputs)


NODE_CLASS_MAPPINGS = {
    "MykeeBeforeAfterTextInjection": MykeeBeforeAfterTextInjection,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeBeforeAfterTextInjection": "Mykee Before/After Text Injection",
}
