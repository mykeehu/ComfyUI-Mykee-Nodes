"""
ComfyUI-Mykee-Nodes / text switch-batch node

"Mykee Text Switch/Batch": 0..49 numbered text fields on one node, each
with a radio button (switch mode) or a checkbox (batch mode) at its end.

- switch mode: only the text picked with the radio button goes out.
- batch mode: every ticked, non-empty text goes out, in field order, as a
  list - so the nodes after it run once per text (one prompt per text for
  a CLIPTextEncode -> KSampler chain). "joined" has them in one string.

Any field can also be fed from its own text_N input socket; the connected
value then replaces the typed one, but the radio / checkbox still decides
whether it is used. The inputs are lazy: an input whose field is not
selected is never evaluated.

The fields, radio and checkboxes are drawn by web/mykee_text_switch.js and
stored in the hidden texts_config widget. For the prompt the frontend only
sends the selected fields, so editing a text that is not selected does not
make anything re-run.
"""

import json

try:
    from comfy_execution.graph import ExecutionBlocker
except ImportError:  # very old ComfyUI
    ExecutionBlocker = None

MAX_TEXTS = 49


def _items(texts_config):
    """Selected fields from the frontend: [{"index": 1-based, "text": str}, ...]."""
    try:
        cfg = json.loads(texts_config) if texts_config else {}
    except (TypeError, ValueError):
        cfg = {}
    items = cfg.get("items") if isinstance(cfg, dict) else None
    result = []
    for item in items if isinstance(items, list) else []:
        try:
            index = int(item.get("index"))
        except (AttributeError, TypeError, ValueError):
            continue
        if 1 <= index <= MAX_TEXTS:
            result.append({"index": index, "text": str(item.get("text") or "")})
    return result


def _unescape(separator):
    return separator.replace("\\n", "\n").replace("\\t", "\t")


class MykeeTextSwitchBatch:
    CATEGORY = "Mykee/Prompt"
    FUNCTION = "main"
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("text", "joined", "count")
    OUTPUT_IS_LIST = (True, False, False)
    OUTPUT_TOOLTIPS = (
        "switch: the selected text. batch: every ticked, non-empty text as a "
        "list - the nodes after it run once per text.",
        "The same texts in one string, joined with the separator.",
        "How many texts went out.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            f"text_{i}": (
                "STRING",
                {
                    "forceInput": True,
                    "lazy": True,
                    "tooltip": f"Optional. Replaces the typed text of field {i}; its radio / checkbox still applies.",
                },
            )
            for i in range(1, MAX_TEXTS + 1)
        }
        return {
            "required": {
                "batch_mode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "batch",
                        "label_off": "switch",
                        "tooltip": "switch: only the text picked with the radio button goes out. "
                        "batch: every ticked, non-empty text goes out, one after the other.",
                    },
                ),
                "text_count": (
                    "INT",
                    {"default": 2, "min": 0, "max": MAX_TEXTS, "tooltip": "Number of text fields."},
                ),
                "separator": (
                    "STRING",
                    {"default": "\\n", "tooltip": "Separator of the joined output (\\n = new line, \\t = tab)."},
                ),
                "texts_config": (
                    "STRING",
                    {"default": "{}", "hidden": True, "socketless": True},
                ),
            },
            "optional": optional,
        }

    def check_lazy_status(self, batch_mode=False, text_count=2, separator="\\n", texts_config="{}", **kwargs):
        # A connected input that has not been evaluated yet shows up as None.
        needed = []
        for item in _items(texts_config):
            key = f"text_{item['index']}"
            if item["index"] <= text_count and key in kwargs and kwargs[key] is None:
                needed.append(key)
        return needed

    def main(self, batch_mode=False, text_count=2, separator="\\n", texts_config="{}", **kwargs):
        texts = []
        for item in _items(texts_config):
            if item["index"] > text_count:
                continue
            key = f"text_{item['index']}"
            value = kwargs.get(key)
            text = str(value) if value is not None else item["text"]
            if batch_mode and not text.strip():
                continue  # batch: empty fields are skipped
            texts.append(text)
            if not batch_mode:
                break  # switch: exactly one

        if not texts:
            message = (
                "[Mykee Text Switch/Batch] No ticked, non-empty text - output blocked."
                if batch_mode
                else "[Mykee Text Switch/Batch] No text field is selected - output blocked."
            )
            print(message)
            if ExecutionBlocker is not None:
                return ([ExecutionBlocker(None)], ExecutionBlocker(None), 0)
            return ([], "", 0)

        return (texts, _unescape(separator).join(texts), len(texts))


NODE_CLASS_MAPPINGS = {
    "MykeeTextSwitchBatch": MykeeTextSwitchBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeTextSwitchBatch": "Mykee Text Switch/Batch",
}
