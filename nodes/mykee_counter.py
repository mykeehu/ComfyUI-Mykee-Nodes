"""
ComfyUI-Mykee-Counter
A counter node for any number of arbitrarily-typed input signals.
"""

MAX_INPUTS = 50


class AnyType(str):
    """Wildcard type - reports itself as compatible with every ComfyUI type,
    so the input signal_N sockets accept an output of any type."""

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


ANY = AnyType("*")


class MykeeCounter:
    """
    Counter node.

    Behavior:
    - The 'value' field is the counter's current value. It can be
      overwritten by hand at any time before starting the queue - counting
      then continues from there.
    - 'padding' (1-10) sets the number of digits in the text (TEXT) output,
      e.g. padding=5 and value=1 gives the text "00001".
    - 'num_inputs' (1-50) controls how many signal_N inputs the node
      watches. The JS frontend shows/hides sockets based on this.
    - If ANY of the watched signal_N inputs is not None (i.e. it received
      some signal/data), the counter increases by exactly ONE - during
      execution it stops at the very first such input; any further
      signals in the same run no longer increase it again.
    - If none of the watched inputs carry a signal (not connected, or the
      source node is inactive/bypassed and so produces no output), this is
      not an error: the node simply doesn't advance the counter.
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, MAX_INPUTS + 1):
            optional[f"signal_{i}"] = (ANY, {"default": None})

        return {
            "required": {
                "value": ("INT", {
                    "default": 0,
                    "min": -1_000_000_000,
                    "max": 1_000_000_000,
                    "step": 1,
                    "tooltip": "The counter's current value. Can be overwritten by hand.",
                }),
                "padding": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of digits in the TEXT output (e.g. 5 -> 00001).",
                }),
                "num_inputs": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": MAX_INPUTS,
                    "step": 1,
                    "tooltip": "How many input signal sockets should be active/visible (1-50).",
                }),
            },
            "optional": optional,
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("INT", "FLOAT", "STRING")
    RETURN_NAMES = ("INT", "FLOAT", "TEXT")
    FUNCTION = "count"
    CATEGORY = "Mykee/Counter"
    OUTPUT_NODE = True

    def count(self, value, padding, num_inputs, unique_id=None, **kwargs):
        triggered = False
        for i in range(1, int(num_inputs) + 1):
            sig = kwargs.get(f"signal_{i}")
            if sig is not None:
                triggered = True
                break  # stops at the very first signal; the rest don't count in this run

        new_value = int(value) + 1 if triggered else int(value)
        text = str(new_value).zfill(int(padding))

        return {
            # send the updated value back to the frontend so the "value"
            # widget auto-updates for the next run
            "ui": {"value": [new_value]},
            "result": (new_value, float(new_value), text),
        }


# Last-known seed value per node (keyed by unique_id), so
# MykeeCounterSeedAdvanced can detect when the seed has changed.
_SEED_STATE = {}


class MykeeCounterSeedAdvanced:
    """
    An extended version of MykeeCounter with a dedicated 'seed' (INT) input.

    Behavior:
    - There's a separate, dedicated INT input/widget called 'seed' (not
      part of the signal_N wildcard inputs). This can be a manually typed
      number, or the output of a connected node (e.g. a seed generator) -
      right-click -> "Convert widget to input" to turn it into a socket if
      you need a connected seed.
    - If 'seed' differs from its value on the previous run: the counter's
      'value' resets to 0, then continues watching the signal_N inputs in
      the same run - if one of them carries a signal, it advances by one
      from there (from 0).
    - If 'seed' has NOT changed: everything proceeds exactly like
      MykeeCounter (advances from the current 'value' if there's a
      signal).
    - On the very first run (no previous seed data yet), this does not
      count as a "change" - it starts normally.
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, MAX_INPUTS + 1):
            optional[f"signal_{i}"] = (ANY, {"default": None})

        return {
            "required": {
                "seed": ("INT", {
                    "default": 0,
                    "min": -1_000_000_000,
                    "max": 1_000_000_000,
                    "step": 1,
                    "tooltip": "Dedicated seed input. If it changes from the previous run, the counter resets to zero.",
                }),
                "value": ("INT", {
                    "default": 0,
                    "min": -1_000_000_000,
                    "max": 1_000_000_000,
                    "step": 1,
                    "tooltip": "The counter's current value. Can be overwritten by hand.",
                }),
                "padding": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of digits in the TEXT output (e.g. 5 -> 00001).",
                }),
                "num_inputs": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": MAX_INPUTS,
                    "step": 1,
                    "tooltip": "How many input signal sockets should be active/visible (1-50).",
                }),
            },
            "optional": optional,
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("INT", "FLOAT", "STRING")
    RETURN_NAMES = ("INT", "FLOAT", "TEXT")
    FUNCTION = "count_seed"
    CATEGORY = "Mykee/Counter"
    OUTPUT_NODE = True

    def count_seed(self, seed, value, padding, num_inputs, unique_id=None, **kwargs):
        last_seed = _SEED_STATE.get(unique_id)
        seed_changed = last_seed is not None and seed != last_seed
        _SEED_STATE[unique_id] = seed

        base = 0 if seed_changed else int(value)

        triggered = False
        for i in range(1, int(num_inputs) + 1):
            sig = kwargs.get(f"signal_{i}")
            if sig is not None:
                triggered = True
                break  # stops at the very first signal; the rest don't count in this run

        new_value = base + 1 if triggered else base
        text = str(new_value).zfill(int(padding))

        return {
            "ui": {"value": [new_value]},
            "result": (new_value, float(new_value), text),
        }


NODE_CLASS_MAPPINGS = {
    "MykeeCounter": MykeeCounter,
    "MykeeCounterSeedAdvanced": MykeeCounterSeedAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeCounter": "Mykee Counter",
    "MykeeCounterSeedAdvanced": "Mykee Counter (Seed Advanced)",
}
