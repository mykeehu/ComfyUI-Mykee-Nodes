"""
ComfyUI-Mykee-Nodes / seed node

"Mykee Seed" is a reworked version of the "Seed" node from rgthree-comfy
(https://github.com/rgthree/rgthree-comfy, MIT License, Copyright (c) 2023
Regis Gaughan, III - see THIRD_PARTY_NOTICES.md). Differences:

- the random range (min_seed / max_seed) is a pair of regular widgets on
  the node instead of hidden node properties, and
- a seed_lock switch: when ON, a seed that was just generated for a run
  (seed_value was -1 / -2 / -3) is written straight back into seed_value,
  so it stays put for the following runs. When OFF, -1 behaves as it always
  did: a fresh random seed on every run, seed_value stays at -1, and
- a history box (history_size = 1..50 entries) listing the seeds of the
  most recent runs; clicking one puts it into seed_value. It lives purely
  in the frontend - history_size is the only part the server ever sees.

The actual seed generation happens in the browser (web/mykee_seed.js): it
rewrites the seed in the prompt right before it is sent to the server, so
the generated seed also ends up in the saved image metadata. This Python
side only passes the seed through - plus a server-side fallback for the
case where a special seed reaches the server anyway (e.g. an API call that
passes -1 directly, without the ComfyUI frontend).
"""

import random

SEED_LIMIT = 1125899906842624  # 2**50 - well inside JS's exact-integer range
SPECIAL_SEEDS = (-1, -2, -3)  # random / increment / decrement (see web/mykee_seed.js)
SEED_WIDGET = "seed_value"

# SystemRandom is unaffected by other extensions calling random.seed() on
# the global generator (which would otherwise make "random" seeds repeat).
_rng = random.SystemRandom()


def _random_seed(min_seed, max_seed):
    lo = max(-SEED_LIMIT, min(SEED_LIMIT, int(min_seed)))
    hi = max(-SEED_LIMIT, min(SEED_LIMIT, int(max_seed)))
    if lo > hi:
        lo, hi = hi, lo
    for _ in range(8):
        seed = _rng.randint(lo, hi)
        if seed not in SPECIAL_SEEDS:
            return seed
    return 0  # only reachable when the range consists of special seeds alone


class MykeeSeed:
    CATEGORY = "Mykee/Utils"
    FUNCTION = "main"
    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("seed", "seed_text")

    @classmethod
    def INPUT_TYPES(cls):
        # NOTE: the seed widget is deliberately NOT called "seed": ComfyUI
        # attaches its own "control after generate" combo to widgets with
        # that exact name, which would fight with this node's own logic.
        return {
            "required": {
                SEED_WIDGET: (
                    "INT",
                    {
                        "default": 0,
                        "min": -SEED_LIMIT,
                        "max": SEED_LIMIT,
                        "tooltip": (
                            "-1 = new random seed on every run (from the "
                            "min_seed..max_seed range). Any other value is "
                            "used as is. -2 / -3 = last used seed + 1 / - 1."
                        ),
                    },
                ),
                "min_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": -SEED_LIMIT,
                        "max": SEED_LIMIT,
                        "tooltip": "Lower bound (inclusive) of the random seed range.",
                    },
                ),
                "max_seed": (
                    "INT",
                    {
                        "default": SEED_LIMIT,
                        "min": -SEED_LIMIT,
                        "max": SEED_LIMIT,
                        "tooltip": "Upper bound (inclusive) of the random seed range.",
                    },
                ),
                "seed_lock": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "locked",
                        "label_off": "unlocked",
                        "tooltip": (
                            "ON: a newly generated seed is written into "
                            "seed_value right away, so the following runs "
                            "keep using it. OFF: -1 generates a new seed on "
                            "every run."
                        ),
                    },
                ),
                "history_size": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 50,
                        "tooltip": (
                            "How many of the most recently used seeds are "
                            "listed in the history box below (click one to "
                            "put it into seed_value)."
                        ),
                    },
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def IS_CHANGED(cls, seed_value=0, **kwargs):
        # A special seed arriving here means "generate a new one" - force a
        # re-run by returning something different every time.
        if seed_value in SPECIAL_SEEDS:
            return _rng.random()
        return seed_value

    def main(
        self,
        seed_value=0,
        min_seed=0,
        max_seed=SEED_LIMIT,
        seed_lock=False,
        history_size=5,
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
    ):
        seed = int(seed_value)

        if seed in SPECIAL_SEEDS:
            # Normally the frontend already replaced these before queueing.
            # Reaching this point means the prompt came from somewhere else
            # (API call, other frontend). Generate one here, and write it
            # back into the prompt / workflow so it lands in the image
            # metadata like a frontend-generated seed would.
            original = seed
            seed = _random_seed(min_seed, max_seed)
            print(
                f"[Mykee Seed] Got {original} on the server, generated seed {seed} "
                f"(seed_lock only works from the ComfyUI frontend)."
            )
            self._write_back(seed, prompt, extra_pnginfo, unique_id)

        # seed_text: the same seed as a string, e.g. for file names, so it
        # does not have to be converted to text separately.
        return (seed, str(seed))

    @staticmethod
    def _write_back(seed, prompt, extra_pnginfo, unique_id):
        if unique_id is None:
            return
        uid = str(unique_id)

        prompt_node = (prompt or {}).get(uid)
        if isinstance(prompt_node, dict) and isinstance(prompt_node.get("inputs"), dict):
            prompt_node["inputs"][SEED_WIDGET] = seed

        workflow = (extra_pnginfo or {}).get("workflow") or {}
        for wf_node in workflow.get("nodes", []):
            if str(wf_node.get("id")) == uid and wf_node.get("widgets_values"):
                wf_node["widgets_values"][0] = seed  # seed_value is the first widget
                break


NODE_CLASS_MAPPINGS = {
    "MykeeSeed": MykeeSeed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeSeed": "Mykee Seed",
}
