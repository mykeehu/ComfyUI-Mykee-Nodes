"""
ComfyUI-Mykee-Nodes / conditioning nodes
"""

try:
    from comfy_execution.graph import ExecutionBlocker
except ImportError:
    # Older ComfyUI without this module - the switch node below falls
    # back to raising a plain RuntimeError for its error cases.
    ExecutionBlocker = None


class MykeeConditioningSwitch:
    """
    A rework of Crystian's "Switch conditioning" node from ComfyUI-Crystools
    (https://github.com/crystian/comfyui-crystools), with more forgiving
    logic for the case where only one of the two inputs is actually wired
    up in the graph.

    The original node treats on_true/on_false as required inputs and
    always honors `boolean` literally, whether or not the input it points
    to is actually connected - so leaving one branch unconnected (e.g.
    while iterating on a graph) is a hard error even though the intent
    ("just use whichever one is there") is obvious.

    This version makes on_true/on_false optional and adds a `switch_is_active`
    widget (same naming/reasoning as MykeeImageSwitch's own widget
    elsewhere in this pack):

    - switch_is_active ON  -> normal switch mode: `boolean` picks between
      on_true/on_false when both are active.
    - switch_is_active OFF -> "bypass" mode (the switch itself is treated as
      inactive): `boolean` is ignored entirely.

    In BOTH modes, whenever exactly one of on_true/on_false is active,
    that one is passed through regardless of `switch_is_active`/`boolean` -
    there's nothing to decide, so nothing is asked of you. The only
    two situations that raise an error are the genuinely ambiguous
    ones:

    - both on_true and on_false are active AND switch_is_active is OFF
      (bypass/inactive) - there's no `boolean` decision being made in
      that mode, so which one should go through is undefined.
    - neither on_true nor on_false is active - nothing to pass through.

    Both error cases block the `conditioning` output via ComfyUI's
    ExecutionBlocker (so a mistake here doesn't crash unrelated parts of
    the workflow) rather than raising a hard traceback, matching
    MykeeImageSwitch's convention; older ComfyUI versions without
    ExecutionBlocker fall back to a plain RuntimeError.

    IMPORTANT: as with MykeeImageSwitch, "disabling the switch" here
    means turning switch_is_active OFF, NOT using ComfyUI's own node
    Bypass/Mute (right-click -> Mode) on this node itself. A natively
    bypassed node's Python code never runs at all - ComfyUI's core
    engine handles that passthrough itself, purely by matching
    input/output slots positionally, which can't be customized from
    node code and would just reproduce the original node's rigidity
    one level up.

    Credit: switch logic/naming (on_true, on_false, boolean) is based on
    Crystian's original "Switch conditioning [Crystools]" node -
    https://github.com/crystian/comfyui-crystools
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch_is_active": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "ON: normal switch mode - `boolean` picks between "
                        "on_true/on_false when both are active. OFF: "
                        "\"bypass\" mode (the switch is treated as "
                        "inactive) - `boolean` is ignored; if both inputs "
                        "are active at the same time this is ambiguous and "
                        "raises an error. Either way, if only one input is "
                        "active it always passes through regardless of "
                        "this setting. Right-click -> 'Convert widget to "
                        "input' to drive this from a BOOLEAN output. Do "
                        "NOT use ComfyUI's own node Bypass/Mute on this "
                        "node itself to disable it - use this toggle "
                        "instead (same reasoning as MykeeImageSwitch)."
                    ),
                }),
                "boolean": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Used only when switch_is_active is ON and both "
                        "on_true and on_false are active: True -> "
                        "on_true, False -> on_false. Ignored when only "
                        "one input is active, or when switch_is_active is OFF."
                    ),
                }),
            },
            "optional": {
                "on_true": ("CONDITIONING",),
                "on_false": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "info")
    FUNCTION = "switch"
    CATEGORY = "Mykee/Conditioning"

    def switch(self, switch_is_active, boolean, on_true=None, on_false=None):
        has_true = on_true is not None
        has_false = on_false is not None

        if has_true and not has_false:
            return (on_true, "only on_true is active -> passthrough on_true")

        if has_false and not has_true:
            return (on_false, "only on_false is active -> passthrough on_false")

        if not has_true and not has_false:
            msg = (
                "Mykee Switch Conditioning: neither on_true nor on_false "
                "is active (both disconnected, or their upstream node is "
                "bypassed/muted). Blocking downstream nodes that need the "
                "conditioning output."
            )
            if ExecutionBlocker is not None:
                return (ExecutionBlocker(msg), msg)
            raise RuntimeError(msg)

        # Both are active from here on.
        if not switch_is_active:
            msg = (
                "Mykee Switch Conditioning: both on_true and on_false are "
                "active, but switch_is_active is OFF (bypass/inactive), so the "
                "`boolean` selector is not in effect and it's ambiguous "
                "which one should be used. Turn switch_is_active ON and pick "
                "one with `boolean`, or disconnect one of the inputs."
            )
            if ExecutionBlocker is not None:
                return (ExecutionBlocker(msg), msg)
            raise RuntimeError(msg)

        if boolean:
            return (on_true, "switch ON, both active -> boolean=True -> using on_true")
        return (on_false, "switch ON, both active -> boolean=False -> using on_false")


NODE_CLASS_MAPPINGS = {
    "MykeeConditioningSwitch": MykeeConditioningSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeConditioningSwitch": "Mykee Switch Conditioning",
}
