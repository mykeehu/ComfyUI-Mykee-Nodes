"""
ComfyUI-Mykee-Nodes / mask nodes
"""


class MykeeInvertMaskToggle:
    """
    An enhanced version of the base 'InvertMask' node with an 'enabled'
    (On/Off) toggle.

    - enabled = True  -> the mask is passed through inverted (1.0 - mask),
      the usual InvertMask behavior.
    - enabled = False -> the input mask is passed through unchanged.

    'enabled' is a plain BOOLEAN widget that can be turned into a socket
    via right-click ("Convert widget to input"), so it can be driven by
    another (sub)graph's BOOLEAN output too, not just toggled by hand.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "If enabled, the mask is inverted. "
                        "If disabled, the input mask is passed through unchanged. "
                        "Right-click -> 'Convert widget to input' -> can also be "
                        "driven from another (sub)graph's BOOLEAN output."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASK",)
    FUNCTION = "invert"
    CATEGORY = "Mykee/Mask"

    def invert(self, mask, enabled):
        if enabled:
            return (1.0 - mask,)
        return (mask,)


NODE_CLASS_MAPPINGS = {
    "MykeeInvertMaskToggle": MykeeInvertMaskToggle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeInvertMaskToggle": "Mykee Invert Mask (Toggle)",
}
