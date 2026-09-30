"""
ComfyUI-Mykee-Nodes / image nodes

MykeeColorBackground:
    Composites a foreground IMAGE (optionally with a MASK acting as its
    alpha channel) onto a solid color background - the same idea as
    comfy_mtb's "Colored Image" node (https://github.com/melMass/comfy_mtb),
    but with a proper top-level color widget: a hex text field with a live
    swatch/native picker (see web/mykee_color_background.js), instead of
    comfy_mtb's old dedicated custom-button color widget. Also supports
    growing the canvas beyond the foreground's own size (per-side pixel
    padding), filled with the same background color.
"""

import torch

try:
    from comfy_execution.graph import ExecutionBlocker
except ImportError:
    # Older ComfyUI without this module - the switch nodes below fall back
    # to raising a plain RuntimeError for the "nothing is active" case.
    ExecutionBlocker = None


def hex_to_rgb01(hex_str, fallback=(1.0, 1.0, 1.0)):
    """'#RRGGBB' / 'RRGGBB' / '#RGB' -> (r, g, b) floats in [0, 1]. Falls
    back to `fallback` for anything that isn't a valid hex color (e.g. an
    empty/partial value while it's being typed, or a disconnected input)."""
    if not isinstance(hex_str, str):
        return fallback
    s = hex_str.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        return fallback
    try:
        r = int(s[0:2], 16) / 255.0
        g = int(s[2:4], 16) / 255.0
        b = int(s[4:6], 16) / 255.0
        return (r, g, b)
    except ValueError:
        return fallback


def rgb01_to_hex(rgb):
    r, g, b = (max(0, min(255, round(c * 255))) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


class MykeeColorBackground:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "foreground_image": ("IMAGE",),
                "color": ("STRING", {
                    "default": "#FFFFFF",
                    "tooltip": (
                        "Background color as a hex code (e.g. #FF8800). Type "
                        "it directly, pick it with the swatch below the "
                        "field, or right-click -> 'Convert widget to input' "
                        "to drive it from a STRING output."
                    ),
                }),
                "mask_invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Invert foreground_mask before it's used as the "
                        "alpha channel for the composite. Affects the "
                        "composited output image only - the mask input "
                        "itself is left untouched."
                    ),
                }),
                "mask_opacity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "Scales the mask's alpha before compositing. 1.0 = "
                        "use the mask as-is; 0.0 = fully transparent (the "
                        "foreground won't show through at all, only the "
                        "background color will)."
                    ),
                }),
                "canvas_expand_top": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 1,
                    "tooltip": "Grow the canvas above the foreground image by this many pixels, filled with the background color.",
                }),
                "canvas_expand_bottom": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 1,
                    "tooltip": "Grow the canvas below the foreground image by this many pixels, filled with the background color.",
                }),
                "canvas_expand_left": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 1,
                    "tooltip": "Grow the canvas to the left of the foreground image by this many pixels, filled with the background color.",
                }),
                "canvas_expand_right": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 1,
                    "tooltip": "Grow the canvas to the right of the foreground image by this many pixels, filled with the background color.",
                }),
            },
            "optional": {
                "foreground_mask": ("MASK", {
                    "tooltip": (
                        "Alpha channel for foreground_image. If not "
                        "connected, the foreground is treated as fully "
                        "opaque (only the canvas_expand_* padding, if any, "
                        "will show the background color)."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "color_info")
    FUNCTION = "composite"
    CATEGORY = "Mykee/Image"

    def composite(self, foreground_image, color, mask_invert, mask_opacity,
                  canvas_expand_top, canvas_expand_bottom,
                  canvas_expand_left, canvas_expand_right,
                  foreground_mask=None):
        if foreground_image.dim() != 4 or foreground_image.shape[-1] < 3:
            raise ValueError(
                "foreground_image must be a batch of IMAGE tensors [B,H,W,C>=3]."
            )

        device = foreground_image.device
        dtype = foreground_image.dtype
        fg = foreground_image[..., :3]
        batch, h, w, _ = fg.shape

        rgb = hex_to_rgb01(color)
        hex_out = rgb01_to_hex(rgb)
        bg_color = torch.tensor(rgb, device=device, dtype=dtype)

        # Alpha mask at the foreground's own resolution/batch size.
        if foreground_mask is not None:
            mask = foreground_mask.to(device=device, dtype=dtype)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            if mask.shape[0] != batch:
                if mask.shape[0] == 1:
                    mask = mask.expand(batch, -1, -1)
                else:
                    reps = (batch + mask.shape[0] - 1) // mask.shape[0]
                    mask = mask.repeat(reps, 1, 1)[:batch]
            if mask.shape[1] != h or mask.shape[2] != w:
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(1), size=(h, w), mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
        else:
            mask = torch.ones((batch, h, w), device=device, dtype=dtype)

        if mask_invert:
            mask = 1.0 - mask
        mask = (mask * mask_opacity).clamp(0.0, 1.0)

        # Composite fg over a same-size background patch first, then place
        # that patch onto the (optionally larger) canvas.
        alpha = mask.unsqueeze(-1)
        patch = fg * alpha + bg_color.view(1, 1, 1, 3) * (1.0 - alpha)

        top, bottom = canvas_expand_top, canvas_expand_bottom
        left, right = canvas_expand_left, canvas_expand_right
        if top or bottom or left or right:
            out_h, out_w = h + top + bottom, w + left + right
            canvas = bg_color.view(1, 1, 1, 3).expand(batch, out_h, out_w, 3).clone()
            canvas[:, top:top + h, left:left + w, :] = patch
            out = canvas
        else:
            out = patch

        out = out.clamp(0.0, 1.0)
        return (out, hex_out)


class MykeeImageSwitch:
    """
    A 2-input image switch with real passthrough for the "one branch is
    inactive" case - unlike most switch nodes, whose single output only
    ever maps to input 1 when something upstream is disabled.

    IMPORTANT: "disabling the switch" here means turning OFF the
    switch_mode widget below, NOT using ComfyUI's own node Bypass/Mute
    (right-click -> Mode). A bypassed node's Python code never runs at
    all - ComfyUI's core engine handles bypass passthrough itself,
    purely by matching input/output slots positionally, and that can't
    be customized from node code. That's exactly the "no passthrough"
    problem this node exists to avoid, so bypassing this node via the
    canvas Mode menu would just reproduce the same problem one level up.
    Use switch_mode instead (same reasoning as MykeeInvertMaskToggle's
    own "enabled" widget elsewhere in this pack).

    image_1 / image_2 are OPTIONAL inputs. If the node feeding one of
    them is itself bypassed/muted, ComfyUI's own passthrough already
    resolves that connected-but-inactive branch to nothing, so this node
    simply sees that input as not present - no special detection needed
    here, it falls out of image_1/image_2 being None.

    - switch_mode ON,  both active   -> `selected` picks 1 or 2.
    - switch_mode ON,  one active    -> that one, regardless of `selected`.
    - switch_mode ON,  neither active -> the image output is blocked (via
      ComfyUI's ExecutionBlocker) for anything downstream that needs it -
      no hard crash, other independent parts of the workflow keep running.
    - switch_mode OFF (batch mode), both active -> both images are passed
      through as a real ComfyUI *list* (OUTPUT_IS_LIST), not merged into
      one stacked tensor - so each keeps its own original width/height.
      Downstream nodes (Color Background, Save Image, ...) automatically
      run once per image at that image's own size, exactly like a batch
      "one after another", without any forced resizing. The trade-off:
      a node that specifically wants one true multi-frame batch tensor
      for batched inference will instead be run twice, once per image.
    - switch_mode OFF, one active    -> that one, unbatched.
    - switch_mode OFF, neither active -> same blocked-output behavior as above.
    """

    OUTPUT_IS_LIST = (True, False)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch_mode": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "ON (switch mode): selected picks between "
                        "image_1/image_2 (or auto-passes through whichever "
                        "one is active, if only one is). OFF (batch mode): "
                        "both active inputs are passed through one after "
                        "another, each at its own original size (a real "
                        "ComfyUI list, not a merged/resized tensor batch); "
                        "if only one is active, it passes through alone. "
                        "Right-click -> 'Convert widget to input' to drive "
                        "this from a BOOLEAN output. Do NOT use ComfyUI's "
                        "own node Bypass/Mute on this node itself to "
                        "'disable' it - that skips this node's code "
                        "entirely and loses the passthrough behavior; use "
                        "this toggle."
                    ),
                }),
                "selected": ("INT", {
                    "default": 1, "min": 1, "max": 2, "step": 1,
                    "tooltip": (
                        "Which input to use when switch_mode is ON and "
                        "both image_1 and image_2 are active. Ignored "
                        "otherwise (auto-passthrough / batch mode take over)."
                    ),
                }),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "switch"
    CATEGORY = "Mykee/Image"

    def switch(self, switch_mode, selected, image_1=None, image_2=None):
        has1 = image_1 is not None
        has2 = image_2 is not None

        if not has1 and not has2:
            msg = (
                "Mykee Image Switch: neither image_1 nor image_2 is active "
                "(both disconnected, or their upstream node is "
                "bypassed/muted). Blocking downstream nodes that need the "
                "image output."
            )
            if ExecutionBlocker is not None:
                return ([ExecutionBlocker(msg)], msg)
            raise RuntimeError(msg)

        if switch_mode:
            if has1 and has2:
                if selected == 1:
                    return ([image_1], "switch ON, both active -> using input 1")
                return ([image_2], "switch ON, both active -> using input 2")
            if has1:
                return ([image_1], "switch ON, only input 1 active -> passthrough input 1")
            return ([image_2], "switch ON, only input 2 active -> passthrough input 2")

        # switch_mode == False -> batch mode: pass through as a real
        # ComfyUI list (see OUTPUT_IS_LIST above) so each image keeps its
        # own original size instead of being resized to match the other.
        if has1 and has2:
            return (
                [image_1, image_2],
                "batch mode, both active -> passing through 2 images (own sizes kept)",
            )
        if has1:
            return ([image_1], "batch mode, only input 1 active -> passthrough input 1")
        return ([image_2], "batch mode, only input 2 active -> passthrough input 2")


MAX_SWITCH_INPUTS = 50


class MykeeImageSwitchAdvanced:
    """
    Same idea as MykeeImageSwitch, but with any number of image inputs
    instead of a fixed 2. New sockets appear automatically: connecting
    something to the currently-last image_N input adds an empty
    image_(N+1) right after it (handled client-side, see
    web/mykee_image_switch_advanced.js). Disconnecting trims trailing
    empty inputs back down to a single spare (never below image_1/2,
    never touching a socket that's still connected).

    Python itself always declares a fixed, generous cap
    (MAX_SWITCH_INPUTS image_N optional inputs) since ComfyUI's node
    schema can't truly be infinite - the JS side just controls how many
    of those are actually visible as sockets on the node at any time.

    IMPORTANT: same caveat as the base switch - "disabling the switch"
    means turning OFF switch_mode below, NOT ComfyUI's own node
    Bypass/Mute (right-click -> Mode), which skips this node's code
    entirely and can't reproduce this behavior.

    - switch_mode ON,  exactly one input active -> that one, regardless
      of `selected` (auto-passthrough).
    - switch_mode ON,  2+ inputs active -> `selected` must be one of the
      active input numbers, or this raises a real error (no guessing) -
      this is a deliberate mistake on the user's part, unlike the "none
      active" case below, so it stays a hard error.
    - switch_mode ON,  none active -> the image output is blocked (via
      ComfyUI's ExecutionBlocker) for anything downstream that needs it -
      no hard crash, other independent parts of the workflow keep running.
    - switch_mode OFF (batch mode), 2+ inputs active -> all of them are
      passed through as a real ComfyUI *list* (OUTPUT_IS_LIST), in
      image_1, image_2, ... order - not merged into one stacked tensor,
      so each keeps its own original width/height. Downstream nodes
      (Color Background, Save Image, ...) automatically run once per
      image at that image's own size, without any forced resizing. The
      trade-off: a node that specifically wants one true multi-frame
      batch tensor for batched inference will instead be run once per
      image.
    - switch_mode OFF, exactly one active -> that one, unbatched.
    - switch_mode OFF, none active -> same blocked-output behavior as above.
    """

    OUTPUT_IS_LIST = (True, False)

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "switch_mode": ("BOOLEAN", {
                "default": True,
                "tooltip": (
                    "ON (switch mode): selected picks which active input "
                    "to use (or auto-passes through the only active one, "
                    "if just one is connected). OFF (batch mode): every "
                    "active input is passed through one after another, "
                    "each at its own original size (a real ComfyUI list, "
                    "not a merged/resized tensor batch), in image_1, "
                    "image_2, ... order; if only one is active, it passes "
                    "through alone. Right-click -> 'Convert widget to "
                    "input' to drive this from a BOOLEAN output. Do NOT "
                    "use ComfyUI's own node Bypass/Mute on this node "
                    "itself to 'disable' it - that skips this node's code "
                    "entirely and loses the passthrough behavior; use "
                    "this toggle."
                ),
            }),
            "selected": ("INT", {
                "default": 1, "min": 1, "max": MAX_SWITCH_INPUTS, "step": 1,
                "tooltip": (
                    "Which input number to use when switch_mode is ON and "
                    "more than one input is active. Must be one of the "
                    "currently active input numbers. Ignored when only one "
                    "input is active, or when switch_mode is OFF (batch "
                    "mode)."
                ),
            }),
        }
        optional = {
            f"image_{i}": ("IMAGE",) for i in range(1, MAX_SWITCH_INPUTS + 1)
        }
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "switch"
    CATEGORY = "Mykee/Image"

    def switch(self, switch_mode, selected, **kwargs):
        active = [
            (i, kwargs[f"image_{i}"])
            for i in range(1, MAX_SWITCH_INPUTS + 1)
            if kwargs.get(f"image_{i}") is not None
        ]

        if not active:
            msg = (
                "Mykee Image Switch (Advanced): no image input is active "
                "(all disconnected, or their upstream nodes are "
                "bypassed/muted). Blocking downstream nodes that need the "
                "image output."
            )
            if ExecutionBlocker is not None:
                return ([ExecutionBlocker(msg)], msg)
            raise RuntimeError(msg)

        if switch_mode:
            if len(active) == 1:
                idx, img = active[0]
                return ([img], f"switch ON, only input {idx} active -> passthrough input {idx}")
            match = next((img for (idx, img) in active if idx == selected), None)
            if match is None:
                active_list = ", ".join(str(idx) for idx, _ in active)
                raise RuntimeError(
                    f"Mykee Image Switch (Advanced): selected={selected} "
                    f"is not one of the currently active inputs "
                    f"({active_list}). Pick one of those, or leave only "
                    f"one input connected for automatic passthrough."
                )
            return ([match], f"switch ON, using selected input {selected}")

        # switch_mode == False -> batch mode: pass through as a real
        # ComfyUI list (see OUTPUT_IS_LIST above) so each image keeps its
        # own original size instead of being resized to match input 1.
        if len(active) == 1:
            idx, img = active[0]
            return ([img], f"batch mode, only input {idx} active -> passthrough input {idx}")
        active_list = "+".join(str(idx) for idx, _ in active)
        return (
            [img for (_, img) in active],
            f"batch mode, passing through {len(active)} images (own sizes kept): inputs {active_list}",
        )


NODE_CLASS_MAPPINGS = {
    "MykeeColorBackground": MykeeColorBackground,
    "MykeeImageSwitch": MykeeImageSwitch,
    "MykeeImageSwitchAdvanced": MykeeImageSwitchAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeColorBackground": "Mykee Color Background",
    "MykeeImageSwitch": "Mykee Image Switch",
    "MykeeImageSwitchAdvanced": "Mykee Image Switch (Advanced)",
}
