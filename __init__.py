import importlib

# Every node module lives in ./nodes/. To add a new node file, drop it into
# that folder and add its module name here (order = registration order; on a
# duplicate node ID the later module wins, same as before the reorganisation).
_NODE_MODULES = [
    "mykee_counter",
    "mykee_mask",
    "mykee_character",
    "mykee_group_bypasser",
    "mykee_audio",
    "mykee_prosody",
    "mykee_voice_match",
    "mykee_voice_cleanup",
    "mykee_dereverb",
    "mykee_room_reducer",
    "mykee_stylegan",
    "mykee_image",
    "mykee_stripe_remover",
    "mykee_prompt_clarity",
    "mykee_prompt_template",
    "mykee_prompt_modifier",
    "mykee_prompt_advanced_ultimate",
    "mykee_conditioning",
    "mykee_seed",
    "mykee_xyz_plot",
    "mykee_text_switch",
    "mykee_text_injection",
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _module_name in _NODE_MODULES:
    _module = importlib.import_module(f".nodes.{_module_name}", __name__)
    NODE_CLASS_MAPPINGS.update(_module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_module.NODE_DISPLAY_NAME_MAPPINGS)

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
