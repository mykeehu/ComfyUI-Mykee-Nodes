"""
ComfyUI-Mykee-Nodes / Group Bypasser

Frontend-only (virtual) node. It never runs on the Python side - all of
its behavior (listing the groups present in the current graph, drawing a
toggle per group, ordering the toggles, drag-to-reorder in manual mode,
and actually flipping each group's nodes between ALWAYS/BYPASS) lives in
the paired JS file: web/mykee_group_bypasser.js.

This stub exists only so ComfyUI can register the node type and show it
in the node-add menu. INPUT_TYPES/RETURN_TYPES are intentionally empty -
the JS side marks the node as `isVirtualNode`, so it is skipped entirely
when the workflow is converted into an API prompt, and this `noop`
function is never actually called.
"""


class MykeeGroupBypasser:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "Mykee/Utils"
    OUTPUT_NODE = True

    def noop(self):
        return {}


NODE_CLASS_MAPPINGS = {
    "MykeeGroupBypasser": MykeeGroupBypasser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MykeeGroupBypasser": "Mykee Group Bypasser",
}
