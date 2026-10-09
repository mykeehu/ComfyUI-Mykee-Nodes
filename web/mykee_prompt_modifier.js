import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const NODE_NAME = "MykeePromptModifier";
const DISPLAY_NAME = "response_display";


// The display widgets are client-side only (serialize = false, so they never
// reach the cache key), which also means ComfyUI doesn't save their text in
// the workflow - switching to another workflow tab and back rebuilds the
// node with empty boxes. The last result is therefore also kept in
// node.properties: that is saved with the workflow, but it is not a widget
// (no widgets_values position shift) and not a backend input (no re-run).
const STORE_KEY = "mykee_display";

function rememberDisplay(node, name, value) {
    node.properties = node.properties || {};
    const store = node.properties[STORE_KEY] && typeof node.properties[STORE_KEY] === "object" ? node.properties[STORE_KEY] : {};
    store[name] = value;
    node.properties[STORE_KEY] = store;
}

function restoreDisplay(node, names) {
    const store = node.properties?.[STORE_KEY];
    if (!store || typeof store !== "object") return;
    for (const name of names) {
        const value = store[name];
        if (typeof value !== "string") continue;
        const w = node.widgets?.find((w) => w.name === name);
        if (!w) continue;
        w.value = value;
        if (w.inputEl) w.inputEl.value = value;
    }
}

app.registerExtension({
    name: "Mykee.PromptModifier",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            // response_display is intentionally NOT declared as a backend
            // input in mykee_prompt_modifier.py (see the note there) - it's
            // created here, purely client-side, exactly like ComfyUI's own
            // PreviewImage node has no "images" input yet still shows a
            // widget. Setting w.serialize = false on a widget that DOES
            // exist server-side turned out not to be a reliable enough fix
            // on its own (still forced a rerun every queue on this
            // person's setup); a widget that was never a real input in the
            // first place sidesteps the question entirely - there's
            // nothing for the cache key to ever pick up.
            const { widget: w } = ComfyWidgets["STRING"](
                this, DISPLAY_NAME, ["STRING", { multiline: true, default: "" }], app
            );
            w.serialize = false;
            if (w.inputEl) {
                w.inputEl.readOnly = true;
                w.inputEl.placeholder = "(filled in after running)";
                w.inputEl.style.opacity = "0.75";
            }

            return ret;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const ret = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            // properties are restored by now; the widgets exist since onNodeCreated.
            restoreDisplay(this, [DISPLAY_NAME]);
            return ret;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const value = message?.[DISPLAY_NAME]?.[0];
            if (value === undefined) return;
            rememberDisplay(this, DISPLAY_NAME, value);
            const w = this.widgets?.find((w) => w.name === DISPLAY_NAME);
            if (w) {
                w.value = value;
                if (w.inputEl) {
                    w.inputEl.value = value;
                    w.inputEl.readOnly = true;
                }
            }
            app.graph.setDirtyCanvas(true, true);
        };
    },
});
