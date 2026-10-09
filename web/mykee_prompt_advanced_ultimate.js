import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const NODE_NAME = "MykeePromptAdvancedUltimate";
const DISPLAY_WIDGET_NAMES = ["response_1_display", "response_2_display", "tags_1_display"];

// The second stage's seed_control (fixed/increment/decrement/randomize)
// is computed in Python after each run and sent back through this same
// key, so the "second_seed" INT widget updates for the next run - same
// mechanism as the read-only display widgets above, and the same one
// MykeeCounter uses for its own "value" widget.
//
// NOTE: an earlier version of this file tried to replicate ComfyUI's
// built-in control_after_generate behavior for second_seed by
// dynamically adding an extra combo widget via node.addWidget() at
// node-creation time. That caused a real bug: ComfyUI saves widget
// values in the workflow JSON by POSITION, and a widget injected after
// the fact isn't accounted for in that position count, so every widget
// after it silently shifted by one slot on each reload. second_seed's
// own "fixed/increment/decrement/randomize" control is now a normal,
// natively-declared widget on the Python side instead (see
// mykee_prompt_advanced_ultimate.py's _sampling_widgets) - it's part of
// the node's widget list from the very start, so save/restore stays
// aligned. This file no longer adds or repositions any widgets, EXCEPT
// for the read-only display widgets below, which is a different,
// deliberate case: see the comment where they're created.
const SEED_SYNC_KEY = "second_seed";


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
    name: "Mykee.PromptAdvancedUltimate",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_NAME) return;

        // Create the read-only display widgets as soon as the node exists.
        // These are deliberately NOT declared as backend inputs in
        // mykee_prompt_advanced_ultimate.py's INPUT_TYPES (see the note
        // there) - a real declared input with w.serialize = false wasn't
        // reliable enough on its own to stop this node rerunning on every
        // single queue even with unchanged inputs, since its own previous
        // output fed back in as part of the next run's cache key. A
        // widget that was never a real backend input in the first place
        // sidesteps that entirely. Because these are appended at
        // node-creation time, always in the same fixed order, they don't
        // hit the position-shift issue described above for second_seed -
        // that issue is specifically about inserting a widget AFTER
        // creation, mid-way through the existing list.
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            for (const name of DISPLAY_WIDGET_NAMES) {
                const { widget: w } = ComfyWidgets["STRING"](
                    this, name, ["STRING", { multiline: true, default: "" }], app
                );
                w.serialize = false;
                if (w.inputEl) {
                    w.inputEl.readOnly = true;
                    w.inputEl.placeholder = "(filled in after running)";
                    w.inputEl.style.opacity = "0.75";
                }
            }

            return ret;
        };

        // After execution, the backend sends response_1/response_2/tags_1
        // and the next second_seed value back via the "ui" dict - write
        // them into the matching widgets.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const ret = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            // properties are restored by now; the widgets exist since onNodeCreated.
            restoreDisplay(this, DISPLAY_WIDGET_NAMES);
            return ret;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            for (const name of DISPLAY_WIDGET_NAMES) {
                const value = message?.[name]?.[0];
                if (value === undefined) continue;

                rememberDisplay(this, name, value);
                const w = this.widgets?.find((w) => w.name === name);
                if (!w) continue;

                w.value = value;
                if (w.inputEl) {
                    w.inputEl.value = value;
                    w.inputEl.readOnly = true;
                }
            }

            const nextSeed = message?.[SEED_SYNC_KEY]?.[0];
            if (nextSeed !== undefined) {
                const seedWidget = this.widgets?.find((w) => w.name === SEED_SYNC_KEY);
                if (seedWidget) {
                    seedWidget.value = nextSeed;
                    if (seedWidget.callback) seedWidget.callback(nextSeed);
                }
            }

            app.graph.setDirtyCanvas(true, true);
        };
    },
});
