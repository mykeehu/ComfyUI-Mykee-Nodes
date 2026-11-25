/*
 * Mykee - KSampler with Sigmas - frontend.
 *
 * The Switch_to_sigmas toggle (bottom of the node) picks the schedule source:
 *   OFF -> steps / scheduler / denoise are active; sigmas_text and
 *          denoise_mode are greyed out;
 *   ON  -> steps / scheduler are greyed out; denoise and denoise_mode are
 *          active (denoise is applied to the schedule); sigmas_text is active
 *          unless the sigmas input is connected (a connected input wins).
 * The backend applies the same rules on its own; this is only the visual
 * feedback.
 *
 * A new node is fitted to its content. A loaded node keeps the size it was
 * saved with (also a height the user set by hand); it only grows if the
 * content would not fit.
 */
import { app } from "../../scripts/app.js";

const NODE_NAME = "MykeeKSamplerSigmas";
const SIGMAS_INPUT = "sigmas";
const SIGMAS_TEXT = "sigmas_text";
const SWITCH = "Switch_to_sigmas";
const SCHEDULE_WIDGETS = ["steps", "scheduler"];
const DENOISE_MODE = "denoise_mode";

function widget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

// Uses the graph's link table too: newer frontends only mirror links in
// input.link, and the mirror can lag behind.
function inputLinked(node, name) {
    const idx = node.inputs?.findIndex((inp) => inp.name === name) ?? -1;
    if (idx < 0) return false;
    if (node.inputs[idx].link != null) return true;
    const table = node.graph?.links;
    const links = table instanceof Map ? [...table.values()] : table ? Object.values(table) : [];
    return links.some((l) => l && String(l.target_id) === String(node.id) && l.target_slot === idx);
}

function setDisabled(w, disabled) {
    if (!w || !!w.disabled === disabled) return false;
    w.disabled = disabled;
    return true;
}

function syncScheduleWidgets(node) {
    const useSigmas = !!widget(node, SWITCH)?.value;
    const socket = inputLinked(node, SIGMAS_INPUT);
    let changed = setDisabled(widget(node, SIGMAS_TEXT), !useSigmas || socket);
    if (setDisabled(widget(node, DENOISE_MODE), !useSigmas)) changed = true;
    for (const name of SCHEDULE_WIDGETS) {
        if (setDisabled(widget(node, name), useSigmas)) changed = true;
    }
    if (changed) node.graph?.setDirtyCanvas?.(true, true);
}

// shrink = false only grows (used after a load, so a saved height is kept).
function fitHeight(node, shrink = true) {
    const size = node.computeSize?.();
    if (!size || !Number.isFinite(size[1])) return;
    const height = shrink ? size[1] : Math.max(node.size[1], size[1]);
    node.setSize([Math.max(node.size[0], size[0]), height]);
    node.graph?.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "Mykee.KSamplerSigmas",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated?.apply(this, arguments);
            const switchWidget = widget(this, SWITCH);
            if (switchWidget) {
                const callback = switchWidget.callback;
                switchWidget.callback = (...args) => {
                    const r = callback?.apply(switchWidget, args);
                    syncScheduleWidgets(this);
                    return r;
                };
            }
            syncScheduleWidgets(this);
            // ComfyUI may still apply its own size after onNodeCreated - fit
            // again, but only for a NEW node: a node loaded from a workflow
            // has been configured by now and keeps its saved size.
            setTimeout(() => {
                if (!this._mykeeConfigured) fitHeight(this, true);
            }, 0);
            return ret;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const ret = onConfigure?.apply(this, arguments);
            this._mykeeConfigured = true;
            // Let the links of the whole graph load first. The saved size is
            // kept; the node only grows if the content would not fit.
            setTimeout(() => {
                syncScheduleWidgets(this);
                fitHeight(this, false);
            }, 50);
            return ret;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const ret = onConnectionsChange?.apply(this, arguments);
            // The link table is updated after this callback on some frontends.
            setTimeout(() => syncScheduleWidgets(this), 0);
            return ret;
        };

        // Safety net (undo/redo, paste, subgraph changes): cheap check per redraw.
        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function () {
            const ret = onDrawForeground?.apply(this, arguments);
            syncScheduleWidgets(this);
            return ret;
        };
    },
});
