/*
 * Mykee Before/After Text Injection - frontend.
 *
 * The backend declares 50 text_N inputs and 50 text_N outputs; this keeps
 * only the first text_count pairs on the node. Outputs are only ever removed
 * / added at the end, so the remaining output slots keep the indices the
 * backend's RETURN_TYPES expect.
 *
 * The node's height follows its content (ComfyUI sizes a new node while all
 * 50 pairs still exist).
 */
import { app } from "../../scripts/app.js";

const NODE_NAME = "MykeeBeforeAfterTextInjection";
const MAX_TEXTS = 50;
const NAME_RE = /^text_(\d+)$/;
// These widgets must not have input sockets: see removeWidgetSockets().
const WIDGET_NAMES = new Set(["text_count", "before_text", "before_separator", "after_text", "after_separator"]);

function textCount(node) {
    const n = Math.round(Number(node.widgets?.find((w) => w.name === "text_count")?.value ?? 2));
    return Number.isFinite(n) ? Math.max(1, Math.min(MAX_TEXTS, n)) : 2;
}

function slotIndex(slot) {
    const match = slot?.name?.match(NAME_RE);
    return match ? Number(match[1]) : null;
}

// Bypass: ComfyUI passes a bypassed node's output N through from input N
// (by slot index). text_1..text_N therefore have to be input slots 0..N-1,
// i.e. the widgets may not have sockets in front of / between them. The
// backend marks them socketless; this also removes the sockets of older
// saved workflows / frontends that ignore that flag. A socket that has a
// link is kept (removing it would drop the link) - then bypass can't pass
// the texts through correctly, which is reported in the console.
function removeWidgetSockets(node) {
    for (let idx = (node.inputs?.length ?? 0) - 1; idx >= 0; idx--) {
        const input = node.inputs[idx];
        if (!WIDGET_NAMES.has(input.widget?.name ?? input.name)) continue;
        if (input.link != null) {
            console.warn(
                `[Mykee Before/After Text Injection] "${input.name}" is connected to a link, so its socket stays - ` +
                    "bypassing this node will not pass the texts through correctly. Disconnect it to fix that."
            );
            continue;
        }
        node.removeInput(idx);
    }
}

function syncSlots(node, shrink = true) {
    const count = textCount(node);
    removeWidgetSockets(node);

    for (let idx = (node.inputs?.length ?? 0) - 1; idx >= 0; idx--) {
        const n = slotIndex(node.inputs[idx]);
        if (n !== null && n > count) node.removeInput(idx);
    }
    const inputs = new Set((node.inputs ?? []).map(slotIndex).filter((n) => n !== null));
    for (let i = 1; i <= count; i++) {
        if (!inputs.has(i)) node.addInput(`text_${i}`, "STRING");
    }

    // Outputs: trim / extend at the end only.
    while ((node.outputs?.length ?? 0) > count) node.removeOutput(node.outputs.length - 1);
    while ((node.outputs?.length ?? 0) < count) node.addOutput(`text_${node.outputs.length + 1}`, "STRING");

    fitHeight(node, shrink);
}

// Fits the node's height to its content; the width stays the user's.
// shrink = false only grows (a loaded workflow keeps a node the user made
// taller for bigger before / after text boxes).
function fitHeight(node, shrink = true) {
    const size = node.computeSize?.();
    if (!size || !Number.isFinite(size[1])) return;
    const height = shrink ? size[1] : Math.max(node.size[1], size[1]);
    node.setSize([Math.max(node.size[0], size[0]), height]);
    node.graph?.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "Mykee.BeforeAfterTextInjection",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated?.apply(this, arguments);
            const countWidget = this.widgets?.find((w) => w.name === "text_count");
            if (countWidget) {
                const callback = countWidget.callback;
                countWidget.callback = (...args) => {
                    const r = callback?.apply(countWidget, args);
                    syncSlots(this);
                    return r;
                };
            }
            syncSlots(this);
            // ComfyUI may still apply its own size after onNodeCreated.
            setTimeout(() => fitHeight(this), 0);
            return ret;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const ret = onConfigure?.apply(this, arguments);
            setTimeout(() => syncSlots(this, false), 50);
            return ret;
        };
    },
});
