/*
 * Mykee Video Info - frontend.
 *
 * A box at the bottom of the node lists the size, frame rate, codec, format
 * and the outgoing file name of the connected video. The values come from the
 * last run (the browser cannot look into the video); they are kept in
 * node.properties, so they survive a workflow-tab switch and a reload. The
 * file name row follows the filename_tag field live while typing.
 *
 * The box is a client-side only widget (never serialized, not a backend
 * input), so it cannot change the node's cache key.
 */
import { app } from "../../scripts/app.js";

const NODE_NAME = "MykeeVideoInfo";
const BOX_NAME = "video_info";
const STORE_KEY = "mykee_display";
const TAG_WIDGET = "filename_tag";
const MAX_TAG_LENGTH = 200; // keep in sync with nodes/mykee_video_info.py
const MIN_NODE_WIDTH = 260;
const ROWS = [
    ["size", "Size"],
    ["frame_rate", "Frame rate"],
    ["codec", "Codec"],
    ["format", "Format"],
    ["filename", "Filename"],
];
const ROW_HEIGHT = 20;
// First guess only (rows + padding and border): the real height is measured
// from the box itself, see createBox().
const BOX_HEIGHT_GUESS = ROWS.length * ROW_HEIGHT + 18;
const BOX_MARGIN = 4; // air under the box, inside the widget row

function widget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function lastRun(node) {
    const value = node.properties?.[STORE_KEY]?.[BOX_NAME];
    return value && typeof value === "object" ? value : null;
}

function rememberLastRun(node, value) {
    node.properties = node.properties || {};
    const store = node.properties[STORE_KEY] && typeof node.properties[STORE_KEY] === "object" ? node.properties[STORE_KEY] : {};
    store[BOX_NAME] = value;
    node.properties[STORE_KEY] = store;
}

// Same cleaning as the backend: no line breaks / control characters.
function cleanTag(tag) {
    return String(tag ?? "").replace(/[\u0000-\u001f\u007f-\u009f]/g, "").slice(0, MAX_TAG_LENGTH);
}

function createBox(node) {
    // The wrapper is the widget element (the frontend sizes it); the box inside
    // keeps its own natural height, so it can never be cut off by a row that
    // is shorter than expected. Its measured height then sets the row height.
    const wrapper = document.createElement("div");
    wrapper.style.cssText = "box-sizing:border-box;width:100%;overflow:visible;pointer-events:none;";
    const box = document.createElement("div");
    box.style.cssText =
        "box-sizing:border-box;width:100%;display:grid;grid-template-columns:auto 1fr;" +
        `grid-auto-rows:minmax(${ROW_HEIGHT}px, auto);column-gap:12px;align-items:center;padding:8px 12px;` +
        "border-radius:6px;border:1px solid var(--border-color, #444);background:var(--comfy-input-bg, #222);" +
        "color:var(--input-text, #ddd);font:12px/1.2 sans-serif;user-select:none;pointer-events:none;";
    wrapper.append(box);
    const cells = {};
    for (const [key, label] of ROWS) {
        const name = document.createElement("div");
        name.textContent = label;
        name.style.cssText = "opacity:0.6;white-space:nowrap;";
        const value = document.createElement("div");
        value.style.cssText =
            "font:600 12px/1.2 monospace;text-align:right;white-space:nowrap;overflow:hidden;" +
            "text-overflow:ellipsis;min-width:0;";
        box.append(name, value);
        cells[key] = value;
    }
    const state = { cells, shown: null, height: BOX_HEIGHT_GUESS + BOX_MARGIN };
    node._mykeeVideoInfo = state;
    const w = node.addDOMWidget(BOX_NAME, "mykee_video_info", wrapper, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => state.height,
        getMaxHeight: () => state.height,
    });
    w.serialize = false;

    // offsetHeight is in layout pixels (canvas zoom does not change it).
    if (typeof ResizeObserver !== "undefined") {
        const observer = new ResizeObserver(() => {
            const measured = box.offsetHeight;
            if (!measured) return; // not laid out (hidden tab, collapsed node)
            const wanted = measured + BOX_MARGIN;
            if (Math.abs(wanted - state.height) < 1) return;
            state.height = wanted;
            fitHeight(node);
        });
        observer.observe(box);
        const onRemoved = node.onRemoved;
        node.onRemoved = function () {
            observer.disconnect();
            return onRemoved?.apply(this, arguments);
        };
    }
}

function refreshBox(node) {
    const el = node._mykeeVideoInfo;
    if (!el) return;
    const info = lastRun(node);
    const dash = "–";
    const values = info
        ? {
            size: `${info.width} × ${info.height}`,
            frame_rate: String(info.frame_rate ?? dash),
            codec: String(info.codec ?? dash),
            format: String(info.format ?? dash),
            filename: String(info.base ?? "") + cleanTag(widget(node, TAG_WIDGET)?.value),
        }
        : { size: dash, frame_rate: dash, codec: dash, format: dash, filename: "run to see the info" };
    const key = JSON.stringify(values);
    if (el.shown === key) return;
    el.shown = key;
    for (const [name] of ROWS) {
        // textContent only: file names are shown as plain text, never as HTML
        el.cells[name].textContent = values[name];
    }
}

// The node is always exactly as tall as its content; the width is the user's.
function fitHeight(node) {
    const size = node.computeSize?.();
    if (!size || !Number.isFinite(size[1])) return;
    node.setSize([Math.max(node.size[0], size[0], MIN_NODE_WIDTH), size[1]]);
    node.graph?.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "Mykee.VideoInfo",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated?.apply(this, arguments);
            createBox(this);
            refreshBox(this);
            // ComfyUI may still apply its own size after onNodeCreated - fit again.
            setTimeout(() => fitHeight(this), 0);
            return ret;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const ret = onConfigure?.apply(this, arguments);
            // properties (the last run) are restored by now
            setTimeout(() => {
                refreshBox(this);
                fitHeight(this); // keeps the saved width, drops a stale saved height
            }, 50);
            return ret;
        };

        // A height change by hand is undone: the content decides the height.
        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            const ret = onResize?.apply(this, arguments);
            const wanted = this.computeSize?.();
            if (size && wanted && Number.isFinite(wanted[1])) size[1] = wanted[1];
            return ret;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const values = message?.[BOX_NAME];
            if (!Array.isArray(values) || !values.length || typeof values[0] !== "object") return;
            rememberLastRun(this, values[0]);
            refreshBox(this);
            this.graph?.setDirtyCanvas?.(true, true);
        };

        // Redrawn on every change, so typing a tag updates the file name row.
        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function () {
            const ret = onDrawForeground?.apply(this, arguments);
            refreshBox(this);
            return ret;
        };
    },
});
