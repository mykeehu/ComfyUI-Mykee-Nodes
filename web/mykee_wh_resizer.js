/*
 * Mykee Width/Height Resizer - frontend.
 *
 * - Shows only the field that belongs to the chosen mode: scale_by for
 *   "scale by", scale_to (and the reference) for "scale to".
 * - A small box at the bottom of the node shows the output size. It is
 *   calculated live from the typed numbers (same maths as the backend). When
 *   width / height come through a connection their value is unknown here, so
 *   the box shows the result of the last run instead.
 */
import { app } from "../../scripts/app.js";

const NODE_NAME = "MykeeWidthHeightResizer";
const PREVIEW_NAME = "size_preview";
const STORE_KEY = "mykee_display";
const MIN_SIZE = 1;
const MAX_SIZE = 65536;
const MIN_NODE_WIDTH = 260;
const PREVIEW_HEIGHT = 64;

function widget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

// ---- math (keep in sync with nodes/mykee_wh_resizer.py) --------------------
function snap(value, div) {
    value = Math.round(value * 1e6) / 1e6; // swallow float noise
    const lowest = Math.max(MIN_SIZE, div);
    const result = div ? Math.floor(value / div + 0.5) * div : Math.floor(value + 0.5);
    return Math.max(lowest, Math.min(MAX_SIZE, result));
}

function referenceValue(width, height, reference) {
    if (reference === "width") return width;
    if (reference === "height") return height;
    if (reference === "shortest side") return Math.min(width, height);
    return Math.max(width, height);
}

function computeSize(width, height, mode, reference, scaleBy, scaleTo, divisibleBy) {
    width = Math.max(MIN_SIZE, Math.trunc(width));
    height = Math.max(MIN_SIZE, Math.trunc(height));
    let newW, newH;
    if (mode === "scale to") {
        const base = referenceValue(width, height, reference);
        newW = (width * scaleTo) / base;
        newH = (height * scaleTo) / base;
    } else {
        newW = width * scaleBy;
        newH = height * scaleBy;
    }
    const div = parseInt(divisibleBy, 10) || 0; // "Off" -> 0
    return [snap(newW, div), snap(newH, div)];
}
// ---- end math ---------------------------------------------------------------

function isLinked(node, name) {
    return !!node.inputs?.some((inp) => inp.name === name && inp.link != null);
}

function liveSize(node) {
    // Unknown while width / height arrive through a connection.
    if (isLinked(node, "image") || isLinked(node, "width") || isLinked(node, "height")) return null;
    const v = (name) => widget(node, name)?.value;
    const width = Number(v("width"));
    const height = Number(v("height"));
    if (!Number.isFinite(width) || !Number.isFinite(height)) return null;
    return computeSize(
        width, height, v("mode"), v("reference"),
        Number(v("scale_by")), Number(v("scale_to")), v("divisible_by")
    );
}

function lastRun(node) {
    const value = node.properties?.[STORE_KEY]?.[PREVIEW_NAME];
    return typeof value === "string" && value ? value : "";
}

function rememberLastRun(node, value) {
    node.properties = node.properties || {};
    const store = node.properties[STORE_KEY] && typeof node.properties[STORE_KEY] === "object" ? node.properties[STORE_KEY] : {};
    store[PREVIEW_NAME] = value;
    node.properties[STORE_KEY] = store;
}

// ---- preview box -------------------------------------------------------------
function createPreview(node) {
    const box = document.createElement("div");
    box.style.cssText =
        "box-sizing:border-box;width:100%;height:100%;display:flex;flex-direction:column;" +
        "align-items:center;justify-content:center;gap:4px;padding:8px 12px;border-radius:6px;" +
        "border:1px solid var(--border-color, #444);background:var(--comfy-input-bg, #222);" +
        "color:var(--input-text, #ddd);user-select:none;pointer-events:none;overflow:hidden;";
    const main = document.createElement("div");
    main.style.cssText = "font:600 16px/1.2 monospace;letter-spacing:0.03em;";
    const sub = document.createElement("div");
    sub.style.cssText = "font:11px/1.3 sans-serif;opacity:0.6;text-align:center;";
    box.append(main, sub);

    const w = node.addDOMWidget(PREVIEW_NAME, "mykee_size_preview", box, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => PREVIEW_HEIGHT,
        getMaxHeight: () => PREVIEW_HEIGHT,
    });
    w.serialize = false;
    node._mykeePreview = { main, sub, shown: null };
}

function refreshPreview(node) {
    const el = node._mykeePreview;
    if (!el) return;
    let text, note;
    const live = liveSize(node);
    if (live) {
        text = `${live[0]} × ${live[1]}`;
        note = "output size";
    } else {
        const last = lastRun(node);
        text = last || "– × –";
        const source = isLinked(node, "image") ? "image" : "connected input";
        note = last ? `${source} - size of the last run` : `${source} - run to see the size`;
    }
    const key = text + "|" + note;
    if (el.shown === key) return;
    el.shown = key;
    el.main.textContent = text;
    el.sub.textContent = note;
}

// ---- widget visibility (same technique as the Size Switch/Batch node) ----------
function setWidgetVisible(w, visible) {
    if (!w) return;
    if (!w._mykeeOrig) {
        w._mykeeOrig = {
            type: w.type,
            hadComputeSize: Object.prototype.hasOwnProperty.call(w, "computeSize"),
            computeSize: w.computeSize,
            hadLayoutSize: Object.prototype.hasOwnProperty.call(w, "computeLayoutSize"),
            computeLayoutSize: w.computeLayoutSize,
        };
    }
    const orig = w._mykeeOrig;
    if (visible) {
        // Type first: while the type is still "hidden", newer frontends
        // ignore writes to widget.hidden (they derive visibility from it).
        w.type = orig.type;
        w.hidden = false;
        if (orig.hadComputeSize) w.computeSize = orig.computeSize;
        else delete w.computeSize;
        if (orig.hadLayoutSize) w.computeLayoutSize = orig.computeLayoutSize;
        else delete w.computeLayoutSize;
    } else {
        w.hidden = true;
        w.type = "hidden";
        w.computeSize = () => [0, -4];
        w.computeLayoutSize = () => ({ minHeight: 0, maxHeight: 0, minWidth: 0 });
    }
}

function fitHeight(node) {
    const size = node.computeSize?.();
    if (!size || !Number.isFinite(size[1])) return;
    node.setSize([Math.max(node.size[0], size[0], MIN_NODE_WIDTH), size[1]]);
    node.graph?.setDirtyCanvas?.(true, true);
}

function syncMode(node) {
    const scaleTo = widget(node, "mode")?.value === "scale to";
    setWidgetVisible(widget(node, "reference"), scaleTo);
    setWidgetVisible(widget(node, "scale_to"), scaleTo);
    setWidgetVisible(widget(node, "scale_by"), !scaleTo);
    fitHeight(node);
}

app.registerExtension({
    name: "Mykee.WidthHeightResizer",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated?.apply(this, arguments);
            createPreview(this);
            const modeWidget = widget(this, "mode");
            if (modeWidget) {
                const callback = modeWidget.callback;
                modeWidget.callback = (...args) => {
                    const r = callback?.apply(modeWidget, args);
                    syncMode(this);
                    return r;
                };
            }
            syncMode(this);
            refreshPreview(this);
            // ComfyUI may still apply its own size after onNodeCreated - fit again.
            setTimeout(() => fitHeight(this), 0);
            return ret;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const ret = onConfigure?.apply(this, arguments);
            // Widget values are restored by now; the small delay lets the
            // links of other nodes load too.
            setTimeout(() => {
                syncMode(this);
                refreshPreview(this);
            }, 50);
            return ret;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const values = message?.[PREVIEW_NAME];
            if (!Array.isArray(values) || !values.length) return;
            // A list on the inputs (e.g. from Size Switch/Batch) runs the node
            // once per entry; show all of them.
            rememberLastRun(this, values.join("  ·  "));
            refreshPreview(this);
            this.graph?.setDirtyCanvas?.(true, true);
        };

        // Redrawn on every change (typing, arrows, connecting a link), so the
        // box needs no hooks on the individual widgets.
        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function () {
            const ret = onDrawForeground?.apply(this, arguments);
            // A connected image replaces the typed width / height: grey them out.
            const fromImage = isLinked(this, "image");
            for (const name of ["width", "height"]) {
                const w = widget(this, name);
                if (w && w.disabled !== fromImage) w.disabled = fromImage;
            }
            refreshPreview(this);
            return ret;
        };
    },
});
