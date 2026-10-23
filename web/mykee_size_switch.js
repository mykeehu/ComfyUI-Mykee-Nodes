/*
 * Mykee Size Switch/Batch - frontend.
 *
 * The backend declares MAX_PAIRS blocks of four widgets each:
 *
 *   use_N     toggle: radio button (switch mode) / checkbox (batch mode)
 *   ratio_N   aspect ratio combo (Custom, 1:1, 16:9, ...)
 *   width_N   INT widget (can be connected from outside)
 *   height_N  INT widget (can be connected from outside)
 *
 * This keeps only the first pair_count blocks visible (and keeps only their
 * width / height sockets on the node). The selection is stored in the hidden
 * pairs_config widget:
 *
 *   { checked: [...], selected: <1-based index>, driver: ["w"|"h", ...] }
 *
 * "driver" is the side of a pair that was edited last: with a ratio it is the
 * base, the other side is calculated from it (same maths as the backend).
 * Typed numbers are rounded to divisible_by right here, so the widgets show
 * what will really be used.
 *
 * The switch-mode selection and the batch-mode ticks are stored separately, so
 * flipping the mode loses neither. Lowering pair_count and raising it again
 * brings the old values back.
 *
 * Only the selected pairs go into the prompt (serializeValue): the values of
 * unselected pairs are replaced by a constant, so editing them does not re-run
 * anything.
 */
import { app } from "../../scripts/app.js";

const NODE_NAME = "MykeeSizeSwitchBatch";
const CONFIG_WIDGET = "pairs_config";
const MAX_PAIRS = 32;
const MIN_SIZE = 16;
const MIN_NODE_WIDTH = 300;
const MAX_SIZE = 16384;

function widget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function isBatch(node) {
    return !!widget(node, "batch_mode")?.value;
}

function pairCount(node) {
    const n = Math.round(Number(widget(node, "pair_count")?.value ?? 1));
    return Number.isFinite(n) ? Math.max(1, Math.min(MAX_PAIRS, n)) : 1;
}

function readState(node) {
    let state = {};
    try {
        state = JSON.parse(widget(node, CONFIG_WIDGET)?.value || "{}") || {};
    } catch {
        state = {};
    }
    // A pair that was never touched counts as ticked, so a freshly added pair
    // is part of a batch right away.
    const checked = Array.isArray(state.checked) ? state.checked.map(Boolean) : [];
    while (checked.length < MAX_PAIRS) checked.push(true);
    const selected = Number.isInteger(state.selected) ? state.selected : 1;
    const driver = Array.isArray(state.driver) ? state.driver.map((d) => (d === "h" ? "h" : "w")) : [];
    while (driver.length < MAX_PAIRS) driver.push("w");
    return { checked: checked.slice(0, MAX_PAIRS), selected, driver: driver.slice(0, MAX_PAIRS) };
}

function writeState(node, state) {
    const w = widget(node, CONFIG_WIDGET);
    if (!w) return;
    w.value = JSON.stringify({ checked: state.checked, selected: state.selected, driver: state.driver });
}

// The pair numbers that actually go out.
function usedPairs(node) {
    const state = readState(node);
    const count = pairCount(node);
    const used = [];
    if (isBatch(node)) {
        for (let i = 1; i <= count; i++) if (state.checked[i - 1]) used.push(i);
    } else if (state.selected >= 1 && state.selected <= count) {
        used.push(state.selected);
    }
    return used;
}

// Which width_N / height_N / image_N sockets are connected? Uses the graph's link
// table: newer frontends only mirror links in input.link, and the mirror can
// lag behind.
function connectedAxes(node) {
    const result = new Set();
    const table = node.graph?.links;
    const links = table instanceof Map ? [...table.values()] : table ? Object.values(table) : [];
    for (const link of links) {
        if (!link || String(link.target_id) !== String(node.id)) continue;
        const name = node.inputs?.[link.target_slot]?.name;
        if (name && /^(width|height|image)_\d+$/.test(name)) result.add(name);
    }
    return result;
}

// The side that leads a pair: the connected one if only one side is
// connected, otherwise the one edited last.
function driverOf(state, i, wLinked, hLinked) {
    if (wLinked !== hLinked) return wLinked ? "w" : "h";
    return state.driver[i - 1];
}

// What the backend gets: only the pairs that are used.
function resolveConfig(node) {
    const state = readState(node);
    const linked = connectedAxes(node);
    const items = usedPairs(node).map((index) => {
        // A connected image replaces both typed sides.
        const imageLinked = linked.has(`image_${index}`);
        const wLinked = imageLinked || linked.has(`width_${index}`);
        const hLinked = imageLinked || linked.has(`height_${index}`);
        return {
            index,
            driver: driverOf(state, index, wLinked, hLinked),
            w_linked: wLinked,
            h_linked: hLinked,
        };
    });
    return JSON.stringify({ items });
}

// ---------------------------------------------------------------------------
// Rounding and aspect ratio (the same maths as the backend)
// ---------------------------------------------------------------------------

function divisor(node) {
    const n = parseInt(widget(node, "divisible_by")?.value, 10);
    return Number.isFinite(n) ? n : 0; // "Off"
}

function snap(value, div) {
    const lowest = Math.max(MIN_SIZE, div);
    const result = div ? Math.floor(value / div + 0.5) * div : Math.floor(value + 0.5);
    return Math.max(lowest, Math.min(MAX_SIZE, result));
}

function parseRatio(text) {
    const match = /^(\d+):(\d+)$/.exec(String(text ?? ""));
    if (!match) return null;
    const a = Number(match[1]);
    const b = Number(match[2]);
    return a > 0 && b > 0 ? [a, b] : null;
}

// Rounds the typed numbers of one pair and, with a ratio, calculates the side
// that does not lead. A side that arrives through a connection is left alone
// (its value is not known here); the backend handles it.
function applyPair(node, i) {
    const wWidget = widget(node, `width_${i}`);
    const hWidget = widget(node, `height_${i}`);
    if (!wWidget || !hWidget) return;
    const linked = connectedAxes(node);
    const imageLinked = linked.has(`image_${i}`);
    const wLinked = imageLinked || linked.has(`width_${i}`);
    const hLinked = imageLinked || linked.has(`height_${i}`);
    const div = divisor(node);
    const ratio = parseRatio(widget(node, `ratio_${i}`)?.value);
    const driver = driverOf(readState(node), i, wLinked, hLinked);

    if (!ratio) {
        if (!wLinked) wWidget.value = snap(Number(wWidget.value), div);
        if (!hLinked) hWidget.value = snap(Number(hWidget.value), div);
        return;
    }
    const [rw, rh] = ratio;
    if (!wLinked && !hLinked) {
        if (driver === "w") {
            wWidget.value = snap(Number(wWidget.value), div);
            hWidget.value = snap((wWidget.value * rh) / rw, div);
        } else {
            hWidget.value = snap(Number(hWidget.value), div);
            wWidget.value = snap((hWidget.value * rw) / rh, div);
        }
    } else if (!wLinked) {
        wWidget.value = snap(Number(wWidget.value), div); // only the base can be rounded
    } else if (!hLinked) {
        // the connected width leads: its value is unknown here, keep the typed height
    }
}

function applyAll(node) {
    for (let i = 1; i <= pairCount(node); i++) applyPair(node, i);
    node.graph?.setDirtyCanvas?.(true, true);
}

// ---------------------------------------------------------------------------
// Showing / hiding the pair blocks and their sockets
// ---------------------------------------------------------------------------

// Shows/hides a widget and makes litegraph reserve no row for it while hidden.
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

function inputIndex(node, name) {
    return (node.inputs ?? []).findIndex((inp) => inp.name === name);
}

// Keeps the width_N / height_N sockets of the visible pairs only, and none of
// the use_N / ratio_N widgets (they have no business being connected).
function syncPairs(node) {
    const count = pairCount(node);
    for (let i = 1; i <= MAX_PAIRS; i++) {
        const visible = i <= count;
        for (const name of [`use_${i}`, `ratio_${i}`, `width_${i}`, `height_${i}`]) {
            setWidgetVisible(widget(node, name), visible);
        }
        for (const name of [`width_${i}`, `height_${i}`]) {
            const idx = inputIndex(node, name);
            if (visible && idx < 0) node.addInput(name, "INT", { widget: { name } });
            else if (!visible && idx >= 0) node.removeInput(idx);
        }
        const imageIdx = inputIndex(node, `image_${i}`);
        if (visible && imageIdx < 0) node.addInput(`image_${i}`, "IMAGE");
        else if (!visible && imageIdx >= 0) node.removeInput(imageIdx);
        for (const name of [`use_${i}`, `ratio_${i}`]) {
            const idx = inputIndex(node, name);
            if (idx >= 0 && node.inputs[idx].link == null) node.removeInput(idx);
        }
    }
}

// Writes the stored selection into the use_N toggles, with labels that fit
// the current mode.
function refreshToggles(node) {
    const state = readState(node);
    const batch = isBatch(node);
    for (let i = 1; i <= MAX_PAIRS; i++) {
        const w = widget(node, `use_${i}`);
        if (!w) continue;
        w.label = `Pair ${i}`;
        w.value = batch ? state.checked[i - 1] : state.selected === i;
        w.options = w.options || {};
        w.options.on = batch ? "☑ include" : "◉ selected";
        w.options.off = batch ? "☐ skip" : "○";
    }
    for (const button of node._mykeeButtons ?? []) setWidgetVisible(button, batch);
}

// The node's height always follows its content (grows AND shrinks). ComfyUI
// sizes a new node while all pairs are still there, so without the shrinking
// a new node would start far too tall.
function fitHeight(node) {
    const size = node.computeSize?.();
    if (!size || !Number.isFinite(size[1])) return;
    node.setSize([Math.max(node.size[0], size[0], MIN_NODE_WIDTH), size[1]]);
    node.graph?.setDirtyCanvas?.(true, true);
}

// Width / height of a pair with a connected image are not used: grey them out.
function refreshDisabled(node) {
    const linked = connectedAxes(node);
    for (let i = 1; i <= MAX_PAIRS; i++) {
        const fromImage = linked.has(`image_${i}`);
        for (const axis of ["width", "height"]) {
            const w = widget(node, `${axis}_${i}`);
            if (w) w.disabled = fromImage;
        }
    }
}

function refreshAll(node) {
    syncPairs(node);
    refreshDisabled(node);
    refreshToggles(node);
    fitHeight(node);
}

// ---------------------------------------------------------------------------
// Pointing-hand cursor over the button widgets
// ---------------------------------------------------------------------------

// litegraph records each widget's drawn row position in widget.last_y after
// every draw pass - used to switch the canvas cursor to a pointing hand while
// the mouse is over one of the button widgets.
function updateButtonCursor(node) {
    const canvasEl = app.canvas && app.canvas.canvas;
    if (!canvasEl) return;
    const rowH = (window.LiteGraph && window.LiteGraph.NODE_WIDGET_HEIGHT) || 20;
    const gm = app.canvas.graph_mouse;
    let hovering = false;
    if (gm && node.pos && node.widgets) {
        const localX = gm[0] - node.pos[0];
        const localY = gm[1] - node.pos[1];
        for (const w of node.widgets) {
            if (w.type !== "button" || typeof w.last_y !== "number") continue;
            if (localX >= 15 && localX <= node.size[0] - 15 && localY >= w.last_y && localY <= w.last_y + rowH) {
                hovering = true;
                break;
            }
        }
    }
    if (hovering && !node._mykeeButtonCursorSet) {
        canvasEl.style.cursor = "pointer";
        node._mykeeButtonCursorSet = true;
    } else if (!hovering && node._mykeeButtonCursorSet) {
        canvasEl.style.cursor = "";
        node._mykeeButtonCursorSet = false;
    }
}

// ---------------------------------------------------------------------------
// Node setup
// ---------------------------------------------------------------------------

function setupNode(node) {
    const configWidget = widget(node, CONFIG_WIDGET);
    if (configWidget) {
        configWidget.hidden = true;
        configWidget.computeSize = () => [0, -4];
        configWidget.serializeValue = () => resolveConfig(node);
    }

    for (let i = 1; i <= MAX_PAIRS; i++) {
        // The selection toggle is UI only: a constant goes into the prompt.
        const use = widget(node, `use_${i}`);
        if (use) {
            use.serializeValue = () => false;
            const callback = use.callback;
            use.callback = function (value) {
                const ret = callback?.apply(this, arguments);
                const state = readState(node);
                if (isBatch(node)) {
                    state.checked[i - 1] = !!value;
                } else {
                    state.selected = i; // radio: clicking the selected one keeps it
                }
                writeState(node, state);
                refreshToggles(node);
                node.graph?.setDirtyCanvas?.(true, true);
                return ret;
            };
        }
        // Unselected pairs send a constant, so editing them re-runs nothing.
        // (A connected socket is not affected: the link is sent instead.)
        const ratio = widget(node, `ratio_${i}`);
        if (ratio) {
            ratio.serializeValue = () => (usedPairs(node).includes(i) ? ratio.value : "Custom");
            const callback = ratio.callback;
            ratio.callback = function () {
                const ret = callback?.apply(this, arguments);
                applyPair(node, i);
                node.graph?.setDirtyCanvas?.(true, true);
                return ret;
            };
        }
        for (const axis of ["width", "height"]) {
            const w = widget(node, `${axis}_${i}`);
            if (!w) continue;
            w.serializeValue = () => (usedPairs(node).includes(i) ? w.value : MIN_SIZE);
            const callback = w.callback;
            w.callback = function () {
                const ret = callback?.apply(this, arguments);
                // The side edited last leads when a ratio is set.
                const state = readState(node);
                state.driver[i - 1] = axis === "width" ? "w" : "h";
                writeState(node, state);
                applyPair(node, i);
                node.graph?.setDirtyCanvas?.(true, true);
                return ret;
            };
        }
    }

    // select all / clear all (batch mode only); at the end of the widget list
    // so the positional widget values of saved workflows stay in place.
    node._mykeeButtons = [];
    for (const [label, value] of [["select all", true], ["clear all", false]]) {
        const button = node.addWidget("button", label, null, () => {
            const state = readState(node);
            for (let i = 0; i < pairCount(node); i++) state.checked[i] = value;
            writeState(node, state);
            refreshToggles(node);
            node.graph?.setDirtyCanvas?.(true, true);
        });
        button.serialize = false;
        node._mykeeButtons.push(button);
    }

    const divisorWidget = widget(node, "divisible_by");
    if (divisorWidget) {
        const callback = divisorWidget.callback;
        divisorWidget.callback = function () {
            const ret = callback?.apply(this, arguments);
            applyAll(node);
            return ret;
        };
    }

    for (const name of ["batch_mode", "pair_count"]) {
        const w = widget(node, name);
        if (!w) continue;
        const callback = w.callback;
        w.callback = function () {
            const ret = callback?.apply(this, arguments);
            refreshAll(node);
            return ret;
        };
    }

    node.mykeeSizeRefresh = () => refreshAll(node);
    node.setSize([Math.max(node.size[0], MIN_NODE_WIDTH), node.size[1]]);
    refreshAll(node);
    // ComfyUI may still apply its own size after onNodeCreated - fit again.
    setTimeout(() => fitHeight(node), 0);
}

app.registerExtension({
    name: "Mykee.SizeSwitchBatch",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated?.apply(this, arguments);
            setupNode(this);
            return ret;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const ret = onConfigure?.apply(this, arguments);
            // Widget values and links are restored by now (links of other
            // nodes may still be loading, hence the small delay).
            setTimeout(() => this.mykeeSizeRefresh?.(), 50);
            return ret;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type) {
            const ret = onConnectionsChange?.apply(this, arguments);
            const INPUT = window.LiteGraph?.INPUT ?? 1;
            if (type === INPUT) {
                setTimeout(() => {
                    refreshDisabled(this);
                    this.graph?.setDirtyCanvas?.(true, true);
                }, 0);
            }
            return ret;
        };

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function () {
            const ret = onDrawForeground?.apply(this, arguments);
            updateButtonCursor(this);
            return ret;
        };
    },
});
