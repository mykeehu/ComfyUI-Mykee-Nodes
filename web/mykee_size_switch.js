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
const MIN_NODE_WIDTH = 320;      // width of a NEW node only; the user may make it narrower
const MAX_SIZE = 16384;
// How far the width / height fields are shifted to the right. The socket of a
// field has a click area that reaches over the first part of the field, and
// the left arrow is only usable between the end of that area and the end of
// the arrow zone (40 + WIDGET_INDENT px from the node's left edge): a bigger
// indent makes that stretch wider.
const WIDGET_INDENT = 32;

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
    node._mykeeCfg = w.value; // our own change: the caller refreshes the toggles itself
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

const DEFAULT_STEP = 8; // the step of the arrows when divisible_by is Off

function gcd(a, b) {
    return b ? gcd(b, a % b) : a;
}

// Sides that keep a ratio EXACTLY and are both divisible by the divisor lie on a
// grid: for 4:3 and a divisor of 8 that is 32 x 24, 64 x 48, ... (1664 x 1248).
// Returns [step of the base side, step of the other side].
function ratioGrid(rw, rh, div, baseIsWidth) {
    const a = div / gcd(rw, div);
    const b = div / gcd(rh, div);
    const m = (a * b) / gcd(a, b);
    return baseIsWidth ? [rw * m, rh * m] : [rh * m, rw * m];
}

// [base, other] for a ratio rw:rh with the base side rounded to the divisor (the
// same maths as the backend). The nearest grid point is used when it is close to
// the typed number (within 4%, or one divisor); otherwise the base is rounded and
// the other side is calculated from it. Fitting the result again changes nothing.
function fitRatio(base, rw, rh, div, baseIsWidth) {
    const b = snap(base, div);
    const factor = baseIsWidth ? rh / rw : rw / rh;
    const fallback = [b, snap(b * factor, div)];
    if (!div) return fallback;
    const [step, otherStep] = ratioGrid(rw, rh, div, baseIsWidth);
    const lowest = Math.max(MIN_SIZE, div);
    let best = null;
    const k0 = Math.floor(b / step);
    for (const k of [k0, k0 + 1]) {
        if (k < 1) continue;
        const value = step * k;
        const other = otherStep * k;
        if (value < lowest || value > MAX_SIZE || other < lowest || other > MAX_SIZE) continue;
        const distance = Math.abs(value - b);
        if (best === null || distance < best[0]) best = [distance, value, other];
    }
    if (best !== null && best[0] <= Math.max(div, b * 0.04)) return [best[1], best[2]];
    return fallback;
}

// The arrows and dragging step by the divisor, so one click always lands on
// the next valid number (with a step of 8 and a divisor of 16, 1024 - 8 would
// be rounded straight back to 1024).
function applyStep(node) {
    const div = divisor(node);
    for (let i = 1; i <= MAX_PAIRS; i++) {
        // With a ratio and a divisor the valid pairs lie on a grid (4:3 and 8:
        // 32 x 24, 64 x 48 ...), so an arrow click moves one grid point of the
        // side it belongs to, and a typed number is never mistaken for a click.
        const ratio = parseRatio(widget(node, `ratio_${i}`)?.value);
        const grid = ratio && div ? ratioGrid(ratio[0], ratio[1], div, true) : null;
        for (const axis of ["width", "height"]) {
            const w = widget(node, `${axis}_${i}`);
            if (!w) continue;
            const step = grid ? grid[axis === "width" ? 0 : 1] : div || DEFAULT_STEP;
            w.options = w.options || {};
            w.options.step2 = step;
            w.options.step = step * 10;
            // The frontend rounds typed / stepped values to (value - min % step)
            // steps, i.e. with an offset when min is not a multiple of the step:
            // min 16 with a step of 32 gives 16, 48, 80 ... so every arrow click
            // was rounded to an off-grid number and then snapped back by us
            // (stuck at 3136). A min that is a multiple of the step removes the
            // offset (this is also the lowest value snap() allows).
            w.options.min = Math.max(MIN_SIZE, grid ? step : div);
        }
    }
}

// Remembers the numbers of a pair, to tell an arrow click from typing.
function rememberPair(node, i) {
    for (const axis of ["width", "height"]) {
        const w = widget(node, `${axis}_${i}`);
        if (w) w._mykeeLast = Number(w.value);
    }
}

// Fallback for frontends that ignore the step above: a change of exactly one
// default step that the rounding would undo moves one divisor instead.
function nudge(node, w, i, axis) {
    const div = divisor(node);
    const prior = w._mykeeLast;
    const value = Number(w.value);
    if (!div || !Number.isFinite(prior)) return;
    // With a ratio the arrows move along the exact-ratio grid, one grid point at a
    // time; otherwise the nearest grid point would be the one the arrow came from.
    const ratio = parseRatio(widget(node, `ratio_${i}`)?.value);
    if (ratio && !connectedAxes(node).has(`${axis === "w" ? "width" : "height"}_${i}`)) {
        const delta = value - prior;
        if (delta !== 0 && Math.abs(delta) <= Math.max(div, DEFAULT_STEP)) {
            const [step] = ratioGrid(ratio[0], ratio[1], div, axis === "w");
            const grid = Math.sign(delta) > 0 ? Math.floor(prior / step) + 1 : Math.ceil(prior / step) - 1;
            w.value = Math.max(Math.max(MIN_SIZE, div), Math.min(MAX_SIZE, grid * step));
        }
        return;
    }
    if (div <= DEFAULT_STEP) return;
    if (Math.abs(value - prior) === DEFAULT_STEP && value % div !== 0) {
        w.value = prior + Math.sign(value - prior) * div;
    }
}

// Rounds the typed numbers of one pair and, with a ratio, calculates the side
// that does not lead. A side that arrives through a connection is left alone
// (its value is not known here); the backend handles it.
function applyPairRaw(node, i) {
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
            [wWidget.value, hWidget.value] = fitRatio(Number(wWidget.value), rw, rh, div, true);
        } else {
            [hWidget.value, wWidget.value] = fitRatio(Number(hWidget.value), rw, rh, div, false);
        }
    } else if (!wLinked) {
        wWidget.value = snap(Number(wWidget.value), div); // only the base can be rounded
    } else if (!hLinked) {
        // the connected width leads: its value is unknown here, keep the typed height
    }
}

function applyPair(node, i) {
    applyPairRaw(node, i);
    rememberPair(node, i);
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

// Draws a number widget a little narrower and shifted to the right, so there
// is a gap between its socket and the field, and the width / height fields
// stand out as a group under the "Pair N" row. The left margin of a widget is
// a fixed value in the frontend, so the drawing is shifted here, and so are
// the clicks (the left arrow zone is measured from the node's left edge).
function indentWidget(w) {
    if (!w || w._mykeeIndented) return;
    w._mykeeIndented = true;

    if (typeof w.drawWidget === "function") {
        const drawWidget = w.drawWidget;
        w.drawWidget = function (ctx, options) {
            ctx.save();
            ctx.translate(WIDGET_INDENT, 0);
            try {
                return drawWidget.call(this, ctx, { ...options, width: (options?.width ?? 0) - WIDGET_INDENT });
            } finally {
                ctx.restore();
            }
        };
    } else if (typeof w.draw === "function") {
        const draw = w.draw; // older frontends: draw(ctx, node, widgetWidth, y, height)
        w.draw = function (ctx, node, widgetWidth, ...rest) {
            ctx.save();
            ctx.translate(WIDGET_INDENT, 0);
            try {
                return draw.call(this, ctx, node, widgetWidth - WIDGET_INDENT, ...rest);
            } finally {
                ctx.restore();
            }
        };
    }

    if (typeof w.onClick === "function") {
        const onClick = w.onClick;
        w.onClick = function (options) {
            const e = options?.e;
            const node = options?.node;
            // Left half only: the right arrow keeps its place.
            if (e && node && typeof e.canvasX === "number" && e.canvasX - node.pos[0] < node.size[0] / 2) {
                const saved = e.canvasX;
                e.canvasX = saved - WIDGET_INDENT;
                try {
                    return onClick.call(this, options);
                } finally {
                    e.canvasX = saved;
                }
            }
            return onClick.call(this, options);
        };
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
//
// The size the user gives the node by dragging its corner is remembered as the
// width plus the EXTRA height above what the content needs, in node.properties
// (saved with the workflow). The extra - not the absolute height - is kept so
// the node still follows its content when pairs are added or removed.
const SIZE_PROP = "mykee_size";
const MAX_USER_SIZE = 4000;
const MIN_USER_WIDTH = 100;      // sanity bound; the real minimum is what the widgets need

function readUserSize(node) {
    const raw = node.properties?.[SIZE_PROP];
    if (!raw || typeof raw !== "object") return null;
    const w = Number(raw.w);
    const extra = Number(raw.extra);
    if (!Number.isFinite(w) || !Number.isFinite(extra)) return null;
    return {
        w: Math.max(MIN_USER_WIDTH, Math.min(MAX_USER_SIZE, Math.round(w))),
        extra: Math.max(0, Math.min(MAX_USER_SIZE, Math.round(extra))),
    };
}

function writeUserSize(node, w, extra) {
    const next = {
        w: Math.max(MIN_USER_WIDTH, Math.min(MAX_USER_SIZE, Math.round(w))),
        extra: Math.max(0, Math.min(MAX_USER_SIZE, Math.round(extra))),
    };
    const old = readUserSize(node);
    if (old && old.w === next.w && old.extra === next.extra) return;
    node.properties = node.properties || {};
    node.properties[SIZE_PROP] = next;
}

// Called while / right after the user drags the node's resize corner.
function rememberUserSize(node) {
    const size = node.computeSize?.();
    if (!size || !Number.isFinite(size[1])) return;
    writeUserSize(node, node.size[0], node.size[1] - size[1]);
}

// Workflows saved before the size was remembered carry only the node's size:
// take the width and the extra height from it (once, with the layout built).
function adoptSavedSize(node, saved) {
    if (readUserSize(node) || !Array.isArray(saved)) return;
    const size = node.computeSize?.();
    if (!size || !Number.isFinite(size[1]) || !Number.isFinite(saved[0]) || !Number.isFinite(saved[1])) return;
    writeUserSize(node, saved[0], saved[1] - size[1]);
}

function fitHeight(node) {
    const size = node.computeSize?.();
    if (!size || !Number.isFinite(size[1])) return;
    const user = readUserSize(node);
    node.setSize([
        Math.max(user ? user.w : node.size[0], size[0]),
        size[1] + (user ? user.extra : 0),
    ]);
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

function refreshAll(node, savedSize) {
    syncPairs(node);
    refreshDisabled(node);
    applyStep(node);
    for (let i = 1; i <= MAX_PAIRS; i++) rememberPair(node, i);
    refreshToggles(node);
    if (savedSize) adoptSavedSize(node, savedSize);
    fitHeight(node);
    node._mykeeSig = layoutSignature(node);
    node._mykeeCfg = widget(node, CONFIG_WIDGET)?.value ?? "";
}

// What the visible layout depends on.
function layoutSignature(node) {
    return [pairCount(node), isBatch(node) ? 1 : 0, divisor(node)].join("|");
}

// Self-healing: some ways of rebuilding a node (e.g. a "Reload node" menu item
// of an extension) copy the widget values into a fresh node WITHOUT calling any
// widget callback or onConfigure, so pair_count showed 8 while only one pair was
// built. Whenever the widget values and the built layout disagree, rebuild.
function ensureSynced(node) {
    if (!node._mykeeReady || node._mykeeSyncPending) return;
    const cfg = widget(node, CONFIG_WIDGET)?.value ?? "";
    if (layoutSignature(node) === node._mykeeSig && cfg === node._mykeeCfg) return;
    node._mykeeSyncPending = true;
    setTimeout(() => {
        node._mykeeSyncPending = false;
        if (layoutSignature(node) !== node._mykeeSig) refreshAll(node);
        else {
            node._mykeeCfg = widget(node, CONFIG_WIDGET)?.value ?? "";
            refreshToggles(node);
            node.graph?.setDirtyCanvas?.(true, true);
        }
    }, 0);
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
                applyStep(node);
                applyPair(node, i);
                node.graph?.setDirtyCanvas?.(true, true);
                return ret;
            };
        }
        for (const axis of ["width", "height"]) {
            const w = widget(node, `${axis}_${i}`);
            if (!w) continue;
            w.label = axis; // the "Pair N" row above says which pair it is
            indentWidget(w);
            w.serializeValue = () => (usedPairs(node).includes(i) ? w.value : MIN_SIZE);
            const callback = w.callback;
            w.callback = function () {
                const ret = callback?.apply(this, arguments);
                // The side edited last leads when a ratio is set.
                nudge(node, w, i, axis === "width" ? "w" : "h");
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
            applyStep(node);
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

    node.mykeeSizeRefresh = (savedSize) => refreshAll(node, savedSize);
    node.setSize([Math.max(node.size[0], MIN_NODE_WIDTH), node.size[1]]);
    refreshAll(node);
    node._mykeeReady = true;
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
        nodeType.prototype.onConfigure = function (info) {
            const ret = onConfigure?.apply(this, arguments);
            // Widget values and links are restored by now (links of other
            // nodes may still be loading, hence the small delay).
            const savedSize = Array.isArray(info?.size) ? [...info.size] : null;
            setTimeout(() => this.mykeeSizeRefresh?.(savedSize), 50);
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

        const onAdded = nodeType.prototype.onAdded;
        nodeType.prototype.onAdded = function () {
            const ret = onAdded?.apply(this, arguments);
            // Values may be copied in right after the node is added (reload).
            for (const delay of [0, 100, 500]) setTimeout(() => ensureSynced(this), delay);
            return ret;
        };

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function () {
            const ret = onDrawForeground?.apply(this, arguments);
            ensureSynced(this);
            updateButtonCursor(this);
            // The size is stored while the user drags the corner, and once more
            // on the frame after the mouse is released.
            const resizing = app.canvas?.resizing_node === this;
            if (resizing || this._mykeeWasResizing) rememberUserSize(this);
            this._mykeeWasResizing = resizing;
            return ret;
        };
    },
});
