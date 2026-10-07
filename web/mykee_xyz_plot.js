/*
 * Mykee XYZ Plot - frontend.
 *
 * The node's x / y / z outputs are "*" typed. Dragging one onto a widget
 * of any node turns that widget into a plot axis: the output takes over the
 * widget's type (like a Primitive node does) and the node's panel shows an
 * editor for the axis values:
 *
 *   INT / FLOAT  comma separated values and/or ranges ("5, 7.5, 10", "4:12:2")
 *   STRING       one value per line
 *   COMBO        searchable multi-select of the widget's own option list
 *   BOOLEAN      true / false checkboxes
 *
 * Queueing: api.queuePrompt is wrapped so that one Run becomes one prompt
 * per plot cell (Z outermost, X innermost). In every cell's prompt the links
 * coming from the x / y / z outputs are replaced by the literal value of that
 * cell - exactly what a Primitive node would put there. So ComfyUI's cache
 * only re-runs what really changed between two cells (e.g. a checkpoint
 * loader on the Z axis loads each model once). The controller node itself
 * gets the cell index / run id in its xyz_config input, and passes the axis
 * labels on to the Assembler through xyz_info.
 *
 * The wrapper is installed when this module loads, i.e. before any
 * extension's setup() runs. That makes it the innermost wrapper, so e.g.
 * Mykee Seed (which wraps in setup) resolves its random seed once per Run,
 * and every cell of the plot gets the same seed.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "MykeeXYZPlot";
const CONFIG_WIDGET = "xyz_config";
const AXES = ["X", "Y", "Z"]; // output slots 0, 1, 2
const OUTPUT_NAMES = ["x", "y", "z"];
const SUPPORTED_TYPES = ["INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"];
const CELL_CONFIRM_LIMIT = 100;
const RANDOM_SEED_MAX = 2 ** 32;
const MIN_WIDTH = 380;
const PANEL_GAP = 6; // .mykee-xyz gap
// Room below the last axis box, so it doesn't touch the node's bottom edge.
const PANEL_BOTTOM_PADDING = 32;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function toast(detail, severity = "warn") {
    if (app.extensionManager?.toast?.add) {
        app.extensionManager.toast.add({ severity, summary: "Mykee XYZ Plot", detail, life: 6000 });
    } else {
        console.warn(`[Mykee XYZ Plot] ${detail}`);
    }
}

function getLink(graph, id) {
    if (!graph || id == null) return undefined;
    return graph.getLink?.(id) ?? graph.links?.get?.(id) ?? graph.links?.[id];
}

function readState(node) {
    const widget = node.widgets?.find((w) => w.name === CONFIG_WIDGET);
    let state = {};
    try {
        state = JSON.parse(widget?.value || "{}") || {};
    } catch {
        state = {};
    }
    state.axes ??= {};
    for (const axis of AXES) {
        state.axes[axis] ??= {};
        state.axes[axis].text ??= "";
        state.axes[axis].selected ??= [];
    }
    return state;
}

function writeState(node, state) {
    const widget = node.widgets?.find((w) => w.name === CONFIG_WIDGET);
    if (widget) widget.value = JSON.stringify({ version: 1, axes: state.axes });
}

function roundFloat(n) {
    return Math.round(n * 1e6) / 1e6;
}

// "5, 7.5, 10" / "4:12:2" (end inclusive, step defaults to 1) / newlines.
function parseNumbers(text, type) {
    const values = [];
    const errors = [];
    for (const raw of String(text).split(/[,;\n]/)) {
        const item = raw.trim();
        if (!item) continue;
        const range = item.split(":").map((s) => s.trim());
        if (range.length === 2 || range.length === 3) {
            const start = Number(range[0]);
            const end = Number(range[1]);
            let step = range.length === 3 ? Number(range[2]) : 1;
            if (![start, end, step].every(Number.isFinite) || step === 0) {
                errors.push(item);
                continue;
            }
            step = Math.abs(step) * (end >= start ? 1 : -1);
            const count = Math.floor(Math.abs(end - start) / Math.abs(step) + 1e-9) + 1;
            for (let i = 0; i < Math.min(count, 10000); i++) values.push(start + i * step);
        } else {
            const n = Number(item);
            if (Number.isFinite(n)) values.push(n);
            else errors.push(item);
        }
    }
    const cast = values.map((n) => (type === "INT" ? Math.round(n) : roundFloat(n)));
    return { values: cast, errors };
}

function comboOptions(widget, node) {
    const values = widget?.options?.values;
    const list = typeof values === "function" ? values(widget, node) : values;
    return Array.isArray(list) ? list.map(String) : [];
}

// ---------------------------------------------------------------------------
// What is an axis output connected to?
// ---------------------------------------------------------------------------

function describeInput(targetNode, slot) {
    const input = targetNode?.inputs?.[slot];
    if (!input) return null;
    const widgetName = input.widget?.name;
    const widget = widgetName ? targetNode.widgets?.find((w) => w.name === widgetName) : undefined;

    let type = input.type;
    if (Array.isArray(type) || widget?.type === "combo") type = "COMBO";
    type = String(type ?? "").toUpperCase();
    if (!SUPPORTED_TYPES.includes(type)) return { supported: false, type, name: input.name };
    if (type === "COMBO" && !widget) return { supported: false, type, name: input.name };

    const param = input.label || input.localized_name || input.name;
    return {
        supported: true,
        type,
        param,
        nodeTitle: targetNode.title || targetNode.type,
        label: `${param} (${targetNode.title || targetNode.type})`,
        options: type === "COMBO" ? comboOptions(widget, targetNode) : null,
        widgetOptions: widget?.options ?? {},
    };
}

// Links leaving an output slot. Newer frontends keep the real link list in a
// link store and output.links[] is only a mirror that can be stale right
// after a disconnect, so the graph's own link table is the source of truth.
function outputLinkList(node, slot, ignoreLinkId) {
    const graph = node.graph;
    if (!graph) return [];
    const table = graph.links;
    let links;
    if (table instanceof Map) links = [...table.values()];
    else if (table && typeof table === "object") links = Object.values(table);
    else links = (node.outputs?.[slot]?.links ?? []).map((id) => getLink(graph, id));
    return links
        .filter((l) => l && String(l.origin_id) === String(node.id) && l.origin_slot === slot && l.id !== ignoreLinkId)
        .sort((a, b) => a.id - b.id);
}

// All targets of an axis output; the first one defines the axis.
function axisTargets(node, slot, ignoreLinkId) {
    const result = [];
    for (const link of outputLinkList(node, slot, ignoreLinkId)) {
        const targetNode = node.graph?.getNodeById?.(link.target_id);
        const info = describeInput(targetNode, link.target_slot);
        if (info) result.push({ ...info, linkId: link.id, targetNode, targetSlot: link.target_slot });
    }
    return result;
}

function axisTarget(node, slot, ignoreLinkId) {
    const targets = axisTargets(node, slot, ignoreLinkId).filter((t) => t.supported);
    if (!targets.length) return null;
    const main = targets[0];
    const extra = targets.length - 1;
    return extra > 0 ? { ...main, label: `${main.label} +${extra}` } : main;
}

// Resolved value list of one axis (what goes into the prompt).
function axisValues(target, axisState) {
    if (!target) return { values: [], errors: [] };
    switch (target.type) {
        case "INT":
        case "FLOAT":
            return parseNumbers(axisState.text, target.type);
        case "STRING":
            return {
                values: String(axisState.text).split("\n").map((s) => s.trim()).filter(Boolean),
                errors: [],
            };
        case "BOOLEAN":
            return {
                values: [true, false].filter((b) => axisState.selected.includes(String(b))),
                errors: [],
            };
        case "COMBO": {
            // Keep the option list's order, drop entries that no longer exist.
            const chosen = new Set(axisState.selected);
            return { values: (target.options ?? []).filter((o) => chosen.has(o)), errors: [] };
        }
        default:
            return { values: [], errors: [] };
    }
}

// The config the backend gets (via serializeValue): only resolved values.
function resolveConfig(node) {
    const state = readState(node);
    const axes = AXES.map((axis, slot) => {
        const target = axisTarget(node, slot);
        if (!target) return null;
        const { values } = axisValues(target, state.axes[axis]);
        return { axis, type: target.type, param: target.param, label: target.label, values };
    });
    return { version: 1, axes };
}

// ---------------------------------------------------------------------------
// Queue fan-out: one prompt per cell
// ---------------------------------------------------------------------------

function isLinkTo(value, sourceKey) {
    return Array.isArray(value) && value.length === 2 && String(value[0]) === sourceKey;
}

function buildCellOutput(output, sourceKey, config, cell, dims, runId) {
    const [nx, ny] = dims;
    const index = [cell % nx, Math.floor(cell / nx) % ny, Math.floor(cell / (nx * ny))];
    const slotValues = {};
    config.axes.forEach((axis, slot) => {
        if (axis) slotValues[slot] = axis.values[index[slot]];
    });

    const cellOutput = { ...output };
    for (const [key, entry] of Object.entries(output)) {
        if (!entry?.inputs) continue;
        let inputs = null;
        for (const [name, value] of Object.entries(entry.inputs)) {
            if (isLinkTo(value, sourceKey) && value[1] in slotValues) {
                inputs ??= { ...entry.inputs };
                inputs[name] = slotValues[value[1]];
            }
        }
        if (inputs) cellOutput[key] = { ...entry, inputs };
    }
    const source = cellOutput[sourceKey];
    cellOutput[sourceKey] = {
        ...source,
        inputs: {
            ...source.inputs,
            [CONFIG_WIDGET]: JSON.stringify({ ...config, run_id: runId, cell, total: dims[0] * dims[1] * dims[2] }),
        },
    };
    return cellOutput;
}

function findPlotNode(output) {
    const keys = Object.keys(output ?? {}).filter((key) => output[key]?.class_type === NODE_NAME);
    if (keys.length > 1) {
        throw new Error("Mykee XYZ Plot: only one active XYZ Plot node is supported per workflow. Mute or bypass the others.");
    }
    return keys[0] ?? null;
}

function planFromPrompt(prompt) {
    const output = prompt?.output;
    const sourceKey = findPlotNode(output);
    if (!sourceKey) return null;

    let config;
    try {
        config = JSON.parse(output[sourceKey].inputs?.[CONFIG_WIDGET] ?? "{}");
    } catch {
        config = {};
    }
    if (!Array.isArray(config.axes)) config.axes = [null, null, null];

    // An axis output that is still linked in the prompt must have values,
    // otherwise its target would receive None.
    for (let slot = 0; slot < 3; slot++) {
        const used = Object.values(output).some((entry) =>
            Object.values(entry?.inputs ?? {}).some((v) => isLinkTo(v, sourceKey) && v[1] === slot)
        );
        if (used && !config.axes[slot]?.values?.length) {
            throw new Error(`Mykee XYZ Plot: the ${AXES[slot]} axis is connected but has no values selected.`);
        }
    }

    const dims = [0, 1, 2].map((slot) => config.axes[slot]?.values?.length || 1);
    return { sourceKey, config, dims, total: dims[0] * dims[1] * dims[2] };
}

const baseQueuePrompt = api.queuePrompt;
api.queuePrompt = async function (number, prompt, ...rest) {
    const plan = planFromPrompt(prompt);
    if (!plan) return baseQueuePrompt.call(this, number, prompt, ...rest);

    const { sourceKey, config, dims, total } = plan;
    if (total > CELL_CONFIRM_LIMIT) {
        const ok = window.confirm(`Mykee XYZ Plot: this will queue ${total} prompts (${dims.join(" × ")}). Continue?`);
        if (!ok) throw new Error("Mykee XYZ Plot: queueing cancelled.");
    }

    const runId = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
    let response;
    for (let cell = 0; cell < total; cell++) {
        const output = buildCellOutput(prompt.output, sourceKey, config, cell, dims, runId);
        response = await baseQueuePrompt.call(this, number, { ...prompt, output }, ...rest);
        const hasErrors = response?.node_errors && Object.keys(response.node_errors).length > 0;
        if (hasErrors || !response?.prompt_id) return response; // let ComfyUI show the error
    }
    return response;
};

// ---------------------------------------------------------------------------
// Panel UI
// ---------------------------------------------------------------------------

const PANEL_CSS = `
.mykee-xyz { display:flex; flex-direction:column; gap:6px; width:100%; height:100%;
  overflow-y:auto; box-sizing:border-box; padding:2px 4px 8px; font:12px sans-serif;
  color:var(--input-text, #ddd); }
.mykee-xyz-axis { border:1px solid var(--border-color, #444); border-radius:6px; padding:5px 6px;
  background:var(--comfy-input-bg, #222); display:flex; flex-direction:column; gap:4px; }
.mykee-xyz-head { display:flex; align-items:center; gap:6px; }
.mykee-xyz-badge { font-weight:bold; border-radius:4px; padding:0 6px; color:#111; }
.mykee-xyz-title { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.mykee-xyz-muted { opacity:.6; font-style:italic; }
.mykee-xyz textarea, .mykee-xyz input[type=text] { width:100%; box-sizing:border-box;
  background:var(--comfy-menu-bg, #111); color:inherit; border:1px solid var(--border-color, #444);
  border-radius:4px; padding:3px 5px; font:12px monospace; resize:vertical; }
.mykee-xyz-row { display:flex; gap:4px; align-items:center; }
.mykee-xyz button { background:var(--comfy-menu-bg, #333); color:inherit; border:1px solid var(--border-color, #555);
  border-radius:4px; padding:1px 7px; cursor:pointer; font-size:11px; }
.mykee-xyz-list { max-height:150px; overflow-y:auto; border:1px solid var(--border-color, #444);
  border-radius:4px; padding:2px 4px; }
.mykee-xyz-list label { display:flex; gap:5px; align-items:center; white-space:nowrap; cursor:pointer; }
.mykee-xyz-list label.hidden { display:none; }
.mykee-xyz-count { opacity:.75; font-size:11px; }
.mykee-xyz-error { color:#f77; font-size:11px; }
.mykee-xyz-total { font-weight:bold; padding:2px 2px 4px; }
`;
const BADGE_COLORS = { X: "#7fc8ff", Y: "#9be08f", Z: "#ffcf73" };

function ensureCss() {
    if (document.getElementById("mykee-xyz-css")) return;
    const style = document.createElement("style");
    style.id = "mykee-xyz-css";
    style.textContent = PANEL_CSS;
    document.head.appendChild(style);
}

function el(tag, props = {}, children = []) {
    const element = document.createElement(tag);
    Object.assign(element, props);
    for (const child of children) if (child) element.append(child);
    return element;
}

// Keeps canvas shortcuts / zoom out of the panel's inputs and lists.
function isolate(element) {
    for (const type of ["keydown", "keyup", "wheel"]) {
        element.addEventListener(type, (e) => e.stopPropagation());
    }
    return element;
}

function renderAxis(node, axis, slot, state, onChange) {
    const target = axisTarget(node, slot);
    const axisState = state.axes[axis];
    const box = el("div", { className: "mykee-xyz-axis" });

    const badge = el("span", { className: "mykee-xyz-badge", textContent: axis });
    badge.style.background = BADGE_COLORS[axis];
    const title = el("span", {
        className: "mykee-xyz-title" + (target ? "" : " mykee-xyz-muted"),
        textContent: target ? `${target.label} · ${target.type}` : `drag the ${OUTPUT_NAMES[slot]} output onto any widget`,
        title: target ? target.label : "",
    });
    const refresh = el("button", { textContent: "↻", title: "Re-read the connected parameter (e.g. after a model list refresh)" });
    refresh.onclick = () => node.mykeeXyzRender?.();
    box.append(el("div", { className: "mykee-xyz-head" }, [badge, title, target ? refresh : null]));
    if (!target) return { box, count: 1 };

    const count = el("div", { className: "mykee-xyz-count" });
    const error = el("div", { className: "mykee-xyz-error" });
    const updateCount = () => {
        const { values, errors } = axisValues(target, axisState);
        count.textContent = target.type === "COMBO"
            ? `${values.length} / ${target.options.length} selected`
            : `${values.length} value${values.length === 1 ? "" : "s"}` +
              (values.length && target.type !== "STRING" ? `: ${values.slice(0, 12).join(", ")}${values.length > 12 ? ", …" : ""}` : "");
        error.textContent = errors.length ? `Not understood: ${errors.join(", ")}` : "";
        return values.length;
    };

    if (target.type === "INT" || target.type === "FLOAT" || target.type === "STRING") {
        const area = isolate(el("textarea", {
            rows: target.type === "STRING" ? 4 : 2,
            value: axisState.text,
            placeholder: target.type === "STRING" ? "one value per line" : "e.g. 5, 7.5, 10   or   4:12:2 (start:end:step)",
        }));
        area.addEventListener("input", () => {
            axisState.text = area.value;
            onChange();
        });
        box.append(area);
        if (target.type === "INT") {
            const dice = el("button", {
                textContent: "🎲 random",
                title: "Fill with random values (as many as are listed now, or 4) - handy for a seed axis",
            });
            dice.onclick = () => {
                const n = parseNumbers(axisState.text, "INT").values.length || 4;
                const lo = Number.isFinite(target.widgetOptions.min) ? Math.max(0, target.widgetOptions.min) : 0;
                const hi = Number.isFinite(target.widgetOptions.max) ? Math.min(RANDOM_SEED_MAX, target.widgetOptions.max) : RANDOM_SEED_MAX;
                const list = Array.from({ length: n }, () => lo + Math.floor(Math.random() * (hi - lo + 1)));
                axisState.text = area.value = list.join(", ");
                onChange();
            };
            box.append(el("div", { className: "mykee-xyz-row" }, [dice]));
        }
    } else if (target.type === "BOOLEAN") {
        const row = el("div", { className: "mykee-xyz-row" });
        for (const value of ["true", "false"]) {
            const check = el("input", { type: "checkbox", checked: axisState.selected.includes(value) });
            check.onchange = () => {
                axisState.selected = axisState.selected.filter((v) => v !== value);
                if (check.checked) axisState.selected.push(value);
                onChange();
            };
            row.append(el("label", {}, [check, document.createTextNode(` ${value}  `)]));
        }
        box.append(row);
    } else if (target.type === "COMBO") {
        const filter = isolate(el("input", { type: "text", placeholder: "filter…" }));
        const list = isolate(el("div", { className: "mykee-xyz-list" }));
        const checks = [];
        for (const option of target.options) {
            const check = el("input", { type: "checkbox", checked: axisState.selected.includes(option) });
            check.onchange = () => {
                axisState.selected = axisState.selected.filter((v) => v !== option);
                if (check.checked) axisState.selected.push(option);
                onChange();
            };
            const label = el("label", { title: option }, [check, document.createTextNode(option)]);
            checks.push({ option, check, label });
            list.append(label);
        }
        const visible = () => checks.filter((c) => !c.label.classList.contains("hidden"));
        filter.addEventListener("input", () => {
            const q = filter.value.toLowerCase();
            for (const c of checks) c.label.classList.toggle("hidden", !!q && !c.option.toLowerCase().includes(q));
        });
        const setVisible = (on) => {
            for (const c of visible()) {
                c.check.checked = on;
                axisState.selected = axisState.selected.filter((v) => v !== c.option);
                if (on) axisState.selected.push(c.option);
            }
            onChange();
        };
        const all = el("button", { textContent: "all", title: "Select all (visible) entries" });
        const none = el("button", { textContent: "none", title: "Deselect all (visible) entries" });
        all.onclick = () => setVisible(true);
        none.onclick = () => setVisible(false);
        box.append(el("div", { className: "mykee-xyz-row" }, [filter, all, none]), list);
    }

    box.append(count, error);
    return { box, updateCount };
}

function setupPlotNode(node) {
    ensureCss();

    // Hide the raw JSON widget (the backend spec already asks for that on
    // newer frontends; this covers older ones).
    const configWidget = node.widgets?.find((w) => w.name === CONFIG_WIDGET);
    if (configWidget) {
        configWidget.hidden = true;
        configWidget.computeSize = () => [0, -4];
        // The prompt gets the resolved values, the workflow keeps the editor state.
        configWidget.serializeValue = () => JSON.stringify(resolveConfig(node));
    }

    const panel = el("div", { className: "mykee-xyz" });
    // Declared before addDOMWidget: getMinHeight may be called right away.
    let contentHeight = estimateHeight(node);
    const domWidget = node.addDOMWidget("xyz_panel", "MYKEE_XYZ_PANEL", panel, {
        serialize: false,
        getMinHeight: () => contentHeight,
    });
    domWidget.serialize = false;

    // The panel's height follows its content, and the node's height follows
    // the panel. First a rough estimate (the DOM may not be laid out yet),
    // then the real height, measured one frame later.
    const measure = () => {
        const kids = [...panel.children];
        const height = kids.reduce((sum, kid) => sum + kid.offsetHeight, 0);
        if (height > 0) contentHeight = height + PANEL_GAP * (kids.length - 1) + PANEL_BOTTOM_PADDING;
    };

    // shrink = false only grows: used when a saved workflow is loaded, so a
    // node the user made taller (e.g. for a longer combo list) keeps its size.
    node.mykeeXyzRender = (shrink = true) => {
        contentHeight = estimateHeight(node);
        const state = readState(node);
        panel.replaceChildren();
        const total = el("div", { className: "mykee-xyz-total" });
        const counters = [];
        const updateTotal = () => {
            const counts = counters.map((c) => (c ? c() : 1));
            const cells = counts.reduce((a, b) => a * Math.max(1, b), 1);
            total.textContent = `Plot: ${counts.join(" × ")} = ${cells} image${cells === 1 ? "" : "s"}`;
        };
        const onChange = () => {
            writeState(node, state);
            updateTotal();
            node.graph?.setDirtyCanvas?.(true, false);
        };
        panel.append(total);
        AXES.forEach((axis, slot) => {
            const { box, updateCount } = renderAxis(node, axis, slot, state, onChange);
            counters.push(updateCount ?? null);
            panel.append(box);
        });
        updateTotal();
        fitHeight(node, shrink);
        requestAnimationFrame(() => {
            measure();
            fitHeight(node, shrink);
        });
    };

    node.setSize([Math.max(node.size[0], MIN_WIDTH), node.size[1]]);
    node.mykeeXyzRender();
    // ComfyUI may still apply its own size after onNodeCreated - fit again.
    setTimeout(() => fitHeight(node, true), 0);
}

// Rough panel height per axis type, used until the real one is measured.
function estimateHeight(node) {
    let height = 26; // "Plot: ..." line
    for (let slot = 0; slot < 3; slot++) {
        const target = axisTarget(node, slot);
        let box = 34; // not connected: header only
        if (target?.type === "INT") box = 124;
        else if (target?.type === "FLOAT") box = 96;
        else if (target?.type === "STRING") box = 140;
        else if (target?.type === "BOOLEAN") box = 76;
        else if (target?.type === "COMBO") box = 96 + Math.min(150, (target.options?.length ?? 0) * 18 + 6);
        height += PANEL_GAP + box;
    }
    return height + PANEL_BOTTOM_PADDING;
}

// Fits the node's height to its content; the width stays the user's.
function fitHeight(node, shrink = true) {
    const size = node.computeSize?.();
    if (!size || !Number.isFinite(size[1])) return;
    const height = shrink ? size[1] : Math.max(node.size[1], size[1]);
    node.setSize([Math.max(node.size[0], size[0]), height]);
    node.graph?.setDirtyCanvas?.(true, true);
}

// Output slot follows the connected widget's type, like a Primitive node.
function updateOutputSlot(node, slot, ignoreLinkId) {
    const output = node.outputs?.[slot];
    if (!output) return;
    const target = axisTarget(node, slot, ignoreLinkId);
    output.type = target ? target.type : "*";
    output.label = target ? `${OUTPUT_NAMES[slot]}: ${target.param}` : OUTPUT_NAMES[slot];
}

function checkNewLink(node, slot, linkInfo) {
    const targetNode = node.graph?.getNodeById?.(linkInfo.target_id);
    const info = describeInput(targetNode, linkInfo.target_slot);
    const first = axisTargets(node, slot).find((t) => t.supported && t.linkId !== linkInfo.id);

    let reason = null;
    if (!info?.supported) {
        reason = `"${info?.name ?? "this input"}" can't be an axis - only INT, FLOAT, STRING, BOOLEAN and combo widgets can.`;
    } else if (first && first.type !== info.type) {
        reason = `The ${AXES[slot]} axis already drives a ${first.type} parameter; it can't also drive a ${info.type} one.`;
    } else if (first && info.type === "COMBO" && first.options.join("\n") !== info.options.join("\n")) {
        reason = `The ${AXES[slot]} axis already drives a combo with a different option list.`;
    }
    if (reason) {
        toast(reason);
        // Not during the event itself - the link is still being set up.
        setTimeout(() => targetNode?.disconnectInput?.(linkInfo.target_slot), 0);
        return false;
    }
    return true;
}

app.registerExtension({
    name: "Mykee.XYZPlot",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated?.apply(this, arguments);
            setupPlotNode(this);
            return ret;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const ret = onConfigure?.apply(this, arguments);
            // Target nodes may not exist yet while the graph is loading.
            setTimeout(() => {
                for (let slot = 0; slot < 3; slot++) updateOutputSlot(this, slot);
                this.mykeeXyzRender?.(false);
            }, 50);
            return ret;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, slot, connected, linkInfo) {
            const ret = onConnectionsChange?.apply(this, arguments);
            const OUTPUT = window.LiteGraph?.OUTPUT ?? 2;
            if (type !== OUTPUT || slot > 2) return ret;
            if (connected && linkInfo && !checkNewLink(this, slot, linkInfo)) return ret;
            if (!connected) {
                // Reset right away, ignoring the link being removed (it may
                // still be listed while the disconnect is in progress).
                updateOutputSlot(this, slot, linkInfo?.id);
            }
            // Defer: during a connect / disconnect the link list is not final yet.
            setTimeout(() => {
                updateOutputSlot(this, slot);
                this.mykeeXyzRender?.();
                this.graph?.setDirtyCanvas?.(true, true);
            }, 0);
            return ret;
        };
    },
});
