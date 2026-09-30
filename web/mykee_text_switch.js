/*
 * Mykee Text Switch/Batch - frontend.
 *
 * Draws the numbered text fields, each with a radio button (switch mode) or
 * a checkbox (batch mode) at its end, plus "select all" / "clear all" in
 * batch mode. Everything is stored in the hidden texts_config widget:
 *
 *   { texts: [...], checked: [...], selected: <1-based index> }
 *
 * The texts are kept for all 49 fields, so lowering text_count and raising
 * it again brings the old texts back. The switch-mode selection and the
 * batch-mode ticks are stored separately, so flipping the mode loses
 * neither.
 *
 * Only the fields that are actually used go into the prompt (serializeValue),
 * so typing into a field that is not selected does not re-run anything.
 *
 * The text_1..text_49 input sockets are declared in Python; only the first
 * text_count of them are kept on the node (same approach as Mykee Image
 * Switch (Advanced)). A connected field is shown read-only.
 */
import { app } from "../../scripts/app.js";

const NODE_NAME = "MykeeTextSwitchBatch";
const CONFIG_WIDGET = "texts_config";
const MAX_TEXTS = 49;
const NAME_RE = /^text_(\d+)$/;
const ROW_HEIGHT = 62;
const TOOLBAR_HEIGHT = 30;

function widget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function isBatch(node) {
    return !!widget(node, "batch_mode")?.value;
}

function textCount(node) {
    const n = Math.round(Number(widget(node, "text_count")?.value ?? 2));
    return Number.isFinite(n) ? Math.max(0, Math.min(MAX_TEXTS, n)) : 2;
}

function readState(node) {
    let state = {};
    try {
        state = JSON.parse(widget(node, CONFIG_WIDGET)?.value || "{}") || {};
    } catch {
        state = {};
    }
    const texts = Array.isArray(state.texts) ? state.texts.map((t) => String(t ?? "")) : [];
    const checked = Array.isArray(state.checked) ? state.checked.map(Boolean) : [];
    while (texts.length < MAX_TEXTS) texts.push("");
    while (checked.length < MAX_TEXTS) checked.push(false);
    const selected = Number.isInteger(state.selected) ? state.selected : 1;
    return { texts: texts.slice(0, MAX_TEXTS), checked: checked.slice(0, MAX_TEXTS), selected };
}

function writeState(node, state) {
    const w = widget(node, CONFIG_WIDGET);
    if (!w) return;
    // Trailing empty / unticked fields are not worth saving.
    let last = MAX_TEXTS;
    while (last > 0 && !state.texts[last - 1] && !state.checked[last - 1]) last--;
    w.value = JSON.stringify({
        texts: state.texts.slice(0, last),
        checked: state.checked.slice(0, last),
        selected: state.selected,
    });
}

// Is text_N connected? Uses the graph's link table: newer frontends only
// mirror links in input.link, and the mirror can lag behind.
function connectedFields(node) {
    const result = new Set();
    const graph = node.graph;
    const table = graph?.links;
    const links = table instanceof Map ? [...table.values()] : table ? Object.values(table) : [];
    for (const link of links) {
        if (!link || String(link.target_id) !== String(node.id)) continue;
        const match = node.inputs?.[link.target_slot]?.name?.match(NAME_RE);
        if (match) result.add(Number(match[1]));
    }
    return result;
}

// What the backend gets: only the fields that are used.
function resolveConfig(node) {
    const state = readState(node);
    const count = textCount(node);
    const connected = connectedFields(node);
    const item = (index) => ({ index, text: connected.has(index) ? "" : state.texts[index - 1] });
    let items = [];
    if (isBatch(node)) {
        for (let i = 1; i <= count; i++) {
            const used = state.checked[i - 1] && (connected.has(i) || state.texts[i - 1].trim());
            if (used) items.push(item(i));
        }
    } else if (state.selected >= 1 && state.selected <= count) {
        items = [item(state.selected)];
    }
    return JSON.stringify({ items });
}

// ---------------------------------------------------------------------------
// Input sockets: keep text_1..text_count
// ---------------------------------------------------------------------------

function syncSockets(node) {
    const count = textCount(node);
    for (let idx = (node.inputs?.length ?? 0) - 1; idx >= 0; idx--) {
        const match = node.inputs[idx].name.match(NAME_RE);
        if (match && Number(match[1]) > count) node.removeInput(idx);
    }
    const present = new Set(
        (node.inputs ?? []).map((inp) => inp.name.match(NAME_RE)?.[1]).filter(Boolean).map(Number)
    );
    for (let i = 1; i <= count; i++) {
        if (!present.has(i)) node.addInput(`text_${i}`, "STRING");
    }
}

// ---------------------------------------------------------------------------
// Panel
// ---------------------------------------------------------------------------

const PANEL_CSS = `
.mykee-ts { display:flex; flex-direction:column; gap:4px; width:100%; height:100%; box-sizing:border-box;
  overflow-y:auto; padding:2px 4px; font:12px sans-serif; color:var(--input-text, #ddd); }
.mykee-ts-bar { display:flex; gap:4px; }
.mykee-ts-bar button { background:var(--comfy-menu-bg, #333); color:inherit; border:1px solid var(--border-color, #555);
  border-radius:4px; padding:2px 8px; cursor:pointer; font-size:11px; }
.mykee-ts-row { display:flex; align-items:stretch; gap:5px; }
.mykee-ts-num { min-width:20px; text-align:right; padding-top:4px; font-weight:bold; opacity:.8; }
.mykee-ts-row textarea { flex:1; min-height:48px; box-sizing:border-box; resize:vertical;
  background:var(--comfy-input-bg, #222); color:inherit; border:1px solid var(--border-color, #444);
  border-radius:4px; padding:3px 5px; font:12px sans-serif; }
.mykee-ts-row textarea:disabled { opacity:.55; font-style:italic; }
.mykee-ts-row.off textarea { opacity:.5; }
.mykee-ts-pick { display:flex; align-items:center; padding:0 2px; }
.mykee-ts-pick input { width:16px; height:16px; cursor:pointer; margin:0; }
.mykee-ts-empty { opacity:.6; font-style:italic; padding:4px; }
`;

function ensureCss() {
    if (document.getElementById("mykee-ts-css")) return;
    const style = document.createElement("style");
    style.id = "mykee-ts-css";
    style.textContent = PANEL_CSS;
    document.head.appendChild(style);
}

function isolate(element) {
    for (const type of ["keydown", "keyup", "wheel"]) {
        element.addEventListener(type, (e) => e.stopPropagation());
    }
    return element;
}

// The node's height always follows its content (grows AND shrinks). The
// initial size is computed by ComfyUI while all 49 declared text_N sockets
// still exist, so without the shrinking a new node starts far too tall.
function fitHeight(node) {
    const size = node.computeSize?.();
    if (!size || !Number.isFinite(size[1])) return;
    node.setSize([Math.max(node.size[0], size[0]), size[1]]);
    node.graph?.setDirtyCanvas?.(true, true);
}

function setupNode(node) {
    ensureCss();
    const radioName = `mykee-ts-${Math.random().toString(36).slice(2)}`;

    const configWidget = widget(node, CONFIG_WIDGET);
    if (configWidget) {
        configWidget.hidden = true;
        configWidget.computeSize = () => [0, -4];
        configWidget.serializeValue = () => resolveConfig(node);
    }

    const panel = document.createElement("div");
    panel.className = "mykee-ts";
    const dom = node.addDOMWidget("texts_panel", "MYKEE_TEXT_PANEL", panel, {
        serialize: false,
        getMinHeight: () => (isBatch(node) ? TOOLBAR_HEIGHT : 0) + Math.max(1, textCount(node)) * ROW_HEIGHT + 8,
    });
    dom.serialize = false;

    const render = () => {
        const state = readState(node);
        const count = textCount(node);
        const batch = isBatch(node);
        const connected = connectedFields(node);
        const save = () => writeState(node, state);
        panel.replaceChildren();

        if (batch && count > 0) {
            const bar = document.createElement("div");
            bar.className = "mykee-ts-bar";
            for (const [label, value] of [["select all", true], ["clear all", false]]) {
                const button = document.createElement("button");
                button.textContent = label;
                button.onclick = () => {
                    for (let i = 0; i < count; i++) state.checked[i] = value;
                    save();
                    render();
                };
                bar.append(button);
            }
            panel.append(bar);
        }
        if (count === 0) {
            const empty = document.createElement("div");
            empty.className = "mykee-ts-empty";
            empty.textContent = "No text fields (text_count = 0).";
            panel.append(empty);
        }

        for (let i = 1; i <= count; i++) {
            const row = document.createElement("div");
            row.className = "mykee-ts-row";

            const num = document.createElement("div");
            num.className = "mykee-ts-num";
            num.textContent = String(i);

            const area = isolate(document.createElement("textarea"));
            if (connected.has(i)) {
                area.disabled = true;
                area.value = "";
                area.placeholder = `← from input text_${i}`;
            } else {
                area.value = state.texts[i - 1];
                area.placeholder = `text ${i}`;
                area.addEventListener("input", () => {
                    state.texts[i - 1] = area.value;
                    save();
                });
            }

            const pick = document.createElement("label");
            pick.className = "mykee-ts-pick";
            const input = document.createElement("input");
            if (batch) {
                input.type = "checkbox";
                input.checked = state.checked[i - 1];
                input.title = "Include this text";
                input.onchange = () => {
                    state.checked[i - 1] = input.checked;
                    row.classList.toggle("off", !input.checked);
                    save();
                };
                row.classList.toggle("off", !input.checked);
            } else {
                input.type = "radio";
                input.name = radioName;
                input.checked = state.selected === i;
                input.title = "Use this text";
                input.onchange = () => {
                    if (!input.checked) return;
                    state.selected = i;
                    save();
                    panel.querySelectorAll(".mykee-ts-row").forEach((r, n) => r.classList.toggle("off", n + 1 !== i));
                };
                row.classList.toggle("off", state.selected !== i);
            }
            pick.append(input);
            row.append(num, area, pick);
            panel.append(row);
        }
        fitHeight(node);
    };
    node.mykeeTextRender = render;

    // Re-render when the mode or the field count changes.
    for (const name of ["batch_mode", "text_count"]) {
        const w = widget(node, name);
        if (!w) continue;
        const callback = w.callback;
        w.callback = function () {
            const ret = callback?.apply(this, arguments);
            if (name === "text_count") syncSockets(node);
            render();
            return ret;
        };
    }

    syncSockets(node);
    node.setSize([Math.max(node.size[0], 340), node.size[1]]);
    render();
    // ComfyUI may still apply its own size after onNodeCreated - fit again.
    setTimeout(() => fitHeight(node), 0);
}

app.registerExtension({
    name: "Mykee.TextSwitchBatch",
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
            setTimeout(() => {
                syncSockets(this);
                this.mykeeTextRender?.();
            }, 50);
            return ret;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type) {
            const ret = onConnectionsChange?.apply(this, arguments);
            const INPUT = window.LiteGraph?.INPUT ?? 1;
            if (type === INPUT) setTimeout(() => this.mykeeTextRender?.(), 0);
            return ret;
        };
    },
});
