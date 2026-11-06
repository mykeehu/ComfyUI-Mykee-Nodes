/*
 * Mykee Text Analyzer - frontend.
 *
 * - Read-only display boxes (notes, answer, status, thinking) with copy
 *   buttons, filled from the "ui" dict after each run. They are client-side
 *   only widgets (never serialized, so they never change the cache key); the
 *   last texts are also kept in node.properties so a workflow-tab switch does
 *   not empty them. Same mechanism as Mykee Prompt Advanced Ultimate.
 * - The second stage's seed control is sent back through the same channel.
 * - While the text_in socket is connected, the "text" box is hidden and the
 *   node shrinks by its height; disconnecting brings it back.
 * - The settings sit on four tabs (Notes / Answer / Advanced / Outputs) under
 *   the always-visible fields (clip, question, text, note_mode, seeds). A tab
 *   that is not shown has its widgets hidden, which keeps the node short. The
 *   widgets stay real widgets (values saved as always, sockets and links kept).
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const NODE_NAME = "MykeeTextAnalyzer";
const DISPLAY_WIDGET_NAMES = [
    "response_1_display", "response_2_display", "status_display", "thinking_1_display", "thinking_2_display",
];
const COPY_LABELS = {
    response_2_display: "📋 Copy answer (response 2)",
    response_1_display: "📋 Copy notes (response 1)",
    status_display: "📋 Copy status",
    thinking_1_display: "📋 Copy thinking 1",
    thinking_2_display: "📋 Copy thinking 2",
};
const PLACEHOLDERS = {
    response_2_display: "(the final answer - filled in after running)",
    response_1_display: "(the notes collected from the chunks - filled in after running)",
    status_display: "(how the run went - filled in after running)",
    thinking_1_display: "(thinking of round 1 - filled in after running with thinking on)",
    thinking_2_display: "(thinking of round 2 - filled in after running with thinking on)",
};
const SEED_SYNC_KEY = "second_seed";
const STORE_KEY = "mykee_display";
const TEXT_WIDGET = "text";
const TEXT_SOCKET = "text_in";

function findWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

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
        const w = findWidget(node, name);
        if (!w) continue;
        w.value = value;
        if (w.inputEl) w.inputEl.value = value;
    }
}

// ---- show / hide a widget (incl. its DOM textarea) -------------------------
function setWidgetVisible(widget, visible) {
    if (!widget) return;
    if (!widget._mykeeOrig || widget.hidden !== true) {
        widget._mykeeOrig = {
            type: widget.type,
            hadComputeSize: Object.prototype.hasOwnProperty.call(widget, "computeSize"),
            computeSize: widget.computeSize,
            hadLayoutSize: Object.prototype.hasOwnProperty.call(widget, "computeLayoutSize"),
            computeLayoutSize: widget.computeLayoutSize,
        };
    }
    const orig = widget._mykeeOrig;
    if (visible) {
        // type first: newer frontends ignore writes to widget.hidden while the type is "hidden"
        widget.type = orig.type;
        widget.hidden = false;
        if (orig.hadComputeSize) widget.computeSize = orig.computeSize;
        else delete widget.computeSize;
        if (orig.hadLayoutSize) widget.computeLayoutSize = orig.computeLayoutSize;
        else delete widget.computeLayoutSize;
    } else {
        widget.hidden = true;
        widget.type = "hidden";
        widget.computeSize = () => [0, -4];
        widget.computeLayoutSize = () => ({ minHeight: 0, maxHeight: 0, minWidth: 0 });
    }
    const el = widget.element || widget.inputEl;
    if (el) {
        el.hidden = !visible;
        el.style.display = visible ? "" : "none";
    }
}

function textSocketConnected(node) {
    const input = node.inputs?.find((i) => i.name === TEXT_SOCKET);
    return !!input && input.link != null;
}

// ---- tabs ---------------------------------------------------------------------
const TABS = [
    { id: "notes", label: "Notes", title: "Round 1: how the notes are written" },
    { id: "answer", label: "Answer", title: "Round 2: how the final answer is written" },
    { id: "advanced", label: "Advanced", title: "Limits and other settings" },
    { id: "outputs", label: "Outputs", title: "Notes, answer, status and thinking" },
];
const TAB_PROP = "mykee_tab";
const TAB_SIZE_PROP = "mykee_tab_heights"; // node height per tab, saved with the workflow
const DEFAULT_TAB = "outputs";
const TAB_BAR_NAME = "_mykee_tabs";
const TAB_BAR_HEIGHT = 30;
const ALWAYS_VISIBLE = ["question", "text", "note_mode", "seed", "second_seed", "second_seed_control"];
const TAB_OF = {};
for (const n of ["max_length", "sampling_mode", "temperature", "top_k", "top_p", "min_p",
    "repetition_penalty", "presence_penalty", "thinking", "thinking_budget", "notes_system_prompt"]) TAB_OF[n] = "notes";
for (const n of ["second_system_prompt", "second_max_length", "second_sampling_mode", "second_temperature",
    "second_top_k", "second_top_p", "second_min_p", "second_repetition_penalty",
    "second_presence_penalty", "second_thinking", "second_thinking_budget"]) TAB_OF[n] = "answer";
for (const n of ["mode", "notes_max_tokens", "input_token_limit", "unload_after_run", "template_mode"]) TAB_OF[n] = "advanced";
for (const n of DISPLAY_WIDGET_NAMES) TAB_OF[n] = "outputs";

function currentTab(node) {
    const t = node.properties?.[TAB_PROP];
    return TABS.some((x) => x.id === t) ? t : DEFAULT_TAB;
}

// Which tab a widget belongs to: "always" = shown above the tabs. A widget the
// table does not know (e.g. the control_after_generate combo the frontend adds
// behind a seed) goes with the widget before it.
function tabAssignments(node) {
    const out = new Map();
    let last = "always";
    for (const w of node.widgets ?? []) {
        if (w.name === TAB_BAR_NAME) continue;
        let tab;
        if (TAB_OF[w.name]) tab = TAB_OF[w.name];
        else if (w.type === "button" || w.name?.endsWith("_display")) tab = "outputs";
        else if (ALWAYS_VISIBLE.includes(w.name)) tab = "always";
        else tab = last;
        out.set(w, tab);
        last = tab;
    }
    return out;
}

function applyTab(node, resize) {
    const current = currentTab(node);
    const hideText = textSocketConnected(node);
    for (const [w, tab] of tabAssignments(node)) {
        let visible = tab === "always" || tab === current;
        if (w.name === TEXT_WIDGET && hideText) visible = false;
        if ((w.hidden === true) === !visible) continue;
        setWidgetVisible(w, visible);
    }
    applyStoredHeights(node, true);
    if (resize) node.setSize([node.size[0], node.computeSize()[1]]);
    node.setDirtyCanvas?.(true, true);
}

function syncTextVisibility(node, resize) {
    applyTab(node, resize);
}

function createTabBar(node) {
    const margin = 15;
    const gap = 4;
    const h = 24;
    const widget = {
        name: TAB_BAR_NAME,
        type: "mykee_tabs",
        value: "",
        serialize: false,
        options: { serialize: false },
        computeSize: (width) => [width ?? 0, TAB_BAR_HEIGHT],
        draw(ctx, n, width, y) {
            const bw = (width - 2 * margin - gap * (TABS.length - 1)) / TABS.length;
            const current = currentTab(n);
            ctx.save();
            ctx.font = "12px sans-serif";
            ctx.textAlign = "center";
            TABS.forEach((tab, i) => {
                const x = margin + i * (bw + gap);
                const active = tab.id === current;
                ctx.fillStyle = active ? "#4a6fa5" : "#2b2b2b";
                ctx.strokeStyle = active ? "#7aa2d6" : "#555";
                ctx.beginPath();
                if (ctx.roundRect) ctx.roundRect(x, y + 3, bw, h, 5);
                else ctx.rect(x, y + 3, bw, h);
                ctx.fill();
                ctx.stroke();
                ctx.fillStyle = active ? "#fff" : "#bbb";
                ctx.fillText(tab.label, x + bw / 2, y + 3 + h / 2 + 4);
            });
            ctx.restore();
        },
        mouse(event, pos, n) {
            if (event.type !== "pointerdown" && event.type !== "mousedown") return false;
            const bw = (n.size[0] - 2 * margin - gap * (TABS.length - 1)) / TABS.length;
            const i = Math.floor((pos[0] - margin) / (bw + gap));
            if (i < 0 || i >= TABS.length || pos[0] - margin - i * (bw + gap) > bw) return false;
            n.properties = n.properties || {};
            const heights = { ...(n.properties[TAB_SIZE_PROP] || {}) };
            heights[currentTab(n)] = n.size[1]; // remember the height of the tab we leave
            n.properties[TAB_SIZE_PROP] = heights;
            n.properties[TAB_PROP] = TABS[i].id;
            applyTab(n, true);
            const wanted = Number(heights[TABS[i].id]);
            if (Number.isFinite(wanted)) n.setSize([n.size[0], Math.max(wanted, n.computeSize()[1])]);
            return true;
        },
    };
    node.widgets = node.widgets || [];
    node.widgets.push(widget);
    // right under the always-visible fields, above every tab's widgets
    const anchor = node.widgets.findIndex((w) => w.name === "second_seed_control");
    if (anchor >= 0) {
        node.widgets.splice(node.widgets.indexOf(widget), 1);
        node.widgets.splice(anchor + 1, 0, widget);
    }
    return widget;
}

// ---- resizable display boxes -----------------------------------------------
// Every box has a drag handle (bottom-right corner). A box nobody touched takes
// its share of the node's height (at least DEFAULT_BOX_HEIGHT); a box that was
// dragged keeps the height it was given, and the node grows or shrinks by the
// difference. The heights are saved with the workflow (node.properties).
const HEIGHTS_KEY = "mykee_box_heights";
const DEFAULT_BOX_HEIGHT = 110;
const MIN_BOX_HEIGHT = 50;
const HANDLE_SIZE = 22;
const BOX_CHROME = 16; // widget row height minus the textarea inside it (measured)

function storedHeights(node) {
    const h = node.properties?.[HEIGHTS_KEY];
    return h && typeof h === "object" ? h : {};
}

function applyBoxHeight(w, height, fixed) {
    w.computeSize = (width) => [width ?? 0, height];
    w.computeLayoutSize = () => (fixed
        ? { minHeight: height, maxHeight: height, minWidth: 0 }
        : { minHeight: height, minWidth: 0 });
}

function applyStoredHeights(node, onlyVisible) {
    const stored = storedHeights(node);
    for (const name of DISPLAY_WIDGET_NAMES) {
        const w = findWidget(node, name);
        if (!w) continue;
        if (onlyVisible && w.hidden === true) continue;
        const h = Number(stored[name]);
        if (Number.isFinite(h) && h >= MIN_BOX_HEIGHT) applyBoxHeight(w, h, true);
        else applyBoxHeight(w, DEFAULT_BOX_HEIGHT, false);
    }
}

function enableBoxResize(node, w) {
    const el = w.inputEl;
    if (!el) return;
    el.style.resize = "vertical";
    el.addEventListener("pointerdown", (ev) => {
        const r = el.getBoundingClientRect();
        if (ev.clientX < r.right - HANDLE_SIZE || ev.clientY < r.bottom - HANDLE_SIZE) return;
        const startH = el.offsetHeight;
        const finish = () => {
            window.removeEventListener("pointerup", finish);
            window.removeEventListener("pointercancel", finish);
            const newH = Math.max(MIN_BOX_HEIGHT, Math.round(el.offsetHeight));
            el.style.height = "";
            if (Math.abs(newH - startH) < 4) return;
            node.properties = node.properties || {};
            const stored = { ...storedHeights(node), [w.name]: newH + BOX_CHROME };
            node.properties[HEIGHTS_KEY] = stored;
            applyBoxHeight(w, newH + BOX_CHROME, true);
            node.setSize([node.size[0], Math.max(node.computeSize()[1], node.size[1] + (newH - startH))]);
            node.setDirtyCanvas?.(true, true);
        };
        window.addEventListener("pointerup", finish);
        window.addEventListener("pointercancel", finish);
    });
}

// ---- Copy button -----------------------------------------------------------
async function copyToClipboard(text) {
    try {
        if (navigator.clipboard?.writeText) {
            await navigator.clipboard.writeText(text);
            return true;
        }
    } catch {
        /* fall through to the legacy path */
    }
    const area = document.createElement("textarea");
    area.value = text;
    area.style.cssText = "position:fixed;left:-9999px;top:0;opacity:0;";
    document.body.appendChild(area);
    area.select();
    let ok = false;
    try {
        ok = document.execCommand("copy");
    } catch {
        ok = false;
    }
    area.remove();
    return ok;
}

function copyToast(severity, detail) {
    if (app.extensionManager?.toast?.add) {
        app.extensionManager.toast.add({ severity, summary: "Mykee", detail, life: 2500 });
    }
}

function addCopyButton(node, displayName, label) {
    const btn = node.addWidget("button", label, null, async () => {
        const display = findWidget(node, displayName);
        const text = String(display?.inputEl?.value ?? display?.value ?? "");
        if (!text.trim()) {
            copyToast("info", "Nothing to copy yet - run the node first.");
            return;
        }
        const ok = await copyToClipboard(text);
        copyToast(ok ? "success" : "error", ok ? "Text copied to the clipboard." : "Copy failed - select the text and copy it manually.");
        if (ok) {
            btn.label = "✅ Copied!";
            node.setDirtyCanvas?.(true, true);
            setTimeout(() => {
                btn.label = label;
                node.setDirtyCanvas?.(true, true);
            }, 1200);
        }
    });
    btn.serialize = false;
    btn.options = btn.options || {};
    btn.options.serialize = false;
    return btn;
}

// pointing hand over the node's buttons (the canvas only gives a crosshair)
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
            if ((w.type !== "button" && w.type !== "mykee_tabs") || typeof w.last_y !== "number") continue;
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

// ---- live view ---------------------------------------------------------------
// While the node runs, Python sends the notes / thinking / status collected so
// far after every chunk ("mykee.text_analyzer.progress"). They go into the same
// display boxes the final result goes into.
function findNodeById(id) {
    const graph = app.graph;
    if (!graph) return null;
    const direct = graph.getNodeById?.(Number(id));
    if (direct) return direct;
    const last = String(id).split(":").pop(); // node inside a subgraph: "outer:inner"
    return graph.getNodeById?.(Number(last)) ?? null;
}

function setDisplay(node, name, value) {
    if (typeof value !== "string") return;
    const w = findWidget(node, name);
    if (!w) return;
    rememberDisplay(node, name, value);
    w.value = value;
    if (w.inputEl) {
        const stick = w.inputEl.scrollTop + w.inputEl.clientHeight >= w.inputEl.scrollHeight - 8;
        w.inputEl.value = value;
        if (stick) w.inputEl.scrollTop = w.inputEl.scrollHeight; // follow the newest text
    }
}

api.addEventListener("mykee.text_analyzer.progress", (event) => {
    const d = event.detail || {};
    const node = findNodeById(d.node);
    if (!node || (node.comfyClass ?? node.type) !== NODE_NAME) return;
    setDisplay(node, "response_1_display", d.response_1);
    setDisplay(node, "thinking_1_display", d.thinking_1);
    setDisplay(node, "status_display", d.status);
    node.setDirtyCanvas?.(true, true);
});

app.registerExtension({
    name: "Mykee.TextAnalyzer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_NAME) return;

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function () {
            const ret = onDrawForeground?.apply(this, arguments);
            updateButtonCursor(this);
            return ret;
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            createTabBar(this);
            for (const name of DISPLAY_WIDGET_NAMES) {
                const { widget: w } = ComfyWidgets["STRING"](
                    this, name, ["STRING", { multiline: true, default: "" }], app
                );
                // client-side only: never part of the saved widgets_values or of the prompt
                w.serialize = false;
                w.options = w.options || {};
                w.options.serialize = false;
                if (w.inputEl) {
                    w.inputEl.readOnly = true;
                    w.inputEl.placeholder = PLACEHOLDERS[name];
                    w.inputEl.style.opacity = "0.75";
                }
                enableBoxResize(this, w);
                addCopyButton(this, name, COPY_LABELS[name]);
            }
            applyStoredHeights(this);
            applyTab(this, true);
            return ret;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const ret = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            restoreDisplay(this, DISPLAY_WIDGET_NAMES);
            applyStoredHeights(this, true);
            syncTextVisibility(this, false); // the saved size already accounts for it
            return ret;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo, ioSlot) {
            const ret = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
            const LG = window.LiteGraph;
            if ((LG ? type === LG.INPUT : type === 1) && this.inputs?.[index]?.name === TEXT_SOCKET) {
                syncTextVisibility(this, !app.configuringGraph);
            }
            return ret;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            for (const name of DISPLAY_WIDGET_NAMES) {
                const value = message?.[name]?.[0];
                if (value === undefined) continue;
                rememberDisplay(this, name, value);
                const w = findWidget(this, name);
                if (!w) continue;
                w.value = value;
                if (w.inputEl) {
                    w.inputEl.value = value;
                    w.inputEl.readOnly = true;
                }
            }
            const nextSeed = message?.[SEED_SYNC_KEY]?.[0];
            if (nextSeed !== undefined) {
                const seedWidget = findWidget(this, SEED_SYNC_KEY);
                if (seedWidget) {
                    seedWidget.value = nextSeed;
                    if (seedWidget.callback) seedWidget.callback(nextSeed);
                }
            }
            app.graph.setDirtyCanvas(true, true);
        };
    },
});
