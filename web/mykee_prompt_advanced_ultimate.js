import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const NODE_NAME = "MykeePromptAdvancedUltimate";
const DISPLAY_WIDGET_NAMES = ["response_1_display", "response_2_display", "tags_1_display"];
const COPY_LABELS = {
    response_1_display: "📋 Copy response 1",
    response_2_display: "📋 Copy response 2",
    tags_1_display: "📋 Copy tags 1",
};

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


// ---- Copy button -----------------------------------------------------------
// A button under a display widget that puts its text on the clipboard.
// navigator.clipboard only exists in secure contexts (https / localhost), so
// ComfyUI opened over a plain-http LAN address falls back to execCommand.
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
        const display = node.widgets?.find((w) => w.name === displayName);
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
    return btn;
}


// Plain litegraph buttons only get the canvas' crosshair - show the pointing
// hand while the mouse is over one of this node's buttons.
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
        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function () {
            const ret = onDrawForeground?.apply(this, arguments);
            updateButtonCursor(this);
            return ret;
        };

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
                addCopyButton(this, name, COPY_LABELS[name]);
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
