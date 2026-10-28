import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const NODE_NAME = "MykeeJsonBboxResizer";
const DISPLAY_NAME = "json_display";
const COPY_LABEL = "\u{1F4CB} Copy JSON";
const PLACEHOLDER = "(the resized JSON appears here after running)";

// Client-side only display of the result (same pattern as the Prompt
// Advanced nodes): not a backend input, so it never triggers a re-run. The
// last result is kept in node.properties so it survives a workflow tab
// switch.
const STORE_KEY = "mykee_display";

function rememberDisplay(node, name, value) {
    node.properties = node.properties || {};
    const store = node.properties[STORE_KEY] && typeof node.properties[STORE_KEY] === "object" ? node.properties[STORE_KEY] : {};
    store[name] = value;
    node.properties[STORE_KEY] = store;
}

function restoreDisplay(node, name) {
    const value = node.properties?.[STORE_KEY]?.[name];
    if (typeof value !== "string") return;
    const w = node.widgets?.find((w) => w.name === name);
    if (!w) return;
    w.value = value;
    if (w.inputEl) w.inputEl.value = value;
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
    name: "Mykee.JsonBboxResizer",
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
            const { widget: w } = ComfyWidgets["STRING"](
                this, DISPLAY_NAME, ["STRING", { multiline: true, default: "" }], app
            );
            w.serialize = false;
            if (w.inputEl) {
                w.inputEl.readOnly = true;
                w.inputEl.placeholder = PLACEHOLDER;
                w.inputEl.style.opacity = "0.75";
            }
            addCopyButton(this, DISPLAY_NAME, COPY_LABEL);
            return ret;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const ret = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            restoreDisplay(this, DISPLAY_NAME);
            return ret;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const values = message?.[DISPLAY_NAME];
            if (!Array.isArray(values) || !values.length) return;
            // A list of JSONs (an image batch from the captioner) runs the
            // node once per entry; show them all, separated by a blank line.
            const value = values.join("\n\n");
            rememberDisplay(this, DISPLAY_NAME, value);
            const w = this.widgets?.find((w) => w.name === DISPLAY_NAME);
            if (w) {
                w.value = value;
                if (w.inputEl) {
                    w.inputEl.value = value;
                    w.inputEl.readOnly = true;
                }
            }
            app.graph.setDirtyCanvas(true, true);
        };
    },
});
