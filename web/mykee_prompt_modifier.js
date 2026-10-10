import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const NODE_NAME = "MykeePromptModifier";
const DISPLAY_NAME = "response_display";


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
    name: "Mykee.PromptModifier",
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

            // response_display is intentionally NOT declared as a backend
            // input in mykee_prompt_modifier.py (see the note there) - it's
            // created here, purely client-side, exactly like ComfyUI's own
            // PreviewImage node has no "images" input yet still shows a
            // widget. Setting w.serialize = false on a widget that DOES
            // exist server-side turned out not to be a reliable enough fix
            // on its own (still forced a rerun every queue on this
            // person's setup); a widget that was never a real input in the
            // first place sidesteps the question entirely - there's
            // nothing for the cache key to ever pick up.
            const { widget: w } = ComfyWidgets["STRING"](
                this, DISPLAY_NAME, ["STRING", { multiline: true, default: "" }], app
            );
            w.serialize = false;
            if (w.inputEl) {
                w.inputEl.readOnly = true;
                w.inputEl.placeholder = "(filled in after running)";
                w.inputEl.style.opacity = "0.75";
            }
            addCopyButton(this, DISPLAY_NAME, "📋 Copy text");

            return ret;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const ret = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            // properties are restored by now; the widgets exist since onNodeCreated.
            restoreDisplay(this, [DISPLAY_NAME]);
            return ret;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const value = message?.[DISPLAY_NAME]?.[0];
            if (value === undefined) return;
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
