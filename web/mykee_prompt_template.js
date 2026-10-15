import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * Mykee Prompt Template.
 *
 * "positive_prompt_text" and "negative_prompt_text" are real, serialized
 * multiline STRING widgets - together they ARE the node's two outputs,
 * always. The "Existing templates" picker, "New", "Save" and "Reload" move
 * both of them to and from a one-file-per-template JSON store (default:
 * ComfyUI/user/default/Prompt templates, or "custom_path" if that widget
 * is filled in):
 *
 *   - Reload re-fetches the list of templates currently on disk (at
 *     whatever "custom_path" is right now) into the "Existing templates"
 *     picker - no restart needed after adding/renaming a template file by
 *     hand, or after changing custom_path.
 *   - Picking a template from "Existing templates" loads it immediately -
 *     overwrites BOTH prompt widgets (older files saved before negative
 *     prompts existed load fine too - negative_prompt_text just comes
 *     back empty). There's no separate Load button.
 *   - New clears template_name and both prompt boxes, for starting a
 *     fresh template from blank rather than editing whatever was last
 *     loaded/typed.
 *   - Save writes both widgets out together under template_name
 *     (sanitized server-side so the resulting file name is valid on both
 *     Windows and Linux - the widget is updated to match whatever name
 *     actually got used, in case it had to change).
 *
 * Unlike the StyleGAN Vector Save node, none of this needs the graph to
 * run - there's no "armed" gate here, Save just fires the REST call
 * immediately.
 *
 * Save/New/Reload are plain built-in litegraph "button" widgets, which
 * (in this ComfyUI frontend build) don't get the pointing-hand cursor
 * litegraph normally gives them on hover - only a crosshair. Fixed below
 * via a small onDrawForeground hook that checks the mouse against each
 * button widget's own row (using the last_y litegraph already records for
 * every widget after drawing it) and sets the canvas cursor by hand.
 */

function findWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

async function fetchJSON(url, options) {
    const res = await api.fetchApi(url, options);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    return data;
}

const MIN_NODE_WIDTH = 320;

function enforceMinWidth(node) {
    const computed = node.computeSize();
    node.setSize([Math.max(computed[0], MIN_NODE_WIDTH), computed[1]]);
}

const NODE_NAME = "MykeePromptTemplate";

// litegraph records each widget's drawn row position in widget.last_y
// after every draw pass - used here (rather than a custom widget) to
// detect hovering over one of the plain "button" widgets below, so the
// canvas cursor can be switched to a pointing hand while over them.
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
    name: "Mykee.PromptTemplate",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_NAME) return;

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            const ret = onDrawForeground ? onDrawForeground.apply(this, arguments) : undefined;
            updateButtonCursor(this);
            return ret;
        };

        // Switching to another workflow tab and back rebuilds the graph: each
        // node is created first (onNodeCreated, with DEFAULT widget values -
        // so custom_path is still empty and the list is read from the default
        // folder) and only THEN configured with the saved widget values. The
        // list therefore has to be re-read here, after custom_path has been
        // restored, otherwise it stays empty / from the wrong folder until
        // Reload is pressed by hand.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const ret = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            if (this._mykeeRefreshTemplateList) this._mykeeRefreshTemplateList();
            return ret;
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const nameWidget = findWidget(this, "template_name");
            const positiveWidget = findWidget(this, "positive_prompt_text");
            const negativeWidget = findWidget(this, "negative_prompt_text");
            const pathWidget = findWidget(this, "custom_path");
            if (!nameWidget || !positiveWidget || !negativeWidget) return ret;

            enforceMinWidth(this);

            const currentPath = () => (pathWidget?.value || "").trim();

            const loadByName = async (name) => {
                if (!name) return;
                try {
                    const data = await fetchJSON(
                        `/mykee/prompt_templates/load?path=${encodeURIComponent(currentPath())}&name=${encodeURIComponent(name)}`
                    );
                    positiveWidget.value = data.prompt ?? "";
                    negativeWidget.value = data.negative_prompt ?? "";
                    if (data.name && data.name !== nameWidget.value) nameWidget.value = data.name;
                    this.setDirtyCanvas(true, true);
                } catch (e) {
                    window.alert("Load failed: " + e.message);
                }
            };

            let existingNames = [];
            const listWidget = this.addWidget(
                "combo",
                "Existing templates",
                "",
                (value) => {
                    if (!value) return;
                    nameWidget.value = value;
                    // Picking one loads it straight away - no separate
                    // Load button anymore (see the New button below).
                    loadByName(value);
                },
                { values: existingNames }
            );
            // template_name is re-derived from disk on every reload anyway,
            // so a stale pick shouldn't persist across page reloads.
            listWidget.serialize = false;

            // Guards against out-of-order responses: if two refreshes are in
            // flight (e.g. one fired at node creation with the default path,
            // another after the saved custom_path was restored), only the
            // newest one is allowed to update the list.
            let refreshSeq = 0;
            const refreshList = async (retriesLeft = 2) => {
                const seq = ++refreshSeq;
                try {
                    const data = await fetchJSON(
                        `/mykee/prompt_templates/list?path=${encodeURIComponent(currentPath())}`
                    );
                    if (seq !== refreshSeq) return existingNames;
                    // Update the SAME array in place, so any reference the
                    // frontend may hold to options.values stays valid.
                    existingNames.length = 0;
                    existingNames.push(...(data.templates || []));
                    if (listWidget.options) listWidget.options.values = existingNames;
                    this.setDirtyCanvas(true, true);
                    return existingNames;
                } catch (e) {
                    console.warn("Mykee Prompt Template: couldn't fetch template list", e);
                    // The server may not be fully ready right after a page
                    // load / workflow switch - try again shortly.
                    if (seq === refreshSeq && retriesLeft > 0) {
                        setTimeout(() => refreshList(retriesLeft - 1), 1000);
                    }
                    return existingNames;
                }
            };
            this._mykeeRefreshTemplateList = refreshList;

            // Re-read the list whenever custom_path changes, so the picker
            // never shows templates from a stale folder.
            if (pathWidget) {
                const origPathCallback = pathWidget.callback;
                pathWidget.callback = function () {
                    const r = origPathCallback ? origPathCallback.apply(this, arguments) : undefined;
                    refreshList();
                    return r;
                };
            }

            const newBtn = this.addWidget("button", "🆕 New", null, () => {
                nameWidget.value = "";
                positiveWidget.value = "";
                negativeWidget.value = "";
                this.setDirtyCanvas(true, true);
            });
            newBtn.serialize = false;

            const saveBtn = this.addWidget("button", "💾 Save", null, async () => {
                const name = (nameWidget.value || "").trim();
                if (!name) {
                    window.alert("Enter a template name to save.");
                    return;
                }
                try {
                    const data = await fetchJSON("/mykee/prompt_templates/save", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            path: currentPath(),
                            name,
                            prompt: positiveWidget.value || "",
                            negative_prompt: negativeWidget.value || "",
                        }),
                    });
                    // The server sanitizes the name for cross-platform file
                    // safety - reflect whatever it actually saved as.
                    if (data.name && data.name !== nameWidget.value) {
                        nameWidget.value = data.name;
                        this.setDirtyCanvas(true, true);
                    }
                    await refreshList();
                    if (app.extensionManager?.toast?.add) {
                        app.extensionManager.toast.add({
                            severity: "success",
                            summary: "Mykee Prompt Template",
                            detail: `'${data.name}.json' saved.`,
                            life: 4000,
                        });
                    }
                } catch (e) {
                    window.alert("Save failed: " + e.message);
                }
            });
            saveBtn.serialize = false;

            const reloadBtn = this.addWidget("button", "🔄 Reload", null, refreshList);
            reloadBtn.serialize = false;

            refreshList();
            enforceMinWidth(this);

            return ret;
        };
    },
});
