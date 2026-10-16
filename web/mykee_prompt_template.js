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
 * "show_negative_prompt" (off by default) decides whether the negative
 * box is visible at all - most prompts today don't use one, and an empty
 * box just wastes space:
 *
 *   - off: negative_prompt_text is hidden, the positive box takes over
 *     the freed height, and the node's negative output is empty (the
 *     Python side enforces that, so hidden stale text never leaks out).
 *     Save then stores an empty negative prompt too.
 *   - on: both boxes are shown and share the node's height.
 *   - Loading a template that has a negative prompt switches it on, so
 *     the text is never loaded invisibly. Workflows saved before this
 *     switch existed come back with it on if they held a negative prompt.
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

// Default node size for a freshly added node (negative box hidden, so the
// positive prompt gets a comfortable height right away).
const DEFAULT_NODE_HEIGHT = 400;

// Shows/hides a widget (incl. its DOM textarea) and makes litegraph reserve
// no row for it while hidden. The original type/size hooks are remembered on
// the widget so they can be restored exactly.
function setWidgetVisible(widget, visible) {
    if (!widget) return;
    if (!widget._mykeeOrig) {
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
        // Type first: while the type is still "hidden", newer frontends
        // ignore writes to widget.hidden (they derive visibility from it).
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
            const showWidget = findWidget(this, "show_negative_prompt");
            if (showWidget) {
                // Workflow saved before the switch existed: no stored value
                // for it. Keep its negative prompt visible (and in use) if it
                // had one, instead of silently hiding/dropping it.
                const stored = arguments[0]?.widgets_values;
                const idx = this.widgets.indexOf(showWidget);
                if (Array.isArray(stored) && stored[idx] === undefined) {
                    const neg = findWidget(this, "negative_prompt_text");
                    showWidget.value = !!(neg && String(neg.value || "").trim());
                }
                if (this._mykeeApplyNegativeVisibility) this._mykeeApplyNegativeVisibility();
            }
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
            const showWidget = findWidget(this, "show_negative_prompt");
            if (!nameWidget || !positiveWidget || !negativeWidget) return ret;

            enforceMinWidth(this);

            // Show/hide the negative box. Hiding leaves the node's height
            // alone, so the positive box simply grows into the freed space;
            // showing makes sure both boxes fit.
            const applyNegativeVisibility = () => {
                const show = showWidget ? !!showWidget.value : true;
                setWidgetVisible(negativeWidget, show);
                if (show) {
                    const needed = this.computeSize();
                    if (this.size[1] < needed[1]) this.setSize([this.size[0], needed[1]]);
                }
                this.setDirtyCanvas(true, true);
            };
            this._mykeeApplyNegativeVisibility = applyNegativeVisibility;
            if (showWidget) {
                const origShowCallback = showWidget.callback;
                showWidget.callback = function () {
                    const r = origShowCallback ? origShowCallback.apply(this, arguments) : undefined;
                    applyNegativeVisibility();
                    return r;
                };
            }

            const currentPath = () => (pathWidget?.value || "").trim();

            const loadByName = async (name) => {
                if (!name) return;
                try {
                    const data = await fetchJSON(
                        `/mykee/prompt_templates/load?path=${encodeURIComponent(currentPath())}&name=${encodeURIComponent(name)}`
                    );
                    positiveWidget.value = data.prompt ?? "";
                    negativeWidget.value = data.negative_prompt ?? "";
                    // Never load a negative prompt invisibly.
                    if (showWidget && negativeWidget.value.trim() && !showWidget.value) {
                        showWidget.value = true;
                        applyNegativeVisibility();
                    }
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
                            // Hidden negative box = no negative prompt (matches
                            // what the node outputs).
                            negative_prompt: showWidget && !showWidget.value ? "" : (negativeWidget.value || ""),
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
            applyNegativeVisibility();
            if (!showWidget || !showWidget.value) {
                // Fresh node (a loaded workflow restores its own size in
                // onConfigure): give the positive box room right away.
                this.setSize([this.size[0], Math.max(this.size[1], DEFAULT_NODE_HEIGHT)]);
            }

            return ret;
        };
    },
});
