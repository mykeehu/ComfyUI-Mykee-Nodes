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
 * The picker itself is not serialized (it is just a view of the folder),
 * but template_name is - so after a workflow-tab switch or page reload the
 * picker is re-pointed at whatever template_name holds, as long as that
 * template still exists on disk. It is also emptied by "New" and when
 * template_name no longer matches a stored template.
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

// ---- template_name output without touching the cache --------------------
// ComfyUI builds a node's cache key from its inputs and its ancestors' inputs.
// With the typed template_name as a real input of this node, renaming a
// template would change the key of everything downstream (encode, sampler,
// decode...) and the whole image would be generated again just to get a new
// file name. So, when the prompt is built, the name is taken out of the node's
// inputs (the node ignores it anyway) and delivered as a plain constant to
// whatever is wired to the "template_name" output. Only that receiving node
// (e.g. Save Image) sees a change and runs again; the rest stays cached.
const NAME_OUTPUT_SLOT = 2;
const INVALID_NAME_CHARS = '<>:"/\\|?*';
const RESERVED_NAMES = new Set(["CON", "PRN", "AUX", "NUL",
    ...Array.from({ length: 9 }, (_, i) => `COM${i + 1}`),
    ...Array.from({ length: 9 }, (_, i) => `LPT${i + 1}`)]);

function cleanNameForFile(name) {
    // Same rules as the backend's template name sanitizing; only the file
    // part is used (a sub-folder in a filename prefix would create folders).
    const base = String(name ?? "").replace(/\\/g, "/").split("/").map((p) => p.trim()).filter(Boolean).pop() ?? "";
    let out = Array.from(base)
        .filter((c) => !INVALID_NAME_CHARS.includes(c) && c.charCodeAt(0) >= 32).join("");
    out = out.trim().replace(/[. ]+$/, "").trim();
    if (RESERVED_NAMES.has(out.toUpperCase())) out = "_" + out;
    out = out.slice(0, 150).trim().replace(/[. ]+$/, "");
    return out || "ComfyUI";
}

function installNameOutputRewrite() {
    const original = app.graphToPrompt;
    if (typeof original !== "function" || original._mykeeNameRewrite) return;
    const wrapped = async function () {
        const result = await original.apply(this, arguments);
        try {
            const output = result?.output;
            if (!output) return result;
            const nameByNode = new Map();
            for (const node of app.graph?._nodes ?? []) {
                if ((node.comfyClass ?? node.type) !== NODE_NAME) continue;
                const w = node.widgets?.find((x) => x.name === "template_name");
                nameByNode.set(String(node.id), cleanNameForFile(w?.value));
            }
            if (!nameByNode.size) return result;
            for (const entry of Object.values(output)) {
                for (const [key, val] of Object.entries(entry?.inputs ?? {})) {
                    if (Array.isArray(val) && val.length === 2 && val[1] === NAME_OUTPUT_SLOT
                        && nameByNode.has(String(val[0]))) {
                        entry.inputs[key] = nameByNode.get(String(val[0]));
                    }
                }
            }
            for (const id of nameByNode.keys()) {
                if (output[id]?.inputs && "template_name" in output[id].inputs) {
                    output[id].inputs.template_name = "";
                }
            }
        } catch (err) {
            console.warn("Mykee Prompt Template: name output rewrite skipped", err);
        }
        return result;
    };
    wrapped._mykeeNameRewrite = true;
    app.graphToPrompt = wrapped;
}

app.registerExtension({
    name: "Mykee.PromptTemplate",
    setup() {
        installNameOutputRewrite();
    },
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
            // The saved node size already includes a visible preview picture.
            // Only trusted when the saved workflow says it was saved with the
            // preview's room included (older workflows never did).
            this._mykeeFromSave = Number(this.properties?.mykee_preview_room) > 0;
            // Show the last loaded template right away (template_name was just
            // restored); the list refresh below then confirms it against disk.
            if (this._mykeeSyncPicker) this._mykeeSyncPicker(true);
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
                    if (this._mykeeUpdatePreview) this._mykeeUpdatePreview();
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

            // Point the (non-serialized) picker at template_name. With
            // optimistic=true the name is shown even before the list has
            // been read from disk; otherwise it is only shown if a template
            // of that name actually exists.
            const syncPicker = (optimistic = false) => {
                const name = (nameWidget.value || "").trim().replace(/\\/g, "/");
                const known = existingNames.includes(name);
                listWidget.value = name && (known || optimistic) ? name : "";
                this.setDirtyCanvas(true, true);
                if (this._mykeeUpdatePreview) this._mykeeUpdatePreview();
            };
            this._mykeeSyncPicker = syncPicker;

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
                    syncPicker();
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
                listWidget.value = "";
                this.setDirtyCanvas(true, true);
                if (this._mykeeUpdatePreview) this._mykeeUpdatePreview();
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

            const reloadBtn = this.addWidget("button", "🔄 Reload", null, () => {
                refreshList();
                if (this._mykeeUpdatePreview) this._mykeeUpdatePreview(true);
            });
            reloadBtn.serialize = false;

            // ---- Preview picture -------------------------------------------
            // An image with the same base name as the template, in the same
            // folder, is shown under the Reload button (a sample of what the
            // template makes). No such image: the widget is hidden and takes
            // no room, the node is as small as before.
            const PREVIEW_MIN_H = 100;
            const PREVIEW_MAX_H = 320;
            const PREVIEW_PAD = 8;
            const previewImg = document.createElement("img");
            previewImg.style.cssText =
                "width:100%;height:100%;object-fit:contain;display:block;border-radius:4px;pointer-events:none;";
            previewImg.draggable = false;
            const previewBox = document.createElement("div");
            previewBox.style.cssText = "width:100%;height:100%;box-sizing:border-box;padding:2px 0;overflow:hidden;";
            previewBox.appendChild(previewImg);

            let previewH = 0; // current height of the preview row
            let previewShown = false;
            const previewWidget = this.addDOMWidget("template_preview", "MYKEE_TEMPLATE_PREVIEW", previewBox, {
                serialize: false,
                getMinHeight: () => previewH,
                getMaxHeight: () => previewH,
            });
            previewWidget.serialize = false;
            previewWidget.computeSize = (width) => [width ?? 0, previewH];
            setWidgetVisible(previewWidget, false);

            let grown = 0; // how much taller the node was made for the preview
            const showPreview = (naturalW, naturalH) => {
                const usable = Math.max(120, this.size[0] - 20);
                const h = Math.round(Math.min(PREVIEW_MAX_H, Math.max(PREVIEW_MIN_H, usable * (naturalH / naturalW))));
                const wasShown = previewShown;
                const oldH = previewH;
                previewH = h;
                if (!wasShown) {
                    previewShown = true;
                    setWidgetVisible(previewWidget, true);
                }
                if (!wasShown && !this._mykeeFromSave) {
                    // The text boxes keep the room they had: the node grows by
                    // the height of the preview.
                    grown = h + PREVIEW_PAD;
                    this.setSize([this.size[0], this.size[1] + grown]);
                } else if (!wasShown) {
                    // A workflow saved with the preview showing already holds
                    // the room for it.
                    grown = h + PREVIEW_PAD;
                } else if (h !== oldH) {
                    grown += h - oldH;
                    this.setSize([this.size[0], this.size[1] + (h - oldH)]);
                }
                this._mykeeFromSave = false;
                this.properties = this.properties || {};
                this.properties.mykee_preview_room = grown;
                const needed = this.computeSize();
                if (this.size[1] < needed[1]) this.setSize([this.size[0], needed[1]]);
                this.setDirtyCanvas(true, true);
            };

            const hidePreview = () => {
                this._mykeeFromSave = false;
                if (!previewShown) { if (this.properties) this.properties.mykee_preview_room = 0; return; }
                previewShown = false;
                previewH = 0;
                previewImg.removeAttribute("src");
                setWidgetVisible(previewWidget, false);
                const floor = this.computeSize();
                // Back to the height the node had without the preview.
                this.setSize([this.size[0], Math.max(floor[1], this.size[1] - grown)]);
                grown = 0;
                if (this.properties) this.properties.mykee_preview_room = 0;
                this.setDirtyCanvas(true, true);
            };

            let previewSeq = 0;
            let previewTimer = null;
            let previewKey = "";
            const runPreviewUpdate = async (force) => {
                const seq = ++previewSeq;
                const name = (nameWidget.value || "").trim();
                if (!name) {
                    previewKey = "";
                    hidePreview();
                    return;
                }
                try {
                    const q = `path=${encodeURIComponent(currentPath())}&name=${encodeURIComponent(name)}`;
                    const info = await fetchJSON(`/mykee/prompt_templates/image_info?${q}`);
                    if (seq !== previewSeq) return; // a newer request is in flight
                    if (!info.exists) {
                        previewKey = "";
                        hidePreview();
                        return;
                    }
                    const key = `${q}&v=${info.version}`;
                    if (key === previewKey && previewShown && !force) return;
                    previewKey = key;
                    const probe = new Image();
                    probe.onload = () => {
                        if (seq !== previewSeq) return;
                        previewImg.src = probe.src;
                        showPreview(probe.naturalWidth || 1, probe.naturalHeight || 1);
                    };
                    probe.onerror = () => {
                        if (seq === previewSeq) hidePreview();
                    };
                    probe.src = api.apiURL(`/mykee/prompt_templates/image?${key}`);
                } catch (e) {
                    if (seq === previewSeq) hidePreview();
                }
            };
            this._mykeeUpdatePreview = (force = false) => {
                clearTimeout(previewTimer);
                previewTimer = setTimeout(() => runPreviewUpdate(force), 150);
            };

            // Typing / pasting a template name re-checks for a picture.
            {
                const node = this;
                const origNameCallback = nameWidget.callback;
                nameWidget.callback = function () {
                    const r = origNameCallback ? origNameCallback.apply(this, arguments) : undefined;
                    node._mykeeUpdatePreview();
                    return r;
                };
            }

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
