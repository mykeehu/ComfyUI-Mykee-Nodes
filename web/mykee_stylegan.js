import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * Mykee StyleGAN nodes.
 *
 * MykeeStyleGANFaceGenerator:
 *   - Fetches /mykee/stylegan/vectors on node creation and builds one FLOAT
 *     slider widget per discovered vector (range/step from that vector's
 *     JSON params). Slider values are kept, as a name->value JSON object, in
 *     the hidden "vector_values" string widget, which IS a real Python
 *     input, so the backend always sees them.
 *   - The refresh button re-fetches models + vectors and rebuilds the
 *     sliders in place (existing ones keep their current value; removed
 *     vectors' sliders are dropped; new ones appear at 0) - no ComfyUI
 *     restart needed after adding/editing a vector.
 *
 * MykeeStyleGANVectorPreview:
 *   - Pure preview/compute - no filesystem writes. Builds w_a/w_b (from
 *     seeds or from projecting two photos) and outputs a preview image
 *     plus a w_pair socket to feed into...
 *
 * MykeeStyleGANVectorSave:
 *   - ...which is the only place a vector actually gets written to disk.
 *     "Load" / "Save" are pure REST calls against the vector's JSON - they
 *     don't need w_pair connected or the graph to run. "Save vector (queue
 *     prompt)" queues the workflow so this node's Python FUNCTION runs and
 *     writes the .npy + .json - bypass this node (or just don't connect
 *     it yet) to preview freely without ever touching disk.
 */

function hideWidget(widget) {
    widget.computeSize = () => [0, -4];
    widget.type = "hidden";
}

function findWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

// Defensive repair for onConfigure: an incompatible/stale saved workflow
// (from before a widget got added/removed/reordered - which has happened
// several times across this pack's history) can restore obviously-invalid
// values (NaN, a combo option that no longer exists) via ComfyUI's
// positional widgets_values restore. Reset just the clearly-broken ones
// back to a sane default rather than leaving garbage on screen - this is
// what "Reload Node" already does by rebuilding from scratch; this makes
// a plain page reload self-heal the same way instead of needing that.
function sanitizeNumberWidget(widget, fallback) {
    if (!widget) return;
    const v = typeof widget.value === "number" ? widget.value : NaN;
    if (!Number.isFinite(v)) widget.value = fallback;
}

function sanitizeBooleanWidget(widget, fallback) {
    if (!widget) return;
    if (typeof widget.value !== "boolean") widget.value = fallback;
}

function sanitizeComboWidget(widget, fallback) {
    if (!widget) return;
    const values = widget.options?.values;
    if (Array.isArray(values) && values.length && !values.includes(widget.value)) {
        widget.value = fallback;
    }
}

// Newer ComfyUI saves BOTH a positional widgets_values array AND a
// name-keyed widgets_values_named object per node. The named one is
// authoritative and immune to positional drift (extra/missing slots from
// widgets that existed - or didn't - when that specific save happened) -
// applying it explicitly after the base configure() runs means a stale
// positional array can never win over it.
function applyNamedWidgetValues(node, info) {
    const named = info && info.widgets_values_named;
    if (!named || typeof named !== "object") return;
    for (const [key, val] of Object.entries(named)) {
        const w = findWidget(node, key);
        if (w) w.value = val;
    }
}

async function fetchJSON(url, options) {
    const res = await api.fetchApi(url, options);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    return data;
}

// Wide enough that the model/seed/truncation combos and the per-vector
// slider labels don't get clipped - litegraph's auto-computed width shrinks
// back down to a cramped default every time widgets are added/removed
// (e.g. on every vector refresh), so this has to be re-applied there too.
const MIN_NODE_WIDTH = 340;

function enforceMinWidth(node) {
    const computed = node.computeSize();
    node.setSize([Math.max(computed[0], MIN_NODE_WIDTH), computed[1]]);
}

// ---------------------------------------------------------------------------
// Shared: per-vector slider UI (discovers facegen_vectors/*.npy, builds one
// number widget per vector mirrored into a hidden "vector_values" JSON
// widget) - used by both the Image Generation and Editor nodes.
// ---------------------------------------------------------------------------

function attachVectorSliderUI(node) {
    const valuesWidget = findWidget(node, "vector_values");
    if (!valuesWidget) return;
    hideWidget(valuesWidget);
    if (!valuesWidget.value) valuesWidget.value = "{}";

    const modelWidget = findWidget(node, "model");
    node._mykeeVectorSliders = {}; // name -> widget
    enforceMinWidth(node);

    const currentValues = () => {
        try {
            return JSON.parse(valuesWidget.value || "{}");
        } catch {
            return {};
        }
    };

    const syncValuesWidget = () => {
        const out = {};
        for (const [name, w] of Object.entries(node._mykeeVectorSliders)) {
            out[name] = w.value;
        }
        valuesWidget.value = JSON.stringify(out);
    };

    const rebuildSliders = (vectorsDict) => {
        const prevValues = currentValues();

        // Remove existing vector sliders (keep everything else).
        for (const w of Object.values(node._mykeeVectorSliders)) {
            const idx = node.widgets.indexOf(w);
            if (idx >= 0) node.widgets.splice(idx, 1);
        }
        node._mykeeVectorSliders = {};

        const names = Object.keys(vectorsDict || {}).sort();
        for (const name of names) {
            const p = vectorsDict[name] || {};
            const min = typeof p.min === "number" ? p.min : -5;
            const max = typeof p.max === "number" ? p.max : 5;
            const startValue = name in prevValues ? prevValues[name] : 0;
            const w = node.addWidget(
                "number",
                name,
                Math.min(Math.max(startValue, min), max),
                (value) => {
                    w.value = value;
                    syncValuesWidget();
                },
                { min, max, step: 0.01, precision: 2 }
            );
            // These are purely a client-side mirror of the real
            // "vector_values" JSON widget (which IS serialized) - there
            // can be any number of them depending on what's in
            // facegen_vectors/ at refresh time, so letting them serialize
            // too would shift every widget after them out of position the
            // moment that count ever differs from a previously-saved
            // workflow (e.g. after adding or removing a vector file).
            w.serialize = false;
            node._mykeeVectorSliders[name] = w;
        }

        syncValuesWidget();
        enforceMinWidth(node);
        node.setDirtyCanvas(true, true);
    };

    const refresh = async () => {
        try {
            const [modelsRes, vectorsRes] = await Promise.all([
                fetchJSON("/mykee/stylegan/models"),
                fetchJSON("/mykee/stylegan/vectors"),
            ]);
            if (modelWidget?.options) {
                modelWidget.options.values = modelsRes.models;
                if (!modelsRes.models.includes(modelWidget.value)) {
                    modelWidget.value = modelsRes.models[0];
                }
            }
            rebuildSliders(vectorsRes.vectors);
        } catch (e) {
            console.warn(`${node.title || node.type}: refresh failed`, e);
        }
    };

    const refreshBtn = node.addWidget("button", "🔄 Refresh vectors / models", null, refresh);
    refreshBtn.serialize = false;
    refresh();
}

// ---------------------------------------------------------------------------
// StyleGAN Image Generation
// ---------------------------------------------------------------------------

const GEN_NODE_NAME = "MykeeStyleGANFaceGenerator";

app.registerExtension({
    name: "Mykee.StyleGAN.ImageGeneration",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== GEN_NODE_NAME) return;

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const ret = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            applyNamedWidgetValues(this, info);
            return ret;
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            attachVectorSliderUI(this);
            return ret;
        };
    },
});

// ---------------------------------------------------------------------------
// StyleGAN Editor
// ---------------------------------------------------------------------------

const EDITOR_NODE_NAME = "MykeeStyleGANEditor";

app.registerExtension({
    name: "Mykee.StyleGAN.Editor",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== EDITOR_NODE_NAME) return;

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const ret = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            applyNamedWidgetValues(this, info);
            sanitizeBooleanWidget(findWidget(this, "keep_cache"), true);
            sanitizeNumberWidget(findWidget(this, "projection_steps"), 300);
            sanitizeNumberWidget(findWidget(this, "projection_seed"), 0);
            return ret;
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            attachStatusWidget(this);
            attachVectorSliderUI(this);

            const seedWidget = findWidget(this, "projection_seed");
            const randBtn = this.addWidget("button", "🎲 Randomize projection_seed", null, () => {
                if (!seedWidget) return;
                seedWidget.value = Math.floor(Math.random() * 0xFFFFFFFF);
                this.setDirtyCanvas(true, true);
            });
            randBtn.serialize = false;
            const idx = this.widgets.indexOf(randBtn);
            if (idx !== -1 && seedWidget) {
                this.widgets.splice(idx, 1);
                const afterIdx = this.widgets.indexOf(seedWidget);
                this.widgets.splice(afterIdx + 1, 0, randBtn);
            }
            enforceMinWidth(this);

            return ret;
        };
    },
});

// ---------------------------------------------------------------------------
// StyleGAN Vector Preview
// ---------------------------------------------------------------------------

const PREVIEW_NODE_NAME = "MykeeStyleGANVectorPreview";

let _measureCtx = null;
function getMeasureCtx() {
    if (!_measureCtx) {
        _measureCtx = document.createElement("canvas").getContext("2d");
        _measureCtx.font = "12px Arial";
    }
    return _measureCtx;
}

function wrapTextLines(ctx, text, maxWidth) {
    const words = String(text || "").split(/\s+/).filter(Boolean);
    const lines = [];
    let current = "";
    for (const word of words) {
        const trial = current ? current + " " + word : word;
        if (current && ctx.measureText(trial).width > maxWidth) {
            lines.push(current);
            current = word;
        } else {
            current = trial;
        }
    }
    if (current) lines.push(current);
    return lines;
}

// Wraps to at most 2 lines, ellipsizing the second line if there's more
// text than that can hold - used so long status messages (which can run
// well past a single line at the node's default width) stay readable
// instead of being clipped or overflowing past the node's edge.
function wrapTextMax2Lines(ctx, text, maxWidth) {
    const lines = wrapTextLines(ctx, text, maxWidth);
    if (lines.length <= 2) return lines.length ? lines : [""];
    let second = lines.slice(1).join(" ");
    while (second.length > 1 && ctx.measureText(second + "…").width > maxWidth) {
        second = second.slice(0, -1);
    }
    return [lines[0], second.trimEnd() + "…"];
}

function attachStatusWidget(node) {
    const STATUS_FONT = "12px Arial";
    const LINE_HEIGHT = 14;
    const statusWidget = {
        name: "mykee_status",
        type: "mykee_status_display",
        value: "",
        // Purely a live display for status pushed from Python during a
        // run - it must NOT be saved into the workflow JSON, or reloading
        // the page/workflow replays whatever old message was showing last
        // session (e.g. a stale "Not saved..." from a previous run)
        // instead of starting blank until something actually runs again.
        serialize: false,
        computeSize: function (width) {
            const ctx = getMeasureCtx();
            ctx.font = STATUS_FONT;
            const maxWidth = Math.max(10, width - 12);
            const lines = wrapTextMax2Lines(ctx, this.value ? String(this.value) : "", maxWidth);
            return [width, lines.length * LINE_HEIGHT + 6];
        },
        draw: function (ctx, node, widgetWidth, y, height) {
            ctx.save();
            ctx.fillStyle = "#9d9d9d";
            ctx.font = STATUS_FONT;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            const maxWidth = Math.max(10, widgetWidth - 12);
            const lines = wrapTextMax2Lines(ctx, this.value ? String(this.value) : "", maxWidth);
            const totalHeight = lines.length * LINE_HEIGHT;
            let lineY = y + height / 2 - totalHeight / 2 + LINE_HEIGHT / 2;
            for (const line of lines) {
                ctx.fillText(line, widgetWidth / 2, lineY);
                lineY += LINE_HEIGHT;
            }
            ctx.restore();
        },
    };
    node.addCustomWidget(statusWidget);
    const idx = node.widgets.indexOf(statusWidget);
    if (idx > 0) {
        node.widgets.splice(idx, 1);
        node.widgets.unshift(statusWidget);
    }
    node._mykeeStatusWidget = statusWidget;
}

app.registerExtension({
    name: "Mykee.StyleGAN.VectorPreview",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== PREVIEW_NODE_NAME) return;

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const ret = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            applyNamedWidgetValues(this, info);
            const modelWidget = findWidget(this, "model");
            sanitizeComboWidget(findWidget(this, "source_mode"), "image");
            sanitizeComboWidget(modelWidget, modelWidget?.options?.values?.[0]);
            sanitizeBooleanWidget(findWidget(this, "keep_cache"), true);
            sanitizeNumberWidget(findWidget(this, "seed_a"), 100);
            sanitizeNumberWidget(findWidget(this, "seed_b"), 200);
            sanitizeNumberWidget(findWidget(this, "seed_truncation"), 0.7);
            sanitizeNumberWidget(findWidget(this, "projection_steps"), 300);
            sanitizeNumberWidget(findWidget(this, "projection_seed"), 0);
            return ret;
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            enforceMinWidth(this);
            attachStatusWidget(this);
            // Deliberately NOT dynamically hiding seed_a/seed_b/
            // seed_truncation/projection_steps based on source_mode - it
            // made resizing look messy. Each field's tooltip says which
            // source_mode it applies to instead.

            // seed_a/seed_b are plain INT widgets (no ComfyUI built-in
            // control_after_generate): two of those on the same node
            // fight each other (their dropdowns end up mirroring one
            // another), and - worse - having either auto-randomize after
            // every queue changes the widget value even in image mode
            // where it's unused, which silently invalidates this node's
            // cache and forces a full re-run (including a slow image
            // re-projection) the next time you queue just to hit Save on
            // the Vector Save node. These buttons reroll on demand only.
            const seedAWidget = findWidget(this, "seed_a");
            const seedBWidget = findWidget(this, "seed_b");
            const randomizeSeed = (widget) => {
                if (!widget) return;
                widget.value = Math.floor(Math.random() * 0xFFFFFFFF);
                this.setDirtyCanvas(true, true);
            };
            const insertRightAfter = (widget, afterWidget) => {
                const idx = this.widgets.indexOf(widget);
                if (idx === -1) return;
                this.widgets.splice(idx, 1);
                const afterIdx = this.widgets.indexOf(afterWidget);
                this.widgets.splice(afterIdx + 1, 0, widget);
            };
            const randABtn = this.addWidget("button", "🎲 Randomize seed_a", null, () => randomizeSeed(seedAWidget));
            randABtn.serialize = false;
            insertRightAfter(randABtn, seedAWidget);
            const randBBtn = this.addWidget("button", "🎲 Randomize seed_b", null, () => randomizeSeed(seedBWidget));
            randBBtn.serialize = false;
            insertRightAfter(randBBtn, seedBWidget);
            const projSeedWidget = findWidget(this, "projection_seed");
            const randProjBtn = this.addWidget("button", "🎲 Randomize projection_seed", null, () => randomizeSeed(projSeedWidget));
            randProjBtn.serialize = false;
            insertRightAfter(randProjBtn, projSeedWidget);
            enforceMinWidth(this);

            return ret;
        };
    },
    async setup() {
        // Shared by both StyleGAN nodes that have a status widget.
        api.addEventListener("mykee.stylegan_status", (event) => {
            const detail = event.detail || {};
            const nodeId = detail.node;
            const text = detail.text;
            if (nodeId === undefined || nodeId === null) return;
            const graphNode = app.graph.getNodeById(Number(nodeId)) ?? app.graph.getNodeById(nodeId);
            if (graphNode && graphNode._mykeeStatusWidget) {
                graphNode._mykeeStatusWidget.value = text;
                enforceMinWidth(graphNode);
                graphNode.setDirtyCanvas(true, true);
            }
        });
    },
});

// ---------------------------------------------------------------------------
// StyleGAN Vector Save
// ---------------------------------------------------------------------------

const SAVE_NODE_NAME = "MykeeStyleGANVectorSave";
const SAVE_PARAM_FIELDS = ["layer_start", "layer_end", "step", "min_value", "max_value"];

app.registerExtension({
    name: "Mykee.StyleGAN.VectorSave",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== SAVE_NODE_NAME) return;

        // "armed" must be a real, serialized backend input (Python needs
        // its actual value every execution) - but that means a saved
        // workflow.json can freeze in a "true" from the instant between
        // the Save button arming it and the post-run disarm event coming
        // back (e.g. if the page was reloaded/saved right then), and
        // loading that workflow later would silently save again on the
        // very next Run. onConfigure fires after ComfyUI restores widget
        // values from a loaded workflow, so force it back to false there
        // no matter what was persisted - it must only ever become true
        // from an explicit Save-button click in the current session.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const ret = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            applyNamedWidgetValues(this, info);
            const armedWidget = findWidget(this, "armed");
            if (armedWidget) armedWidget.value = false;
            // Defensive: an already-shifted saved workflow from before the
            // serialize:false fixes below could have left vector_name
            // holding a stray null/undefined - don't let that render as
            // the literal text "null".
            const nameWidgetCfg = findWidget(this, "vector_name");
            if (nameWidgetCfg && nameWidgetCfg.value == null) nameWidgetCfg.value = "";
            sanitizeBooleanWidget(findWidget(this, "allow_overwrite"), false);
            sanitizeNumberWidget(findWidget(this, "layer_start"), 0);
            sanitizeNumberWidget(findWidget(this, "layer_end"), 17);
            sanitizeNumberWidget(findWidget(this, "step"), 0.25);
            sanitizeNumberWidget(findWidget(this, "min_value"), -2.0);
            sanitizeNumberWidget(findWidget(this, "max_value"), 2.0);
            return ret;
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const nameWidget = findWidget(this, "vector_name");
            if (!nameWidget) return ret;
            enforceMinWidth(this);
            attachStatusWidget(this);

            // "armed" is only ever set by the Save button/the disarm
            // listener below - hide it so it can't be flipped on by
            // accident (which would make the next plain Run save), and
            // force it to false right now too (belt-and-suspenders
            // alongside onConfigure above, for the brand-new-node case).
            const armedWidgetInit = findWidget(this, "armed");
            if (armedWidgetInit) {
                hideWidget(armedWidgetInit);
                armedWidgetInit.value = false;
            }

            let existingVectorNames = [];
            const listWidget = this.addWidget(
                "combo",
                "Existing vectors",
                "",
                (value) => {
                    if (value) nameWidget.value = value;
                },
                { values: existingVectorNames }
            );
            // Just a convenience picker that copies its selection into
            // vector_name - its own value is re-derived from disk on every
            // load anyway, so don't let a stale pick persist across
            // reloads either.
            listWidget.serialize = false;

            const refreshList = async () => {
                try {
                    const data = await fetchJSON("/mykee/stylegan/vectors");
                    existingVectorNames = Object.keys(data.vectors || {}).sort();
                    if (listWidget.options) listWidget.options.values = existingVectorNames;
                    return data.vectors || {};
                } catch (e) {
                    console.warn("Mykee StyleGAN Vector Save: couldn't fetch vector list", e);
                    return {};
                }
            };
            this._mykeeRefreshVectorList = refreshList;

            const loadBtn = this.addWidget("button", "📂 Load", null, async () => {
                const name = (nameWidget.value || "").trim();
                if (!name) {
                    window.alert("Enter a vector name to load.");
                    return;
                }
                const vectors = await refreshList();
                const entry = vectors[name];
                if (!entry) {
                    window.alert(`No '${name}.npy' found in the facegen_vectors folder.`);
                    return;
                }
                const setVal = (fieldName, key) => {
                    const w = findWidget(this, fieldName);
                    if (w && key in entry) w.value = entry[key];
                };
                setVal("layer_start", "layer_start");
                setVal("layer_end", "layer_end");
                setVal("step", "step");
                setVal("min_value", "min");
                setVal("max_value", "max");
                this.setDirtyCanvas(true, true);
            });
            loadBtn.serialize = false;

            const saveJsonBtn = this.addWidget("button", "💾 Save", null, async () => {
                const name = (nameWidget.value || "").trim();
                if (!name) {
                    window.alert("Enter a vector name to save.");
                    return;
                }
                const body = { name };
                for (const fieldName of SAVE_PARAM_FIELDS) {
                    const w = findWidget(this, fieldName);
                    if (!w) continue;
                    const key = fieldName === "min_value" ? "min" : fieldName === "max_value" ? "max" : fieldName;
                    body[key] = w.value;
                }
                try {
                    await fetchJSON("/mykee/stylegan/save_vector_params", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(body),
                    });
                    await refreshList();
                    if (app.extensionManager?.toast?.add) {
                        app.extensionManager.toast.add({
                            severity: "success",
                            summary: "Mykee StyleGAN Vector Save",
                            detail: `'${name}.json' saved.`,
                            life: 4000,
                        });
                    }
                } catch (e) {
                    window.alert("Save failed: " + e.message);
                }
            });
            saveJsonBtn.serialize = false;

            const saveVectorBtn = this.addWidget("button", "✨ Save vector (queue prompt)", null, async () => {
                const name = (nameWidget.value || "").trim();
                if (!name) {
                    window.alert("A vector name is required to save.");
                    return;
                }
                if (!this.inputs?.some((i) => i.name === "w_pair" && i.link != null)) {
                    window.alert("Connect w_pair from a StyleGAN Vector Preview node first.");
                    return;
                }
                const overwriteWidget = findWidget(this, "allow_overwrite");
                const vectors = await refreshList();
                if (name in vectors) {
                    if (!overwriteWidget?.value) {
                        window.alert(`'${name}.npy' already exists. Turn on 'allow_overwrite' if you really want to replace it, or pick a different vector_name.`);
                        return;
                    }
                    if (!window.confirm(`This will overwrite the existing '${name}' vector. Continue?`)) return;
                }
                // "armed" is the safety gate that keeps a plain Run/Queue
                // press from saving by accident (this node is an
                // OUTPUT_NODE, so it runs on any queue, not just from this
                // button) - arm it right before queuing; the node disarms
                // itself the instant it actually runs.
                const armedWidget = findWidget(this, "armed");
                if (armedWidget) armedWidget.value = true;
                app.queuePrompt(0, 1);
            });
            saveVectorBtn.serialize = false;

            refreshList();
            enforceMinWidth(this);

            return ret;
        };
    },
    async setup() {
        api.addEventListener("mykee.stylegan_disarm", (event) => {
            const detail = event.detail || {};
            const nodeId = detail.node;
            if (nodeId === undefined || nodeId === null) return;
            const graphNode = app.graph.getNodeById(Number(nodeId)) ?? app.graph.getNodeById(nodeId);
            const armedWidget = graphNode && findWidget(graphNode, "armed");
            if (armedWidget) {
                armedWidget.value = false;
                graphNode.setDirtyCanvas(true, true);
            }
        });
        api.addEventListener("mykee.stylegan_vector_saved", (event) => {
            const detail = event.detail || {};
            const nodeId = detail.node;
            if (nodeId === undefined || nodeId === null) return;
            const graphNode = app.graph.getNodeById(Number(nodeId)) ?? app.graph.getNodeById(nodeId);
            graphNode?._mykeeRefreshVectorList?.();
        });
    },
});
