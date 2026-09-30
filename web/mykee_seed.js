/*
 * Mykee Seed - frontend.
 *
 * Reworked from the "Seed" node of rgthree-comfy
 * (https://github.com/rgthree/rgthree-comfy, MIT License, Copyright (c) 2023
 * Regis Gaughan, III - see THIRD_PARTY_NOTICES.md). Same idea: the seed
 * widget holds either a fixed seed or a special value, and the real seed is
 * swapped into the prompt right before it is sent to the server (so the
 * generated seed also lands in the saved image metadata). Special values:
 *
 *    -1  new random seed (from the min_seed..max_seed widgets)
 *    -2  last used seed + 1
 *    -3  last used seed - 1
 *
 * New compared to the original:
 *  - min_seed / max_seed are regular widgets (not node properties).
 *  - seed_lock: when ON, a seed generated from a special value is written
 *    straight back into the seed widget, i.e. it is "locked" without
 *    having to press the "use last seed" button.
 *  - A read-only history box at the bottom of the node lists the seeds of
 *    the most recent runs (newest on top, at most history_size of them,
 *    a re-used seed moves to the top instead of being listed twice).
 *    Clicking a line puts that seed into the seed widget. The box is
 *    client-side only (never sent to the backend, so it cannot affect
 *    caching); the list itself is stored in the node's properties, which
 *    keeps it across reloads and inside the saved workflow.
 *
 * Unlike rgthree, this does not depend on rgthree's own event bus: it wraps
 * api.queuePrompt itself (once), and every live Mykee Seed node gets a look
 * at the outgoing prompt.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const NODE_NAME = "MykeeSeed";
const SEED_WIDGET = "seed_value"; // not "seed": avoids ComfyUI's own control_after_generate combo

const SEED_LIMIT = 1125899906842624;
const SPECIAL_RANDOM = -1;
const SPECIAL_INCREMENT = -2;
const SPECIAL_DECREMENT = -3;
const SPECIAL_SEEDS = [SPECIAL_RANDOM, SPECIAL_INCREMENT, SPECIAL_DECREMENT];

const LAST_SEED_LABEL = "♻️ (Use Last Queued Seed)";
const MODE_NEVER = 2; // muted
const MODE_BYPASS = 4;

const HISTORY_PROPERTY = "mykee_seed_history";
const HISTORY_MIN = 1;
const HISTORY_MAX = 50;
const HISTORY_DEFAULT = 5;

// The widget-computed default size feels a bit cramped (long seed values /
// button labels get tight, the history box is squashed) - pad it out a bit.
const SIZE_PADDING = [40, 30];

function paddedSize(size) {
    return [size[0] + SIZE_PADDING[0], size[1] + SIZE_PADDING[1]];
}

// All Mykee Seed nodes that currently exist (validated on use, see below).
const liveNodes = new Set();

function clampInt(value, fallback) {
    const n = Math.round(Number(value));
    if (!Number.isFinite(n)) return fallback;
    return Math.max(-SEED_LIMIT, Math.min(SEED_LIMIT, n));
}

// Random integer in [min, max] (bounds swapped if reversed), never one of
// the special values.
function randomSeedInRange(minValue, maxValue) {
    let lo = clampInt(minValue, 0);
    let hi = clampInt(maxValue, SEED_LIMIT);
    if (lo > hi) [lo, hi] = [hi, lo];
    for (let i = 0; i < 8; i++) {
        const seed = Math.min(hi, lo + Math.floor(Math.random() * (hi - lo + 1)));
        if (!SPECIAL_SEEDS.includes(seed)) return seed;
    }
    return 0; // only reachable when the range consists of special values alone
}

// Decides the seed for this run from the widget's current value and the
// last used seed. No side effects.
function resolveSeed(state) {
    const inputSeed = Number(state.seedWidget.value);
    const wasSpecial = SPECIAL_SEEDS.includes(inputSeed);
    if (!wasSpecial) return { seed: inputSeed, wasSpecial };

    let seed = null;
    if (typeof state.lastSeed === "number" && !SPECIAL_SEEDS.includes(state.lastSeed)) {
        if (inputSeed === SPECIAL_INCREMENT) seed = state.lastSeed + 1;
        else if (inputSeed === SPECIAL_DECREMENT) seed = state.lastSeed - 1;
    }
    // No usable last seed, or we stepped into a special value / out of range: randomize.
    if (seed == null || SPECIAL_SEEDS.includes(seed) || Math.abs(seed) > SEED_LIMIT) {
        seed = randomSeedInRange(state.minWidget?.value, state.maxWidget?.value);
    }
    return { seed, wasSpecial };
}

// The node's entry in the serialized workflow (root graph or subgraph
// definition), so the real seed can be stored in the image metadata.
function findWorkflowNode(workflow, node) {
    if (!workflow) return null;
    let nodes = workflow.nodes ?? [];
    const subgraph = workflow.definitions?.subgraphs?.find((s) => s.id === node.graph?.id);
    if (subgraph) nodes = subgraph.nodes ?? [];
    return nodes.find((n) => String(n.id) === String(node.id)) ?? null;
}

function updateLastSeedButton(state) {
    const button = state.lastSeedButton;
    if (!button) return;
    if (state.lastSeed != null && state.lastSeed !== Number(state.seedWidget.value)) {
        button.label = `♻️ ${state.lastSeed}`;
        button.disabled = false;
    } else {
        button.label = LAST_SEED_LABEL;
        button.disabled = true;
    }
}

function historyLimit(state, override) {
    const raw = override ?? state.historyWidget?.value;
    // Workflows saved before history_size existed can leave "" / undefined in the widget.
    if (raw === "" || raw == null) return HISTORY_DEFAULT;
    const n = Math.round(Number(raw));
    if (!Number.isFinite(n)) return HISTORY_DEFAULT;
    return Math.max(HISTORY_MIN, Math.min(HISTORY_MAX, n));
}

function readStoredHistory(node) {
    const raw = node.properties?.[HISTORY_PROPERTY];
    if (!Array.isArray(raw)) return [];
    // Only real numbers / numeric strings (Number(null) would silently become 0).
    return raw
        .filter((v) => typeof v === "number" || (typeof v === "string" && v.trim() !== ""))
        .map(Number)
        .filter((n) => Number.isFinite(n));
}

// Replaces the history (trimmed to the limit), then updates the stored copy and the box.
function setHistory(node, state, list, limitOverride) {
    state.history = list.slice(0, historyLimit(state, limitOverride));
    node.properties ??= {};
    node.properties[HISTORY_PROPERTY] = [...state.history];
    if (state.historyBox) state.historyBox.value = state.history.join("\n");
}

// Newest on top; a seed that is used again moves to the top instead of being listed twice.
function recordSeed(node, state, seed) {
    setHistory(node, state, [seed, ...state.history.filter((s) => s !== seed)]);
}

// Called for every outgoing prompt. Rewrites this node's seed in place.
function applySeedToPrompt(node, prompt) {
    const state = node.mykeeSeedState;
    if (!state) return;
    if (node.mode === MODE_NEVER || node.mode === MODE_BYPASS) return;

    const output = prompt?.output;
    if (!output) return;

    // Root graph nodes are keyed "12", nodes inside subgraphs "5:12".
    const id = String(node.id);
    const keys = Object.keys(output).filter((key) => {
        if (key !== id && !key.endsWith(`:${id}`)) return false;
        const entry = output[key];
        // Skip if the widget was turned into a (linked) input - not ours to overwrite.
        return (
            entry?.class_type === NODE_NAME &&
            entry.inputs?.[SEED_WIDGET] !== undefined &&
            !Array.isArray(entry.inputs[SEED_WIDGET])
        );
    });
    // Not part of this queue (e.g. partial queue): don't consume a seed.
    if (!keys.length) return;

    const { seed, wasSpecial } = resolveSeed(state);

    for (const key of keys) {
        output[key].inputs[SEED_WIDGET] = seed;
        if (!Array.isArray(output[key].inputs.history_size)) {
            output[key].inputs.history_size = historyLimit(state);
        }
    }

    const workflowNode = findWorkflowNode(prompt.workflow, node);
    if (workflowNode?.widgets_values) {
        const serialized = node.widgets.filter((w) => w.options?.serialize !== false && w.serialize !== false);
        const index = serialized.indexOf(state.seedWidget);
        if (index >= 0) workflowNode.widgets_values[index] = seed;
    }

    state.lastSeed = seed;
    recordSeed(node, state, seed);

    // seed_lock: a freshly generated seed becomes the widget's value.
    if (wasSpecial && state.lockWidget?.value) {
        state.seedWidget.value = seed;
    }

    updateLastSeedButton(state);
    node.setDirtyCanvas?.(true, true);
}

function setupSeedNode(node) {
    const find = (name) => node.widgets?.find((w) => w.name === name);
    const seedWidget = find(SEED_WIDGET);
    if (!seedWidget) return;

    const state = {
        seedWidget,
        minWidget: find("min_seed"),
        maxWidget: find("max_seed"),
        lockWidget: find("seed_lock"),
        historyWidget: find("history_size"),
        lastSeed: undefined,
        lastSeedButton: null,
        history: [],
        historyBox: null,
    };
    node.mykeeSeedState = state;

    // New nodes start on "random"; a loaded workflow overwrites this in configure().
    seedWidget.value = SPECIAL_RANDOM;
    seedWidget.label = "seed";

    node.addWidget(
        "button",
        "🎲 Randomize Each Time",
        "",
        () => {
            seedWidget.value = SPECIAL_RANDOM;
            updateLastSeedButton(state);
            node.setDirtyCanvas?.(true, true);
        },
        { serialize: false }
    );

    node.addWidget(
        "button",
        "🎲 New Fixed Random",
        "",
        () => {
            seedWidget.value = randomSeedInRange(state.minWidget?.value, state.maxWidget?.value);
            updateLastSeedButton(state);
            node.setDirtyCanvas?.(true, true);
        },
        { serialize: false }
    );

    state.lastSeedButton = node.addWidget(
        "button",
        LAST_SEED_LABEL,
        "",
        () => {
            if (state.lastSeed != null) seedWidget.value = state.lastSeed;
            updateLastSeedButton(state);
            node.setDirtyCanvas?.(true, true);
        },
        { serialize: false }
    );
    state.lastSeedButton.label = LAST_SEED_LABEL;
    state.lastSeedButton.disabled = true;

    // History box: a read-only multiline text widget, created client-side only
    // (NOT declared in INPUT_TYPES and never serialized - see the header note).
    const { widget: box } = ComfyWidgets["STRING"](
        node,
        "seed_history",
        ["STRING", { multiline: true, default: "" }],
        app
    );
    box.serialize = false;
    box.options = { ...(box.options ?? {}), serialize: false };
    state.historyBox = box;
    const el = box.inputEl;
    if (el) {
        el.readOnly = true;
        el.placeholder = "(seeds of recent runs appear here - click one to reuse it)";
        el.title = "Click a seed to put it into the seed field";
        el.style.cursor = "pointer";
        el.style.fontSize = "0.8rem";
        el.addEventListener("click", () => {
            // A text selection (e.g. to copy a seed) is not a click on a line.
            if (el.selectionStart !== el.selectionEnd) return;
            const line = el.value.slice(0, el.selectionStart).split("\n").length - 1;
            const seed = state.history[line];
            if (seed === undefined) return;
            seedWidget.value = seed;
            updateLastSeedButton(state);
            node.setDirtyCanvas?.(true, true);
        });
    }

    // Lowering history_size trims the list right away.
    if (state.historyWidget) {
        const originalCallback = state.historyWidget.callback;
        state.historyWidget.callback = function (value, ...rest) {
            const ret = originalCallback ? originalCallback.call(this, value, ...rest) : undefined;
            setHistory(node, state, state.history, value);
            return ret;
        };
    }

    liveNodes.add(node);
    node.setSize?.(paddedSize(node.computeSize()));
}

app.registerExtension({
    name: "Mykee.Seed",

    async setup() {
        // Wrap api.queuePrompt once (guard against a second registration).
        if (api.__mykeeSeedHooked) return;
        api.__mykeeSeedHooked = true;

        const originalQueuePrompt = api.queuePrompt;
        api.queuePrompt = async function (number, prompt, ...args) {
            try {
                for (const node of [...liveNodes]) {
                    // Drop nodes that no longer exist in their graph (deleted, workflow replaced).
                    if (node.graph?.getNodeById?.(node.id) !== node) {
                        liveNodes.delete(node);
                        continue;
                    }
                    applySeedToPrompt(node, prompt);
                }
            } catch (error) {
                console.error("[Mykee Seed] Could not apply seed to prompt:", error);
            }
            return originalQueuePrompt.call(this, number, prompt, ...args);
        };
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            setupSeedNode(this);
            return ret;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            liveNodes.delete(this);
            return onRemoved ? onRemoved.apply(this, arguments) : undefined;
        };

        // After a workflow is loaded, the widgets hold the saved values: refresh the button and the history box.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const ret = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            const state = this.mykeeSeedState;
            if (state) {
                if (state.historyWidget) state.historyWidget.value = historyLimit(state);
                setHistory(this, state, readStoredHistory(this));
                updateLastSeedButton(state);
            }
            return ret;
        };
    },
});
