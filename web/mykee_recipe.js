/*
 * Mykee Recipe - frontend.
 *
 * The node has a "recipe_name" output (slot 0), an "xyz_info" output (slot 1,
 * for the XYZ Plot Assembler) and up to ten parameter outputs p1 .. p10
 * (slots 2..11, "slots" sets how many are shown). Like the XYZ Plot
 * axes and the Model Template outputs, a parameter output is dragged onto any
 * widget (cfg, sampler_name, ckpt_name, a text box, a boolean, ...): the
 * output takes over the widget's type (like a Primitive node does), gets a key
 * (by default the widget's name - editable) and the node shows an editor for
 * the value. What is set here is what the connected widget receives.
 *
 * Four tabs (same tab bar as the Text Analyzer):
 *
 *   Parameters  the connected parameters with their keys and value editors,
 *               a recipe picker (loads the picked recipe into the parameters),
 *               the recipe name and "Send to Recipe".
 *   Recipe      the current recipe as editable JSON text, a pager over all the
 *               recipes in the node, "Load Recipe" (recipe -> parameters).
 *   Batch       a switch (off by default) and, when on, the list of the recipes
 *               on the node with check boxes (Select All / Select None):
 *               Run queues one prompt per ticked recipe, one after another.
 *               "Send parameters" also writes the parameter names and values
 *               under the recipe name in the plot labels (xyz_info).
 *   JSON file   the whole recipe file as editable JSON text, plus json_name /
 *               custom_path / the list of existing files / Save / Load.
 *
 * Recipes find their targets by KEY, never by node class: a recipe entry is
 * applied to the connected output that carries the same key. Only if no key
 * matches, an entry whose stored widget-name hint equals the widget's name is
 * used - when exactly one entry fits. See nodes/mykee_recipe.py for the file
 * format and the rules; every applied / skipped value is reported in a toast.
 *
 * Safety: everything that comes from a file, the JSON text boxes or a shared
 * workflow goes through cleanDoc() first (strict schema, limits, no
 * prototype-polluting keys, finite numbers) and every value is checked against
 * the connected widget (type, min / max, "is it in the list") before it is
 * used. User data is only ever put into the DOM as text, never as HTML.
 *
 * The state (parameters, recipes, pager position) lives in the hidden
 * "recipe_config" widget (JSON) and is saved with the workflow. At queue time
 * its serializeValue sends only the resolved values of the connected outputs,
 * so editing recipes never invalidates ComfyUI's cache.
 *
 * Sizes: the node's size is saved with the workflow, each tab remembers its
 * own node height, and a loaded workflow only ever grows the node, never
 * shrinks it. The two text boxes follow the node's height.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "MykeeRecipe";
const CONFIG_WIDGET = "recipe_config";
const MAX_SLOTS = 10;
const FIRST_PARAM_OUTPUT = 2; // output 0 is recipe_name, output 1 is xyz_info
const XYZ_OUTPUT = "xyz_info";
const XYZ_INFO_TYPE = "MYKEE_XYZ_INFO";
// Plot controllers: a batch can't be combined with them (both fan out the queue).
const PLOT_CLASSES = new Set(["MykeeXYZPlot", "MykeeSimpleXPlot"]);
const BATCH_CONFIRM_LIMIT = 100;
const MAX_LABEL = 600;
const SUPPORTED_TYPES = ["INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"];

// Limits - the same as in nodes/mykee_recipe.py.
const MAX_RECIPES = 500;
const MAX_PARAMS = 64;
const MAX_KEY = 64;
const MAX_NAME = 120;
const MAX_HINT = 100;
const MAX_STRING = 200000;
const MAX_COMBO = 1000;
const SAFE_INT = 2 ** 53;
const BAD_KEYS = new Set(["__proto__", "constructor", "prototype"]);

const MIN_WIDTH = 400;
const PANEL_GAP = 6;
const PANEL_BOTTOM_PADDING = 32;
const RECIPE_PANE_MIN_HEIGHT = 300;
const JSON_PANE_MIN_HEIGHT = 200;
const BATCH_PANE_MIN_HEIGHT = 260;

const TABS = [
    { id: "params", label: "Parameters", title: "The parameters connected to the outputs, and Send to Recipe" },
    { id: "recipe", label: "Recipe", title: "The current recipe (editable) and the pager over all recipes" },
    { id: "batch", label: "Batch", title: "Run several recipes one after another" },
    { id: "json", label: "JSON file", title: "The whole recipe file, and Save / Load" },
];
const TAB_PROP = "mykee_recipe_tab";
const TAB_SIZE_PROP = "mykee_recipe_tab_heights"; // node height per tab, saved with the workflow
const BOX_HEIGHT_PROP = "mykee_recipe_box_heights"; // heights of the text boxes in the parameter list
const DEFAULT_TAB = "params";
const TAB_BAR_NAME = "_mykee_recipe_tabs";
const TAB_BAR_HEIGHT = 30;
const BADGE_COLOR = "#7fc8ff";

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function toast(detail, severity = "warn", life = 6000) {
    if (app.extensionManager?.toast?.add) {
        app.extensionManager.toast.add({ severity, summary: "Mykee Recipe", detail, life });
    } else {
        console.warn(`[Mykee Recipe] ${detail}`);
    }
}

async function fetchJSON(url, options) {
    const res = await api.fetchApi(url, options);
    let data = {};
    try {
        data = await res.json();
    } catch {
        /* no JSON body */
    }
    if (!res.ok) throw new Error(data.error || res.statusText || "request failed");
    return data;
}

function findWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function getLink(graph, id) {
    if (!graph || id == null) return undefined;
    return graph.getLink?.(id) ?? graph.links?.get?.(id) ?? graph.links?.[id];
}

const own = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key);
const isPlainObject = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const clamp = (n, lo, hi) => Math.min(hi, Math.max(lo, n));
const roundFloat = (n) => Math.round(n * 1e6) / 1e6;

// ---------------------------------------------------------------------------
// Schema cleaning (mirror of clean_doc() in nodes/mykee_recipe.py)
// ---------------------------------------------------------------------------

function cleanLabel(value, limit) {
    if (typeof value !== "string") return "";
    // eslint-disable-next-line no-control-regex
    return value.replace(/[\u0000-\u001f\u007f]/g, "").trim().slice(0, limit).trim();
}

function cleanKey(value) {
    const key = cleanLabel(value, MAX_KEY);
    return !key || BAD_KEYS.has(key) ? null : key;
}

function cleanText(value, limit, what) {
    if (typeof value !== "string") throw new Error(`${what} must be text`);
    const text = value.replace(/\u0000/g, "");
    if (text.length > limit) throw new Error(`${what} is too long (over ${limit} characters)`);
    return text;
}

function inferType(value) {
    if (typeof value === "boolean") return "BOOLEAN";
    if (typeof value === "number") return Number.isInteger(value) ? "INT" : "FLOAT";
    if (typeof value === "string") return "STRING";
    throw new Error("unsupported value (only numbers, text and true/false)");
}

// Returns {type, value} or throws.
function cleanValue(type, value) {
    const t = type == null ? inferType(value) : String(type).toUpperCase();
    if (!SUPPORTED_TYPES.includes(t)) throw new Error(`unknown type "${String(type).slice(0, 20)}"`);
    if (t === "BOOLEAN") {
        if (typeof value !== "boolean") throw new Error("a BOOLEAN value must be true or false");
        return { type: t, value };
    }
    if (t === "INT" || t === "FLOAT") {
        if (typeof value !== "number") throw new Error(`a ${t} value must be a number`);
        if (!Number.isFinite(value)) throw new Error("the number is not finite");
        if (t === "INT") {
            if (!Number.isInteger(value)) throw new Error("an INT value must be a whole number");
            if (Math.abs(value) > SAFE_INT) throw new Error("the number is out of range");
        } else if (Math.abs(value) > 1e300) {
            throw new Error("the number is out of range");
        }
        return { type: t, value };
    }
    return { type: t, value: cleanText(value, t === "COMBO" ? MAX_COMBO : MAX_STRING, `a ${t} value`) };
}

function cleanHint(hint) {
    if (!isPlainObject(hint)) return null;
    const out = {};
    for (const field of ["node", "widget"]) {
        const text = cleanLabel(hint[field], MAX_HINT);
        if (text) out[field] = text;
    }
    return Object.keys(out).length ? out : null;
}

function cleanParam(entry) {
    let hint = null;
    let clean;
    if (isPlainObject(entry)) {
        if (entry.type != null && typeof entry.type !== "string") throw new Error("type must be text");
        hint = cleanHint(entry.hint);
        clean = cleanValue(entry.type ?? null, entry.value);
    } else {
        clean = cleanValue(null, entry);
    }
    return hint ? { ...clean, hint } : clean;
}

function cleanRecipe(raw, position, warnings) {
    if (!isPlainObject(raw)) {
        warnings.push(`recipe ${position}: not an object - skipped`);
        return null;
    }
    const name = cleanLabel(raw.name, MAX_NAME) || `Recipe ${position}`;
    const paramsRaw = raw.params ?? {};
    if (!isPlainObject(paramsRaw)) {
        warnings.push(`'${name}': params must be an object - skipped`);
        return null;
    }
    const params = {};
    let count = 0;
    for (const [keyRaw, entry] of Object.entries(paramsRaw)) {
        if (count >= MAX_PARAMS) {
            warnings.push(`'${name}': more than ${MAX_PARAMS} entries - the rest was dropped`);
            break;
        }
        const key = cleanKey(keyRaw);
        if (key === null) {
            warnings.push(`'${name}': invalid key ${JSON.stringify(String(keyRaw).slice(0, 30))} - skipped`);
            continue;
        }
        try {
            params[key] = cleanParam(entry);
            count++;
        } catch (e) {
            warnings.push(`'${name}': '${key}' skipped - ${e.message}`);
        }
    }
    return { name, params };
}

// Accepts {recipes:[...]}, a bare list, or one recipe object. Throws if there
// is no usable recipe list. Returns {recipes, warnings}.
function cleanDoc(data) {
    if (isPlainObject(data) && !own(data, "recipes") && own(data, "params")) data = { recipes: [data] };
    else if (Array.isArray(data)) data = { recipes: data };
    if (!isPlainObject(data) || !Array.isArray(data.recipes)) {
        throw new Error('not a recipe file (expected {"version": 1, "recipes": [...]})');
    }
    const warnings = [];
    let raw = data.recipes;
    if (raw.length > MAX_RECIPES) {
        warnings.push(`more than ${MAX_RECIPES} recipes - the rest was dropped`);
        raw = raw.slice(0, MAX_RECIPES);
    }
    const recipes = [];
    raw.forEach((item, i) => {
        const recipe = cleanRecipe(item, i + 1, warnings);
        if (recipe) recipes.push(recipe);
    });
    return { recipes, warnings };
}

// ---------------------------------------------------------------------------
// Matching a recipe value to a connected widget
// ---------------------------------------------------------------------------

// Windows lists "sub\model.safetensors", Linux "sub/model.safetensors" - a
// recipe made on one and loaded on the other must still match.
const normName = (s) => String(s).replace(/\\/g, "/");
const baseName = (s) => normName(s).split("/").pop().toLowerCase();

// Finds `wanted` in `options`: exact (ignoring the slash style), then
// case-insensitive, then by file name alone when that is unambiguous.
// Returns {value, note?} with the list's own spelling, or null.
function matchOption(options, wanted) {
    const w = normName(wanted);
    let hit = options.find((o) => normName(o) === w);
    if (hit !== undefined) return { value: hit };
    const lw = w.toLowerCase();
    hit = options.find((o) => normName(o).toLowerCase() === lw);
    if (hit !== undefined) return { value: hit };
    const wb = baseName(w);
    const candidates = options.filter((o) => baseName(o) === wb);
    if (candidates.length === 1) return { value: candidates[0], note: "matched by file name" };
    return null;
}

function typeCompatible(entryType, targetType) {
    const numeric = (t) => t === "INT" || t === "FLOAT";
    const texty = (t) => t === "STRING" || t === "COMBO";
    return entryType === targetType || (numeric(entryType) && numeric(targetType)) || (texty(entryType) && texty(targetType));
}

// Checks one recipe value against a connected widget. `target` carries type,
// options (combo) and widgetOptions (min / max). Returns {ok, value, note} or
// {ok:false, reason}.
function coerceValue(target, entryType, value) {
    const fail = (reason) => ({ ok: false, reason });
    if (!typeCompatible(entryType, target.type)) return fail(`a ${entryType} value doesn't fit a ${target.type} parameter`);
    switch (target.type) {
        case "INT":
        case "FLOAT": {
            if (typeof value !== "number" || !Number.isFinite(value)) return fail("not a number");
            if (target.type === "INT" && !Number.isInteger(value)) return fail(`${value} is not a whole number`);
            const lo = Number.isFinite(target.widgetOptions?.min) ? target.widgetOptions.min : -Infinity;
            const hi = Number.isFinite(target.widgetOptions?.max) ? target.widgetOptions.max : Infinity;
            const limited = clamp(value, lo, hi);
            const out = target.type === "INT" ? Math.round(limited) : roundFloat(limited);
            return { ok: true, value: out, note: limited !== value ? `limited to ${out}` : undefined };
        }
        case "BOOLEAN":
            return typeof value === "boolean" ? { ok: true, value } : fail("not true / false");
        case "STRING":
            return typeof value === "string" ? { ok: true, value } : fail("not text");
        case "COMBO": {
            if (typeof value !== "string") return fail("not text");
            const hit = matchOption(target.options ?? [], value);
            return hit ? { ok: true, value: hit.value, note: hit.note } : fail(`'${value.slice(0, 80)}' is not in the list`);
        }
        default:
            return fail("unsupported parameter type");
    }
}

function defaultValue(target) {
    const lo = Number.isFinite(target.widgetOptions?.min) ? target.widgetOptions.min : -Infinity;
    const hi = Number.isFinite(target.widgetOptions?.max) ? target.widgetOptions.max : Infinity;
    switch (target.type) {
        case "INT":
        case "FLOAT":
            return clamp(0, lo, hi);
        case "BOOLEAN":
            return false;
        case "COMBO":
            return target.options?.[0] ?? "";
        default:
            return "";
    }
}

// Step of the number box's arrows: the widget's own step, so the arrows behave
// like the original widget's. ComfyUI keeps it in options.step2 (newer
// frontends) or as options.step = 10 x the real step (older ones). Without
// any of that a FLOAT steps by its precision (round / precision, else 0.01),
// an INT by 1.
function stepFor(target) {
    const o = target.widgetOptions ?? {};
    let step = null;
    if (Number.isFinite(o.step2) && o.step2 > 0) step = o.step2;
    else if (Number.isFinite(o.step) && o.step > 0) step = o.step / 10;
    if (target.type === "INT") return step !== null && Number.isInteger(step) && step >= 1 ? step : 1;
    if (step === null) {
        let decimals = 2;
        if (Number.isFinite(o.precision) && o.precision >= 0) decimals = Math.min(6, Math.round(o.precision));
        else if (Number.isFinite(o.round) && o.round > 0) decimals = clamp(Math.ceil(-Math.log10(o.round) - 1e-9), 0, 6);
        step = decimals > 0 ? 10 ** -decimals : 1;
    }
    return roundFloat(step);
}

// Pure core of "Load Recipe": slots = [{s, key, target}] (the CONNECTED
// parameter outputs). Returns what to do with each slot and which recipe keys
// matched nothing. Applying is left to the caller.
function planApply(slots, recipe) {
    const entries = recipe.params;
    const used = new Set();
    const picks = new Map(); // slot index -> {key, entry, via}
    for (const { s, key } of slots) {
        if (key && own(entries, key)) {
            picks.set(s, { key, entry: entries[key], via: "key" });
            used.add(key);
        }
    }
    for (const { s, target } of slots) {
        if (picks.has(s) || !target.widgetName) continue;
        const candidates = Object.entries(entries).filter(
            ([k, e]) => !used.has(k) && e.hint?.widget === target.widgetName && typeCompatible(e.type, target.type)
        );
        if (candidates.length === 1) {
            const [k, entry] = candidates[0];
            picks.set(s, { key: k, entry, via: "widget name" });
            used.add(k);
        }
    }
    const results = slots.map(({ s, key, target }) => {
        const pick = picks.get(s);
        if (!pick) return { s, key, status: "unchanged" };
        const res = coerceValue(target, pick.entry.type, pick.entry.value);
        if (!res.ok) return { s, key, status: "rejected", reason: res.reason };
        const notes = [res.note, pick.via === "widget name" ? `matched by widget name from '${pick.key}'` : null].filter(Boolean);
        return { s, key, status: "applied", value: res.value, note: notes.join("; ") || undefined };
    });
    const unusedKeys = Object.keys(entries).filter((k) => !used.has(k));
    return { results, unusedKeys };
}

// ---------------------------------------------------------------------------
// What is an output connected to?
// ---------------------------------------------------------------------------

function describeInput(targetNode, slot) {
    const input = targetNode?.inputs?.[slot];
    if (!input) return null;
    const widgetName = input.widget?.name;
    const widget = widgetName ? targetNode.widgets?.find((w) => w.name === widgetName) : undefined;

    let type = input.type;
    if (Array.isArray(type) || widget?.type === "combo") type = "COMBO";
    type = String(type ?? "").toUpperCase();
    if (!SUPPORTED_TYPES.includes(type)) return { supported: false, type, name: input.name };
    if (type === "COMBO" && !widget) return { supported: false, type, name: input.name };

    const param = input.label || input.localized_name || input.name;
    const values = widget?.options?.values;
    const list = typeof values === "function" ? values(widget, targetNode) : values;
    return {
        supported: true,
        type,
        param,
        widgetName: widgetName ?? input.name,
        nodeClass: String(targetNode.comfyClass ?? targetNode.type ?? ""),
        label: `${param} (${targetNode.title || targetNode.type})`,
        options: type === "COMBO" ? (Array.isArray(list) ? list.map(String) : []) : null,
        widgetOptions: widget?.options ?? {},
        currentValue: widget?.value,
    };
}

// Links leaving an output slot. The graph's own link table is the source of
// truth - output.links[] can be stale right after a disconnect.
function outputLinkList(node, outputIndex, ignoreLinkId) {
    const graph = node.graph;
    if (!graph) return [];
    const table = graph.links;
    let links;
    if (table instanceof Map) links = [...table.values()];
    else if (table && typeof table === "object") links = Object.values(table);
    else links = (node.outputs?.[outputIndex]?.links ?? []).map((id) => getLink(graph, id));
    return links
        .filter((l) => l && String(l.origin_id) === String(node.id) && l.origin_slot === outputIndex && l.id !== ignoreLinkId)
        .sort((a, b) => a.id - b.id);
}

function slotTargets(node, s, ignoreLinkId) {
    const result = [];
    for (const link of outputLinkList(node, s + FIRST_PARAM_OUTPUT, ignoreLinkId)) {
        const targetNode = node.graph?.getNodeById?.(link.target_id);
        const info = describeInput(targetNode, link.target_slot);
        if (info) result.push({ ...info, linkId: link.id, targetNode, targetSlot: link.target_slot });
    }
    return result;
}

// The first supported target defines the parameter.
function slotTarget(node, s, ignoreLinkId) {
    const targets = slotTargets(node, s, ignoreLinkId).filter((t) => t.supported);
    if (!targets.length) return null;
    const main = targets[0];
    const extra = targets.length - 1;
    return extra > 0 ? { ...main, label: `${main.label} +${extra}` } : main;
}

// ---------------------------------------------------------------------------
// State (kept in node._mykeeRc, saved in the hidden recipe_config widget)
// ---------------------------------------------------------------------------

function newState() {
    return { version: 1, params: [], recipes: [], index: 0, file: "", dirty: false, batch: { enabled: false, sendParams: false, selected: [] } };
}

// The widget value may come from a shared workflow: treat it as untrusted.
function parseState(raw) {
    let data = {};
    try {
        data = JSON.parse(raw || "{}") || {};
    } catch {
        data = {};
    }
    const st = newState();
    if (Array.isArray(data.params)) {
        for (let s = 0; s < MAX_SLOTS; s++) {
            const p = data.params[s];
            if (!isPlainObject(p)) continue;
            const v = p.value;
            const scalar = typeof v === "number" ? Number.isFinite(v) : typeof v === "boolean" || typeof v === "string";
            st.params[s] = {
                key: cleanKey(p.key) ?? "",
                value: scalar ? (typeof v === "string" ? v.slice(0, MAX_STRING) : v) : undefined,
            };
        }
    }
    try {
        st.recipes = cleanDoc({ recipes: data.recipes ?? [] }).recipes;
    } catch {
        st.recipes = [];
    }
    st.index = st.recipes.length ? clamp(Number.isInteger(data.index) ? data.index : 0, 0, st.recipes.length - 1) : 0;
    st.file = cleanLabel(data.file, 200);
    st.dirty = data.dirty === true;
    if (isPlainObject(data.batch)) {
        const selected = Array.isArray(data.batch.selected) ? data.batch.selected : [];
        st.batch = {
            enabled: data.batch.enabled === true,
            sendParams: data.batch.sendParams === true,
            selected: [...new Set(selected.filter((n) => typeof n === "string").map((n) => cleanLabel(n, MAX_NAME)).filter(Boolean))].slice(0, MAX_RECIPES),
        };
    }
    return st;
}

const S = (node) => (node._mykeeRc ??= parseState(findWidget(node, CONFIG_WIDGET)?.value));

function persist(node) {
    const widget = findWidget(node, CONFIG_WIDGET);
    if (widget) widget.value = JSON.stringify(S(node));
    node.graph?.setDirtyCanvas?.(true, false);
}

function param(node, s) {
    const st = S(node);
    return (st.params[s] ??= { key: "", value: undefined });
}

function slotCount(node) {
    const w = findWidget(node, "slots");
    const n = Number.isFinite(Number(w?.value)) ? Math.round(Number(w.value)) : 3;
    return clamp(n, 1, MAX_SLOTS);
}

function paramKey(node, s, target) {
    return param(node, s).key || target?.widgetName || `p${s + 1}`;
}

function uniqueKey(node, base, exceptSlot) {
    const taken = new Set();
    for (let s = 0; s < slotCount(node); s++) {
        if (s !== exceptSlot && S(node).params[s]?.key) taken.add(S(node).params[s].key);
    }
    let key = cleanKey(base) ?? "param";
    if (!taken.has(key)) return key;
    for (let i = 2; i < 100; i++) {
        const candidate = `${key.slice(0, MAX_KEY - 4)}_${i}`;
        if (!taken.has(candidate)) return candidate;
    }
    return key;
}

// What the backend gets: only the connected outputs, with their values.
function resolveParams(node) {
    const params = {};
    for (let s = 0; s < slotCount(node); s++) {
        const target = slotTarget(node, s);
        if (!target) continue;
        let value = param(node, s).value;
        if (value === undefined) value = defaultValue(target);
        params[`p${s + 1}`] = { type: target.type, value };
    }
    const config = { version: 1, params };
    const batch = S(node).batch;
    if (batch.enabled) {
        // The queue wrapper below turns this into one prompt per recipe (and
        // strips it again), so the backend only ever sees one recipe at a time.
        config.batch = { items: planBatch(connectedSlots(node), S(node).recipes, batch.selected, batch.sendParams) };
    }
    return config;
}

// ---------------------------------------------------------------------------
// Batch: one prompt per selected recipe
// ---------------------------------------------------------------------------

const MODEL_EXT = /\.(safetensors|ckpt|pt|pth|bin|gguf|sft|onnx|pkl)$/i;

// A value as text for the plot labels. short: model files lose folder and
// extension, long texts are cut. Mirrors the XYZ Plot's labels.
function valueText(type, value, short = true) {
    let text;
    if (type === "FLOAT") {
        text = Number(value).toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
        if (text === "" || text === "-0") text = "0";
    } else if (type === "BOOLEAN") {
        text = value ? "true" : "false";
    } else {
        text = String(value);
    }
    if (short) {
        if (MODEL_EXT.test(text)) text = normName(text).split("/").pop().replace(MODEL_EXT, "");
        text = text.replace(/\s+/g, " ").trim();
        if (text.length > 40) text = text.slice(0, 39) + "…";
    } else if (text.length > 200) {
        text = text.slice(0, 199) + "…";
    }
    return text;
}

// Pure core of a batch run. slots = [{s, key, target, current}] (the connected
// outputs and the value now on the node); recipes = the recipes on the node;
// selected = recipe names ticked in the list. Per ticked recipe (in list
// order) it returns what every connected output gets: the recipe's value, or
// the node's own value when the recipe has no (usable) entry for it.
function planBatch(slots, recipes, selected, sendParams) {
    const wanted = new Set(selected);
    const items = [];
    for (const recipe of recipes) {
        if (!wanted.has(recipe.name)) continue;
        const { results } = planApply(slots, recipe);
        const params = {};
        const short = [];
        const full = [];
        const warn = [];
        for (const { s, target, current } of slots) {
            const r = results.find((x) => x.s === s);
            const applied = r?.status === "applied";
            params[`p${s + 1}`] = { type: target.type, value: applied ? r.value : current };
            if (applied) {
                short.push(`${r.key}: ${valueText(target.type, r.value)}`);
                full.push(`${r.key}: ${valueText(target.type, r.value, false)}`);
            } else if (r?.status === "rejected") {
                warn.push(`${recipe.name} / ${r.key}: ${r.reason}`);
            }
        }
        items.push({
            name: recipe.name,
            params,
            label: (sendParams ? [recipe.name, ...short].join("\n") : recipe.name).slice(0, MAX_LABEL),
            full: (sendParams ? [recipe.name, ...full].join("\n") : recipe.name).slice(0, MAX_LABEL * 2),
            warn,
        });
    }
    return items;
}

function connectedSlots(node) {
    const slots = [];
    for (let s = 0; s < slotCount(node); s++) {
        const target = slotTarget(node, s);
        if (!target) continue;
        let current = param(node, s).value;
        if (current === undefined) current = defaultValue(target);
        slots.push({ s, key: paramKey(node, s, target), target, current });
    }
    return slots;
}

// ---------------------------------------------------------------------------
// Dynamic outputs
// ---------------------------------------------------------------------------

// Workflows saved before the xyz_info output existed have recipe_name, p1 ..
// pn: insert the new output and move the saved links of p1 .. pn with it.
function migrateOutputs(node) {
    const outs = node.outputs;
    if (!Array.isArray(outs) || !outs.length || outs[1]?.name === XYZ_OUTPUT) return;
    outs.splice(1, 0, { name: XYZ_OUTPUT, type: XYZ_INFO_TYPE, links: null });
    const links = node.graph?.links;
    for (let i = FIRST_PARAM_OUTPUT; i < outs.length; i++) {
        for (const id of outs[i]?.links ?? []) {
            const link = links?.get ? links.get(id) : links?.[id];
            if (link) link.origin_slot = i;
        }
    }
}

function updateOutputSlot(node, s, ignoreLinkId) {
    const output = node.outputs?.[s + FIRST_PARAM_OUTPUT];
    if (!output) return;
    const target = slotTarget(node, s, ignoreLinkId);
    output.type = target ? target.type : "*";
    output.label = target ? paramKey(node, s, target) : `p${s + 1}`;
}

// Makes the number of outputs match `n` (1 + n: recipe_name comes first).
// Outputs are only ever removed from the end, so the slot numbers the backend
// knows stay valid. Returns the count that was really set.
function setOutputCount(node, wanted) {
    let n = clamp(Math.round(Number(wanted)) || 1, 1, MAX_SLOTS);
    let lowest = 1; // can't go below the highest connected output
    for (let s = 0; s < MAX_SLOTS; s++) {
        if (outputLinkList(node, s + FIRST_PARAM_OUTPUT).length) lowest = s + 1;
    }
    if (n < lowest) {
        toast(`Output p${lowest} is still connected - disconnect it first.`);
        n = lowest;
    }
    while (node.outputs.length - FIRST_PARAM_OUTPUT < n) node.addOutput(`p${node.outputs.length - FIRST_PARAM_OUTPUT + 1}`, "*");
    while (node.outputs.length - FIRST_PARAM_OUTPUT > n) {
        if (node.outputs[node.outputs.length - 1]?.links?.length) break; // never drop a connected output
        node.removeOutput(node.outputs.length - 1);
    }
    n = node.outputs.length - FIRST_PARAM_OUTPUT;
    const widget = findWidget(node, "slots");
    if (widget && widget.value !== n) widget.value = n;
    for (let s = 0; s < n; s++) updateOutputSlot(node, s);
    return n;
}

function checkNewLink(node, s, linkInfo) {
    const targetNode = node.graph?.getNodeById?.(linkInfo.target_id);
    const info = describeInput(targetNode, linkInfo.target_slot);
    const first = slotTargets(node, s).find((t) => t.supported && t.linkId !== linkInfo.id);

    let reason = null;
    if (!info?.supported) {
        reason = `"${info?.name ?? "this input"}" can't be a recipe parameter - only INT, FLOAT, STRING, BOOLEAN and combo widgets can.`;
    } else if (first && first.type !== info.type) {
        reason = `Output p${s + 1} already drives a ${first.type} parameter; it can't also drive a ${info.type} one.`;
    } else if (first && info.type === "COMBO" && first.options.join("\n") !== info.options.join("\n")) {
        reason = `Output p${s + 1} already drives a combo with a different option list.`;
    }
    if (reason) {
        toast(reason);
        // Not during the event itself - the link is still being set up.
        setTimeout(() => targetNode?.disconnectInput?.(linkInfo.target_slot), 0);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Panel UI
// ---------------------------------------------------------------------------

const PANEL_CSS = `
.mykee-rc { display:flex; flex-direction:column; gap:6px; width:100%; height:100%;
  overflow-y:auto; box-sizing:border-box; padding:2px 4px 8px; font:12px sans-serif;
  color:var(--input-text, #ddd); }
.mykee-rc-slot { border:1px solid var(--border-color, #444); border-radius:6px; padding:5px 6px;
  background:var(--comfy-input-bg, #222); display:flex; flex-direction:column; gap:4px; }
.mykee-rc-head { display:flex; align-items:center; gap:6px; }
.mykee-rc-badge { font-weight:bold; border-radius:4px; padding:0 6px; color:#111; white-space:nowrap; background:${BADGE_COLOR}; }
.mykee-rc-muted { opacity:.6; font-style:italic; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.mykee-rc-sub { opacity:.7; font-size:11px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.mykee-rc input[type=text], .mykee-rc input[type=number], .mykee-rc textarea, .mykee-rc select {
  width:100%; box-sizing:border-box; background:var(--comfy-menu-bg, #111); color:inherit;
  border:1px solid var(--border-color, #444); border-radius:4px; padding:3px 5px; font:12px sans-serif; }
.mykee-rc textarea { font:12px/1.35 Consolas, Menlo, monospace; resize:vertical; }
.mykee-rc .key { flex:1; font-weight:bold; }
.mykee-rc .dup { border-color:#f77; }
.mykee-rc select.missing { border-color:#f77; }
.mykee-rc button { background:var(--comfy-menu-bg, #333); color:inherit; border:1px solid var(--border-color, #555);
  border-radius:4px; padding:1px 7px; cursor:pointer; font-size:11px; }
.mykee-rc button:disabled { opacity:.4; cursor:default; }
.mykee-rc-pane { display:flex; flex-direction:column; gap:6px; width:100%; height:100%; box-sizing:border-box;
  padding:2px 4px 6px; font:12px sans-serif; color:var(--input-text, #ddd); }
.mykee-rc-pane textarea { flex:1 1 auto; min-height:90px; resize:none; }
.mykee-rc-row { display:flex; gap:6px; align-items:center; flex:0 0 auto; }
.mykee-rc-name { flex:1; font-weight:bold; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.mykee-rc-status { flex:0 0 auto; font-size:11px; opacity:.85; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.mykee-rc-status.ok { color:#8fd18f; }
.mykee-rc-status.bad { color:#f77; opacity:1; }
.mykee-rc-check { display:flex; gap:6px; align-items:center; cursor:pointer; }
.mykee-rc-batch { display:flex; flex-direction:column; gap:6px; flex:1 1 auto; min-height:0; }
.mykee-rc-list { flex:1 1 auto; min-height:80px; overflow-y:auto; border:1px solid var(--border-color, #444);
  border-radius:4px; padding:3px 6px; background:var(--comfy-input-bg, #222); }
.mykee-rc-list label { display:flex; gap:6px; align-items:center; cursor:pointer; padding:1px 0; }
.mykee-rc-list label span { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
`;

function ensureCss() {
    if (document.getElementById("mykee-rc-css")) return;
    const style = document.createElement("style");
    style.id = "mykee-rc-css";
    style.textContent = PANEL_CSS;
    document.head.appendChild(style);
}

function el(tag, props = {}, children = []) {
    const element = document.createElement(tag);
    Object.assign(element, props);
    for (const child of children) if (child) element.append(child);
    return element;
}

// Keeps canvas shortcuts / zoom out of the panel's inputs.
function isolate(element) {
    for (const type of ["keydown", "keyup", "wheel"]) {
        element.addEventListener(type, (e) => e.stopPropagation());
    }
    return element;
}

// ---- tabs (same mechanism as the Text Analyzer) ----------------------------

function currentTab(node) {
    const t = node.properties?.[TAB_PROP];
    return TABS.some((x) => x.id === t) ? t : DEFAULT_TAB;
}

// Shows / hides a widget (incl. its DOM element) and gives a hidden one no height.
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
    const element = widget.element || widget.inputEl;
    if (element) {
        element.hidden = !visible;
        element.style.display = visible ? "" : "none";
    }
}

function applyTab(node, resize) {
    const current = currentTab(node);
    for (const [tab, widgets] of Object.entries(node._mykeeRcTabWidgets ?? {})) {
        for (const w of widgets) {
            const visible = tab === current;
            if ((w.hidden === true) === !visible) continue;
            setWidgetVisible(w, visible);
        }
    }
    if (resize) node.setSize([node.size[0], node.computeSize()[1]]);
    node.setDirtyCanvas?.(true, true);
}

function createTabBar(node) {
    const margin = 15;
    const gap = 4;
    const h = 24;
    return {
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
            if (TABS[i].id === currentTab(n)) return true;
            commitAll(n); // a half-typed edit is stored (or reported) before the tab goes away
            n.properties = n.properties || {};
            const heights = { ...(n.properties[TAB_SIZE_PROP] || {}) };
            heights[currentTab(n)] = n.size[1]; // remember the height of the tab we leave
            n.properties[TAB_SIZE_PROP] = heights;
            n.properties[TAB_PROP] = TABS[i].id;
            applyTab(n, true);
            // Parameters is content-driven (applyTab just fitted it); the text-box tabs get back the height they had.
            const wanted = Number(heights[TABS[i].id]);
            if (TABS[i].id !== "params" && Number.isFinite(wanted)) n.setSize([n.size[0], Math.max(wanted, n.computeSize()[1])]);
            n._mykeeRcRefreshAll?.();
            if (TABS[i].id === "params") {
                // The panel was hidden (so it had no measurable height, only an
                // estimate): measure the real content now that it is visible.
                n._mykeeRcMeasure?.();
                requestAnimationFrame(() => n._mykeeRcMeasure?.());
            }
            return true;
        },
    };
}

// Fits the node's height to its content; the width stays the user's.
function fitHeight(node, shrink = true) {
    const size = node.computeSize?.();
    if (!size || !Number.isFinite(size[1])) return;
    const height = shrink ? size[1] : Math.max(node.size[1], size[1]);
    node.setSize([Math.max(node.size[0], size[0]), height]);
    node.graph?.setDirtyCanvas?.(true, true);
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

// ---- Parameters tab ---------------------------------------------------------

function boxHeights(node) {
    const h = node.properties?.[BOX_HEIGHT_PROP];
    return isPlainObject(h) ? h : {};
}

// A resizable text box keeps the height the user gave it (saved with the workflow).
function rememberBoxHeight(node, area, name) {
    area.addEventListener("pointerup", () => {
        const height = Math.round(area.offsetHeight);
        if (!(height >= 40) || height === Math.round(Number(boxHeights(node)[name]))) return;
        node.properties = node.properties || {};
        node.properties[BOX_HEIGHT_PROP] = { ...boxHeights(node), [name]: height };
        node._mykeeRcMeasure?.();
    });
}

function renderSlot(node, s, onChange) {
    const target = slotTarget(node, s);
    const box = el("div", { className: "mykee-rc-slot" });
    const badge = el("span", { className: "mykee-rc-badge", textContent: `p${s + 1}` });

    if (!target) {
        box.append(
            el("div", { className: "mykee-rc-head" }, [
                badge,
                el("span", { className: "mykee-rc-muted", textContent: `drag output p${s + 1} onto any widget` }),
            ])
        );
        return box;
    }

    const p = param(node, s);
    if (!p.key) p.key = uniqueKey(node, target.widgetName, s);
    if (p.value === undefined) p.value = defaultValue(target);

    // key (the name the recipe stores the value under)
    const key = isolate(el("input", { type: "text", className: "key", value: p.key, maxLength: MAX_KEY,
        title: "The key this parameter is stored under in a recipe. Rename it to make different widgets / nodes share one recipe entry." }));
    const dup = () => {
        const others = [];
        for (let o = 0; o < slotCount(node); o++) if (o !== s && slotTarget(node, o)) others.push(param(node, o).key);
        key.classList.toggle("dup", others.includes(key.value.trim()));
    };
    key.addEventListener("input", dup);
    key.onchange = () => {
        const clean = cleanKey(key.value);
        if (clean === null) {
            toast(`'${key.value}' can't be used as a key.`);
            key.value = p.key;
        } else {
            p.key = clean;
            key.value = clean;
            updateOutputSlot(node, s);
        }
        dup();
        onChange();
    };
    dup();
    const refresh = el("button", { textContent: "↻", title: "Re-read the connected parameter (e.g. after a model list refresh)" });
    refresh.onclick = () => node._mykeeRcRenderParams?.();
    box.append(el("div", { className: "mykee-rc-head" }, [badge, key, refresh]));
    box.append(el("div", { className: "mykee-rc-sub", textContent: `${target.label} · ${target.type}`, title: target.label }));

    // value editor
    if (target.type === "INT" || target.type === "FLOAT") {
        const input = isolate(el("input", { type: "number", step: String(stepFor(target)) }));
        const lo = target.widgetOptions?.min;
        const hi = target.widgetOptions?.max;
        if (Number.isFinite(lo)) input.min = String(lo);
        if (Number.isFinite(hi)) input.max = String(hi);
        input.value = String(p.value);
        input.onchange = () => {
            const res = coerceValue(target, target.type, Number(input.value));
            if (res.ok) p.value = res.value;
            input.value = String(p.value);
            onChange();
        };
        box.append(input);
    } else if (target.type === "STRING") {
        const area = isolate(el("textarea", { rows: 3, value: String(p.value ?? ""), spellcheck: false }));
        const stored = Number(boxHeights(node)[`p${s + 1}`]);
        if (stored >= 40) area.style.height = `${stored}px`;
        area.addEventListener("input", () => {
            p.value = area.value.slice(0, MAX_STRING);
            onChange();
        });
        rememberBoxHeight(node, area, `p${s + 1}`);
        box.append(area);
    } else if (target.type === "BOOLEAN") {
        const check = el("input", { type: "checkbox", checked: p.value === true });
        check.onchange = () => {
            p.value = check.checked;
            onChange();
        };
        box.append(el("label", { className: "mykee-rc-check" }, [check, document.createTextNode("true")]));
    } else if (target.type === "COMBO") {
        const select = isolate(el("select", {}));
        const current = String(p.value ?? "");
        const listed = target.options.includes(current);
        if (current && !listed) {
            // The remembered entry isn't in the list (any more) - show that
            // instead of silently switching to something else.
            select.append(el("option", { value: current, textContent: `⚠ ${current} (not in list)` }));
            select.classList.add("missing");
        }
        for (const option of target.options) select.append(el("option", { value: option, textContent: option }));
        select.value = current;
        select.title = current;
        select.onchange = () => {
            p.value = select.value;
            select.title = select.value;
            select.classList.toggle("missing", !target.options.includes(select.value));
            onChange();
        };
        box.append(select);
    }
    return box;
}

// Rough panel height until the real one is measured.
function estimateParamsHeight(node) {
    let height = 0;
    for (let s = 0; s < slotCount(node); s++) {
        const t = slotTarget(node, s);
        const box = !t ? 36 : t.type === "STRING" ? 150 : 100;
        height += PANEL_GAP + box;
    }
    return height + PANEL_BOTTOM_PADDING;
}

// ---- Recipe / JSON tabs -----------------------------------------------------

function setStatus(element, text, ok) {
    element.textContent = text;
    element.classList.toggle("ok", ok === true);
    element.classList.toggle("bad", ok === false);
    element.title = text;
}

function recipeText(recipe) {
    return JSON.stringify(recipe, null, 2);
}

function docText(st) {
    return JSON.stringify({ version: 1, recipes: st.recipes }, null, 2);
}

// Writes `text` into a text box unless somebody is typing in it.
function setTextIfIdle(area, text) {
    if (document.activeElement === area) return;
    area.value = text;
}

// Edits in the two text boxes are parsed as you type (status line) and stored
// when they are valid: after a short pause, on blur, and before anything that
// reads the recipes. commitAll() returns false when a box holds invalid JSON.
function commitAll(node) {
    const ui = node._mykeeRcUI;
    if (!ui) return true;
    return ui.commitRecipe() && ui.commitJson();
}

function sendToRecipe(node) {
    if (!commitAll(node)) {
        toast("Fix the JSON text first (see the red status line).");
        return;
    }
    const st = S(node);
    const name = cleanLabel(findWidget(node, "recipe_name")?.value, MAX_NAME);
    if (!name) {
        toast("Enter a recipe name first (recipe_name).");
        return;
    }
    const entries = {};
    const seen = new Map();
    for (let s = 0; s < slotCount(node); s++) {
        const target = slotTarget(node, s);
        if (!target) continue;
        const key = paramKey(node, s, target);
        if (seen.has(key)) {
            toast(`Outputs p${seen.get(key) + 1} and p${s + 1} have the same key '${key}' - rename one of them.`);
            return;
        }
        seen.set(key, s);
        try {
            const clean = cleanParam({ type: target.type, value: param(node, s).value ?? defaultValue(target) });
            entries[key] = { ...clean, hint: cleanHint({ node: target.nodeClass, widget: target.widgetName }) ?? undefined };
            if (!entries[key].hint) delete entries[key].hint;
        } catch (e) {
            toast(`p${s + 1} (${key}): ${e.message}`);
            return;
        }
    }
    if (!Object.keys(entries).length) {
        toast("Connect at least one parameter output to a widget first.");
        return;
    }
    const lower = name.toLowerCase();
    let index = st.recipes.findIndex((r) => r.name.toLowerCase() === lower);
    let message;
    if (index >= 0) {
        const before = st.recipes[index].params;
        const added = Object.keys(entries).filter((k) => !own(before, k)).length;
        st.recipes[index] = { name: st.recipes[index].name, params: { ...before, ...entries } };
        message = `Recipe '${st.recipes[index].name}' updated (${Object.keys(entries).length} value${Object.keys(entries).length === 1 ? "" : "s"}${added ? `, ${added} new` : ""}).`;
    } else {
        if (st.recipes.length >= MAX_RECIPES) {
            toast(`A recipe file can hold at most ${MAX_RECIPES} recipes.`);
            return;
        }
        st.recipes.push({ name, params: entries });
        index = st.recipes.length - 1;
        message = `Recipe '${name}' added (${Object.keys(entries).length} value${Object.keys(entries).length === 1 ? "" : "s"}).`;
    }
    st.index = index;
    st.dirty = true;
    persist(node);
    node._mykeeRcRefreshAll?.();
    toast(message + " Save it to a file on the 'JSON file' tab.", "success", 4500);
}

function applyRecipeToNode(node, recipe) {
    const slots = [];
    for (let s = 0; s < slotCount(node); s++) {
        const target = slotTarget(node, s);
        if (target) slots.push({ s, key: paramKey(node, s, target), target });
    }
    if (!slots.length) {
        toast("No parameter output is connected to a widget - nothing to apply.");
        return;
    }
    const { results, unusedKeys } = planApply(slots, recipe);
    const applied = [];
    const unchanged = [];
    const rejected = [];
    for (const r of results) {
        if (r.status === "applied") {
            param(node, r.s).value = r.value;
            applied.push(r.note ? `${r.key} (${r.note})` : r.key);
        } else if (r.status === "rejected") {
            rejected.push(`${r.key}: ${r.reason}`);
        } else {
            unchanged.push(r.key);
        }
    }
    const nameWidget = findWidget(node, "recipe_name");
    if (nameWidget) nameWidget.value = recipe.name;
    persist(node);
    node._mykeeRcRenderParams?.();

    const parts = [applied.length ? `applied: ${applied.join(", ")}` : "nothing applied"];
    if (rejected.length) parts.push(`NOT applied (left unchanged): ${rejected.join("; ")}`);
    if (unchanged.length) parts.push(`not in the recipe (left unchanged): ${unchanged.join(", ")}`);
    if (unusedKeys.length) parts.push(`no output with this key (skipped): ${unusedKeys.join(", ")}`);
    toast(`'${recipe.name}' - ${parts.join(" | ")}`, rejected.length ? "warn" : applied.length ? "success" : "info", rejected.length ? 10000 : 5000);
}

// ---------------------------------------------------------------------------
// Queue fan-out: one prompt per ticked recipe
// ---------------------------------------------------------------------------

const isLinkTo = (value, key) => Array.isArray(value) && value.length === 2 && String(value[0]) === key;

// The batch in a prompt, or null. The batch description is what the node's
// serializeValue put into recipe_config; it comes through the prompt, so it is
// validated here before anything is queued.
function batchFromPrompt(prompt) {
    const output = prompt?.output;
    if (!output) return null;
    const found = [];
    for (const [key, entry] of Object.entries(output)) {
        if (entry?.class_type !== NODE_NAME) continue;
        let config;
        try {
            config = JSON.parse(entry.inputs?.recipe_config ?? "{}");
        } catch {
            continue;
        }
        if (isPlainObject(config?.batch) && Array.isArray(config.batch.items)) found.push({ key, items: config.batch.items });
    }
    if (!found.length) return null;
    if (found.length > 1) {
        throw new Error("Mykee Recipe: only one Recipe node with Batch switched on is supported per workflow. Switch it off on the others.");
    }
    if (Object.values(output).some((entry) => PLOT_CLASSES.has(entry?.class_type))) {
        throw new Error("Mykee Recipe: Batch can't be combined with an XYZ Plot / Simple X Plot node (both fan out the queue). Mute or bypass the plot node, or switch Batch off.");
    }
    const { key } = found[0];
    const items = found[0].items.slice(0, MAX_RECIPES);
    if (!items.length) {
        throw new Error("Mykee Recipe: Batch is on but no recipe is ticked on the Batch tab.");
    }
    const valid = items.every(
        (it) => isPlainObject(it) && typeof it.name === "string" && isPlainObject(it.params)
            && typeof it.label === "string" && typeof it.full === "string"
    );
    if (!valid) throw new Error("Mykee Recipe: the batch description is invalid - reload the page and try again.");
    return { key, items };
}

// The prompt for one recipe: the links from p1 .. pn become the recipe's
// literal values (exactly what a Primitive node would put there), so ComfyUI's
// cache only re-runs what really differs between two recipes. The Recipe node
// itself gets the recipe's name and the batch description for xyz_info.
function buildBatchOutput(output, key, items, cell, runId) {
    const item = items[cell];
    const cellOutput = { ...output };
    for (const [k, entry] of Object.entries(output)) {
        if (!entry?.inputs) continue;
        let inputs = null;
        for (const [name, value] of Object.entries(entry.inputs)) {
            if (!isLinkTo(value, key) || !Number.isInteger(value[1]) || value[1] < FIRST_PARAM_OUTPUT) continue;
            const p = item.params[`p${value[1] - FIRST_PARAM_OUTPUT + 1}`];
            if (!p || !own(p, "value")) continue;
            inputs ??= { ...entry.inputs };
            inputs[name] = p.value;
        }
        if (inputs) cellOutput[k] = { ...entry, inputs };
    }
    const source = cellOutput[key];
    cellOutput[key] = {
        ...source,
        inputs: {
            ...source.inputs,
            recipe_name: item.name,
            recipe_config: JSON.stringify({
                version: 1,
                params: item.params,
                batch: {
                    run_id: runId,
                    cell,
                    labels: items.map((it) => it.label),
                    full: items.map((it) => it.full),
                },
            }),
        },
    };
    return cellOutput;
}

function notify(detail, severity = "info", life = 5000) {
    const t = app.extensionManager?.toast;
    if (t?.add) t.add({ severity, summary: "Mykee Recipe", detail, life });
    else console.info("[Mykee Recipe] " + detail);
}

const baseQueuePrompt = api.queuePrompt;
api.queuePrompt = async function (number, prompt, ...rest) {
    const plan = batchFromPrompt(prompt);
    if (!plan) return baseQueuePrompt.call(this, number, prompt, ...rest);
    const { key, items } = plan;
    if (items.length > BATCH_CONFIRM_LIMIT) {
        const ok = window.confirm(`Mykee Recipe: this will queue ${items.length} prompts (one per recipe). Continue?`);
        if (!ok) throw new Error("Mykee Recipe: queueing cancelled.");
    }
    const warnings = items.flatMap((it) => (Array.isArray(it.warn) ? it.warn.map(String) : []));
    if (warnings.length) {
        notify(`${warnings.length} value${warnings.length === 1 ? "" : "s"} not applied (the node's own value is used): ${warnings.slice(0, 3).join(" | ")}`, "warn", 10000);
    }
    const runId = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
    let response;
    for (let cell = 0; cell < items.length; cell++) {
        const output = buildBatchOutput(prompt.output, key, items, cell, runId);
        response = await baseQueuePrompt.call(this, number, { ...prompt, output }, ...rest);
        const hasErrors = response?.node_errors && Object.keys(response.node_errors).length > 0;
        if (hasErrors || !response?.prompt_id) return response; // let ComfyUI show the error
    }
    notify(`Batch queued: ${items.length} recipe${items.length === 1 ? "" : "s"}.`, "info", 3000);
    return response;
};

// ---------------------------------------------------------------------------
// Node setup
// ---------------------------------------------------------------------------

function setupNode(node) {
    ensureCss();

    const slotsW = findWidget(node, "slots");
    const nameW = findWidget(node, "recipe_name");
    const jsonW = findWidget(node, "json_name");
    const pathW = findWidget(node, "custom_path");
    const configW = findWidget(node, CONFIG_WIDGET);
    if (!slotsW || !nameW || !jsonW || !pathW || !configW) return;

    // Hide the raw JSON widget (the backend spec already asks for that on
    // newer frontends; this covers older ones). The prompt gets only the
    // resolved values, the workflow keeps the whole state.
    configW.hidden = true;
    configW.computeSize = () => [0, -4];
    configW.serializeValue = () => JSON.stringify(resolveParams(node));

    node._mykeeRc = parseState(configW.value);
    setOutputCount(node, slotsW.value);

    const origSlotsCallback = slotsW.callback;
    slotsW.callback = function () {
        const r = origSlotsCallback ? origSlotsCallback.apply(this, arguments) : undefined;
        setOutputCount(node, slotsW.value);
        node._mykeeRcRenderParams?.();
        return r;
    };

    const currentPath = () => String(pathW.value || "").trim();

    // --- Parameters tab: panel + Send button ----------------------------
    const params = el("div", { className: "mykee-rc" });
    let paramsHeight = estimateParamsHeight(node);
    const paramsDom = node.addDOMWidget("recipe_params", "MYKEE_RECIPE_PARAMS", params, {
        serialize: false,
        getMinHeight: () => paramsHeight,
    });
    paramsDom.serialize = false;

    const sendBtn = node.addWidget("button", "➕ Send to Recipe", null, () => sendToRecipe(node));
    sendBtn.serialize = false;
    sendBtn.options = { ...(sendBtn.options || {}), serialize: false };

    // Recipe picker: choosing a recipe loads it into the parameters and fills
    // in recipe_name, so a new recipe can be made from an existing one right
    // here (pick, change a value, type a new name, Send to Recipe).
    const pickerW = node.addWidget("combo", "recipe_picker", "", (value) => {
        if (!value) return;
        if (!commitAll(node)) {
            toast("Fix the JSON text first (see the red status line).");
            syncRecipePicker();
            return;
        }
        const st = S(node);
        const index = st.recipes.findIndex((r) => r.name === value);
        if (index < 0) return;
        st.index = index;
        applyRecipeToNode(node, st.recipes[index]);
        node._mykeeRcRefreshAll?.();
    }, { values: [""], serialize: false });
    pickerW.serialize = false;
    pickerW.tooltip = "Pick a recipe to load it into the parameters (same as 'Load Recipe').";

    // The list follows the recipes on the node; it shows the recipe whose name is in recipe_name.
    const syncRecipePicker = () => {
        const names = S(node).recipes.map((r) => r.name);
        pickerW.options.values = ["", ...names];
        const current = String(nameW.value || "").trim().toLowerCase();
        pickerW.value = names.find((n) => n.toLowerCase() === current) ?? "";
        node.setDirtyCanvas?.(true, true);
    };
    node._mykeeRcSyncPicker = syncRecipePicker;
    const origNameCallback = nameW.callback;
    nameW.callback = function () {
        const r = origNameCallback ? origNameCallback.apply(this, arguments) : undefined;
        syncRecipePicker();
        return r;
    };

    // The Parameters tab's height always follows its content (grows AND
    // shrinks). The two text-box tabs keep the height the user gave them.
    const measure = () => {
        const kids = [...params.children];
        const height = kids.reduce((sum, kid) => sum + kid.offsetHeight, 0);
        if (height > 0) paramsHeight = height + PANEL_GAP * (kids.length - 1) + PANEL_BOTTOM_PADDING;
        if (currentTab(node) === "params") fitHeight(node, true);
    };
    node._mykeeRcMeasure = measure;

    // Anything that changes a parameter box's height later (a text box the
    // user resizes, a list that wraps, fonts loading) re-measures the panel.
    let measureQueued = false;
    const queueMeasure = () => {
        if (measureQueued) return;
        measureQueued = true;
        requestAnimationFrame(() => {
            measureQueued = false;
            measure();
        });
    };
    const observer = typeof ResizeObserver !== "undefined" ? new ResizeObserver(queueMeasure) : null;

    node._mykeeRcRenderParams = () => {
        paramsHeight = estimateParamsHeight(node);
        params.replaceChildren();
        const onChange = () => persist(node);
        observer?.disconnect();
        for (let s = 0; s < slotCount(node); s++) params.append(renderSlot(node, s, onChange));
        for (const box of params.children) observer?.observe(box);
        persist(node);
        if (currentTab(node) === "params") fitHeight(node, true);
        requestAnimationFrame(() => measure());
    };

    // --- Recipe tab ------------------------------------------------------
    const prevBtn = el("button", { textContent: "◀", title: "Previous recipe" });
    const nextBtn = el("button", { textContent: "▶", title: "Next recipe" });
    const counter = el("span", { textContent: "0 / 0" });
    const nameLabel = el("span", { className: "mykee-rc-name" });
    const recipeArea = isolate(el("textarea", {
        spellcheck: false,
        placeholder: "No recipe yet. Set the parameters on the first tab and press 'Send to Recipe', "
            + "press New to write one by hand, or load a recipe file on the JSON file tab.",
    }));
    const recipeStatus = el("div", { className: "mykee-rc-status" });
    const loadRecipeBtn = el("button", { textContent: "⬆ Load Recipe", title: "Put this recipe's values into the parameters (first tab)" });
    const newBtn = el("button", { textContent: "➕ New", title: "Add an empty recipe to write by hand" });
    const delBtn = el("button", { textContent: "🗑 Delete", title: "Delete this recipe from the node (the file changes when you Save)" });
    const recipePane = el("div", { className: "mykee-rc-pane mykee-rc" }, [
        el("div", { className: "mykee-rc-row" }, [prevBtn, counter, nextBtn, nameLabel]),
        recipeArea,
        recipeStatus,
        el("div", { className: "mykee-rc-row" }, [loadRecipeBtn, newBtn, delBtn]),
    ]);
    const recipeDom = node.addDOMWidget("recipe_editor", "MYKEE_RECIPE_EDITOR", recipePane, {
        serialize: false,
        getMinHeight: () => RECIPE_PANE_MIN_HEIGHT,
    });
    recipeDom.serialize = false;

    // --- Batch tab -------------------------------------------------------
    const batchOn = el("input", { type: "checkbox" });
    const sendOn = el("input", { type: "checkbox" });
    const allBtn = el("button", { textContent: "Select All", title: "Tick every recipe" });
    const noneBtn = el("button", { textContent: "Select None", title: "Untick every recipe" });
    const batchCount = el("span", { className: "mykee-rc-status" });
    const batchList = el("div", { className: "mykee-rc-list" });
    const batchBody = el("div", { className: "mykee-rc-batch" }, [
        el("label", { className: "mykee-rc-check", title: "Also write the parameter names and values under the recipe name in the plot labels (xyz_info)" }, [
            sendOn, el("span", { textContent: "Send parameters (names and values under the recipe name in the plot)" }),
        ]),
        el("div", { className: "mykee-rc-row" }, [allBtn, noneBtn, batchCount]),
        batchList,
    ]);
    const batchPane = el("div", { className: "mykee-rc-pane mykee-rc" }, [
        el("label", { className: "mykee-rc-check", title: "Off: the node works as before. On: Run queues one prompt per ticked recipe." }, [
            batchOn, el("span", { textContent: "Batch - run the ticked recipes one after another" }),
        ]),
        batchBody,
    ]);
    const batchDom = node.addDOMWidget("recipe_batch", "MYKEE_RECIPE_BATCH", batchPane, {
        serialize: false,
        getMinHeight: () => BATCH_PANE_MIN_HEIGHT,
    });
    batchDom.serialize = false;

    const showBatch = () => {
        const st = S(node);
        batchOn.checked = st.batch.enabled;
        sendOn.checked = st.batch.sendParams;
        batchBody.style.display = st.batch.enabled ? "" : "none";
        batchList.replaceChildren();
        const chosen = new Set(st.batch.selected);
        for (const recipe of st.recipes) {
            const box = el("input", { type: "checkbox", checked: chosen.has(recipe.name) });
            box.addEventListener("change", () => {
                const set = new Set(S(node).batch.selected);
                if (box.checked) set.add(recipe.name);
                else set.delete(recipe.name);
                S(node).batch.selected = [...set];
                updateBatchCount();
                persist(node);
            });
            batchList.append(el("label", { title: recipe.name }, [box, el("span", { textContent: recipe.name })]));
        }
        if (!st.recipes.length) batchList.append(el("div", { className: "mykee-rc-muted", textContent: "No recipe on the node - load a recipe file on the JSON file tab." }));
        updateBatchCount();
    };
    const updateBatchCount = () => {
        const st = S(node);
        const names = new Set(st.recipes.map((r) => r.name));
        const n = st.batch.selected.filter((x) => names.has(x)).length;
        batchCount.textContent = `${n} of ${st.recipes.length} ticked`;
    };
    batchOn.addEventListener("change", () => {
        S(node).batch.enabled = batchOn.checked;
        showBatch();
        persist(node);
    });
    sendOn.addEventListener("change", () => {
        S(node).batch.sendParams = sendOn.checked;
        persist(node);
    });
    allBtn.addEventListener("click", () => {
        S(node).batch.selected = S(node).recipes.map((r) => r.name);
        showBatch();
        persist(node);
    });
    noneBtn.addEventListener("click", () => {
        S(node).batch.selected = [];
        showBatch();
        persist(node);
    });

    // --- JSON file tab ---------------------------------------------------
    const jsonArea = isolate(el("textarea", { spellcheck: false, placeholder: "The recipe file's content (JSON) appears here." }));
    const jsonStatus = el("div", { className: "mykee-rc-status" });
    const jsonPane = el("div", { className: "mykee-rc-pane mykee-rc" }, [jsonArea, jsonStatus]);
    const jsonDom = node.addDOMWidget("recipe_json", "MYKEE_RECIPE_JSON", jsonPane, {
        serialize: false,
        getMinHeight: () => JSON_PANE_MIN_HEIGHT,
    });
    jsonDom.serialize = false;

    let recipeTimer = null;
    let jsonTimer = null;
    let recipeEdited = false;
    let jsonEdited = false;

    const updateRecipeHeader = () => {
        const st = S(node);
        const n = st.recipes.length;
        counter.textContent = n ? `${st.index + 1} / ${n}` : "0 / 0";
        nameLabel.textContent = n ? st.recipes[st.index].name : "";
        nameLabel.title = nameLabel.textContent;
        prevBtn.disabled = st.index <= 0;
        nextBtn.disabled = st.index >= n - 1;
        loadRecipeBtn.disabled = !n;
        delBtn.disabled = !n;
        recipeArea.disabled = !n;
    };

    const showRecipe = () => {
        const st = S(node);
        updateRecipeHeader();
        if (recipeEdited) return; // unparsable text the user is still working on stays as it is
        if (st.recipes.length) {
            recipeArea.value = recipeText(st.recipes[st.index]);
            const n = Object.keys(st.recipes[st.index].params).length;
            setStatus(recipeStatus, `✔ ${n} entr${n === 1 ? "y" : "ies"}`, true);
        } else {
            recipeArea.value = "";
            setStatus(recipeStatus, "", null);
        }
    };

    const showJson = () => {
        const st = S(node);
        if (jsonEdited) return; // unparsable text the user is still working on stays as it is
        const n = st.recipes.length;
        jsonArea.value = n ? docText(st) : "";
        setStatus(jsonStatus, `${n} recipe${n === 1 ? "" : "s"}${st.file ? ` · file: ${st.file}` : ""}${st.dirty ? " · unsaved changes" : ""}`, null);
    };

    node._mykeeRcRefreshAll = () => {
        showRecipe();
        showBatch();
        showJson();
        syncRecipePicker();
        persist(node);
    };

    // Recipe text -> state. Returns false only when the text is invalid.
    const commitRecipe = () => {
        clearTimeout(recipeTimer);
        if (!recipeEdited) return true;
        const st = S(node);
        if (!st.recipes.length) return true;
        let raw;
        try {
            raw = JSON.parse(recipeArea.value);
        } catch (e) {
            setStatus(recipeStatus, `✖ ${e.message}`, false);
            return false;
        }
        const warnings = [];
        const clean = cleanRecipe(raw, st.index + 1, warnings);
        if (!clean) {
            setStatus(recipeStatus, `✖ ${warnings[0] || "not a recipe"}`, false);
            return false;
        }
        st.recipes[st.index] = clean;
        st.dirty = true;
        recipeEdited = false;
        const n = Object.keys(clean.params).length;
        setStatus(recipeStatus, `✔ ${n} entr${n === 1 ? "y" : "ies"}${warnings.length ? ` · ${warnings.length} skipped: ${warnings[0]}` : ""}`, warnings.length ? null : true);
        updateRecipeHeader();
        setTextIfIdle(jsonArea, docText(st));
        persist(node);
        return true;
    };

    const commitJson = () => {
        clearTimeout(jsonTimer);
        if (!jsonEdited) return true;
        const st = S(node);
        let doc;
        try {
            doc = cleanDoc(JSON.parse(jsonArea.value));
        } catch (e) {
            setStatus(jsonStatus, `✖ ${e.message}`, false);
            return false;
        }
        st.recipes = doc.recipes;
        st.index = st.recipes.length ? clamp(st.index, 0, st.recipes.length - 1) : 0;
        st.dirty = true;
        jsonEdited = false;
        const n = st.recipes.length;
        setStatus(jsonStatus, `✔ ${n} recipe${n === 1 ? "" : "s"}${doc.warnings.length ? ` · ${doc.warnings.length} skipped: ${doc.warnings[0]}` : ""}`, doc.warnings.length ? null : true);
        updateRecipeHeader();
        setTextIfIdle(recipeArea, st.recipes.length ? recipeText(st.recipes[st.index]) : "");
        persist(node);
        return true;
    };
    node._mykeeRcUI = { commitRecipe, commitJson };

    recipeArea.addEventListener("input", () => {
        recipeEdited = true;
        clearTimeout(recipeTimer);
        try {
            const raw = JSON.parse(recipeArea.value);
            const w = [];
            const c = cleanRecipe(raw, S(node).index + 1, w);
            setStatus(recipeStatus, c ? `✔ valid${w.length ? ` · ${w.length} would be skipped` : ""}` : `✖ ${w[0] || "not a recipe"}`, c ? true : false);
            if (c) recipeTimer = setTimeout(commitRecipe, 700);
        } catch (e) {
            setStatus(recipeStatus, `✖ ${e.message}`, false);
        }
    });
    recipeArea.addEventListener("blur", () => {
        if (commitRecipe()) showRecipe();
    });
    jsonArea.addEventListener("input", () => {
        jsonEdited = true;
        clearTimeout(jsonTimer);
        try {
            const d = cleanDoc(JSON.parse(jsonArea.value));
            setStatus(jsonStatus, `✔ valid · ${d.recipes.length} recipe${d.recipes.length === 1 ? "" : "s"}${d.warnings.length ? ` · ${d.warnings.length} would be skipped` : ""}`, true);
            jsonTimer = setTimeout(commitJson, 700);
        } catch (e) {
            setStatus(jsonStatus, `✖ ${e.message}`, false);
        }
    });
    jsonArea.addEventListener("blur", () => {
        if (commitJson()) showJson();
    });

    const goTo = (index) => {
        if (!commitAll(node)) {
            toast("Fix the JSON text first (see the red status line).");
            return;
        }
        const st = S(node);
        st.index = clamp(index, 0, Math.max(0, st.recipes.length - 1));
        persist(node);
        showRecipe();
    };
    prevBtn.onclick = () => goTo(S(node).index - 1);
    nextBtn.onclick = () => goTo(S(node).index + 1);

    loadRecipeBtn.onclick = () => {
        if (!commitAll(node)) {
            toast("Fix the JSON text first (see the red status line).");
            return;
        }
        const st = S(node);
        if (st.recipes[st.index]) applyRecipeToNode(node, st.recipes[st.index]);
    };

    newBtn.onclick = () => {
        if (!commitAll(node)) {
            toast("Fix the JSON text first (see the red status line).");
            return;
        }
        const st = S(node);
        if (st.recipes.length >= MAX_RECIPES) {
            toast(`A recipe file can hold at most ${MAX_RECIPES} recipes.`);
            return;
        }
        let name = "New recipe";
        for (let i = 2; st.recipes.some((r) => r.name === name); i++) name = `New recipe ${i}`;
        st.recipes.push({ name, params: {} });
        st.index = st.recipes.length - 1;
        st.dirty = true;
        node._mykeeRcRefreshAll();
        recipeArea.focus();
    };

    delBtn.onclick = () => {
        const st = S(node);
        if (!st.recipes.length) return;
        if (!window.confirm(`Delete recipe '${st.recipes[st.index].name}' from the node?`)) return;
        st.recipes.splice(st.index, 1);
        st.index = clamp(st.index, 0, Math.max(0, st.recipes.length - 1));
        st.dirty = true;
        node._mykeeRcRefreshAll();
    };

    // --- JSON file tab: picker, Save / Load ------------------------------
    const existing = [];
    const listW = node.addWidget(
        "combo",
        "Existing recipe files",
        "",
        (value) => {
            if (!value) return;
            jsonW.value = value;
            loadFile(value);
        },
        { values: existing }
    );
    // Re-derived from disk on every reload, so a stale pick shouldn't persist.
    listW.serialize = false;

    // Points the (non-serialized) picker at json_name, if such a file exists.
    const syncPicker = () => {
        const name = String(jsonW.value || "").trim().replace(/\\/g, "/");
        listW.value = name && existing.includes(name) ? name : "";
        node.setDirtyCanvas(true, true);
    };

    let refreshSeq = 0;
    const refreshList = async (retriesLeft = 2) => {
        const seq = ++refreshSeq;
        try {
            const data = await fetchJSON(`/mykee/recipes/list?path=${encodeURIComponent(currentPath())}`);
            if (seq !== refreshSeq) return existing;
            // Update the SAME array in place, so any reference the frontend
            // may hold to options.values stays valid.
            existing.length = 0;
            existing.push(...(Array.isArray(data.files) ? data.files.map(String) : []));
            if (listW.options) listW.options.values = existing;
            syncPicker();
            node.setDirtyCanvas(true, true);
        } catch (e) {
            console.warn("Mykee Recipe: couldn't fetch the file list", e);
            if (seq === refreshSeq && retriesLeft > 0) setTimeout(() => refreshList(retriesLeft - 1), 1000);
        }
        return existing;
    };
    node._mykeeRcRefreshList = refreshList;

    // The picker follows the file name as it is typed / restored.
    const origJsonCallback = jsonW.callback;
    jsonW.callback = function () {
        const r = origJsonCallback ? origJsonCallback.apply(this, arguments) : undefined;
        syncPicker();
        return r;
    };

    // Re-read the list whenever custom_path changes.
    const origPathCallback = pathW.callback;
    pathW.callback = function () {
        const r = origPathCallback ? origPathCallback.apply(this, arguments) : undefined;
        refreshList();
        return r;
    };

    const loadFile = async (name) => {
        name = String(name || "").trim();
        if (!name) {
            window.alert("Enter a file name (json_name) or pick one from the list.");
            return;
        }
        if (!commitAll(node)) {
            toast("Fix the JSON text first (see the red status line).");
            return;
        }
        const st = S(node);
        if (st.dirty && st.recipes.length && !window.confirm("The recipes on the node have unsaved changes - replace them with the file's content?")) return;
        try {
            const data = await fetchJSON(
                `/mykee/recipes/load?path=${encodeURIComponent(currentPath())}&name=${encodeURIComponent(name)}`
            );
            const doc = cleanDoc(data.doc); // never trust the response blindly either
            st.recipes = doc.recipes;
            st.index = 0;
            st.file = cleanLabel(data.name, 200) || name;
            st.dirty = false;
            jsonW.value = st.file;
            syncPicker();
            node._mykeeRcRefreshAll();
            const extra = [...(Array.isArray(data.warnings) ? data.warnings.map(String) : []), ...doc.warnings];
            toast(
                `'${st.file}.json' loaded: ${st.recipes.length} recipe${st.recipes.length === 1 ? "" : "s"}.` +
                    (extra.length ? ` ${extra.length} problem${extra.length === 1 ? "" : "s"}: ${extra.slice(0, 3).join(" | ")}` : ""),
                extra.length ? "warn" : "success",
                extra.length ? 9000 : 4000
            );
            node.setDirtyCanvas(true, true);
        } catch (e) {
            window.alert("Load failed: " + e.message);
        }
    };

    const saveFile = async () => {
        const name = String(jsonW.value || "").trim();
        if (!name) {
            window.alert("Enter a file name (json_name) to save.");
            return;
        }
        if (!commitAll(node)) {
            toast("Fix the JSON text first (see the red status line).");
            return;
        }
        const st = S(node);
        if (!st.recipes.length) {
            window.alert("Nothing to save - there is no recipe on the node yet.");
            return;
        }
        await refreshList();
        const sameAsLoaded = name.toLowerCase() === st.file.toLowerCase();
        if (!sameAsLoaded && existing.some((n) => n.toLowerCase() === name.toLowerCase())
            && !window.confirm(`'${name}.json' already exists - overwrite it?`)) return;
        try {
            const data = await fetchJSON("/mykee/recipes/save", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ path: currentPath(), name, doc: { version: 1, recipes: st.recipes } }),
            });
            st.file = cleanLabel(data.name, 200) || name;
            st.dirty = false;
            jsonW.value = st.file;
            await refreshList(); // re-reads the folder and points the picker at the saved file
            syncPicker();
            node._mykeeRcRefreshAll();
            toast(`'${st.file}.json' saved (${st.recipes.length} recipe${st.recipes.length === 1 ? "" : "s"}).`, "success", 4000);
            node.setDirtyCanvas(true, true);
        } catch (e) {
            window.alert("Save failed: " + e.message);
        }
    };

    const saveBtn = node.addWidget("button", "💾 Save to Recipe JSON", null, saveFile);
    const loadBtn = node.addWidget("button", "📂 Load Recipe JSON", null, () => loadFile(jsonW.value));
    const reloadBtn = node.addWidget("button", "🔄 Reload list", null, async () => {
        await refreshList();
        node._mykeeRcRenderParams?.(); // also re-reads the connected lists
    });
    for (const b of [saveBtn, loadBtn, reloadBtn]) {
        b.serialize = false;
        b.options = { ...(b.options || {}), serialize: false };
    }

    // --- layout: tab bar first, then each tab's widgets in order ---------
    const tabBar = createTabBar(node);
    node._mykeeRcTabWidgets = {
        params: [slotsW, paramsDom, pickerW, nameW, sendBtn],
        recipe: [recipeDom],
        batch: [batchDom],
        json: [jsonW, pathW, listW, saveBtn, loadBtn, reloadBtn, jsonDom],
    };
    const ordered = [tabBar, slotsW, paramsDom, pickerW, nameW, sendBtn, recipeDom, batchDom, jsonW, pathW, listW, saveBtn, loadBtn, reloadBtn, jsonDom, configW];
    const rest = node.widgets.filter((w) => !ordered.includes(w));
    node.widgets = [...ordered, ...rest];

    node.setSize([Math.max(node.size[0], MIN_WIDTH), node.size[1]]);
    applyTab(node, true); // a fresh node: sized to its content once (a loaded one is configured right after)
    node._mykeeRcRenderParams();
    node._mykeeRcRefreshAll();
    refreshList();
    // ComfyUI may still apply its own size after onNodeCreated - fit again. On
    // the text-box tabs a loaded workflow keeps its saved height (that is the
    // text box's size); the Parameters tab always follows its content.
    setTimeout(() => {
        if (!node._mykeeConfiguredAt || currentTab(node) === "params") fitHeight(node, true);
    }, 0);
}

app.registerExtension({
    name: "Mykee.Recipe",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function () {
            const ret = onDrawForeground?.apply(this, arguments);
            updateButtonCursor(this);
            return ret;
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated?.apply(this, arguments);
            setupNode(this);
            return ret;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const ret = onConfigure?.apply(this, arguments);
            if (!this._mykeeRcTabWidgets) return ret; // setupNode did not run
            this._mykeeConfiguredAt = Date.now();
            migrateOutputs(this); // a workflow saved before the xyz_info output existed
            // The saved widget values have just been restored: re-read the state
            // (a workflow may come from anywhere, so it is cleaned again) and make
            // the outputs match the saved slot count.
            this._mykeeRc = parseState(findWidget(this, CONFIG_WIDGET)?.value);
            setOutputCount(this, slotCount(this));
            applyTab(this, false); // the saved size already belongs to the saved tab
            this._mykeeRcRefreshList?.();
            // Target nodes may not exist yet while the graph is loading: settle
            // again once the links exist. The Parameters tab fits its content; the
            // text-box tabs keep the saved height (and only grow if it is too small).
            const settle = () => {
                for (let s = 0; s < slotCount(this); s++) updateOutputSlot(this, s);
                this._mykeeRcRenderParams?.();
                this._mykeeRcRefreshAll?.();
                fitHeight(this, currentTab(this) === "params");
            };
            setTimeout(settle, 50);
            setTimeout(settle, 400);
            return ret;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, slot, connected, linkInfo) {
            const ret = onConnectionsChange?.apply(this, arguments);
            const OUTPUT = window.LiteGraph?.OUTPUT ?? 2;
            if (type !== OUTPUT || slot < FIRST_PARAM_OUTPUT) return ret;
            const s = slot - FIRST_PARAM_OUTPUT;
            if (connected && linkInfo && !checkNewLink(this, s, linkInfo)) return ret;

            // A link made by hand (not while a workflow is loading): the widget's
            // name becomes the key and its current value the parameter's value, so
            // nothing has to be looked up and typed in again.
            let adopt = null;
            if (connected && linkInfo && !app.configuringGraph && Date.now() - (this._mykeeConfiguredAt || 0) > 1500) {
                const already = slotTargets(this, s).some((t) => t.supported && t.linkId !== linkInfo.id);
                const info = already ? null : describeInput(this.graph?.getNodeById?.(linkInfo.target_id), linkInfo.target_slot);
                if (info?.supported) adopt = info;
            }
            if (!connected) {
                // Reset right away, ignoring the link being removed (it may still
                // be listed while the disconnect is in progress).
                updateOutputSlot(this, s, linkInfo?.id);
            }
            // Defer: during a connect / disconnect the link list is not final yet.
            setTimeout(() => {
                if (adopt) {
                    const p = param(this, s);
                    p.key = uniqueKey(this, adopt.widgetName, s);
                    const res = coerceValue(adopt, adopt.type, adopt.currentValue);
                    p.value = res.ok ? res.value : defaultValue(adopt);
                }
                updateOutputSlot(this, s);
                this._mykeeRcRenderParams?.();
                this.graph?.setDirtyCanvas?.(true, true);
            }, 0);
            return ret;
        };
    },
});

// Exposed for the unit tests only (nothing in ComfyUI reads this).
export const __testing = { planBatch, valueText, batchFromPrompt, buildBatchOutput, migrateOutputs, cleanDoc, cleanRecipe, cleanParam, cleanValue, matchOption, coerceValue, planApply, defaultValue, parseState, stepFor };
