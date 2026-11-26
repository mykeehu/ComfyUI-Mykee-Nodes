/*
 * Mykee Universal Generator - frontend.
 *
 * The node shows the RECIPE only (compact, collapsible groups); the library
 * is browsed and edited in a modal window that exists only while it is open.
 *
 * Node panel
 *   - search box: type to find any value of the library, click to add it
 *   - Browse: opens the library window
 *   - recipe header: Load / Save / Reset all
 *   - one collapsible group per category (reset per group), lines can be
 *     moved up/down and removed; a fixed line can be edited by hand (it is
 *     then marked "edited"), random lines show their last picks
 *   - tabs: Scene (always there) and one tab per subject (person, creature,
 *     ...) with a free-text name that the prompt uses; subjects can be
 *     renamed, duplicated, moved and deleted
 *   - Manual group: free text lines (per tab)
 *   - the last run's tags and prompt
 *
 * Library window
 *   - left: category tree (max 2 levels) with + Category / + Property,
 *     rename / delete (delete = move to the _trash folder)
 *   - middle: the selected property's values (browse mode: tick / mode /
 *     count) or its editor (edit mode)
 *   - right: what will change in the recipe when you press Send
 *   Send writes only the properties you changed, so manual lines, edited
 *   lines that you did not touch and lines whose library file is gone stay.
 *
 * Safety: every value from the server, a file or a workflow is cleaned
 * (cleanRecipe) and only ever put into the DOM as text, never as HTML.
 *
 * Height: the node's height always follows its content (it grows and
 * shrinks); a height the user adds by dragging (room for the system_prompt
 * text box) is remembered and kept. The width is the user's.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "MykeeUniversalGenerator";
const STATE_WIDGET = "recipe_state";
const TITLE = "Mykee Universal Generator";
const EXTRA_PROP = "mugen_extra_height";
const LAST_PROP = "mugen_last_output";
const MIN_WIDTH = 420;
// Free space under the panel, inside the node. The last box must never
// touch or poke out of the node's bottom edge.
const BOTTOM_GAP = 12;

// Limits - the same as in nodes/mykee_universal_generator.py.
const MAX_NAME = 80;
const MAX_LABEL = 120;
const MAX_ITEM_TEXT = 1000;
const MAX_ITEMS = 2000;
const MAX_ENTRIES = 500;
const MAX_MANUAL_TEXT = 2000;
const MAX_EDIT_TEXT = 2000;
const MAX_DEPTH = 2;
const ID_RE = /^[A-Za-z0-9_-]{1,32}$/;
const MODES = ["fixed", "random_checked", "random_all"];
const MODE_LABEL = { fixed: "fixed", random_checked: "random from checked", random_all: "random from all" };
const MANUAL_GROUP = "\u0000manual";
const SCENE = "scene";
const MAX_SUBJECTS = 16;
const MAX_SUBJECT_NAME = 120;
const SCOPES = ["", "scene", "subject"];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function toast(detail, severity = "warn", life = 5000) {
    if (app.extensionManager?.toast?.add) {
        app.extensionManager.toast.add({ severity, summary: TITLE, detail, life });
    } else {
        console.warn(`[${TITLE}] ${detail}`);
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
    if (!res.ok) {
        const err = new Error(data.error || res.statusText || "request failed");
        err.status = res.status;
        throw err;
    }
    return data;
}

const postJSON = (url, body) => fetchJSON(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
});

function el(tag, props = {}, children = []) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
        if (v === undefined || v === null) continue;
        if (k === "className") e.className = v;
        else if (k === "text") e.textContent = v;
        else if (k === "title") e.title = v;
        else if (k === "style") e.style.cssText = v;
        else if (k.startsWith("on")) e.addEventListener(k.slice(2).toLowerCase(), v);
        else if (k in e) e[k] = v;
        else e.setAttribute(k, v);
    }
    for (const c of [].concat(children)) {
        if (c === null || c === undefined || c === false) continue;
        e.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return e;
}

const btn = (text, onClick, cls = "", title = "") =>
    el("button", { type: "button", className: `mugen-btn ${cls}`, text, title, onclick: (e) => { e.stopPropagation(); onClick(e); } });

function findWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function newId(taken = new Set()) {
    for (;;) {
        const id = Math.random().toString(36).slice(2, 10);
        if (id.length >= 6 && !taken.has(id)) return id;
    }
}

const str = (v, limit) => (typeof v === "string" ? v.replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/g, "").slice(0, limit) : "");
const int = (v, lo, hi, d) => (typeof v === "number" && Number.isFinite(v) ? Math.min(hi, Math.max(lo, Math.trunc(v))) : d);
const propKey = (category, file) => `${category}\u0001${file}`;
const catTitle = (path) => path.split("/").join(" › ");

function validName(name) {
    if (typeof name !== "string" || name !== name.trim() || name.length < 1 || name.length > MAX_NAME) return false;
    if (/[<>:"/\\|?*\u0000-\u001f\u007f]/.test(name)) return false;
    if (name.startsWith(".") || name.endsWith(".") || name.endsWith(" ")) return false;
    const stem = name.split(".")[0].toUpperCase();
    if (/^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$/.test(stem)) return false;
    return name.toLowerCase() !== "_trash";
}

function askName(message, value = "") {
    const v = window.prompt(message, value);
    if (v === null) return null;
    const name = v.trim();
    if (!validName(name)) {
        toast("Invalid name: 1-80 characters, no / \\ : * ? \" < > |, no leading dot and no trailing dot.", "error");
        return null;
    }
    return name;
}

// ---------------------------------------------------------------------------
// Recipe state (mirror of clean_recipe() in the Python file)
// ---------------------------------------------------------------------------

function cleanItem(it) {
    if (!it || typeof it !== "object") return null;
    const prompt = str(it.prompt, MAX_ITEM_TEXT).trim();
    if (!prompt) return null;
    return {
        id: typeof it.id === "string" && ID_RE.test(it.id) ? it.id : "",
        prompt,
        label: str(it.label, MAX_LABEL).trim() || prompt.slice(0, MAX_LABEL),
    };
}

function cleanRecipe(data) {
    const src = data && typeof data === "object" && !Array.isArray(data) ? data : {};
    const subjects = [];
    const sids = new Set();
    for (const sub of Array.isArray(src.subjects) ? src.subjects.slice(0, MAX_SUBJECTS) : []) {
        if (!sub || typeof sub !== "object" || typeof sub.sid !== "string" || !ID_RE.test(sub.sid) || sids.has(sub.sid) || sub.sid === SCENE) continue;
        sids.add(sub.sid);
        const name = str(sub.name, MAX_SUBJECT_NAME).replace(/\s+/g, " ").trim();
        subjects.push({ sid: sub.sid, name: name || `person ${subjects.length + 1}` });
    }
    const uids = new Set();
    const entries = [];
    for (const e of Array.isArray(src.entries) ? src.entries.slice(0, MAX_ENTRIES) : []) {
        if (!e || typeof e !== "object") continue;
        let uid = typeof e.uid === "string" && ID_RE.test(e.uid) && !uids.has(e.uid) ? e.uid : newId(uids);
        uids.add(uid);
        // no (or an unknown) subject = the scene; old recipes load as scene-only
        const subject = sids.has(e.subject) ? e.subject : SCENE;
        if (e.kind === "manual") {
            entries.push({ uid, subject, kind: "manual", text: str(e.text, MAX_MANUAL_TEXT) });
            continue;
        }
        if (e.kind !== "lib") continue;
        const parts = typeof e.category === "string" ? e.category.split("/") : [];
        if (!parts.length || parts.length > MAX_DEPTH || !parts.every(validName) || !validName(e.property)) continue;
        const items = (Array.isArray(e.items) ? e.items.slice(0, MAX_ITEMS) : []).map(cleanItem).filter(Boolean);
        const mode = MODES.includes(e.mode) ? e.mode : "fixed";
        const lo = int(e.min, 1, MAX_ITEMS, 1);
        const hi = int(e.max, lo, MAX_ITEMS, lo);
        const edited = !!e.edited && mode === "fixed";
        entries.push({
            uid, subject, kind: "lib", category: e.category, property: e.property,
            propLabel: str(e.propLabel, MAX_LABEL) || e.property,
            order: int(e.order, -9999, 9999, 100),
            ptype: e.ptype === "single" ? "single" : "multi",
            mode, min: lo, max: hi, items, edited,
            text: edited ? str(e.text, MAX_EDIT_TEXT) : "",
        });
    }
    const collapsed = Array.isArray(src.ui?.collapsed)
        ? src.ui.collapsed.filter((c) => typeof c === "string" && c.length < 200).slice(0, 200) : [];
    const active = sids.has(src.ui?.active) ? src.ui.active : SCENE;
    return { version: 2, name: str(src.name, MAX_NAME).trim(), subjects, entries, ui: { collapsed, active } };
}

const groupName = (e) => (e.kind === "manual" ? MANUAL_GROUP : e.category);
const groupKey = (e) => `${e.subject}\u0002${groupName(e)}`;

// Keeps every group contiguous: the scene first, then the subjects in their
// order; inside a tab the groups keep their order of first appearance and
// the Manual group comes last.
function normalize(recipe) {
    const tabs = [SCENE, ...recipe.subjects.map((x) => x.sid)];
    const out = [];
    for (const tab of tabs) {
        const order = [];
        const byGroup = new Map();
        for (const e of recipe.entries) {
            if (e.subject !== tab) continue;
            const g = groupName(e);
            if (!byGroup.has(g)) {
                byGroup.set(g, []);
                if (g !== MANUAL_GROUP) order.push(g);
            }
            byGroup.get(g).push(e);
        }
        if (byGroup.has(MANUAL_GROUP)) order.push(MANUAL_GROUP);
        for (const g of order) out.push(...byGroup.get(g));
    }
    recipe.entries = out;
}

function insertLibEntry(recipe, entry) {
    normalize(recipe);
    const list = recipe.entries;
    const tab = (e) => e.subject === entry.subject;
    const same = list.map((e, i) => [e, i]).filter(([e]) => tab(e) && e.kind === "lib" && e.category === entry.category);
    let at;
    if (same.length) {
        const after = same.find(([e]) => e.order > entry.order);
        at = after ? after[1] : same[same.length - 1][1] + 1;
    } else {
        // a new group goes before the first group (of the same tab) whose lines all come later
        const firsts = new Map();
        list.forEach((e, i) => {
            if (!tab(e) || e.kind !== "lib") return;
            const f = firsts.get(e.category);
            if (!f) firsts.set(e.category, { i, min: e.order });
            else f.min = Math.min(f.min, e.order);
        });
        const later = [...firsts.values()].filter((f) => f.min > entry.order).sort((a, b) => a.i - b.i)[0];
        const manualAt = list.findIndex((e) => tab(e) && e.kind === "manual");
        let tabEnd = -1;
        list.forEach((e, i) => { if (tab(e)) tabEnd = i; });
        at = later ? later.i : manualAt >= 0 ? manualAt : tabEnd >= 0 ? tabEnd + 1 : list.length;
    }
    list.splice(at, 0, entry);
    normalize(recipe);
}

function findEntry(recipe, tab, category, property) {
    return recipe.entries.find((e) => e.kind === "lib" && e.subject === tab && e.category === category && e.property === property);
}

// Which tab a property's line belongs to when `active` is the chosen tab:
// scope "scene" -> always the scene, "subject" -> the active subject (null
// when the scene is active: it needs a person tab), "" -> the active tab.
function targetTab(prop, active) {
    if (prop?.scope === "scene") return SCENE;
    if (prop?.scope === "subject") return active === SCENE ? null : active;
    return active;
}

function tabName(recipe, tab) {
    if (tab === SCENE) return "Scene";
    return recipe.subjects.find((x) => x.sid === tab)?.name || "?";
}

function entryValuesText(e, picks) {
    if (e.mode === "fixed") return e.edited ? e.text : e.items.map((i) => i.prompt).join(", ");
    const p = picks?.[e.uid];
    return Array.isArray(p) && p.length ? p.join(", ") : "";
}

// ---------------------------------------------------------------------------
// Library cache (shared by every node)
// ---------------------------------------------------------------------------

const LIB = { data: null, loading: null };

async function loadLibrary(force = false) {
    if (LIB.data && !force) return LIB.data;
    if (LIB.loading && !force) return LIB.loading;
    LIB.loading = (async () => {
        const raw = await fetchJSON("/mykee/ugen/library");
        const properties = new Map();
        for (const p of Array.isArray(raw.properties) ? raw.properties : []) {
            if (!p || typeof p.category !== "string" || typeof p.file !== "string") continue;
            const items = (Array.isArray(p.items) ? p.items : []).map(cleanItem).filter((i) => i && i.id);
            properties.set(propKey(p.category, p.file), {
                category: p.category, file: p.file,
                label: str(p.label, MAX_LABEL) || p.file,
                type: p.type === "multi" ? "multi" : "single",
                max: int(p.max, 0, MAX_ITEMS, 0),
                order: int(p.order, -9999, 9999, 100),
                scope: SCOPES.includes(p.scope) ? p.scope : "",
                items,
            });
        }
        const categories = (Array.isArray(raw.categories) ? raw.categories : [])
            .filter((c) => c && typeof c.path === "string" && c.path.split("/").every(validName))
            .map((c) => ({ path: c.path, name: c.path.split("/").pop(), depth: c.path.split("/").length }));
        if (Array.isArray(raw.errors) && raw.errors.length) {
            toast(`Some library files could not be read:\n${raw.errors.slice(0, 5).map((e) => str(e, 300)).join("\n")}`, "warn", 9000);
        }
        LIB.data = { categories, properties };
        LIB.loading = null;
        return LIB.data;
    })();
    try {
        return await LIB.loading;
    } catch (e) {
        LIB.loading = null;
        throw e;
    }
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

function injectStyle() {
    if (document.getElementById("mugen-style")) return;
    const css = `
.mugen{box-sizing:border-box;width:100%;overflow:hidden;font:12px/1.35 Inter,system-ui,sans-serif;color:#ddd;pointer-events:auto}
.mugen *{box-sizing:border-box}
.mugen-inner{display:flex;flex-direction:column;gap:6px;padding:2px 2px 0}
.mugen-btn{cursor:pointer;background:#383838;color:#e0e0e0;border:1px solid #4a4a4a;border-radius:5px;padding:3px 8px;font:inherit;white-space:nowrap}
.mugen-btn:hover{background:#444}
.mugen-btn.primary{background:#3a5f99;border-color:#3a5f99;color:#fff;font-weight:600}
.mugen-btn.primary:hover{background:#4670b0}
.mugen-btn.danger{background:#4a2a2a;border-color:#6a3a3a;color:#f2b8b0}
.mugen-btn.ok{background:#3d7a4a;border-color:#3d7a4a;color:#fff;font-weight:600}
.mugen-btn.icon{background:none;border:none;color:#9a9a9a;padding:1px 4px}
.mugen-btn.icon:hover{color:#fff;background:#3a3a3a}
.mugen-btn.link{background:none;border:none;color:#7fb0ff;padding:2px 0}
.mugen-btn:disabled{opacity:.4;cursor:default}
.mugen-row{display:flex;gap:6px;align-items:center}
.mugen-input{flex:1;min-width:0;background:#1e1e1e;border:1px solid #444;border-radius:5px;color:#e0e0e0;padding:4px 7px;font:inherit}
.mugen-input:focus{outline:none;border-color:#5b8fd9}
.mugen-results{background:#1c1c1c;border:1px solid #3c3c3c;border-radius:5px;max-height:none}
.mugen-result{cursor:pointer;padding:4px 8px;display:flex;gap:4px}
.mugen-result:hover{background:#2a3a55}
.mugen-dim{color:#9a9a9a}
.mugen-head{display:flex;justify-content:space-between;align-items:center;gap:6px}
.mugen-title{font-weight:600;letter-spacing:.04em;color:#bdbdbd;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mugen-group{background:#242424;border:1px solid #3a3a3a;border-radius:6px}
.mugen-group.manual{border-style:dashed;border-color:#3f5a55}
.mugen-ghead{display:flex;align-items:center;gap:6px;padding:4px 6px;cursor:pointer;user-select:none}
.mugen-gname{flex:1;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mugen-gbody{display:flex;flex-direction:column;gap:3px;padding:3px 6px 6px;border-top:1px solid #3a3a3a}
.mugen-line{display:flex;gap:6px;align-items:center;min-height:22px}
.mugen-plabel{width:92px;flex:none;color:#9a9a9a;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mugen-val{flex:1;min-width:0;display:flex;gap:5px;align-items:center;flex-wrap:wrap;cursor:text;word-break:break-word}
.mugen-badge{border-radius:4px;padding:0 5px;font-size:10.5px;white-space:nowrap}
.mugen-badge.rnd{background:#3b3155;color:#cdb8ff}
.mugen-badge.edit{background:#4a3b1f;color:#f0c674}
.mugen-badge.miss{background:#4a2626;color:#f2a69c}
.mugen-badge.count{background:#333;color:#bbb}
.mugen-out{background:#1e1e1e;border:1px solid #3a3a3a;border-radius:6px;padding:5px 8px;white-space:pre-wrap;word-break:break-word;user-select:text;cursor:text}
.mugen-outlabel{font-size:10.5px;color:#9a9a9a;margin-bottom:2px;display:flex;justify-content:space-between}
.mugen-empty{color:#8a8a8a;padding:6px 2px}
.mugen-tabs{display:flex;flex-wrap:wrap;gap:3px;border-bottom:1px solid #3a3a3a;padding-bottom:4px}
.mugen-tab{cursor:pointer;background:#2c2c2c;color:#bdbdbd;border:1px solid #3a3a3a;border-radius:5px 5px 0 0;padding:3px 9px;font:inherit;max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mugen-tab:hover{background:#353535}
.mugen-tab.on{background:#3a5f99;border-color:#3a5f99;color:#fff;font-weight:600}
.mugen-tab.add{background:none;border-style:dashed;color:#7fb0ff}
.mugen-overlay{position:fixed;inset:0;z-index:10000;background:rgba(0,0,0,.6);display:flex;align-items:center;justify-content:center;font:13px/1.35 Inter,system-ui,sans-serif;color:#e0e0e0}
.mugen-dialog{width:min(1200px,94vw);height:min(800px,90vh);background:#262626;border:1px solid #3c3c3c;border-radius:12px;display:flex;flex-direction:column;overflow:hidden;box-shadow:0 10px 40px rgba(0,0,0,.6)}
.mugen-dialog *{box-sizing:border-box}
.mugen-dhead,.mugen-dfoot{display:flex;align-items:center;gap:10px;padding:10px 14px;background:#1f1f1f}
.mugen-dhead{border-bottom:1px solid #3a3a3a}
.mugen-dfoot{border-top:1px solid #3a3a3a;justify-content:space-between}
.mugen-dbody{flex:1;display:flex;min-height:0}
.mugen-tree{width:260px;flex:none;border-right:1px solid #3a3a3a;display:flex;flex-direction:column;min-height:0}
.mugen-treelist{flex:1;overflow:auto;padding:4px 6px}
.mugen-tnode{display:flex;align-items:center;gap:4px;padding:4px 6px;border-radius:4px;cursor:pointer;user-select:none}
.mugen-tnode:hover{background:#2e2e2e}
.mugen-tnode.sel{background:#2f4466}
.mugen-tnode .mugen-tname{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mugen-tnode .mugen-tacts{display:none}
.mugen-tnode:hover .mugen-tacts{display:flex}
.mugen-main{flex:1;display:flex;flex-direction:column;min-width:0;min-height:0}
.mugen-mhead{padding:12px 16px;border-bottom:1px solid #3a3a3a;display:flex;flex-direction:column;gap:8px}
.mugen-mbody{flex:1;overflow:auto;padding:10px 16px}
.mugen-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:4px 14px}
.mugen-item{display:flex;gap:8px;align-items:center;padding:5px 8px;border-radius:5px;cursor:pointer;user-select:none}
.mugen-item:hover{background:#2e2e2e}
.mugen-item.on{background:#2c3a50}
.mugen-item.off{opacity:.5;cursor:default}
.mugen-check{width:14px;height:14px;flex:none;border:1px solid #777;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;color:#fff}
.mugen-item.on .mugen-check{background:#5b8fd9;border-color:#5b8fd9}
.mugen-seg{display:flex;background:#2e2e2e;border-radius:6px;padding:2px}
.mugen-seg button{cursor:pointer;background:none;border:0;color:#bdbdbd;border-radius:5px;padding:4px 10px;font:inherit}
.mugen-seg button.on{background:#3a5f99;color:#fff}
.mugen-side{width:280px;flex:none;border-left:1px solid #3a3a3a;background:#222;display:flex;flex-direction:column;min-height:0}
.mugen-sidelist{flex:1;overflow:auto;padding:0 14px 10px;display:flex;flex-direction:column;gap:8px}
.mugen-add{color:#8fd18f}.mugen-rem{color:#f2a69c}.mugen-chg{color:#cdb8ff}
.mugen-field{display:flex;flex-direction:column;gap:3px;min-width:0}
.mugen-field>.mugen-input,.mugen-field>.mugen-select{width:100%;min-width:0;flex:none}
.mugen-field>span{font-size:11px;color:#9a9a9a}
.mugen-select{background:#1e1e1e;border:1px solid #444;border-radius:5px;color:#e0e0e0;padding:4px 6px;font:inherit}
.mugen-num{width:70px;flex:none}
.mugen-erow{display:flex;gap:8px;align-items:center;margin-bottom:4px}
.mugen-textarea{width:100%;min-height:120px;background:#1e1e1e;border:1px solid #444;border-radius:5px;color:#e0e0e0;padding:6px;font:inherit;resize:vertical}
`;
    document.head.append(el("style", { id: "mugen-style", text: css }));
}

// ---------------------------------------------------------------------------
// Node state access
// ---------------------------------------------------------------------------

function S(node) {
    if (!node._mugen) node._mugen = cleanRecipe({});
    return node._mugen;
}

function persist(node) {
    const w = findWidget(node, STATE_WIDGET);
    if (w) w.value = JSON.stringify(S(node));
    node.setDirtyCanvas?.(true, true);
}

function lastOutput(node) {
    const v = node.properties?.[LAST_PROP];
    return v && typeof v === "object" ? v : {};
}

function fitHeight(node) {
    const size = node.computeSize?.();
    if (!size || !Number.isFinite(size[1])) return;
    const extra = Math.max(0, Number(node.properties?.[EXTRA_PROP]) || 0);
    node._mugenSizing = true;
    try {
        node.setSize([Math.max(node.size[0], size[0], MIN_WIDTH), size[1] + extra]);
    } finally {
        node._mugenSizing = false;
    }
    node.graph?.setDirtyCanvas?.(true, true);
}

// ---------------------------------------------------------------------------
// Recipe edits from the node
// ---------------------------------------------------------------------------

function addItemFromSearch(node, prop, item) {
    const st = S(node);
    const tab = targetTab(prop, st.ui.active);
    if (!tab) {
        toast(`'${prop.label}' belongs to a person - add or select a person tab first.`, "info");
        return;
    }
    const e = findEntry(st, tab, prop.category, prop.file);
    const snap = { id: item.id, prompt: item.prompt, label: item.label };
    if (!e) {
        insertLibEntry(st, {
            uid: newId(new Set(st.entries.map((x) => x.uid))), subject: tab, kind: "lib",
            category: prop.category, property: prop.file, propLabel: prop.label, order: prop.order,
            ptype: prop.type, mode: "fixed", min: 1, max: 1, items: [snap], edited: false, text: "",
        });
        if (tab !== st.ui.active) toast(`'${prop.label}' is a scene property - added to the Scene tab.`, "info", 2500);
    } else if (e.mode === "random_all") {
        toast(`'${prop.label}' is set to random from all - every value is already in play.`, "info");
        return;
    } else if (e.items.some((i) => i.id === item.id)) {
        toast(`'${item.prompt}' is already in the recipe.`, "info");
        return;
    } else if (prop.type === "single" && e.mode === "fixed") {
        e.items = [snap];
        e.edited = false;
        e.text = "";
    } else {
        if (e.mode === "fixed" && prop.max > 0 && e.items.length >= prop.max) {
            toast(`'${prop.label}' allows at most ${prop.max} value(s).`, "warn");
            return;
        }
        e.items.push(snap);
        e.edited = false;
        e.text = "";
    }
    st.ui.collapsed = st.ui.collapsed.filter((c) => c !== `${tab}\u0002${prop.category}`);
    persist(node);
    node._mugenRender?.();
}

function moveEntry(node, uid, dir) {
    const st = S(node);
    normalize(st);
    const i = st.entries.findIndex((e) => e.uid === uid);
    const j = i + dir;
    if (i < 0 || j < 0 || j >= st.entries.length) return;
    if (groupKey(st.entries[i]) !== groupKey(st.entries[j])) return;
    [st.entries[i], st.entries[j]] = [st.entries[j], st.entries[i]];
    persist(node);
    node._mugenRender?.();
}

function removeEntry(node, uid) {
    const st = S(node);
    st.entries = st.entries.filter((e) => e.uid !== uid);
    persist(node);
    node._mugenRender?.();
}

// ---------------------------------------------------------------------------
// The node panel
// ---------------------------------------------------------------------------

function buildPanel(node) {
    const root = el("div", { className: "mugen" });
    const inner = el("div", { className: "mugen-inner" });
    root.append(inner);
    // keep canvas shortcuts / dragging out of the panel's inputs
    for (const ev of ["keydown", "keyup", "keypress"]) root.addEventListener(ev, (e) => e.stopPropagation());

    let panelHeight = 120;
    const dom = node.addDOMWidget("mugen_panel", "MYKEE_UGEN_PANEL", root, {
        serialize: false,
        getMinHeight: () => panelHeight,
        getMaxHeight: () => panelHeight,
    });
    dom.serialize = false;

    // The widget's height = content + BOTTOM_GAP + slack. "slack" is learned:
    // the frontend may give the element a little less than the widget height
    // (its own margins), so after layout the real room is checked and any
    // shortfall is added once - the last box is never cut off.
    let slack = 0;
    const measure = () => {
        const content = Math.ceil(inner.offsetHeight); // layout height, not affected by the canvas zoom
        if (content <= 0) return;
        const room = root.clientHeight;
        if (room > 0 && Math.abs(room - (panelHeight - slack)) <= 40 + slack) {
            const shortfall = content + BOTTOM_GAP - room;
            if (shortfall > 1 && Math.abs(panelHeight - (content + BOTTOM_GAP + slack)) <= 1) slack += shortfall;
        }
        const h = content + BOTTOM_GAP + slack;
        if (Math.abs(h - panelHeight) > 1) {
            panelHeight = h;
            fitHeight(node);
            queueMeasure(); // check the real room once more after the resize
        }
    };
    let queued = false;
    const queueMeasure = () => {
        if (queued) return;
        queued = true;
        requestAnimationFrame(() => {
            queued = false;
            measure();
        });
    };
    if (typeof ResizeObserver !== "undefined") {
        const ro = new ResizeObserver(queueMeasure);
        ro.observe(inner);
        ro.observe(root);
    }
    node._mugenMeasure = queueMeasure;

    // --- search ---------------------------------------------------------
    const search = el("input", { className: "mugen-input", type: "text", placeholder: "Search the library…", spellcheck: false });
    const results = el("div", { className: "mugen-results", style: "display:none" });
    const browseBtn = btn("+ Browse", () => openLibrary(node), "primary", "Open the library window");
    let searchTimer = 0;
    const runSearch = async () => {
        const q = search.value.trim().toLowerCase();
        results.replaceChildren();
        if (!q) {
            results.style.display = "none";
            return;
        }
        let lib;
        try {
            lib = await loadLibrary();
        } catch (e) {
            toast(`Could not load the library: ${e.message}`, "error");
            return;
        }
        const words = q.split(/\s+/).filter(Boolean);
        const hits = [];
        outer: for (const p of lib.properties.values()) {
            for (const it of p.items) {
                const hay = `${p.category} ${p.label} ${it.prompt} ${it.label}`.toLowerCase();
                if (words.every((w) => hay.includes(w))) {
                    hits.push([p, it]);
                    if (hits.length >= 14) break outer;
                }
            }
        }
        if (!hits.length) results.append(el("div", { className: "mugen-empty", text: "No match." }));
        for (const [p, it] of hits) {
            results.append(el("div", {
                className: "mugen-result",
                title: "Add to the recipe",
                onmousedown: (e) => e.preventDefault(),
                onclick: () => {
                    addItemFromSearch(node, p, it);
                    search.value = "";
                    results.replaceChildren();
                    results.style.display = "none";
                },
            }, [el("span", { className: "mugen-dim", text: `${catTitle(p.category)} › ${p.label} › ` }), el("span", { text: it.prompt })]));
        }
        results.style.display = "";
    };
    search.addEventListener("input", () => {
        clearTimeout(searchTimer);
        searchTimer = setTimeout(runSearch, 150);
    });
    search.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            search.value = "";
            runSearch();
        } else if (e.key === "Enter") {
            results.querySelector(".mugen-result")?.click();
        }
    });

    // --- recipe header + load list -------------------------------------
    const nameSpan = el("span", { className: "mugen-title" });
    const loadList = el("div", { className: "mugen-results", style: "display:none" });
    const toggleLoadList = async () => {
        if (loadList.style.display !== "none") {
            loadList.style.display = "none";
            return;
        }
        loadList.replaceChildren(el("div", { className: "mugen-empty", text: "Loading…" }));
        loadList.style.display = "";
        try {
            const { recipes } = await fetchJSON("/mykee/ugen/recipes");
            loadList.replaceChildren();
            if (!recipes?.length) loadList.append(el("div", { className: "mugen-empty", text: "No saved recipes yet." }));
            for (const name of recipes || []) {
                if (typeof name !== "string") continue;
                loadList.append(el("div", { className: "mugen-result", onclick: () => loadRecipe(node, name) }, [el("span", { text: name })]));
            }
        } catch (e) {
            loadList.replaceChildren(el("div", { className: "mugen-empty", text: `Error: ${e.message}` }));
        }
    };
    node._mugenCloseLoadList = () => { loadList.style.display = "none"; };

    const head = el("div", { className: "mugen-head" }, [
        nameSpan,
        el("div", { className: "mugen-row" }, [
            btn("Load", toggleLoadList, "", "Load a saved recipe"),
            btn("Save", () => saveRecipe(node), "", "Save the recipe to a file"),
            btn("Reset all", () => {
                const st = S(node);
                if ((st.entries.length || st.subjects.length) && !window.confirm("Clear the whole recipe (every tab and person)?")) return;
                st.entries = [];
                st.subjects = [];
                st.name = "";
                st.ui.collapsed = [];
                st.ui.active = SCENE;
                persist(node);
                node._mugenRender();
            }, "danger", "Start a new, empty recipe"),
        ]),
    ]);

    const tabsBox = el("div", { className: "mugen-tabs" });
    const subjectBox = el("div", { className: "mugen-row" });
    const groupsBox = el("div", { style: "display:flex;flex-direction:column;gap:6px" });
    const outBox = el("div", { style: "display:flex;flex-direction:column;gap:6px" });

    inner.append(el("div", { className: "mugen-row" }, [search, browseBtn]), results, head, loadList, tabsBox, subjectBox, groupsBox, outBox);

    // --- subjects --------------------------------------------------------
    const allUids = () => new Set(S(node).entries.map((x) => x.uid));
    const freeSid = () => newId(new Set([SCENE, ...S(node).subjects.map((x) => x.sid)]));
    const addSubject = () => {
        const st = S(node);
        if (st.subjects.length >= MAX_SUBJECTS) {
            toast(`At most ${MAX_SUBJECTS} subjects.`, "warn");
            return;
        }
        const sid = freeSid();
        st.subjects.push({ sid, name: `person ${st.subjects.length + 1}` });
        st.ui.active = sid;
        persist(node);
        render();
        subjectBox.querySelector("input")?.select();
    };
    const duplicateSubject = (sid) => {
        const st = S(node);
        const src = st.subjects.find((x) => x.sid === sid);
        if (!src || st.subjects.length >= MAX_SUBJECTS) return;
        const nsid = freeSid();
        st.subjects.splice(st.subjects.indexOf(src) + 1, 0, { sid: nsid, name: `${src.name} copy`.slice(0, MAX_SUBJECT_NAME) });
        const taken = allUids();
        for (const e of st.entries.filter((x) => x.subject === sid)) {
            const uid = newId(taken);
            taken.add(uid);
            st.entries.push({ ...JSON.parse(JSON.stringify(e)), uid, subject: nsid });
        }
        st.ui.active = nsid;
        persist(node);
        render();
    };
    const moveSubject = (sid, dir) => {
        const st = S(node);
        const i = st.subjects.findIndex((x) => x.sid === sid);
        const j = i + dir;
        if (i < 0 || j < 0 || j >= st.subjects.length) return;
        [st.subjects[i], st.subjects[j]] = [st.subjects[j], st.subjects[i]];
        persist(node);
        render();
    };
    const deleteSubject = (sid) => {
        const st = S(node);
        const sub = st.subjects.find((x) => x.sid === sid);
        if (!sub) return;
        const n = st.entries.filter((x) => x.subject === sid).length;
        if (n && !window.confirm(`Delete '${sub.name}' and its ${n} line(s)?`)) return;
        const idx = st.subjects.indexOf(sub);
        st.subjects = st.subjects.filter((x) => x !== sub);
        st.entries = st.entries.filter((x) => x.subject !== sid);
        st.ui.collapsed = st.ui.collapsed.filter((c) => !c.startsWith(`${sid}\u0002`));
        st.ui.active = st.subjects[Math.min(idx, st.subjects.length - 1)]?.sid || SCENE;
        persist(node);
        render();
    };

    // --- rendering ------------------------------------------------------
    const render = () => {
        const st = S(node);
        normalize(st);
        const picks = node._mugenPicks || {};
        const lib = LIB.data;
        nameSpan.textContent = `RECIPE${st.name ? " · " + st.name : ""}`;
        const active = st.ui.active;

        // tab bar: Scene, the subjects, + Person
        tabsBox.replaceChildren();
        for (const tab of [SCENE, ...st.subjects.map((x) => x.sid)]) {
            const n = st.entries.filter((e) => e.subject === tab).length;
            const name = tabName(st, tab);
            tabsBox.append(el("button", {
                type: "button", className: `mugen-tab${tab === active ? " on" : ""}`,
                text: n ? `${name} (${n})` : name, title: name,
                onclick: (ev) => {
                    ev.stopPropagation();
                    st.ui.active = tab;
                    persist(node);
                    render();
                },
            }));
        }
        tabsBox.append(btn("+ Person", addSubject, "mugen-tab add", "Add a subject (person, creature, ...) with its own attributes"));

        // the active subject's name and actions
        subjectBox.replaceChildren();
        const sub = st.subjects.find((x) => x.sid === active);
        subjectBox.style.display = sub ? "" : "none";
        if (sub) {
            const idx = st.subjects.indexOf(sub);
            const nameIn = el("input", {
                className: "mugen-input", type: "text", value: sub.name, maxLength: MAX_SUBJECT_NAME, spellcheck: false,
                title: "How the prompt calls this subject - e.g. a description or a name with a description",
            });
            nameIn.addEventListener("change", () => {
                sub.name = str(nameIn.value, MAX_SUBJECT_NAME).replace(/\s+/g, " ").trim() || `person ${idx + 1}`;
                persist(node);
                render();
            });
            const left = btn("◀", () => moveSubject(sub.sid, -1), "icon", "Move earlier (earlier in the prompt)");
            const right = btn("▶", () => moveSubject(sub.sid, 1), "icon", "Move later");
            left.disabled = idx === 0;
            right.disabled = idx === st.subjects.length - 1;
            subjectBox.append(el("span", { className: "mugen-dim", text: "Name" }), nameIn, left, right,
                btn("⧉", () => duplicateSubject(sub.sid), "icon", "Duplicate this subject with all its lines"),
                btn("✕", () => deleteSubject(sub.sid), "icon", "Delete this subject"));
        }

        groupsBox.replaceChildren();
        const tabEntries = st.entries.filter((e) => e.subject === active);
        if (!tabEntries.length) {
            groupsBox.append(el("div", { className: "mugen-empty", text: active === SCENE
                ? "Scene: setting, lighting, camera, style... - search above or open Browse. Add people with + Person."
                : "No attributes yet - search above or open Browse." }));
        }

        const groups = [];
        for (const e of tabEntries) {
            const g = groupKey(e);
            let grp = groups.find((x) => x.key === g);
            if (!grp) groups.push((grp = { key: g, name: groupName(e), entries: [] }));
            grp.entries.push(e);
        }
        if (!groups.some((g) => g.name === MANUAL_GROUP)) groups.push({ key: `${active}\u0002${MANUAL_GROUP}`, name: MANUAL_GROUP, entries: [] });

        for (const g of groups) {
            const manual = g.name === MANUAL_GROUP;
            const collapsed = st.ui.collapsed.includes(g.key);
            const box = el("div", { className: `mugen-group${manual ? " manual" : ""}` });
            const ghead = el("div", {
                className: "mugen-ghead",
                onclick: () => {
                    st.ui.collapsed = collapsed ? st.ui.collapsed.filter((c) => c !== g.key) : [...st.ui.collapsed, g.key];
                    persist(node);
                    render();
                },
            }, [
                el("span", { className: "mugen-dim", text: collapsed ? "▸" : "▾" }),
                el("span", { className: "mugen-gname", text: manual ? "Manual" : catTitle(g.name), style: manual ? "color:#8fd8cb" : "" }),
                collapsed && g.entries.length ? el("span", { className: "mugen-badge count", text: String(g.entries.length) }) : null,
                g.entries.length ? btn("↺", () => {
                    if (!window.confirm(`Remove every line of '${manual ? "Manual" : catTitle(g.name)}'?`)) return;
                    st.entries = st.entries.filter((e) => groupKey(e) !== g.key);
                    persist(node);
                    render();
                }, "icon", "Reset this group") : null,
            ]);
            box.append(ghead);
            if (!collapsed) {
                const body = el("div", { className: "mugen-gbody" });
                g.entries.forEach((e, idx) => body.append(renderLine(node, e, idx, g.entries.length, picks, lib)));
                if (manual) {
                    body.append(btn("+ Add manual line", () => {
                        st.entries.push({ uid: newId(allUids()), subject: active, kind: "manual", text: "" });
                        persist(node);
                        render();
                        const inputs = groupsBox.querySelectorAll(".mugen-group.manual input");
                        inputs[inputs.length - 1]?.focus();
                    }, "link"));
                }
                box.append(body);
            }
            groupsBox.append(box);
        }

        // last run's output
        const last = lastOutput(node);
        outBox.replaceChildren();
        if (last.tags || last.prompt) {
            const copyBtn = (text) => btn("copy", () => navigator.clipboard?.writeText(text).then(
                () => toast("Copied.", "success", 1500), () => toast("The browser did not allow copying.", "warn")), "link");
            outBox.append(
                el("div", {}, [el("div", { className: "mugen-outlabel" }, [el("span", { text: "TAGS (last run)" }), copyBtn(last.tags || "")]),
                    el("div", { className: "mugen-out", text: last.tags || "" })]),
                el("div", {}, [el("div", { className: "mugen-outlabel" }, [el("span", { text: "PROMPT (last run)" }), copyBtn(last.prompt || "")]),
                    el("div", { className: "mugen-out", text: last.prompt || "" })]),
            );
        }
        queueMeasure();
    };
    node._mugenRender = render;
    return dom;
}

function renderLine(node, e, idx, count, picks, lib) {
    const moveBtns = [
        btn("▲", () => moveEntry(node, e.uid, -1), "icon", "Move up"),
        btn("▼", () => moveEntry(node, e.uid, 1), "icon", "Move down"),
        btn("✕", () => removeEntry(node, e.uid), "icon", "Remove"),
    ];
    moveBtns[0].disabled = idx === 0;
    moveBtns[1].disabled = idx === count - 1;

    if (e.kind === "manual") {
        const input = el("input", { className: "mugen-input", type: "text", value: e.text, placeholder: "free text…", spellcheck: false });
        input.addEventListener("change", () => {
            e.text = str(input.value, MAX_MANUAL_TEXT);
            persist(node);
        });
        return el("div", { className: "mugen-line" }, [input, ...moveBtns]);
    }

    const missing = lib && !lib.properties.has(propKey(e.category, e.property));
    const val = el("div", { className: "mugen-val" });
    if (e.mode === "fixed") {
        val.title = "Click to edit this line by hand";
        val.append(el("span", { text: entryValuesText(e) }));
        if (e.edited) {
            val.append(el("span", { className: "mugen-badge edit", text: "edited" }));
            val.append(btn("⟲", () => {
                e.edited = false;
                e.text = "";
                persist(node);
                node._mugenRender();
            }, "icon", "Restore the picked values"));
        }
        val.addEventListener("click", (ev) => {
            if (ev.target.closest("button") || val.querySelector("input")) return;
            const original = entryValuesText(e);
            const input = el("input", { className: "mugen-input", type: "text", value: original, spellcheck: false });
            val.replaceChildren(input);
            input.focus();
            let done = false;
            const commit = (save) => {
                if (done) return;
                done = true;
                const v = str(input.value, MAX_EDIT_TEXT).trim();
                if (save && v && v !== original) {
                    e.edited = true;
                    e.text = v;
                    persist(node);
                }
                node._mugenRender();
            };
            input.addEventListener("keydown", (k) => {
                if (k.key === "Enter") commit(true);
                if (k.key === "Escape") commit(false);
            });
            input.addEventListener("blur", () => commit(true));
        });
    } else {
        const n = e.items.length;
        const range = e.ptype === "single" ? "1" : e.min === e.max ? `${Math.min(e.min, n)}` : `${Math.min(e.min, n)}–${Math.min(e.max, n)}`;
        val.style.cursor = "default";
        val.append(el("span", { className: "mugen-badge rnd", text: `random ${range} of ${n}` }));
        const p = entryValuesText(e, picks);
        val.append(el("span", { className: "mugen-dim", text: p || "(picked when run)" }));
    }
    if (missing) val.append(el("span", { className: "mugen-badge miss", text: "source missing", title: "The library file of this line is gone - the line still works from its saved copy." }));
    return el("div", { className: "mugen-line" }, [
        el("span", { className: "mugen-plabel", text: e.propLabel, title: `${catTitle(e.category)} › ${e.propLabel}` }),
        val,
        ...moveBtns,
    ]);
}

async function loadRecipe(node, name) {
    const st = S(node);
    if (st.entries.length && !window.confirm(`Replace the current recipe with '${name}'?`)) return;
    try {
        const { recipe } = await fetchJSON(`/mykee/ugen/recipe/load?name=${encodeURIComponent(name)}`);
        const clean = cleanRecipe(recipe);
        clean.name = name;
        node._mugen = clean;
        node._mugenPicks = {};
        persist(node);
        node._mugenCloseLoadList?.();
        node._mugenRender();
        toast(`Recipe '${name}' loaded.`, "success", 2500);
    } catch (e) {
        toast(`Load failed: ${e.message}`, "error");
    }
}

async function saveRecipe(node) {
    const st = S(node);
    const name = askName("Recipe name:", st.name || "");
    if (!name) return;
    const body = { name, recipe: { ...st, ui: undefined }, overwrite: false };
    try {
        await postJSON("/mykee/ugen/recipe/save", body);
    } catch (e) {
        if (e.status !== 409 || !window.confirm(`'${name}' already exists. Overwrite it?`)) {
            if (e.status !== 409) toast(`Save failed: ${e.message}`, "error");
            return;
        }
        try {
            await postJSON("/mykee/ugen/recipe/save", { ...body, overwrite: true });
        } catch (e2) {
            toast(`Save failed: ${e2.message}`, "error");
            return;
        }
    }
    st.name = name;
    persist(node);
    node._mugenRender();
    toast(`Recipe '${name}' saved.`, "success", 2500);
}

// ---------------------------------------------------------------------------
// The library window
// ---------------------------------------------------------------------------

// The tab a property's line lives in while the window sends to `active`
// (a property missing from the library stays with the active tab).
function tabForKey(key, lib, active) {
    const prop = lib.properties.get(key);
    return prop ? targetTab(prop, active) : active;
}

function draftFromRecipe(recipe, active, lib) {
    const draft = new Map();
    for (const e of recipe.entries) {
        if (e.kind !== "lib") continue;
        const key = propKey(e.category, e.property);
        if (e.subject !== tabForKey(key, lib, active)) continue;
        draft.set(key, {
            mode: e.mode, min: e.min, max: e.max,
            // random from all uses every value: nothing is "ticked" by hand
            checked: new Set(e.mode === "random_all" ? [] : e.items.map((i) => i.id).filter(Boolean)),
        });
    }
    return draft;
}

const sameSet = (a, b) => a.size === b.size && [...a].every((x) => b.has(x));

function draftActive(d) {
    return !!d && (d.mode === "random_all" || d.checked.size > 0);
}

// Differences between the draft and the recipe as it was when the window opened.
function computeChanges(orig, draft, recipe, lib, active) {
    const keys = new Set([...orig.keys(), ...draft.keys()]);
    const out = [];
    const textOf = (key, id) => {
        const p = lib.properties.get(key);
        const it = p?.items.find((i) => i.id === id);
        if (it) return it.prompt;
        const [cat, file] = key.split("\u0001");
        return findEntry(recipe, tabForKey(key, lib, active), cat, file)?.items.find((i) => i.id === id)?.prompt || id;
    };
    for (const key of keys) {
        const o = orig.get(key);
        const d = draft.get(key);
        const oa = draftActive(o);
        const da = draftActive(d);
        if (!oa && !da) continue;
        const lines = [];
        if (!da) {
            lines.push(["rem", oa && o.mode === "random_all" ? "random from all" : `${o.checked.size} value(s)`]);
        } else {
            if (!oa || o.mode !== d.mode) {
                if (oa) lines.push(["chg", `${MODE_LABEL[o.mode]} → ${MODE_LABEL[d.mode]}`]);
                else if (d.mode !== "fixed") lines.push(["chg", MODE_LABEL[d.mode]]);
            }
            if (oa && d.mode !== "fixed" && o.mode !== "fixed" && (o.min !== d.min || o.max !== d.max)) {
                lines.push(["chg", `count ${o.min}–${o.max} → ${d.min}–${d.max}`]);
            }
            if (d.mode !== "random_all") {
                const before = oa && o.mode !== "random_all" ? o.checked : new Set();
                for (const id of d.checked) if (!before.has(id)) lines.push(["add", textOf(key, id)]);
                for (const id of before) if (!d.checked.has(id)) lines.push(["rem", textOf(key, id)]);
            }
        }
        if (!lines.length) continue;
        const [cat, file] = key.split("\u0001");
        const tab = tabForKey(key, lib, active);
        const label = lib.properties.get(key)?.label || findEntry(recipe, tab, cat, file)?.propLabel || file;
        const where = tab !== active && tab ? `[${tabName(recipe, tab)}] ` : "";
        out.push({ key, title: `${where}${catTitle(cat)} › ${label}`, lines });
    }
    return out;
}

function applyDraft(node, orig, draft, lib, active) {
    const st = S(node);
    const keys = new Set([...orig.keys(), ...draft.keys()]);
    let changed = 0;
    for (const key of keys) {
        const o = orig.get(key);
        const d = draft.get(key);
        const same = draftActive(o) === draftActive(d) && (!draftActive(d)
            || (o.mode === d.mode && o.min === d.min && o.max === d.max && sameSet(o.checked, d.checked)));
        if (same) continue; // untouched: leave the line exactly as it is
        const [cat, file] = key.split("\u0001");
        const prop = lib.properties.get(key);
        const tab = tabForKey(key, lib, active);
        if (!tab) continue; // a person property while the scene is chosen - never ticked
        const entry = findEntry(st, tab, cat, file);
        changed++;
        if (!draftActive(d)) {
            if (entry) st.entries = st.entries.filter((e) => e !== entry);
            continue;
        }
        let items;
        if (d.mode === "random_all") {
            items = (prop?.items || entry?.items || []).map((i) => ({ ...i }));
        } else {
            const old = new Map((entry?.items || []).map((i) => [i.id, i]));
            const order = prop ? prop.items.map((i) => i.id) : [...d.checked];
            items = [];
            for (const id of order) if (d.checked.has(id)) {
                const it = prop?.items.find((i) => i.id === id) || old.get(id);
                if (it) items.push({ id: it.id, prompt: it.prompt, label: it.label });
            }
            for (const id of d.checked) if (!order.includes(id) && old.get(id)) items.push({ ...old.get(id) });
        }
        if (entry) {
            entry.mode = d.mode;
            entry.min = d.min;
            entry.max = d.max;
            entry.items = items;
            entry.edited = false; // the picked values changed - the hand edit no longer matches them
            entry.text = "";
            if (prop) {
                entry.propLabel = prop.label;
                entry.order = prop.order;
                entry.ptype = prop.type;
            }
        } else if (prop) {
            insertLibEntry(st, {
                uid: newId(new Set(st.entries.map((x) => x.uid))), subject: tab, kind: "lib",
                category: cat, property: file, propLabel: prop.label, order: prop.order,
                ptype: prop.type, mode: d.mode, min: d.min, max: d.max, items, edited: false, text: "",
            });
        }
    }
    persist(node);
    node._mugenRender?.();
    return changed;
}

function renameRecipeRefs(node, fromCat, fromFile, toCat, toFile) {
    // fromFile null = a whole category (and its sub-categories) was renamed
    const st = S(node);
    for (const e of st.entries) {
        if (e.kind !== "lib") continue;
        if (fromFile === null) {
            if (e.category === fromCat || e.category.startsWith(fromCat + "/")) e.category = toCat + e.category.slice(fromCat.length);
        } else if (e.category === fromCat && e.property === fromFile) {
            e.category = toCat;
            e.property = toFile;
        }
    }
    st.ui.collapsed = st.ui.collapsed.map((c) => (fromFile === null && (c === fromCat || c.startsWith(fromCat + "/")) ? toCat + c.slice(fromCat.length) : c));
    persist(node);
}

async function openLibrary(node) {
    if (document.querySelector(".mugen-overlay")) return;
    injectStyle();
    let lib;
    try {
        lib = await loadLibrary(true);
    } catch (e) {
        toast(`Could not load the library: ${e.message}`, "error");
        return;
    }

    let target = S(node).ui.active;
    let orig = draftFromRecipe(S(node), target, lib);
    let draft = draftFromRecipe(S(node), target, lib);
    const ui = {
        selCat: null, selProp: null, view: "browse", expanded: new Set(), filter: "",
        edit: null, dirty: false,
    };
    // open the categories that have something picked
    for (const key of orig.keys()) {
        const cat = key.split("\u0001")[0];
        ui.expanded.add(cat.split("/")[0]);
        ui.expanded.add(cat);
    }

    const overlay = el("div", { className: "mugen-overlay" });
    const dialog = el("div", { className: "mugen-dialog", role: "dialog", "aria-label": "Universal Generator library" });
    overlay.append(dialog);
    for (const ev of ["keydown", "keyup", "keypress", "wheel", "pointerdown", "mousedown"]) overlay.addEventListener(ev, (e) => e.stopPropagation());

    const close = (force = false) => {
        if (!force && ui.dirty && !window.confirm("The property editor has unsaved changes. Close anyway?")) return;
        const pending = computeChanges(orig, draft, S(node), lib, target).length;
        if (!force && pending && !window.confirm(`${pending} change(s) were not sent to the recipe. Close anyway?`)) return;
        overlay.remove();
        node._mugenCloseLibrary = null;
    };
    node._mugenCloseLibrary = () => close(true);
    overlay.addEventListener("keydown", (e) => {
        if (e.key === "Escape") close();
    });

    const filterInput = el("input", { className: "mugen-input", type: "text", placeholder: "Filter categories, properties and values…", spellcheck: false });
    filterInput.addEventListener("input", () => {
        ui.filter = filterInput.value.trim().toLowerCase();
        renderTree();
    });
    const targetSel = el("select", { className: "mugen-select", title: "The tab the picked values go to (scene properties always go to the Scene)" });
    const fillTargets = () => {
        const st = S(node);
        targetSel.replaceChildren(...[SCENE, ...st.subjects.map((x) => x.sid)].map((t) =>
            el("option", { value: t, text: tabName(st, t), selected: t === target })));
    };
    fillTargets();
    targetSel.addEventListener("change", () => {
        const pending = computeChanges(orig, draft, S(node), lib, target).length;
        if (pending && !window.confirm(`${pending} change(s) for '${tabName(S(node), target)}' were not sent. Discard them?`)) {
            targetSel.value = target;
            return;
        }
        const st = S(node);
        target = [SCENE, ...st.subjects.map((x) => x.sid)].includes(targetSel.value) ? targetSel.value : SCENE;
        st.ui.active = target; // the node shows the same tab
        persist(node);
        node._mugenRender?.();
        orig = draftFromRecipe(st, target, lib);
        draft = draftFromRecipe(st, target, lib);
        renderAll();
    });
    dialog.append(el("div", { className: "mugen-dhead" }, [
        el("span", { text: "Library", style: "font-weight:600;font-size:14px" }),
        filterInput,
        el("span", { className: "mugen-dim", text: "Sending to" }),
        targetSel,
        btn("⟳", async () => {
            if (ui.dirty && !window.confirm("Discard the unsaved editor changes?")) return;
            await reload();
        }, "icon", "Reload the library from disk"),
        btn("✕", () => close(), "icon", "Close"),
    ]));

    const treeList = el("div", { className: "mugen-treelist" });
    const tree = el("div", { className: "mugen-tree" }, [
        el("div", { className: "mugen-row", style: "padding:8px" }, [
            btn("+ Category", () => newCategory(), "", "New category (inside the selected top-level category, or at the top)"),
            btn("+ Property", () => newProperty(), "", "New property (list) in the selected category"),
        ]),
        treeList,
    ]);
    const main = el("div", { className: "mugen-main" });
    const sideList = el("div", { className: "mugen-sidelist" });
    const side = el("div", { className: "mugen-side" }, [
        el("div", { text: "Changes to send", style: "padding:12px 14px 8px;font-weight:600" }),
        sideList,
        el("div", { className: "mugen-dim", style: "padding:10px 14px;border-top:1px solid #3a3a3a;font-size:12px", text: "Manual lines, untouched lines and lines whose library file is gone stay as they are." }),
    ]);
    dialog.append(el("div", { className: "mugen-dbody" }, [tree, main, side]));

    const sendBtn = btn("Send to recipe", () => {
        if (ui.dirty && !window.confirm("The property editor has unsaved changes (they are not part of the selection). Send anyway?")) return;
        const n = applyDraft(node, orig, draft, lib, target);
        orig = draftFromRecipe(S(node), target, lib);
        draft = draftFromRecipe(S(node), target, lib);
        toast(n ? `Recipe updated (${n} propert${n === 1 ? "y" : "ies"}).` : "Nothing to send.", n ? "success" : "info", 2500);
        renderAll();
    }, "primary");
    dialog.append(el("div", { className: "mugen-dfoot" }, [
        btn("Clear selection", () => {
            if (!window.confirm("Untick everything? (The recipe changes only when you press Send.)")) return;
            for (const d of draft.values()) {
                d.checked.clear();
                if (d.mode === "random_all") d.mode = "fixed";
            }
            renderAll();
        }, "danger", "Untick every value of every property"),
        el("div", { className: "mugen-row" }, [btn("Close", () => close()), sendBtn]),
    ]));

    document.body.append(overlay);
    filterInput.focus();

    // --- data helpers ------------------------------------------------------
    const propsOf = (cat) => [...lib.properties.values()].filter((p) => p.category === cat)
        .sort((a, b) => a.order - b.order || a.label.localeCompare(b.label));
    const childrenOf = (path) => lib.categories.filter((c) => (path === null ? c.depth === 1 : c.depth === path.split("/").length + 1 && c.path.startsWith(path + "/")));
    const getDraft = (key, prop) => {
        if (!draft.has(key)) draft.set(key, { mode: "fixed", min: 1, max: Math.max(1, Math.min(2, prop?.items.length || 1)), checked: new Set() });
        return draft.get(key);
    };

    async function reload() {
        try {
            lib = await loadLibrary(true);
        } catch (e) {
            toast(`Could not load the library: ${e.message}`, "error");
            return;
        }
        if (ui.selProp && !lib.properties.has(ui.selProp)) ui.selProp = null;
        if (ui.selCat && !lib.categories.some((c) => c.path === ui.selCat)) ui.selCat = null;
        ui.view = "browse";
        ui.edit = null;
        ui.dirty = false;
        node._mugenRender?.();
        renderAll();
    }

    function guardEditor() {
        return !ui.dirty || window.confirm("Discard the unsaved editor changes?");
    }

    // --- tree --------------------------------------------------------------
    function matchesFilter(p) {
        if (!ui.filter) return true;
        const hay = `${p.category} ${p.label} ${p.file} ${p.items.map((i) => i.prompt + " " + i.label).join(" ")}`.toLowerCase();
        return ui.filter.split(/\s+/).every((w) => hay.includes(w));
    }

    function renderTree() {
        treeList.replaceChildren();
        const renderCat = (c) => {
            const props = propsOf(c.path).filter(matchesFilter);
            const kids = childrenOf(c.path);
            const kidRows = [];
            for (const k of kids) {
                const r = renderCat(k);
                if (r) kidRows.push(r);
            }
            if (ui.filter && !props.length && !kidRows.length && !c.path.toLowerCase().includes(ui.filter)) return null;
            const open = ui.filter ? true : ui.expanded.has(c.path);
            const pad = (c.depth - 1) * 16;
            const frag = el("div");
            const row = el("div", {
                className: `mugen-tnode${ui.selCat === c.path && !ui.selProp ? " sel" : ""}`,
                style: `padding-left:${6 + pad}px`,
                onclick: () => {
                    if (!guardEditor()) return;
                    ui.dirty = false;
                    if (ui.expanded.has(c.path) && ui.selCat === c.path && !ui.selProp) ui.expanded.delete(c.path);
                    else ui.expanded.add(c.path);
                    ui.selCat = c.path;
                    ui.selProp = null;
                    ui.view = "browse";
                    renderAll();
                },
            }, [
                el("span", { className: "mugen-dim", text: open ? "▾" : "▸" }),
                el("span", { className: "mugen-tname", text: c.name, style: "font-weight:600" }),
                el("span", { className: "mugen-tacts" }, [
                    btn("✎", () => renameCategory(c.path), "icon", "Rename category"),
                    btn("🗑", () => deleteCategory(c.path), "icon", "Move category to the trash"),
                ]),
            ]);
            frag.append(row);
            if (open) {
                for (const kr of kidRows) frag.append(kr);
                for (const p of props) {
                    const key = propKey(p.category, p.file);
                    const d = draft.get(key);
                    let badge = null;
                    if (d?.mode === "random_all") badge = el("span", { className: "mugen-badge rnd", text: "rnd all" });
                    else if (d?.checked.size) badge = el("span", { className: d.mode === "fixed" ? "mugen-badge count" : "mugen-badge rnd", text: String(d.checked.size) });
                    frag.append(el("div", {
                        className: `mugen-tnode${ui.selProp === key ? " sel" : ""}`,
                        style: `padding-left:${22 + pad}px`,
                        onclick: () => {
                            if (ui.selProp === key) return;
                            if (!guardEditor()) return;
                            ui.dirty = false;
                            ui.selProp = key;
                            ui.selCat = p.category;
                            ui.view = "browse";
                            renderAll();
                        },
                    }, [el("span", { className: "mugen-tname", text: p.label }), badge]));
                }
            }
            return frag;
        };
        treeList.append(el("div", {
            className: `mugen-tnode${ui.selCat === null && !ui.selProp ? " sel" : ""}`,
            title: "Select the root: + Category then creates a top-level category",
            onclick: () => {
                if (!guardEditor()) return;
                ui.dirty = false;
                ui.selCat = null;
                ui.selProp = null;
                ui.view = "browse";
                renderAll();
            },
        }, [el("span", { className: "mugen-tname mugen-dim", text: "Library root" })]));
        for (const c of childrenOf(null)) {
            const r = renderCat(c);
            if (r) treeList.append(r);
        }
        if (!lib.categories.length) treeList.append(el("div", { className: "mugen-empty", text: "No categories yet - press + Category." }));
    }

    // --- middle: browse ------------------------------------------------------
    function renderBrowse() {
        main.replaceChildren();
        const prop = lib.properties.get(ui.selProp);
        if (!prop) {
            main.append(el("div", { className: "mugen-mbody" }, [el("div", { className: "mugen-empty", text: ui.selCat
                ? `Category '${catTitle(ui.selCat)}' - pick a property on the left, or add one with + Property.`
                : "Pick a property on the left." })]));
            return;
        }
        const dest = targetTab(prop, target);
        if (!dest) {
            main.append(el("div", { className: "mugen-mbody" }, [
                el("div", { style: "font-size:17px;font-weight:600;margin-bottom:8px", text: prop.label }),
                el("div", { className: "mugen-empty", text: "This property belongs to a person. Choose a person in 'Sending to' at the top, or add one with '+ Person' on the node first." }),
                btn("Edit property", () => { ui.view = "edit"; ui.edit = null; renderAll(); }),
            ]));
            return;
        }
        const d = getDraft(ui.selProp, prop);
        const segBtns = MODES.map((m) => el("button", {
            type: "button", className: d.mode === m ? "on" : "", text: MODE_LABEL[m][0].toUpperCase() + MODE_LABEL[m].slice(1),
            onclick: () => {
                d.mode = m;
                renderAll();
            },
        }));
        const countRow = [];
        if (d.mode !== "fixed" && prop.type === "multi") {
            const lo = el("input", { className: "mugen-input mugen-num", type: "number", min: 1, max: MAX_ITEMS, value: d.min });
            const hi = el("input", { className: "mugen-input mugen-num", type: "number", min: 1, max: MAX_ITEMS, value: d.max });
            const upd = () => {
                d.min = int(Number(lo.value), 1, MAX_ITEMS, 1);
                d.max = int(Number(hi.value), d.min, MAX_ITEMS, d.min);
                lo.value = d.min;
                hi.value = d.max;
                renderSide();
            };
            lo.addEventListener("change", upd);
            hi.addEventListener("change", upd);
            countRow.push(el("span", { className: "mugen-dim", text: "count" }), lo, el("span", { text: "–" }), hi);
        }
        const typeText = prop.type === "single" ? "single" : `multi${prop.max ? ` · max ${prop.max}` : ""}`;
        main.append(el("div", { className: "mugen-mhead" }, [
            el("div", { className: "mugen-head" }, [
                el("div", {}, [
                    el("div", { className: "mugen-dim", text: catTitle(prop.category) + (dest !== target ? `  →  goes to ${tabName(S(node), dest)}` : ""), style: "font-size:12px" }),
                    el("div", { style: "font-size:17px;font-weight:600" }, [prop.label, el("span", { className: "mugen-dim", style: "font-size:12px;font-weight:400", text: ` · ${typeText} · ${prop.file}.json` })]),
                ]),
                btn("Edit property", () => {
                    ui.view = "edit";
                    ui.edit = null;
                    renderAll();
                }),
            ]),
            el("div", { className: "mugen-row", style: "flex-wrap:wrap" }, [
                el("span", { className: "mugen-dim", text: "Mode" }), el("div", { className: "mugen-seg" }, segBtns), ...countRow,
                el("span", { style: "flex:1" }),
                d.mode !== "random_all" && prop.type === "multi" ? btn("Select all", () => {
                    prop.items.forEach((i) => d.checked.add(i.id));
                    renderAll();
                }, "link") : null,
                d.mode !== "random_all" ? btn("Select none", () => {
                    d.checked.clear();
                    renderAll();
                }, "link") : null,
            ]),
        ]));
        const grid = el("div", { className: "mugen-grid" });
        const single = prop.type === "single" && d.mode === "fixed";
        for (const it of prop.items) {
            const on = d.mode === "random_all" || d.checked.has(it.id);
            grid.append(el("div", {
                className: `mugen-item${on ? " on" : ""}${d.mode === "random_all" ? " off" : ""}`,
                title: it.prompt,
                onclick: () => {
                    if (d.mode === "random_all") return;
                    if (d.checked.has(it.id)) d.checked.delete(it.id);
                    else {
                        if (single) d.checked.clear();
                        else if (d.mode === "fixed" && prop.max > 0 && d.checked.size >= prop.max) {
                            toast(`'${prop.label}' allows at most ${prop.max} value(s).`, "warn", 2500);
                            return;
                        }
                        d.checked.add(it.id);
                    }
                    renderAll();
                },
            }, [
                el("span", { className: "mugen-check", text: on ? "✓" : "" }),
                el("span", { text: it.prompt, style: "flex:1;min-width:0;overflow-wrap:anywhere" }),
                it.label !== it.prompt ? el("span", { className: "mugen-dim", text: it.label }) : null,
            ]));
        }
        const body = el("div", { className: "mugen-mbody" }, [grid]);
        if (!prop.items.length) body.append(el("div", { className: "mugen-empty", text: "This list is empty - press Edit property to add values." }));
        main.append(body);
    }

    // --- middle: edit --------------------------------------------------------
    function renderEdit() {
        main.replaceChildren();
        const prop = lib.properties.get(ui.selProp);
        if (!prop) {
            ui.view = "browse";
            renderBrowse();
            return;
        }
        if (!ui.edit) {
            ui.edit = {
                file: prop.file, category: prop.category, label: prop.label, type: prop.type, max: prop.max, order: prop.order, scope: prop.scope,
                items: prop.items.map((i) => ({ id: i.id, prompt: i.prompt, label: i.label === i.prompt ? "" : i.label })),
                paste: false,
            };
        }
        const ed = ui.edit;
        const dirty = () => {
            ui.dirty = true;
            badge.style.display = "";
        };
        const badge = el("span", { className: "mugen-badge edit", text: "unsaved", style: ui.dirty ? "" : "display:none" });

        const field = (label, input) => el("label", { className: "mugen-field" }, [el("span", { text: label }), input]);
        const fileIn = el("input", { className: "mugen-input", type: "text", value: ed.file, spellcheck: false });
        fileIn.addEventListener("input", () => { ed.file = fileIn.value.trim(); dirty(); });
        const labelIn = el("input", { className: "mugen-input", type: "text", value: ed.label, maxLength: MAX_LABEL });
        labelIn.addEventListener("input", () => { ed.label = labelIn.value; dirty(); });
        const typeSel = el("select", { className: "mugen-select" }, ["single", "multi"].map((t) => el("option", { value: t, text: t, selected: ed.type === t })));
        typeSel.addEventListener("change", () => { ed.type = typeSel.value; dirty(); });
        const maxIn = el("input", { className: "mugen-input mugen-num", type: "number", min: 0, max: MAX_ITEMS, value: ed.max, title: "Most values in multi (0 = no limit)" });
        maxIn.addEventListener("change", () => { ed.max = int(Number(maxIn.value), 0, MAX_ITEMS, 0); maxIn.value = ed.max; dirty(); });
        const orderIn = el("input", { className: "mugen-input mugen-num", type: "number", min: -9999, max: 9999, value: ed.order, title: "Position in the prompt (lower = earlier)" });
        orderIn.addEventListener("change", () => { ed.order = int(Number(orderIn.value), -9999, 9999, 100); orderIn.value = ed.order; dirty(); });
        const catSel = el("select", { className: "mugen-select" }, lib.categories.map((c) => el("option", { value: c.path, text: catTitle(c.path), selected: ed.category === c.path })));
        const scopeSel = el("select", { className: "mugen-select", title: "scene: always the Scene tab · person: always a person tab · any: the active tab" },
            [["", "any"], ["scene", "scene"], ["subject", "person"]].map(([v, t]) => el("option", { value: v, text: t, selected: ed.scope === v })));
        scopeSel.addEventListener("change", () => { ed.scope = scopeSel.value; dirty(); });
        catSel.addEventListener("change", () => { ed.category = catSel.value; dirty(); });

        main.append(el("div", { className: "mugen-mhead" }, [
            el("div", { className: "mugen-head" }, [
                el("div", { className: "mugen-row" }, [el("span", { style: "font-size:17px;font-weight:600", text: `Editing: ${prop.label}` }), badge]),
                el("div", { className: "mugen-row" }, [
                    btn("Cancel", () => {
                        if (!guardEditor()) return;
                        ui.dirty = false;
                        ui.edit = null;
                        ui.view = "browse";
                        renderAll();
                    }),
                    btn("Save property", () => saveEdit(), "ok"),
                ]),
            ]),
            // two rows, so every field (and its drop-down) fits at any window width
            el("div", { style: "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;align-items:end" }, [
                field("File name (.json)", fileIn), field("Display name", labelIn), field("Category", catSel),
            ]),
            el("div", { style: "display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;align-items:end" }, [
                field("Belongs to", scopeSel), field("Type", typeSel), field("Max (0 = no limit)", maxIn), field("Order", orderIn),
            ]),
        ]));

        const rows = el("div");
        const renderRows = () => {
            rows.replaceChildren(el("div", { className: "mugen-erow mugen-dim", style: "font-size:11px" }, [
                el("span", { text: "PROMPT", style: "flex:3" }), el("span", { text: "LABEL (copied from the prompt until you change it)", style: "flex:2" }), el("span", { style: "width:78px" }),
            ]));
            ed.items.forEach((it, i) => {
                const p = el("input", { className: "mugen-input", type: "text", value: it.prompt, maxLength: MAX_ITEM_TEXT, style: "flex:3" });
                const l = el("input", { className: "mugen-input", type: "text", value: it.label, placeholder: it.prompt, maxLength: MAX_LABEL, style: "flex:2" });
                p.addEventListener("input", () => { it.prompt = p.value; l.placeholder = p.value; dirty(); });
                l.addEventListener("input", () => { it.label = l.value; dirty(); });
                const up = btn("▲", () => { [ed.items[i - 1], ed.items[i]] = [ed.items[i], ed.items[i - 1]]; dirty(); renderRows(); }, "icon", "Move up");
                const dn = btn("▼", () => { [ed.items[i + 1], ed.items[i]] = [ed.items[i], ed.items[i + 1]]; dirty(); renderRows(); }, "icon", "Move down");
                up.disabled = i === 0;
                dn.disabled = i === ed.items.length - 1;
                rows.append(el("div", { className: "mugen-erow" }, [p, l, up, dn,
                    btn("✕", () => { ed.items.splice(i, 1); dirty(); renderRows(); }, "icon", "Delete row")]));
            });
        };
        renderRows();

        const pasteBox = el("textarea", { className: "mugen-textarea", placeholder: "One value per line. Optional: prompt | label" });
        const pasteWrap = el("div", { style: ed.paste ? "margin-top:8px" : "display:none" }, [
            pasteBox,
            el("div", { className: "mugen-row", style: "margin-top:6px" }, [
                btn("Add these lines", () => {
                    const have = new Set(ed.items.map((i) => i.prompt.trim().toLowerCase()));
                    let added = 0;
                    for (const line of pasteBox.value.split(/\r?\n/)) {
                        const [pr, ...rest] = line.split("|");
                        const prompt = str(pr, MAX_ITEM_TEXT).trim();
                        if (!prompt || have.has(prompt.toLowerCase()) || ed.items.length >= MAX_ITEMS) continue;
                        have.add(prompt.toLowerCase());
                        const label = str(rest.join("|"), MAX_LABEL).trim();
                        ed.items.push({ id: "", prompt, label: label === prompt ? "" : label });
                        added++;
                    }
                    pasteBox.value = "";
                    ed.paste = false;
                    pasteWrap.style.display = "none";
                    if (added) dirty();
                    renderRows();
                    toast(`${added} value(s) added.`, "info", 2000);
                }, "primary"),
                btn("Cancel", () => { ed.paste = false; pasteWrap.style.display = "none"; }),
            ]),
        ]);
        const openPaste = async () => {
            ed.paste = true;
            pasteWrap.style.display = "";
            try {
                if (navigator.clipboard?.readText) pasteBox.value = await navigator.clipboard.readText();
            } catch {
                /* clipboard not allowed here - paste by hand */
            }
            pasteBox.focus();
        };

        main.append(el("div", { className: "mugen-mbody" }, [
            rows,
            el("div", { className: "mugen-row", style: "margin-top:8px" }, [
                btn("+ Add row", () => { ed.items.push({ id: "", prompt: "", label: "" }); dirty(); renderRows(); rows.querySelector(".mugen-erow:last-child input")?.focus(); }),
                btn("Paste list", openPaste, "", "Paste many values at once (one per line)"),
                el("span", { style: "flex:1" }),
                btn("Duplicate", () => duplicateProperty(prop)),
                btn("Move to trash", () => deleteProperty(prop), "danger"),
            ]),
            pasteWrap,
        ]));
    }

    async function saveEdit() {
        const prop = lib.properties.get(ui.selProp);
        const ed = ui.edit;
        if (!prop || !ed) return;
        if (!validName(ed.file)) {
            toast("Invalid file name: 1-80 characters, no / \\ : * ? \" < > |, no leading or trailing dot.", "error");
            return;
        }
        const items = ed.items.filter((i) => i.prompt.trim()).map((i) => ({
            ...(i.id ? { id: i.id } : {}), prompt: i.prompt.trim(), label: i.label.trim() || i.prompt.trim(),
        }));
        const data = { version: 1, label: ed.label.trim() || ed.file, type: ed.type, max: ed.max, order: ed.order, scope: ed.scope, items };
        try {
            const res = await postJSON("/mykee/ugen/property/save", {
                category: ed.category, file: ed.file, data, old_category: prop.category, old_file: prop.file,
            });
            const newKey = propKey(ed.category, ed.file);
            if (newKey !== ui.selProp) {
                renameRecipeRefs(node, prop.category, prop.file, ed.category, ed.file);
                for (const m of [orig, draft]) if (m.has(ui.selProp)) {
                    m.set(newKey, m.get(ui.selProp));
                    m.delete(ui.selProp);
                }
            }
            // ids that are gone from the list can no longer be ticked
            const ids = new Set((res.property?.items || []).map((i) => i.id));
            const d = draft.get(newKey);
            if (d) for (const id of [...d.checked]) if (!ids.has(id)) d.checked.delete(id);
            ui.dirty = false;
            ui.edit = null;
            ui.view = "browse";
            ui.selProp = newKey;
            ui.selCat = ed.category;
            await reload();
            toast(`'${data.label}' saved.`, "success", 2000);
        } catch (e) {
            toast(`Save failed: ${e.message}`, "error");
        }
    }

    // --- category / property actions ----------------------------------------
    async function newCategory() {
        if (!guardEditor()) return;
        const sel = ui.selCat ? ui.selCat.split("/") : [];
        const parent = sel.length >= 1 && sel.length < MAX_DEPTH ? sel.slice(0, 1).join("/") : sel.length >= MAX_DEPTH ? sel.slice(0, MAX_DEPTH - 1).join("/") : "";
        const name = askName(parent ? `New sub-category in '${catTitle(parent)}':` : "New top-level category:");
        if (!name) return;
        try {
            const { path } = await postJSON("/mykee/ugen/category/create", { parent, name });
            if (parent) ui.expanded.add(parent);
            ui.selCat = path;
            ui.selProp = null;
            await reload();
        } catch (e) {
            toast(`Could not create the category: ${e.message}`, "error");
        }
    }

    async function renameCategory(path) {
        if (!guardEditor()) return;
        const parts = path.split("/");
        const name = askName("New name for the category:", parts[parts.length - 1]);
        if (!name || name === parts[parts.length - 1]) return;
        try {
            const res = await postJSON("/mykee/ugen/category/rename", { path, name });
            renameRecipeRefs(node, path, null, res.path, null);
            const remap = (m) => {
                for (const key of [...m.keys()]) {
                    const [cat, file] = key.split("\u0001");
                    if (cat === path || cat.startsWith(path + "/")) {
                        m.set(propKey(res.path + cat.slice(path.length), file), m.get(key));
                        m.delete(key);
                    }
                }
            };
            remap(orig);
            remap(draft);
            ui.expanded = new Set([...ui.expanded].map((c) => (c === path || c.startsWith(path + "/") ? res.path + c.slice(path.length) : c)));
            if (ui.selCat && (ui.selCat === path || ui.selCat.startsWith(path + "/"))) ui.selCat = res.path + ui.selCat.slice(path.length);
            ui.selProp = null;
            await reload();
        } catch (e) {
            toast(`Rename failed: ${e.message}`, "error");
        }
    }

    async function deleteCategory(path) {
        if (!guardEditor()) return;
        if (!window.confirm(`Move the category '${catTitle(path)}' with everything in it to the trash folder?\n(Recipe lines that use it keep working from their saved copy.)`)) return;
        try {
            await postJSON("/mykee/ugen/category/delete", { path });
            if (ui.selCat === path || ui.selCat?.startsWith(path + "/")) {
                ui.selCat = null;
                ui.selProp = null;
            }
            await reload();
        } catch (e) {
            toast(`Delete failed: ${e.message}`, "error");
        }
    }

    async function newProperty() {
        if (!guardEditor()) return;
        if (!ui.selCat) {
            toast("Select a category on the left first.", "info");
            return;
        }
        const name = askName(`New property (list) in '${catTitle(ui.selCat)}' - file name:`);
        if (!name) return;
        try {
            const res = await postJSON("/mykee/ugen/property/save", {
                category: ui.selCat, file: name, data: { version: 1, label: name, type: "multi", max: 0, order: 100, scope: "", items: [] },
            });
            ui.expanded.add(ui.selCat);
            ui.selProp = propKey(res.property.category, res.property.file);
            await reload();
            ui.view = "edit";
            renderAll();
        } catch (e) {
            toast(`Could not create the property: ${e.message}`, "error");
        }
    }

    async function duplicateProperty(prop) {
        if (!guardEditor()) return;
        const name = askName("File name of the copy:", `${prop.file} copy`.slice(0, MAX_NAME));
        if (!name) return;
        try {
            const res = await postJSON("/mykee/ugen/property/save", {
                category: prop.category, file: name,
                data: { version: 1, label: `${prop.label} copy`.slice(0, MAX_LABEL), type: prop.type, max: prop.max, order: prop.order, scope: prop.scope, items: prop.items },
            });
            ui.dirty = false;
            ui.edit = null;
            ui.selProp = propKey(res.property.category, res.property.file);
            await reload();
        } catch (e) {
            toast(`Duplicate failed: ${e.message}`, "error");
        }
    }

    async function deleteProperty(prop) {
        if (!window.confirm(`Move '${prop.label}' (${prop.file}.json) to the trash folder?\n(Recipe lines that use it keep working from their saved copy.)`)) return;
        try {
            await postJSON("/mykee/ugen/property/delete", { category: prop.category, file: prop.file });
            const key = propKey(prop.category, prop.file);
            draft.delete(key);
            orig.delete(key);
            ui.dirty = false;
            ui.edit = null;
            ui.selProp = null;
            await reload();
        } catch (e) {
            toast(`Delete failed: ${e.message}`, "error");
        }
    }

    // --- right side -----------------------------------------------------------
    function renderSide() {
        const changes = computeChanges(orig, draft, S(node), lib, target);
        sideList.replaceChildren();
        let add = 0;
        let rem = 0;
        let chg = 0;
        for (const c of changes) {
            const box = el("div", {}, [el("div", { className: "mugen-dim", style: "font-size:12px", text: c.title })]);
            for (const [kind, text] of c.lines) {
                if (kind === "add") add++;
                else if (kind === "rem") rem++;
                else chg++;
                box.append(el("div", { className: `mugen-${kind}`, text: `${kind === "add" ? "+" : kind === "rem" ? "−" : "~"} ${text}` }));
            }
            sideList.append(box);
        }
        if (!changes.length) sideList.append(el("div", { className: "mugen-dim", text: "No changes yet." }));
        sendBtn.textContent = changes.length ? `Send to recipe (+${add} / −${rem} / ~${chg})` : "Send to recipe";
        sendBtn.title = "+ added values, − removed values, ~ changed settings (mode or count)";
    }

    function renderAll() {
        renderTree();
        if (ui.view === "edit") renderEdit();
        else renderBrowse();
        renderSide();
    }

    renderAll();
}

// ---------------------------------------------------------------------------
// Node setup
// ---------------------------------------------------------------------------

function setupNode(node) {
    injectStyle();
    const stateW = findWidget(node, STATE_WIDGET);
    if (!stateW) return;
    stateW.hidden = true;
    stateW.computeSize = () => [0, -4];
    // the prompt gets the recipe without the panel's view state (collapsed
    // groups), so folding a group never invalidates ComfyUI's cache
    stateW.serializeValue = () => {
        const { ui, ...rest } = S(node);
        return JSON.stringify(rest);
    };
    try {
        node._mugen = cleanRecipe(JSON.parse(stateW.value || "{}"));
    } catch {
        node._mugen = cleanRecipe({});
    }
    // system_prompt goes to the very top (above use_llm). This runs before a
    // saved workflow is configured, so widgets_values keep matching.
    const sysW = findWidget(node, "system_prompt");
    if (sysW) node.widgets = [sysW, ...node.widgets.filter((w) => w !== sysW)];
    node.properties = node.properties || {};
    buildPanel(node);
    node.setSize([Math.max(node.size[0], MIN_WIDTH), node.size[1]]);
    node._mugenRender();
    loadLibrary().then(() => node._mugenRender?.()).catch(() => {});
    setTimeout(() => fitHeight(node), 0);
    setTimeout(() => { node._mugenReady = true; }, 800);
}

app.registerExtension({
    name: "Mykee.UniversalGenerator",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            setupNode(this);
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            if (!this._mugenRender) return r;
            this._mugenReady = false;
            try {
                this._mugen = cleanRecipe(JSON.parse(findWidget(this, STATE_WIDGET)?.value || "{}"));
            } catch {
                this._mugen = cleanRecipe({});
            }
            this._mugenRender();
            const settle = () => {
                this._mugenRender?.();
                fitHeight(this);
            };
            setTimeout(settle, 50);
            setTimeout(settle, 400);
            setTimeout(() => { this._mugenReady = true; }, 900);
            return r;
        };

        // A height the user adds by dragging is remembered (room for the
        // system_prompt box); our own content-fitting never counts as that.
        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            const r = onResize?.apply(this, arguments);
            if (this._mugenSizing || !this._mugenReady || !this.computeSize) return r;
            const min = this.computeSize()[1];
            this.properties = this.properties || {};
            this.properties[EXTRA_PROP] = Math.max(0, Math.round(size[1] - min));
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const tags = message?.tags_display?.[0];
            const prompt = message?.prompt_display?.[0];
            if (typeof tags === "string" || typeof prompt === "string") {
                this.properties = this.properties || {};
                this.properties[LAST_PROP] = { tags: str(tags, 200000), prompt: str(prompt, 200000) };
            }
            try {
                const picks = JSON.parse(message?.picks?.[0] || "{}");
                const clean = {};
                if (picks && typeof picks === "object" && !Array.isArray(picks)) {
                    for (const [k, v] of Object.entries(picks)) {
                        if (ID_RE.test(k) && Array.isArray(v)) clean[k] = v.filter((x) => typeof x === "string").map((x) => str(x, MAX_ITEM_TEXT));
                    }
                }
                this._mugenPicks = clean;
            } catch {
                /* ignore */
            }
            this._mugenRender?.();
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this._mugenCloseLibrary?.();
            return onRemoved?.apply(this, arguments);
        };
    },
});

// Exposed for the unit tests only (nothing in ComfyUI reads this).
export const __testing = { cleanRecipe, normalize, insertLibEntry, computeChanges, applyDraft, draftFromRecipe, validName, propKey, targetTab, SCENE };
