/*
 * Mykee Model Template - frontend.
 *
 * The node's seven outputs (model, clip_type, clip_1, clip_2, clip_3, vae, vae_2) are "*"
 * typed. Dragging one onto a combo widget of a loader node (ckpt_name,
 * unet_name, clip_name, vae_name, ...) makes the output take over that
 * widget's type, like a Primitive node does, and the node's panel shows a
 * dropdown with the loader's own file list. The selected file name is what
 * the loader receives.
 *
 * Templates (one JSON file each, see nodes/mykee_model_template.py) are
 * saved from, and loaded into, those dropdowns:
 *
 *   - Save writes the selection of the CONNECTED outputs only.
 *   - Loading applies an entry only when its output is connected AND the
 *     file name is in the connected widget's list. Anything else is left
 *     untouched - e.g. a template with a model and a VAE, loaded onto a node
 *     with no CLIP connected, selects the model and the VAE and nothing else.
 *     A toast tells what was applied and what was skipped (and why).
 *
 * Template Notes: a free-text / Markdown field at the bottom of the node. It
 * is shown rendered when the text looks like Markdown (otherwise as plain
 * text) and turns into an auto-growing textarea for editing. The notes are
 * stored in the template JSON ("notes") and in the workflow, never sent to
 * the backend at queue time (editing them doesn't invalidate the cache).
 *
 * The selection lives in the hidden "model_config" widget (JSON), the same
 * way the XYZ Plot node keeps its settings. At queue time the widget's
 * serializeValue sends only the connected outputs' values to the backend.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "MykeeModelTemplate";
const CONFIG_WIDGET = "model_config";
const SLOTS = ["model", "clip_type", "clip_1", "clip_2", "clip_3", "vae", "vae_2"]; // output slots 0..6
const SLOT_TITLES = { model: "Model", clip_type: "CLIP type", clip_1: "CLIP 1", clip_2: "CLIP 2", clip_3: "CLIP 3", vae: "VAE", vae_2: "VAE 2" };
const SLOT_HINTS = {
    model: "drag the model output onto a checkpoint / diffusion model list",
    clip_type: "drag the clip_type output onto the type list of a CLIP loader",
    clip_1: "drag the clip_1 output onto a CLIP list",
    clip_2: "drag the clip_2 output onto a second CLIP list (dual / triple loaders)",
    clip_3: "drag the clip_3 output onto a third CLIP list (triple loaders)",
    vae: "drag the vae output onto a VAE list",
    vae_2: "drag the vae_2 output onto a second VAE list (models that need two VAEs)",
};
const BADGE_COLORS = { model: "#7fc8ff", clip_type: "#6fd6c4", clip_1: "#9be08f", clip_2: "#c3e88d", clip_3: "#dcf0a8", vae: "#ffcf73", vae_2: "#ffb347" };
const MIN_WIDTH = 340;
const PANEL_GAP = 6;
// Panel padding (2px top + 8px bottom) sits inside the measured box too, so it
// is part of this on top of the real gap below the last row.
const PANEL_BOTTOM_PADDING = 32;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function toast(detail, severity = "warn", life = 6000) {
    if (app.extensionManager?.toast?.add) {
        app.extensionManager.toast.add({ severity, summary: "Mykee Model Template", detail, life });
    } else {
        console.warn(`[Mykee Model Template] ${detail}`);
    }
}

async function fetchJSON(url, options) {
    const res = await api.fetchApi(url, options);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    return data;
}

function findWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function getLink(graph, id) {
    if (!graph || id == null) return undefined;
    return graph.getLink?.(id) ?? graph.links?.get?.(id) ?? graph.links?.[id];
}

function readState(node) {
    const widget = findWidget(node, CONFIG_WIDGET);
    let state = {};
    try {
        state = JSON.parse(widget?.value || "{}") || {};
    } catch {
        state = {};
    }
    if (!state.values || typeof state.values !== "object") state.values = {};
    if (typeof state.notes !== "string") state.notes = "";
    return state;
}

// Writes the selection. The notes are always taken from the stored state, so a
// slot panel holding an older snapshot can't overwrite freshly typed notes.
function writeState(node, state) {
    const widget = findWidget(node, CONFIG_WIDGET);
    if (widget) widget.value = JSON.stringify({ version: 1, values: state.values, notes: readState(node).notes });
}

function writeNotes(node, notes) {
    const widget = findWidget(node, CONFIG_WIDGET);
    if (widget) widget.value = JSON.stringify({ version: 1, values: readState(node).values, notes });
}

function comboOptions(widget, node) {
    const values = widget?.options?.values;
    const list = typeof values === "function" ? values(widget, node) : values;
    return Array.isArray(list) ? list.map(String) : [];
}

// Windows lists "sub\model.safetensors", Linux "sub/model.safetensors" - a
// template saved on one and loaded on the other must still match.
const normName = (s) => String(s).replace(/\\/g, "/");

// Finds `wanted` in `options`: exact (ignoring the slash style) first, then
// case-insensitive. Returns the list's own spelling, or null.
function matchOption(options, wanted) {
    const w = normName(wanted);
    let hit = options.find((o) => normName(o) === w);
    if (hit !== undefined) return hit;
    const lw = w.toLowerCase();
    hit = options.find((o) => normName(o).toLowerCase() === lw);
    return hit !== undefined ? hit : null;
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
    if (type !== "COMBO" || !widget) return { supported: false, type, name: input.name };

    const param = input.label || input.localized_name || input.name;
    return {
        supported: true,
        type,
        param,
        nodeTitle: targetNode.title || targetNode.type,
        label: `${param} (${targetNode.title || targetNode.type})`,
        options: comboOptions(widget, targetNode),
    };
}

// Links leaving an output slot. The graph's own link table is the source of
// truth - output.links[] can be stale right after a disconnect.
function outputLinkList(node, slot, ignoreLinkId) {
    const graph = node.graph;
    if (!graph) return [];
    const table = graph.links;
    let links;
    if (table instanceof Map) links = [...table.values()];
    else if (table && typeof table === "object") links = Object.values(table);
    else links = (node.outputs?.[slot]?.links ?? []).map((id) => getLink(graph, id));
    return links
        .filter((l) => l && String(l.origin_id) === String(node.id) && l.origin_slot === slot && l.id !== ignoreLinkId)
        .sort((a, b) => a.id - b.id);
}

function slotTargets(node, slot, ignoreLinkId) {
    const result = [];
    for (const link of outputLinkList(node, slot, ignoreLinkId)) {
        const targetNode = node.graph?.getNodeById?.(link.target_id);
        const info = describeInput(targetNode, link.target_slot);
        if (info) result.push({ ...info, linkId: link.id, targetNode, targetSlot: link.target_slot });
    }
    return result;
}

// The first supported target defines the list shown for the slot.
function slotTarget(node, slot, ignoreLinkId) {
    const targets = slotTargets(node, slot, ignoreLinkId).filter((t) => t.supported);
    if (!targets.length) return null;
    const main = targets[0];
    const extra = targets.length - 1;
    return extra > 0 ? { ...main, label: `${main.label} +${extra}` } : main;
}

// Connected outputs only, with a value (what is saved / sent to the backend).
function resolveValues(node) {
    const state = readState(node);
    const values = {};
    SLOTS.forEach((name, slot) => {
        if (!slotTarget(node, slot)) return;
        const v = state.values[name];
        if (typeof v === "string" && v) values[name] = v;
    });
    return values;
}

// ---------------------------------------------------------------------------
// Minimal, safe Markdown renderer (no dependencies, offline). Everything is
// HTML-escaped first; only a fixed set of tags is produced and links are
// limited to http(s) / mailto.
// ---------------------------------------------------------------------------

const escapeHtml = (s) =>
    String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");

// Heuristic: does this text use Markdown syntax worth rendering?
function looksLikeMarkdown(text) {
    return [
        /^ {0,3}#{1,6}\s+\S/m, // headings
        /^ {0,3}([-*+]|\d+[.)])\s+\S/m, // lists
        /^ {0,3}```/m, // code fences
        /^ {0,3}>\s?\S/m, // quotes
        /^ {0,3}([-*_])( ?\1){2,}\s*$/m, // rules
        /^\s*\|?.+\|.+\n\s*\|?\s*:?-{2,}:?\s*\|/m, // tables
        /\*\*[^*\n]+\*\*|__[^_\n]+__|~~[^~\n]+~~/, // bold / strike
        /`[^`\n]+`/, // inline code
        /\[[^\]\n]+\]\((https?:\/\/|mailto:)[^)\s]+\)/, // links
    ].some((re) => re.test(text));
}

function renderInline(text) {
    const stash = [];
    const hold = (html) => `\u0000${stash.push(html) - 1}\u0000`;
    let s = String(text);
    // code spans first - their content is literal
    s = s.replace(/(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/g, (_, __, code) => hold(`<code>${escapeHtml(code.trim())}</code>`));
    // backslash escapes
    s = s.replace(/\\([\\`*_{}\[\]()#+\-.!|>~<])/g, (_, ch) => hold(escapeHtml(ch)));
    s = escapeHtml(s);
    s = s.replace(/\[([^\]]+)\]\(((?:https?:\/\/|mailto:)[^\s)]+)\)/g, (_, label, url) =>
        hold(`<a href="${url}" target="_blank" rel="noopener noreferrer">${label}</a>`)
    );
    s = s.replace(/&lt;((?:https?:\/\/|mailto:)[^\s&]+)&gt;/g, (_, url) =>
        hold(`<a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`)
    );
    s = s.replace(/(^|[\s(])(https?:\/\/[^\s<]+[^\s<.,;:!?)])/g, (_, pre, url) =>
        pre + hold(`<a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`)
    );
    s = s.replace(/\*\*\*(?=\S)([\s\S]*?\S)\*\*\*/g, "<strong><em>$1</em></strong>");
    s = s.replace(/\*\*(?=\S)([\s\S]*?\S)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/(^|[^\w])__(?=\S)([\s\S]*?\S)__(?!\w)/g, "$1<strong>$2</strong>");
    s = s.replace(/\*(?=\S)([^*]*?\S)\*/g, "<em>$1</em>");
    s = s.replace(/(^|[^\w])_(?=\S)([^_]*?\S)_(?!\w)/g, "$1<em>$2</em>");
    s = s.replace(/~~(?=\S)([\s\S]*?\S)~~/g, "<del>$1</del>");
    s = s.replace(/(?: {2,}|\\)\n/g, "<br>");
    // restore (stash entries may contain further placeholders, so loop)
    for (let i = 0; i < 3 && s.includes("\u0000"); i++) s = s.replace(/\u0000(\d+)\u0000/g, (_, n) => stash[+n]);
    return s;
}

function splitTableRow(line) {
    let l = line.trim();
    if (l.startsWith("|")) l = l.slice(1);
    if (l.endsWith("|") && !l.endsWith("\\|")) l = l.slice(0, -1);
    return l.split(/(?<!\\)\|/).map((c) => c.trim().replace(/\\\|/g, "|"));
}

const RE_LIST_ITEM = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;
const RE_TABLE_SEP = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/;
const RE_HR = /^ {0,3}([-*_])( ?\1){2,}\s*$/;
const RE_FENCE = /^ {0,3}(```+|~~~+)\s*([\w+-]*)/;

function buildList(items, start, baseIndent) {
    const ordered = /\d/.test(items[start].marker);
    let html = ordered ? "<ol>" : "<ul>";
    let i = start;
    while (i < items.length && items[i].indent >= baseIndent) {
        if (i > start && items[i].indent === baseIndent && /\d/.test(items[i].marker) !== ordered) break;
        let text = items[i].text;
        let cls = "";
        const task = /^\[([ xX])\]\s+(.*)$/.exec(text);
        if (task) {
            text = `${task[1] === " " ? "☐" : "☑"} ${task[2]}`;
            cls = ' class="task"';
        }
        html += `<li${cls}>${renderInline(text)}`;
        i++;
        if (i < items.length && items[i].indent > baseIndent) {
            const [sub, next] = buildList(items, i, items[i].indent);
            html += sub;
            i = next;
        }
        html += "</li>";
    }
    html += ordered ? "</ol>" : "</ul>";
    return [html, i];
}

function renderMarkdown(src) {
    const lines = String(src).replace(/\r\n?/g, "\n").replace(/\t/g, "    ").split("\n");
    const out = [];
    let i = 0;
    const isBlockStart = (line, next) =>
        !line.trim() ||
        RE_FENCE.test(line) ||
        /^ {0,3}#{1,6}\s/.test(line) ||
        RE_HR.test(line) ||
        /^ {0,3}>/.test(line) ||
        RE_LIST_ITEM.test(line) ||
        (line.includes("|") && next !== undefined && RE_TABLE_SEP.test(next) && next.includes("-"));

    while (i < lines.length) {
        const line = lines[i];
        if (!line.trim()) {
            i++;
            continue;
        }
        const fence = RE_FENCE.exec(line);
        if (fence) {
            const marker = fence[1][0];
            const code = [];
            i++;
            while (i < lines.length && !new RegExp(`^ {0,3}${marker}{${fence[1].length},}\\s*$`).test(lines[i])) code.push(lines[i++]);
            i++; // closing fence
            out.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
            continue;
        }
        const heading = /^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$/.exec(line);
        if (heading) {
            out.push(`<h${heading[1].length}>${renderInline(heading[2])}</h${heading[1].length}>`);
            i++;
            continue;
        }
        if (RE_HR.test(line)) {
            out.push("<hr>");
            i++;
            continue;
        }
        if (/^ {0,3}>/.test(line)) {
            const quote = [];
            while (i < lines.length && /^ {0,3}>/.test(lines[i])) quote.push(lines[i++].replace(/^ {0,3}> ?/, ""));
            out.push(`<blockquote>${renderMarkdown(quote.join("\n"))}</blockquote>`);
            continue;
        }
        if (line.includes("|") && i + 1 < lines.length && RE_TABLE_SEP.test(lines[i + 1]) && lines[i + 1].includes("-")) {
            const head = splitTableRow(line);
            const aligns = splitTableRow(lines[i + 1]).map((c) =>
                /^:-+:$/.test(c) ? "center" : /^-+:$/.test(c) ? "right" : /^:-+$/.test(c) ? "left" : ""
            );
            const cell = (tag, text, n) =>
                `<${tag}${aligns[n] ? ` style="text-align:${aligns[n]}"` : ""}>${renderInline(text)}</${tag}>`;
            let html = "<table><thead><tr>" + head.map((c, n) => cell("th", c, n)).join("") + "</tr></thead><tbody>";
            i += 2;
            while (i < lines.length && lines[i].trim() && lines[i].includes("|")) {
                html += "<tr>" + splitTableRow(lines[i++]).map((c, n) => cell("td", c, n)).join("") + "</tr>";
            }
            out.push(html + "</tbody></table>");
            continue;
        }
        if (RE_LIST_ITEM.test(line)) {
            const items = [];
            while (i < lines.length) {
                const m = RE_LIST_ITEM.exec(lines[i]);
                if (m) {
                    items.push({ indent: m[1].length, marker: m[2], text: m[3] });
                    i++;
                } else if (lines[i].trim() && /^\s{2,}\S/.test(lines[i]) && items.length && !RE_FENCE.test(lines[i])) {
                    items[items.length - 1].text += " " + lines[i].trim(); // wrapped item text
                    i++;
                } else if (!lines[i].trim() && i + 1 < lines.length && RE_LIST_ITEM.test(lines[i + 1])) {
                    i++; // blank line between items: same list
                } else break;
            }
            let k = 0;
            while (k < items.length) {
                const [html, next] = buildList(items, k, items[k].indent);
                out.push(html);
                k = next;
            }
            continue;
        }
        // paragraph
        const para = [line];
        i++;
        while (i < lines.length && !isBlockStart(lines[i], lines[i + 1])) para.push(lines[i++]);
        out.push(`<p>${renderInline(para.join("\n"))}</p>`);
    }
    return out.join("\n");
}

// ---------------------------------------------------------------------------
// Panel UI
// ---------------------------------------------------------------------------

const PANEL_CSS = `
.mykee-mt { display:flex; flex-direction:column; gap:6px; width:100%; height:100%;
  overflow-y:auto; box-sizing:border-box; padding:2px 4px 8px; font:12px sans-serif;
  color:var(--input-text, #ddd); }
.mykee-mt-slot { border:1px solid var(--border-color, #444); border-radius:6px; padding:5px 6px;
  background:var(--comfy-input-bg, #222); display:flex; flex-direction:column; gap:4px; }
.mykee-mt-head { display:flex; align-items:center; gap:6px; }
.mykee-mt-badge { font-weight:bold; border-radius:4px; padding:0 6px; color:#111; white-space:nowrap; }
.mykee-mt-title { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.mykee-mt-muted { opacity:.6; font-style:italic; }
.mykee-mt select { width:100%; box-sizing:border-box; background:var(--comfy-menu-bg, #111);
  color:inherit; border:1px solid var(--border-color, #444); border-radius:4px; padding:3px 5px; font:12px sans-serif; }
.mykee-mt select.missing { border-color:#f77; }
.mykee-mt-notes { display:flex; flex-direction:column; gap:4px; width:100%; height:100%;
  box-sizing:border-box; padding:2px 4px 6px; font:12px sans-serif; color:var(--input-text, #ddd); }
.mykee-mt-notes-head { display:flex; align-items:center; gap:6px; flex:0 0 auto; }
.mykee-mt-notes-title { font-weight:bold; flex:1; }
.mykee-mt-notes-tag { opacity:.6; font-size:10px; border:1px solid var(--border-color, #555); border-radius:3px; padding:0 4px; }
.mykee-mt-notes textarea { flex:0 0 auto; width:100%; box-sizing:border-box; resize:none; overflow:hidden; min-height:60px;
  background:var(--comfy-input-bg, #222); color:inherit; border:1px solid var(--border-color, #444);
  border-radius:4px; padding:5px 6px; font:12px/1.4 sans-serif; }
.mykee-mt-notes-view { flex:0 0 auto; box-sizing:border-box; border:1px solid var(--border-color, #444); border-radius:4px;
  background:var(--comfy-input-bg, #222); padding:6px 8px; line-height:1.4; overflow-wrap:anywhere;
  overflow-y:auto; cursor:text; min-height:24px; }
.mykee-mt-notes-view.plain { white-space:pre-wrap; }
.mykee-mt-notes-view > :first-child { margin-top:0; }
.mykee-mt-notes-view > :last-child { margin-bottom:0; }
.mykee-mt-notes-view h1, .mykee-mt-notes-view h2, .mykee-mt-notes-view h3,
.mykee-mt-notes-view h4, .mykee-mt-notes-view h5, .mykee-mt-notes-view h6 { margin:.7em 0 .35em; line-height:1.25; }
.mykee-mt-notes-view h1 { font-size:1.5em; border-bottom:1px solid var(--border-color, #444); padding-bottom:.2em; }
.mykee-mt-notes-view h2 { font-size:1.3em; border-bottom:1px solid var(--border-color, #444); padding-bottom:.15em; }
.mykee-mt-notes-view h3 { font-size:1.15em; }
.mykee-mt-notes-view h4, .mykee-mt-notes-view h5, .mykee-mt-notes-view h6 { font-size:1em; }
.mykee-mt-notes-view p, .mykee-mt-notes-view ul, .mykee-mt-notes-view ol,
.mykee-mt-notes-view blockquote, .mykee-mt-notes-view pre, .mykee-mt-notes-view table { margin:.45em 0; }
.mykee-mt-notes-view ul, .mykee-mt-notes-view ol { padding-left:20px; }
.mykee-mt-notes-view ul ul, .mykee-mt-notes-view ol ol, .mykee-mt-notes-view ul ol, .mykee-mt-notes-view ol ul { margin:.1em 0; }
.mykee-mt-notes-view li.task { list-style:none; margin-left:-18px; }
.mykee-mt-notes-view code { background:rgba(127,127,127,.25); border-radius:3px; padding:0 4px;
  font:11px/1.4 Consolas, Menlo, monospace; }
.mykee-mt-notes-view pre { background:rgba(0,0,0,.35); border-radius:4px; padding:6px 8px; overflow-x:auto; }
.mykee-mt-notes-view pre code { background:none; padding:0; white-space:pre; }
.mykee-mt-notes-view blockquote { border-left:3px solid var(--border-color, #666); margin-left:0; padding:0 0 0 8px; opacity:.85; }
.mykee-mt-notes-view hr { border:0; border-top:1px solid var(--border-color, #555); margin:.7em 0; }
.mykee-mt-notes-view table { border-collapse:collapse; }
.mykee-mt-notes-view th, .mykee-mt-notes-view td { border:1px solid var(--border-color, #555); padding:2px 6px; }
.mykee-mt-notes-view th { background:rgba(127,127,127,.2); }
.mykee-mt-notes-view a { color:#6cb6ff; }
.mykee-mt-notes-empty { opacity:.5; font-style:italic; }
.mykee-mt button, .mykee-mt-notes button { background:var(--comfy-menu-bg, #333); color:inherit; border:1px solid var(--border-color, #555);
  border-radius:4px; padding:1px 7px; cursor:pointer; font-size:11px; }
`;

function ensureCss() {
    if (document.getElementById("mykee-mt-css")) return;
    const style = document.createElement("style");
    style.id = "mykee-mt-css";
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

function renderSlot(node, name, slot, state, onChange) {
    const target = slotTarget(node, slot);
    const box = el("div", { className: "mykee-mt-slot" });

    const badge = el("span", { className: "mykee-mt-badge", textContent: SLOT_TITLES[name] });
    badge.style.background = BADGE_COLORS[name];
    const title = el("span", {
        className: "mykee-mt-title" + (target ? "" : " mykee-mt-muted"),
        textContent: target ? target.label : SLOT_HINTS[name],
        title: target ? target.label : SLOT_HINTS[name],
    });
    const refresh = el("button", {
        textContent: "↻",
        title: "Re-read the connected list (e.g. after a model list refresh)",
    });
    refresh.onclick = () => node.mykeeMtRender?.();
    box.append(el("div", { className: "mykee-mt-head" }, [badge, title, target ? refresh : null]));
    if (!target) return box;

    const options = target.options;
    let current = state.values[name];
    if ((typeof current !== "string" || !current) && options.length) {
        // Nothing chosen yet: take the loader's own first entry, so the
        // output is never empty.
        current = options[0];
        state.values[name] = current;
        writeState(node, state);
    }
    const select = isolate(el("select", {}));
    const listed = options.includes(current);
    if (current && !listed) {
        // The remembered file isn't in the list (any more) - show that
        // instead of silently switching to something else.
        select.append(el("option", { value: current, textContent: `⚠ ${current} (not in list)` }));
        select.classList.add("missing");
    }
    for (const option of options) select.append(el("option", { value: option, textContent: option }));
    select.value = current;
    select.title = current;
    select.onchange = () => {
        state.values[name] = select.value;
        select.title = select.value;
        select.classList.toggle("missing", !options.includes(select.value));
        onChange();
    };
    box.append(select);
    return box;
}

function estimateHeight(node) {
    let height = 0;
    SLOTS.forEach((_, slot) => {
        height += PANEL_GAP + (slotTarget(node, slot) ? 62 : 38);
    });
    return height + PANEL_BOTTOM_PADDING;
}

// Fits the node's height to its content; the width stays the user's.
function fitHeight(node, shrink = true) {
    const size = node.computeSize?.();
    if (!size || !Number.isFinite(size[1])) return;
    const height = shrink ? size[1] : Math.max(node.size[1], size[1]);
    node.setSize([Math.max(node.size[0], size[0]), height]);
    node.graph?.setDirtyCanvas?.(true, true);
}

// Output slot follows the connected widget's type, like a Primitive node.
function updateOutputSlot(node, slot, ignoreLinkId) {
    const output = node.outputs?.[slot];
    if (!output) return;
    const target = slotTarget(node, slot, ignoreLinkId);
    output.type = target ? target.type : "*";
    output.label = target ? `${SLOTS[slot]}: ${target.param}` : SLOTS[slot];
}

function checkNewLink(node, slot, linkInfo) {
    const targetNode = node.graph?.getNodeById?.(linkInfo.target_id);
    const info = describeInput(targetNode, linkInfo.target_slot);
    const first = slotTargets(node, slot).find((t) => t.supported && t.linkId !== linkInfo.id);

    let reason = null;
    if (!info?.supported) {
        reason = `"${info?.name ?? "this input"}" can't be used - only combo (file list) widgets can.`;
    } else if (first && first.options.join("\n") !== info.options.join("\n")) {
        reason = `The ${SLOTS[slot]} output is already connected to a list with different entries; it can't also drive this one.`;
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
// Template load: only into connected outputs, only entries that are in the list
// ---------------------------------------------------------------------------

function applyTemplate(node, templateValues, templateNotes) {
    const state = readState(node);
    const applied = [];
    const notInList = [];
    const notConnected = [];
    SLOTS.forEach((name, slot) => {
        const wanted = templateValues?.[name];
        if (typeof wanted !== "string" || !wanted) return; // not in the template: untouched
        const target = slotTarget(node, slot);
        if (!target) {
            notConnected.push(name);
            return;
        }
        const hit = matchOption(target.options, wanted);
        if (hit === null) {
            notInList.push(`${name}: ${wanted}`);
            return;
        }
        state.values[name] = hit;
        applied.push(name);
    });
    writeState(node, state);
    // Notes are not tied to a connection or a list: loading a template always
    // replaces them - with its notes, or with an empty field if it has none.
    if (typeof templateNotes === "string") {
        writeNotes(node, templateNotes);
        if (templateNotes.trim()) applied.push("notes");
    }
    node.mykeeMtRender?.();
    node.mykeeMtNotesSync?.(true);
    return { applied, notInList, notConnected };
}

function reportApply(name, { applied, notInList, notConnected }) {
    const parts = [];
    parts.push(applied.length ? `applied: ${applied.join(", ")}` : "nothing applied");
    if (notInList.length) parts.push(`not in the list (left unchanged): ${notInList.join("; ")}`);
    if (notConnected.length) parts.push(`output not connected (skipped): ${notConnected.join(", ")}`);
    const severity = notInList.length ? "warn" : applied.length ? "success" : "info";
    toast(`'${name}' - ${parts.join(" | ")}`, severity, notInList.length ? 9000 : 4500);
}

// ---------------------------------------------------------------------------
// Button cursor (plain litegraph buttons only get a crosshair otherwise)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Template Notes (bottom of the node): plain text or Markdown
// ---------------------------------------------------------------------------

const NOTES_MIN_HEIGHT = 96;
const NOTES_MAX_BODY = 1200; // beyond this the text scrolls inside the box
const NOTES_CHROME_FALLBACK = 22 + 4 + 8; // header row + gap + padding (used while hidden)
const NOTES_SAFETY = 4; // spare pixels so rounding never produces a scrollbar

function setupNotes(node) {
    const head = el("div", { className: "mykee-mt-notes-head" });
    const title = el("span", { className: "mykee-mt-notes-title", textContent: "Template Notes" });
    const tag = el("span", { className: "mykee-mt-notes-tag" });
    const toggle = el("button", { textContent: "✏ Edit", title: "Switch between editing and the formatted view" });
    head.append(title, tag, toggle);

    const area = isolate(
        el("textarea", { placeholder: "Notes for this template - plain text or Markdown (headings, lists, **bold**, `code`, tables, links)...", spellcheck: false })
    );
    const view = el("div", { className: "mykee-mt-notes-view", title: "Double-click to edit" });
    const box = el("div", { className: "mykee-mt-notes" }, [head, area, view]);

    let notesHeight = NOTES_MIN_HEIGHT;
    let editing = false;
    let lastWidth = 0;

    const domWidget = node.addDOMWidget("template_notes", "MYKEE_TEMPLATE_NOTES", box, {
        serialize: false,
        getMinHeight: () => notesHeight,
    });
    domWidget.serialize = false;

    const text = () => readState(node).notes;

    // Fits the widget (and node) to the content. grow-only while typing so a
    // node the user made taller isn't shrunk under their hands.
    const measure = (shrink) => {
        let body = 0;
        if (editing) {
            area.style.height = "auto";
            body = Math.max(area.scrollHeight + 2, 60);
            area.style.height = `${body}px`;
        } else {
            view.style.maxHeight = "";
            body = view.scrollHeight + 2;
            if (body > NOTES_MAX_BODY) {
                body = NOTES_MAX_BODY;
                view.style.maxHeight = `${NOTES_MAX_BODY}px`;
            }
        }
        if (body <= 2 && !editing) return; // hidden (display:none) - keep the last height
        const chrome = head.offsetHeight ? head.offsetHeight + 4 + 8 : NOTES_CHROME_FALLBACK;
        const next = Math.max(NOTES_MIN_HEIGHT, body + chrome + NOTES_SAFETY);
        const delta = next - notesHeight;
        if (delta === 0 && !shrink) return;
        notesHeight = next;
        if (shrink) {
            fitHeight(node, true);
        } else {
            // Typing: move the node's height by exactly what the notes grew or
            // shrank, so no slack ends up in (and pushes down) the widgets above.
            const size = node.computeSize?.();
            const height = Math.max(size?.[1] ?? 0, node.size[1] + delta);
            node.setSize([node.size[0], height]);
            node.graph?.setDirtyCanvas?.(true, true);
        }
    };

    const showView = () => {
        const value = text();
        const isMd = looksLikeMarkdown(value);
        tag.textContent = isMd ? "Markdown" : "Plain text";
        if (!value.trim()) {
            view.className = "mykee-mt-notes-view mykee-mt-notes-empty";
            view.textContent = "No notes - double-click or press Edit to write some (plain text or Markdown).";
        } else if (isMd) {
            view.className = "mykee-mt-notes-view";
            view.innerHTML = renderMarkdown(value);
        } else {
            view.className = "mykee-mt-notes-view plain";
            view.textContent = value;
        }
    };

    const setEditing = (on, shrink = true, focus = true) => {
        editing = on;
        area.style.display = on ? "" : "none";
        view.style.display = on ? "none" : "";
        toggle.textContent = on ? "👁 View" : "✏ Edit";
        if (on) {
            area.value = text();
            tag.textContent = looksLikeMarkdown(area.value) ? "Markdown" : "Plain text";
        } else {
            showView();
        }
        measure(shrink);
        if (on && focus) {
            area.focus();
            area.setSelectionRange(area.value.length, area.value.length);
        }
    };

    area.addEventListener("input", () => {
        writeNotes(node, area.value);
        tag.textContent = looksLikeMarkdown(area.value) ? "Markdown" : "Plain text";
        measure(false);
        node.graph?.setDirtyCanvas?.(true, false);
    });
    area.addEventListener("blur", () => {
        if (editing && text().trim()) setEditing(false);
    });
    area.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && text().trim()) area.blur();
    });
    view.addEventListener("dblclick", () => setEditing(true));
    // Links open normally; any other click on the view is a plain selection.
    toggle.onclick = () => setEditing(!editing);

    // Wrapping changes with the node width, and so does the height.
    if (typeof ResizeObserver !== "undefined") {
        new ResizeObserver(() => {
            const w = box.clientWidth;
            if (!w || w === lastWidth) return;
            lastWidth = w;
            requestAnimationFrame(() => measure(false));
        }).observe(box);
    }

    // Pulls the notes from the stored state into the UI (after a workflow
    // load, a template load, ...). An empty note opens in edit mode.
    node.mykeeMtNotesSync = (shrink = true) => {
        if (editing && document.activeElement === area) return; // never fight the user's typing
        setEditing(!text().trim(), shrink, false); // programmatic: don't steal focus
    };

    setEditing(true, false);
    node.mykeeMtNotesSync(false);
}

// ---------------------------------------------------------------------------
// Node setup
// ---------------------------------------------------------------------------

function setupNode(node) {
    ensureCss();

    const nameWidget = findWidget(node, "template_name");
    const pathWidget = findWidget(node, "custom_path");
    const configWidget = findWidget(node, CONFIG_WIDGET);
    if (!nameWidget || !configWidget) return;

    // Hide the raw JSON widget (the backend spec already asks for that on
    // newer frontends; this covers older ones).
    configWidget.hidden = true;
    configWidget.computeSize = () => [0, -4];
    // The prompt gets only the connected outputs' values, the workflow keeps
    // the whole editor state.
    configWidget.serializeValue = () => JSON.stringify({ version: 1, values: resolveValues(node) });

    const currentPath = () => (pathWidget?.value || "").trim();

    // --- template picker + buttons -------------------------------------
    let existingNames = [];
    const loadByName = async (name) => {
        if (!name) return;
        try {
            const data = await fetchJSON(
                `/mykee/model_templates/load?path=${encodeURIComponent(currentPath())}&name=${encodeURIComponent(name)}`
            );
            if (data.name && data.name !== nameWidget.value) nameWidget.value = data.name;
            reportApply(data.name || name, applyTemplate(node, data.values, data.notes));
            node.setDirtyCanvas(true, true);
        } catch (e) {
            window.alert("Load failed: " + e.message);
        }
    };

    const listWidget = node.addWidget(
        "combo",
        "Existing templates",
        "",
        (value) => {
            if (!value) return;
            nameWidget.value = value;
            // Picking one loads it straight away - same as the Prompt Template node.
            loadByName(value);
        },
        { values: existingNames }
    );
    // Re-derived from disk on every reload, so a stale pick shouldn't persist.
    listWidget.serialize = false;

    const refreshList = async () => {
        try {
            const data = await fetchJSON(`/mykee/model_templates/list?path=${encodeURIComponent(currentPath())}`);
            existingNames = data.templates || [];
            if (listWidget.options) listWidget.options.values = existingNames;
        } catch (e) {
            console.warn("Mykee Model Template: couldn't fetch template list", e);
        }
        return existingNames;
    };

    const newBtn = node.addWidget("button", "🆕 New", null, () => {
        // Starts a fresh template name; the current selections stay, so a
        // similar setup can be saved under a new name quickly.
        nameWidget.value = "";
        node.setDirtyCanvas(true, true);
    });
    newBtn.serialize = false;

    const saveBtn = node.addWidget("button", "💾 Save", null, async () => {
        const name = (nameWidget.value || "").trim();
        if (!name) {
            window.alert("Enter a template name to save.");
            return;
        }
        const values = resolveValues(node);
        const notes = readState(node).notes;
        if (!Object.keys(values).length && !notes.trim()) {
            window.alert("Nothing to save - connect at least one output to a list or write some notes first.");
            return;
        }
        try {
            const data = await fetchJSON("/mykee/model_templates/save", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ path: currentPath(), name, values, notes }),
            });
            if (data.name && data.name !== nameWidget.value) {
                nameWidget.value = data.name;
                node.setDirtyCanvas(true, true);
            }
            await refreshList();
            const savedParts = [...Object.keys(values), ...(notes.trim() ? ["notes"] : [])];
            toast(`'${data.name}.json' saved (${savedParts.join(", ")}).`, "success", 4000);
        } catch (e) {
            window.alert("Save failed: " + e.message);
        }
    });
    saveBtn.serialize = false;

    const reloadBtn = node.addWidget("button", "🔄 Reload", null, async () => {
        await refreshList();
        node.mykeeMtRender?.(); // also re-reads the connected lists
    });
    reloadBtn.serialize = false;

    // --- slot panel ------------------------------------------------------
    const panel = el("div", { className: "mykee-mt" });
    let contentHeight = estimateHeight(node);
    const domWidget = node.addDOMWidget("model_panel", "MYKEE_MODEL_PANEL", panel, {
        serialize: false,
        getMinHeight: () => contentHeight,
    });
    domWidget.serialize = false;

    const measure = () => {
        const kids = [...panel.children];
        const height = kids.reduce((sum, kid) => sum + kid.offsetHeight, 0);
        if (height > 0) contentHeight = height + PANEL_GAP * (kids.length - 1) + PANEL_BOTTOM_PADDING;
    };

    // shrink = false only grows: used when a saved workflow is loaded, so a
    // node the user made taller keeps its size.
    node.mykeeMtRender = (shrink = true) => {
        contentHeight = estimateHeight(node);
        const state = readState(node);
        panel.replaceChildren();
        const onChange = () => {
            writeState(node, state);
            node.graph?.setDirtyCanvas?.(true, false);
        };
        SLOTS.forEach((name, slot) => panel.append(renderSlot(node, name, slot, state, onChange)));
        fitHeight(node, shrink);
        requestAnimationFrame(() => {
            measure();
            fitHeight(node, shrink);
        });
    };

    setupNotes(node);

    node.setSize([Math.max(node.size[0], MIN_WIDTH), node.size[1]]);
    node.mykeeMtRender();
    refreshList();
    // ComfyUI may still apply its own size after onNodeCreated - fit again.
    setTimeout(() => fitHeight(node, true), 0);
}

app.registerExtension({
    name: "Mykee.ModelTemplate",

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
            // Target nodes may not exist yet while the graph is loading.
            setTimeout(() => {
                for (let slot = 0; slot < SLOTS.length; slot++) updateOutputSlot(this, slot);
                this.mykeeMtRender?.(false);
                this.mykeeMtNotesSync?.(false);
            }, 50);
            return ret;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, slot, connected, linkInfo) {
            const ret = onConnectionsChange?.apply(this, arguments);
            const OUTPUT = window.LiteGraph?.OUTPUT ?? 2;
            if (type !== OUTPUT || slot >= SLOTS.length) return ret;
            if (connected && linkInfo && !checkNewLink(this, slot, linkInfo)) return ret;
            if (!connected) {
                // Reset right away, ignoring the link being removed (it may
                // still be listed while the disconnect is in progress).
                updateOutputSlot(this, slot, linkInfo?.id);
            }
            // Defer: during a connect / disconnect the link list is not final yet.
            setTimeout(() => {
                updateOutputSlot(this, slot);
                this.mykeeMtRender?.();
                this.graph?.setDirtyCanvas?.(true, true);
            }, 0);
            return ret;
        };
    },
});
