import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * Mykee Video Titler - front end.
 *
 * The Python side declares every setting as a normal widget. This extension adds:
 *
 *   - five tabs (Text / Style / Layout / Titles / Output) in the pack's tab bar style;
 *   - the Meme Maker's font picker and color swatches (same routes, only a
 *     font family name is stored);
 *   - a player: the source video is played by the browser itself (ComfyUI's
 *     own /view route serves the file of the loader behind the video input, or
 *     the preview copy the node writes when it runs), with seek bar, frame
 *     steps, play / pause and the cut points (trim_start / trim_end);
 *   - the caption layer over the player, rendered by the same Python code as
 *     the result (POST /mykee/titler/preview). It shows the title of the list
 *     that is on screen at the player's position; inside the Start - End range
 *     of the Titles tab it shows the captions being typed instead, when they
 *     belong to the selected title or no title of the list is there yet;
 *   - the Titles tab: start / end fields (time or frames), buttons that take
 *     the player's position, and the list of titles. The list is kept in time
 *     order, overlaps are refused, and every row can be edited by hand. It
 *     lives in the hidden "titles" widget as JSON, so it is saved with the
 *     workflow and sent to the node as it is.
 *
 * The node's height always follows its content; the width stays the user's.
 * File and title texts are only ever written with textContent / value.
 */

const NODE_NAME = "MykeeVideoTitler";
const MIN_WIDTH = 420;
const TAB_PROP = "mykee_titler_tab";
const UNITS_PROP = "mykee_titler_units";
const INFO_PROP = "mykee_titler_info";
const RUN_PROP = "mykee_titler_run";
const DEFAULT_TAB = "titles";
const TAB_BAR_NAME = "_mykee_titler_tabs";
const TAB_BAR_HEIGHT = 30;
const PLAYER_NAME = "titler_player";
const PANEL_NAME = "titles_panel";
const CONTROLS_HEIGHT = 108; // seek bar + transport + cut row + status line
const STAGE_MIN_HEIGHT = 90;
const STAGE_MAX_HEIGHT = 340;
const DEFAULT_SIZE = [1024, 576];
const DEFAULT_FPS = 25;
const NO_FONT = "(default)";
const PREVIEW_GRAY = "#707070";
const SWATCH_HEIGHT = 40;
const MAX_TITLES = 2000; // keep in sync with nodes/mykee_video_titler.py
const LIST_MAX_HEIGHT = 260;
const VIDEO_EXT_RE = /\.(mp4|m4v|mov|mkv|webm|avi|wmv|flv|mpg|mpeg|ts|mts|m2ts|3gp|ogv)$/i;

const TABS = [
    { id: "text", label: "Text" },
    { id: "style", label: "Style" },
    { id: "layout", label: "Layout" },
    { id: "titles", label: "Titles" },
    { id: "output", label: "Output" },
];

// widget name -> tab
const TAB_OF = {
    top_font_size: "text", top_align: "text", top_lines: "text", top_margin: "text",
    bottom_font_size: "text", bottom_align: "text", bottom_lines: "text", bottom_margin: "text",
    font_picker: "style", reload_fonts: "style", bold: "style", italic: "style",
    uppercase: "style", text_color: "style", text_color_swatch: "style", stroke_color: "style",
    stroke_color_swatch: "style", stroke_width: "style", line_height: "style",
    placement: "layout", padding: "layout", frame_color: "layout", frame_color_swatch: "layout", frame_width: "layout",
    frame_width_mode: "layout", frame_height_mode: "layout", frame_height: "layout",
    [PANEL_NAME]: "titles", text_mode: "titles", subtitle_font_size: "titles", subtitle_position: "titles",
    trim_start: "output", trim_end: "output", filename_tag: "output", save_output: "output", overwrite: "output",
};

const FRAME_KEYS = ["placement", "frame_color", "frame_width", "frame_width_mode", "frame_height_mode", "frame_height"];
const STYLE_KEYS = [
    "top_text", "bottom_text", "font", "top_font_size", "bottom_font_size", "bold", "italic", "uppercase", "text_color",
    "stroke_color", "stroke_width", "line_height", "padding", "top_align", "top_lines", "bottom_align", "bottom_lines",
    "top_margin", "bottom_margin",
];
const COLOR_WIDGETS = ["text_color", "stroke_color", "frame_color"];
const TEXT_WIDGETS = ["top_text", "bottom_text"];
const TEXT_FIELDS = ["top_text", "bottom_text"]; // always visible above the tabs
const OTHER_WATCHED = ["text_mode", "subtitle_font_size", "subtitle_position", "trim_start", "trim_end"];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const findWidget = (node, name) => node.widgets?.find((w) => w.name === name);

function normalizeHex(value, fallback) {
    if (typeof value !== "string") return fallback;
    let s = value.trim();
    if (!s.startsWith("#")) s = "#" + s;
    if (/^#[0-9a-fA-F]{6}$/.test(s)) return s.toUpperCase();
    if (/^#[0-9a-fA-F]{3}$/.test(s)) {
        const [, r, g, b] = s;
        return `#${r}${r}${g}${g}${b}${b}`.toUpperCase();
    }
    return fallback;
}

function el(tag, props = {}, children = []) {
    const element = document.createElement(tag);
    Object.assign(element, props);
    for (const child of children) if (child) element.append(child);
    return element;
}

function css(element, style) {
    Object.assign(element.style, style);
    return element;
}

function toast(severity, detail) {
    app.extensionManager?.toast?.add?.({ severity, summary: "Mykee Video Titler", detail, life: 3500 });
}

const BUTTON_STYLE = {
    cursor: "pointer", font: "12px sans-serif", padding: "3px 8px", borderRadius: "4px", lineHeight: "16px",
    border: "1px solid var(--border-color, #555)", background: "var(--comfy-input-bg, #2b2b2b)",
    color: "var(--input-text, #ddd)", whiteSpace: "nowrap", flex: "0 0 auto",
};
const INPUT_STYLE = {
    font: "12px monospace", padding: "2px 5px", borderRadius: "4px", boxSizing: "border-box", minWidth: "0",
    border: "1px solid var(--border-color, #555)", background: "var(--comfy-input-bg, #222)",
    color: "var(--input-text, #ddd)",
};

function button(label, title, onClick) {
    const b = css(el("button", { type: "button", textContent: label, title }), BUTTON_STYLE);
    b.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        onClick(event);
    });
    return b;
}

let fontsPromise = null;
function loadFonts(refresh = false) {
    if (!fontsPromise || refresh) {
        fontsPromise = api
            .fetchApi(`/mykee/meme/fonts${refresh ? "?refresh=1" : ""}`)
            .then((res) => (res.ok ? res.json() : { fonts: [], default: "" }))
            .catch(() => ({ fonts: [], default: "" }));
    }
    return fontsPromise;
}

// ---- time <-> text -----------------------------------------------------------

function fps(node) {
    const value = Number(node._vt?.info?.fps);
    return Number.isFinite(value) && value > 0 ? value : DEFAULT_FPS;
}

function units(node) {
    return node.properties?.[UNITS_PROP] === "frames" ? "frames" : "time";
}

const round6 = (value) => Math.round(value * 1e6) / 1e6;
const frameOf = (node, seconds) => Math.floor(seconds * fps(node) + 0.5);
const snap = (node, seconds) => round6(Math.floor(seconds * fps(node) + 1e-4) / fps(node));

function formatClock(seconds) {
    const total = Math.max(0, Math.round(seconds * 1000));
    const ms = total % 1000;
    const s = Math.floor(total / 1000) % 60;
    const m = Math.floor(total / 60000) % 60;
    const h = Math.floor(total / 3600000);
    const two = (n) => String(n).padStart(2, "0");
    return `${two(h)}:${two(m)}:${two(s)}.${String(ms).padStart(3, "0")}`;
}

function formatTime(node, seconds) {
    return units(node) === "frames" ? String(frameOf(node, seconds)) : formatClock(seconds);
}

// "1:02:03.5", "02:03,5", "12.5" (time) or "150" (frames) -> seconds, NaN when not understood
function parseTime(node, text) {
    const s = String(text ?? "").trim().replace(",", ".");
    if (!s) return NaN;
    if (units(node) === "frames") {
        return /^\d+$/.test(s) ? round6(parseInt(s, 10) / fps(node)) : NaN;
    }
    if (!/^(\d+:){0,2}\d+(\.\d+)?$/.test(s)) return NaN;
    const value = s.split(":").reduce((sum, part) => sum * 60 + parseFloat(part), 0);
    return Number.isFinite(value) ? round6(value) : NaN;
}

// ---- tabs / visibility (same technique as the Meme Maker) ----------------------

function currentTab(node) {
    const t = node.properties?.[TAB_PROP];
    return TABS.some((x) => x.id === t) ? t : DEFAULT_TAB;
}

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
        widget.type = orig.type; // type first: newer frontends ignore writes to hidden while it is "hidden"
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

function wantedVisible(node, name) {
    if (TAB_OF[name] !== currentTab(node)) return false;
    const placement = findWidget(node, "placement")?.value;
    if (name.startsWith("frame_") && placement !== "frame") return false;
    if (name === "frame_height" && findWidget(node, "frame_height_mode")?.value !== "fixed") return false;
    if (name.startsWith("subtitle_") && findWidget(node, "text_mode")?.value !== "subtitle layer") return false;
    return true;
}

function applyVisibility(node) {
    for (const w of node.widgets ?? []) {
        if (!(w.name in TAB_OF)) continue;
        const visible = wantedVisible(node, w.name);
        if ((w.hidden === true) === !visible) continue;
        setWidgetVisible(w, visible);
    }
}

// The node's height follows its content (grow and shrink); the width stays.
function fitHeight(node) {
    const size = node.computeSize?.();
    if (!size || !Number.isFinite(size[1])) return;
    node.setSize([Math.max(node.size[0], MIN_WIDTH, size[0]), size[1]]);
    node.graph?.setDirtyCanvas?.(true, true);
}

function relayout(node) {
    applyVisibility(node);
    fitHeight(node);
}

function createTabBar() {
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
            n.properties = n.properties || {};
            n.properties[TAB_PROP] = TABS[i].id;
            relayout(n);
            return true;
        },
    };
}

// pointing hand over the canvas-drawn buttons and the tab bar
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

// ---- reading the graph ---------------------------------------------------------

function linkOf(node, id) {
    const links = node.graph?.links;
    if (!links || id == null) return null;
    return (typeof links.get === "function" ? links.get(id) : links[id]) ?? null;
}

// The text of a connected node, when it is a plain value (Primitive, text nodes).
function linkedText(node, name) {
    try {
        const input = node.inputs?.find((i) => i.name === name || i.widget?.name === name);
        const link = linkOf(node, input?.link);
        const origin = link && node.graph?.getNodeById?.(link.origin_id);
        const widget = origin?.widgets?.find((w) => typeof w.value === "string" && w.type !== "button");
        return typeof widget?.value === "string" ? widget.value : null;
    } catch {
        return null;
    }
}

// {filename, subfolder, type} of the nearest video file in the nodes behind the
// video input, when it is a file ComfyUI's /view route can serve. null otherwise.
function upstreamVideo(node) {
    try {
        const start = linkOf(node, node.inputs?.find((i) => i.name === "video")?.link);
        if (!start) return null;
        const queue = [start.origin_id];
        const seen = new Set();
        while (queue.length && seen.size < 100) {
            const id = queue.shift();
            if (seen.has(id)) continue;
            seen.add(id);
            const origin = node.graph?.getNodeById?.(id);
            if (!origin) continue;
            for (const w of origin.widgets ?? []) {
                if (typeof w.value !== "string" || w.value.length > 1024 || /[\r\n]/.test(w.value)) continue;
                let value = w.value.trim();
                let type = "input";
                const note = value.match(/\s*\[(input|output|temp)\]\s*$/);
                if (note) {
                    type = note[1];
                    value = value.slice(0, note.index);
                }
                if (!VIDEO_EXT_RE.test(value)) continue;
                value = value.replace(/\\/g, "/");
                if (value.startsWith("/") || /^[A-Za-z]:/.test(value) || value.split("/").includes("..")) return null;
                const cut = value.lastIndexOf("/");
                return { filename: value.slice(cut + 1), subfolder: cut >= 0 ? value.slice(0, cut) : "", type };
            }
            for (const input of origin.inputs ?? []) {
                const link = linkOf(node, input.link);
                if (link) queue.push(link.origin_id);
            }
        }
    } catch {
        // nothing usable behind the input
    }
    return null;
}

const refKey = (ref) => (ref ? `${ref.type}|${ref.subfolder}|${ref.filename}` : "");

// ---- the form (the caption being edited) ----------------------------------------

function collect(node, keys) {
    const out = {};
    for (const key of keys) {
        const w = findWidget(node, key);
        if (!w) continue;
        let value = w.value;
        if (TEXT_WIDGETS.includes(key)) {
            const linked = linkedText(node, key);
            if (linked !== null) value = linked;
        }
        out[key] = value;
    }
    return out;
}

const formStyle = (node) => collect(node, STYLE_KEYS);
const frameSettings = (node) => collect(node, FRAME_KEYS);

// ---- the list of titles -----------------------------------------------------------

function getTitles(node) {
    try {
        const raw = JSON.parse(findWidget(node, "titles")?.value || "[]");
        const list = Array.isArray(raw) ? raw : Array.isArray(raw?.titles) ? raw.titles : [];
        return list.filter((t) => t && typeof t === "object" && Number.isFinite(t.start) && Number.isFinite(t.end));
    } catch {
        return [];
    }
}

// null when the list is fine, otherwise what is wrong with it
function listProblem(node, list) {
    if (list.length > MAX_TITLES) return `At most ${MAX_TITLES} titles fit in the list.`;
    for (const t of list) {
        if (!(t.start >= 0) || !(t.end > t.start)) {
            return `The end (${formatTime(node, t.end)}) must be after the start (${formatTime(node, t.start)}).`;
        }
    }
    for (let i = 1; i < list.length; i++) {
        if (list[i].start < list[i - 1].end - 1e-6) {
            return `Titles may not overlap: ${formatTime(node, list[i - 1].start)} - ${formatTime(node, list[i - 1].end)} ` +
                `and ${formatTime(node, list[i].start)} - ${formatTime(node, list[i].end)}.`;
        }
    }
    return null;
}

// Sorts, checks and stores the list. Returns the stored list, or null when it was refused.
function commitTitles(node, list, keep) {
    const sorted = [...list].sort((a, b) => a.start - b.start || a.end - b.end);
    const problem = listProblem(node, sorted);
    if (problem) {
        toast("error", problem);
        renderList(node); // puts back what was typed over
        return null;
    }
    const w = findWidget(node, "titles");
    if (w) w.value = JSON.stringify(sorted);
    node._vt.selected = keep ? sorted.indexOf(keep) : -1;
    node._vt.cache.clear();
    node._vt.wantedKey = null; // the common canvas may have changed too
    renderList(node);
    updateOverlay(node);
    node.graph?.setDirtyCanvas?.(true, true);
    return sorted;
}

function titleAt(list, seconds) {
    for (let i = 0; i < list.length; i++) {
        if (seconds >= list[i].start - 1e-6 && seconds < list[i].end - 1e-6) return i;
    }
    return -1;
}

// ---- player ------------------------------------------------------------------------

function videoSize(node) {
    const info = node._vt.info;
    if (info && info.width >= 16 && info.height >= 16) return [info.width, info.height];
    const v = node._vt.video;
    if (v.videoWidth >= 16 && v.videoHeight >= 16) return [v.videoWidth, v.videoHeight];
    return DEFAULT_SIZE;
}

function duration(node) {
    const d = node._vt.video.duration;
    if (Number.isFinite(d) && d > 0) return d;
    const known = Number(node._vt.info?.duration);
    return Number.isFinite(known) && known > 0 ? known : 0;
}

function stageHeight(node, width) {
    const ratio = node._vt?.ratio || DEFAULT_SIZE[1] / DEFAULT_SIZE[0];
    const h = Math.round(Math.max(40, (width ?? node.size[0]) - 30) * ratio);
    return Math.min(STAGE_MAX_HEIGHT, Math.max(STAGE_MIN_HEIGHT, h));
}

// Sizes the stage to the output picture's shape and puts the video where the
// picture sits in it (inside the frame bands in frame placement).
function layoutStage(node) {
    const vt = node._vt;
    const shape = vt?.shape;
    if (!vt || !shape) return;
    const availW = Math.max(40, node.size[0] - 30);
    const planned = stageHeight(node, node.size[0]);
    const measured = vt.holder.clientHeight || 0;
    const availH = Math.max(20, (measured > 20 ? Math.min(measured, planned) : planned) - 4);
    const w = Math.min(availW, (availH * shape.totalW) / shape.total);
    vt.stage.style.width = `${Math.round(w)}px`;
    vt.stage.style.height = `${Math.round((w * shape.total) / shape.totalW)}px`;
    Object.assign(vt.video.style, {
        left: `${(shape.left / shape.totalW) * 100}%`,
        width: `${(shape.width / shape.totalW) * 100}%`,
        top: `${(shape.top / shape.total) * 100}%`,
        height: `${(shape.height / shape.total) * 100}%`,
    });
}

function setStatus(node, text, isError) {
    const status = node._vt?.status;
    if (!status) return;
    status.textContent = text || "";
    status.title = text || "";
    status.style.color = isError ? "#ff8a80" : "#d8b45a";
}

function trimValues(node) {
    const start = Number(findWidget(node, "trim_start")?.value) || 0;
    const end = Number(findWidget(node, "trim_end")?.value) || 0;
    return [Math.max(0, start), Math.max(0, end)];
}

function updateReadouts(node) {
    const vt = node._vt;
    if (!vt) return;
    const t = vt.video.currentTime || 0;
    const d = duration(node);
    vt.timeLabel.textContent = `${formatTime(node, t)} / ${formatTime(node, d)}`;
    if (!vt.seeking) {
        vt.seek.max = String(d || 1);
        vt.seek.value = String(Math.min(t, d || 1));
    }
    vt.playBtn.textContent = vt.video.paused ? "▶" : "⏸";
    const [start, end] = trimValues(node);
    vt.trimLabel.textContent = start > 0 || end > 0
        ? `Cut: ${formatTime(node, start)} – ${end > 0 ? formatTime(node, end) : "end"}`
        : "Cut: whole video";
}

function seekTo(node, seconds) {
    const vt = node._vt;
    const d = duration(node);
    const half = 0.5 / fps(node);
    // the middle of the frame, so the browser never lands on the one before it
    const target = Math.max(0, seconds + half);
    vt.video.currentTime = d > half ? Math.min(d - half / 2, target) : target; // length not known yet: no upper limit
}

function stepFrames(node, count) {
    const vt = node._vt;
    vt.video.pause();
    const frame = Math.floor((vt.video.currentTime || 0) * fps(node) + 1e-4) + count;
    seekTo(node, Math.max(0, frame) / fps(node));
}

function setTrim(node, which, seconds) {
    const w = findWidget(node, which);
    if (!w) return;
    w.value = Math.round(seconds * 1000) / 1000;
    w.callback?.(w.value, app.canvas, node);
    updateReadouts(node);
    node.graph?.setDirtyCanvas?.(true, true);
}

async function probe(node, ref) {
    try {
        const res = await api.fetchApi("/mykee/titler/probe", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(ref),
        });
        if (!res.ok) return null;
        return await res.json();
    } catch {
        return null;
    }
}

// Decides what the player shows: the last run's source when it belongs to the
// loader that is connected now, otherwise that loader's own file.
async function refreshSource(node) {
    const vt = node._vt;
    if (!vt) return;
    const upstream = upstreamVideo(node);
    const key = refKey(upstream);
    vt.upstreamKey = key;
    const run = node.properties?.[RUN_PROP];
    const fromRun = run && run.key === key && run.source ? run.source : null;
    const ref = fromRun || upstream;
    // a preview copy is rewritten under the same name: the run's stamp tells them apart
    const wanted = refKey(ref) + (fromRun && ref.type === "temp" ? `|${run.stamp || 0}` : "");
    if (wanted === vt.sourceKey) return;
    // the same video again (after a run): stay where the player was
    vt.resume = vt.sourceKey !== null && key === vt.loadedUpstreamKey ? vt.video.currentTime || 0 : 0;
    vt.loadedUpstreamKey = key;
    vt.sourceKey = wanted;
    vt.cache.clear();
    vt.wantedKey = null;
    if (!ref) {
        vt.video.removeAttribute("src");
        vt.video.load();
        setStatus(node, node.inputs?.find((i) => i.name === "video")?.link != null
            ? "Run the node once to load the video into the player."
            : "Connect a video.", false);
        updateOverlay(node);
        return;
    }
    const query = new URLSearchParams({ filename: ref.filename, subfolder: ref.subfolder || "", type: ref.type });
    if (ref.type === "temp") query.set("rand", String(Date.now())); // never a cached copy of an older preview
    vt.video.src = api.apiURL(`/view?${query.toString()}`);
    setStatus(node, "", false);
    if (fromRun && run.info) {
        vt.info = run.info;
    } else {
        const info = await probe(node, ref);
        if (vt.sourceKey !== wanted) return; // another video was connected meanwhile
        if (info && info.fps > 0) {
            vt.info = { width: info.width, height: info.height, fps: info.fps, frames: info.frames, duration: info.duration };
            if (info.playable === false) {
                setStatus(node, `The browser cannot play ${info.codec || "this codec"} - run the node once to get a preview copy.`, true);
            }
        }
    }
    vt.cache.clear();
    vt.wantedKey = null;
    renderList(node);
    updateOverlay(node);
    updateReadouts(node);
}

// ---- caption layer over the player ----------------------------------------------------

function cueAt(node, seconds) {
    if (findWidget(node, "text_mode")?.value !== "subtitle layer") return null;
    for (const cue of node._vt.cues ?? []) {
        if (seconds >= cue.start && seconds < cue.end) return cue;
    }
    return null;
}

// What should be on screen right now: {show, subtitle}.
//   - empty list: the captions on the node, over the whole video;
//   - inside the Start - End range of the Titles tab (End empty = from Start on):
//     the captions on the node, when they belong to the selected title or no
//     title of the list is on screen there - so a title is seen while it is typed;
//   - otherwise the title of the list at the player's position, if there is one.
function wantedOverlay(node) {
    const vt = node._vt;
    const list = getTitles(node);
    const t = vt.video.currentTime || 0;
    const stored = (title) => Object.fromEntries(STYLE_KEYS.filter((k) => k in title).map((k) => [k, title[k]]));
    let show = null;
    if (!list.length) {
        show = formStyle(node);
    } else {
        const i = titleAt(list, t);
        const start = parseTime(node, vt.startInput?.value);
        const end = parseTime(node, vt.endInput?.value);
        const inRange = Number.isFinite(start) && t >= start - 1e-6 && (!Number.isFinite(end) || t < end - 1e-6);
        if (inRange && (i < 0 || i === vt.selected)) show = formStyle(node);
        else if (i >= 0) show = stored(list[i]);
    }
    const cue = cueAt(node, t);
    const subtitle = cue ? {
        text: cue.text,
        position: findWidget(node, "subtitle_position")?.value,
        font_size: findWidget(node, "subtitle_font_size")?.value,
        style: formStyle(node),
    } : null;
    return { show, subtitle };
}

function applyOverlay(node, data) {
    const vt = node._vt;
    vt.overlay.src = "data:image/png;base64," + data.png;
    vt.shape = { width: data.width, height: data.height, top: data.img_y, left: data.img_x, total: data.total_h, totalW: data.total_w };
    const ratio = data.total_h / data.total_w;
    const changed = Math.abs(ratio - (vt.ratio || 0)) > 0.002;
    vt.ratio = ratio;
    if (changed) fitHeight(node);
    layoutStage(node);
    const warnings = [];
    if (data.fallback) warnings.push(`Font not found on this system - using '${data.font}'.`);
    if (data.missing) warnings.push(`'${data.font}' has no glyph for: ${data.missing}`);
    if (warnings.length) setStatus(node, warnings.join(" "), false);
}

async function updateOverlay(node) {
    const vt = node._vt;
    if (!vt) return;
    const wanted = wantedOverlay(node);
    const key = JSON.stringify(wanted);
    if (key === vt.wantedKey) return;
    vt.wantedKey = key;
    const cached = vt.cache.get(key);
    if (cached) {
        applyOverlay(node, cached);
        return;
    }
    const [width, height] = videoSize(node);
    const list = getTitles(node);
    const titles = list.map((t) => Object.fromEntries(STYLE_KEYS.filter((k) => k in t).map((k) => [k, t[k]])));
    try {
        const res = await api.fetchApi("/mykee/titler/preview", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ width, height, frame: frameSettings(node), titles, ...wanted }),
        });
        let data = {};
        try {
            data = await res.json();
        } catch {
            data = {};
        }
        const numbers = [data.width, data.height, data.total_w, data.total_h, data.img_x, data.img_y];
        if (!res.ok || typeof data.png !== "string" || !numbers.every((v) => Number.isFinite(v) && v >= 0) || !(data.total_w > 0) || !(data.total_h > 0)) {
            if (vt.wantedKey === key) setStatus(node, data.error || "the preview could not be rendered", true);
            return;
        }
        if (vt.cache.size > 300) vt.cache.clear();
        vt.cache.set(key, data);
        if (vt.wantedKey === key) applyOverlay(node, data);
    } catch {
        if (vt.wantedKey === key) setStatus(node, "the preview could not be rendered", true);
    }
}

function scheduleOverlay(node, delay = 250) {
    clearTimeout(node._vt?.timer);
    if (node._vt) node._vt.timer = setTimeout(() => updateOverlay(node), delay);
}

// Anything that changes the picture: drop the rendered layers and draw again.
function formChanged(node) {
    const vt = node._vt;
    if (!vt) return;
    vt.cache.clear();
    vt.wantedKey = null;
    updateReadouts(node);
    scheduleOverlay(node);
}

function createPlayer(node) {
    const vt = node._vt;
    const video = el("video", { preload: "metadata", playsInline: true, controls: false });
    css(video, { position: "absolute", objectFit: "fill", background: PREVIEW_GRAY });
    const overlay = el("img", { alt: "", draggable: false });
    css(overlay, { position: "absolute", left: "0", top: "0", width: "100%", height: "100%", pointerEvents: "none" });
    const stage = css(el("div", {}, [video, overlay]), {
        position: "relative", margin: "0 auto", borderRadius: "4px", overflow: "hidden", background: "#1c1c1c", cursor: "pointer",
    });
    const holder = css(el("div", {}, [stage]), { flex: "1 1 0", minHeight: "0", width: "100%", overflow: "hidden" });
    if (typeof ResizeObserver !== "undefined") new ResizeObserver(() => layoutStage(node)).observe(holder);

    const seek = el("input", { type: "range", min: "0", max: "1", step: "0.001", value: "0", title: "Position" });
    css(seek, { width: "100%", margin: "4px 0 2px", cursor: "pointer", height: "16px" });

    const toggle = () => {
        if (video.paused) {
            video.play().catch(() => setStatus(node, "The browser could not start the video.", true));
        } else {
            video.pause();
        }
    };
    const playBtn = button("▶", "Play / pause", toggle);
    css(playBtn, { minWidth: "34px" });
    const timeLabel = css(el("div"), { font: "12px monospace", flex: "1 1 auto", textAlign: "center", whiteSpace: "nowrap", overflow: "hidden" });
    const transport = css(el("div", {}, [
        button("⏮", "Jump to the start", () => seekTo(node, 0)),
        button("◀", "One frame back", () => stepFrames(node, -1)),
        playBtn,
        button("▶|", "One frame forward", () => stepFrames(node, 1)),
        button("⏭", "Jump to the end", () => seekTo(node, duration(node))),
        timeLabel,
    ]), { display: "flex", gap: "4px", alignItems: "center", height: "26px" });

    const trimLabel = css(el("div"), { font: "12px sans-serif", flex: "1 1 auto", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", opacity: "0.85" });
    const trimRow = css(el("div", {}, [
        button("[ Cut start", "The output starts at the player's position", () => setTrim(node, "trim_start", snap(node, video.currentTime || 0))),
        button("Cut end ]", "The output ends at the player's position", () => setTrim(node, "trim_end", snap(node, video.currentTime || 0))),
        button("No cut", "Keep the whole video", () => {
            setTrim(node, "trim_start", 0);
            setTrim(node, "trim_end", 0);
        }),
        trimLabel,
    ]), { display: "flex", gap: "4px", alignItems: "center", height: "26px", marginTop: "4px" });

    const status = css(el("div"), {
        font: "11px sans-serif", height: "18px", lineHeight: "18px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
    });
    const box = css(el("div", {}, [holder, seek, transport, trimRow, status]), {
        display: "flex", flexDirection: "column", width: "100%", height: "100%", boxSizing: "border-box",
    });

    Object.assign(vt, { video, overlay, stage, holder, seek, playBtn, timeLabel, trimLabel, status });

    stage.addEventListener("click", toggle);
    seek.addEventListener("pointerdown", () => { vt.seeking = true; });
    seek.addEventListener("input", () => { video.currentTime = Number(seek.value) || 0; });
    const endSeek = () => { vt.seeking = false; };
    seek.addEventListener("change", endSeek);
    seek.addEventListener("pointerup", endSeek);

    const refresh = () => {
        updateReadouts(node);
        updateOverlay(node);
    };
    const loop = () => {
        if (video.paused || vt.removed) return;
        refresh();
        requestAnimationFrame(loop);
    };
    video.addEventListener("play", () => requestAnimationFrame(loop));
    for (const name of ["pause", "seeked", "timeupdate", "loadedmetadata", "durationchange", "ended"]) {
        video.addEventListener(name, refresh);
    }
    video.addEventListener("loadedmetadata", () => {
        if (vt.resume > 0) {
            seekTo(node, Math.min(vt.resume, duration(node)));
            vt.resume = 0;
        }
        vt.cache.clear();
        vt.wantedKey = null;
        renderList(node);
        refresh();
    });
    video.addEventListener("error", () => {
        if (!video.getAttribute("src")) return;
        setStatus(node, "The browser cannot play this video - run the node once to get a preview copy.", true);
    });

    const w = node.addDOMWidget(PLAYER_NAME, "MYKEE_TITLER_PLAYER", box, { serialize: false });
    w.serialize = false;
    w.computeSize = (width) => [width, stageHeight(node, width) + CONTROLS_HEIGHT];
    w.computeLayoutSize = () => {
        const h = stageHeight(node) + CONTROLS_HEIGHT;
        return { minHeight: h, maxHeight: h, minWidth: 0 };
    };
}

// ---- Titles tab ------------------------------------------------------------------------

function styleSummary(title) {
    const parts = [];
    if (title.top_text) parts.push(`top: ${title.top_text}`);
    if (title.bottom_text) parts.push(`bottom: ${title.bottom_text}`);
    return parts.join("\n") || "(no text)";
}

function loadIntoForm(node, title) {
    for (const key of STYLE_KEYS) {
        if (!(key in title)) continue;
        const w = findWidget(node, key);
        if (!w) continue;
        w.value = title[key];
        const input = w.inputEl || w.element;
        if (TEXT_WIDGETS.includes(key) && input && "value" in input) input.value = String(title[key] ?? "");
    }
    node._vt.sync?.();
}

function renderList(node) {
    const vt = node._vt;
    if (!vt?.list) return;
    const list = getTitles(node);
    vt.list.replaceChildren();
    vt.unitsBtn.textContent = units(node) === "frames" ? "Units: frames" : "Units: time";
    const hint = units(node) === "frames" ? "frame number" : "hh:mm:ss.mmm";
    vt.startInput.placeholder = hint;
    vt.endInput.placeholder = hint;
    vt.count.textContent = list.length
        ? `${list.length} title${list.length === 1 ? "" : "s"}`
        : "No titles: the captions above cover the whole video.";
    vt.updateBtn.disabled = !(vt.selected >= 0 && vt.selected < list.length);
    vt.updateBtn.style.opacity = vt.updateBtn.disabled ? "0.45" : "1";

    list.forEach((title, index) => {
        const timeField = (key) => {
            const input = css(el("input", { type: "text", value: formatTime(node, title[key]), title: `${key} (${hint})` }), {
                ...INPUT_STYLE, width: "100%",
            });
            input.addEventListener("change", () => {
                const seconds = parseTime(node, input.value);
                if (!Number.isFinite(seconds)) {
                    toast("error", `"${input.value}" is not a valid ${units(node) === "frames" ? "frame number" : "time"}.`);
                    renderList(node);
                    return;
                }
                const next = getTitles(node);
                next[index] = { ...next[index], [key]: seconds };
                commitTitles(node, next, next[index]);
            });
            return input;
        };
        const textField = (key, label) => {
            const input = css(el("textarea", { rows: 1, value: String(title[key] ?? ""), placeholder: label, title: label }), {
                ...INPUT_STYLE, font: "12px sans-serif", width: "100%", height: "24px", resize: "vertical", minHeight: "24px",
            });
            input.addEventListener("change", () => {
                const next = getTitles(node);
                next[index] = { ...next[index], [key]: input.value };
                commitTitles(node, next, next[index]);
            });
            return input;
        };
        const number = css(el("div", { textContent: String(index + 1), title: "Select this title" }), {
            font: "12px monospace", minWidth: "22px", textAlign: "right", cursor: "pointer", opacity: "0.7",
        });
        number.addEventListener("click", () => {
            vt.selected = vt.selected === index ? -1 : index;
            renderList(node);
            vt.wantedKey = null;
            updateOverlay(node);
        });
        const top = css(el("div", {}, [
            number,
            timeField("start"),
            timeField("end"),
            button("▶", "Jump to the start of this title", () => seekTo(node, title.start)),
            button("✎", "Load this title into the fields above for editing", () => {
                vt.selected = index;
                vt.startInput.value = formatTime(node, title.start);
                vt.endInput.value = formatTime(node, title.end);
                loadIntoForm(node, title);
                vt.wantedKey = null;
                vt.video.pause();
                seekTo(node, title.start);
                renderList(node);
                updateReadouts(node);
                updateOverlay(node);
            }),
            button("✕", "Delete this title", () => {
                const next = getTitles(node);
                next.splice(index, 1);
                commitTitles(node, next, null);
            }),
        ]), { display: "grid", gridTemplateColumns: "auto 1fr 1fr auto auto auto", gap: "4px", alignItems: "center" });
        const bottom = css(el("div", {}, [textField("top_text", "top text"), textField("bottom_text", "bottom text")]), {
            display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px", marginTop: "3px", paddingLeft: "26px",
        });
        const row = css(el("div", { title: styleSummary(title) }, [top, bottom]), {
            padding: "5px", borderRadius: "5px", marginBottom: "4px",
            border: `1px solid ${vt.selected === index ? "#7aa2d6" : "var(--border-color, #444)"}`,
            background: vt.selected === index ? "rgba(74,111,165,0.25)" : "transparent",
        });
        vt.list.append(row);
    });
}

// Loads a JSON saved by this node (already validated by the server): the list
// is replaced, the cut and the frame settings are set. Settings the server
// could not use here are missing from it; a title then keeps the node's own.
function applyProject(node, project) {
    const vt = node._vt;
    const style = formStyle(node);
    const titles = (Array.isArray(project.titles) ? project.titles : [])
        .filter((t) => t && Number.isFinite(t.start) && Number.isFinite(t.end) && t.end > t.start)
        .map((t) => {
            const own = t.style && typeof t.style === "object" ? t.style : {};
            const picked = Object.fromEntries(STYLE_KEYS.filter((k) => k in own).map((k) => [k, own[k]]));
            return { start: round6(t.start), end: round6(t.end), ...style, top_text: "", bottom_text: "", ...picked };
        });
    const setWidget = (name, value) => {
        const w = findWidget(node, name);
        if (w && value !== undefined && value !== null) w.value = value;
    };
    if (project.frame && typeof project.frame === "object") {
        for (const key of FRAME_KEYS) if (key in project.frame) setWidget(key, project.frame[key]);
    }
    if (project.trim && Number.isFinite(project.trim.start) && Number.isFinite(project.trim.end)) {
        setWidget("trim_start", project.trim.start);
        setWidget("trim_end", project.trim.end);
    }
    if (project.form && typeof project.form === "object") loadIntoForm(node, project.form);
    const stored = commitTitles(node, titles, null);
    vt.sync?.();
    if (!stored) return;
    const ignored = Array.isArray(project.ignored) ? project.ignored.filter((x) => typeof x === "string") : [];
    toast(ignored.length ? "warn" : "success",
        `Loaded ${stored.length} title${stored.length === 1 ? "" : "s"} from the saved JSON (the list was replaced).` +
        (ignored.length ? ` Left out: ${ignored.join(", ")}.` : ""));
}

async function importText(node) {
    const vt = node._vt;
    let cues = null;
    let project = null;
    const linked = linkedText(node, "text");
    if (linked !== null && linked.trim()) {
        try {
            const res = await api.fetchApi("/mykee/titler/parse", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: linked, duration: duration(node) }),
            });
            const data = await res.json();
            if (res.ok && data.kind === "project" && data.project && typeof data.project === "object") project = data.project;
            else if (res.ok && Array.isArray(data.cues)) cues = data.cues;
        } catch {
            cues = null;
        }
    } else if (vt.project) {
        project = vt.project; // from the last run (the text itself is not readable here)
    }
    if (project) {
        applyProject(node, project);
        return;
    }
    if (!cues) cues = vt.cues?.length ? vt.cues : null;
    if (!cues || !cues.length) {
        toast("warn", node.inputs?.find((i) => i.name === "text")?.link != null
            ? "The text is not known here yet - run the node once, then import again."
            : "Connect a text (SRT, WebVTT, plain text or a JSON saved by this node) to the text input first.");
        return;
    }
    const style = formStyle(node);
    const side = findWidget(node, "subtitle_position")?.value === "top" ? "top_text" : "bottom_text";
    const list = getTitles(node);
    let added = 0;
    let skipped = 0;
    for (const cue of cues) {
        if (!(cue && Number.isFinite(cue.start) && Number.isFinite(cue.end) && cue.end > cue.start && typeof cue.text === "string")) continue;
        const clash = list.some((t) => cue.start < t.end - 1e-6 && cue.end > t.start + 1e-6);
        if (clash || list.length >= MAX_TITLES) {
            skipped++;
            continue;
        }
        list.push({ start: round6(cue.start), end: round6(cue.end), ...style, top_text: "", bottom_text: "", [side]: cue.text });
        added++;
    }
    if (added && commitTitles(node, list, null)) {
        toast("success", `Imported ${added} title${added === 1 ? "" : "s"}` + (skipped ? `, skipped ${skipped} that overlap existing ones.` : "."));
    } else if (!added) {
        toast("warn", "Nothing was imported: every cue overlaps a title that is already in the list.");
    }
}

function createTitlesPanel(node) {
    const vt = node._vt;
    const label = (text) => css(el("div", { textContent: text }), { font: "12px sans-serif", minWidth: "36px", opacity: "0.8" });
    const field = () => css(el("input", { type: "text" }), { ...INPUT_STYLE, flex: "1 1 auto", width: "100%" });
    vt.startInput = field();
    vt.endInput = field();
    const rangeChanged = () => {
        vt.wantedKey = null;
        updateOverlay(node); // the range decides where the typed captions are shown
    };
    for (const input of [vt.startInput, vt.endInput]) input.addEventListener("input", rangeChanged);
    const position = () => formatTime(node, snap(node, vt.video.currentTime || 0));
    const row = (children) => css(el("div", {}, children), { display: "flex", gap: "4px", alignItems: "center", marginBottom: "4px" });

    vt.unitsBtn = button("Units: time", "Show and type the times as time or as frame numbers", () => {
        const before = [parseTime(node, vt.startInput.value), parseTime(node, vt.endInput.value)];
        node.properties = node.properties || {};
        node.properties[UNITS_PROP] = units(node) === "frames" ? "time" : "frames";
        vt.startInput.value = Number.isFinite(before[0]) ? formatTime(node, before[0]) : "";
        vt.endInput.value = Number.isFinite(before[1]) ? formatTime(node, before[1]) : "";
        renderList(node);
        updateReadouts(node);
    });

    const read = () => {
        const start = parseTime(node, vt.startInput.value);
        const end = parseTime(node, vt.endInput.value);
        if (!Number.isFinite(start) || !Number.isFinite(end)) {
            toast("error", "Fill in the start and the end first (type them, or take them from the player's position).");
            return null;
        }
        return { start, end };
    };

    const addBtn = button("+ Add to the list", "Add the captions above with this start and end to the list", () => {
        const times = read();
        if (!times) return;
        const title = { ...times, ...formStyle(node) };
        commitTitles(node, [...getTitles(node), title], title);
    });
    vt.updateBtn = button("Update selected", "Replace the selected title with the captions above and this start and end", () => {
        const times = read();
        const list = getTitles(node);
        if (!times || !(vt.selected >= 0 && vt.selected < list.length)) return;
        const title = { ...times, ...formStyle(node) };
        list[vt.selected] = title;
        commitTitles(node, list, title);
    });

    vt.list = css(el("div"), { maxHeight: `${LIST_MAX_HEIGHT}px`, overflowY: "auto", marginTop: "2px" });
    vt.list.addEventListener("wheel", (event) => {
        if (vt.list.scrollHeight > vt.list.clientHeight) event.stopPropagation(); // scroll the list, not the canvas zoom
    }, { passive: true });
    vt.count = css(el("div"), { font: "11px sans-serif", opacity: "0.7", flex: "1 1 auto", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" });

    const content = css(el("div", {}, [
        row([label("Start"), vt.startInput, button("⇤ Position", "Use the player's position as the start", () => { vt.startInput.value = position(); rangeChanged(); })]),
        row([label("End"), vt.endInput, button("Position ⇥", "Use the player's position as the end (the title is gone from this frame on)", () => { vt.endInput.value = position(); rangeChanged(); })]),
        row([addBtn, vt.updateBtn, css(el("div"), { flex: "1 1 auto" }), vt.unitsBtn]),
        vt.list,
        row([
            vt.count,
            button("Import text", "SRT / WebVTT / plain text: adds its cues to the list as titles with the current style. A JSON saved by this node: loads it back (replaces the list, sets the cut and the frame settings).", () => importText(node)),
            button("Clear list", "Remove every title", () => {
                if (getTitles(node).length) commitTitles(node, [], null);
            }),
        ]),
    ]), { boxSizing: "border-box", width: "100%", padding: "2px 0", color: "var(--input-text, #ddd)" });
    const wrapper = css(el("div", {}, [content]), { boxSizing: "border-box", width: "100%", overflow: "visible" });

    vt.panelHeight = 150;
    const w = node.addDOMWidget(PANEL_NAME, "MYKEE_TITLER_PANEL", wrapper, { serialize: false });
    w.serialize = false;
    w.computeSize = (width) => [width, vt.panelHeight];
    w.computeLayoutSize = () => ({ minHeight: vt.panelHeight, maxHeight: vt.panelHeight, minWidth: 0 });
    if (typeof ResizeObserver !== "undefined") {
        vt.panelObserver = new ResizeObserver(() => {
            const measured = content.offsetHeight; // layout pixels - the canvas zoom does not change it
            if (!measured) return; // hidden (another tab) or collapsed
            const wanted = measured + 6;
            if (Math.abs(wanted - vt.panelHeight) < 1) return;
            vt.panelHeight = wanted;
            if (currentTab(node) === "titles") fitHeight(node);
        });
        vt.panelObserver.observe(content);
    }
}


// Widgets added after the first release. Workflows store widget values by
// position, so in a workflow saved before they existed every value after
// them would land one field too far. Such a workflow (no layout mark in its
// properties) is read with the old order, and the new widgets keep their
// default. Handles both ways frontends store the values (only the saved
// widgets, or every widget by index).
const LAYOUT_PROP = "mykee_titler_layout";
const LAYOUT_VERSION = 2;
const ADDED_WIDGETS = ["top_margin", "bottom_margin"];

function migrateWidgetValues(node, info) {
    const saved = info?.properties?.[LAYOUT_PROP];
    node.properties = node.properties || {};
    node.properties[LAYOUT_PROP] = LAYOUT_VERSION;
    const values = info?.widgets_values;
    if (saved === LAYOUT_VERSION || !Array.isArray(values) || !node.widgets) return;
    const before = node.widgets.filter((w) => !ADDED_WIDGETS.includes(w.name));
    const stored = before.filter((w) => w.serialize !== false);
    const source = values.length <= stored.length ? stored : before; // compact list / one slot per widget
    source.forEach((w, index) => {
        const value = values[index];
        if (w.serialize === false || value === undefined || value === null) return;
        w.value = value;
        const input = w.inputEl || w.element;
        if (typeof value === "string" && input && "value" in input && input.tagName === "TEXTAREA") input.value = value;
    });
    for (const name of ADDED_WIDGETS) {
        const w = node.widgets.find((x) => x.name === name);
        if (w) w.value = 0;
    }
}

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

function setupNode(node) {
    if (node._vt) return;
    node.properties = node.properties || {};
    node._vt = {
        cache: new Map(), selected: -1, info: null, cues: [], sourceKey: null, wantedKey: null,
        shape: null, ratio: DEFAULT_SIZE[1] / DEFAULT_SIZE[0],
    };
    const vt = node._vt;
    const run = node.properties[RUN_PROP];
    if (run && typeof run === "object") {
        vt.info = run.info || null;
        vt.cues = Array.isArray(run.cues) ? run.cues : [];
    }

    createPlayer(node);
    createTitlesPanel(node);
    setWidgetVisible(findWidget(node, "titles"), false);

    // --- font: hidden string widget + picker -------------------------------
    const fontW = findWidget(node, "font");
    setWidgetVisible(fontW, false);
    const pickerW = node.addWidget("combo", "font_picker", NO_FONT, (value) => {
        fontW.value = value === NO_FONT ? "" : value;
        formChanged(node);
        node.setDirtyCanvas?.(true, true);
    }, { values: [NO_FONT], serialize: false });
    pickerW.serialize = false;
    pickerW.tooltip = "Pick one of the fonts installed on this system.";
    const syncPicker = () => {
        const names = pickerW.options.values;
        if (fontW.value && !names.includes(fontW.value)) names.splice(1, 0, fontW.value); // a font from another machine stays visible
        pickerW.value = fontW.value ? fontW.value : NO_FONT;
    };
    const fillFonts = (list) => {
        const names = Array.isArray(list?.fonts) ? list.fonts.filter((n) => typeof n === "string") : [];
        const keep = fontW.value && !names.includes(fontW.value) ? [fontW.value] : [];
        pickerW.options.values = [NO_FONT, ...keep, ...names];
        syncPicker();
        node.setDirtyCanvas?.(true, true);
    };
    loadFonts().then(fillFonts);
    const reloadW = node.addWidget("button", "reload_fonts", "Reload font list", () => {
        loadFonts(true).then((list) => {
            fillFonts(list);
            formChanged(node);
        });
    });
    reloadW.label = "Reload font list";
    reloadW.serialize = false;

    // --- color swatches ------------------------------------------------------
    const swatches = {};
    for (const name of COLOR_WIDGETS) {
        const colorW = findWidget(node, name);
        if (!colorW) continue;
        colorW.value = normalizeHex(colorW.value, "#FFFFFF");
        const picker = el("input", { type: "color", value: colorW.value, title: `Pick ${name.replace("_", " ")}` });
        css(picker, {
            width: "100%", height: "24px", padding: "0", margin: "0", cursor: "pointer", background: "transparent", boxSizing: "border-box",
            border: "1px solid var(--border-color, #4a4a4a)", borderRadius: "4px",
        });
        picker.addEventListener("input", () => {
            colorW.value = picker.value.toUpperCase();
            colorW.callback?.(colorW.value, app.canvas, node);
            node.graph?.setDirtyCanvas(true, true);
        });
        const swatchW = node.addDOMWidget(`${name}_swatch`, "COLORSWATCH", picker, { serialize: false });
        swatchW.serialize = false;
        swatchW.computeSize = (width) => [width, SWATCH_HEIGHT];
        swatchW.computeLayoutSize = () => ({ minHeight: SWATCH_HEIGHT, maxHeight: SWATCH_HEIGHT, minWidth: 0 });
        swatches[name] = { picker, colorW };
    }
    const syncSwatches = () => {
        for (const { picker, colorW } of Object.values(swatches)) picker.value = normalizeHex(colorW.value, picker.value);
    };

    // --- order: texts, tab bar, player, then the settings ---------------------
    node.widgets.push(createTabBar());
    const byName = new Map(node.widgets.map((w) => [w.name, w]));
    const order = [...TEXT_FIELDS, TAB_BAR_NAME, PLAYER_NAME];
    const textGroup = ["top_font_size", "top_align", "top_lines", "top_margin", "bottom_font_size", "bottom_align", "bottom_lines", "bottom_margin"];
    order.push(...textGroup.filter((n) => byName.has(n)));
    for (const w of node.widgets) {
        const special = w.name.endsWith("_swatch") || ["font_picker", "reload_fonts", PANEL_NAME].includes(w.name);
        if (order.includes(w.name) || special) continue;
        if (w.name === "titles") order.push(PANEL_NAME);
        order.push(w.name);
        if (w.name === "font") order.push("font_picker", "reload_fonts");
        if (COLOR_WIDGETS.includes(w.name)) order.push(`${w.name}_swatch`);
    }
    const ordered = order.map((n) => byName.get(n)).filter(Boolean);
    for (const w of node.widgets) if (!ordered.includes(w)) ordered.push(w);
    node.widgets = ordered;

    // --- changes -> preview / layout ---------------------------------------
    for (const key of [...STYLE_KEYS, ...FRAME_KEYS, ...OTHER_WATCHED]) {
        const w = findWidget(node, key);
        if (!w) continue;
        const original = w.callback;
        w.callback = function () {
            const r = original ? original.apply(this, arguments) : undefined;
            if (COLOR_WIDGETS.includes(key)) {
                const hex = normalizeHex(w.value, null);
                if (hex) {
                    w.value = hex;
                    syncSwatches();
                }
            }
            if (key === "placement" || key === "frame_height_mode" || key === "text_mode") relayout(node);
            formChanged(node);
            return r;
        };
        if (TEXT_WIDGETS.includes(key)) {
            for (const input of new Set([w.element, w.inputEl])) {
                input?.addEventListener?.("input", () => formChanged(node));
            }
        }
    }

    vt.sync = () => {
        syncPicker();
        syncSwatches();
        relayout(node);
        vt.cache.clear();
        vt.wantedKey = null;
        renderList(node);
        updateReadouts(node);
        scheduleOverlay(node, 50);
    };
    node.properties[LAYOUT_PROP] = LAYOUT_VERSION;
    if (!TABS.some((t) => t.id === node.properties[TAB_PROP])) node.properties[TAB_PROP] = DEFAULT_TAB;
    node.setSize([Math.max(node.size[0], MIN_WIDTH), node.size[1]]);
    relayout(node);
    renderList(node);
    updateReadouts(node);
    scheduleOverlay(node, 100);
    setTimeout(() => {
        relayout(node);
        refreshSource(node);
    }, 0);
    // Another file picked in the loader gives this node no event, so look regularly
    // (a timer, not a draw hook: it must not depend on the canvas redrawing this node).
    vt.poll = setInterval(() => {
        if (node.graph && refKey(upstreamVideo(node)) !== vt.upstreamKey) refreshSource(node);
    }, 700);
}

app.registerExtension({
    name: "Mykee.VideoTitler",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            setupNode(this);
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            const vt = this._vt;
            if (!vt) return r;
            migrateWidgetValues(this, arguments[0]);
            const run = this.properties?.[RUN_PROP];
            if (run && typeof run === "object") {
                vt.info = run.info || vt.info;
                vt.cues = Array.isArray(run.cues) ? run.cues : [];
            }
            const resync = () => {
                vt.sync?.();
                refreshSource(this);
            };
            resync();
            setTimeout(resync, 0);
            setTimeout(resync, 200); // widget values and the other nodes' links are in by now
            return r;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
            if (this._vt && !app.configuringGraph) {
                setTimeout(() => refreshSource(this), 0);
                formChanged(this);
            }
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            const info = message?.mykee_titler?.[0];
            const vt = this._vt;
            if (!vt || !info || typeof info !== "object") return r;
            const source = info.source && typeof info.source.filename === "string" && ["input", "output", "temp"].includes(info.source.type)
                ? { filename: info.source.filename, subfolder: String(info.source.subfolder || ""), type: info.source.type }
                : null;
            const cues = Array.isArray(info.cues) ? info.cues.slice(0, MAX_TITLES) : [];
            const known = { width: info.width, height: info.height, fps: info.fps, frames: info.frames, duration: info.duration };
            this.properties = this.properties || {};
            this.properties[RUN_PROP] = { key: refKey(upstreamVideo(this)), source, info: known, cues, stamp: Date.now() };
            vt.info = known;
            vt.cues = cues;
            vt.project = info.cue_kind === "project" && info.project && typeof info.project === "object" ? info.project : null;
            vt.cache.clear();
            vt.wantedKey = null;
            refreshSource(this);
            updateOverlay(this);
            const lines = [];
            if (info.saved?.video) lines.push(`Saved ${info.saved.video} + ${info.saved.json}`);
            if (Array.isArray(info.warnings)) lines.push(...info.warnings.filter((w) => typeof w === "string"));
            setStatus(this, lines.join("  ·  "), false);
            renderList(this);
            updateReadouts(this);
            return r;
        };

        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function () {
            const r = onResize ? onResize.apply(this, arguments) : undefined;
            if (this._vt) layoutStage(this);
            return r;
        };

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function () {
            const r = onDrawForeground ? onDrawForeground.apply(this, arguments) : undefined;
            updateButtonCursor(this);
            return r;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            const vt = this._vt;
            if (vt) {
                vt.removed = true;
                clearTimeout(vt.timer);
                clearInterval(vt.poll);
                vt.panelObserver?.disconnect();
                try {
                    vt.video.pause();
                    vt.video.removeAttribute("src");
                    vt.video.load();
                } catch {
                    // already gone
                }
            }
            return onRemoved ? onRemoved.apply(this, arguments) : undefined;
        };
    },
});
