import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * Mykee Meme Maker - front end.
 *
 * The Python side declares every setting as a normal widget (so any of them
 * can still be converted to an input). This extension adds:
 *
 *   - three tabs (Text / Style / Layout) in the Text Analyzer's tab bar style;
 *   - a font picker: the list of the system's fonts comes from
 *     /mykee/meme/fonts, the choice is written into the (hidden) "font" widget
 *     - only a family name is stored, never a path;
 *   - color swatches (native color input) under the three hex fields;
 *   - a live preview, rendered by the same Python code as the node itself
 *     (POST /mykee/meme/preview), so the font, wrapping and outline are what
 *     the node will produce. It is drawn on a gray picture of the last seen
 *     input size (known after the first run, 1024x576 before) and uses the
 *     text of a connected Primitive / text node when it can read it;
 *   - the connected picture under the preview before the first run: the file
 *     of the loader behind the image / video input is shown through ComfyUI's
 *     own /view route (the first frame of a video). With both inputs connected
 *     and active the preview says that only one should be connected.
 *
 * The node's height always follows its content (the preview's height follows
 * the picture's shape); the width stays the user's.
 */

const NODE_NAME = "MykeeMemeMaker";
const MIN_WIDTH = 380;
const TAB_PROP = "mykee_meme_tab";
const SIZE_PROP = "mykee_meme_size";
const DEFAULT_TAB = "text";
const TAB_BAR_NAME = "_mykee_meme_tabs";
const TAB_BAR_HEIGHT = 30;
const PREVIEW_STATUS_HEIGHT = 20;
const PREVIEW_MIN_HEIGHT = 80;
const PREVIEW_MAX_HEIGHT = 320;
const DEFAULT_SIZE = [1024, 1024];
const PREVIEW_SIZE_WIDGETS = ["preview_width", "preview_height"]; // always visible, above the tabs
const TEXT_FIELDS = ["top_text", "bottom_text"]; // always visible above the tabs too (so a text can be connected)
const NO_FONT = "(default)";
const PREVIEW_GRAY = "#707070";
const SWATCH_HEIGHT = 40; // 24px swatch + a gap below it
const RUN_KEY_PROP = "mykee_meme_run_key"; // which connected file the last run's size / first frame belong to
const BOTH_MESSAGE = "For the preview, connect only one of image / video.";
const MEDIA_EXT = {
    image: /\.(png|jpe?g|jfif|webp|gif|bmp|avif)$/i,
    video: /\.(mp4|m4v|mov|mkv|webm|ogv)$/i,
};

const TABS = [
    { id: "text", label: "Text" },
    { id: "style", label: "Style" },
    { id: "layout", label: "Layout" },
];

// widget name -> tab
const TAB_OF = {
    top_font_size: "text", top_align: "text", top_lines: "text", top_margin: "text",
    bottom_font_size: "text", bottom_align: "text", bottom_lines: "text", bottom_margin: "text",
    font_picker: "style", reload_fonts: "style", bold: "style", italic: "style",
    uppercase: "style", text_color: "style", text_color_swatch: "style", stroke_color: "style",
    stroke_color_swatch: "style", stroke_width: "style", line_height: "style",
    placement: "layout", padding: "layout", frame_color: "layout", frame_color_swatch: "layout", frame_width: "layout", frame_width_mode: "layout",
    frame_height_mode: "layout", frame_height: "layout",
};

const PARAM_KEYS = [
    "top_text", "bottom_text", "font", "top_font_size", "bottom_font_size", "bold", "italic", "uppercase", "text_color",
    "stroke_color", "stroke_width", "line_height", "padding", "placement", "frame_color", "frame_width", "frame_width_mode",
    "frame_height_mode", "frame_height", "top_align", "top_lines", "bottom_align", "bottom_lines",
    "top_margin", "bottom_margin",
];
// settings of the preview only (not sent as caption settings)
const COLOR_WIDGETS = ["text_color", "stroke_color", "frame_color"];
const TEXT_WIDGETS = ["top_text", "bottom_text"];

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

// ---- tabs ------------------------------------------------------------------

function currentTab(node) {
    const t = node.properties?.[TAB_PROP];
    return TABS.some((x) => x.id === t) ? t : DEFAULT_TAB;
}

// Shows / hides a widget (incl. its DOM element); a hidden one gets no height.
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

// Which widgets are shown: the current tab's, minus the frame settings that
// don't apply to the chosen placement.
function wantedVisible(node, name) {
    if (TAB_OF[name] !== currentTab(node)) return false;
    const placement = findWidget(node, "placement")?.value;
    if (name.startsWith("frame_color") && placement !== "frame") return false;
    if ((name === "frame_width" || name === "frame_width_mode" || name === "frame_height_mode" || name === "frame_height") && placement !== "frame") return false;
    if (name === "frame_height" && findWidget(node, "frame_height_mode")?.value !== "fixed") return false;
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

// ---- the connected picture (before the node has run) ------------------------

function linkOf(node, id) {
    const links = node.graph?.links;
    if (!links || id == null) return null;
    return (typeof links.get === "function" ? links.get(id) : links[id]) ?? null;
}

// The link on an input, when the node it comes from is active (not muted / bypassed).
function activeLink(node, name) {
    const link = linkOf(node, node.inputs?.find((i) => i.name === name)?.link);
    if (!link) return null;
    const origin = node.graph?.getNodeById?.(link.origin_id);
    return origin && (origin.mode ?? 0) === 0 ? link : null;
}

// {filename, subfolder, type} of the nearest media file in the nodes behind a
// link, when it is a file ComfyUI's /view route can serve. null otherwise.
function upstreamFile(node, start, pattern) {
    try {
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
                if (!pattern.test(value)) continue;
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

// The last run's size / first frame belong to another file than the one connected now.
function runIsStale(node) {
    const runKey = node.properties?.[RUN_KEY_PROP];
    const wanted = node._mykeeMemeWantedKey;
    if (typeof wanted !== "string") return false; // nothing looked at yet
    // A run from before the file was recorded: unknown which picture it was, so a connected file wins.
    if (typeof runKey !== "string") return wanted !== "";
    return runKey !== wanted;
}

// Looks at what is connected and loads its file as the preview's background.
function refreshMedia(node) {
    if (!node._mykeeMemeReady) return;
    const image = activeLink(node, "image");
    const video = activeLink(node, "video");
    const both = !!image && !!video;
    const kind = both ? null : image ? "image" : video ? "video" : null;
    const ref = kind ? upstreamFile(node, image || video, MEDIA_EXT[kind]) : null;
    const key = both ? "both" : ref ? `${kind}|${ref.type}|${ref.subfolder}|${ref.filename}` : "";
    if (key === node._mykeeMemeWantedKey) return;
    node._mykeeMemeWantedKey = key;
    node._mykeeMemeBoth = both;
    node._mykeeMemeMedia = null;
    node._mykeeMemeFieldsWin = false; // a new picture: its own size is the preview's again
    const bgVideo = node._mykeeMemeVideo;
    const finish = () => {
        layoutStage(node);
        schedulePreview(node, 50);
    };
    if (!ref) {
        bgVideo?.removeAttribute("src");
        bgVideo?.load();
        finish();
        return;
    }
    const query = new URLSearchParams({ filename: ref.filename, subfolder: ref.subfolder, type: ref.type });
    const url = api.apiURL(`/view?${query.toString()}`);
    const loaded = (width, height) => {
        if (node._mykeeMemeWantedKey !== key || !(width >= 16) || !(height >= 16)) return;
        node._mykeeMemeMedia = { kind, url, width, height };
        finish();
    };
    if (kind === "image") {
        bgVideo?.removeAttribute("src");
        bgVideo?.load();
        const probe = new Image();
        probe.onload = () => loaded(probe.naturalWidth, probe.naturalHeight);
        probe.onerror = finish; // a format the browser cannot show: the gray picture stays
        probe.src = url;
    } else if (bgVideo) {
        bgVideo.onloadedmetadata = () => {
            loaded(bgVideo.videoWidth, bgVideo.videoHeight);
            try {
                bgVideo.currentTime = 0.001; // makes the browser draw the first frame
            } catch {
                // the frame just stays empty
            }
        };
        bgVideo.onerror = finish;
        bgVideo.src = url;
    }
    finish();
}

// ---- preview ---------------------------------------------------------------

// The picture shape the preview is drawn for:
//   - before the node has run: the preview_width / preview_height fields (1024 x 1024);
//   - after a run: the real picture (so its first frame can sit under the
//     caption), until a field is edited again (the edit wins until the next
//     run). Connected fields (a number node) always win.
function lastSize(node) {
    let linkedAny = false;
    const [w, h] = PREVIEW_SIZE_WIDGETS.map((name) => {
        const linked = linkedValue(node, name, "number");
        if (linked !== null) linkedAny = true;
        const value = linked !== null ? linked : findWidget(node, name)?.value;
        return Number.isFinite(value) ? Math.round(value) : 0;
    });
    const fields = w >= 16 && h >= 16 && w <= 65536 && h <= 65536 ? [w, h] : null;
    const s = node.properties?.[SIZE_PROP];
    const valid = Array.isArray(s) && s.length === 2 && s.every((v) => Number.isFinite(v) && v >= 16 && v <= 65536);
    const run = valid && !runIsStale(node) ? s : null;
    if (run && !node._mykeeMemeFieldsWin && !linkedAny) return run;
    // not run yet with this picture: the connected file's own size
    const media = node._mykeeMemeMedia;
    if (media && !node._mykeeMemeFieldsWin && !linkedAny) return [media.width, media.height];
    return fields || run || DEFAULT_SIZE;
}

// The text of a connected node, when it is a plain value (Primitive, text
// nodes with a string widget). null = use the widget's own text.
function linkedValue(node, name, type = "string") {
    try {
        const input = node.inputs?.find((i) => i.name === name || i.widget?.name === name);
        if (!input || input.link == null) return null;
        const link = node.graph?.links?.get ? node.graph.links.get(input.link) : node.graph?.links?.[input.link];
        const origin = link && node.graph?.getNodeById?.(link.origin_id);
        const widget = origin?.widgets?.find((w) => typeof w.value === type && w.type !== "button");
        return typeof widget?.value === type ? widget.value : null;
    } catch {
        return null;
    }
}

const linkedText = (node, name) => linkedValue(node, name, "string");

function collectParams(node) {
    const params = {};
    for (const key of PARAM_KEYS) {
        const w = findWidget(node, key);
        if (!w) continue;
        let value = w.value;
        if (TEXT_WIDGETS.includes(key)) {
            const linked = linkedText(node, key);
            if (linked !== null) value = linked;
        }
        params[key] = value;
    }
    return params;
}

function previewHeight(node, width) {
    const ratio = node._mykeeMemeRatio || DEFAULT_SIZE[1] / DEFAULT_SIZE[0];
    const h = Math.round(Math.max(40, (width ?? node.size[0]) - 30) * ratio);
    return Math.min(PREVIEW_MAX_HEIGHT, Math.max(PREVIEW_MIN_HEIGHT, h)) + PREVIEW_STATUS_HEIGHT;
}

// Sizes the stage to the picture's shape (as large as the preview area allows) and
// puts the background (the last frame, or gray) under the caption layer.
function layoutStage(node) {
    const stage = node._mykeeMemeStage;
    const shape = node._mykeeMemeShape;
    if (!stage || !shape) return;
    const availW = Math.max(40, node.size[0] - 30);
    // The room the frontend really gives the preview (it can be less than computeSize said):
    // the stage must stay inside it, or it runs over the status line below.
    const planned = previewHeight(node, node.size[0]) - PREVIEW_STATUS_HEIGHT;
    const measured = node._mykeeMemeHolder?.clientHeight || 0;
    const availH = Math.max(20, (measured > 20 ? Math.min(measured, planned) : planned) - 4);
    const w = Math.min(availW, (availH * shape.totalW) / shape.total);
    stage.style.width = `${Math.round(w)}px`;
    stage.style.height = `${Math.round((w * shape.total) / shape.totalW)}px`;
    const pic = node._mykeeMemePic;
    pic.style.left = `${(shape.left / shape.totalW) * 100}%`;
    pic.style.width = `${(shape.width / shape.totalW) * 100}%`;
    pic.style.top = `${(shape.top / shape.total) * 100}%`;
    pic.style.height = `${(shape.height / shape.total) * 100}%`;
    // Under the caption: the first frame of the last run, else the connected file itself, else gray.
    const fits = (m) => !!m && Math.abs(m.width / m.height - shape.width / shape.height) < 0.02;
    const thumb = !runIsStale(node) && fits(node._mykeeMemeThumb) ? node._mykeeMemeThumb : null;
    const media = !thumb && !node._mykeeMemeBoth && fits(node._mykeeMemeMedia) ? node._mykeeMemeMedia : null;
    const bgVideo = node._mykeeMemeVideo;
    if (bgVideo) bgVideo.style.display = media?.kind === "video" ? "block" : "none";
    const still = thumb || (media?.kind === "image" ? media : null);
    pic.style.background = still ? `url("${still.url}") 0 0 / 100% 100% no-repeat` : PREVIEW_GRAY;
}

function setStatus(node, text, isError) {
    const status = node._mykeeMemeStatus;
    if (!status) return;
    status.textContent = text;
    status.style.color = isError ? "#ff8a80" : "#d8b45a";
    status.title = text;
}

async function renderPreview(node) {
    const token = (node._mykeeMemeToken = (node._mykeeMemeToken || 0) + 1);
    const [width, height] = lastSize(node);
    try {
        const res = await api.fetchApi("/mykee/meme/preview", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ params: collectParams(node), width, height }),
        });
        let data = {};
        try {
            data = await res.json();
        } catch {
            data = {};
        }
        if (token !== node._mykeeMemeToken) return; // a newer request is on its way
        if (!res.ok || !data.png) {
            setStatus(node, data.error || "the preview could not be rendered", true);
            return;
        }
        const totalW = data.total_w ?? data.width;
        const left = data.img_x ?? 0;
        const top = data.img_y ?? data.top_band;
        const nums = [data.width, data.height, top, left, data.total_h, totalW];
        if (!nums.every((v) => Number.isFinite(v) && v >= 0) || !(data.total_h > 0) || !(data.width > 0) || !(totalW > 0)) {
            setStatus(node, "the preview could not be rendered", true);
            return;
        }
        node._mykeeMemeImg.src = "data:image/png;base64," + data.png;
        node._mykeeMemeShape = { width: data.width, height: data.height, top, left, total: data.total_h, totalW };
        const ratio = data.total_h / totalW;
        const changed = Math.abs(ratio - (node._mykeeMemeRatio || 0)) > 0.002;
        node._mykeeMemeRatio = ratio;
        if (changed) fitHeight(node);
        layoutStage(node);
        const warnings = [];
        if (node._mykeeMemeBoth) warnings.push(BOTH_MESSAGE);
        if (data.fallback) warnings.push(`Font not found on this system - using '${data.font}'.`);
        if (data.missing) warnings.push(`'${data.font}' has no glyph for: ${data.missing}`);
        // a caption that had to be shrunk (to its line limit / the frame): say what it ended up at
        ["top", "bottom"].forEach((side, i) => {
            const used = Array.isArray(data.sizes) ? data.sizes[i] : null;
            const set = Number(findWidget(node, `${side}_font_size`)?.value);
            if (Number.isFinite(used) && Number.isFinite(set) && used < set - 0.05) warnings.push(`${side} text shrunk to ${used}% (set ${set}%).`);
        });
        node._mykeeMemeWarn = warnings.join(" ");
        setStatus(node, node._mykeeMemeWarn, false);
    } catch {
        if (token === node._mykeeMemeToken) setStatus(node, "the preview could not be rendered", true);
    }
}

function schedulePreview(node, delay = 250) {
    clearTimeout(node._mykeeMemeTimer);
    node._mykeeMemeTimer = setTimeout(() => renderPreview(node), delay);
}


// Widgets added after the first release. Workflows store widget values by
// position, so in a workflow saved before they existed every value after
// them would land one field too far. Such a workflow (no layout mark in its
// properties) is read with the old order, and the new widgets keep their
// default. Handles both ways frontends store the values (only the saved
// widgets, or every widget by index).
const LAYOUT_PROP = "mykee_meme_layout";
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
    if (node._mykeeMemeReady) return;
    node._mykeeMemeReady = true;

    // --- preview widget ----------------------------------------------------
    // The server renders only the caption layer (transparent PNG, with the
    // frame bands in frame mode); it is laid over a gray picture, or over the
    // first frame of the last run when there is one of the same shape.
    const pic = el("div");
    Object.assign(pic.style, { position: "absolute", left: "0", width: "100%", background: PREVIEW_GRAY });
    // the first frame of a connected video file (shown until the node has run)
    const bgVideo = el("video", { muted: true, preload: "auto", playsInline: true });
    Object.assign(bgVideo.style, { position: "absolute", left: "0", top: "0", width: "100%", height: "100%", objectFit: "fill", display: "none" });
    pic.append(bgVideo);
    node._mykeeMemeVideo = bgVideo;
    const overlay = el("img", { alt: "preview", draggable: false });
    Object.assign(overlay.style, { position: "absolute", left: "0", top: "0", width: "100%", height: "100%" });
    const stage = el("div", {}, [pic, overlay]);
    Object.assign(stage.style, {
        position: "relative", margin: "0 auto", borderRadius: "4px", overflow: "hidden",
        background: "repeating-conic-gradient(#2a2a2a 0% 25%, #333 0% 50%) 50% / 16px 16px",
    });
    const holder = el("div", {}, [stage]);
    Object.assign(holder.style, { flex: "1 1 0", minHeight: "0", width: "100%", overflow: "hidden" });
    node._mykeeMemeHolder = holder;
    if (typeof ResizeObserver !== "undefined") new ResizeObserver(() => layoutStage(node)).observe(holder);
    const status = el("div");
    Object.assign(status.style, {
        font: "11px sans-serif", height: `${PREVIEW_STATUS_HEIGHT}px`, lineHeight: `${PREVIEW_STATUS_HEIGHT}px`,
        overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
    });
    const box = el("div", {}, [holder, status]);
    Object.assign(box.style, { display: "flex", flexDirection: "column", width: "100%", height: "100%", boxSizing: "border-box" });
    node._mykeeMemeImg = overlay;
    node._mykeeMemePic = pic;
    node._mykeeMemeStage = stage;
    node._mykeeMemeStatus = status;
    const previewW = node.addDOMWidget("meme_preview", "MYKEE_MEME_PREVIEW", box, { serialize: false });
    previewW.serialize = false;
    previewW.computeSize = (width) => [width, previewHeight(node, width)];
    previewW.computeLayoutSize = () => {
        const h = previewHeight(node);
        return { minHeight: h, maxHeight: h, minWidth: 0 };
    };

    // --- font: hidden string widget + picker -------------------------------
    const fontW = findWidget(node, "font");
    setWidgetVisible(fontW, false);
    const pickerW = node.addWidget("combo", "font_picker", NO_FONT, (value) => {
        fontW.value = value === NO_FONT ? "" : value;
        schedulePreview(node, 50);
        node.setDirtyCanvas?.(true, true);
    }, { values: [NO_FONT], serialize: false });
    pickerW.serialize = false;
    pickerW.tooltip = "Pick one of the fonts installed on this system.";
    const syncPicker = () => {
        pickerW.value = fontW.value ? fontW.value : NO_FONT;
    };
    const fillFonts = (list) => {
        const names = Array.isArray(list?.fonts) ? list.fonts.filter((n) => typeof n === "string") : [];
        const keep = fontW.value && !names.includes(fontW.value) ? [fontW.value] : []; // a font from another machine stays visible
        pickerW.options.values = [NO_FONT, ...keep, ...names];
        syncPicker();
        node.setDirtyCanvas?.(true, true);
    };
    loadFonts().then(fillFonts);
    const reloadW = node.addWidget("button", "reload_fonts", "Reload font list", () => {
        loadFonts(true).then((list) => {
            fillFonts(list);
            schedulePreview(node, 50);
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
        Object.assign(picker.style, {
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
        // The element is 24px tall; the rest is a gap, so the next field doesn't sit on the swatch.
        swatchW.computeSize = (width) => [width, SWATCH_HEIGHT];
        swatchW.computeLayoutSize = () => ({ minHeight: SWATCH_HEIGHT, maxHeight: SWATCH_HEIGHT, minWidth: 0 });
        swatches[name] = { picker, colorW };
    }
    const syncSwatches = () => {
        for (const { picker, colorW } of Object.values(swatches)) picker.value = normalizeHex(colorW.value, picker.value);
    };

    // --- order: tab bar, preview, then the settings (swatch under its field) --
    const tabBar = createTabBar();
    node.widgets.push(tabBar);
    const byName = new Map(node.widgets.map((w) => [w.name, w]));
    const order = [...TEXT_FIELDS, ...PREVIEW_SIZE_WIDGETS, TAB_BAR_NAME, "meme_preview"];
    for (const w of node.widgets) {
        if (order.includes(w.name) || w.name.endsWith("_swatch") || w.name === "font_picker" || w.name === "reload_fonts") continue;
        order.push(w.name);
        if (w.name === "font") order.push("font_picker", "reload_fonts");
        if (COLOR_WIDGETS.includes(w.name)) order.push(`${w.name}_swatch`);
    }
    // the Text tab reads top caption settings first, then the bottom ones
    const textGroup = ["top_font_size", "top_align", "top_lines", "top_margin", "bottom_font_size", "bottom_align", "bottom_lines", "bottom_margin"];
    const rest = order.filter((n) => !textGroup.includes(n));
    rest.splice(rest.indexOf("meme_preview") + 1, 0, ...textGroup.filter((n) => byName.has(n)));
    order.splice(0, order.length, ...rest);
    const ordered = order.map((n) => byName.get(n)).filter(Boolean);
    for (const w of node.widgets) if (!ordered.includes(w)) ordered.push(w);
    node.widgets = ordered;

    // --- changes -> preview / layout ---------------------------------------
    for (const key of [...PARAM_KEYS, ...PREVIEW_SIZE_WIDGETS]) {
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
            if (key === "placement" || key === "frame_height_mode") relayout(node);
            if (PREVIEW_SIZE_WIDGETS.includes(key)) node._mykeeMemeFieldsWin = true; // an edit beats the last run's size
            schedulePreview(node);
            return r;
        };
        if (TEXT_WIDGETS.includes(key)) {
            for (const input of new Set([w.element, w.inputEl])) {
                input?.addEventListener?.("input", () => schedulePreview(node));
            }
        }
    }

    node._mykeeMemeSync = () => {
        syncPicker();
        syncSwatches();
        relayout(node);
        schedulePreview(node, 50);
    };
    node.properties = node.properties || {};
    node.properties[LAYOUT_PROP] = LAYOUT_VERSION;
    if (!TABS.some((t) => t.id === node.properties[TAB_PROP])) node.properties[TAB_PROP] = DEFAULT_TAB;
    node.setSize([Math.max(node.size[0], MIN_WIDTH), node.size[1]]);
    relayout(node);
    schedulePreview(node, 100);
    setTimeout(() => {
        relayout(node);
        refreshMedia(node);
    }, 0);
    // Another file picked in the loader, or a node muted / bypassed, gives this node no
    // event, so look regularly. A timer, not a draw hook: it must not depend on the canvas
    // redrawing this node. refreshMedia does nothing when the connected file is the same.
    node._mykeeMemePoll = setInterval(() => {
        if (node.graph) refreshMedia(node);
    }, 700);
}

app.registerExtension({
    name: "Mykee.MemeMaker",
    // A node whose result came from the cache is not executed again, so it gets no
    // onExecuted: the real picture (the last run's size and first frame) is still
    // the right one - drop an edited-size override and show it again.
    setup() {
        api.addEventListener("execution_cached", (event) => {
            try {
                const ids = new Set((event?.detail?.nodes ?? []).map(String));
                for (const node of app.graph?._nodes ?? []) {
                    if (node.comfyClass !== NODE_NAME || !ids.has(String(node.id)) || !node._mykeeMemeReady) continue;
                    if (!node._mykeeMemeFieldsWin) continue;
                    node._mykeeMemeFieldsWin = false;
                    schedulePreview(node, 50);
                }
            } catch {
                // the preview just stays as it is
            }
        });
    },
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
            migrateWidgetValues(this, arguments[0]);
            const resync = () => {
                this._mykeeMemeSync?.();
                refreshMedia(this);
            };
            resync();
            setTimeout(resync, 0);
            setTimeout(resync, 200); // widget values from the workflow are in by now
            return r;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
            if (this._mykeeMemeReady) {
                schedulePreview(this);
                if (!app.configuringGraph) setTimeout(() => refreshMedia(this), 0);
            }
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            const info = message?.mykee_meme?.[0];
            if (info && Number.isFinite(info.width) && Number.isFinite(info.height)) {
                this.properties = this.properties || {};
                this.properties[SIZE_PROP] = [info.width, info.height];
                this.properties[RUN_KEY_PROP] = this._mykeeMemeWantedKey ?? "";
                this._mykeeMemeFieldsWin = false; // the real picture is known again
                // the first input frame (a small JPEG the node sends back) becomes the preview's background
                const ok = typeof info.thumb === "string" && /^data:image\/jpeg;base64,[A-Za-z0-9+/=]+$/.test(info.thumb);
                this._mykeeMemeThumb = ok ? { url: info.thumb, width: info.width, height: info.height } : null;
                schedulePreview(this, 50);
            }
            return r;
        };

        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function () {
            const r = onResize ? onResize.apply(this, arguments) : undefined;
            if (this._mykeeMemeReady) layoutStage(this);
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
            clearTimeout(this._mykeeMemeTimer);
            clearInterval(this._mykeeMemePoll);
            this._mykeeMemeToken = -1;
            return onRemoved ? onRemoved.apply(this, arguments) : undefined;
        };
    },
});
