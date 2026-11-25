/*
 * Mykee - Sigmas Editor - frontend.
 *
 * The node shows an editable chart of a sigma schedule. Every point is one
 * sigma value; there are steps + 1 points, evenly spaced along the x axis
 * (x = step index), so the points ARE the steps.
 *
 *   - drag a point up / down to change its value (it always stays between its
 *     neighbours, so the curve keeps decreasing),
 *   - double-click a point to delete it, double-click between two points to
 *     insert a new one there (steps follows),
 *   - type the values into the field under the chart (up to 6 decimals),
 *   - change `steps`: the current curve is resampled with a monotone cubic
 *     spline (PCHIP), so its shape is kept and it still decreases,
 *   - Generate: build a curve from a formula (flow shift, flux base/max
 *     shift, Karras, exponential, linear) from the `start` sigma down to 0.
 *     `start` always shows the first point of the current curve, so Generate
 *     keeps the current start (e.g. 0.7 for img2img) unless you type another,
 *   - flux (base/max shift): the shift is computed from width / height like
 *     ComfyUI's ModelSamplingFlux node. width / height are node widgets (they
 *     can be connected) and are greyed out for the other curves. While such a
 *     curve is not edited after Generate it stays "auto": it follows width /
 *     height (also connected ones: the backend regenerates it at run time and
 *     sends the result back here). Any edit freezes the curve,
 *   - Load from input: after a run, copy the sigmas of the connected
 *     scheduler into the editor.
 *
 * The values are stored in the hidden `sigma_values` widget (a plain comma
 * separated list); the backend reads that. The chart colours come from the
 * ComfyUI theme (CSS variables) and follow a theme switch.
 *
 * The node's height always follows its content; the width and the height
 * the user dragged the value field to are kept (also after reload).
 */
import { app } from "../../scripts/app.js";

const NODE_NAME = "MykeeSigmasEditor";
const VALUES_WIDGET = "sigma_values";
const AUTO_WIDGET = "auto_flux";
const SIZE_WIDGETS = ["width", "height"];
const MAX_STEPS = 500;            // keep in sync with nodes/mykee_sigmas_editor.py
const DECIMALS = 6;
const EPS = 1e-6;                 // smallest gap between neighbouring points
const CHART_HEIGHT = 230;
const ROW_HEIGHT = 22;            // generator row, tool row
const MSG_HEIGHT = 16;
const GAP = 5;                    // flex gap between the panel's 5 parts
const PANEL_PADDING = 6;          // top 2 + bottom 4
const DOM_MARGIN = 4;             // margin the frontend keeps around the DOM widget (top and bottom)
const SLACK = 4;                  // rounding / border safety
const MIN_AREA_HEIGHT = 44;
const MAX_AREA_HEIGHT = 600;
const MIN_NODE_WIDTH = 430;
const HIT_RADIUS = 9;             // px, grabbing a point
const PAD = { left: 52, right: 14, top: 12, bottom: 24 };

const PROP_AREA = "mykee_sigma_area_height";
const PROP_GEN = "mykee_sigma_generator";
const PROP_INPUT = "mykee_input_sigmas";

// fields: which inputs of the generator row a curve uses (the rest is hidden).
const GENERATORS = {
    flow: { label: "flow shift", fields: ["shift", "start"] },
    flux: { label: "flux base/max", fields: ["base", "max", "start"] },
    karras: { label: "Karras", fields: ["rho", "start", "min"] },
    exponential: { label: "exponential", fields: ["start", "min"] },
    linear: { label: "linear", fields: ["start"] },
};

// ComfyUI ModelSamplingFlux: base_shift at 256 image tokens, max_shift at
// 4096, linear in between and beyond; the shift itself is e^mu.
// Keep in sync with flux_shift() in nodes/mykee_sigmas_editor.py.
function fluxShift(base, max, width, height) {
    const mm = (max - base) / (4096 - 256);
    const b = base - mm * 256;
    const mu = ((width * height) / 256) * mm + b;
    return Math.exp(Math.min(mu, 50));
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function widget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function round6(v) {
    return Math.round(v * 1e6) / 1e6;
}

function formatValues(values) {
    return values.map((v) => v.toFixed(DECIMALS)).join(", ");
}

function clampNum(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
}

// "1.0, 0.98 0.5;0" / "[1.0, 0.5, 0.0]" -> { values } or { error }
function parseValues(text) {
    const cleaned = String(text ?? "")
        .replace(/tensor\(/gi, " ")
        .replace(/[[\]()]/g, " ")
        .trim();
    if (!cleaned) return { error: "No values." };
    const parts = cleaned.split(/[\s,;]+/).filter(Boolean);
    if (parts.length < 2) return { error: "At least 2 values are needed (1 step)." };
    if (parts.length > MAX_STEPS + 1) return { error: `Too many values (max ${MAX_STEPS + 1}).` };
    const values = [];
    for (const part of parts) {
        const v = Number(part);
        if (!Number.isFinite(v)) return { error: `"${part.slice(0, 30)}" is not a number.` };
        if (v < 0) return { error: "Negative values are not allowed." };
        values.push(round6(v));
    }
    for (let i = 1; i < values.length; i++) {
        if (!(values[i] < values[i - 1])) {
            return { error: `Value ${i + 1} (${values[i]}) must be smaller than value ${i} (${values[i - 1]}).` };
        }
    }
    return { values };
}

// Makes a list strictly decreasing (EPS gaps) without going below 0.
function enforceDecreasing(values) {
    const out = values.map((v) => round6(Math.max(0, v)));
    for (let i = 1; i < out.length; i++) {
        if (out[i] > out[i - 1] - EPS) out[i] = round6(Math.max(0, out[i - 1] - EPS));
    }
    return out;
}

// Monotone cubic (Fritsch-Carlson / PCHIP) through (x[i], y[i]); keeps a
// decreasing curve decreasing.
function pchip(xs, ys) {
    const n = xs.length;
    const h = [], d = [];
    for (let i = 0; i < n - 1; i++) {
        h.push(xs[i + 1] - xs[i]);
        d.push((ys[i + 1] - ys[i]) / h[i]);
    }
    const m = new Array(n).fill(0);
    if (n === 2) {
        m[0] = m[1] = d[0];
    } else {
        for (let i = 1; i < n - 1; i++) {
            if (d[i - 1] * d[i] <= 0) m[i] = 0;
            else {
                const w1 = 2 * h[i] + h[i - 1];
                const w2 = h[i] + 2 * h[i - 1];
                m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i]);
            }
        }
        const end = (h0, h1, d0, d1) => {
            let v = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
            if (Math.sign(v) !== Math.sign(d0)) v = 0;
            else if (Math.sign(d0) !== Math.sign(d1) && Math.abs(v) > Math.abs(3 * d0)) v = 3 * d0;
            return v;
        };
        m[0] = end(h[0], h[1], d[0], d[1]);
        m[n - 1] = end(h[n - 2], h[n - 3], d[n - 2], d[n - 3]);
    }
    return (x) => {
        let i = 0;
        while (i < n - 2 && x > xs[i + 1]) i++;
        const t = clampNum((x - xs[i]) / h[i], 0, 1);
        const t2 = t * t, t3 = t2 * t;
        return (2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h[i] * m[i]
            + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h[i] * m[i + 1];
    };
}

// Same curve, new number of steps.
function resample(values, steps) {
    const m = values.length;
    if (m - 1 === steps) return values.slice();
    const xs = values.map((_, i) => i / (m - 1));
    const f = pchip(xs, values);
    const out = [];
    for (let j = 0; j <= steps; j++) out.push(j === 0 ? values[0] : j === steps ? values[m - 1] : f(j / steps));
    return enforceDecreasing(out);
}

// ---------------------------------------------------------------------------
// Generators (the usual scheduler formulas)
// ---------------------------------------------------------------------------

function generate(kind, steps, start, end, shift, rho) {
    const n = steps;
    const out = [];
    if (kind === "flow") {
        // Flow-model schedule: t runs evenly to 0, sigma = s*t / (1 + (s-1)*t).
        // t0 is chosen so that the first sigma equals `start` (start < 1 is
        // the same as the KSampler's denoise with the step count kept).
        const s = shift;
        const t0 = start / (s - (s - 1) * start);
        for (let i = 0; i <= n; i++) {
            const t = t0 * (1 - i / n);
            out.push((s * t) / (1 + (s - 1) * t));
        }
    } else if (kind === "karras") {
        // Karras et al. 2022: evenly spaced in sigma^(1/rho), then 0.
        const a = Math.pow(start, 1 / rho), b = Math.pow(end, 1 / rho);
        for (let i = 0; i < n; i++) {
            const r = n === 1 ? 0 : i / (n - 1);
            out.push(Math.pow(a + r * (b - a), rho));
        }
        out.push(0);
    } else if (kind === "exponential") {
        // Evenly spaced in log(sigma), then 0.
        const la = Math.log(start), lb = Math.log(end);
        for (let i = 0; i < n; i++) {
            const r = n === 1 ? 0 : i / (n - 1);
            out.push(Math.exp(la + r * (lb - la)));
        }
        out.push(0);
    } else {
        for (let i = 0; i <= n; i++) out.push(start * (1 - i / n));
    }
    out[out.length - 1] = 0;
    return enforceDecreasing(out);
}

// ---------------------------------------------------------------------------
// Theme colours
// ---------------------------------------------------------------------------

function cssVar(names, fallback) {
    const styles = [getComputedStyle(document.documentElement), getComputedStyle(document.body)];
    for (const name of names) {
        for (const st of styles) {
            const v = st.getPropertyValue(name).trim();
            if (v) return v;
        }
    }
    return fallback;
}

function themeColors() {
    return {
        bg: cssVar(["--comfy-input-bg"], "#222"),
        grid: cssVar(["--border-color"], "#444"),
        text: cssVar(["--descrip-text", "--input-text", "--fg-color"], "#999"),
        strong: cssVar(["--input-text", "--fg-color"], "#ddd"),
        accent: cssVar(["--p-primary-color", "--primary-color", "--p-primary-500"], "#4a9eff"),
        point: cssVar(["--comfy-menu-bg", "--bg-color"], "#333"),
        error: cssVar(["--error-text"], "#ff6b6b"),
    };
}

// Every open editor is redrawn when the theme (CSS variables) changes.
const liveEditors = new Set();
let themeObserver = null;
function watchTheme() {
    if (themeObserver || typeof MutationObserver === "undefined") return;
    let timer = null;
    themeObserver = new MutationObserver(() => {
        clearTimeout(timer);
        timer = setTimeout(() => liveEditors.forEach((ed) => ed.draw()), 50);
    });
    for (const el of [document.documentElement, document.body]) {
        themeObserver.observe(el, { attributes: true, attributeFilter: ["class", "style", "data-theme"] });
    }
    // ComfyUI palettes may also swap a <style> element in <head>.
    themeObserver.observe(document.head, { childList: true, subtree: true, characterData: true });
}

// ---------------------------------------------------------------------------
// Panel
// ---------------------------------------------------------------------------

const PANEL_CSS = `
.mykee-se { display:flex; flex-direction:column; gap:${GAP}px; width:100%; height:100%; box-sizing:border-box;
  padding:2px 4px 4px; font:12px sans-serif; color:var(--input-text, #ddd); overflow:hidden; }
.mykee-se canvas { width:100%; display:block; flex:0 0 auto; border-radius:4px; border:1px solid var(--border-color, #444);
  box-sizing:border-box; touch-action:none; }
.mykee-se-row { display:flex; align-items:center; gap:4px; height:${ROW_HEIGHT}px; min-width:0; flex:0 0 auto; }
.mykee-se-row label { opacity:.75; font-size:11px; white-space:nowrap; }
.mykee-se-row input, .mykee-se-row select { background:var(--comfy-input-bg, #222); color:inherit;
  border:1px solid var(--border-color, #444); border-radius:4px; padding:1px 4px; font:11px sans-serif;
  height:20px; box-sizing:border-box; min-width:0; }
.mykee-se-row input { flex:1 1 56px; width:56px; min-width:34px; }
.mykee-se-row select { cursor:pointer; flex:0 1 auto; max-width:120px; }
.mykee-se-row button { flex:0 0 auto; }
.mykee-se-row input:disabled { opacity:.4; }
.mykee-se button { background:var(--comfy-menu-bg, #333); color:inherit; border:1px solid var(--border-color, #555);
  border-radius:4px; padding:1px 8px; font:11px sans-serif; height:20px; cursor:pointer; white-space:nowrap; }
.mykee-se button:disabled { opacity:.4; cursor:default; }
.mykee-se-info { margin-left:auto; opacity:.7; font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.mykee-se textarea { width:100%; box-sizing:border-box; flex:0 0 auto; resize:vertical; min-height:${MIN_AREA_HEIGHT}px;
  max-height:${MAX_AREA_HEIGHT}px; background:var(--comfy-input-bg, #222); color:inherit;
  border:1px solid var(--border-color, #444); border-radius:4px; padding:3px 5px; font:11px monospace; }
.mykee-se textarea.bad { border-color:var(--error-text, #ff6b6b); }
.mykee-se-msg { height:${MSG_HEIGHT}px; line-height:${MSG_HEIGHT}px; flex:0 0 auto; font-size:11px; opacity:.8; white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis; }
.mykee-se-msg.bad { color:var(--error-text, #ff6b6b); opacity:1; }
`;

function ensureCss() {
    if (document.getElementById("mykee-se-css")) return;
    const style = document.createElement("style");
    style.id = "mykee-se-css";
    style.textContent = PANEL_CSS;
    document.head.appendChild(style);
}

function isolate(element) {
    for (const type of ["keydown", "keyup", "keypress", "wheel"]) {
        element.addEventListener(type, (e) => e.stopPropagation());
    }
    return element;
}

function areaHeight(node) {
    const h = Number(node.properties?.[PROP_AREA]);
    return Number.isFinite(h) ? clampNum(Math.round(h), MIN_AREA_HEIGHT, MAX_AREA_HEIGHT) : 64;
}

function panelHeight(node) {
    return CHART_HEIGHT + ROW_HEIGHT * 2 + areaHeight(node) + MSG_HEIGHT
        + GAP * 4 + PANEL_PADDING + DOM_MARGIN * 2 + SLACK;
}

function fitHeight(node) {
    const size = node.computeSize?.();
    if (!size || !Number.isFinite(size[1])) return;
    node.setSize([Math.max(node.size[0], size[0]), size[1]]);
    node.graph?.setDirtyCanvas?.(true, true);
}

// sigma_values and auto_flux only carry data: hide them and drop their sockets.
function hideValuesWidget(node) {
    for (const name of [VALUES_WIDGET, AUTO_WIDGET]) {
        const w = widget(node, name);
        if (w && w.type !== "hidden") {
            w.hidden = true;
            w.type = "hidden";
            w.computeSize = () => [0, -4];
            w.computeLayoutSize = () => ({ minHeight: 0, maxHeight: 0, minWidth: 0 });
        }
        const idx = node.inputs?.findIndex((inp) => inp.name === name) ?? -1;
        if (idx >= 0) {
            if (node.inputs[idx].link != null) node.disconnectInput?.(idx);
            node.removeInput(idx);
        }
    }
}

// Uses the graph's link table too: newer frontends only mirror links in
// input.link, and the mirror can lag behind.
function inputLinked(node, name) {
    const idx = node.inputs?.findIndex((inp) => inp.name === name) ?? -1;
    if (idx < 0) return false;
    if (node.inputs[idx].link != null) return true;
    const table = node.graph?.links;
    const links = table instanceof Map ? [...table.values()] : table ? Object.values(table) : [];
    return links.some((l) => l && String(l.target_id) === String(node.id) && l.target_slot === idx);
}

function readAuto(node) {
    const raw = widget(node, AUTO_WIDGET)?.value;
    if (typeof raw !== "string" || !raw.trim()) return null;
    try {
        const a = JSON.parse(raw);
        const ok = a && [a.base, a.max, a.start].every((v) => Number.isFinite(Number(v)));
        return ok ? { base: Number(a.base), max: Number(a.max), start: Number(a.start) } : null;
    } catch {
        return null;
    }
}

function writeAuto(node, auto) {
    const w = widget(node, AUTO_WIDGET);
    if (w) w.value = auto ? JSON.stringify({ base: auto.base, max: auto.max, start: auto.start }) : "";
}

function readGen(node) {
    const g = node.properties?.[PROP_GEN];
    const src = g && typeof g === "object" ? g : {};
    const num = (v, d, lo, hi) => (Number.isFinite(Number(v)) ? clampNum(Number(v), lo, hi) : d);
    return {
        kind: GENERATORS[src.kind] ? src.kind : "flow",
        shift: num(src.shift, 3.0, 0.01, 100),
        base: num(src.base, 0.5, 0, 100),
        max: num(src.max, 1.15, 0, 100),
        rho: num(src.rho, 7.0, 0.1, 100),
        start: num(src.start, 1.0, 0.000001, 1000),
        end: num(src.end, 0.002, 0.000001, 1000),
    };
}

function writeGen(node, gen) {
    node.properties = node.properties || {};
    node.properties[PROP_GEN] = { ...gen };
}

function storedInput(node) {
    const raw = node.properties?.[PROP_INPUT];
    if (!Array.isArray(raw) || raw.length < 2 || raw.length > MAX_STEPS + 1) return null;
    const values = raw.map(Number);
    return values.every((v) => Number.isFinite(v) && v >= 0) ? values : null;
}

class SigmaEditor {
    constructor(node) {
        this.node = node;
        this.values = [1, 0.75, 0.5, 0.25, 0];
        this.drag = -1;
        this.hover = -1;
        this.build();
        this.loadFromWidget();
    }

    // ---- data ---------------------------------------------------------------
    loadFromWidget() {
        const parsed = parseValues(widget(this.node, VALUES_WIDGET)?.value);
        if (parsed.values) this.values = parsed.values;
        this.commit({ syncSteps: true, keepText: false, keepAuto: true });
        this.syncSizeWidgets();
    }

    // Writes the values everywhere: hidden widget, steps, text field, chart.
    // Any change that is not an auto regeneration ends the "auto" state.
    commit({ syncSteps = true, keepText = false, keepAuto = false } = {}) {
        if (!keepAuto) writeAuto(this.node, null);
        const w = widget(this.node, VALUES_WIDGET);
        if (w) w.value = formatValues(this.values);
        if (syncSteps) {
            const sw = widget(this.node, "steps");
            if (sw && sw.value !== this.values.length - 1) sw.value = this.values.length - 1;
        }
        if (!keepText) {
            this.area.value = formatValues(this.values);
            this.area.classList.remove("bad");
        }
        this.setMessage("");
        if (this.startInput && document.activeElement !== this.startInput) {
            this.startInput.value = String(this.values[0]);
            if (!this.loading) {
                const gen = readGen(this.node);
                gen.start = this.values[0] > 0 ? this.values[0] : gen.start;
                writeGen(this.node, gen);
            }
        }
        this.updateInfo();
        this.draw();
        this.node.graph?.setDirtyCanvas?.(true, false);
    }

    setValues(values, opts) {
        this.values = values;
        this.commit(opts);
    }

    setMessage(text, bad = false) {
        this.msg.textContent = text || "Drag a point · double-click: insert / delete · type the values below";
        this.msg.classList.toggle("bad", !!bad);
    }

    updateInfo() {
        const steps = this.values.length - 1;
        let text = `${steps} step${steps === 1 ? "" : "s"} · start ${this.values[0].toFixed(DECIMALS)}`;
        const auto = readAuto(this.node);
        if (auto) {
            const linked = SIZE_WIDGETS.some((n) => inputLinked(this.node, n));
            const last = this.lastAuto;
            if (linked) {
                text += last ? ` · auto: ${last[0]}×${last[1]}, shift ${last[2].toFixed(2)}` : " · auto: size from input at run";
            } else {
                const [wd, ht] = this.size2();
                text += ` · auto: shift ${fluxShift(auto.base, auto.max, wd, ht).toFixed(2)}`;
            }
        }
        this.info.textContent = text;
        this.info.title = auto
            ? "This flux curve follows width / height until you edit it."
            : "";
        const input = storedInput(this.node);
        this.loadBtn.disabled = !input;
        this.loadBtn.textContent = input ? `Load from input (${input.length - 1})` : "Load from input";
        this.loadBtn.title = input
            ? "Copy the sigmas that arrived on the sigmas input in the last run into the editor."
            : "Connect a scheduler to the sigmas input and run the workflow once.";
    }

    // ---- DOM ----------------------------------------------------------------
    build() {
        ensureCss();
        const node = this.node;
        const panel = document.createElement("div");
        panel.className = "mykee-se";

        this.canvas = document.createElement("canvas");
        this.canvas.style.height = `${CHART_HEIGHT}px`;
        this.ctx = this.canvas.getContext("2d");

        // Generator row.
        const gen = readGen(node);
        const genRow = document.createElement("div");
        genRow.className = "mykee-se-row";
        const kind = isolate(document.createElement("select"));
        for (const [key, g] of Object.entries(GENERATORS)) {
            const opt = document.createElement("option");
            opt.value = key;
            opt.textContent = g.label;
            kind.append(opt);
        }
        kind.value = gen.kind;
        this.kindSelect = kind;
        const FIELD_INFO = {
            shift: ["shift", "Flow shift: higher = the curve stays longer at high noise (as the model's own shift)."],
            base: ["base", "base_shift of ModelSamplingFlux (shift at 256 image tokens, e.g. 0.5)."],
            max: ["max", "max_shift of ModelSamplingFlux (shift at 4096 image tokens, e.g. 1.15). The shift is computed from width / height."],
            rho: ["rho", "Karras rho (7 is the usual value)."],
            start: ["start", "First sigma of the generated curve. Shows the first point of the current curve, so Generate keeps it unless you type another (lower = more of an input image is kept)."],
            min: ["min", "Smallest sigma before the final 0 (Karras / exponential)."],
        };
        const inputs = {};
        const labels = {};
        genRow.append(kind);
        for (const [key, [text, title]] of Object.entries(FIELD_INFO)) {
            const label = document.createElement("label");
            label.textContent = text;
            label.title = title;
            const input = isolate(document.createElement("input"));
            input.type = "number";
            input.step = "any";
            input.title = title;
            inputs[key] = input;
            labels[key] = label;
            genRow.append(label, input);
        }
        inputs.shift.value = String(gen.shift);
        inputs.base.value = String(gen.base);
        inputs.max.value = String(gen.max);
        inputs.rho.value = String(gen.rho);
        inputs.min.value = String(gen.end);
        inputs.start.value = String(this.values[0]);
        this.startInput = inputs.start;
        const genBtn = document.createElement("button");
        genBtn.textContent = "Generate";
        genBtn.title = "Replace the curve with the chosen formula (current steps, from the start value).";
        genRow.append(genBtn);

        const syncGenFields = () => {
            const used = GENERATORS[kind.value].fields;
            for (const key of Object.keys(FIELD_INFO)) {
                const show = used.includes(key);
                labels[key].style.display = show ? "" : "none";
                inputs[key].style.display = show ? "" : "none";
            }
            this.syncSizeWidgets();
        };
        const saveGen = () => {
            const cur = readGen(node);
            cur.kind = kind.value;
            const num = (key) => Number(inputs[key].value);
            for (const [key, prop] of [["shift", "shift"], ["base", "base"], ["max", "max"], ["rho", "rho"],
                ["min", "end"], ["start", "start"]]) {
                if (inputs[key].value !== "" && Number.isFinite(num(key)) && num(key) > 0) cur[prop] = num(key);
            }
            writeGen(node, readGen({ properties: { [PROP_GEN]: cur } }));
        };
        // Put the saved generator settings (stored in the workflow) back into
        // the fields. Called after a workflow load, when node.properties has
        // been restored - the fields were filled with defaults before that.
        this.restoreGen = () => {
            const g = readGen(node);
            kind.value = g.kind;
            inputs.shift.value = String(g.shift);
            inputs.base.value = String(g.base);
            inputs.max.value = String(g.max);
            inputs.rho.value = String(g.rho);
            inputs.min.value = String(g.end);
            const saved = node.properties?.[PROP_GEN];
            inputs.start.value = String(saved && Number(saved.start) > 0 ? g.start : this.values[0]);
            syncGenFields();
        };
        kind.addEventListener("change", () => { saveGen(); syncGenFields(); });
        for (const input of Object.values(inputs)) {
            input.addEventListener("change", saveGen);
            input.addEventListener("input", saveGen);
        }
        inputs.start.addEventListener("blur", () => {
            // An empty / invalid start goes back to the first point.
            if (!(Number(inputs.start.value) > 0)) inputs.start.value = String(this.values[0]);
        });
        genBtn.addEventListener("click", () => {
            saveGen();
            const g = readGen(node);
            const steps = this.values.length - 1;
            let start = Number(inputs.start.value);
            if (!(start > 0) || !Number.isFinite(start)) {
                this.setMessage("start must be a number above 0.", true);
                return;
            }
            start = round6(Math.min(start, 1000));
            if ((g.kind === "flow" || g.kind === "flux") && start > 1) start = 1; // flow sigmas live in 0..1
            if (GENERATORS[g.kind].fields.includes("min") && !(g.end < start)) {
                this.setMessage("min must be smaller than start.", true);
                return;
            }
            if (g.kind === "flux") {
                this.setFlux({ base: g.base, max: g.max, start }, steps);
            } else {
                this.setValues(generate(g.kind, steps, start, g.end, g.shift, g.rho));
            }
            this.raiseSigmaMax(this.values[0]);
        });
        syncGenFields();

        // Tool row.
        const toolRow = document.createElement("div");
        toolRow.className = "mykee-se-row";
        this.loadBtn = document.createElement("button");
        this.loadBtn.addEventListener("click", () => {
            const input = storedInput(node);
            if (!input) return;
            this.setValues(enforceDecreasing(input));
            this.raiseSigmaMax(this.values[0]);
        });
        const copyBtn = document.createElement("button");
        copyBtn.textContent = "Copy";
        copyBtn.title = "Copy the values to the clipboard.";
        copyBtn.addEventListener("click", () => {
            navigator.clipboard?.writeText(formatValues(this.values)).then(
                () => this.setMessage("Copied to the clipboard."),
                () => this.setMessage("Copy failed (clipboard not available).", true),
            );
        });
        this.info = document.createElement("div");
        this.info.className = "mykee-se-info";
        toolRow.append(this.loadBtn, copyBtn, this.info);

        // Value field.
        this.area = isolate(document.createElement("textarea"));
        this.area.spellcheck = false;
        this.area.style.height = `${areaHeight(node)}px`;
        this.area.title = "Sigma values, separated by commas or spaces (up to 6 decimals). Each must be smaller than the one before.";
        let typingTimer = null;
        const applyTyped = (final) => {
            const parsed = parseValues(this.area.value);
            if (parsed.values) {
                this.area.classList.remove("bad");
                if (formatValues(parsed.values) === formatValues(this.values)) {
                    if (final) this.area.value = formatValues(this.values);
                    this.setMessage("");
                    return;
                }
                this.setValues(parsed.values, { keepText: !final });
                this.raiseSigmaMax(this.values[0]);
            } else if (final) {
                // Leaving the field with an invalid text: back to the curve.
                this.area.value = formatValues(this.values);
                this.area.classList.remove("bad");
                this.setMessage(`Not applied: ${parsed.error}`, true);
            } else {
                this.area.classList.add("bad");
                this.setMessage(parsed.error, true);
            }
        };
        this.area.addEventListener("input", () => {
            clearTimeout(typingTimer);
            typingTimer = setTimeout(() => applyTyped(false), 400);
        });
        this.area.addEventListener("blur", () => {
            clearTimeout(typingTimer);
            applyTyped(true);
        });
        if (typeof ResizeObserver !== "undefined") {
            // A drag on the resize handle sets an inline height; remember it.
            new ResizeObserver(() => {
                if (!this.area.style.height || !this.area.offsetHeight) return;
                const h = clampNum(Math.round(this.area.offsetHeight), MIN_AREA_HEIGHT, MAX_AREA_HEIGHT);
                if (Math.abs(h - areaHeight(node)) <= 1) return;
                node.properties = node.properties || {};
                node.properties[PROP_AREA] = h;
                clearTimeout(this.fitTimer);
                this.fitTimer = setTimeout(() => fitHeight(node), 30);
            }).observe(this.area);
            // The chart follows the node width.
            new ResizeObserver(() => this.draw()).observe(panel);
        }

        this.msg = document.createElement("div");
        this.msg.className = "mykee-se-msg";

        panel.append(this.canvas, genRow, toolRow, this.area, this.msg);
        this.panel = panel;

        const dom = node.addDOMWidget("sigma_editor", "MYKEE_SIGMA_EDITOR", panel, {
            serialize: false,
            hideOnZoom: false,
            margin: DOM_MARGIN,
            getMinHeight: () => panelHeight(node),
            getMaxHeight: () => panelHeight(node),
        });
        dom.serialize = false;

        this.bindPointer();
    }

    // Width / height as the editor sees them (the widgets' own values).
    size2() {
        const v = (n) => clampNum(Math.round(Number(widget(this.node, n)?.value)) || 1024, 16, 16384);
        return [v("width"), v("height")];
    }

    // A flux curve from the current width / height, kept "auto".
    setFlux(auto, steps) {
        const [wd, ht] = this.size2();
        const shift = fluxShift(auto.base, auto.max, wd, ht);
        writeAuto(this.node, auto);
        this.setValues(generate("flow", steps, auto.start, 0, shift, 7), { keepAuto: true });
    }

    // Re-run an auto flux curve after steps / width / height changed.
    refreshAuto(steps) {
        const auto = readAuto(this.node);
        if (!auto) return false;
        this.setFlux(auto, steps ?? this.values.length - 1);
        return true;
    }

    // width / height are only used by the flux curve: grey them out otherwise.
    syncSizeWidgets() {
        const flux = this.kindSelect?.value === "flux";
        let changed = false;
        for (const name of SIZE_WIDGETS) {
            const w = widget(this.node, name);
            if (w && !!w.disabled !== !flux) {
                w.disabled = !flux;
                changed = true;
            }
        }
        if (changed) this.node.graph?.setDirtyCanvas?.(true, true);
    }

    raiseSigmaMax(top) {
        const w = widget(this.node, "sigma_max");
        if (w && top > Number(w.value)) {
            w.value = Math.ceil(top * 100) / 100;
            this.draw();
        }
    }

    // ---- geometry -----------------------------------------------------------
    size() {
        const width = Math.max(50, this.canvas.clientWidth || this.panel.clientWidth || 300);
        return { width, height: CHART_HEIGHT };
    }

    axisTop() {
        const max = Number(widget(this.node, "sigma_max")?.value) || 1;
        return Math.max(max, this.values[0] || 0, 1e-6);
    }

    logView() {
        return !!widget(this.node, "log_view")?.value;
    }

    logFloor() {
        const positive = this.values.filter((v) => v > 0);
        const minPos = positive.length ? Math.min(...positive) : this.axisTop();
        return Math.min(this.axisTop() * 1e-3, minPos * 0.5);
    }

    xOf(i) {
        const { width } = this.size();
        const n = this.values.length - 1;
        return PAD.left + ((width - PAD.left - PAD.right) * i) / Math.max(1, n);
    }

    yOf(v) {
        const top = this.axisTop();
        const h = CHART_HEIGHT - PAD.top - PAD.bottom;
        let r;
        if (this.logView()) {
            const lo = Math.log10(this.logFloor()), hi = Math.log10(top);
            r = (Math.log10(Math.max(v, this.logFloor())) - lo) / (hi - lo);
        } else {
            r = v / top;
        }
        return PAD.top + h * (1 - clampNum(r, 0, 1));
    }

    valueAt(y) {
        const top = this.axisTop();
        const h = CHART_HEIGHT - PAD.top - PAD.bottom;
        const r = clampNum(1 - (y - PAD.top) / h, 0, 1);
        if (this.logView()) {
            if (r <= 0) return 0;
            const lo = Math.log10(this.logFloor()), hi = Math.log10(top);
            return Math.pow(10, lo + r * (hi - lo));
        }
        return r * top;
    }

    // Pointer position in the chart's own pixels (works at any canvas zoom).
    localPos(e) {
        const rect = this.canvas.getBoundingClientRect();
        const { width } = this.size();
        return {
            x: ((e.clientX - rect.left) * width) / (rect.width || 1),
            y: ((e.clientY - rect.top) * CHART_HEIGHT) / (rect.height || 1),
        };
    }

    hit(pos) {
        let best = -1, bestD = HIT_RADIUS * HIT_RADIUS;
        this.values.forEach((v, i) => {
            const dx = this.xOf(i) - pos.x, dy = this.yOf(v) - pos.y;
            const d = dx * dx + dy * dy;
            if (d <= bestD) { bestD = d; best = i; }
        });
        return best;
    }

    // ---- interaction --------------------------------------------------------
    bindPointer() {
        const c = this.canvas;
        c.addEventListener("pointerdown", (e) => {
            e.stopPropagation();
            if (e.button !== 0) return;
            const idx = this.hit(this.localPos(e));
            if (idx < 0) return;
            this.drag = idx;
            c.setPointerCapture?.(e.pointerId);
            e.preventDefault();
        });
        c.addEventListener("pointermove", (e) => {
            const pos = this.localPos(e);
            if (this.drag >= 0) {
                e.stopPropagation();
                const i = this.drag;
                const upper = i === 0 ? this.axisTop() : this.values[i - 1] - EPS;
                const lower = i === this.values.length - 1 ? 0 : this.values[i + 1] + EPS;
                if (upper < lower) return;
                this.values[i] = round6(clampNum(this.valueAt(pos.y), lower, upper));
                this.commit({ syncSteps: false });
                return;
            }
            const idx = this.hit(pos);
            if (idx !== this.hover) {
                this.hover = idx;
                c.style.cursor = idx >= 0 ? "ns-resize" : "default";
                this.draw();
            }
        });
        const endDrag = (e) => {
            if (this.drag < 0) return;
            this.drag = -1;
            c.releasePointerCapture?.(e.pointerId);
            this.draw();
        };
        c.addEventListener("pointerup", endDrag);
        c.addEventListener("pointercancel", endDrag);
        c.addEventListener("pointerleave", () => {
            if (this.drag < 0 && this.hover >= 0) {
                this.hover = -1;
                this.draw();
            }
        });
        c.addEventListener("dblclick", (e) => {
            e.stopPropagation();
            e.preventDefault();
            const pos = this.localPos(e);
            const idx = this.hit(pos);
            if (idx >= 0) {
                if (this.values.length <= 2) {
                    this.setMessage("At least 2 points (1 step) are needed.", true);
                    return;
                }
                const values = this.values.slice();
                values.splice(idx, 1);
                this.hover = -1;
                this.setValues(values);
                return;
            }
            if (this.values.length - 1 >= MAX_STEPS) {
                this.setMessage(`At most ${MAX_STEPS} steps.`, true);
                return;
            }
            // Insert between the two points around the click.
            const n = this.values.length - 1;
            const { width } = this.size();
            const u = (pos.x - PAD.left) / Math.max(1, width - PAD.left - PAD.right);
            if (u <= 0 || u >= 1) return;
            const left = Math.min(n - 1, Math.floor(u * n));
            const a = this.values[left], b = this.values[left + 1];
            if (a - b < 2 * EPS) {
                this.setMessage("No room for a new point there.", true);
                return;
            }
            const mid = a > 0 && b > 0 ? Math.sqrt(a * b) : (a + b) / 2;
            const values = this.values.slice();
            values.splice(left + 1, 0, round6(clampNum(mid, b + EPS, a - EPS)));
            this.setValues(values);
        });
    }

    // ---- drawing ------------------------------------------------------------
    draw() {
        const { width, height } = this.size();
        const dpr = window.devicePixelRatio || 1;
        const c = this.canvas;
        if (c.width !== Math.round(width * dpr) || c.height !== Math.round(height * dpr)) {
            c.width = Math.round(width * dpr);
            c.height = Math.round(height * dpr);
        }
        const ctx = this.ctx;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const col = themeColors();

        ctx.fillStyle = col.bg;
        ctx.fillRect(0, 0, width, height);
        ctx.font = "10px sans-serif";
        ctx.lineWidth = 1;

        // Horizontal grid + y labels.
        const top = this.axisTop();
        const ticks = [];
        if (this.logView()) {
            const lo = Math.ceil(Math.log10(this.logFloor())), hi = Math.floor(Math.log10(top));
            for (let p = lo; p <= hi; p++) ticks.push(Math.pow(10, p));
            ticks.push(top);
        } else {
            for (let k = 0; k <= 4; k++) ticks.push((top * k) / 4);
        }
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        for (const t of ticks) {
            const y = Math.round(this.yOf(t)) + 0.5;
            ctx.strokeStyle = col.grid;
            ctx.globalAlpha = 0.6;
            ctx.beginPath();
            ctx.moveTo(PAD.left, y);
            ctx.lineTo(width - PAD.right, y);
            ctx.stroke();
            ctx.globalAlpha = 1;
            ctx.fillStyle = col.text;
            const label = t === 0 ? "0" : t >= 100 ? t.toFixed(0) : t >= 1 ? t.toFixed(2) : t.toPrecision(2);
            ctx.fillText(label, PAD.left - 6, y);
        }

        // x labels (step indexes), thinned out when crowded.
        const n = this.values.length - 1;
        const span = (width - PAD.left - PAD.right) / Math.max(1, n);
        const every = Math.max(1, Math.ceil(22 / Math.max(span, 1)));
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillStyle = col.text;
        for (let i = 0; i <= n; i++) {
            if (i % every && i !== n) continue;
            ctx.fillText(String(i), this.xOf(i), CHART_HEIGHT - PAD.bottom + 6);
        }

        // Curve.
        ctx.strokeStyle = col.accent;
        ctx.lineWidth = 2;
        ctx.beginPath();
        this.values.forEach((v, i) => {
            const x = this.xOf(i), y = this.yOf(v);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Points (smaller when crowded).
        const r = span < 8 ? 2 : span < 16 ? 3 : 4.5;
        this.values.forEach((v, i) => {
            const active = i === this.drag || i === this.hover;
            ctx.beginPath();
            ctx.arc(this.xOf(i), this.yOf(v), active ? r + 2 : r, 0, Math.PI * 2);
            ctx.fillStyle = active ? col.accent : col.point;
            ctx.fill();
            ctx.lineWidth = 1.5;
            ctx.strokeStyle = col.accent;
            ctx.stroke();
        });

        // Tooltip for the point under the pointer / being dragged.
        const show = this.drag >= 0 ? this.drag : this.hover;
        if (show >= 0) {
            const text = `#${show}  ${this.values[show].toFixed(DECIMALS)}`;
            ctx.font = "11px monospace";
            const tw = ctx.measureText(text).width + 10;
            let x = this.xOf(show) + 8, y = this.yOf(this.values[show]) - 24;
            if (x + tw > width - 2) x = this.xOf(show) - 8 - tw;
            if (y < 2) y = this.yOf(this.values[show]) + 8;
            ctx.fillStyle = col.point;
            ctx.globalAlpha = 0.92;
            ctx.fillRect(x, y, tw, 18);
            ctx.globalAlpha = 1;
            ctx.strokeStyle = col.grid;
            ctx.lineWidth = 1;
            ctx.strokeRect(x + 0.5, y + 0.5, tw - 1, 17);
            ctx.fillStyle = col.strong;
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.fillText(text, x + 5, y + 9);
        }
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

app.registerExtension({
    name: "Mykee.SigmasEditor",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated?.apply(this, arguments);
            hideValuesWidget(this);
            this._mykeeSigma = new SigmaEditor(this);
            liveEditors.add(this._mykeeSigma);
            watchTheme();

            // steps: resample the current curve to the new number of steps.
            const stepsWidget = widget(this, "steps");
            if (stepsWidget) {
                const callback = stepsWidget.callback;
                stepsWidget.callback = (...args) => {
                    const r = callback?.apply(stepsWidget, args);
                    const ed = this._mykeeSigma;
                    const steps = clampNum(Math.round(Number(stepsWidget.value)) || 1, 1, MAX_STEPS);
                    if (ed && steps !== ed.values.length - 1 && !ed.refreshAuto(steps)) {
                        ed.setValues(resample(ed.values, steps));
                    }
                    return r;
                };
            }
            for (const name of ["sigma_max", "log_view"]) {
                const w = widget(this, name);
                if (!w) continue;
                const callback = w.callback;
                w.callback = (...args) => {
                    const r = callback?.apply(w, args);
                    this._mykeeSigma?.draw();
                    return r;
                };
            }

            for (const name of SIZE_WIDGETS) {
                const w = widget(this, name);
                if (!w) continue;
                const callback = w.callback;
                w.callback = (...args) => {
                    const r = callback?.apply(w, args);
                    const ed = this._mykeeSigma;
                    if (ed && !ed.refreshAuto()) ed.updateInfo();
                    return r;
                };
            }

            this.setSize([Math.max(this.size[0], MIN_NODE_WIDTH), this.size[1]]);
            // ComfyUI may still apply its own size after onNodeCreated - fit again.
            setTimeout(() => {
                fitHeight(this);
                this._mykeeSigma?.draw();
            }, 0);
            return ret;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const ret = onConfigure?.apply(this, arguments);
            // Widget values and properties are restored by now: the stored
            // values win over the steps widget.
            setTimeout(() => {
                hideValuesWidget(this);
                const ed = this._mykeeSigma;
                if (ed) {
                    ed.area.style.height = `${areaHeight(this)}px`;
                    ed.loading = true;
                    ed.loadFromWidget();
                    ed.loading = false;
                    ed.restoreGen?.();
                }
                fitHeight(this);
            }, 50);
            return ret;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const ed = this._mykeeSigma;
            const autoValues = message?.auto_values;
            const autoInfo = message?.auto_info;
            if (ed && readAuto(this) && Array.isArray(autoValues) && autoValues.length >= 2
                && autoValues.length <= MAX_STEPS + 1) {
                const values = autoValues.map(Number);
                if (values.every((v) => Number.isFinite(v) && v >= 0)) {
                    if (Array.isArray(autoInfo) && autoInfo.length === 3 && autoInfo.every((v) => Number.isFinite(Number(v)))) {
                        ed.lastAuto = autoInfo.map(Number);
                    }
                    ed.setValues(enforceDecreasing(values), { keepAuto: true });
                }
            }
            const input = message?.input_sigmas;
            if (!Array.isArray(input) || input.length < 2 || input.length > MAX_STEPS + 1) return;
            const values = input.map(Number);
            if (!values.every((v) => Number.isFinite(v) && v >= 0)) return;
            this.properties = this.properties || {};
            this.properties[PROP_INPUT] = values;
            this._mykeeSigma?.updateInfo();
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const ret = onConnectionsChange?.apply(this, arguments);
            setTimeout(() => this._mykeeSigma?.updateInfo(), 0);
            return ret;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            if (this._mykeeSigma) liveEditors.delete(this._mykeeSigma);
            return onRemoved?.apply(this, arguments);
        };
    },
});
