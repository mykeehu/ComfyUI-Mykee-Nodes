import { app } from "../../scripts/app.js";

/**
 * Mykee Group Bypasser
 *
 * A virtual node that shows one on/off switch per Group present in the
 * current graph. Flipping a switch sets every node inside that group to
 * ALWAYS (0) or BYPASS (4).
 *
 * `LiteGraph` is used here as an ambient global (window.LiteGraph), the
 * same way most ComfyUI custom-node JS extensions do it - the frontend
 * exposes it on window for exactly this kind of backward compatibility.
 * All colors are read live from it every draw, so the node automatically
 * follows whatever color palette (dark/light/custom) is active.
 *
 * Sort modes (node.properties.sortMode):
 *  - "position":  top-to-bottom (then left-to-right) by group bounding box.
 *  - "execution": groups are ranked by the earliest node (by ComfyUI's
 *                 own execution/topological order) they contain.
 *  - "manual":    fully free order, drag-and-drop via the handle on the
 *                 left of each row (only shown in this mode).
 *
 * For "position" and "execution", a group nested inside another group is
 * always placed immediately after its (closest) parent group. "manual"
 * mode ignores nesting - it's a flat, fully free list.
 *
 * Whenever the row order changes (auto re-sort or a manual drop), the
 * rows that moved slide smoothly into their new spot instead of jumping.
 *
 * A copy-icon button in the title bar copies every currently active
 * (non-bypassed) group listed here - and every node inside each of
 * them - to the clipboard, using ComfyUI's own copy/paste format, so
 * they can be pasted straight into another workflow.
 *
 * Each row also passively re-syncs its ON/OFF state (every ~15 draw
 * frames, same cadence as the group-list poll) from that group's actual
 * current bypass state - so toggling a group's own header switch, or
 * another Bypasser node (ours or a different pack's) changing node modes
 * directly, shows up here too, not just changes made from this node.
 */

const NODE_NAME = "MykeeGroupBypasser";
const MODE_ALWAYS = 0;
const MODE_BYPASS = 4;
const ROW_H = 40;
const ROW_TYPE = "MYKEE_GROUP_TOGGLE";
const REORDER_ANIM_MS = 220;
const SWITCH_ON_COLOR = "#89B4E0";
const LABEL_SWITCH_GAP = 16;
const MIN_NODE_WIDTH = 200;
const LABEL_FONT = "13px Arial";
const COPY_ICON = "\uE957"; // PrimeIcons pi-copy
const CHECK_ICON = "\uE909"; // PrimeIcons pi-check

const SORT_LABELS = {
    position: "Position (top to bottom)",
    execution: "Execution order",
    manual: "Manual order",
};
const SORT_KEYS = Object.keys(SORT_LABELS);

// ---------------------------------------------------------------------
// Geometry / graph helpers
// ---------------------------------------------------------------------

function groupBounding(group) {
    return group.boundingRect || group._bounding;
}

function centerInRect(rect, x, y) {
    return x >= rect[0] && x <= rect[0] + rect[2] && y >= rect[1] && y <= rect[1] + rect[3];
}

function rectFullyContains(outer, inner) {
    return (
        outer[0] <= inner[0] &&
        outer[1] <= inner[1] &&
        outer[0] + outer[2] >= inner[0] + inner[2] &&
        outer[1] + outer[3] >= inner[1] + inner[3]
    );
}

function nodesInGroup(graph, group) {
    const b = groupBounding(group);
    const result = [];
    for (const n of graph.nodes) {
        const nb = n.boundingRect;
        if (!nb) continue;
        const cx = nb[0] + nb[2] / 2;
        const cy = nb[1] + nb[3] / 2;
        if (centerInRect(b, cx, cy)) result.push(n);
    }
    return result;
}

function setGroupBypassed(graph, group, bypass) {
    for (const n of nodesInGroup(graph, group)) {
        n.mode = bypass ? MODE_BYPASS : MODE_ALWAYS;
    }
}

function isGroupBypassed(graph, group) {
    const nodes = nodesInGroup(graph, group);
    if (nodes.length === 0) return false;
    return nodes.every((n) => n.mode === MODE_BYPASS);
}

// Immediate-parent map: for each group, the smallest-area group that
// fully contains it (its closest ancestor), if any.
function buildHierarchy(groups) {
    const bounds = new Map(groups.map((g) => [g, groupBounding(g)]));
    const area = (b) => b[2] * b[3];

    const parentOf = new Map();
    for (const g of groups) {
        let best = null;
        let bestArea = Infinity;
        for (const other of groups) {
            if (other === g) continue;
            if (rectFullyContains(bounds.get(other), bounds.get(g))) {
                const a = area(bounds.get(other));
                if (a < bestArea) {
                    bestArea = a;
                    best = other;
                }
            }
        }
        parentOf.set(g, best);
    }

    const childrenOf = new Map(groups.map((g) => [g, []]));
    const roots = [];
    for (const g of groups) {
        const p = parentOf.get(g);
        if (p) childrenOf.get(p).push(g);
        else roots.push(g);
    }
    return { roots, childrenOf };
}

function orderGroups(graph, sortMode, manualOrder) {
    const groups = graph.groups || [];
    if (groups.length === 0) return [];

    if (sortMode === "manual") {
        const idx = new Map((manualOrder || []).map((t, i) => [t, i]));
        return [...groups].sort((a, b) => {
            const ia = idx.has(a.title) ? idx.get(a.title) : Number.MAX_SAFE_INTEGER;
            const ib = idx.has(b.title) ? idx.get(b.title) : Number.MAX_SAFE_INTEGER;
            if (ia !== ib) return ia - ib;
            return (a.title || "").localeCompare(b.title || "");
        });
    }

    let execIndex = null;
    if (sortMode === "execution") {
        const order = graph.computeExecutionOrder(false);
        execIndex = new Map(order.map((n, i) => [n.id, i]));
    }

    const rank = (g) => {
        if (sortMode === "execution") {
            let minIdx = Infinity;
            for (const n of nodesInGroup(graph, g)) {
                const i = execIndex.get(n.id);
                if (i !== undefined && i < minIdx) minIdx = i;
            }
            return [minIdx === Infinity ? Number.MAX_SAFE_INTEGER : minIdx];
        }
        // position: top-to-bottom, then left-to-right
        const b = groupBounding(g);
        return [b[1], b[0]];
    };

    const { roots, childrenOf } = buildHierarchy(groups);
    const sortSiblings = (list) =>
        [...list].sort((a, b) => {
            const ra = rank(a);
            const rb = rank(b);
            for (let i = 0; i < Math.max(ra.length, rb.length); i++) {
                const d = (ra[i] ?? 0) - (rb[i] ?? 0);
                if (d) return d;
            }
            return (a.title || "").localeCompare(b.title || "");
        });

    const result = [];
    const visit = (list) => {
        for (const g of sortSiblings(list)) {
            result.push(g);
            visit(childrenOf.get(g) || []);
        }
    };
    visit(roots);
    return result;
}

// ---------------------------------------------------------------------
// Small drawing helpers
// ---------------------------------------------------------------------

function hexToRgb(hex) {
    const norm = hex.length === 4 ? hex.replace(/^#?([a-f\d])([a-f\d])([a-f\d])$/i, "#$1$1$2$2$3$3") : hex;
    const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(norm);
    if (!m) return [136, 136, 136];
    return [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)];
}

function lerpColor(hexA, hexB, t) {
    const a = hexToRgb(hexA);
    const b = hexToRgb(hexB);
    const r = Math.round(a[0] + (b[0] - a[0]) * t);
    const g = Math.round(a[1] + (b[1] - a[1]) * t);
    const bl = Math.round(a[2] + (b[2] - a[2]) * t);
    return `rgb(${r}, ${g}, ${bl})`;
}

function easeOutCubic(t) {
    return 1 - Math.pow(1 - t, 3);
}

function roundedRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, w, h, r);
    else ctx.rect(x, y, w, h);
}

// ---------------------------------------------------------------------
// The per-group row widget
// ---------------------------------------------------------------------

function layoutMetrics(widgetWidth, manual) {
    const margin = 15;
    const rowX = margin;
    const rowW = widgetWidth - margin * 2;
    const handleW = manual ? 26 : 0;
    const switchW = 38;
    const switchH = 18;
    const padRight = 14;
    const textX = rowX + (manual ? handleW + 10 : 14);
    const switchX = rowX + rowW - padRight - switchW;
    return { margin, rowX, rowW, handleW, switchW, switchH, padRight, textX, switchX };
}

let _measureCtx = null;
function measureLabelWidth(text) {
    if (!_measureCtx) _measureCtx = document.createElement("canvas").getContext("2d");
    _measureCtx.font = LABEL_FONT;
    return _measureCtx.measureText(text || "").width;
}

// Width the node needs so the widest group label sits comfortably next
// to the switch, with no wasted space and no truncation (unless a
// label is absurdly long).
function computeRequiredWidth(groups, manual) {
    const { margin, handleW } = layoutMetrics(0, manual);
    const leftOffset = margin + (manual ? handleW + 10 : 14);
    const rightSide = LABEL_SWITCH_GAP + 38 /* switchW */ + 14 /* padRight */ + margin;
    let maxLabelW = 0;
    for (const g of groups) {
        const w = measureLabelWidth(g.title || "(unnamed group)");
        if (w > maxLabelW) maxLabelW = w;
    }
    return Math.max(MIN_NODE_WIDTH, Math.ceil(leftOffset + maxLabelW + rightSide));
}

function makeGroupToggleWidget(group) {
    const widget = {
        type: ROW_TYPE,
        name: `group::${group.title}`,
        value: true,
        options: {},
        serialize: false,
        groupRef: group,
        _switchVal: undefined,
        _animOffset: 0,
        _animStart: undefined,
        computeSize(width) {
            return [width, ROW_H];
        },
        draw(ctx, node, widgetWidth, y, _H, lowQuality) {
            const LG = window.LiteGraph || {};
            const manual = node.properties.sortMode === "manual";
            const dragging = node._mykeeDragState && node._mykeeDragState.widget === widget;
            const { rowX, rowW, switchW, switchH, textX, switchX } = layoutMetrics(widgetWidth, manual);

            // Row-reorder settle animation (FLIP): ease any pending
            // vertical offset back to zero.
            let offset = 0;
            if (widget._animStart !== undefined) {
                const t = Math.min(1, (performance.now() - widget._animStart) / REORDER_ANIM_MS);
                offset = widget._animOffset * (1 - easeOutCubic(t));
                if (t >= 1) {
                    widget._animStart = undefined;
                    widget._animOffset = 0;
                } else {
                    node.graph?.setDirtyCanvas(true, false);
                }
            }

            const pillH = ROW_H - 12;
            const pillY = y + (ROW_H - pillH) / 2 + offset;

            ctx.save();
            if (dragging) ctx.globalAlpha = 0.32;

            // Row background - same shape/colors as ComfyUI's own widgets.
            ctx.fillStyle = LG.WIDGET_BGCOLOR || "#222";
            roundedRect(ctx, rowX, pillY, rowW, pillH, pillH * 0.4);
            ctx.fill();
            if (!lowQuality) {
                ctx.strokeStyle = LG.WIDGET_OUTLINE_COLOR || "#666";
                ctx.lineWidth = 1;
                ctx.stroke();
            }

            // Drag handle (grip icon) - manual mode only.
            if (manual && !lowQuality) {
                ctx.fillStyle = LG.WIDGET_SECONDARY_TEXT_COLOR || "#999";
                const hx = rowX + 10;
                const hy = pillY + pillH / 2;
                for (let i = -1; i <= 1; i++) {
                    ctx.beginPath();
                    ctx.arc(hx, hy + i * 5, 1.6, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.beginPath();
                    ctx.arc(hx + 6, hy + i * 5, 1.6, 0, Math.PI * 2);
                    ctx.fill();
                }
            }

            // Label.
            if (!lowQuality) {
                ctx.fillStyle = LG.WIDGET_TEXT_COLOR || "#DDD";
                ctx.font = LABEL_FONT;
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                const maxTextW = Math.max(10, switchX - LABEL_SWITCH_GAP - textX);
                let label = group.title || "(unnamed group)";
                while (ctx.measureText(label).width > maxTextW && label.length > 1) {
                    label = label.slice(0, -1);
                }
                if (label !== (group.title || "(unnamed group)")) label += "…";
                ctx.fillText(label, textX, pillY + pillH / 2 + 1);
            }

            // On/off switch, with an animated sliding knob + color fade.
            if (widget._switchVal === undefined) widget._switchVal = widget.value ? 1 : 0;
            const target = widget.value ? 1 : 0;
            if (Math.abs(widget._switchVal - target) > 0.003) {
                widget._switchVal += (target - widget._switchVal) * 0.35;
                node.graph?.setDirtyCanvas(true, false);
            } else {
                widget._switchVal = target;
            }
            const sv = widget._switchVal;

            const swY = pillY + pillH / 2 - switchH / 2;
            const offColor = LG.WIDGET_OUTLINE_COLOR || "#666";
            ctx.fillStyle = lerpColor(offColor, SWITCH_ON_COLOR, sv);
            roundedRect(ctx, switchX, swY, switchW, switchH, switchH / 2);
            ctx.fill();

            const knobR = switchH / 2 - 2;
            const knobX = switchX + switchH / 2 + sv * (switchW - switchH);
            const knobY = swY + switchH / 2;
            ctx.beginPath();
            ctx.fillStyle = "#fff";
            ctx.shadowColor = "rgba(0,0,0,0.35)";
            ctx.shadowBlur = 2;
            ctx.arc(knobX, knobY, knobR, 0, Math.PI * 2);
            ctx.fill();

            ctx.restore();
        },
        mouse(event, pos, node) {
            const manual = node.properties.sortMode === "manual";
            const [x] = pos;
            const { rowX, handleW } = layoutMetrics(node.size[0], manual);

            if (event.type === "pointerdown") {
                if (manual && x >= rowX && x <= rowX + handleW) {
                    node._mykeeDragState = {
                        widget,
                        startY: event.canvasY,
                        lastCanvasY: event.canvasY,
                        startIndex: node._mykeeToggleWidgets.indexOf(widget),
                    };
                    node._mykeeUpdateCursor(true);
                    node.graph.setDirtyCanvas(true, true);
                    return true;
                }
                widget.value = !widget.value;
                setGroupBypassed(node.graph, group, !widget.value);
                node.graph.setDirtyCanvas(true, true);
                return true;
            }

            if (event.type === "pointermove") {
                const state = node._mykeeDragState;
                if (state && state.widget === widget) {
                    state.lastCanvasY = event.canvasY;
                    node.graph.setDirtyCanvas(true, true);
                    return true;
                }
            }

            if (event.type === "pointerup") {
                const state = node._mykeeDragState;
                if (state && state.widget === widget) {
                    node._mykeeCommitDrag(state, event);
                    return true;
                }
            }
            return false;
        },
    };
    return widget;
}

// ---------------------------------------------------------------------
// Node lifecycle
// ---------------------------------------------------------------------

app.registerExtension({
    name: "Mykee.GroupBypasser",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        nodeType.prototype.isVirtualNode = true;

        nodeType.prototype._mykeeComputeSignature = function () {
            const groups = (this.graph && this.graph.groups) || [];
            return groups
                .map((g) => {
                    const b = groupBounding(g);
                    return `${g.title}#${Math.round(b[0])},${Math.round(b[1])},${Math.round(b[2])},${Math.round(b[3])}`;
                })
                .join("|");
        };

        // Rebuilds the toggle-row list from the current graph state.
        // Existing rows are re-used (matched by group reference) so
        // their values, switch-animation and identity survive a
        // rebuild - only the *positions* change, which is exactly what
        // drives the settle animation below.
        nodeType.prototype._mykeeRebuild = function () {
            if (!this.graph) return;
            const graph = this.graph;
            const ordered = orderGroups(graph, this.properties.sortMode, this.properties.manualOrder);

            const prevWidgets = this._mykeeToggleWidgets || [];
            const prevIndexOf = new Map(prevWidgets.map((w, i) => [w, i]));
            const byGroup = new Map(prevWidgets.map((w) => [w.groupRef, w]));

            const newToggleWidgets = ordered.map((g) => {
                let w = byGroup.get(g);
                if (!w) w = makeGroupToggleWidget(g);
                w.value = !isGroupBypassed(graph, g);
                return w;
            });

            const now = performance.now();
            for (const [i, w] of newToggleWidgets.entries()) {
                const prevI = prevIndexOf.get(w);
                if (prevI === undefined || prevI === i) continue;
                w._animOffset = (prevI - i) * ROW_H;
                w._animStart = now;
            }

            this._mykeeToggleWidgets = newToggleWidgets;
            this.widgets = [this._sortWidget, ...newToggleWidgets];
            this._mykeeGroupSignature = this._mykeeComputeSignature();
            this._sortWidget.value = SORT_LABELS[this.properties.sortMode];

            const autoSize = this.computeSize();
            const requiredWidth = computeRequiredWidth(ordered, this.properties.sortMode === "manual");
            this.setSize([requiredWidth, autoSize[1]]);
            graph.setDirtyCanvas(true, true);
        };

        // Cheap re-sync of each row's ON/OFF value from the actual current
        // bypass state of its group's nodes - independent of the full
        // rebuild above, which only fires when the *set* of groups or
        // their geometry changes. Without this, flipping a group's own
        // header toggle, or another Bypasser node (ours or anyone else's)
        // changing a node's mode directly, would never be reflected here:
        // the row would keep showing whatever it last knew, since nothing
        // about the group's title/position changed to trigger a rebuild.
        nodeType.prototype._mykeeSyncToggleValues = function () {
            if (!this.graph) return false;
            let changed = false;
            for (const w of this._mykeeToggleWidgets || []) {
                const shouldBeOn = !isGroupBypassed(this.graph, w.groupRef);
                if (w.value !== shouldBeOn) {
                    w.value = shouldBeOn;
                    changed = true;
                }
            }
            return changed;
        };

        // Finishes a manual drag: works out the drop index, updates
        // manualOrder, then lets _mykeeRebuild() re-derive the order and
        // animate every row that moved into its new spot.
        nodeType.prototype._mykeeCommitDrag = function (state, event) {
            const rows = this._mykeeToggleWidgets;
            const deltaRows = Math.round((event.canvasY - state.startY) / ROW_H);
            const targetIndex = Math.max(0, Math.min(rows.length - 1, state.startIndex + deltaRows));
            const curIndex = rows.indexOf(state.widget);

            if (targetIndex !== curIndex) {
                const newTitles = rows.map((w) => w.groupRef.title);
                newTitles.splice(curIndex, 1);
                newTitles.splice(targetIndex, 0, state.widget.groupRef.title);
                this.properties.manualOrder = newTitles;
            }

            this._mykeeDragState = null;
            this._mykeeUpdateCursor(false);
            this._mykeeRebuild();
        };

        nodeType.prototype._mykeeUpdateCursor = function (forceGrabbing) {
            const canvasEl = app.canvas && app.canvas.canvas;
            if (!canvasEl) return;
            if (forceGrabbing) {
                canvasEl.style.cursor = "grabbing";
                this._mykeeCursorSet = true;
                return;
            }
            if (this.properties.sortMode !== "manual" || !app.canvas.graph_mouse) {
                if (this._mykeeCursorSet) {
                    canvasEl.style.cursor = "";
                    this._mykeeCursorSet = false;
                }
                return;
            }
            const m = app.canvas.graph_mouse;
            const localX = m[0] - this.pos[0];
            const localY = m[1] - this.pos[1];
            const { rowX, handleW } = layoutMetrics(this.size[0], true);
            let hovering = false;
            for (const w of this._mykeeToggleWidgets || []) {
                if (localY >= w.y && localY <= w.y + ROW_H && localX >= rowX && localX <= rowX + handleW) {
                    hovering = true;
                    break;
                }
            }
            if (hovering) {
                canvasEl.style.cursor = "grab";
                this._mykeeCursorSet = true;
            } else if (this._mykeeCursorSet) {
                canvasEl.style.cursor = "";
                this._mykeeCursorSet = false;
            }
        };

        // Copies every *active* (non-bypassed) group this node lists -
        // plus every node inside each of them - to the clipboard, using
        // ComfyUI's own clipboard format, so a normal Ctrl+V (in this
        // browser, in any tab/workflow, or - best effort - even a
        // different window) pastes them exactly like a native
        // multi-select copy would.
        nodeType.prototype._mykeeCopyToClipboard = function () {
            if (!this.graph || !app.canvas || typeof app.canvas.copyToClipboard !== "function") return;
            const activeWidgets = (this._mykeeToggleWidgets || []).filter((w) => w.value);
            if (activeWidgets.length === 0) return;

            const items = new Set();
            for (const w of activeWidgets) {
                const g = w.groupRef;
                items.add(g);
                for (const n of nodesInGroup(this.graph, g)) items.add(n);
            }

            const serializedData = app.canvas.copyToClipboard(items);
            this._mykeeFlashCopyButton();

            // Best effort: also put the same payload on the real OS
            // clipboard, in the exact "text/html" shape ComfyUI's own
            // Ctrl+C uses, so a plain Ctrl+V works outside this
            // browser's local-storage clipboard too. Silently skipped
            // if the browser/context doesn't allow it - the litegraph-
            // internal copy above already succeeded either way.
            try {
                if (serializedData && navigator.clipboard && window.ClipboardItem) {
                    const bytes = new TextEncoder().encode(serializedData);
                    const chunkSize = 0x8000;
                    const chunks = [];
                    for (let offset = 0; offset < bytes.length; offset += chunkSize) {
                        chunks.push(String.fromCharCode(...bytes.subarray(offset, offset + chunkSize)));
                    }
                    const b64 = btoa(chunks.join(""));
                    const html = `<meta charset="utf-8"><div><span data-metadata="${b64}"></span></div><span style="white-space:pre-wrap;">Text</span>`;
                    const blob = new Blob([html], { type: "text/html" });
                    navigator.clipboard.write([new ClipboardItem({ "text/html": blob })]).catch(() => {});
                }
            } catch (err) {
                // Non-fatal.
            }
        };

        nodeType.prototype._mykeeFlashCopyButton = function () {
            const button = this._mykeeCopyButton;
            if (!button) return;
            button.text = CHECK_ICON;
            this.graph?.setDirtyCanvas(true, true);
            clearTimeout(this._mykeeCopyFlashTimer);
            this._mykeeCopyFlashTimer = setTimeout(() => {
                button.text = COPY_ICON;
                this.graph?.setDirtyCanvas(true, true);
            }, 700);
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            this.properties = this.properties || {};
            if (!SORT_KEYS.includes(this.properties.sortMode)) {
                this.properties.sortMode = "position";
            }
            if (!Array.isArray(this.properties.manualOrder)) {
                this.properties.manualOrder = [];
            }

            if (!this.title || this.title === NODE_NAME) {
                this.title = "Mykee Group Bypasser";
            }

            this._mykeeDragState = null;
            this._mykeeToggleWidgets = [];
            this._mykeeFrame = 0;
            this._mykeeCursorSet = false;

            this._mykeeCopyButton = this.addTitleButton({
                name: "mykee_copy",
                text: COPY_ICON,
                fontSize: 20,
                height: 24,
            });

            const self = this;
            this._sortWidget = this.addWidget(
                "combo",
                "sort",
                SORT_LABELS[this.properties.sortMode],
                (value) => {
                    const mode = SORT_KEYS.find((k) => SORT_LABELS[k] === value) || "position";
                    self.properties.sortMode = mode;
                    if (mode === "manual" && self.properties.manualOrder.length === 0) {
                        self.properties.manualOrder = (self._mykeeToggleWidgets || []).map((w) => w.groupRef.title);
                    }
                    self._mykeeRebuild();
                },
                { values: Object.values(SORT_LABELS) }
            );
            this._sortWidget.serialize = false;

            this._mykeeRebuild();

            return ret;
        };

        const onTitleButtonClick = nodeType.prototype.onTitleButtonClick;
        nodeType.prototype.onTitleButtonClick = function (button, canvas) {
            if (button.name === "mykee_copy") {
                this._mykeeCopyToClipboard();
                return;
            }
            onTitleButtonClick?.apply(this, arguments);
        };


        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            onDrawForeground?.apply(this, arguments);
            if (!this.graph || this.flags?.collapsed) return;

            const state = this._mykeeDragState;

            if (state) {
                const rows = this._mykeeToggleWidgets;
                const deltaRows = Math.round((state.lastCanvasY - state.startY) / ROW_H);
                const targetIndex = Math.max(0, Math.min(rows.length - 1, state.startIndex + deltaRows));
                const curIndex = rows.indexOf(state.widget);

                // Insertion-line indicator: shows exactly where the row
                // would land if released right now.
                if (targetIndex !== curIndex) {
                    const targetWidget = rows[targetIndex];
                    if (targetWidget) {
                        const above = targetIndex < curIndex;
                        const lineY = above ? targetWidget.y - 1 : targetWidget.y + ROW_H - 3;
                        ctx.save();
                        ctx.strokeStyle = SWITCH_ON_COLOR;
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.moveTo(15, lineY);
                        ctx.lineTo(this.size[0] - 15, lineY);
                        ctx.stroke();
                        ctx.restore();
                    }
                }

                // Floating copy of the dragged row, following the
                // cursor, drawn last so it's on top of every other row.
                const followY = state.widget.y + (state.lastCanvasY - state.startY);
                ctx.save();
                ctx.shadowColor = "rgba(0,0,0,0.45)";
                ctx.shadowBlur = 10;
                state.widget.draw(ctx, this, this.size[0], followY, ROW_H, false);
                ctx.restore();

                // Cursor + continuous redraw while dragging.
                this._mykeeUpdateCursor(true);
                this.graph.setDirtyCanvas(true, false);
                return;
            }

            this._mykeeUpdateCursor(false);

            // Keep polling for hover (cursor) + group changes while in
            // manual mode or every ~15 frames otherwise - cheap enough,
            // avoids hooking every possible graph-mutation event.
            if (this.properties.sortMode === "manual") {
                this.graph.setDirtyCanvas(true, false);
            }
            this._mykeeFrame = (this._mykeeFrame + 1) % 15;
            if (this._mykeeFrame !== 0) return;
            if (this._mykeeComputeSignature() !== this._mykeeGroupSignature) {
                this._mykeeRebuild();
            } else if (this._mykeeSyncToggleValues()) {
                this.graph.setDirtyCanvas(true, false);
            }
        };

        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            getExtraMenuOptions?.apply(this, arguments);
            options.push(
                null,
                {
                    content: "Refresh",
                    callback: () => this._mykeeRebuild(),
                }
            );
            if (this.properties.sortMode === "manual") {
                options.push({
                    content: "Clear manual order (revert to Position)",
                    callback: () => {
                        this.properties.manualOrder = [];
                        this.properties.sortMode = "position";
                        this._mykeeRebuild();
                    },
                });
            }
        };
    },
});
