import { app } from "../../scripts/app.js";

/**
 * Mykee Color Background.
 *
 * The Python side declares "color" as a plain STRING widget (so it's
 * typable and can be right-click "Convert widget to input"'d to a STRING
 * socket, both out of the box). This extension only adds a second, purely
 * visual DOM widget under it: a native <input type="color"> swatch that
 * shows the current color and opens the browser's own picker on click.
 *
 * The two stay in sync in both directions:
 *   - picking a color in the swatch writes the hex into the "color" widget
 *   - typing/pasting a hex into the "color" field (or a workflow load)
 *     updates the swatch preview
 *
 * This is the same "hidden/real widget as source of truth + a friendlier
 * widget layered on top" approach already used elsewhere in this pack
 * (see mykee_stylegan.js) - it deliberately avoids replacing or hiding the
 * built-in text widget, unlike comfy_mtb's "Colored Image" node, whose
 * color widget is a bespoke button that bypasses the normal text field
 * entirely (so it can't be typed into or converted to an input).
 */

function findWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

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

app.registerExtension({
    name: "Mykee.ColorBackground",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MykeeColorBackground") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const colorWidget = findWidget(this, "color");
            if (!colorWidget || this._mykeeColorSwatchAdded) return ret;
            this._mykeeColorSwatchAdded = true;

            colorWidget.value = normalizeHex(colorWidget.value, "#FFFFFF");

            const picker = document.createElement("input");
            picker.type = "color";
            picker.value = colorWidget.value;
            picker.title = "Pick background color";
            Object.assign(picker.style, {
                width: "100%",
                height: "24px",
                padding: "0",
                border: "1px solid var(--border-color, #4a4a4a)",
                borderRadius: "4px",
                background: "transparent",
                cursor: "pointer",
            });

            const syncSwatchFromWidget = () => {
                picker.value = normalizeHex(colorWidget.value, picker.value);
            };

            picker.addEventListener("input", () => {
                colorWidget.value = picker.value.toUpperCase();
                if (typeof colorWidget.callback === "function") {
                    colorWidget.callback(colorWidget.value, app.canvas, this);
                }
                this.graph?.setDirtyCanvas(true, true);
            });

            // Keep the swatch preview in sync when the hex text is edited
            // by hand (or fed in via a converted STRING input's default).
            const origCallback = colorWidget.callback;
            colorWidget.callback = (value, ...rest) => {
                const hex = normalizeHex(value, picker.value);
                colorWidget.value = hex;
                picker.value = hex;
                return origCallback ? origCallback.call(this, hex, ...rest) : undefined;
            };

            const domWidget = this.addDOMWidget("color_swatch", "COLORSWATCH", picker, {
                serialize: false,
            });
            domWidget.serialize = false;
            domWidget.computeSize = (width) => [width, 28];

            syncSwatchFromWidget();

            const onConfigure = this.onConfigure;
            this.onConfigure = function () {
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                syncSwatchFromWidget();
                return r;
            };

            return ret;
        };
    },
});
