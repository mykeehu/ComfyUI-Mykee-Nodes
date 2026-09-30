import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * Mykee Voice/Accent Match - status widget
 *
 * Adds a small, read-only status line right under the node's title,
 * showing the same progress text this node prints to the console
 * ("Separating vocals...", "Loading Seed-VC model...", "Converting
 * voice...", "Mixing...", "Done") - this pipeline's stages can each take
 * a while (model downloads, diffusion steps), so a glance at the node
 * itself is more useful than having to check the console.
 *
 * The backend sends updates over the "mykee-voice-match-status" websocket
 * message (see _send_status() in mykee_voice_match.py), keyed by this
 * node's own unique_id so multiple instances of this node in the same
 * graph don't cross-update each other's status.
 */

const NODE_NAME = "MykeeVoiceAccentMatch";

app.registerExtension({
    name: "Mykee.VoiceAccentMatch.Status",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const statusWidget = {
                name: "mykee_status",
                type: "mykee_status_display",
                value: "Idle",
                computeSize: function (width) {
                    return [width, 20];
                },
                draw: function (ctx, node, widgetWidth, y, height) {
                    ctx.save();
                    ctx.fillStyle = "#9d9d9d";
                    ctx.font = "12px Arial";
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    ctx.fillText(this.value ? String(this.value) : "Idle", widgetWidth / 2, y + height / 2);
                    ctx.restore();
                },
            };

            this.addCustomWidget(statusWidget);
            // Move it from the end (where addCustomWidget appends it) to
            // the very top of the widget list, right under the title.
            const idx = this.widgets.indexOf(statusWidget);
            if (idx > 0) {
                this.widgets.splice(idx, 1);
                this.widgets.unshift(statusWidget);
            }

            this._mykeeStatusWidget = statusWidget;
            return ret;
        };
    },
    async setup() {
        api.addEventListener("mykee-voice-match-status", (event) => {
            const detail = event.detail || {};
            const nodeId = detail.node;
            const text = detail.text;
            if (nodeId === undefined || nodeId === null) return;

            const graphNode =
                app.graph.getNodeById(Number(nodeId)) ?? app.graph.getNodeById(nodeId);
            if (graphNode && graphNode._mykeeStatusWidget) {
                graphNode._mykeeStatusWidget.value = text;
                graphNode.setDirtyCanvas(true, true);
            }
        });
    },
});
