import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const NODE_NAME = "MykeePromptAdvanced";
const DISPLAY_NAME = "response_display";

app.registerExtension({
    name: "Mykee.PromptAdvanced",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            // Purely a client-side display mirror - see the matching note
            // in mykee_prompt_modifier.js/.py for why this is no longer a
            // declared backend input at all (a plain w.serialize = false
            // on a real input wasn't reliable enough on its own to stop
            // this node rerunning on every queue).
            const { widget: w } = ComfyWidgets["STRING"](
                this, DISPLAY_NAME, ["STRING", { multiline: true, default: "" }], app
            );
            w.serialize = false;
            if (w.inputEl) {
                w.inputEl.readOnly = true;
                w.inputEl.placeholder = "(filled in after running)";
                w.inputEl.style.opacity = "0.75";
            }

            return ret;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const value = message?.[DISPLAY_NAME]?.[0];
            if (value === undefined) return;
            const w = this.widgets?.find((w) => w.name === DISPLAY_NAME);
            if (w) {
                w.value = value;
                if (w.inputEl) {
                    w.inputEl.value = value;
                    w.inputEl.readOnly = true;
                }
            }
            app.graph.setDirtyCanvas(true, true);
        };
    },
});
