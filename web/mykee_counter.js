import { app } from "../../scripts/app.js";

const MAX_INPUTS = 50;
const NODE_NAMES = ["MykeeCounter", "MykeeCounterSeedAdvanced"];

app.registerExtension({
    name: "Mykee.Counter",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!NODE_NAMES.includes(nodeData.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const numInputsWidget = this.widgets?.find((w) => w.name === "num_inputs");

            // Adds/removes the signal_N input sockets based on the
            // "num_inputs" widget's value. Never deletes a connected
            // socket, so a connection is never accidentally lost.
            const syncInputs = (count) => {
                count = Math.max(1, Math.min(MAX_INPUTS, Math.round(count)));

                for (let i = 1; i <= MAX_INPUTS; i++) {
                    const name = `signal_${i}`;
                    const idx = this.inputs?.findIndex((inp) => inp.name === name) ?? -1;

                    if (i <= count) {
                        if (idx === -1) {
                            this.addInput(name, "*");
                        }
                    } else if (idx !== -1) {
                        const input = this.inputs[idx];
                        if (!input.link) {
                            this.removeInput(idx);
                        }
                    }
                }

                this.setSize(this.computeSize());
                app.graph.setDirtyCanvas(true, true);
            };

            if (numInputsWidget) {
                syncInputs(numInputsWidget.value ?? 1);

                const origCallback = numInputsWidget.callback;
                numInputsWidget.callback = (value, ...rest) => {
                    const res = origCallback ? origCallback.call(numInputsWidget, value, ...rest) : undefined;
                    syncInputs(value);
                    return res;
                };
            }

            return ret;
        };

        // After execution, the backend sends back the updated counter
        // value ("ui.value"), which we write back into the "value" widget
        // so the next run continues from there. It can still be
        // overwritten by hand at any time before queuing - that override
        // is always available.
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            const newValue = message?.value?.[0];
            if (newValue === undefined) return;

            const valueWidget = this.widgets?.find((w) => w.name === "value");
            if (valueWidget) {
                valueWidget.value = newValue;
            }
        };
    },
});
