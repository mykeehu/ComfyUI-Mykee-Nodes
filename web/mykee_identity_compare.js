import { app } from "../../scripts/app.js";

const MAX_PEOPLE = 10;
const NODE_NAME = "MykeeIdentityCompare";

// For a given person index, the (name, type) of each output this node
// produces for that person - kept in one place so add/remove stays in
// sync with the Python node's RETURN_TYPES/RETURN_NAMES order.
const outputSpecsFor = (i) => [
    [`similarity_${i}`, "FLOAT"],
    [`passed_${i}`, "BOOLEAN"],
    [`annotated_image_${i}`, "IMAGE"],
    [`mask_${i}`, "MASK"],
];

app.registerExtension({
    name: "Mykee.IdentityCompare",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const numPeopleWidget = this.widgets?.find((w) => w.name === "num_people");

            // Adds/removes source_image_N inputs and similarity_N/passed_N/
            // annotated_image_N/mask_N outputs based on the "num_people"
            // widget's value. Never deletes a connected socket, so a
            // connection is never accidentally lost by nudging the widget.
            const syncSockets = (count) => {
                count = Math.max(1, Math.min(MAX_PEOPLE, Math.round(count)));

                // --- inputs: source_image_N ---
                for (let i = 1; i <= MAX_PEOPLE; i++) {
                    const name = `source_image_${i}`;
                    const idx = this.inputs?.findIndex((inp) => inp.name === name) ?? -1;

                    if (i <= count) {
                        if (idx === -1) {
                            this.addInput(name, "IMAGE");
                        }
                    } else if (idx !== -1) {
                        const input = this.inputs[idx];
                        if (!input.link) {
                            this.removeInput(idx);
                        }
                    }
                }

                // --- outputs: similarity_N, passed_N, annotated_image_N, mask_N ---
                for (let i = 1; i <= MAX_PEOPLE; i++) {
                    for (const [name, type] of outputSpecsFor(i)) {
                        const idx = this.outputs?.findIndex((out) => out.name === name) ?? -1;

                        if (i <= count) {
                            if (idx === -1) {
                                this.addOutput(name, type);
                            }
                        } else if (idx !== -1) {
                            const output = this.outputs[idx];
                            if (!output.links || output.links.length === 0) {
                                this.removeOutput(idx);
                            }
                        }
                    }
                }

                this.setSize(this.computeSize());
                app.graph.setDirtyCanvas(true, true);
            };
            // Exposed so onConfigure (below) can re-sync once the saved
            // widget value has actually been loaded - see the comment there.
            this.mykeeSyncIdentityCompareSockets = syncSockets;

            if (numPeopleWidget) {
                syncSockets(numPeopleWidget.value ?? 1);

                const origCallback = numPeopleWidget.callback;
                numPeopleWidget.callback = (value, ...rest) => {
                    const res = origCallback ? origCallback.call(numPeopleWidget, value, ...rest) : undefined;
                    syncSockets(value);
                    return res;
                };
            }

            return ret;
        };

        // onNodeCreated runs before the saved workflow's widget values are
        // applied, so syncSockets() above always runs against the widget's
        // *default* value (1), not whatever num_people was actually saved
        // as. For any saved value other than the default, that leaves the
        // node with the wrong socket count - e.g. a node saved at num_people
        // 1 still got built for the default and, historically, ended up
        // with sockets doubled up in pairs. onConfigure fires once the real
        // saved values are in place, so re-sync sockets against the value
        // that's actually there now.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const ret = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            const numPeopleWidget = this.widgets?.find((w) => w.name === "num_people");
            if (numPeopleWidget && this.mykeeSyncIdentityCompareSockets) {
                this.mykeeSyncIdentityCompareSockets(numPeopleWidget.value ?? 1);
            }
            return ret;
        };
    },
});
