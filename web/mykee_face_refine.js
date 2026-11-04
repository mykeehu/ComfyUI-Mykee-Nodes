import { app } from "../../scripts/app.js";

// Face-refine nodes:
//   MykeeFaceTracks - live "num_people" switch: adds/removes the source_image_N
//                     inputs. Its outputs are fixed lists (crops, masks,
//                     ref_faces, self_faces, person).
//   MykeeFacePaste  - the num_people widget is not needed any more (the number
//                     of people comes from the tracks): it is hidden, and its
//                     widget socket removed. The widget itself stays, so saved
//                     workflows keep loading.
// A socket that already has a connection is never removed.

const MAX_PEOPLE = 10;
const HIDE_WIDGET = "num_people";

function hideNumPeople(node) {
    const widget = node.widgets?.find((w) => w.name === HIDE_WIDGET);
    if (widget) {
        widget.hidden = true;
        widget.computeSize = () => [0, -4];
    }
    // newer frontends give every widget an invisible socket; it would sit on
    // top of the first real input and catch dropped links
    const idx = node.inputs?.findIndex((inp) => inp.name === HIDE_WIDGET) ?? -1;
    if (idx >= 0) {
        if (node.inputs[idx].link != null) node.disconnectInput?.(idx);
        node.removeInput(idx);
    }
}

app.registerExtension({
    name: "Mykee.FaceRefine",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "MykeeFacePaste") {
            const created = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const ret = created ? created.apply(this, arguments) : undefined;
                hideNumPeople(this);
                this.setSize(this.computeSize());
                return ret;
            };
            const configured = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const ret = configured ? configured.apply(this, arguments) : undefined;
                setTimeout(() => {
                    hideNumPeople(this);
                    app.graph.setDirtyCanvas(true, true);
                }, 50);
                return ret;
            };
            return;
        }
        if (nodeData.name !== "MykeeFaceTracks") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const numPeopleWidget = this.widgets?.find((w) => w.name === "num_people");

            const syncSockets = (count) => {
                count = Math.max(1, Math.min(MAX_PEOPLE, Math.round(count)));
                for (let i = 1; i <= MAX_PEOPLE; i++) {
                    const name = `source_image_${i}`;
                    const idx = this.inputs?.findIndex((inp) => inp.name === name) ?? -1;
                    if (i <= count) {
                        if (idx === -1) this.addInput(name, "IMAGE");
                    } else if (idx !== -1 && !this.inputs[idx].link) {
                        this.removeInput(idx);
                    }
                }
                this.setSize(this.computeSize());
                app.graph.setDirtyCanvas(true, true);
            };
            this.mykeeSyncFaceRefineSockets = syncSockets;

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

        // onNodeCreated runs before the saved widget values are applied, so
        // re-sync once the real num_people value is in place.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const ret = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            const numPeopleWidget = this.widgets?.find((w) => w.name === "num_people");
            if (numPeopleWidget && this.mykeeSyncFaceRefineSockets) {
                this.mykeeSyncFaceRefineSockets(numPeopleWidget.value ?? 1);
            }
            return ret;
        };
    },
});
