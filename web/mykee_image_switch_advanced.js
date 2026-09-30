import { app } from "../../scripts/app.js";

/**
 * Mykee Image Switch (Advanced)
 *
 * Python always declares a fixed, generous cap of image_1..image_MAX
 * optional inputs (ComfyUI's node schema can't truly be infinite), but
 * only 2 are shown on the node initially - the rest are removed right
 * after creation. From then on this extension purely manages which of
 * those pre-declared sockets are currently visible:
 *
 *   - connecting to the currently-last image_N socket adds an empty
 *     image_(N+1) right after it, so there's always one spare to grab.
 *   - disconnecting trims trailing empty sockets back down to a single
 *     spare, never below image_1/image_2, and never touches a socket
 *     that's still connected.
 *
 * `LiteGraph` is used here as an ambient global (window.LiteGraph), the
 * same way mykee_group_bypasser.js does it in this pack.
 */

const MAX_INPUTS = 50;
const MIN_INPUTS = 2;
const NAME_RE = /^image_(\d+)$/;

function imageInputName(i) {
    return `image_${i}`;
}

// [{ inp, idx }], sorted by numeric suffix ascending.
function imageInputs(node) {
    return (node.inputs || [])
        .map((inp, idx) => ({ inp, idx }))
        .filter(({ inp }) => NAME_RE.test(inp.name))
        .sort((a, b) => parseInt(a.inp.name.match(NAME_RE)[1], 10) - parseInt(b.inp.name.match(NAME_RE)[1], 10));
}

function trimTrailingEmpty(node) {
    let list = imageInputs(node);
    while (list.length > MIN_INPUTS) {
        const last = list[list.length - 1];
        const secondLast = list[list.length - 2];
        if (last.inp.link == null && secondLast.inp.link == null) {
            node.removeInput(last.idx);
            list = imageInputs(node);
        } else {
            break;
        }
    }
}

function ensureSpareAfterLastConnected(node) {
    const list = imageInputs(node);
    if (!list.length) return;
    const last = list[list.length - 1];
    if (last.inp.link != null) {
        const num = parseInt(last.inp.name.match(NAME_RE)[1], 10);
        if (num < MAX_INPUTS) {
            node.addInput(imageInputName(num + 1), "IMAGE");
        }
    }
}

app.registerExtension({
    name: "Mykee.ImageSwitchAdvanced",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MykeeImageSwitchAdvanced") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            // Freshly created from defs: every image_1..image_MAX socket
            // exists (ComfyUI builds one per declared optional input).
            // Collapse that down to the starting pair; a loaded workflow
            // instead restores exactly the sockets it saved, so this is a
            // no-op there beyond the safety trim below.
            let list = imageInputs(this);
            for (let n = list.length - 1; n >= 0; n--) {
                const { inp, idx } = list[n];
                const num = parseInt(inp.name.match(NAME_RE)[1], 10);
                if (num > MIN_INPUTS && inp.link == null) {
                    this.removeInput(idx);
                }
            }

            this.setSize(this.computeSize());
            return ret;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const ret = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            // Belt-and-suspenders: a workflow saved at an odd moment
            // shouldn't leave the node without its trailing spare input.
            ensureSpareAfterLastConnected(this);
            trimTrailingEmpty(this);
            return ret;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo, ioSlot) {
            onConnectionsChange?.apply(this, arguments);

            const INPUT = window.LiteGraph?.INPUT ?? 1;
            if (type !== INPUT) return;

            const input = this.inputs?.[index];
            if (!input || !NAME_RE.test(input.name)) return;

            if (connected) {
                ensureSpareAfterLastConnected(this);
            } else {
                trimTrailingEmpty(this);
            }

            this.setSize(this.computeSize());
            this.graph?.setDirtyCanvas(true, true);
        };
    },
});
