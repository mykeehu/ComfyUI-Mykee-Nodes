/*
 * Mykee Get Filename - frontend.
 *
 * The first output passes the input on, so it takes the type of whatever is
 * connected to the "media" input (IMAGE, VIDEO, AUDIO, ...): the socket gets
 * that type's colour and can be connected wherever that type is accepted.
 * With nothing connected it is a wildcard again.
 */
import { app } from "../../scripts/app.js";

const NODE_NAME = "MykeeGetFilename";
const INPUT_NAME = "media";
const ANY = "*";

function linkOf(node, linkId) {
    const links = node.graph?.links;
    if (!links || linkId == null) return null;
    return (typeof links.get === "function" ? links.get(linkId) : links[linkId]) ?? null;
}

// Type of the output that feeds the "media" input, "*" when unknown.
function incomingType(node) {
    const input = node.inputs?.find((i) => i.name === INPUT_NAME);
    const link = linkOf(node, input?.link);
    if (!link) return ANY;
    const origin = node.graph?.getNodeById?.(link.origin_id);
    const type = origin?.outputs?.[link.origin_slot]?.type ?? link.type;
    return typeof type === "string" && type && type !== ANY ? type : ANY;
}

function syncOutputType(node) {
    const output = node.outputs?.[0];
    if (!output) return;
    const type = incomingType(node);
    if (output.type === type) return;
    output.type = type;
    // the links that leave this output carry the type too (their colour)
    for (const id of output.links ?? []) {
        const link = linkOf(node, id);
        if (link) link.type = type;
    }
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "Mykee.GetFilename",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const ret = onConnectionsChange?.apply(this, arguments);
            if (!app.configuringGraph) syncOutputType(this);
            return ret;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const ret = onConfigure?.apply(this, arguments);
            // the links of the other nodes are loaded a moment later
            setTimeout(() => syncOutputType(this), 50);
            return ret;
        };
    },
});
