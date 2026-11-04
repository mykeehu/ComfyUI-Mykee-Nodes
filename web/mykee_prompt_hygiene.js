/*
 * Mykee prompt hygiene.
 *
 * The frontend puts EVERY widget of a node into the prompt that is sent to the
 * backend, unless widget.options.serialize === false. (widget.serialize = false
 * only keeps a widget out of the saved workflow; it does NOT keep it out of
 * the prompt.) A client-side-only widget - a read-only response box, a "Copy"
 * / "Save" button, a template picker - therefore travelled to the backend as an
 * extra input. Its value changes (the last response text, the picked name...),
 * and ComfyUI builds its cache key from all inputs of the prompt, so the node
 * and everything after it re-ran on every queue although nothing was touched.
 *
 * This extension marks, on every Mykee node, each widget that is NOT a declared
 * backend input as "not for the prompt". Declared inputs are never touched.
 */
import { app } from "../../scripts/app.js";

function declaredNames(node) {
    const input = node.constructor?.nodeData?.input;
    const names = new Set();
    if (!input) return null;
    for (const group of ["required", "optional", "hidden"]) {
        for (const name of Object.keys(input[group] ?? {})) names.add(name);
    }
    return names;
}

function keepOutOfPrompt(node) {
    const declared = declaredNames(node);
    if (!declared) return; // unknown definition: do nothing rather than guess
    for (const w of node.widgets ?? []) {
        if (!w?.name || declared.has(w.name)) continue;
        if (w.options?.serialize === false) continue;
        w.options = w.options || {};
        w.options.serialize = false;
    }
}

app.registerExtension({
    name: "Mykee.PromptHygiene",
    nodeCreated(node) {
        const cls = node.comfyClass ?? node.constructor?.comfyClass ?? node.type;
        if (typeof cls !== "string" || !cls.startsWith("Mykee")) return;
        keepOutOfPrompt(node);
        // Some widgets are added a moment after the node is created.
        setTimeout(() => keepOutOfPrompt(node), 0);
        setTimeout(() => keepOutOfPrompt(node), 1000);
        const onConfigure = node.onConfigure;
        node.onConfigure = function () {
            const ret = onConfigure?.apply(this, arguments);
            keepOutOfPrompt(this);
            return ret;
        };
    },
});
