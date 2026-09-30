import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Listens for the "mykee.toast" custom message (sent from Python via
// PromptServer.instance.send_sync) and shows it using ComfyUI's own
// native Toast API - the same top-right-corner notification system the
// rest of the app uses, instead of a custom popup implementation.
app.registerExtension({
    name: "Mykee.Toast",
    async setup() {
        api.addEventListener("mykee.toast", (event) => {
            const detail = event.detail || {};
            const {
                severity = "warn",
                summary = "Mykee",
                detail: message = "",
                life = 6000,
            } = detail;

            if (app.extensionManager?.toast?.add) {
                app.extensionManager.toast.add({
                    severity,
                    summary,
                    detail: message,
                    life,
                });
            } else {
                // Very old frontend without the Toast API - fall back to
                // a console warning so the message isn't lost silently.
                console.warn(`[Mykee] ${summary}: ${message}`);
            }
        });
    },
});
