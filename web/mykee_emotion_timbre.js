import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * Mykee Emotion Timbre - preset UI
 *
 * - Picking a named preset loads its values into the sliders and keeps
 *   showing that preset's name (it does NOT reset to "custom").
 * - Changing any slider by hand switches the dropdown to "custom" - the
 *   label should never claim a mood that isn't actually what's about to
 *   run. (Programmatic changes made while applying a preset don't count
 *   as "by hand" and don't trigger this.)
 * - "Save preset..." asks for a name and POSTs the current slider values
 *   to the server, which writes them into mood_presets.json (overwriting
 *   if the name already exists). The new/updated name then shows up in
 *   the dropdown and becomes the selected value - no ComfyUI restart
 *   needed, no code changes needed.
 * - "Delete preset" removes the currently-selected preset from
 *   mood_presets.json. "custom" can't be deleted (or saved over).
 *
 * The full set of presets (names + values) is fetched from the server on
 * node creation and refreshed after every save/delete, rather than being
 * duplicated as a hardcoded object in this file.
 */

const NODE_NAME = "MykeeEmotionTimbre";

const SLIDER_NAMES = [
    "tempo_scale",
    "rate_variability",
    "pitch_shift_semitones",
    "pitch_range_scale",
    "energy_range_scale",
    "pause_scale",
    "breathiness",
    "brightness",
    "roughness",
    "vibrato_depth_semitones",
    "vibrato_rate_hz",
    "micro_jitter_semitones",
];

app.registerExtension({
    name: "Mykee.EmotionTimbre",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const presetWidget = this.widgets?.find((w) => w.name === "preset");
            if (!presetWidget) return ret;

            let currentPresets = {};
            let applyingPreset = false;

            const setPresetOptions = (presetsDict) => {
                currentPresets = presetsDict || {};
                const names = Object.keys(currentPresets);
                if (presetWidget.options) {
                    presetWidget.options.values = ["custom", ...names];
                }
            };

            const applyPreset = (name) => {
                const values = currentPresets[name];
                if (!values) return;

                applyingPreset = true;
                for (const sliderName of SLIDER_NAMES) {
                    if (!(sliderName in values)) continue;
                    const w = this.widgets?.find((w2) => w2.name === sliderName);
                    if (!w) continue;
                    w.value = values[sliderName];
                    w.callback?.(w.value);
                }
                applyingPreset = false;

                this.setDirtyCanvas(true, true);
            };

            const origPresetCallback = presetWidget.callback;
            presetWidget.callback = (value, ...rest) => {
                const res = origPresetCallback
                    ? origPresetCallback.call(presetWidget, value, ...rest)
                    : undefined;
                if (value !== "custom") {
                    applyPreset(value);
                }
                return res;
            };

            for (const sliderName of SLIDER_NAMES) {
                const w = this.widgets?.find((w2) => w2.name === sliderName);
                if (!w) continue;
                const origSliderCallback = w.callback;
                w.callback = (value, ...rest) => {
                    const res = origSliderCallback ? origSliderCallback.call(w, value, ...rest) : undefined;
                    if (!applyingPreset && presetWidget.value !== "custom") {
                        presetWidget.value = "custom";
                        this.setDirtyCanvas(true, true);
                    }
                    return res;
                };
            }

            this.addWidget("button", "Save preset...", null, async () => {
                const defaultName = presetWidget.value !== "custom" ? presetWidget.value : "";
                const name = window.prompt("Preset name:", defaultName);
                if (name === null) return;
                const trimmed = name.trim();
                if (!trimmed) return;
                if (trimmed.toLowerCase() === "custom") {
                    window.alert('"custom" is reserved and can\'t be used as a preset name.');
                    return;
                }

                const values = {};
                for (const sliderName of SLIDER_NAMES) {
                    const w = this.widgets?.find((w2) => w2.name === sliderName);
                    if (w) values[sliderName] = w.value;
                }

                try {
                    const res = await api.fetchApi("/mykee/emotion_timbre/save_preset", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ name: trimmed, values }),
                    });
                    const data = await res.json();
                    if (!res.ok) {
                        window.alert("Save failed: " + (data.error || res.statusText));
                        return;
                    }
                    setPresetOptions(data.presets);
                    presetWidget.value = trimmed;
                    this.setDirtyCanvas(true, true);
                } catch (e) {
                    window.alert("Save failed: " + e);
                }
            });

            this.addWidget("button", "Delete preset", null, async () => {
                const name = presetWidget.value;
                if (!name || name === "custom") {
                    window.alert('"custom" can\'t be deleted.');
                    return;
                }
                if (!window.confirm(`Delete preset "${name}"?`)) return;

                try {
                    const res = await api.fetchApi("/mykee/emotion_timbre/delete_preset", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ name }),
                    });
                    const data = await res.json();
                    if (!res.ok) {
                        window.alert("Delete failed: " + (data.error || res.statusText));
                        return;
                    }
                    setPresetOptions(data.presets);
                    presetWidget.value = "custom";
                    this.setDirtyCanvas(true, true);
                } catch (e) {
                    window.alert("Delete failed: " + e);
                }
            });

            (async () => {
                try {
                    const res = await api.fetchApi("/mykee/emotion_timbre/presets");
                    const data = await res.json();
                    setPresetOptions(data.presets);
                } catch (e) {
                    console.warn("Mykee Emotion Timbre: couldn't fetch presets from server", e);
                }
            })();

            return ret;
        };
    },
});
