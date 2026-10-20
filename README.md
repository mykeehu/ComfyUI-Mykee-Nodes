# ComfyUI-Mykee-Nodes

A custom node collection for ComfyUI: counters and utilities, character-consistency
nodes, audio processing, StyleGAN face generation/editing, image and latent
fixers, switches, and LLM-based prompt nodes. More Mykee nodes may be added
here in the future.

Please note: I am not a programmer, so the code and descriptions for the
nodes were written by Claude; I was involved in the design and testing
of the nodes. Please use the nodes with this in mind!

Some nodes are modifications or enhancements of nodes created by other
developers, these are specifically noted. Thank you to them!

## Installation

1. Copy this folder into `ComfyUI/custom_nodes/`, or use
   `git clone https://github.com/mykeehu/ComfyUI-Mykee-Nodes/` in `ComfyUI/custom_nodes/` folder,
   so the end result is `ComfyUI/custom_nodes/ComfyUI-Mykee-Nodes/`.
2. Restart ComfyUI.

To install the additional packages required for certain nodes, see the
requirements.txt file for more information,
and use the following command to install them as needed:

```
pip install -r requirements.txt
```

## Documentation

Every node has its own page with a screenshot, inputs/outputs and notes in the
[`documents`](documents/) folder. Click a node name below to open it.

### Counters & mask

| Node | Description |
|---|---|
| [Mykee Counter](documents/counter.md) | Counter that advances by one whenever any of its `signal_N` inputs carries data. |
| [Mykee Counter (Seed Advanced)](documents/counter-seed-advanced.md) | Same counter, but resets to zero whenever the `seed` value changes. |
| [Mykee Invert Mask (Toggle)](documents/invert-mask-toggle.md) | ComfyUI's InvertMask with an on/off switch. |

### Character consistency

Overview, models and pipeline integration: [Mykee/Character - character-consistency nodes](documents/character-consistency.md)

| Node | Description |
|---|---|
| [Mykee Face Detect + Align/Crop (YuNet)](documents/face-detect-align-crop-yunet.md) | Finds the best face; returns aligned face, context crop and masks. |
| [Mykee Face Embed (SFace)](documents/face-embed-sface.md) | 128-dim face embedding. |
| [Mykee Subject Embed (DINOv2)](documents/subject-embed-dinov2.md) | 768-dim body/clothing ("subject") embedding. |
| [Mykee Identity Score (cosine)](documents/identity-score-cosine.md) | Cosine similarity between two embeddings, with a pass threshold. |
| [Mykee Identity Compare (Multi-Person)](documents/identity-compare-multi-person.md) | All-in-one identity check of a generated image against several references. |
| [Mykee Character Reference Pack](documents/character-reference-pack.md) | Builds a reference image batch and prompt prefix for up to 9 references. |

### Audio

| Node | Description |
|---|---|
| [Mykee Audio Merger](documents/audio-merger.md) | Mixes a narrator track with a background bed. |
| [Mykee Silence Remover](documents/silence-remover.md) | Caps long silent gaps and pads the start/end. |
| [Mykee Stereo To Mono](documents/stereo-to-mono.md) | Forces the output to mono or stereo. |
| [Mykee Audio Leveling](documents/audio-leveling.md) | Fixes level drift and normalizes peak loudness. |
| [Mykee Emotion Timbre](documents/emotion-timbre.md) | Emotion/prosody control based on the WORLD vocoder. |
| [Mykee Voice/Accent Match](documents/voice-accent-match.md) | Reshapes a generated voice's timbre/accent toward a reference sample. |
| [Mykee Audio DSP Cleanup](documents/audio-dsp-cleanup.md) | Fast CPU-only fixes: level, tone, hum, background noise. |
| [Mykee Audio Dereverb](documents/audio-dereverb.md) | Late-reverb suppression with WPE. |
| [Mykee Room Reducer](documents/room-reducer.md) | Reduces the "distant mic / roomy" character of speech. |
| [Mykee AI Voice Restore](documents/ai-voice-restore.md) | VoiceFixer-based speech restoration. |

### StyleGAN

Overview, folder layout, layer reference and caveats: [StyleGAN Facegenerator](documents/stylegan.md)

| Node | Description |
|---|---|
| [Mykee StyleGAN Image Generation](documents/stylegan-image-generation.md) | Generates images from a seed with saved direction-vector sliders. |
| [Mykee StyleGAN Editor](documents/stylegan-editor.md) | Edits an existing photo through GAN inversion and the same vectors. |
| [Mykee StyleGAN Vector Preparation](documents/stylegan-vector-preparation.md) | Builds and previews a new direction vector (writes nothing to disk). |
| [Mykee StyleGAN Vector Save](documents/stylegan-vector-save.md) | Saves the prepared vector and its parameters. |

### Image, latent & conditioning

| Node | Description |
|---|---|
| [Mykee Color Background](documents/color-background.md) | Composites an image (with optional mask) onto a solid color. |
| [Mykee Image Switch](documents/image-switch.md) | 2-input image switch with real passthrough when a branch is inactive. |
| [Mykee Image Switch (Advanced)](documents/image-switch-advanced.md) | Image switch/batch with any number of inputs. |
| [Mykee Switch Conditioning](documents/switch-conditioning.md) | Forgiving conditioning switch (rework of Crystools' node). |
| [Mykee Stripe Remover](documents/stripe-remover.md) | Removes faint periodic stripes from images (DiT banding). |
| [Mykee Latent Nyquist Notch](documents/latent-nyquist-notch.md) | Removes the 2-latent-pixel pattern from a LATENT before VAE decode. |
| [Mykee Latent Tone Notch](documents/latent-tone-notch.md) | Removes a fixed-frequency stripe pattern of any period from a LATENT. |

### Prompt & text

| Node | Description |
|---|---|
| [Mykee Prompt Template](documents/prompt-template.md) | Small prompt library (positive/negative) with load/save. |
| [Mykee Prompt Advanced](documents/prompt-advanced.md) | LLM chat/instruction node with editable system prompt and optional image input. |
| [Mykee Prompt Advanced - Prompt Modifier](documents/prompt-modifier.md) | Rewrites only the details an edit instruction addresses. |
| [Mykee Prompt Advanced Ultimate](documents/prompt-advanced-ultimate.md) | Two-stage version: describe first, then answer/edit from the text. |
| [Mykee MOSS Captioner (JSON)](documents/moss-captioner-json.md) | Captions images as MOSS-Image structured JSON. |
| [Mykee Text Switch/Batch](documents/text-switch-batch.md) | Up to 49 text fields with switch or batch output. |
| [Mykee Before/After Text Injection](documents/before-after-text-injection.md) | Adds the same text before/after many texts at once. |

### Utils

| Node | Description |
|---|---|
| [Mykee Model Template](documents/model-template.md) | Remembers which CLIP/VAE files belong to which model. |
| [Mykee Group Bypasser](documents/group-bypasser.md) | One on/off switch per group, sortable. |
| [Mykee Seed](documents/seed.md) | Rework of rgthree's Seed node with a lockable seed and history. |
| [Mykee XYZ Plot / XYZ Plot Assembler](documents/xyz-plot.md) | Universal XYZ plot: any widget of any node can be an axis. |

## Credits

Third-party code and licenses are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Individual pages credit the nodes they are based on.
