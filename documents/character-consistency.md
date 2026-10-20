# Mykee/Character - character-consistency nodes

[← Back to the main README](../README.md)

A ComfyUI-native rebuild of [Inline Studio](https://github.com/inlineresearch/Inline-Studio)'s
"Encode Character" logic. Inline Studio does the same thing with 3 small,
freely redistributable models (see the project's release notes):
**YuNet** (MIT) detects the face, **SFace** (Apache-2.0) computes a
128-dim face embedding from it, and **DINOv2-base** (Apache-2.0) computes
a 768-dim "subject" (body/clothing) embedding from the full reference
image. The score (cosine similarity) can be used to select/filter
generated takes - the same way Inline Studio scores every render.

## Installation / models

```
pip install -r requirements.txt
```

Place the YuNet and SFace `.onnx` files, as well as the `dinov2-base`
folder (in HuggingFace format), under `ComfyUI/models/annotators/` (the
nodes look here automatically, and list files from here in their
dropdowns too):

- `face_detection_yunet_2023mar.onnx` -
  https://huggingface.co/opencv/face_detection_yunet
- `face_recognition_sface_2021dec.onnx` -
  https://huggingface.co/opencv/face_recognition_sface
- `dinov2-base/` folder -
  https://huggingface.co/facebook/dinov2-base

If you type a HuggingFace repo id into the `dinov2-base` dropdown field
(e.g. `facebook/dinov2-base`) instead of a local folder, `transformers`
will download it automatically on first run.

## Nodes

- [Mykee Face Detect + Align/Crop (YuNet)](face-detect-align-crop-yunet.md)
- [Mykee Face Embed (SFace)](face-embed-sface.md)
- [Mykee Subject Embed (DINOv2)](subject-embed-dinov2.md)
- [Mykee Identity Score (cosine)](identity-score-cosine.md)
- [Mykee Identity Compare (Multi-Person)](identity-compare-multi-person.md)
- [Mykee Character Reference Pack](character-reference-pack.md)

## Suggested integration into existing pipelines

- **MiniMax H3 (video):** `Face Detect+Align/Crop` -> `context_crop` for
  every reference image -> `Character Reference Pack` ->
  `reference_batch` into the H3 node's image references,
  `ref_prompt_prefix` at the start of the prompt.
- **Krea 2 Edit (image), with your "Reference Latent+" node:** connect
  `context_crop` to one of Reference Latent+'s `image_N` inputs, and
  `context_crop_face_mask` to the matching `mask_N` (due to the
  coordinate match, only this mask output works here - not plain
  `face_mask`, see [Face Detect + Align/Crop](face-detect-align-crop-yunet.md)). If you only want to strongly preserve facial
  identity and leave clothing/background to the prompt (which also
  matches your own prompt rules), turn off the corresponding image_N's
  `imageN_hair`/`imageN_body`/`imageN_clothes`/`imageN_background`
  toggles, and leave only `imageN_face` on - our mask marks "face" more
  precisely than the node's own internal auto-detection likely can.
- **Take selection in both cases:** the `Identity Score` computed from
  the reference's `aligned_face` and a frame of the generated
  image/video's SFace embeddings helps automatically filter out takes
  that drifted from the identity.
