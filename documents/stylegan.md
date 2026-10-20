# StyleGAN Facegenerator / StyleGAN Facegenerator Vector Editor

[← Back to the main README](../README.md)

Four nodes (`Mykee/StyleGAN` category) built around a StyleGAN2/3 generator
`.pkl` (e.g. `stylegan2-ffhq-1024x1024.pkl`).

## Nodes

- [Mykee StyleGAN Image Generation](stylegan-image-generation.md)
- [Mykee StyleGAN Editor](stylegan-editor.md)
- [Mykee StyleGAN Vector Preparation](stylegan-vector-preparation.md)
- [Mykee StyleGAN Vector Save](stylegan-vector-save.md)

## Folder layout

```
ComfyUI/models/stylegan/*.pkl                          - generator checkpoints
ComfyUI/models/stylegan/vgg16.pt                        - auto-downloaded on first vector generation
ComfyUI/models/stylegan/facegen_vectors/<name>.npy      - a direction vector (W-space delta)
ComfyUI/models/stylegan/facegen_vectors/<name>.json     - that vector's parameters
```

Each `<name>.json` looks like:

```json
{
  "layer_start": 0,
  "layer_end": 17,
  "step": 0.25,
  "min": -2.0,
  "max": 2.0,
  "source_mode": "image",
  "seed_a": 100,
  "seed_b": 200,
  "seed_truncation": 0.7,
  "projection_steps": 300,
  "projection_seed": 12345
}
```

- `layer_start`/`layer_end` - inclusive range of W+ layers the vector is
  added to - see "Layer reference" below for what each range tends to
  control on this model.
- `step` - a flat multiplier applied on top of whatever the Image
  Generation/Editor slider is set to.
- `min`/`max` - the Image Generation/Editor slider's range for that vector.
- `source_mode`/`seed_a`/`seed_b`/`seed_truncation`/`projection_steps`/
  `projection_seed` - written automatically by the Vector Save node,
  recording exactly how the vector was built (whichever of these actually
  applied for that `source_mode`) so you can reproduce it later - given
  the same model and, for `image` mode, the same two source photos. The
  images themselves (and their filenames) are deliberately never stored,
  only the generation settings. None of these six drive the Image
  Generation/Editor sliders - they're purely for your own reference, and
  hand-editing or removing them is harmless.

## Layer reference

Which W+ layer range tends to control what, per model. Add a new
sub-section here for any other StyleGAN2/3 model you use with this pack -
the layer count and what each range controls depends on the model's
output resolution and what it was trained on, so this table doesn't
transfer directly to a different generator.

### `stylegan2-ffhq-1024x1024.pkl` (18 layers, 0-17)

A 1024px model has 18 W+ layers (`log2(1024) x 2 - 2`), one pair per
resolution doubling from 4x4 up to 1024x1024. They roughly split into
three groups - this is the original StyleGAN paper's own style-mixing
breakdown, not something specific to this pack:

| Layers | Resolution | Tends to control |
|---|---|---|
| 0-3 | 4x4 - 8x8 | Coarse: pose/camera angle, overall face shape/skull structure, general hairstyle silhouette (short vs long), presence of glasses |
| 4-7 | 16x16 - 32x32 | Middle: finer facial features - eye/eyebrow/nose/mouth shape, more detailed hairstyle |
| 8-17 | 64x64 - 1024x1024 | Fine: color (skin, eye, hair), lighting, micro-texture/skin detail, background |

Vectors are rarely purely one category - a real attribute like "gender"
mixes coarse (jaw/brow structure) and middle (finer features) signal, and
restricting layers only reduces entanglement, it doesn't eliminate it.
Concrete `layer_start`-`layer_end` ranges used by the original
`generate2.py` Gradio app, as a starting point:

| Vector | Layers | Typical `step` |
|---|---|---|
| gender | 0-8 | x1.5 (soft) or x1.0 with `min`/`max` -10/10 (hard) |
| age | 1-7 | x2.5 |
| smile | 4-5 | x2.5 |
| skincolor | 10 | x0.25 |
| eyecolor | 11-12 | x0.15 |

Color-only attributes (eye/skin color) sitting entirely in the fine
range (8-17) are the hardest to isolate well via photo-based vector
building - see the note on VGG-based projection under-weighting color in
"Notes / caveats" below.

### `stylegan-human-v2-1024x512.pkl` (full-body, not a face model)

The [StyleGAN-Human](https://github.com/stylegan-human/StyleGAN-Human)
project's SHHQ-trained generator - built on the same NVIDIA stylegan2-ada
codebase this pack vendors, so every node here works on it exactly the
same way as the FFHQ face model, just with a full-body person instead of
a face. Auto-downloads from Google Drive the same way the FFHQ model
auto-downloads from NVIDIA - see "Notes / caveats" below.

At 1024x512 (2:1, not square) its exact `num_ws` layer count - and so
which layer ranges map to which part of the body - hasn't been verified
against this pack's code yet; test with `layer_start=0, layer_end=31`
(the widest range the Save node's layer fields allow) to see the full
effect of a vector before narrowing it down, the same exploratory way you
would for the coarse/middle/fine breakdown above.

## Notes / caveats

- If `models/stylegan` has no `.pkl` at all, the **model** combo on the
  Image Generation, Editor, and Vector Preparation nodes offers
  `stylegan2-ffhq-1024x1024.pkl` and `stylegan-human-v2-1024x512.pkl` by
  default (alongside whatever real files are already there) - running
  any of them with one of those selected auto-downloads it:
  - `stylegan2-ffhq-1024x1024.pkl` from NVIDIA's official CDN
    (`https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl`,
    ~300MB, one-time).
  - `stylegan-human-v2-1024x512.pkl` from the StyleGAN-Human project's
    Google Drive (see `requirements.txt` for the `requests` dependency
    this needs and what to do if Google Drive's download quota is hit).

  Either way, if that machine has no internet access, download the file
  by hand from the URL in the error message and place it in
  `models/stylegan/` yourself.
- The first time you project a photo (Editor, or Vector Preparation's
  `image` mode), the node downloads NVIDIA's VGG16 feature-detector
  checkpoint (~58MB) into `models/stylegan/vgg16.pt` - needs one-time
  internet access (see `requirements.txt` for the manual download URL if
  that machine is offline).
- GAN inversion quality depends a lot on the input photos being reasonably
  close to FFHQ-style framing (front-facing, centered, similar to the
  StyleGAN training data) - very different crops/angles between image_a and
  image_b will still "work" but the resulting vector may carry pose/framing
  changes along with whatever attribute you meant to isolate. Use
  `layer_start`/`layer_end` (on the Save node) to restrict the vector to
  the layers that actually carry the attribute you want - see "Layer
  reference" above.
- Photo-based vectors for color-only attributes (eye color, skin tone) are
  the hardest to build well: the projection's VGG16 feature-distance loss
  is much more sensitive to structure/shape than to exact color, and the
  colored region (an iris, say) is a tiny fraction of the frame - so there's
  little pressure on the optimizer to reproduce it precisely. If a
  color-vector isn't picking up strongly enough, try: a more exaggerated
  color difference between the two source photos, more `projection_steps`,
  and keeping `layer_start`/`layer_end` tightly restricted to the fine
  layers that actually carry color (see "Layer reference" above) so no
  structural noise from the projection leaks into the saved vector.
- The Image Generation, Editor, and Vector Preparation nodes load the
  generator lazily and cache it in memory per `.pkl` path (same process,
  so it's shared between them and repeated runs) - the first generation/
  preview/edit after choosing a model will be slower while it loads. The
  Save node never touches the generator at all - it only works with the
  `w_pair` it's given plus plain file I/O.
- The bias_act/upfirdn2d/filtered_lrelu ops each try to JIT-compile a fast
  CUDA kernel the first time they're used (needs an MSVC + CUDA build
  toolchain on PATH) and fall back automatically to a slower but fully
  portable pure-PyTorch implementation if that's not available - either way
  it works, the CUDA path is just faster. No precompiled `.pyd` is shipped
  with this pack, since one built elsewhere is tied to that exact
  Python/torch build and won't load on a different ComfyUI install (that's
  the `DLL load failed while importing bias_act_plugin` error some builds
  of this node used to hit).
