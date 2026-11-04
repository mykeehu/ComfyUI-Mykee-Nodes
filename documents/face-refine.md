# Face refine - overview

[← Back to the main README](../README.md)  ·  Category: `Mykee/Face Refine`

Small faces in a generated image (a full-body shot, a character sheet, a
video) are often distorted, because the model gives them only a few latent
tokens. The usual fix is a "face detailer": cut the face out, enlarge it,
re-generate it at low denoise, paste it back. The common detailer nodes expect
a plain 4D latent and a standard VAE, so they cannot be used with every model
(MiniMax H3, for example, has a packed video+audio latent).

These nodes split the job into independent pieces, so only the part that is
specific to a model has to be model-specific:

| Piece | Node | Model specific? |
|---|---|---|
| detect, track, identify, cut out | [Mykee Face Tracks](face-tracks.md) | no |
| turn crops into latents | VAE Encode (any model) or [Mykee H3 Face Latent](h3-face-latent.md) | H3 only |
| one crop at a time | [Mykee Image Batch To List](image-batch-to-list.md) | no |
| re-generate | your own sampler chain, your conditioning, your reference images | - |
| put it back | [Mykee Face Paste](face-paste.md) | no |

Works for a single image and for a video (a frame batch) the same way: an image
is simply a 1-frame video.

## What keeps the identity

1. **The right reference.** `ref_faces` is the face of your source image cut
   with the same framing and size as the crops, and `self_faces` is the best
   face found in the input itself (for a character sheet: the large close-up
   panel). Feed one or both to your model's reference input instead of the whole
   sheet - the face then gets most of the reference tokens instead of a few.
2. **Low denoise** (about 0.35-0.5): the pose, expression and lighting stay,
   the model only redraws the detail. Higher values let the reference win but
   can change the head.
3. **Per-track identity.** Who a face is gets decided once per track (the same
   face followed through the frames) with SFace, and a face can only go to one
   person - no swapping between people, no flipping between frames.
4. **Identity guard** in the paste node: a refined face is only used when it is
   not less similar to the reference than the original face was.

## Recipe: any image model (Qwen Image Edit, Flux, Chroma, SDXL, ...)

```
Load Image ─► Mykee Face Tracks ─┬─ crops ─► VAE Encode ─► KSampler ─► VAE Decode ─┐ refined_all
 (+ source_image_N per person)   │                 ▲  denoise 0.2-0.5             │
                                 │   your conditioning (+ ref_faces / self_faces  │
                                 │   as reference images, a short face prompt)    │
                                 └─ tracks ───────────────────────────────────────┴─► Mykee Face Paste ─► image
                                      Load Image (the original) ───────────────────────────▲
```

The crops come as a **list**, so the whole chain runs once per crop. If your model takes the crop as an
**image in its conditioning** (Qwen Image Edit and similar), the crop is
`image_1` of the edit node (the image to edit), `ref_faces` / `self_faces`
follow as further reference images, and the latent is a VAE Encode of the same
crop (not the edit node's own latent) with denoise about 0.2-0.4. Use the same model and
conditioning style as for the original image, but with a short prompt that fits
a face close-up - not the prompt of the whole scene (a "3 panel character
sheet" prompt would try to draw panels into the crop).

## Recipe: several people, one chain

Set `num_people` on Face Tracks and connect a `source_image_N` per person. The
`crops`, `ref_faces` and `self_faces` lists are aligned, so ONE refine chain
refines every crop with the reference of its own person; the decoded results go
to Face Paste `refined_all`. Nothing is duplicated per person (up to 10).

## Recipe: MiniMax H3 (ref2va)

```
Mykee Face Tracks ─ crops ─► Mykee H3 Face Latent ─► SamplerCustomAdvanced ─► VAE Decode ─► Mykee Face Paste
                   ref_faces ──┐         ▲ (list: one latent per crop)                   ▲
                   self_faces ─┴► H3 Reference Conditioning (width = height = crop_size, length 5,
                                  short face prompt) ──► positive (guider) and LATENT (av_latent)
```

- Set the crop size (a multiple of 16, e.g. 768 or 1024) in Face Tracks, and the
  same width / height in the H3 Reference Conditioning node of the refine pass.
- `length` stays 5 (a still). The audio VAE is not needed.
- BasicScheduler `denoise` about 0.35-0.5. The Turbo LoRA setup of the
  generation pass can be reused.
- VAE Decode returns 5 frames per crop; Face Paste takes the middle one
  (`frame_pick`).

## Recipe: video

Wire the frames in as `images`; everything else is the same. Faces are followed
across the frames (a missed detection is bridged by interpolation), every track
keeps one constant crop scale and a smoothed position, and the guard decides
per track, so the result does not flicker between refined and original. The
refined video has exactly the same frame count and timing as the input, so the
audio can be put back afterwards.

## Limits worth knowing

- Faces need to be detected: a back view has no face (and nothing to fix), and
  strong profile views are cropped by the detector box instead of the landmarks.
- SFace is less reliable for very small or very blurry faces. The guard is
  therefore "must not get worse", not a fixed score.
- Models that already draw the face well do not need this; models that re-draw
  it at low denoise still depend on a good reference.
- The detector and embedder models (YuNet, SFace) are the same ones as for
  the other character nodes (`models/annotators`).
