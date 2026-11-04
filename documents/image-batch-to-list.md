# Mykee Image Batch To List

[← Back to the main README](../README.md)  ·  [Face refine overview](face-refine.md)  ·  Category: `Mykee/Face Refine`

Turns an image batch into a list, so every node after it runs once per entry
instead of once for the whole batch.

## When you need it

When the crop goes into your model's **conditioning** as an image (Qwen Image
Edit and similar edit models). A batch of crops would be read there as several
reference images of ONE generation, and a single image would come back - Face
Paste then stops with `refined input has 1 image(s) ... but 2 crop(s) were
expected`. With the list, the whole chain runs once per crop:

```
images ─► Image Batch To List ─┬─► text-encode / edit node (image_1 = the crop)
                                └─► VAE Encode ─► KSampler ─► VAE Decode ─► Mykee Face Paste (refined_all)
```

Not needed for [Mykee H3 Face Latent](h3-face-latent.md) (it already outputs a
list), nor for a plain VAE Encode -> sampler chain without image conditioning.

## Inputs

- **images** - any image batch you want processed one by one. (The crops of
  [Mykee Face Tracks](face-tracks.md) already come as a list.)
- **chunk_size** - images per list entry. 1 (default) = each image on its own.
  Larger values are faster when the chain has no image conditioning (plain
  VAE Encode -> sampler), e.g. for a video with many crops.

## Output

- **images** - a list of image batches, in the original order. Reference images
  that are NOT in the list (`ref_faces`, `self_faces`) are reused for every
  entry.
