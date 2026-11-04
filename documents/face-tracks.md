# Mykee Face Tracks (detect + crop)

[← Back to the main README](../README.md)  ·  [Face refine overview](face-refine.md)  ·  Category: `Mykee/Face Refine`

Finds every face in an image or video, follows each face through the frames,
decides who it is and cuts out one square, eye-levelled crop per face and
frame. It does not change anything in your image.

A single IMAGE is a 1-frame video, so there is one code path for both.

## Inputs

- **images** - one image, or a video as a frame batch.
- **num_people** (1-10) - live switch: adds/removes the `source_image_N` inputs.
- **source_image_N** (optional) - a reference image of person N (a photo, a
  character card). Its largest face is the identity reference. Without any
  source image and with `num_people` = 1, every detected face is person 1.
- **chunk_size** - crops per list entry (an entry never mixes people). 1 (default)
  = every crop is processed on its own, which is needed when the crop goes into
  the conditioning of an edit model. Larger = faster for a plain VAE Encode ->
  sampler chain without image conditioning (video).
- **model_file_yunet / model_file_sface** - the same models as the other
  character nodes (`models/annotators`).
- **crop_size** - side of every crop (multiple of 16). 768 or 1024 suit most
  models.
- **crop_factor** - how much context a crop has. 2.0 = the FFHQ head window.
- **align** - `level_eyes` rotates every crop so the eyes are horizontal (the
  paste node rotates it back). Profile views and unreliable landmarks are never
  rotated.
- **score_threshold / min_face_size_px / detect_upscale** - detection.
  `min_face_size_px` is an *absolute* size on purpose: a small full-body face
  must not be dropped just because another panel has a big close-up. Use
  `detect_upscale` > 1 if tiny faces are missed.
- **match_threshold** - minimum SFace similarity for a face to count as a given
  person. Faces below it are left untouched (listed in `summary`).
- **exclusive_matching** - each person may own at most one track. Only for
  images where every person appears exactly once; leave off for character
  sheets (one person, several faces).
- **source_which_face** - which face of a `source_image_N` is the reference when
  the image shows several (character sheet). Same options as Mykee Face Detect,
  plus `most_frontal`: the face turned most towards the camera (the largest among
  the frontal ones) - use it when a profile view is bigger than the frontal
  close-up, so `largest` would pick the wrong one.
- **assign_mode** - how faces in the input are given to persons.
  `by_similarity` (default): the face most like `source_image_N` becomes person
  N - for refining and for keeping identities. `left_to_right`, `right_to_left`,
  `top_to_bottom`, `bottom_to_top`: persons are handed out by POSITION (the
  first face in that order = person 1 = `source_image_1`, ...). Use these for
  **face swaps**, where the sources are different people and similarity means
  nothing. Only the `num_people` best detections are used, so small or
  low-score false positives are ignored; `match_threshold` and
  `exclusive_matching` do not apply. The summary lists which face got which
  person.
- **skip_self_reference** - still images only: when a person has several faces,
  the best one (the self face) is left out of the crops, it is your reference.
- **mask_grow_pct / mask_forehead_pct / mask_feather_pct** - the soft face mask
  of `masks` (YuNet's box starts at the eyebrows, hence the forehead
  extension; feather is a percentage of the crop size).
- **edge_fill** - what fills the crop where the window reaches past the image
  border (partial faces).
- Video: **detect_stride**, **max_gap**, **link_iou**, **window_size_mode**,
  **smooth_radius**, **embed_samples** - detection spacing, how long a missed
  face stays the same track, linking sensitivity, constant or per-frame crop
  scale, smoothing of the crop window, frames used for the identity embedding.

- **source_which_face / source_face_index** - which face of a `source_image_N`
  is the reference person when that image shows several (for example a
  character sheet). Same options as `which_face` in Mykee Face Detect
  (largest, highest_score, leftmost, ..., top_left, ..., x_order / y_order with
  the index). Default `largest`. It applies to every `source_image_N`; the
  summary reports how many faces were found and which rule was used.

**Detections:** a second box on the same face (for example one that also takes
in forehead and hair) is dropped, the higher-scoring one is kept. A box counts
as a duplicate when most of it lies inside a better one, or when its own eye /
nose landmarks fall inside a better box. For a single
image the summary lists every detected face with its position, size and
detector score, so a surprising crop can be traced back to a detection.

## Outputs

- **summary** - what was found, who was matched with which similarity, warnings.
- **tracks** (`FACE_TRACKS`) - the geometry and identity data for
  [Mykee Face Paste](face-paste.md).
- **crops, masks, ref_faces, self_faces, person** - **lists**, aligned entry by
  entry, for ALL people (person 1 first, in the order the paste node expects).
  ComfyUI runs the nodes after a list once per entry and pairs the lists entry
  by entry, so every crop arrives together with ITS OWN person's reference: one
  refine chain serves every person, and nothing has to be duplicated per person.
  - **crops** - the eye-levelled crops.
  - **masks** - soft face masks in crop space.
  - **ref_faces** - the face of the person's `source_image_N`, cut with the same
    framing and size as the crops (falls back to the self face if no source
    image is wired).
  - **self_faces** - the best face found in the input (large, frontal, high
    detection score), e.g. the close-up panel of a character sheet.
  - **person** - the person number of each entry.

If no face was found at all, one grey placeholder entry keeps the chain valid
and the paste node ignores it.

## How identity is decided

Every track gets one SFace embedding (mean of its best frames) and goes to the
person it resembles most, if the similarity is at least `match_threshold`. A
person may own many tracks, but a track belongs to exactly one person, and the
decision is made per track, never per frame. When two persons score almost the
same, the summary says so.
