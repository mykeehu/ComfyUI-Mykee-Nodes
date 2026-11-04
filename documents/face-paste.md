# Mykee Face Paste (blend back)

[← Back to the main README](../README.md)  ·  [Face refine overview](face-refine.md)  ·  Category: `Mykee/Face Refine`

Puts refined face crops back into the original image or video frames, exactly
where they came from: every crop was cut with a stored transform (position,
scale, eye-levelling rotation), and this node applies its inverse. Faces cut off
by the image border, tilted heads and moving faces all land correctly.

## Inputs

- **images** - the original image / frames (the same input as Face Tracks).
- **tracks** - from [Mykee Face Tracks](face-tracks.md).
- **refined_all** - the refined results for ALL crops, in the same order as the
  `crops` list of Face Tracks (person 1, 2, ...): simply the decoded output of
  your refine chain (one image per crop, or a list; for H3 the 5-frame decodes,
  where **frame_pick** (middle / first / last) chooses the frame that is the
  face). The number of people comes from the tracks. Not connected = nothing is
  changed. (`num_people` is a leftover widget, hidden, so that saved workflows
  keep loading.)
- **mask_grow_pct / mask_forehead_pct / mask_feather_pct** - size and softness
  of the blend mask (an ellipse over the face, not the whole crop, so the
  surroundings stay exactly as they were).
- **color_match / color_strength** - moves the refined face's tone (LAB mean and
  standard deviation inside the mask) toward the original, so no colour seam
  shows.
- **identity_guard / guard_margin / guard_samples** - see below.
- **model_file_yunet / model_file_sface / score_threshold** - used by the guard.
- **guard_measure** - `in_place` (default): the guard compares the face in the
  ORIGINAL frame with the same face in the finished composite, at the original
  resolution, so the numbers are what you will actually see (and match Mykee
  Identity Compare). `crop`: compares the resampled 768-px crops (older
  behaviour, numbers can differ from Identity Compare).
- **min_detail_ratio** - experimental, **off by default (0)**. `in_place` only:
  rejects a track when the pasted face has less than this fraction of the
  original face's pixel-level Laplacian sharpness. That measure is dominated
  by pixel-level grain, so it can reject refinements that look sharper to the
  eye; use it only if a clearly blurry refinement slipped through.

## Outputs

- **images** - the frames with the refined faces blended in.
- **summary** - per track: pasted or rejected, with the similarity before and
  after.
- **before_after** - one image per track: original crop | refined crop,
  labelled PASTED or REJECTED. Look at this first when tuning.

## Identity guard

Before anything is pasted, the original face and the finished (pasted) face of a
track are compared with the person's reference (`source_image_N`, or the best face of the
input if none was given) with SFace. The refined track is used only if it is
not less similar than the original, minus `guard_margin`; a refined crop in
which no face can be found is rejected too. In `in_place` mode only frames with a real detection are measured (not
interpolated ones). The decision is made **per track**
(on `guard_samples` frames of it), so a video never flickers between refined
and original. Rejected tracks keep their original pixels.

The guard is "must not get worse" rather than a fixed score because SFace is
noisy on tiny faces.
