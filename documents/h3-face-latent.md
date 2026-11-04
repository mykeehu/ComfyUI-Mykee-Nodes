# Mykee H3 Face Latent

[← Back to the main README](../README.md)  ·  [Face refine overview](face-refine.md)  ·  Category: `Mykee/Face Refine`

MiniMax H3 only. Turns face crops into H3 video+audio latents, so they can be
refined by the H3 sampler (denoise below 1) like any other image. Other models
do not need this node - a normal VAE Encode does the job.

## Why it exists

H3 samples a nested latent: a 24-channel video latent at 1/16 resolution and an
audio latent. A still image is a 5-frame clip. Face detailers built for a plain
4D latent cannot handle that. This node builds the right latent from a crop:

1. the crop is repeated 5 times,
2. encoded with the H3 video VAE (the same recipe H3's own Fun ControlNet uses,
   including the shape check),
3. combined with the audio part of the template latent.

No audio VAE is needed.

## Inputs

- **crops** - `crops` of [Mykee Face Tracks](face-tracks.md).
- **vae** - the MiniMax H3 video VAE.
- **av_latent** - the `LATENT` output of an *H3 Reference Conditioning* node set
  up for the refine pass: width = height = the crop size of Face Tracks,
  length = 5. Only its shape and its audio part are used.

## Output

- **latents** - a list with one latent per crop. A list on purpose: the
  sampler (and the VAE Decode after it) then runs once per crop with batch size
  1, which is what H3's conditioning expects.

## Setup for the refine pass

- H3 Reference Conditioning: the same model setup as for generating the image,
  with a short face prompt (not the scene prompt), `ref_faces` and
  `self_faces` of Face Tracks as reference images, width = height = crop size,
  length = 5. Its `positive` goes to the guider, its `LATENT` to `av_latent`.
- BasicScheduler `denoise` roughly 0.35-0.5.
- After the sampler: VAE Decode, then [Mykee Face Paste](face-paste.md) - it
  takes the middle of the 5 decoded frames (`frame_pick`).

If the crop size does not match the latent, or the VAE gives an unexpected
shape, the node stops with a message saying which of the two is wrong.
