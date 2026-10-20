# Mykee Latent Tone Notch

[← Back to the main README](../README.md)  ·  Category: `Mykee/Latent`

Removes a faint, fixed-frequency stripe pattern from a **LATENT**, before the
VAE decode. It is the general version of the *Latent Nyquist Notch*: that node
only knows the 2-latent-pixel period (0.5 cycles per latent pixel), this one
finds - or is told - **any** frequency between 0.02 and 0.5 cycles per latent
pixel, along the width (vertical stripes) and/or the height (horizontal
stripes). It is **experimental**: run it once with `mode = measure_only` first
and read the status.

Why a separate node: some stripe artifacts have a period that is not a fixed
pixel count. For example a fixed number of stripes across the image width (say
48) gives a period that grows with the resolution (48 cycles over a 128 px wide
latent = 0.375 cycles/px = 2.67 latent px). The Nyquist notch cannot see that.
The patch-grid stripes of DiT models (Nyquist) are just one special case here.

Place it like the Nyquist notch: after the sampler pass that produces the
stripes, directly before `VAE Decode` (or before the latent upscale of a
two-pass workflow).

## Outputs

- **latent** - the cleaned latent.
- **removed** - what was subtracted (a LATENT).
- **status** - per axis and per frequency: `f` (cycles per latent pixel), the
  period in latent pixels, the number of cycles across the picture, the
  prominence and the envelope.

## How it works

1. **Auto mode** takes the power spectrum of every latent channel along the
   axis (averaged over a few bands of the other axis, so a stripe that runs
   across the picture adds up while noise does not), divides it by its own
   local median ("floor"), and averages the channels. A sharp peak that is
   `min_prominence` times above the floor is a stripe frequency. It is refined
   between the spectrum bins and snapped to a whole number of cycles across the
   picture when it is within 0.2 of one.
2. The stripe is a **local weighted least-squares fit** of
   `a*cos(2*pi*f*t) + b*sin(2*pi*f*t)` with a Gaussian window (`smoothing`), so
   its amplitude and phase may drift slowly over the picture. Cells with an
   amplitude far above the typical one are real fine structure and are
   excluded (`protect_structure`). The amplitude is limited.
3. The fit is subtracted. Up to `max_components` frequencies per axis are
   removed one after another, the spectrum being measured again on what is
   left. With nothing above the threshold the latent passes through untouched.

## Settings

- **enabled** - off = the latent passes through untouched.
- **mode** - `remove`, or `measure_only` (nothing changes; the status also
  lists the strongest spectral peaks that are *below* the threshold, which
  shows what the node sees).
- **columns / rows** - which axes to look at. `columns` = vertical stripes (the
  pattern changes along the width), `rows` = horizontal stripes.
- **frequency_mode** - `auto` (find the frequencies) or `manual` (use
  `manual_value` / `manual_unit`).
- **min_prominence** - auto: a peak is removed only when it stands this many
  times above the spectrum around it (about 1 for noise). Default 2.5.
- **max_components** - auto: how many different frequencies are removed per
  axis (default 2: the stripe and, for example, a weaker harmonic).
- **strength** - how much of the fitted stripe is subtracted (1 = all).
- **smoothing** - latent pixels over which amplitude and phase are fitted
  (default 8). Larger = narrower band: it touches less real detail but follows
  a drifting pattern less closely.
- **protect_structure** - cells whose amplitude is this many times above the
  typical one are treated as real structure and excluded; 0 = off.
- **max_amplitude** - largest correction per latent value; 0 = automatic
  (6 x the median amplitude).
- **min_frequency** - auto: lowest frequency searched (cycles per latent
  pixel). Lower frequencies mix with real image content.
- **snap_to_whole_cycles** - auto: use an exact whole number of cycles across
  the picture when the measured frequency is close to one.
- **manual_value / manual_unit** - manual mode: the frequency as cycles per
  latent pixel, as a period in latent pixels, or as the number of cycles across
  the picture (the width for `columns`, the height for `rows`).
- **log_to_console** - also print the status to the console.

## Notes

- The checkerboard (grid) component and signature pooling of the Nyquist notch
  are not part of this node; use the Nyquist notch for those.
- Real, regular texture exactly at the stripe frequency (fences, fabric, fine
  hair) cannot be told apart from a stripe. A higher `min_prominence`, a larger
  `smoothing` and `protect_structure` limit the damage.
- Harmonics above the latent's Nyquist frequency do not exist in the latent;
  they appear only after the VAE decode and need the image-space *Stripe
  Remover* instead.
