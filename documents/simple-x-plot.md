# Mykee Simple X Plot

[← Back to the main README](../README.md)  ·  Category: `Mykee/Utils`  ·  See also: [Mykee XYZ Plot / Assembler](xyz-plot.md)

The one-axis version of [Mykee XYZ Plot](xyz-plot.md): **any widget parameter
of any node** can be the axis, but there is only **X** - no Y / Z axes and no
loop order. The result is a single row of images, one per value.

Everything else works exactly like the full XYZ Plot, because both nodes share
the same frontend code and the same Assembler:

- drag the **x** output onto a widget (cfg, steps, sampler_name, ckpt_name,
  lora strength, seed, a prompt text, a boolean, ...); the output takes over
  the widget's type and the node shows the value editor (INT / FLOAT lists
  and ranges like `5, 7.5, 10` or `4:12:2`, STRING one value per line, combo
  multi-select with filter, BOOLEAN checkboxes);
- optional **seed** input, passed through on the **seed** output and printed
  in the plot's title;
- **skip_if_unchanged** - re-running with nothing changed does not generate the
  plot again (see the XYZ Plot document for details);
- one run = one prompt per value, so only the nodes whose value really changed
  re-run between the images.

## Setup

1. Add **Mykee Simple X Plot** and drag its **x** output onto the widget you
   want to vary.
2. Connect **xyz_info** to a **Mykee XYZ Plot Assembler** (the same Assembler
   the full XYZ Plot uses), and the image to plot to its **images** input.
3. Run. The top of the panel shows the size, e.g. `Plot: 5 images`.

The Assembler outputs the finished row as **plot** (and as **plot_per_z**, a
batch with that one image), plus the **info** text with the seed and the full
value list.

Notes:

- Only one active plot controller (Mykee XYZ Plot **or** Mykee Simple X Plot)
  per workflow; mute or bypass the other one.
- Above 100 cells you are asked to confirm before queueing.
