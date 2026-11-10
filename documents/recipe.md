# Mykee Recipe

[← Back to the main README](../README.md)  ·  Category: `Mykee/Utils`  ·  Related: [Mykee Model Template](model-template.md), [Mykee XYZ Plot](xyz-plot.md)

Build **recipes** out of **any widget parameter of any node**: a recipe is a
named set of values (cfg, sampler, scheduler, a model, a LoRA, a prompt text, a
boolean, ...). Keep many recipes in one JSON file, switch between them, and
load a recipe back into the parameters with one click.

Like the XYZ Plot and the Model Template, the node does not know any other
node: you drag its outputs onto the widgets you want it to control. The node
then holds the value, and the connected widget receives it.

## Setup

1. Add **Mykee Recipe**. Set **slots** (1 - 10) to the number of parameters
   you need; the node shows that many outputs, `p1` ... `p10` (below `recipe_name` and `xyz_info`).
2. Drag a parameter output (`p1`, `p2`, ...) onto a widget (the same way a
   Primitive node is connected): `cfg`, `sampler_name`, `ckpt_name`, a text
   box, a switch, ... The output takes over the widget's type and gets a
   **key** (by default the widget's name, editable). The current value of the
   widget becomes the parameter's value.
   Supported: INT, FLOAT, STRING (text), BOOLEAN and combo widgets.
3. On the **Parameters** tab, set the values. Type a **recipe_name** and press
   **➕ Send to Recipe**.

The **recipe_name** output carries the name of the current recipe (cleaned like
a file name) - handy as a file name prefix on a Save Image node. The
**xyz_info** output below it goes to the XYZ Plot Assembler (see Batch).

## The four tabs

**Parameters** - the connected parameters: key, the widget it drives, and a
value editor (number, text box, checkbox or the widget's own list). Below:
`recipe_name` and **Send to Recipe**.

- **recipe_picker** (above `recipe_name`) lists the recipes on the node. Picking
  one loads it into the parameters and fills in `recipe_name` (same as Load
  Recipe, no need to switch tabs). To make a new recipe from an existing one:
  pick it, change a value, type a new `recipe_name`, press Send to Recipe. The
  picker shows the recipe whose name is in `recipe_name`.
- A recipe with that name already exists: its entries are **updated** (other
  entries of that recipe are kept). Otherwise a new recipe is added.
- Two outputs with the same key can't be sent (rename one - the field turns red).

**Recipe** - the current recipe as editable JSON text, with a pager
(◀ 2 / 5 ▶) over all recipes on the node.

- **⬆ Load Recipe** puts the recipe's values into the parameters (see the
  rules below).
- **➕ New** adds an empty recipe to write by hand, **🗑 Delete** removes the
  current one (the file only changes when you Save).
- Edits are checked as you type (status line under the text). Valid text is
  stored automatically; invalid text is never stored, and moving on (pager,
  Load, Save) is blocked until it is fixed.

**Batch** - run several recipes one after another (see below).

**JSON file** - the whole file as editable JSON text, plus:

- **json_name** - the file name (without `.json`; a sub-folder such as
  `portraits/sdxl` is allowed).
- **custom_path** - optional folder instead of
  `ComfyUI/user/default/Recipes`.
- **Existing recipe files** - the files in that folder; picking one loads it.
- **💾 Save to Recipe JSON**, **📂 Load Recipe JSON**, **🔄 Reload list**.

Saving to a file other than the one you loaded asks before overwriting an
existing file; loading asks before it throws away unsaved changes.

## Batch

On the **Batch** tab, a switch (off by default) turns batch mode on. Then a list
of the recipes on the node appears, each with a check box, plus **Select All**
and **Select None**. When you press Run, one prompt is queued for every ticked
recipe, in list order, one after another.

In each prompt the links from the `p` outputs are replaced by that recipe's
values (exactly what a Primitive node would put there), so ComfyUI's cache only
re-runs what really differs between two recipes - e.g. a checkpoint loader is
not run again for the next recipe if the model is the same. The same matching
rules as for Load Recipe apply: an output the recipe has no (usable) entry for
keeps the value shown on the node (a value that doesn't fit the widget is
reported when you press Run). `recipe_name` carries the current recipe's name.

**xyz_info** - connect it to a **Mykee XYZ Plot Assembler**: the recipes become
the columns of one plot (the title says `X: recipe`). Switch **Send parameters**
on to also write the parameter names and values under the recipe name in the
plot labels (only the values the recipe sets; long texts are shortened, the
Assembler's info text keeps them in full). The Assembler is optional - without
it you simply get one image per recipe.

Notes:

- Only the recipes on the node can be ticked (load a file on the JSON file tab
  first). The ticks are stored with the workflow, by recipe name.
- A workflow may have only one Recipe node with Batch on, and Batch can't be
  combined with an XYZ Plot / Simple X Plot node in the same workflow (both
  fan out the queue) - mute or bypass one of them. With more than 100 ticked
  recipes you are asked to confirm.
- Switched off, the node behaves exactly as before.
- Workflows saved before `xyz_info` existed are upgraded when loaded (the
  existing links of `p1` ... are kept).

## How a recipe finds its targets (the key)

A recipe is **not** tied to a node class. It stores values under **keys**, and
the key of a parameter output is what connects a recipe entry to a widget:

```json
{"version": 1,
 "recipes": [
   {"name": "Portrait fast",
    "params": {
      "cfg":     {"type": "FLOAT", "value": 4.5, "hint": {"node": "KSampler", "widget": "cfg"}},
      "sampler": {"type": "COMBO", "value": "euler_ancestral"}}}
 ]}
```

So if a node from another author does the same job, connect the output to *its*
widget and give the output the same key - the recipe works unchanged.

**Load Recipe** applies the recipe to the connected outputs by key:

- an entry whose key matches **no connected output** is skipped (listed in the
  message);
- a connected output the recipe has **no entry for keeps its value**;
- if no key matches an output, the entry whose stored `hint.widget` equals the
  widget's name is used - but only when exactly one entry fits;
- a value is applied only if it fits the widget: numbers are limited to the
  widget's min / max, an INT must be whole, a combo value must exist in the
  widget's list. For file names, `\` and `/`, upper / lower case and - when it
  is unambiguous - a different folder are tolerated. A value that does not fit
  is **not** applied and the message says why.

A toast lists what was applied, what was skipped and why.

Hand-written shortcuts are accepted and cleaned when loaded: a bare value
(`"cfg": 4.5`), a single recipe object, or a plain list of recipes.

## Notes and limits

- Only widget parameters (INT, FLOAT, STRING, BOOLEAN, combo) can be recipe
  entries - not MODEL / LATENT / IMAGE connections, and not custom widgets
  such as the rows of a LoRA-stack node.
- The seed is just another parameter: connect a seed widget if you want it in
  the recipe.
- Lowering **slots** is refused while a higher output is still connected.
- Limits per file: 500 recipes, 64 entries per recipe, 5 MB.
- The recipes are stored in the workflow too (so a workflow carries them), and
  editing recipes never invalidates ComfyUI's cache - only the values of the
  connected outputs reach the backend.

## Security

The save / load routes treat every input as untrusted: the folder must be
inside ComfyUI's user / input / output folder or listed in
`allowed_template_dirs.txt`; the file name is reduced to a safe relative name,
symbolic links are refused, only `<name>.json` is read or written, a save never
overwrites a JSON file that is not a recipe file, and the content is validated
against a strict schema on load, on save and again in the browser (types,
lengths, counts, finite numbers, no `__proto__` / `constructor` keys). Recipe
text is only ever shown as text, never as HTML.
