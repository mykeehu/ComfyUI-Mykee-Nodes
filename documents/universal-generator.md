# Mykee Universal Generator

[← Back to the main README](../README.md)  ·  Category: `Mykee/Prompt`

Builds a prompt from a library of attribute lists. You pick values (eye
colours, hair styles, places, lighting, camera, style... anything), the node
collects them into a **recipe**, and turns the recipe into a comma-separated
tag list and, with an LLM, into a natural description.

## The library

Everything lives in `ComfyUI/user/default/Mykee Universal Generator/`:

```
library/<category>/<property>.json
library/<category>/<sub-category>/<property>.json
recipes/<recipe name>.json
_trash/            deleted categories, properties and recipes end up here
```

- **Category** = folder, at most two levels deep.
- **Property** = one JSON file, a list of values to pick from.

The first time the node is used, a small starter library is copied there
from the pack. After that the folder is yours; the pack never overwrites it.

A property file:

```json
{
  "version": 1,
  "label": "Eye color",
  "type": "multi",
  "max": 2,
  "order": 20,
  "items": [
    {"id": "k3f9a1c2", "prompt": "green eyes", "label": "green"}
  ]
}
```

| Field | Meaning |
|---|---|
| `label` | Display name (default: the file name). |
| `type` | `single` (one value) or `multi`. |
| `max` | Most values in `multi` (0 = no limit). |
| `order` | Where the property goes in the prompt (lower = earlier). New recipe lines are inserted by it. |
| `scope` | `scene`: always goes to the Scene tab (setting, lighting, camera, style). `subject`: always goes to a person tab (body, clothing, pose, position). Empty: goes to the active tab. |
| `items[].prompt` | The text that goes into the prompt. |
| `items[].label` | What the picker shows (defaults to the prompt; handy for a label in your own language). |
| `items[].id` | Keeps recipe lines attached to their value. Created automatically. |

You can edit these files by hand too; press ⟳ in the library window to
reload them.

## Scene and persons

The recipe has a **Scene** tab that is always there (the picture as a whole)
and any number of **person** tabs (`+ Person`) - people, creatures, anything
that needs its own attributes. A picture without people simply has no
person tabs.

- Every person has a free-text **name** that the prompt uses as given: a
  description (`old man`, `young woman`, `elf with silver hair`) or a name
  with a description (`Hank, a tall man`). Two people of the same kind are
  fine - the name tells them apart.
- ◀ ▶ change the order of the persons (the prompt follows it), ⧉ duplicates
  a person with all its lines (handy when only a few things differ), ✕
  deletes it.
- Each tab has its own groups and its own Manual lines, and each person's
  random lines are drawn on their own: changing one person never changes
  another one's picks.
- The search box and the library window send to the **active tab**, except
  that `scene` properties always go to the Scene and `subject` properties
  need a person tab. The window's **Sending to** list switches the target
  tab (the node follows it).
- The `Subject` category of the starter library has Position (foreground,
  background, through a window...), Pose and Action - use them to place the
  persons in the picture.

Recipes saved before persons existed load as scene-only.

## The node

- **Search box**: type any word, click a result to add it to the recipe.
- **+ Browse**: opens the library window (below).
- **Load / Save / Reset all**: recipe files in the `recipes` folder. Reset
  all starts an empty recipe.
- **Recipe groups**: one per category, collapsible (▸/▾); ↺ clears the group.
  Lines move with ▲ ▼ (the order is the prompt's order) and go with ✕.
  - Click a fixed line to **edit it by hand**: it gets an `edited` mark and
    keeps your text; ⟲ restores the picked values.
  - A random line shows `random N of M` and the values drawn in the last run.
  - `source missing`: the library file was deleted or renamed outside the
    window; the line still works from the copy stored in the recipe.
- **Manual**: free text lines that are not in the library. The library
  window never touches them.
- **TAGS / PROMPT (last run)**: the outputs of the last run, with copy
  buttons.

The node's height follows its content. If you drag it taller (more room for
the `system_prompt` box), that extra height is kept.

## The library window

- **Left**: the category tree. `Library root`, a category or a property can
  be selected. **+ Category** creates a category in the selected top-level
  category (or at the top when the root is selected); **+ Property** creates
  a list in the selected category. Hover a category for ✎ (rename) and 🗑
  (move to `_trash`). The filter box searches categories, properties and
  values.
- **Middle (browse)**: tick the values. `single` properties work as a radio
  button. **Mode**:
  - `Fixed`: the ticked values.
  - `Random from checked`: draws from the ticked values.
  - `Random from all`: draws from every value of the property.
  - In random modes of `multi` properties, `count` sets how many are drawn
    (min-max).
- **Middle (Edit property)**: file name, display name, category (move),
  type, max, order and the list itself. The prompt column comes first; the
  label is copied from the prompt until you type your own. **Paste list**
  adds many values at once, one per line, optionally `prompt | label`.
  Duplicate and Move to trash are here too.
- **Right (Changes to send)**: what the recipe will get when you press
  **Send**: `+` added values, `−` removed values, `~` a changed setting (mode
  or count). The button shows the totals.

Send only writes the properties you changed. Manual lines, edited lines you
did not touch and lines whose library file is gone stay as they are. If you
change the ticks of an edited line, the hand edit is dropped (it no longer
matches the values).

## Inputs

| Input | Meaning |
|---|---|
| `use_llm` | On (default): the LLM writes the `prompt` output. Off: `prompt` is the tag list too, no clip needed. |
| `seed` | Drives the random lines and the LLM sampling. Same seed = same picks and same text. Set it to randomize for a new variation each run. |
| `max_length`, `temperature`, `top_k`, `top_p`, `min_p`, `repetition_penalty` | LLM sampling. |
| `thinking`, `thinking_budget`, `template_mode`, `unload_after_run` | Same as in [Mykee Prompt Advanced](prompt-advanced.md). |
| `clip` (optional) | A generation-capable text model (e.g. from Load CLIP). |
| `system_prompt` (optional) | Empty = built-in instruction. Type or connect your own; `{max_length}` is replaced with the token limit. |

## Outputs

| Output | Meaning |
|---|---|
| `tags` | Every value in recipe order, comma separated, duplicates dropped. With persons: the Scene tags, then one block per person that starts with the person's name, the blocks joined with ` BREAK `. |
| `prompt` | The LLM's description (it gets the scene and each person as separate blocks and is told never to move an attribute from one person to another), or the tag list when `use_llm` is off, no clip is connected or generation fails (a toast says why). |

## Security

The HTTP routes of this node accept input from anyone who can reach the
ComfyUI server, so:

- names are checked against a strict whitelist (no `/ \ : * ? " < > |`, no
  `..`, no leading dot, no Windows device names), category paths are at most
  two names deep;
- every resolved path must stay inside `Mykee Universal Generator`, and
  symbolic links are refused;
- only `.json` files are written, and every JSON (library, recipe, workflow
  state) is schema-checked with size limits;
- delete only moves things into `_trash`.
