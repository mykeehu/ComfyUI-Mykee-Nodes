# Mykee Meme Maker

[← Back to the main README](../README.md)  ·  Category: `Mykee/Image`  ·  Related: [Mykee Color Background](color-background.md)

Puts a **top** and / or **bottom caption** on an image, an image batch or a
video - either **on the picture** or in a **frame** (a colored band above /
below it). Any font installed on the system can be used. The caption is
rendered once and composited onto every frame, so a video costs hardly more
than an image.

## Inputs and outputs

| | |
|---|---|
| `image` | An image or an image batch. |
| `video` | A video, instead of `image` (connect only one of them). |
| **image** (output) | The captioned image / batch (the frames, when the input is a video). |
| **video** (output) | The captioned video, with the original audio and frame rate - only when the input is a video. |

`top_text` and `bottom_text` can be connected from other nodes (right-click the
text box > convert to input, or drag a link onto it).

## The live preview

The top of the node shows a preview that is rendered by the **same code as the
node itself** (so font, wrapping, outline and frame look exactly like the
result). After a run, the **first frame of the connected image / video** is the
background of the preview, so you can see what the text has to fit. Before the
first run (or if the picture's shape does not match, see below) the background
is gray. The preview's shape is the size of the last run, 1024 x 576 before
that.

`preview_width` / `preview_height` (default **1024** x 1024) are always shown
**above the tabs**, and can be connected to a number node. They set the shape of
the preview picture until the node has run; after a run the preview follows the
real picture (its shape and its first frame). Edit one of the two fields to
override that again until the next run; a connected field always wins. They only
steer the preview: **they never change the result.** The first frame is only shown when its shape matches the
preview's (otherwise it would be stretched), else you get the gray picture. A
connected Primitive / text node's text is used in the preview too. The
background picture is not saved with the workflow - it comes back after the
next run.

Under the preview you get a warning if the font was not found or has no glyph
for one of the letters (for example a meme font without `ő` / `ű`).

## Tabs

`top_text` and `bottom_text` sit **above the tabs** (always visible), so a text node can be
connected to either of them.

**Text** - for each of the two captions:
- `font_size` (in % of the image width - separate for the top and the bottom caption),
- the alignment (left / center / right),
- `lines`: **0 = automatic wrap** (as many lines as needed, the font size is
  not touched); **N = at most N lines** - if the text needs more, the font is
  shrunk until it fits.

The `font_size` is the **ceiling**: with a line limit the font is only ever
made smaller, never larger, to fit the limit (e.g. a long text at limit 1 gets
a smaller font; a short text keeps the set size). With 0 the size is whatever
you set. The preview tells you when a caption had to be shrunk and to what size.

Explicit line breaks in the text are kept. A word wider than a line is broken.

**Style**
- `font` list (the fonts installed on this system) and **Reload font list**,
- `bold`, `italic`, `uppercase` (separate switches; the font family's own bold /
  italic file is used when it has one, otherwise the style is synthesized),
- `text_color`, `stroke_color` (hex, with a color picker under each),
- `stroke_width` in % of the font size (0 = no outline),
- `line_height` (1.0 = the font's own line spacing).

**Layout**
- `placement`: **on image** or **frame**,
- `padding`: the space between text and the edge of the picture / band, in % of
  the image width,
- frame only: `frame_color`; `frame_height_mode` **fit text** (the band is as
  tall as the text plus padding) or **fixed** (`frame_height` in % of the image
  width; the text is shrunk to fit the band). A band is only added for a side
  that has text.
- frame only: `frame_width` (in % of the image width, 0 = none) adds a border
  of `frame_color` all around the picture. The text bands sit above / below the
  bordered picture, and the text wraps to its (wider) width; font size, padding
  and band height stay relative to the original image width. A border alone
  (no text) works too.
  `frame_width_mode`: **around image** (the border hugs the picture, the text
  bands sit outside it) or **around meme** (the border goes around the whole
  captioned meme, bands included; the text wraps to the image width).

Everything is relative to the picture, so the same settings look the same at
any resolution. In frame mode the total height (and, with a border, the width) is made even
(video encoders need it). On the picture itself the text is never shrunk for height: it is
placed at the top / bottom padding and may run over the edge if there is too
much of it.

## Fonts

Only fonts installed on the system are offered: on Windows the `Windows\Fonts`
folder and the per-user `AppData\Local\Microsoft\Windows\Fonts`; on macOS and
Linux the usual system and user font folders. The workflow stores only the
**family name**. A workflow opened on a computer that lacks the font falls back
to a default font (the preview says so and the console prints it).

## Security

The two routes (`/mykee/meme/fonts`, `/mykee/meme/preview`) accept **no file or
folder name** at all. The font list is built from the fixed system folders only
(symbolic links are not followed, files must lie inside those folders), and a
font name from the browser is only looked up in that list. The preview request
is validated like any untrusted input: text length, number ranges, hex colors,
choice values and picture size are limited, and anything invalid is refused
with a message.

The line breaks are measured at one fixed reference size and scaled, so the preview and the real picture wrap exactly the same way at any resolution.
