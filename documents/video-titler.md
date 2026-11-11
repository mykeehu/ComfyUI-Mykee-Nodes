# Mykee Video Titler

[← Back to the main README](../README.md)  ·  Category: `Mykee/Image`

The video version of the [Mykee Meme Maker](meme-maker.md): top / bottom
captions that appear and disappear at set times. The node has a player to find
those times, can cut the video, can read a subtitle file, and saves the titled
video together with a JSON file that holds the whole timing list.

Inputs:

- **video** - the video to put titles on.
- **text** (optional) - the text of an SRT or WebVTT subtitle file, plain
  text, or the JSON this node saved earlier. Plain text gives one title per
  paragraph (or per line when there are no empty lines), spread evenly over the
  video. A saved JSON is recognised by its content (see **Import text**).
- **top_text**, **bottom_text** and every caption setting on the **Text**,
  **Style** and **Layout** tabs - the same as in the Meme Maker. Here they are
  the captions of the title you are editing.

Outputs:

- **video** - the titled (and cut) video.
- **filename** (STRING) - the file name of the source video without folder and
  extension, with **filename_tag** appended. It is found the same way as in
  [Mykee Get Filename](get-filename.md). This is the name the node saves under.

## The player

The player sits under the two text fields. It plays the source video itself:
the file of the loader behind the **video** input is loaded as soon as it is
connected. When the video is not a file the browser can play as it is
(generated in the workflow, trimmed before this node, or in a codec browsers
do not know), run the node once: it writes a preview copy into ComfyUI's temp
folder and the player loads that.

- **⏮ ◀ ▶/⏸ ▶| ⏭** - jump to the start, one frame back, play / pause, one
  frame forward, jump to the end. Clicking the picture also plays / pauses.
- **[ Cut start**, **Cut end ]**, **No cut** - set where the output video
  starts and ends, at the player's position. The same values are on the
  **Output** tab as **trim_start** / **trim_end** (seconds of the source video;
  trim_end 0 = until the end). The cut does not move the titles: their times
  always refer to the source video.

The captions over the player are rendered by the same code as the result, so
the font, wrapping and outline are what you will get. The player shows the
title of the list that is on screen at its position. Inside the **Start** -
**End** range of the Titles tab it shows the captions you are typing instead,
when they belong to the selected title (the one loaded with **✎**) or when no
title of the list is there yet - so you see a title while you write it, before
it is added or updated.

## The Titles tab

- **Start**, **End** - when the title appears and disappears. Type them, or
  press **⇤ Position** / **Position ⇥** to take the player's position. The
  title is gone from the end position on, so the end of one title can be the
  start of the next.
- **Units: time / frames** - shows and reads every time as `hh:mm:ss.mmm` or
  as a frame number.
- **+ Add to the list** - adds the captions that are on the node now (texts and
  every style setting) with this start and end.
- **Update selected** - replaces the selected title with the captions on the
  node and this start and end.
- **The list** - always in time order. Every row can be edited by hand: the
  start, the end, the top and the bottom text. **▶** jumps to the title, **✎**
  loads it into the fields for editing, selects it and jumps to its start,
  **✕** deletes it.
  Titles may not overlap: a change that would make two titles overlap is
  refused with a message and the row is put back.
- **Import text** - what it does depends on the connected text:
  - *SRT, WebVTT or plain text*: its cues are added to the list as titles with
    the current style. Cues that would overlap a title that is already in the
    list are skipped.
  - *a JSON saved by this node*: it is loaded back, so you can carry on
    editing later. The list is **replaced** by the saved titles (with all their
    settings), and the cut and the frame settings are set. Whatever cannot be
    used here is left out and listed in the message - a font that is not
    installed on this machine, an invalid value, a title without valid times or
    one that overlaps another. A title whose setting was left out uses the
    node's current value for it.

  If the text comes from a node whose content the browser cannot see, run the
  node once first.
- **text_mode** - what the node itself does with the text input:
  - `import to list` - nothing is drawn from it (use **Import text**).
  - `subtitle layer` - the cues are drawn as they are, as a third layer above
    the titles, with the current style, **subtitle_font_size** and
    **subtitle_position**.

With an empty list the captions on the node are shown over the whole video,
like in the Meme Maker.

The **Layout** settings (placement, frame color, frame width and height) belong
to the whole video, not to a title - a video cannot change its size halfway. In
`frame` placement the bands are as tall as the tallest title needs, for the
whole video.

## The Output tab

- **trim_start**, **trim_end** - the cut (see the player).
- **filename_tag** - appended to the file name.
- **save_output** - when on, the node writes `<name><tag>.mp4` and
  `<name><tag>.json` into ComfyUI's `output` folder, with the same name. The
  video is saved as H.264 MP4 with its audio.
- **overwrite** - off (default): an existing file is kept and a counter is
  added to the new name (`_0001`, ...), for both files. On: files of the same
  name are replaced.

## The JSON file

```
{
  "version": 1,
  "video":  { "file", "source", "width", "height", "frame_rate", "frame_count", "duration" },
  "trim":   { "start", "end", "start_frame", "end_frame", "until_end" },
  "frame":  { "placement", "frame_color", "frame_width", ... },
  "titles": [
    { "start", "end", "start_frame", "end_frame", "output_start", "output_end",
      "top_text", "bottom_text", "font", "top_font_size", "bottom_font_size",
      "bold", "italic", "uppercase", "text_color", "stroke_color", "stroke_width",
      "line_height", "padding", "top_align", "top_lines", "bottom_align", "bottom_lines",
      "top_margin", "bottom_margin" }
  ],
  "subtitles": { "position", "font_size", "cues": [ { "start", "end", "text" } ] }
}
```

`start` / `end` are seconds of the source video, `output_start` / `output_end`
seconds of the saved (cut) video. `subtitles` is only there in `subtitle layer`
mode; `whole_video_caption` replaces `titles` content when the list is empty.

Notes:

- The whole video is decoded into memory, like in the Meme Maker; very long
  videos need a lot of RAM.
- Changing any setting re-runs the node (and saves again when **save_output**
  is on). While you are still working on the titles, turn **save_output** off.
- Security: the node only writes into ComfyUI's `output` folder (the result)
  and `temp` folder (the preview copy), under a bare, cleaned file name. Its
  HTTP routes are described in the main README.
