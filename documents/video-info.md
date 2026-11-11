# Mykee Video Info

[← Back to the main README](../README.md)  ·  Category: `Mykee/Utils`

Shows the basic properties of a video on the node and passes the useful ones
on. Connect it to the **VIDEO** output of ComfyUI's **Load Video** node (or of
any other node that outputs a VIDEO).

Inputs:

- **video** - the video to inspect.
- **filename_tag** - text appended to the file name on the **filename** output.
  Example: the file is `video.mp4` and the tag is `_edit` → the output is
  `video_edit`. Leave it empty to pass the name on unchanged.

Outputs:

- **width**, **height** (INT) - size of the video in pixels.
- **frame_rate** (FLOAT) - frames per second.
- **filename** (STRING) - the file name of the video without folder and
  extension, with **filename_tag** appended. Connect it e.g. to the
  `filename_prefix` of a save node.

**Info box:** the box at the bottom of the node lists the size, frame rate,
codec, container format and the outgoing file name. It is filled in when the
node runs and shows the values of the last run; the file name row follows the
**filename_tag** field live while you type. The node runs on its own, so
nothing has to be connected to its outputs to see the info.

Codec and format are shown only. They have no output because every node that
takes a codec or format uses its own list, so a generic output could not be
connected to them.

Notes:

- A video file is only probed (its header is read); no frame is decoded, so the
  node is fast even with long videos.
- A video that exists only in memory (e.g. from **Create Video**) has no codec,
  format or file name yet: the box shows `not encoded yet`, and the
  **filename** output starts with `video`.
- The **filename** output never contains a folder, only the name.
