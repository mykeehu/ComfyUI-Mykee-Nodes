# Mykee Get Filename

[← Back to the main README](../README.md)  ·  Category: `Mykee/Utils`

Passes any media through unchanged and outputs the name of the file it was
loaded from. Connect it to an image, video, audio (or any other) output.

Inputs:

- **media** - any output. It is passed on unchanged.
- **filename_tag** - text appended to the file name on the **filename** output.
  Example: the file is `video.mp4` and the tag is `_edit` → the output is
  `video_edit`. Leave it empty to pass the name on unchanged.

Outputs:

- **media** - the input, unchanged. The socket takes the type of whatever is
  connected to the input (IMAGE, VIDEO, AUDIO, ...).
- **filename** (STRING) - the file name without folder and extension, with
  **filename_tag** appended. Connect it e.g. to the `filename_prefix` of a save
  node.

How the name is found: an image or audio does not carry its file name, so the
node reads the workflow. It walks back from its input, node by node, until it
finds a node with a media file in one of its fields (Load Image, Load Video,
Load Audio, and the loaders of other node packs alike) and takes that name.
The nearest one wins, so the nodes between the loader and this node (resize,
crop, ...) do not matter.

Notes:

- When no file is found behind the input (e.g. a generated image), the name is
  `file`, and a line in the console says so.
- When several loaders feed the input (e.g. two images composited), the one
  closest to this node is used.
- The **filename** output never contains a folder, only the name. The node
  reads the workflow data only; it does not open any file.
- Changing **filename_tag** re-runs this node and everything connected to its
  outputs. If the tag changes often, take only the **filename** from this node
  and feed the generation from the loader directly, so only the saving is
  repeated.
