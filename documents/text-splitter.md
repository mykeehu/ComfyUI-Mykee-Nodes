# Mykee Text Splitter

[← Back to the main README](../README.md)  ·  Category: `Mykee/Prompt`

Cuts a long text into chunks that a chat model can digest, **without ever
cutting inside a sentence**. Chunk sizes are measured in tokens with the
tokenizer of the connected clip, so they are exact for that model.

Typical use: a text that is far too long for one model call (a book chapter,
a log, a transcript). Split it here, then send the chunks on - the nodes after
the `chunks` output run once per chunk. It also works on its own, just to make
numbered `.txt` pieces of a file.

## Inputs

- **clip** - the model whose tokenizer measures the chunks, and (auto mode)
  whose memory limit is probed. Connect the same clip the analyser will use.
- **text** - the text to split. Connect a text to this widget to use that
  instead of typing (the box is then greyed out).
- **limit_mode**
  - **auto** - the node probes the connected model: it runs real prompt
    passes of growing size, catches the out-of-memory error and settles on the
    biggest size that fits. The result is remembered per model and GPU, so the
    next run does not probe again.
  - **manual** - `chunk_tokens` is the chunk size.
- **chunk_tokens** - manual mode: chunk size in tokens.
- **reserve_tokens** - auto mode: room kept free for the question, the system
  prompt and the answer (including thinking). Subtracted from the probed size.
- **safety_percent** - auto mode: only this share of the probed size is used,
  because the real run needs slightly more memory than the probe.
- **max_chunk_tokens** - auto mode: upper limit of a chunk (0 = none). Small
  models read very long texts worse even when the memory would allow it, so the
  default stops at 16384 tokens.
- **paragraph_fill_percent** - when the next paragraph does not fit whole: if the
  chunk already holds at least this share of the limit, it is closed at the
  paragraph end; otherwise it is filled up to the last sentence end that fits.
  100 = always fill to the limit, 0 = always close at paragraph ends.
- **overlap_sentences** - number of sentences repeated at the start of the next
  chunk (0 = off).
- **save_chunks / save_folder / file_name** - also write the chunks to
  `<file_name>_001.txt`, `_002.txt`, ... in the folder (created if missing).
  `file_name` empty = `text`. Off by default. The split itself is cached, so
  switching saving on after a run writes the remembered chunks instead of
  splitting again.
- **reprobe** - auto mode: forget the remembered probe result and measure
  again (e.g. after freeing VRAM). Switch it off again afterwards; while it is
  on, the text is split anew on every run.

## Outputs

- **chunks** - a list of texts (the nodes after it run once per chunk).
- **chunk_count** - how many chunks there are.
- **clip** - the connected clip, passed through.
- **info** - what was measured and how the text was split (also printed to the
  console).

## How the cuts are made

Chunks end at the last sentence end that fits (at a paragraph end when the chunk
is already well filled, see `paragraph_fill_percent`). Abbreviations (dr., pl.,
stb., ...), initials (J. K.), ordinal numbers (5. június) and the dialogue
dashes of Hungarian text are not treated as sentence ends. After the split, every
chunk is counted again as a whole, and if it is over the limit it steps back to the
previous sentence end. Only a sentence that is longer than the limit on its own is
cut smaller - at clause marks first, then at words - and the info text says so.

## Notes

- The auto probe needs a CUDA device. Without one (or if the probe fails) the
  `chunk_tokens` value is used and the info text says why.
- The probe leaves the model loaded, so the analyser that uses the same clip
  starts faster.
- Without a working tokenizer the node estimates 3 characters per token and
  says so in the info text.
