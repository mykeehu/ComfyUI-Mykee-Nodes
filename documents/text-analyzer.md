# Mykee Text Analyzer

[← Back to the main README](../README.md)  ·  Category: `Mykee/Prompt`

Answers a question about a **long text** that does not fit into the model in one
piece. Feed it the chunks of the [Mykee Text Splitter](text-splitter.md) (or a text
that is short enough to go in on its own).

It works in two rounds, like [Mykee Prompt Advanced Ultimate](prompt-advanced-ultimate.md):

1. **Round 1 - response_1 (the notes).** The model reads the chunks one by one with
   the question in front of it and writes notes: direct answers, and also indirect
   evidence (hints, circumstances, changes, what people say or do), so that an answer
   the text never states outright can still be worked out. A chunk with nothing
   relevant is answered with `NONE` and skipped.
2. **Round 2 - response_2 (the final answer).** The answer is written from the notes
   only. It says when the answer is an inference (and what it rests on), and when the
   text simply does not tell. Notes that are too long for one call are condensed in
   groups first.

## Inputs

- **clip** - the model for the notes (and the answer, unless clip_2 is connected).
- **question** - what you want to know. The notes are written *for this question*.
- **context** - optional background you know to be true, e.g. who is who and how the
  people are related. It is given to every call as reliable `BACKGROUND`, and helps the model
  not to mix up characters. Changing it reads the text again.
- **text** - the text, if it is short enough for the model. Can also be connected.
- **text_in** (socket) - the text to analyse: a single text or the splitter's list of
  chunks. While it is connected, the text box is hidden and ignored.
- **chunk_count** (socket, optional) - the splitter's count; only used to warn if a
  chunk got lost on the way.
- **mode** - `understanding` (notes collect direct and indirect evidence). A keyword
  index mode is planned.
- **note_mode**
  - **rolling** (default) - every chunk is read together with the notes of the
    earlier ones, so clues in different places can be connected. The notes are
    condensed by the model when they grow past `notes_max_tokens`: the model regroups
    them by topic and merges details that show the same thing. It aims at
    `notes_max_tokens` minus `max_length`, so the next chunk's notes still fit, and
    tries once more if the first pass stays too long. A condensed text that was cut off
    by the model's own limit is discarded and the notes are kept as they were
    (the status says so).
  - **independent** - every chunk is read on its own; the notes are only put together
    for round 2.
- **notes_system_prompt** - round 1's system prompt (empty = built-in default for the
  note_mode). Its user turn is labelled `BACKGROUND` (if given), `QUESTION`, `NOTES SO FAR` (rolling only) and
  `TEXT CHUNK`. A chunk with nothing relevant should be answered with `NONE`.
- **max_length, sampling_mode, seed, temperature, top_k, top_p, min_p,
  repetition_penalty, presence_penalty, thinking** - round 1 (per chunk), same as on
  the Ultimate node. **thinking_budget** is the room for thinking.
- **second_system_prompt** and the **second_...** settings - round 2 (the answer),
  with its own seed control and **second_thinking_budget**.
- **notes_max_tokens** - rolling mode: size at which the notes are condensed (they are
  condensed down to about this value minus `max_length`); also the size notes are
  condensed to when they are too long for round 2.
- **input_token_limit** - a chunk (or typed text) longer than this is refused with a
  hint to use the splitter; the node never cuts text itself. 0 = no check.
- **clip_2** (optional) - a different model for round 2; `clip` is taken out of memory
  first.
- **template_mode** (optional) - as on the Ultimate node.
- **unload_after_run** - free the clip used last when the run is done.

## Outputs

- **response_1** - the notes.
- **response_2** - the final answer.
- **status** - chunks, chunks with notes, what was reused, what happened per chunk.
- **thinking_1**, **thinking_2** - the thinking of the two rounds (thinking on).

The node also shows these in boxes with copy buttons. A progress bar counts the
chunks. While it runs, the notes, the thinking and the status boxes fill in live: they
are updated after every chunk (not word by word - the model's output of one chunk
arrives at once).

## Caching and continuing

The notes depend on the question, so they are remembered together with the question,
the chunks, the model, the note mode and round 1's settings:

- a **new question** (or any round 1 setting, or different text) reads the text again;
- changing **only round 2's settings** (answer prompt, sampling, seed...) reuses the
  notes - only the final answer is generated again;
- a run that was **interrupted or failed** (e.g. out of memory at chunk 14) continues
  at the chunk where it stopped when queued again.

The memory lasts until ComfyUI is restarted.

## Tabs

Under the fields you use all the time (clip, question, context, text, note_mode and the two seeds)
the settings sit on four tabs, so the node stays short:

- **Notes** - round 1: its system prompt, max_length, sampling, penalties, thinking.
- **Answer** - round 2: the same for the final answer.
- **Advanced** - mode, notes_max_tokens, input_token_limit, unload_after_run, template_mode.
- **Outputs** - the display boxes (notes, answer, status, thinking) with their copy buttons.

The widgets of a tab that is not shown are only hidden: their values are saved as usual,
and a connected widget keeps its link (and stays visible as a greyed row, so you can see it
is driven from outside). To connect something new to a setting, open its tab first.

## Display boxes

Every box under the node has a drag handle in its bottom-right corner: pull it to make
the box taller or shorter (the node grows or shrinks with it). The heights are saved with
the workflow.
