# Third-party notices

## rgthree-comfy

The "Mykee Seed" node (`nodes/mykee_seed.py`, `web/mykee_seed.js`) is a
reworked version of the "Seed" node from
[rgthree-comfy](https://github.com/rgthree/rgthree-comfy), used under the
MIT License:

```
MIT License

Copyright (c) 2023 Regis Gaughan, III (rgthree)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## MOSS-Image

The built-in caption prompt of the "Mykee JSON Captioner" node
(`nodes/mykee_json_captioner.py`) is the "Caption Prompt" published in the
MOSS-Image user guide (`USERGUIDE.md`, section "Image Captioning") of
[OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image), used under
the Apache License, Version 2.0
(<https://www.apache.org/licenses/LICENSE-2.0>). The prompt text is
reproduced verbatim. The conversion of bounding boxes to the centered
`yxyx` layout format follows the arithmetic of `compile_prompt.py` from the
same repository (Apache-2.0). The rest of the node is original work.

## ComfyUI-MOSS-QwenVL-Layout

The `coord_mode`, `item_detail` ("parts") and `box_padding` options of the
"Mykee JSON Captioner" node (the choice between normalized 0-1000 and
absolute-pixel boxes with the wording of that box instruction, the
body-part / clothing / background splitting rule, and the box padding) are
adapted from
[ComfyUI-MOSS-QwenVL-Layout](https://github.com/leroy989/ComfyUI-MOSS-QwenVL-Layout)
by leroy989, used under the MIT License:

```
MIT License

Copyright (c) 2026 leroy989

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
