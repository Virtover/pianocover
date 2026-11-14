# Third-party code

`pianocover/apc/model.py`, `labels.py` and `loss.py` are adapted from the projects below
(attention rewritten for scaled_dot_product_attention, loss made sync-free, labels vectorised).
Both are under the MIT License.

## hFT-Transformer

https://github.com/sony/hFT-Transformer, files `model/model_spec2midi.py` and
`corpus/conv_note2label.py`. The source code is MIT (`LICENSE-CODE`); the repository's
checkpoints are a separate matter, see docs/MODEL_LICENSES.md.

    MIT License

    Copyright (c) 2023 Sony Group Corporation

## AMT-APC

https://github.com/310hz/amt-apc (commit 78c1afb), files `models/_models.py`,
`train/_loss.py`, `data/sv/extract.py`.

    MIT License

    Copyright (c) 2024 komiya

For both, the full license text is:

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
