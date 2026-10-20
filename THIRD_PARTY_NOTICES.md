# Third-Party Notices

This project incorporates code from `ironjr/grokfast`:
- Source: https://github.com/ironjr/grokfast (commit `93b2248e351d4bb9e4dda8ac32a6999b8970404b`)
- File: `grokfast.py`
- Upstream License: MIT License, Copyright (c) 2024 Jaerin Lee

`grokfast.py` adapts upstream's `gradfilter_ema` and `gradfilter_ma` by adding guards for missing or frozen parameter gradients, cloning tensors to avoid gradient aliasing, and initializing state directly from the first raw gradient.

---

### Upstream MIT License

MIT License

Copyright (c) 2024 Jaerin Lee

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
