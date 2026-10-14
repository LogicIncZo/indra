# indra — LogicIncZo fork (tamil-ocr branch)

Fork of `datalab-to/indra` (Apache-2.0) adding **Tamil OCR**. Upstream: inference-only
package for Datalab's Indra line-OCR models (det + rec + order ONNX). Their rec.onnx
covers Latin/Greek/Cyrillic/Devanagari/Bengali; a separate binary script.onnx routes
hanzi lines to rec_cjk.onnx. Tamil was absent at every layer.

## What this branch adds

- `indra_ocr/vocab.py`: `Vocab.tamil()` / `build_tamil_tokens()` — Latin base +
  Tamil block U+0B80–0BFF. `indic_valid()`/`indic_page_ok()` extended to Tamil
  (virama U+0BCD after consonant; anusvara U+0B82; **no nukta**: U+0BBC is
  `<reserved>` through Unicode 18.0, checked against the live chart 2026-10-14;
  visarga U+0B83 is category Lo, so it never reaches the mark checks).
- `indra_ocr/pipeline.py`: optional third route — if the model folder has
  `rec_tamil.onnx` + `meta.json` key `vocab_tamil`, lines routed to class 2 by a
  3-class script model go to the Tamil recognizer; without it, route 2 falls
  back to the multi recognizer. `script_routes()` replaces `hanzi_lines()`
  (binary script.onnx still works).
- `train/`: synthetic-data trainer for our own weights (no upstream training
  code is published, so the trainer is ours):
  - `ta_data.py` — renders line crops with the system Noto fonts and pushes
    them through `indra_ocr.rec_preproc.crop_line`, so train/inference share
    one scale rule. Augments: blur/noise/JPEG/brightness/rotation/squeeze.
  - `ta_model.py` — CRNN (conv stride 8 + BiLSTM) and 3-class script CNN; ONNX
    export bakes greedy argmax: inputs `x` [B,1,32,W] f32 + `widths` [B] i64 →
    `best` [B,T] i64, `pmax` [B,T] f32, `lengths` [B] i64 (T = W/8). Same I/O
    contract as upstream rec.onnx.
  - `ta_train.py` — CTC training loop + CER eval; writes `rec_tamil.onnx`,
    `script3.onnx`, `ta_meta.json` (carries `vocab_tamil`).
- `tests/`: Tamil vocab/validator tests + pipeline routing tests (fake script
  sessions, binary and 3-class).

## Conventions

- Run `python3 -m pytest tests/ -q` before committing (36 tests).
- Corpus provenance: Tirukkural verses (public domain, ~5th c.) from
  `tk120404/thirukkural` (GitHub, MIT). 8 of 2,660 lines dropped: the dataset
  carries U+0BA6/U+0BA7, `<reserved>` codepoints (dataset corruption — mapped
  to nothing; do not "fix" by adding them to the vocab).
- Our weights are trained from scratch on public-domain text + open fonts, so
  they are not derivatives of Datalab's OpenRAIL-M-gated checkpoints.
- Model folder layout for inference: upstream files + `rec_tamil.onnx` +
  `script3.onnx` + `vocab_tamil` in meta.json (see `train/models/ta_meta.json`).

## Verified facts (do not re-derive)

- Upstream rec.onnx is a single multi-script recognizer; charset lives in
  `meta.json` `vocab` (class 0 = CTC blank).
- UCD in Python 3.12 is 15.0; live unicode.org charts are 18.0 (2026-10). Code
  must not rely on unicodedata for chars assigned after 15.0 — pin explicit
  codepoints.
- Upstream ships no training code (repo = inference + bench + tests, 1 commit
  at fork time).
