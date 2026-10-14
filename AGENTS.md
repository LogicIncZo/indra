# Indra fork — Tamil OCR (LogicIncZo/indra)

Fork of datalab-to/indra (Apache-2.0, ~1M-param CPU OCR) adding Tamil
recognition, which upstream ships without (Latin, Cyrillic, Greek,
Devanagari, Bengali, CJK only). Upstream announcement thread:
https://x.com/VikParuchuri/status/2110058908463374364

## What this fork adds (branch `tamil-ocr`, merged to main)

- `indra_ocr/vocab.py` — `build_tamil_tokens()` / `Vocab.tamil()`: Latin
  base + full Unicode 18.0 Tamil block (no nukta case needed: U+0BBC is
  unassigned). Tamil-aware `indic_valid()` (same virama rules, no nukta).
- `indra_ocr/pipeline.py` — 3-way script router. `script3.onnx`
  supersedes the stock binary `script.onnx` (which can only name hanzi).
  Routes: 0=multi, 1=hanzi, 2=tamil; missing recognizers fall back to
  multi. **Key fix:** lines routed to `rec_tamil` emit *character*
  tokens (`as_chars=True`), not vocab ids — otherwise `assemble()`
  re-decodes the ids with the stock vocab and produces Greek/Cyrillic.
- `train/` — self-contained synthetic trainer (see below).
- `train/models/` — deployable model dir: stock onnx + `rec_tamil.onnx`
  + `script3.onnx` + `meta.json` (with `vocab_tamil`) + `ta_meta.json`.
  This dir is what `--model-dir` points at.

## Deployment

```bash
pip install -e .            # workers import this code, not PyPI 0.3.1
indra-ocr page.png --model-dir train/models --mode auto
```

IMPORTANT: the installed `indra-ocr` (PyPI) has no Tamil support. If
output for Tamil lines comes back as Greek/Cyrillic, you are running
stock code (or a worker imported the stale install). The editable
install must shadow PyPI's; `pip uninstall -y indra-ocr` first if in
doubt. `OrtOCR` model-dir lookup order: `--model-dir`, then
`INDRA_OCR_MODEL_DIR`, then the HF download.

## Training pipeline (`train/`)

```bash
python3 train/ta_train.py --n-per-class 4000 --epochs 50 \
    --script-epochs 6 --pages 700 --page-epochs 12 --lr 1e-3 \
    --out train/models
# router-only refresh after fixing page-scale crops:
python3 train/ta_train.py --script-only --pages 600 --script-epochs 10 \
    --lr 1e-3 --out train/models
```

- `ta_data.py` — render lines (Noto Tamil fonts, Tirukkural corpus from
  tk120404/thirukkural + generated coverage lines), augment, crop via
  the shared `crop_line` rule. Page-realistic scale: ~11px/char.
- `ta_pages.py` — render synthetic mixed-script pages -> real detector
  (`models_stock/`) -> `crop_lines` training crops. This matches
  inference distribution; flat-line training alone does NOT transfer
  (router especially).
- `ta_model.py` — CRNN + script CNN matching upstream ONNX contract:
  rec in [B,1,32,W] -> greedy `best` [B,T], `pmax`, `lengths`, stride 8;
  script accepts (x, widths), emits argmax int64. Export uses legacy
  tracer (`dynamo=False`, LSTM-safe).
- `ta_train.py` — CTCLoss trainer; `--pages` stage trains on detector
  crops; held-out val uses a FRESH seed (`seed=77`) — never report CER
  on the training distribution.

## Results (2026-10-14)

- page val CER 0.047 (held-out synthetic pages, through real detector)
- line val CER 0.134
- e2e mixed Tamil/Latin test page: English lines perfect, Tamil readable
  (~90% char acc); typical slips: அரு/அரசு, முல/முதல, dropped period.

## Known gaps / next steps

- Corpus is only Tirukkural + synthetic coverage: modern prose,
  numerals-in-context, and 1980s-era print will degrade. Next: real
  scanned books (e.g. ta.wikisource PDFs) + a small human-checked set.
- No confidence-calibrated tamil/multi fallback (router errors go to the
  wrong recognizer silently). Candidate: decode top-2 routes and pick by
  `indic_valid` + mean conf.
- Export-vs-torch verification gate not yet added to `export_*_onnx`.
- PyPI `indra-ocr` package is unrelated to this fork; deploy from repo.
