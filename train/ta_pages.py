"""Synthetic mixed-script pages -> detector -> training crops.

Renders multi-line pages (Tirukkural Tamil + English + Hanzi), runs the real
det.onnx detector, and yields (crop ink map, script kind, enc) exactly as
inference will see them.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import numpy as np

from ta_data import TAMIL, MULTI, HANZI, load_texts, make_line

HERE = Path(__file__).parent
PAGE_W, PAGE_H = 1400, 1900
MARGIN = 90


def render_page(rng: random.Random, texts: dict, lines_per_page: int = 24):
    """One page image (uint8, 255=white) + list of (text, kind) in order."""
    canvas = np.full((PAGE_H, PAGE_W), 255, np.uint8)
    labels = []
    y = MARGIN
    while y < PAGE_H - 160 and len(labels) < lines_per_page:
        kind = rng.choices([TAMIL, MULTI, HANZI], weights=[10, 4, 3])[0]
        pool = texts[kind]
        text = pool[rng.randrange(len(pool))]
        try:
            strip = make_line(text, kind, rng)
        except Exception:
            continue
        h, w = strip.shape
        if w > PAGE_W - 2 * MARGIN:
            continue
        x = MARGIN + rng.randrange(0, max(1, PAGE_W - 2 * MARGIN - w))
        if y + h >= PAGE_H - 100:
            break
        canvas[y : y + h, x : x + w] = (255 - strip * 255).astype(np.uint8)
        labels.append((text, kind))
        y += h + rng.randrange(10, 26)
    if len(labels) < 8:
        return None, None
    if rng.random() < 0.6:
        nprng = np.random.default_rng(rng.randrange(2**32))
        mask = nprng.random(canvas.shape) < 0.002
        canvas[mask] = rng.choice([0, 180])
    return canvas, labels


def build_page_crops(pipeline, vocab, n_pages: int, texts: dict, seed: int = 5,
                     lines_per_page: int = 24):
    """Detector-crop samples: [(img float ink map, kind, enc)]."""
    from indra_ocr.pipeline import load_pages
    from indra_ocr.line_mask import crop_lines
    import imageio.v2 as imageio

    rng = random.Random(seed)
    out = []
    made = 0
    while made < n_pages:
        page, labels = render_page(rng, texts, lines_per_page)
        if page is None:
            continue
        made += 1
        with tempfile.NamedTemporaryFile(suffix=".png", dir="/tmp") as f:
            imageio.imwrite(f.name, page)
            for _i, gray in load_pages(Path(f.name)):
                boxes = pipeline.detect(gray)
                if not len(boxes):
                    continue
                crops = crop_lines(gray, boxes[:, :4], pipeline.height, pipeline.mask)
        if len(crops) != len(labels):
            continue
        for c, (text, kind) in zip(crops, labels):
            enc = vocab.encode(text) if kind != HANZI else None
            out.append((c.astype(np.float32), kind, enc))
    return out


if __name__ == "__main__":
    print("module; use from ta_train.py --pages")
