"""Train the Tamil recognizer (rec_tamil.onnx) and the 3-way script router
(script3.onnx) on synthetic line images.

Usage:
    python3 train/ta_train.py --n-per-class 2600 --epochs 8 --out models/
Produces models/rec_tamil.onnx, models/script3.onnx, models/ta_meta.json.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from ta_data import TAMIL, HANZI, MULTI, load_texts, make_line
from ta_model import RecCRNN, ScriptCNN, export_rec_onnx, export_script_onnx

from indra_ocr.rec_preproc import pad_batch
from indra_ocr.vocab import Vocab


def build_dataset(vocab: Vocab, texts: dict, n_tamil: int, seed: int = 13):
    """Render lines; returns list of (image [32,W] float32, class, label)."""
    rng = random.Random(seed)
    out = []
    plans = [(TAMIL, texts[TAMIL], n_tamil), (MULTI, texts[MULTI], n_tamil // 3), (HANZI, texts[HANZI], n_tamil // 3)]
    for kind, pool, n in plans:
        for _ in range(n):
            t = rng.choice(pool)
            img = make_line(t, kind, rng)
            enc = vocab.encode(t) if kind != HANZI else None
            if enc:
                out.append((img.astype(np.float32), kind, enc))
            else:
                out.append((img.astype(np.float32), kind, None))
    return out


def collate(batch):
    ws = [b[0].shape[1] for b in batch]
    Wm = int(np.ceil(max(ws) / 8) * 8)
    x = np.zeros((len(batch), 1, 32, Wm), np.float32)  # 0 = paper, 1 = ink
    for i, (img, _, _) in enumerate(batch):
        x[i, 0, :, : img.shape[1]] = img
    widths = torch.tensor(ws, dtype=torch.int64)
    return torch.from_numpy(x), widths, batch


def train_rec(model, data, vocab, epochs, lr, device, overfit_steps: int = 0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    n = len(data)
    rng = random.Random(3)
    if overfit_steps:
        data = data[:6]
        n = len(data)
    order = list(range(n))
    gstep = 0
    for ep in range(epochs):
        rng.shuffle(order)
        tot, nb = 0.0, 0
        for i in range(0, n, 64):
            if overfit_steps and gstep >= overfit_steps:
                break
            chunk = [data[j] for j in order[i : i + 64]]
            Wm = int(np.ceil(max(c[0].shape[1] for c in chunk) / 8) * 8)
            T = Wm // 8
            # CTC wants input/target lengths for every row, so drop rows that
            # cannot contribute (hanzi lines, labels longer than T) first.
            usable = [c for c in chunk if c[2] and len(c[2]) <= T]
            if not usable:
                continue
            targets, tlens = [], []
            for _, _, enc in usable:
                targets += enc
                tlens.append(len(enc))
            x, widths, _ = collate(usable)
            x, widths = x.to(device), widths.to(device)
            logits = model(x)  # RecCRNN.forward already returns log-probs
            logp = logits
            inlens = (widths.float() / 8).ceil().long().clamp(max=logits.shape[1])
            loss = ctc(
                logp.permute(1, 0, 2),  # CTCLoss wants (T, N, C)
                torch.tensor(targets, dtype=torch.long, device=device),
                inlens,
                torch.tensor(tlens, dtype=torch.long, device=device),
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            gstep += 1
            tot += float(loss)
            nb += 1
            if overfit_steps and gstep % 100 == 0:
                with torch.no_grad():
                    b0 = logp.argmax(-1)[0].cpu().numpy()
                set_decode_chars(vocab.chars)
                ref0 = "".join(vocab.chars[c - 1] for c in usable[0][2])
                print(f"    probe loss {float(loss):.4f} pred {ctc_decode(b0, vocab.n_classes)!r} ref {ref0!r}", flush=True)
        print(f"  epoch {ep + 1}/{epochs}  loss {tot / max(nb, 1):.3f}", flush=True)
    return model


@torch.no_grad()
def cer(model, data, vocab, device, limit=400):
    set_decode_chars(vocab.chars)
    rng = random.Random(11)
    sample = rng.sample(data, min(limit, len(data)))
    dists, lens = [], []
    for i in range(0, len(sample), 64):
        chunk = sample[i : i + 64]
        x, widths, batch = collate(chunk)
        logits = model(x.to(device))  # log-probs
        logp = logits
        best = logp.argmax(-1).cpu().numpy()
        if i == 0:
            for k in range(min(2, len(batch))):
                _, _, e0 = batch[k]
                if e0:
                    r0 = "".join(vocab.chars[ch - 1] for ch in e0)
                    print(f"    cer-sample pred {ctc_decode(best[k], vocab.n_classes)!r} ref {r0!r}", flush=True)
        for k, (_, _, enc) in enumerate(batch):
            if not enc:
                continue
            pred = ctc_decode(best[k], len(vocab.chars))
            ref = "".join(vocab.chars[c - 1] for c in enc)
            dists.append(_lev(pred, ref))
            lens.append(len(ref))
    return sum(dists) / max(sum(lens), 1)


_DECODE_CHARS = [""]


def ctc_decode(ids: np.ndarray, n_classes: int) -> str:
    """Greedy CTC: collapse repeats, drop blanks (0)."""
    out, prev = [], 0
    for i in ids:
        i = int(i)
        if i != 0 and i != prev:
            out.append(_DECODE_CHARS[i - 1] if i <= len(_DECODE_CHARS) else "?")
        prev = i
    return "".join(out)


def set_decode_chars(chars: list[str]):
    global _DECODE_CHARS
    _DECODE_CHARS = list(chars)


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def script_cer(model, data, device):
    classes = {MULTI: 0, HANZI: 1, TAMIL: 2}
    model.eval()
    ok = 0
    with torch.no_grad():
        for i in range(0, len(data), 64):
            chunk = data[i : i + 64]
            arr, _w = pad_batch([c[0] for c in chunk])
            x = torch.from_numpy(arr).to(device)
            logits = model(x)
            for k, c in enumerate(chunk):
                ok += int(logits[k].argmax().item() == classes[c[1]])
    return ok / max(len(data), 1)


def train_script(model, data, epochs, lr, device):
    classes = {MULTI: 0, HANZI: 1, TAMIL: 2}
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = len(data)
    rng = random.Random(5)
    order = list(range(n))
    for ep in range(epochs):
        rng.shuffle(order)
        tot, correct, seen = 0.0, 0, 0
        for i in range(0, n, 64):
            chunk = [data[j] for j in order[i : i + 64]]
            x, widths, batch = collate(chunk)
            x, widths = x.to(device), widths.to(device)
            y = torch.tensor([classes[b[1]] for b in batch], device=device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            correct += int((logits.argmax(-1) == y).sum())
            seen += len(batch)
        print(f"  epoch {ep + 1}/{epochs}  loss {tot:.2f}  acc {correct / seen:.3f}", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=0)
    ap.add_argument("--script-only", action="store_true")
    ap.add_argument("--page-epochs", type=int, default=12)
    ap.add_argument("--overfit-steps", type=int, default=0,
                    help="debug: train the recognizer on the first N samples only")
    ap.add_argument("--n-per-class", type=int, default=2600)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--script-epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=str(HERE / "models"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cpu"
    torch.manual_seed(7)

    vocab = Vocab.tamil()
    print(f"vocab: {vocab.n_classes} classes; rendering data...", flush=True)
    texts = load_texts()
    t0 = time.time()
    data = build_dataset(vocab, texts, args.n_per_class)
    tamil_n = sum(1 for d in data if d[1] == TAMIL)
    print(f"rendered {len(data)} lines ({tamil_n} tamil) in {time.time() - t0:.0f}s")

    pdata = None
    if args.pages or args.script_only:
        import sys
        sys.path.insert(0, str(HERE))
        from indra_ocr.pipeline import OrtOCR
        from pathlib import Path as _P
        from ta_pages import build_page_crops
        base_dir = _P(__file__).resolve().parent.parent
        pipe = OrtOCR(base_dir / "train" / "models_stock")
        pdata = build_page_crops(pipe, vocab, args.pages or 500, texts, seed=5)

    if args.script_only:
        assert pdata is not None
        sdata = data + pdata
        smodel = ScriptCNN(3).to(device)
        smodel = train_script(smodel, sdata, args.script_epochs, args.lr, device)
        e = script_cer(smodel, pdata, device)
        print(f"page script acc {e:.3f}")
        export_script_onnx(smodel, str(out / "script3.onnx"))
        print(f"wrote {out}/script3.onnx")
        return

    if args.script_only:
        import sys
        sys.path.insert(0, str(HERE))
        from indra_ocr.pipeline import OrtOCR
        from pathlib import Path as _P
        from ta_pages import build_page_crops
        base_dir = _P(__file__).resolve().parent.parent
        pipe = OrtOCR(base_dir / "train" / "models_stock")
        pdata = build_page_crops(pipe, vocab, max(args.pages, 300), texts, seed=5)
        pval = build_page_crops(pipe, vocab, max(args.pages // 5, 60), texts, seed=999)
        smodel = ScriptCNN(3).to(device)
        smodel = train_script(smodel, data + pdata, args.script_epochs, args.lr, device)
        e = script_cer(smodel, pval, device)
        print(f"script page val acc {e:.3f}", flush=True)
        export_script_onnx(smodel, str(out / "script3.onnx"))
        print(f"wrote {out}/script3.onnx")
        return

    print("training script router (script3.onnx)...", flush=True)
    smodel = ScriptCNN(3).to(device)
    smodel = train_script(smodel, data, args.script_epochs, args.lr, device)
    export_script_onnx(smodel, str(out / "script3.onnx"))

    print("training tamil recognizer (rec_tamil.onnx)...", flush=True)
    rmodel = RecCRNN(vocab.n_classes).to(device)
    rmodel = train_rec(rmodel, data, vocab, args.epochs, args.lr, device, args.overfit_steps)

    if args.pages:
        pval = build_page_crops(pipe, vocab, max(args.pages // 6, 8), texts, seed=999)
        print(f"page crops: {len(pdata)} train / {len(pval)} val", flush=True)
        rmodel = train_rec(rmodel, pdata, vocab, args.page_epochs, args.lr * 0.3, device)
        rmodel.eval()
        e = cer(rmodel, pval, vocab, device)
        print(f"page val CER {e:.3f}")
        heldout = build_dataset(vocab, texts, max(args.n_per_class // 8, 120), seed=77)
        e = cer(rmodel, heldout, vocab, device)
        print(f"line val CER {e:.3f}")
    else:
        rmodel.eval()
        heldout = build_dataset(vocab, texts, max(args.n_per_class // 8, 120), seed=77)
        e = cer(rmodel, heldout, vocab, device)
        print(f"val CER {e:.3f}")
    export_rec_onnx(rmodel, str(out / "rec_tamil.onnx"))

    import json

    meta = {
        "vocab_tamil": vocab.to_json(),
        "source": "LogicIncZo/indra tamil-ocr branch; synthetic Tirukkural + Noto fonts",
        "trained": time.strftime("%Y-%m-%d"),
    }
    (out / "ta_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"wrote {out}/rec_tamil.onnx, script3.onnx, ta_meta.json")


if __name__ == "__main__":
    main()
