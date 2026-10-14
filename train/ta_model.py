"""CRNN models matching the Indra recognizer I/O contract, + ONNX export.

rec contract (pipeline.recognize): inputs x [B,1,32,W] float32 and widths
[B] int64; outputs best [B,T] int64 (greedy ids, 0=blank), pmax [B,T]
float32, lengths [B] int64; timestep stride 8 (three 2x pools).
script contract (pipeline.hanzi_lines): same inputs; binary model emits
[B,1], script3 emits [B,3] logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _Trunk(nn.Module):
    """c1..c3 only: height 32 -> 4, width /8, so the recognizer timestep
    stride matches rec_preproc.WIDTH_MULTIPLE.  c4 is used only by the
    script classifier (deep=True)."""

    def __init__(self):
        super().__init__()
        self.c1 = self._blk(1, 32)
        self.c2 = self._blk(32, 64)
        self.c3 = self._blk(64, 128)
        self.c4 = self._blk(128, 256)

    @staticmethod
    def _blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2))

    def forward(self, x, deep=False):
        h = self.c1(x)
        h = self.c2(h)
        h = self.c3(h)
        if deep:
            h = self.c4(h)
        return h  # deep=False: [B,128,4,W/8]; deep=True: [B,256,2,W/16]


class RecCRNN(nn.Module):
    def __init__(self, n_classes: int, hidden: int = 128):
        super().__init__()
        self.trunk = _Trunk()
        self.lstm = nn.LSTM(128 * 4, hidden, bidirectional=True, batch_first=True)
        self.out = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        h = self.trunk(x)  # [B,128,4,T]
        h = h.permute(0, 3, 2, 1).flatten(2)  # [B,T,512]
        h, _ = self.lstm(h)
        return self.out(h).log_softmax(-1)  # [B,T,C]


class ScriptCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.trunk = _Trunk()
        self.out = nn.Linear(256, n_classes)

    def forward(self, x):
        h = self.trunk(x, deep=True).amax(dim=3).mean(dim=2)  # [B,256]
        return self.out(h)  # [B,n]


def export_rec_onnx(model: RecCRNN, path: str):
    m = model.eval()
    dummy_x = torch.zeros(2, 1, 32, 64)
    dummy_w = torch.tensor([64, 32], dtype=torch.int64)

    class _RecInfer(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, widths):
            logp = self.m(x)  # [B,T,C]
            best = logp.argmax(-1).to(torch.int64)  # [B,T]
            pmax = logp.exp().amax(-1)  # [B,T]
            lengths = torch.div(widths, 8, rounding_mode="floor").to(torch.int64)
            return best, pmax, lengths

    torch.onnx.export(
        _RecInfer(m),
        (dummy_x, dummy_w),
        path,
        input_names=["x", "widths"],
        output_names=["best", "pmax", "lengths"],
        dynamic_axes={
            "x": {0: "b", 3: "w"},
            "widths": {0: "b"},
            "best": {0: "b", 1: "t"},
            "pmax": {0: "b", 1: "t"},
            "lengths": {0: "b"},
        },
        dynamo=False,
        opset_version=17,
    )
    return path


def export_script_onnx(model: ScriptCNN, path: str):
    m = model.eval()
    dummy_x = torch.zeros(2, 1, 32, 64)
    dummy_w = torch.tensor([64, 32], dtype=torch.int64)

    class _ScriptInfer(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, widths):
            logits = self.m(x)  # [B,3]
            return logits * (widths.float() > 0)[:, None]  # widths stay in the graph

    torch.onnx.export(
        _ScriptInfer(m),
        (dummy_x, dummy_w),
        path,
        input_names=["x", "widths"],
        output_names=["logits"],
        dynamic_axes={"x": {0: "b", 3: "w"}, "widths": {0: "b"}, "logits": {0: "b"}},
        dynamo=False,
        opset_version=17,
    )
    return path
