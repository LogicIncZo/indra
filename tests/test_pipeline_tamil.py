import numpy as np
import pytest

from indra_ocr.pipeline import OrtOCR as Pipeline


class FakeScript:
    """script.onnx stub: routes by crop width parity (deterministic)."""

    def __init__(self, three):
        self.three = three

    def run(self, _, feed):
        arr = feed["x"]
        n = arr.shape[0]
        if self.three:
            p = np.zeros((n, 3), np.float32)
            for k in range(n):
                p[k, (k + 1) % 3] = 5.0
            return [p]
        p = np.array([[6.0], [-6.0]] * (n // 2) + [[6.0]] * (n % 2), np.float32)[:n]
        return [p]


def _bare_pipeline(three, has_tamil):
    p = Pipeline.__new__(Pipeline)
    p.script = FakeScript(three)
    p.rec_tamil = object() if has_tamil else None
    return p


def _crops(n, w=64):
    return [np.random.rand(32, w).astype(np.float32) for _ in range(n)]


def test_binary_script_model_keeps_hanzi_routes():
    p = _bare_pipeline(three=False, has_tamil=False)
    assert p.script_routes(_crops(4)) == [1, 0, 1, 0]


def test_three_class_model_routes_tamil_only_when_loaded():
    p = _bare_pipeline(three=True, has_tamil=True)
    assert p.script_routes(_crops(3)) == [1, 2, 0]


def test_script_routes_are_pure_classifier_output():
    p = _bare_pipeline(three=True, has_tamil=False)
    assert p.script_routes(_crops(3)) == [1, 2, 0]
    # Fallback onto the multi recognizer when a route has no model is
    # applied in recognize(); script_routes never sees the model set.