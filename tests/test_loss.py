import numpy as np
import pytest
import torch

from torch_spots.detection.loss import (DotNetLosses, weighted_categorical_crossentropy,
                                        weighted_focal_loss)

EPS = 1e-7


def np_deepcell_wcce(y_true, y_pred, n_classes=2):
    """NumPy transcription of ``deepcell/losses.py::weighted_categorical_crossentropy``
    (deepcell-tf 0.12.10, channels-last, class axis last), returning the mean over pixels."""
    y_pred = y_pred / y_pred.sum(-1, keepdims=True)
    y_pred = np.clip(y_pred, EPS, 1.0 - EPS)
    total_sum = y_true.sum()
    class_sum = y_true.sum(axis=(0, 1, 2), keepdims=True)
    w = 1.0 / n_classes * (total_sum / (class_sum + 1.0))
    return (-(y_true * np.log(y_pred) * w).sum(-1)).mean()


def np_deepcell_focal(y_true, y_pred, n_classes=2, gamma=2.0):
    """NumPy transcription of ``deepcell/losses.py::weighted_focal_loss`` (mean over pixels)."""
    y_pred = y_pred / y_pred.sum(-1, keepdims=True)
    y_pred = np.clip(y_pred, EPS, 1.0 - EPS)
    total_sum = y_true.sum()
    class_sum = y_true.sum(axis=(0, 1, 2), keepdims=True)
    w = 1.0 / n_classes * (total_sum / (class_sum + 1.0))
    temp = (1.0 - y_pred) ** gamma * np.log(y_pred) * w
    return (-(y_true * temp).sum(-1)).mean()


def _batch(rng, b, w, density):
    spot = (rng.random((b, w, w)) < density).astype(np.float64)
    y_true = np.stack([1 - spot, spot], axis=-1)
    logits = rng.normal(size=(b, w, w, 2)) * 2
    p = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    return y_true, p


def _to_torch(y_true, p):   # (B, W, W, C) -> (B, C, W, W), float64 so only the formula is tested
    f = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).contiguous()
    return f(y_true), f(p)


def test_worked_example():
    y_true = torch.tensor([[[[1.0, 1.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]])  # (1, 2, 2, 2)
    # background: pixels (0,0), (0,1), (1,0); spot: (1,1)
    y_pred = torch.tensor([[[[0.9, 0.9], [0.9, 0.5]], [[0.1, 0.1], [0.1, 0.5]]]])
    loss = weighted_categorical_crossentropy(y_true, y_pred).mean()
    assert loss.item() == pytest.approx(0.2128, abs=1e-4)


def test_worked_example_weights():
    from torch_spots.detection.loss import class_weights
    y_true = torch.tensor([[[[1.0, 1.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]])
    w = class_weights(y_true).flatten()
    assert w.tolist() == pytest.approx([0.5, 1.0])      # (background, spot)


def test_agreement_with_deepcell_formula_randomized():
    rng = np.random.default_rng(0)
    n = 0
    for width in (32, 64, 128):
        for density in (0.001, 0.01, 0.05, 0.1):
            for b in (1, 2, 3, 4):
                for rep in range(5):
                    y_true, p = _batch(rng, b, width, density)
                    t, q = _to_torch(y_true, p)
                    got = weighted_categorical_crossentropy(t, q).mean().item()
                    ref = np_deepcell_wcce(y_true, p)
                    assert got == pytest.approx(ref, rel=1e-6)
                    n += 1
    assert n >= 200


def test_focal_uses_same_weights_and_matches_deepcell():
    rng = np.random.default_rng(1)
    for _ in range(40):
        y_true, p = _batch(rng, int(rng.integers(1, 5)), 64, float(rng.uniform(0.001, 0.1)))
        t, q = _to_torch(y_true, p)
        got = weighted_focal_loss(t, q, gamma=2.0).item()
        assert got == pytest.approx(np_deepcell_focal(y_true, p), rel=1e-6)


def _stub_losses(reg, cls, **kw):
    m = DotNetLosses(**kw)
    m.regression_loss = lambda *a: torch.tensor(reg)
    m.classification_loss = lambda *a: torch.tensor(cls)
    return m


LABELS = {"detections": None, "offsets": None}


def test_equal_weighting_with_default_weights():
    m = _stub_losses(0.30, 0.20)
    assert m.regression_weight == 1.0 and m.classification_weight == 1.0
    l2 = 0.01   # the classification-head penalty is added by the training loop
    assert (m(LABELS, LABELS) + l2).item() == pytest.approx(0.51)
    assert m.last_regression == pytest.approx(0.30) and m.last_classification == pytest.approx(0.20)


def test_weights_one_to_five_reproduce_reg_plus_5_cls():
    m = _stub_losses(0.30, 0.20, regression_weight=1.0, classification_weight=5.0)
    assert m(LABELS, LABELS).item() == pytest.approx(0.30 + 5 * 0.20)   # no division by 6


def test_published_weighting_is_the_reference():
    """Reference weights (1/6, 5/6) from the published model: (0.30 + 5 * 0.20) / 6 + 0.01."""
    from torch_spots.detection.train import REFERENCE_TRAINING
    cfg = {k: v for k, v in REFERENCE_TRAINING['loss'].items() if k != 'focal'}
    m = _stub_losses(0.30, 0.20, **cfg)
    assert (m(LABELS, LABELS) + 0.01).item() == pytest.approx(0.22667, abs=1e-5)
