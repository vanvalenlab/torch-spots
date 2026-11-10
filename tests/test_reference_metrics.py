import numpy as np
import pytest

from torch_spots.detection.reference_metrics import (
    DEFAULT_TAU,
    aggregate,
    match_mutual_nearest_neighbor,
    score_image,
    sum_of_min_distance,
)


def P(*pts):
    return np.array(pts, dtype=float).reshape(-1, 2)


# ---- matching scenarios -------------------------------------------------

def test_distance_convention():
    s = score_image(P([0, 0]), P([3, 4]), tau=5)
    assert s.tp == 1 and s.rmse == pytest.approx(5 / np.sqrt(2))


def test_close_pair_kept_even_when_other_assignment_cheaper():
    row, col = match_mutual_nearest_neighbor(P([0, 0], [10, 0]), P([1, 0], [-8.5, 0]), 1)
    assert list(row) == [0] and list(col) == [0]


def test_non_mutual_neighbours_not_matched():
    s = score_image(P([0, 0], [1, 0]), P([0.6, 0], [1.9, 0]), 1)
    assert s.tp == 1 and s.precision == 0.5 and s.recall == 0.5


def test_threshold_inclusive():
    row, _ = match_mutual_nearest_neighbor(P([0, 0]), P([1, 0]), 1)
    assert len(row) == 1


def test_default_tau_recorded():
    assert DEFAULT_TAU == 1.0
    assert score_image(P([0, 0]), P([0, 0])).tau == 1.0


# ---- per-image scores ----------------------------------------------------

def test_partial_match():
    true = P([0, 0], [10, 10], [20, 20], [30, 30])
    pred = P([0, 0], [10, 10])
    s = score_image(true, pred)
    assert (s.precision, s.recall) == (1.0, 0.5)
    assert s.f1 == pytest.approx(2 / 3) and s.jac == pytest.approx(0.5)
    assert (s.tp, s.fp, s.fn) == (2, 0, 2)


def test_no_matches():
    s = score_image(P([0, 0]), P([50, 50]))
    assert (s.precision, s.recall, s.f1, s.jac) == (0, 0, 0, 0)
    assert s.rmse is None and s.d_md is None


def test_rmse_single_pair_unit_distance():
    assert score_image(P([0, 0]), P([1, 0])).rmse == pytest.approx(1 / np.sqrt(2))


def test_dmd_ignores_unmatched_outliers():
    s = score_image(P([0, 0]), P([0, 0.5], [50, 50]))
    assert s.d_md == pytest.approx(0.5)


def test_dmd_function_direct():
    assert sum_of_min_distance(P([0, 0]), P([0, 2])) == pytest.approx(2.0)


@pytest.mark.parametrize("true,pred", [
    (P([0, 0]), np.empty((0, 2))),
    (np.empty((0, 2)), P([0, 0])),
    (np.empty((0, 2)), np.empty((0, 2))),
])
def test_degenerate_images(true, pred):
    s = score_image(true, pred)
    assert s.f1 == 0 and s.jac == 0 and s.rmse is None and s.d_md is None
    assert s.fp == len(pred) and s.fn == len(true)


# ---- aggregation ----------------------------------------------------------

def _fake(f1, n_true, tp, rmse=None):
    from torch_spots.detection.reference_metrics import ReferenceScores
    return ReferenceScores(1.0, n_true, tp, tp, 0, n_true - tp, f1, f1, f1, f1, rmse, rmse)


def test_two_images_macro():
    agg = aggregate([_fake(1.0, 1, 1), _fake(0.5, 1000, 500)])
    assert agg["f1"] == pytest.approx(0.75)


def test_undefined_rmse_excluded():
    agg = aggregate([_fake(1.0, 5, 5, rmse=0.4), _fake(0.0, 5, 0, rmse=None)])
    assert agg["rmse"] == pytest.approx(0.4)
    assert agg["rmse_n_images"] == 1 and agg["n_images"] == 2


def test_totals_available():
    scores = [score_image(P([0, 0], [9, 9]), P([0, 0], [30, 30], [40, 40])),
              score_image(P([5, 5]), P([5, 5]))]
    agg = aggregate(scores)
    assert agg["supplementary_micro"] == {
        "tp": sum(s.tp for s in scores),
        "fp": sum(s.n_pred - s.tp for s in scores),
        "fn": sum(s.n_true - s.tp for s in scores),
    }
    assert all(s.fp == s.n_pred - s.tp and s.fn == s.n_true - s.tp for s in scores)


def test_mixed_tau_refused():
    a = score_image(P([0, 0]), P([0, 0]), tau=1)
    b = score_image(P([0, 0]), P([0, 0]), tau=2)
    with pytest.raises(ValueError):
        aggregate([a, b])


# ---- randomized comparison against the original code ----------------------

def _random_pair(rng):
    n = int(rng.integers(0, 40))
    extent = float(rng.choice([8, 30, 128]))
    true = rng.uniform(0, extent, size=(n, 2))
    keep = rng.random(n) < rng.uniform(0.3, 1.0)
    # perturb with radius clustered around tau to stress the threshold
    radius = rng.choice([0.2, 0.9, 1.0, 1.1, 2.5]) * rng.uniform(0.5, 1.0, size=(n, 1))
    angle = rng.uniform(0, 2 * np.pi, size=n)
    jitter = radius * np.stack([np.sin(angle), np.cos(angle)], axis=1)
    pred = (true + jitter)[keep]
    extra = rng.uniform(0, extent, size=(int(rng.integers(0, 6)), 2))
    pred = np.vstack([pred, extra])
    if rng.random() < 0.15:  # ties / duplicates
        pred = np.round(pred)
        true = np.round(true)
    return true, pred


def test_matching_identical_to_oracle(oracle):
    pm, _ = oracle
    rng = np.random.default_rng(1234)
    n_checked = 0
    for _ in range(600):
        true, pred = _random_pair(rng)
        if len(true) == 0 or len(pred) == 0:
            continue
        tau = float(rng.choice([0.5, 1.0, 2.0]))
        r0, c0 = pm.match_points_mutual_nearest_neighbor(true, pred, threshold=tau)
        r1, c1 = match_mutual_nearest_neighbor(true, pred, tau)
        assert list(zip(r0, c0)) == list(zip(r1, c1))
        n_checked += 1
    assert n_checked >= 500


def test_scores_agree_with_oracle_stats_points(oracle):
    pm, _ = oracle
    rng = np.random.default_rng(99)
    for i in range(600):
        true, pred = _random_pair(rng)
        if i % 25 == 0:
            true = np.empty((0, 2))
        if i % 31 == 0:
            pred = np.empty((0, 2))
        tau = float(rng.choice([0.5, 1.0, 2.0]))
        ref = pm.stats_points(true, pred, tau)
        got = score_image(true, pred, tau)
        assert got.precision == pytest.approx(ref["precision"], abs=1e-12)
        assert got.recall == pytest.approx(ref["recall"], abs=1e-12)
        assert got.f1 == pytest.approx(ref["F1"], abs=1e-12)
        assert got.jac == pytest.approx(ref["JAC"], abs=1e-12)
        for a, b in ((got.rmse, ref["RMSE"]), (got.d_md, ref["d_md"])):
            if b is None:
                assert a is None
            else:
                assert a == pytest.approx(b, abs=1e-12)
