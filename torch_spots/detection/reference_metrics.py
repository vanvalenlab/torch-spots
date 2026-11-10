"""Reference metric set: the SpotNet (deepcell_spots 0.4.2) point-detection metrics.

These reproduce the definitions the published SpotNet benchmark was computed with, so
results are comparable with it. They are the only metrics used to judge a benchmark or a
divergence from the reference configuration. Everything else lives in
:mod:`torch_spots.detection.metrics` and is labelled additional.

Citations are ``deepcell_spots/<file>:<line>`` against the vendored 0.4.2 copy.

Points are ``[y, x]`` pixel coordinates; distances are Euclidean in pixels
(``deepcell_spots/utils/postprocessing_utils.py:117-119``).
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.spatial

#: Default distance threshold tau in pixels (``deepcell_spots/point_metrics.py:442-446``, ``:369``).
DEFAULT_TAU = 1.0

#: Names of the per-image reference metrics (used to keep additional metrics from colliding).
REFERENCE_FIELDS = (
    "tp", "fp", "fn", "n_true", "n_pred",
    "precision", "recall", "f1", "jac", "rmse", "d_md", "tau",
)


def _as_points(pts):
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 2)
    return pts.reshape(-1, 2)


def match_mutual_nearest_neighbor(true_pts, pred_pts, tau: float = DEFAULT_TAU):
    """Mutual-nearest-neighbour matching with an inclusive distance threshold.

    A true and a predicted point are matched only when each is the other's nearest
    neighbour and their distance is ``<= tau``; each point is matched at most once.
    Source: ``deepcell_spots/point_metrics.py:113-158`` (comparison at ``:153``).

    Returns:
        (true_idx, pred_idx): integer arrays of equal length, ordered by true index.
    """
    true_pts = _as_points(true_pts)
    pred_pts = _as_points(pred_pts)
    empty = np.array([], dtype=int)
    if len(true_pts) == 0 or len(pred_pts) == 0:
        return empty, empty

    tree_true = scipy.spatial.cKDTree(true_pts, leafsize=2)
    _, nearest_true_of_pred = tree_true.query(pred_pts)  # for each pred, nearest true
    tree_pred = scipy.spatial.cKDTree(pred_pts, leafsize=2)
    dist_to_nearest_pred, nearest_pred_of_true = tree_pred.query(true_pts)  # for each true

    mutual = nearest_true_of_pred[nearest_pred_of_true] == np.arange(len(true_pts))
    close = dist_to_nearest_pred <= tau
    matched = mutual & close
    return np.where(matched)[0], nearest_pred_of_true[matched]


def sum_of_min_distance(pts1, pts2) -> float:
    """Normalised sum of minimal distances d_md between two non-empty point sets:
    ``0.5 * (mean_i min_j d(p1_i, p2_j) + mean_j min_i d(p1_i, p2_j))``.
    Source: ``deepcell_spots/point_metrics.py:17-56`` (``normalized=True``).
    """
    pts1 = _as_points(pts1)
    pts2 = _as_points(pts2)
    d21, _ = scipy.spatial.cKDTree(pts1, leafsize=2).query(pts2)
    d12, _ = scipy.spatial.cKDTree(pts2, leafsize=2).query(pts1)
    return float(0.5 * (np.mean(d21) + np.mean(d12)))


@dataclass
class ReferenceScores:
    """Reference-set scores for one image. ``rmse`` and ``d_md`` are ``None`` when no pair is matched."""

    tau: float
    n_true: int
    n_pred: int
    tp: int  # matched pairs (point_metrics.py:329-330)
    fp: int  # n_pred - tp
    fn: int  # n_true - tp
    precision: float
    recall: float
    f1: float
    jac: float
    rmse: Optional[float]
    d_md: Optional[float]
    matched_true_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int), repr=False)
    matched_pred_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int), repr=False)

    def as_dict(self) -> dict:
        """Scalar fields only, named by :data:`REFERENCE_FIELDS`."""
        return {k: getattr(self, k) for k in REFERENCE_FIELDS}


def score_image(true_pts, pred_pts, tau: float = DEFAULT_TAU) -> ReferenceScores:
    """Reference scores for one image.

    - precision = TP / N_pred, recall = TP / N_true, F1 = 2PR/(P+R) (0 if P or R is 0),
      JAC = F1 / (2 - F1) (``point_metrics.py:329-342``).
    - RMSE = sqrt(sum(dy^2 + dx^2) / n_matched / 2) over matched pairs (``:349-351``).
    - d_md over matched points only (``:356-357``).
    - Empty true or predicted set: P = R = F1 = JAC = 0, RMSE and d_md undefined
      (``:308-323``, ``:345-347``).
    """
    true_pts = _as_points(true_pts)
    pred_pts = _as_points(pred_pts)
    n_true, n_pred = len(true_pts), len(pred_pts)

    if n_true == 0 or n_pred == 0:
        return ReferenceScores(tau, n_true, n_pred, 0, n_pred, n_true, 0.0, 0.0, 0.0, 0.0, None, None)

    row, col = match_mutual_nearest_neighbor(true_pts, pred_pts, tau)
    tp = len(row)
    precision = tp / n_pred
    recall = tp / n_true
    f1 = 0.0 if (precision == 0 or recall == 0) else 2 * precision * recall / (precision + recall)
    jac = f1 / (2 - f1)

    rmse = d_md = None
    if tp > 0:
        sq = np.sum((true_pts[row] - pred_pts[col]) ** 2)
        rmse = float(np.sqrt(sq / tp / 2))
        d_md = sum_of_min_distance(true_pts[row], pred_pts[col])

    return ReferenceScores(
        tau, n_true, n_pred, tp, n_pred - tp, n_true - tp,
        float(precision), float(recall), float(f1), float(jac), rmse, d_md, row, col,
    )


def aggregate(scores, tau: Optional[float] = None) -> dict:
    """Test-set aggregation of per-image :class:`ReferenceScores`.

    Macro (unweighted per-image) means of precision, recall, F1 and JAC over all images
    (``point_metrics.py:397-400``). RMSE and d_md are averaged over the images where they are
    defined, and the number of such images is reported. Summed TP/FP/FN are supplementary.
    """
    scores = list(scores)
    n_images = len(scores)
    if n_images == 0:
        raise ValueError("cannot aggregate zero images")
    taus = {s.tau for s in scores}
    if len(taus) != 1:
        raise ValueError(f"images were scored with different tau values: {sorted(taus)}")

    def macro(name):
        return float(np.mean([getattr(s, name) for s in scores]))

    def defined_mean(name):
        vals = [getattr(s, name) for s in scores if getattr(s, name) is not None]
        return (float(np.mean(vals)) if vals else None), len(vals)

    rmse, n_rmse = defined_mean("rmse")
    d_md, n_dmd = defined_mean("d_md")
    return {
        "tau": taus.pop(),
        "n_images": n_images,
        "precision": macro("precision"),
        "recall": macro("recall"),
        "f1": macro("f1"),
        "jac": macro("jac"),
        "rmse": rmse,
        "rmse_n_images": n_rmse,
        "d_md": d_md,
        "d_md_n_images": n_dmd,
        "supplementary_micro": {
            "tp": int(sum(s.tp for s in scores)),
            "fp": int(sum(s.fp for s in scores)),
            "fn": int(sum(s.fn for s in scores)),
        },
    }
