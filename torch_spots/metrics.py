"""
PointMetrics
============
Evaluates a set of inferred 2-D points against a set of ground-truth 2-D
points using optimal bipartite matching (Hungarian algorithm) plus a suite
of spatial error metrics.

Inputs
------
true_pts  : array-like, shape (M, 2)   – ground-truth locations
inferred_pts : array-like, shape (N, 2) – model predictions

Both arrays may have different lengths (M ≠ N).  Coordinates can be
sub-pixel floats.

Dependencies
------------
numpy, scipy  (scipy.optimize.linear_sum_assignment for the Hungarian solver)
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PointMetricsResult:
    """All computed metrics, plus the raw matched/unmatched indices."""

    # --- matching parameters ------------------------------------------------
    threshold: float                    # τ used for this result

    # --- confusion matrix ---------------------------------------------------
    TP: int
    FP: int
    FN: int

    # --- derived detection metrics ------------------------------------------
    precision: float                    # TP / (TP + FP)
    recall: float                       # TP / (TP + FN)
    f1: float                           # harmonic mean of precision & recall

    # --- spatial error on matched pairs (pixels) ----------------------------
    mean_dist: float
    median_dist: float
    rmse: float
    p95_dist: float                     # 95th-percentile distance
    max_dist: float
    bias_x: float                       # mean signed Δx  (inferred − true)
    bias_y: float                       # mean signed Δy  (inferred − true)

    # --- threshold-free spatial summary -------------------------------------
    chamfer_distance: float             # mean of both directed nearest-neighbour means

    # --- raw index arrays (into the original input arrays) -----------------
    matched_true_idx: np.ndarray        # shape (TP,)
    matched_inferred_idx: np.ndarray    # shape (TP,)
    matched_distances: np.ndarray       # shape (TP,)  – per-pair Euclidean distance
    unmatched_true_idx: np.ndarray      # FN indices into true_pts
    unmatched_inferred_idx: np.ndarray  # FP indices into inferred_pts

    def summary(self) -> str:
        lines = [
            f"Threshold τ = {self.threshold:.2f} px",
            "",
            "Detection",
            f"  TP        : {self.TP}",
            f"  FP        : {self.FP}",
            f"  FN        : {self.FN}",
            f"  Precision : {self.precision:.4f}",
            f"  Recall    : {self.recall:.4f}",
            f"  F1        : {self.f1:.4f}",
            "",
            "Spatial error (matched pairs)",
            f"  Mean dist : {self.mean_dist:.4f} px",
            f"  Median    : {self.median_dist:.4f} px",
            f"  RMSE      : {self.rmse:.4f} px",
            f"  95th pct  : {self.p95_dist:.4f} px",
            f"  Max dist  : {self.max_dist:.4f} px",
            f"  Bias Δx   : {self.bias_x:+.4f} px",
            f"  Bias Δy   : {self.bias_y:+.4f} px",
            "",
            "Threshold-free",
            f"  Chamfer   : {self.chamfer_distance:.4f} px",
        ]
        return "\n".join(lines)
    
    def return_fields(self):

        fields = np.array([
            "true_positive",
            "false_positive",
            'false_negative',
            'precision',
            'recall',
            'f1',
            'mean_dist',
            'median_dist',
            'rmse',
            'p95',
            'max_dist',
            'bias_x',
            'bias_y',
            'chamfer'
        ])

        return fields
    
    def return_numpy(self):

        metrics = np.array([
            self.TP, 
            self.FP, 
            self.FN, 
            self.precision, 
            self.recall, 
            self.f1, 
            self.mean_dist,
            self.median_dist,
            self.rmse,
            self.p95_dist,
            self.max_dist,
            self.bias_x,
            self.bias_y,
            self.chamfer_distance
        ], dtype=float)

        return metrics

    def __repr__(self) -> str:
        return (
            f"PointMetricsResult(TP={self.TP}, FP={self.FP}, FN={self.FN}, "
            f"precision={self.precision:.3f}, recall={self.recall:.3f}, "
            f"f1={self.f1:.3f}, rmse={self.rmse:.3f} px, "
            f"chamfer={self.chamfer_distance:.3f} px)"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PointMetrics:
    """
    Compute detection and spatial-accuracy metrics for 2-D point sets.

    Parameters
    ----------
    true_pts : array-like, shape (M, 2)
    inferred_pts : array-like, shape (N, 2)

    Example
    -------
    >>> pm = PointMetrics(true_pts, inferred_pts)
    >>> result = pm.evaluate(threshold=2.5)
    >>> print(result.summary())

    # Sweep across thresholds
    >>> results = pm.evaluate_thresholds([0.5, 1.0, 2.0, 5.0, 10.0])
    """

    def __init__(
        self,
        true_pts: np.ndarray,
        inferred_pts: np.ndarray,
    ) -> None:
        
        self.M = len(true_pts)
        self.N = len(inferred_pts)

        self.true_pts = np.asarray(true_pts, dtype=float)
        self.inferred_pts = np.asarray(inferred_pts, dtype=float)


        # Pre-compute the full pairwise distance matrix (M × N) once
        self._dist_matrix: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def dist_matrix(self) -> np.ndarray:
        """Cached (M, N) pairwise Euclidean distance matrix."""
        if self._dist_matrix is None:
            if len(self.true_pts) == 0 or len(self.inferred_pts) == 0:
                self._dist_matrix = np.empty((len(self.true_pts), len(self.inferred_pts)))
            else:
                self._dist_matrix = cdist(self.true_pts, self.inferred_pts, metric="euclidean")
        return self._dist_matrix

    def _optimal_matching(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the Hungarian algorithm on the full cost matrix.

        Returns
        -------
        true_idx   : (K,) indices into true_pts
        infer_idx  : (K,) indices into inferred_pts  (K = min(M, N))
        distances  : (K,) Euclidean distance for each pair
        """
        if len(self.true_pts) == 0 or len(self.inferred_pts) == 0:
            return np.array([], int), np.array([], int), np.array([], float)

        row_ind, col_ind = linear_sum_assignment(self.dist_matrix)
        distances = self.dist_matrix[row_ind, col_ind]
        return row_ind, col_ind, distances

    @staticmethod
    def _safe_div(num: float, den: float) -> float:
        return num / den if den > 0 else 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, threshold: float) -> PointMetricsResult:
        """
        Evaluate all metrics using a distance threshold τ.

        A matched pair (true_i, inferred_j) is a True Positive only when
        their Euclidean distance ≤ threshold.  Unmatched or over-threshold
        pairs contribute to FP / FN.

        Parameters
        ----------
        threshold : float
            Maximum pixel distance for a valid match (τ).

        Returns
        -------
        PointMetricsResult
        """

        # --- Handle degenerate cases ----------------------------------------
        if self.M == 0 and self.N == 0:
            return self._empty_result(threshold)
        if self.M == 0:
            return PointMetricsResult(
                threshold=threshold, TP=0, FP=self.N, FN=0,
                precision=0.0, recall=0.0, f1=0.0,
                mean_dist=0.0, median_dist=0.0, rmse=0.0,
                p95_dist=0.0, max_dist=0.0, bias_x=0.0, bias_y=0.0,
                chamfer_distance=self._chamfer(),
                matched_true_idx=np.array([], int),
                matched_inferred_idx=np.array([], int),
                matched_distances=np.array([], float),
                unmatched_true_idx=np.array([], int),
                unmatched_inferred_idx=np.arange(self.N),
            )
        if self.N == 0:
            return PointMetricsResult(
                threshold=threshold, TP=0, FP=0, FN=self.M,
                precision=0.0, recall=0.0, f1=0.0,
                mean_dist=0.0, median_dist=0.0, rmse=0.0,
                p95_dist=0.0, max_dist=0.0, bias_x=0.0, bias_y=0.0,
                chamfer_distance=self._chamfer(),
                matched_true_idx=np.array([], int),
                matched_inferred_idx=np.array([], int),
                matched_distances=np.array([], float),
                unmatched_true_idx=np.arange(self.M),
                unmatched_inferred_idx=np.array([], int),
            )

        # --- Hungarian matching ---------------------------------------------
        row_ind, col_ind, dists = self._optimal_matching()

        valid_mask = dists <= threshold
        tp_true_idx  = row_ind[valid_mask]
        tp_infer_idx = col_ind[valid_mask]
        tp_dists     = dists[valid_mask]

        TP = int(valid_mask.sum())

        # Unmatched true  → FN
        unmatched_true_set  = set(range(self.M)) - set(tp_true_idx.tolist())
        # Over-threshold pairs also contribute their true side to FN
        over_true  = set(row_ind[~valid_mask].tolist())
        over_infer = set(col_ind[~valid_mask].tolist())
        # Points not considered by the rectangular matching at all
        extra_true  = set(range(self.M))  - set(row_ind.tolist())
        extra_infer = set(range(self.N))  - set(col_ind.tolist())

        fn_indices = np.array(sorted(unmatched_true_set | over_true | extra_true), dtype=int)
        fp_indices = np.array(sorted(
            (set(range(self.N)) - set(tp_infer_idx.tolist())) | extra_infer
        ), dtype=int)

        FN = len(fn_indices)
        FP = len(fp_indices)

        # --- Detection metrics ----------------------------------------------
        precision = self._safe_div(TP, TP + FP)
        recall    = self._safe_div(TP, TP + FN)
        f1        = self._safe_div(2 * precision * recall, precision + recall)

        # --- Spatial error --------------------------------------------------
        if TP > 0:
            mean_dist   = float(np.mean(tp_dists))
            median_dist = float(np.median(tp_dists))
            rmse        = float(np.sqrt(np.mean(tp_dists ** 2)))
            p95_dist    = float(np.percentile(tp_dists, 95))
            max_dist    = float(np.max(tp_dists))
            delta       = self.inferred_pts[tp_infer_idx] - self.true_pts[tp_true_idx]
            bias_x      = float(np.mean(delta[:, 0]))
            bias_y      = float(np.mean(delta[:, 1]))
        else:
            mean_dist = median_dist = rmse = p95_dist = max_dist = 0.0
            bias_x = bias_y = 0.0

        return PointMetricsResult(
            threshold=threshold,
            TP=TP, FP=FP, FN=FN,
            precision=precision, recall=recall, f1=f1,
            mean_dist=mean_dist, median_dist=median_dist, rmse=rmse,
            p95_dist=p95_dist, max_dist=max_dist,
            bias_x=bias_x, bias_y=bias_y,
            chamfer_distance=self._chamfer(),
            matched_true_idx=tp_true_idx,
            matched_inferred_idx=tp_infer_idx,
            matched_distances=tp_dists,
            unmatched_true_idx=fn_indices,
            unmatched_inferred_idx=fp_indices,
        )

    def evaluate_thresholds(
        self,
        thresholds: list[float],
    ) -> list[PointMetricsResult]:
        """
        Evaluate metrics across multiple τ values.

        Returns a list of PointMetricsResult, one per threshold, in the
        same order as the input list.  Useful for precision-recall curves.
        """
        return [self.evaluate(tau) for tau in thresholds]

    def precision_recall_curve(
        self,
        thresholds: list[float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convenience method: returns (thresholds, precisions, recalls) as
        numpy arrays, ready for plotting.
        """
        results = self.evaluate_thresholds(thresholds)
        taus  = np.array(thresholds)
        precs = np.array([r.precision for r in results])
        recs  = np.array([r.recall    for r in results])
        return taus, precs, recs

    # ------------------------------------------------------------------
    # Threshold-free metric
    # ------------------------------------------------------------------

    def _chamfer(self) -> float:
        """
        Chamfer distance: mean of both directed nearest-neighbour averages.

          C = 0.5 * (mean_i min_j d(T_i, I_j)  +  mean_j min_i d(I_j, T_i))

        Returns 0.0 if either set is empty.
        """
        if len(self.true_pts) == 0 or len(self.inferred_pts) == 0:
            return 0.0
        D = self.dist_matrix                     # (M, N)
        t2i = float(D.min(axis=1).mean())        # each true point → nearest inferred
        i2t = float(D.min(axis=0).mean())        # each inferred point → nearest true
        return (t2i + i2t) / 2.0

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result(threshold: float) -> PointMetricsResult:
        empty = np.array([], dtype=int)
        return PointMetricsResult(
            threshold=threshold, TP=0, FP=0, FN=0,
            precision=0.0, recall=0.0, f1=0.0,
            mean_dist=0.0, median_dist=0.0, rmse=0.0,
            p95_dist=0.0, max_dist=0.0, bias_x=0.0, bias_y=0.0,
            chamfer_distance=0.0,
            matched_true_idx=empty, matched_inferred_idx=empty,
            matched_distances=np.array([], float),
            unmatched_true_idx=empty, unmatched_inferred_idx=empty,
        )

    def __repr__(self) -> str:
        return (
            f"PointMetrics(M={len(self.true_pts)} true pts, "
            f"N={len(self.inferred_pts)} inferred pts)"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    true_pts     = rng.uniform(0, 512, size=(30, 2))
    inferred_pts = true_pts[:25] + rng.normal(0, 3, size=(25, 2))  # 5 FN, small noise

    pm = PointMetrics(true_pts, inferred_pts)
    result = pm.evaluate(threshold=5.0)
    print(result.summary())
    print()

    # Precision-recall curve across thresholds
    taus, precs, recs = pm.precision_recall_curve([0.5, 1, 2, 5, 10, 20, 50])
    print("τ (px)  Precision  Recall    F1")
    for tau, p, r in zip(taus, precs, recs):
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        print(f"  {tau:5.1f}   {p:.3f}      {r:.3f}     {f1:.3f}")