"""Evaluate a detection model on the SpotNet v1.1 test (or validation) split.

Scores every image with the SpotNet point metrics (:mod:`torch_spots.detection.reference_metrics`)
and writes a run directory containing:

- ``per_image.csv``: per-image metrics, plus the additional metrics from
  :mod:`torch_spots.detection.metrics`;
- ``summary.json``: macro-averaged scores, micro TP/FP/FN totals, the operating point and the
  dataset checked.
"""
import json
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd
import zarr

from torch_spots.dataset import DEFAULT_DATA_DIR, verify_store
from torch_spots.detection.inference import EVAL_MIN_DISTANCE, EVAL_THRESHOLD, SpotDetection
from torch_spots.detection.metrics import PointMetrics
from torch_spots.detection.reference_metrics import DEFAULT_TAU, REFERENCE_FIELDS, aggregate, score_image

#: Splits that can be scored.
SCORABLE_SPLITS = ("test", "val")

#: Values ``norm_method`` accepts on the command line (``none`` is no normalisation).
NORM_METHODS = ("none", "std", "whole_image")


def run_evaluation(model_path, data_path, device="cpu", threshold=EVAL_THRESHOLD,
                   min_distance=EVAL_MIN_DISTANCE, tau=DEFAULT_TAU, output_dir=None, split="test",
                   norm_method=None):
    """Evaluate ``model_path`` on the v1.1 ``split`` store at ``data_path``; return the summary dict.

    ``model_path=None`` evaluates the published pretrained model. ``norm_method`` is the in-model
    normalisation the checkpoint was trained with (``None`` for the published model). Raises
    :class:`torch_spots.dataset.DatasetMismatchError` (before any scoring) if the store is not
    the pinned v1.1 split.
    """
    if split not in SCORABLE_SPLITS:
        raise ValueError(f"split must be one of {SCORABLE_SPLITS}, got {split!r}")
    z = zarr.open(str(data_path), mode="r")
    dataset = verify_store(z, split)

    X, y, y_inds = z["X"], z["y"], z["y_inds"]
    detector = SpotDetection(model_path=model_path, device=device, norm_method=norm_method)

    rows, ref_scores = [], []
    additional_names = PointMetrics._empty_result(threshold=tau).return_fields()
    for i in range(X.shape[0]):
        pred = detector._eval_predict(X[i:i + 1], threshold=threshold, min_distance=min_distance)
        true = np.asarray(y[i, 0:int(y_inds[i])])
        ref = score_image(true, pred, tau=tau)
        add = PointMetrics(true, pred).evaluate(threshold=tau).return_numpy()
        ref_scores.append(ref)
        rows.append({"image_id": i, **ref.as_dict(), **dict(zip(additional_names, add))})

    df = pd.DataFrame(rows)
    agg = aggregate(ref_scores)

    summary = {
        "n_images_scored": len(ref_scores),
        "split": split,
        "operating_point": {"threshold": float(threshold), "min_distance": min_distance, "tau": float(tau)},
        "reference_metrics": {k: v for k, v in agg.items() if k not in ("supplementary_micro", "tau", "n_images")},
        "supplementary_micro": agg["supplementary_micro"],
        "additional_metrics_macro_mean": {c: float(df[c].mean()) for c in additional_names},
        "model_path": str(detector.model_path),
        "norm_method": norm_method,
        "dataset": dataset,
        "reference_fields": list(REFERENCE_FIELDS),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    if output_dir is None:
        output_dir = Path("data/eval") / datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "per_image.csv", index=False)
    with open(output_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    summary["output_dir"] = str(output_dir)
    return summary


@click.command()
@click.option('--device', default='cuda:0', show_default=True,
              help="Device to run inference on: 'cpu', 'cuda', or a specific GPU such as 'cuda:1'.")
@click.option('--model-path', default=None,
              help="Path to a SpotNet state dict (default: the published pretrained model).")
@click.option('--norm-method', type=click.Choice(NORM_METHODS), default="none", show_default=True,
              help="In-model normalisation the checkpoint was trained with.")
@click.option('--split', type=click.Choice(SCORABLE_SPLITS), default="test", show_default=True,
              help="Which pinned split to score.")
@click.option('--data-path', default=None,
              help="Converted SpotNet v1.1 store for the split (default: <data dir>/<split>.zarr); "
                   "verified against the pinned split.")
@click.option('--threshold', default=EVAL_THRESHOLD, show_default=True, help="Spot probability threshold.")
@click.option('--min-distance', default=EVAL_MIN_DISTANCE, show_default=True, type=int,
              help="Minimum peak separation in pixels.")
@click.option('--tau', default=DEFAULT_TAU, show_default=True, help="Matching distance threshold in pixels.")
@click.option('--output-dir', default=None, help="Run directory to write (default: data/eval/<timestamp>).")
def main(device, model_path, norm_method, split, data_path, threshold, min_distance, tau, output_dir):
    data_path = data_path or str(DEFAULT_DATA_DIR / f"{split}.zarr")
    summary = run_evaluation(model_path, data_path, device, threshold, min_distance, tau, output_dir,
                             split=split, norm_method=None if norm_method == "none" else norm_method)
    ref = summary["reference_metrics"]
    click.echo(f"Scored {summary['n_images_scored']} {split} images at {summary['operating_point']}")
    click.echo(f"F1 {ref['f1']:.4f}  precision {ref['precision']:.4f}  recall {ref['recall']:.4f}  "
               f"JAC {ref['jac']:.4f}")
    click.echo(f"Wrote {summary['output_dir']}")


if __name__ == "__main__":
    main()
