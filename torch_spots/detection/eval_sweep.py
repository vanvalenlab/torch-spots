"""Operating-point sweep over probability threshold, minimum peak distance and tau.

Every sweep is scored with the reference metric set and its grid always contains the reference
operating point (``EVAL_THRESHOLD``, 1 px, tau = 1 px); that point's result is reported separately
(``reference_point.json``). Sweeps are for exploration: only the reference point is a benchmark result.
"""
import json
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm

from torch_spots.dataset import DEFAULT_DATA_DIR, verify_store
from torch_spots.detection.eval import DEFAULT_MODEL_PATH, REFERENCE_OPERATING_POINT, SCORABLE_SPLITS
from torch_spots.detection.inference import SpotDetection
from torch_spots.detection.reference_metrics import aggregate, score_image

DEFAULT_THRESHOLDS = (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999)
DEFAULT_MIN_DISTANCES = (1, 2, 3, 4, 5)
DEFAULT_TAUS = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)


def _with_value(values, required):
    values = [v for v in values if not np.isclose(v, required)]
    return sorted(set(float(v) if isinstance(required, float) else v for v in values) | {required})


def build_grid(thresholds=DEFAULT_THRESHOLDS, min_distances=DEFAULT_MIN_DISTANCES, taus=DEFAULT_TAUS):
    """Cartesian grid of ``(threshold, min_distance, tau)`` that always contains the reference point."""
    ref = REFERENCE_OPERATING_POINT
    thresholds = _with_value(thresholds, ref["threshold"])
    min_distances = _with_value([int(m) for m in min_distances], ref["min_distance"])
    taus = _with_value(taus, ref["tau"])
    return [(t, m, tau) for t in thresholds for m in min_distances for tau in taus]


def is_reference_point(threshold, min_distance, tau):
    ref = REFERENCE_OPERATING_POINT
    return (np.isclose(threshold, ref["threshold"]) and min_distance == ref["min_distance"]
            and np.isclose(tau, ref["tau"]))


def run_sweep(model_path, data_path, device="cpu", grid=None, output_dir=None, split="test"):
    """Run the sweep and return the per-configuration DataFrame (reference point flagged).

    ``split`` is ``'test'`` or ``'val'``; the store is verified against that pinned split and
    ``reference_point.json`` records the split and its provenance.
    """
    if split not in SCORABLE_SPLITS:
        raise ValueError(f"split must be one of {SCORABLE_SPLITS}, got {split!r}")
    z_test = zarr.open(str(data_path), mode="r")
    provenance = verify_store(z_test, split)
    X, y, y_inds = z_test["X"], z_test["y"], z_test["y_inds"]
    detector = SpotDetection(model_path=str(model_path), device=device)
    grid = build_grid() if grid is None else build_grid(*grid) if isinstance(grid, tuple) else grid
    true_spots = [np.asarray(y[i, 0:int(y_inds[i])]) for i in range(X.shape[0])]

    by_op = {}
    for t, m in sorted({(t, m) for t, m, _ in grid}):
        by_op[(t, m)] = [detector._eval_predict(X[i:i + 1], threshold=t, min_distance=m)
                         for i in tqdm(range(X.shape[0]), desc=f"threshold={t} min_distance={m}", leave=False)]

    rows = []
    for t, m, tau in grid:
        scores = [score_image(true_spots[i], p, tau=tau) for i, p in enumerate(by_op[(t, m)])]
        agg = aggregate(scores)
        rows.append({"threshold": t, "min_distance": m, "tau": tau,
                     "is_reference_point": bool(is_reference_point(t, m, tau)),
                     **{k: agg[k] for k in ("n_images", "precision", "recall", "f1", "jac", "rmse",
                                            "rmse_n_images", "d_md", "d_md_n_images")},
                     **{f"supplementary_{k}": v for k, v in agg["supplementary_micro"].items()}})
    df = pd.DataFrame(rows)

    output_dir = Path(output_dir or Path("data/eval") / f"sweep_{datetime.now():%Y%m%d%H%M%S}")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "eval_sweep.csv", index=False)
    ref_row = df[df["is_reference_point"]].iloc[0].to_dict()
    with open(output_dir / "reference_point.json", "w") as fh:
        json.dump({"operating_point": REFERENCE_OPERATING_POINT, "split": split, "dataset": provenance,
                   "benchmark_eligible": split == "test", "result": ref_row}, fh, indent=2, default=float)
    return df


@click.command()
@click.option('--device', default='cuda:0', show_default=True, help="Device for inference.")
@click.option('--model-path', default=DEFAULT_MODEL_PATH, show_default=True, help="Path to the checkpoint.")
@click.option('--split', type=click.Choice(SCORABLE_SPLITS), default="test", show_default=True,
              help="Which pinned split to score ('val' is never a benchmark result).")
@click.option('--data-path', default=None,
              help="Converted SpotNet v1.1 store for the split (default: <data dir>/<split>.zarr).")
@click.option('--output-dir', default=None, help="Directory for eval_sweep.csv and reference_point.json.")
def main(device, model_path, split, data_path, output_dir):
    data_path = data_path or str(DEFAULT_DATA_DIR / f"{split}.zarr")
    df = run_sweep(model_path, data_path, device, output_dir=output_dir, split=split)
    ref = df[df["is_reference_point"]].iloc[0]
    click.echo(f"Reference point ({REFERENCE_OPERATING_POINT['threshold']}, 1 px, tau=1 px): F1 {ref['f1']:.4f}")


if __name__ == "__main__":
    main()
