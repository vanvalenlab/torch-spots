"""Benchmark evaluation (benchmark-evaluation capability).

Scores a trained detector on the pinned SpotNet v1.1 test split with the reference metric set
(:mod:`torch_spots.detection.reference_metrics`) at the reference operating point, and writes one
run directory containing:

- ``per_image.csv``: per-image reference metrics and, separately named, additional metrics;
- ``summary.json``: macro scores, supplementary micro totals, defined-image counts, operating
  point, code revision, configuration, seed and dataset provenance.
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd
import zarr

from torch_spots.dataset import DEFAULT_DATA_DIR, verify_store
from torch_spots.detection.dotnet import MODEL_ARCHITECTURE_VERSION, IncompatibleCheckpointError
from torch_spots.detection.inference import EVAL_MIN_DISTANCE, EVAL_THRESHOLD, SpotDetection
from torch_spots.detection.metrics import PointMetrics
from torch_spots.detection.reference_metrics import DEFAULT_TAU, REFERENCE_FIELDS, aggregate, score_image

#: Identifies the scoring protocol. Reports refuse to combine runs with different values.
PROTOCOL = "spotnet-v1.1/reference-metrics/1"

REFERENCE_OPERATING_POINT = {
    "threshold": EVAL_THRESHOLD,
    "min_distance": EVAL_MIN_DISTANCE,
    "tau": DEFAULT_TAU,
}

DEFAULT_MODEL_PATH = "data/model/20260513091742/saved_model_best_dict.pth"


def git_revision():
    """Return ``{'commit': sha or None, 'dirty': bool or None}`` for the torch_spots checkout."""
    here = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=here,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def read_training_provenance(model_path):
    """Read ``training_config.json`` next to the checkpoint.

    Returns ``(training_record_or_None, unknown_provenance)``. A record without a seed or
    dataset provenance counts as unknown (not benchmark-eligible).
    """
    cfg_path = Path(model_path).resolve().parent / "training_config.json"
    if not cfg_path.is_file():
        return None, True
    with open(cfg_path) as fh:
        record = json.load(fh)
    ok = "seed" in record and isinstance(record.get("dataset"), dict) and \
        all(s in record["dataset"] for s in ("train", "val"))
    return record, not ok


#: Splits that can be scored. Only ``test`` results are benchmark results.
SCORABLE_SPLITS = ("test", "val")


def run_evaluation(model_path, data_path, device="cpu", threshold=EVAL_THRESHOLD,
                   min_distance=EVAL_MIN_DISTANCE, tau=DEFAULT_TAU, output_dir=None, split="test"):
    """Evaluate ``model_path`` on the v1.1 ``split`` store at ``data_path``; return the summary dict.

    ``split`` is ``'test'`` (the default, a benchmark result) or ``'val'`` (for choosing the
    operating point; never benchmark-eligible). Raises
    :class:`torch_spots.dataset.DatasetMismatchError` (before any scoring) if the store is not
    the pinned v1.1 split.
    """
    if split not in SCORABLE_SPLITS:
        raise ValueError(f"split must be one of {SCORABLE_SPLITS}, got {split!r}")
    z_test = zarr.open(str(data_path), mode="r")
    test_provenance = verify_store(z_test, split)

    training, unknown = read_training_provenance(model_path)

    X, y, y_inds = z_test["X"], z_test["y"], z_test["y_inds"]
    detector = SpotDetection(model_path=str(model_path), device=device)

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

    additional_cols = [c for c in additional_names]
    summary = {
        "protocol": PROTOCOL,
        "n_images_scored": len(ref_scores),
        "operating_point": {"threshold": float(threshold), "min_distance": min_distance, "tau": float(tau)},
        "reference_operating_point": REFERENCE_OPERATING_POINT,
        "reference_metrics": {k: v for k, v in agg.items() if k not in ("supplementary_micro", "tau", "n_images")},
        "supplementary_micro": agg["supplementary_micro"],
        "additional_metrics_macro_mean": {c: float(df[c].mean()) for c in additional_cols},
        "unknown_provenance": unknown,
        "benchmark_eligible": split == "test" and (not unknown) and (
            {"threshold": float(threshold), "min_distance": min_distance, "tau": float(tau)}
            == REFERENCE_OPERATING_POINT),
        "model_architecture_version": MODEL_ARCHITECTURE_VERSION,
        "seed": None if training is None else training.get("seed"),
        "training": training,
        "split": split,
        f"{split}_dataset": test_provenance,
        "model_path": str(model_path),
        "evaluation_config": {"device": device, "data_path": str(data_path)},
        "code": git_revision(),
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
@click.option('--model-path', default=DEFAULT_MODEL_PATH, show_default=True,
              help="Path to the checkpoint state dict. Training provenance is read from "
                   "training_config.json in the same directory; checkpoints without it are marked "
                   "unknown_provenance and are not benchmark-eligible.")
@click.option('--split', type=click.Choice(SCORABLE_SPLITS), default="test", show_default=True,
              help="Which pinned split to score. 'val' is for choosing the operating point and is "
                   "never a benchmark result.")
@click.option('--data-path', default=None,
              help="Converted SpotNet v1.1 store for the split (default: <data dir>/<split>.zarr); "
                   "verified against the pinned split.")
@click.option('--threshold', default=EVAL_THRESHOLD, show_default=True, help="Spot probability threshold.")
@click.option('--min-distance', default=EVAL_MIN_DISTANCE, show_default=True, type=int,
              help="Minimum peak separation in pixels.")
@click.option('--tau', default=DEFAULT_TAU, show_default=True, help="Matching distance threshold in pixels.")
@click.option('--output-dir', default=None, help="Run directory to write (default: data/eval/<timestamp>).")
def main(device, model_path, split, data_path, threshold, min_distance, tau, output_dir):
    data_path = data_path or str(DEFAULT_DATA_DIR / f"{split}.zarr")
    try:
        summary = run_evaluation(model_path, data_path, device, threshold, min_distance, tau, output_dir,
                                 split=split)
    except IncompatibleCheckpointError as err:
        raise click.ClickException(str(err))
    ref = summary["reference_metrics"]
    click.echo(f"Scored {summary['n_images_scored']} {split} images at {summary['operating_point']}")
    click.echo(f"F1 {ref['f1']:.4f}  precision {ref['precision']:.4f}  recall {ref['recall']:.4f}  "
               f"JAC {ref['jac']:.4f}")
    if split != "test":
        click.echo(f"NOTE: {split} split - not a benchmark result.")
    if summary["unknown_provenance"]:
        click.echo("NOTE: unknown_provenance - not benchmark-eligible.")
    click.echo(f"Wrote {summary['output_dir']}")


if __name__ == "__main__":
    main()
