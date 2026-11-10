"""Per-configuration benchmark report over three seeds, and divergence verdicts.

``build_report`` combines the run directories written by :mod:`torch_spots.detection.eval` for one
configuration (seeds 0, 1, 2). ``compare`` applies the evidence bar to a candidate divergence
against a reference report (benchmark-evaluation: accuracy / quality-of-life divergence evidence).
Only reference-set F1 is used.
"""
import json
from pathlib import Path

import click
import numpy as np
import zarr

from torch_spots.dataset import DEFAULT_DATA_DIR, verify_store
from torch_spots.detection.eval import PROTOCOL, REFERENCE_OPERATING_POINT
from torch_spots.detection.reference_metrics import DEFAULT_TAU, aggregate, score_image

SEEDS = (0, 1, 2)
_EPS = 1e-9

#: The published SpotDetection-8 model's mean reference-set F1 on the pinned test split at the
#: reference operating point, measured with ``tools/published_model/predict_published.py`` and
#: ``report published``. The report stops with an error if this is ever unset.
PUBLISHED_MODEL_F1 = 0.8100009821287942
#: Where :data:`PUBLISHED_MODEL_F1` came from (model archive, versions, operating point, dataset).
PUBLISHED_MODEL_PROVENANCE = {
    "model": "SpotDetection-8",
    "model_archive_sha256": "7aba51b9ea198e225abff220057a090f6dbb6dcc87448c8bffbc91ee1db3ae36",
    "application": "deepcell_spots 0.4.2 SpotDetection.predict, default preprocessing (percentile clip on, reflect padding)",
    "versions": {"deepcell_spots": "0.4.2", "deepcell": "0.12.10", "tensorflow": "2.8.4"},
    "operating_point": {"threshold": 0.993, "min_distance": 1, "tau": 1.0},
    "dataset": {"dataset_version": "SpotNet-v1_1", "split": "test", "source_crc32": "36c922d7", "n_images": 94},
    "source": "data/eval/published_spotdetection8/summary.json (per_threshold['0.993'])",
}

#: training-config keys that may differ between seeds of one configuration
_PER_RUN_KEYS = {"seed", "dataset", "device", "model_path", "run_info", "data_path",
                  "best_epoch", "best_val_objective"}


class ReportError(ValueError):
    """The runs cannot be combined into a benchmark report."""


def _load_summary(run_dir):
    path = Path(run_dir) / "summary.json"
    if not path.is_file():
        raise ReportError(f"{run_dir}: no summary.json")
    with open(path) as fh:
        return json.load(fh)


def _published_model(published):
    """The published-model score to judge against: ``published`` (``{"f1", "provenance"}``) if given,
    else the recorded constants. Raises :class:`ReportError` while no score is recorded."""
    if published is None:
        published = {"f1": PUBLISHED_MODEL_F1, "provenance": PUBLISHED_MODEL_PROVENANCE}
    if published.get("f1") is None:
        raise ReportError("the published-model score is not recorded yet (PUBLISHED_MODEL_F1 is unset); "
                          "measure it with tools/published_model/predict_published.py and "
                          "`report published`, then record it in torch_spots/detection/report.py")
    provenance = dict(published.get("provenance") or {})
    point = provenance.get("operating_point")
    if point is not None and point != REFERENCE_OPERATING_POINT:
        raise ReportError(f"the published-model score was measured at {point}, not at the reference "
                          f"operating point {REFERENCE_OPERATING_POINT}; the two must match")
    return {"f1": float(published["f1"]), "provenance": provenance}


def build_report(run_dirs, published=None):
    """Combine exactly three seed runs of one configuration into a report dict.

    The configuration reproduces the published model when its mean F1 is at least the published
    model's score (a one-sided test: scoring higher does not count against it). ``published`` is
    ``{"f1": ..., "provenance": ...}`` and defaults to the recorded constants.
    """
    published = _published_model(published)
    run_dirs = [str(d) for d in run_dirs]
    if len(run_dirs) != len(SEEDS):
        raise ReportError(f"expected {len(SEEDS)} run directories (one per seed {SEEDS}), got {len(run_dirs)}")
    summaries = [_load_summary(d) for d in run_dirs]

    for d, s in zip(run_dirs, summaries):
        if s.get("split", "test") != "test":
            raise ReportError(f"{d}: this is a {s['split']}-split result, not a benchmark result; "
                              "only test-split runs can be reported")
        if s.get("unknown_provenance", True):
            raise ReportError(f"{d}: unknown_provenance - not usable for a benchmark or divergence result")
        if s.get("protocol") != PROTOCOL:
            raise ReportError(f"{d}: protocol {s.get('protocol')!r} != {PROTOCOL!r}")
        if s.get("operating_point") != REFERENCE_OPERATING_POINT:
            raise ReportError(f"{d}: operating point {s.get('operating_point')} is not the reference "
                              f"{REFERENCE_OPERATING_POINT}")
        if s.get("n_images_scored") != s.get("test_dataset", {}).get("n_images"):
            raise ReportError(f"{d}: not all test images were scored")
        if not s.get("code", {}).get("commit"):
            raise ReportError(f"{d}: result is missing the code revision")

    seeds = [s["seed"] for s in summaries]
    if sorted(seeds) != list(SEEDS):
        raise ReportError(f"seeds {sorted(seeds, key=str)} do not match the protocol seeds {list(SEEDS)}")

    first = summaries[0]
    for d, s in zip(run_dirs, summaries):
        if s["test_dataset"] != first["test_dataset"]:
            raise ReportError(f"{d}: test dataset differs from the other runs")
        if s["training"]["dataset"] != first["training"]["dataset"]:
            raise ReportError(f"{d}: training dataset provenance differs from the other runs")
        cfg = {k: v for k, v in s["training"].items() if k not in _PER_RUN_KEYS}
        cfg0 = {k: v for k, v in first["training"].items() if k not in _PER_RUN_KEYS}
        if cfg != cfg0:
            raise ReportError(f"{d}: training configuration differs from the other runs (not one configuration)")

    ordered = sorted(zip(seeds, run_dirs, summaries), key=lambda t: t[0])
    f1s = [s["reference_metrics"]["f1"] for _, _, s in ordered]
    mean = sum(f1s) / len(f1s)
    rng = max(f1s) - min(f1s)
    reproduces = mean >= published["f1"] - _EPS
    return {
        "protocol": PROTOCOL,
        "operating_point": REFERENCE_OPERATING_POINT,
        "seeds": [t[0] for t in ordered],
        "per_seed_f1": f1s,
        "mean_f1": mean,
        "range_f1": rng,
        "published_model": published,
        "reproduces_published_model": bool(reproduces),
        "verdict": (f"PASS: mean F1 {mean:.4f} is at least the published model's {published['f1']:.4f}"
                    if reproduces else
                    f"FAIL: mean F1 {mean:.4f} is below the published model's {published['f1']:.4f}; "
                    "published model not reproduced"),
        "test_dataset": first["test_dataset"],
        "training_dataset": first["training"]["dataset"],
        "runs": [{"seed": sd, "run_dir": d, "f1": s["reference_metrics"]["f1"], "code": s["code"],
                  "training": s["training"], "evaluation_config": s["evaluation_config"],
                  "model_path": s["model_path"]} for sd, d, s in ordered],
    }


#: Metrics kept per threshold when scoring the published model.
_PUBLISHED_FIELDS = ("n_images", "precision", "recall", "f1", "jac", "rmse", "d_md")


def score_published(points_path, data_path=None, tau=DEFAULT_TAU):
    """Score the published model's points (``tools/published_model/predict_published.py``).

    ``points_path`` is the ``points.json`` that script wrote: ``{"provenance": ..., "points":
    {threshold: [points of image 0, points of image 1, ...]}}``. The test store is verified
    against the pinned test split first, and every threshold must have one entry per test image.
    Returns the summary dict (per-threshold reference metrics plus the provenance).
    """
    with open(points_path) as fh:
        record = json.load(fh)
    if not isinstance(record, dict) or "points" not in record or "provenance" not in record:
        raise ReportError(f"{points_path}: expected 'provenance' and 'points'; "
                          "write it with tools/published_model/predict_published.py")
    provenance, points = record["provenance"], record["points"]

    data_path = data_path or str(DEFAULT_DATA_DIR / "test.zarr")
    z = zarr.open(str(data_path), mode="r")
    test_provenance = verify_store(z, "test")
    n = int(z["X"].shape[0])
    if provenance.get("test_split", {}).get("crc32") != test_provenance["source_crc32"]:
        raise ReportError(f"{points_path}: points were made on a split with CRC-32 "
                          f"{provenance.get('test_split', {}).get('crc32')!r}, not the pinned test split "
                          f"{test_provenance['source_crc32']!r}")
    true_spots = [np.asarray(z["y"][i, 0:int(z["y_inds"][i])]) for i in range(n)]

    per_threshold = {}
    for key in sorted(points, key=float):
        pred = points[key]
        if len(pred) != n:
            raise ReportError(f"{points_path}: threshold {key} has {len(pred)} images, expected {n}")
        agg = aggregate([score_image(true_spots[i], np.asarray(pred[i], dtype=float).reshape(-1, 2), tau=tau)
                         for i in range(n)])
        per_threshold[key] = {k: agg[k] for k in _PUBLISHED_FIELDS}
    return {
        "protocol": PROTOCOL,
        "kind": "published_model",
        "tau": float(tau),
        "n_images_scored": n,
        "test_dataset": test_provenance,
        "per_threshold": per_threshold,
        "provenance": provenance,
        "points_file": str(points_path),
    }


def _check_comparable(ref, cand):
    if ref["protocol"] != cand["protocol"] or ref["operating_point"] != cand["operating_point"]:
        raise ReportError("reference and candidate use different protocols or operating points")
    if ref["seeds"] != cand["seeds"]:
        raise ReportError(f"reference seeds {ref['seeds']} != candidate seeds {cand['seeds']}")
    if ref["test_dataset"] != cand["test_dataset"] or ref["training_dataset"] != cand["training_dataset"]:
        raise ReportError("reference and candidate used different datasets")


def _benefit_measured(benefit):
    return isinstance(benefit, dict) and bool(benefit.get("metric")) and \
        isinstance(benefit.get("value"), (int, float)) and not isinstance(benefit.get("value"), bool)


def compare(reference, candidate, benefit=None):
    """Verdicts for a candidate divergence against a reference report.

    - Accuracy: supported only if candidate mean F1 exceeds the reference mean by more than the
      reference's seed range (strictly).
    - Quality of life: adoptable only if candidate mean F1 >= reference mean - reference range AND
      ``benefit`` is a measurement, ``{"metric": <name>, "value": <number>}``.
    - If the reference does not reproduce the published model, no divergence is adopted against it.
    """
    _check_comparable(reference, candidate)
    margin = reference["range_f1"]
    delta = candidate["mean_f1"] - reference["mean_f1"]

    if not reference["reproduces_published_model"]:
        reason = "reference does not reproduce the published model; no divergence is adopted against it"
        return {"delta_mean_f1": delta, "margin": margin, "accuracy_supported": False,
                "quality_of_life_adoptable": False, "accuracy_reason": reason, "quality_of_life_reason": reason}

    acc = delta - margin > _EPS
    non_inferior = delta + margin >= -_EPS
    measured = _benefit_measured(benefit)
    return {
        "delta_mean_f1": delta,
        "margin": margin,
        "accuracy_supported": bool(acc),
        "accuracy_reason": (f"mean F1 exceeds the reference mean by {delta:.4f} > seed range {margin:.4f}" if acc
                            else f"mean F1 difference {delta:+.4f} does not exceed seed range {margin:.4f}"),
        "quality_of_life_adoptable": bool(non_inferior and measured),
        "quality_of_life_reason": (
            "non-inferior and benefit measured" if non_inferior and measured else
            "benefit not measured" if non_inferior else
            f"mean F1 is {-delta:.4f} below the reference mean, more than the seed range {margin:.4f}"),
    }


def format_report(report):
    lines = [report["verdict"], "",
             f"protocol {report['protocol']}   operating point {report['operating_point']}", ""]
    for r in report["runs"]:
        lines.append(f"  seed {r['seed']}: F1 {r['f1']:.4f}   commit {str(r['code']['commit'])[:10]}")
    lines += ["", f"  mean F1 {report['mean_f1']:.4f}   range {report['range_f1']:.4f}"]
    return "\n".join(lines)


@click.group()
def main():
    """Combine seed runs into a benchmark report, or compare two reports."""


@main.command("report")
@click.argument("run_dirs", nargs=-1, required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--output", default="report.json", show_default=True, help="Where to write the report JSON.")
def report_cmd(run_dirs, output):
    """Build the 3-seed report from RUN_DIRS (one eval run directory per seed)."""
    try:
        rep = build_report(run_dirs)
    except ReportError as e:
        raise click.ClickException(str(e))
    with open(output, "w") as fh:
        json.dump(rep, fh, indent=2)
    click.echo(format_report(rep))
    click.echo(f"\nWrote {output}")


@main.command("published")
@click.argument("points_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--data-path", default=None,
              help="Converted SpotNet v1.1 test store (default: <data dir>/test.zarr); verified against the pin.")
@click.option("--output", default=None, help="Where to write summary.json (default: next to POINTS_FILE).")
def published_cmd(points_file, data_path, output):
    """Score the published model's POINTS_FILE on the pinned test split at every threshold."""
    try:
        summary = score_published(points_file, data_path)
    except ReportError as e:
        raise click.ClickException(str(e))
    output = output or str(Path(points_file).with_name("summary.json"))
    with open(output, "w") as fh:
        json.dump(summary, fh, indent=2)
    click.echo(f"Scored {summary['n_images_scored']} test images")
    for key, m in summary["per_threshold"].items():
        click.echo(f"  threshold {key:>6}: F1 {m['f1']:.4f}  precision {m['precision']:.4f}  "
                   f"recall {m['recall']:.4f}")
    click.echo(f"Wrote {output}")


@main.command("compare")
@click.argument("reference", type=click.Path(exists=True, dir_okay=False))
@click.argument("candidate", type=click.Path(exists=True, dir_okay=False))
@click.option("--benefit-metric", default=None, help="Name of the measured benefit, e.g. training_time_reduction.")
@click.option("--benefit-value", default=None, type=float, help="Measured value of the benefit.")
def compare_cmd(reference, candidate, benefit_metric, benefit_value):
    """Accuracy and quality-of-life verdicts for CANDIDATE against REFERENCE report JSON files."""
    ref, cand = (json.load(open(p)) for p in (reference, candidate))
    benefit = None if benefit_metric is None or benefit_value is None else \
        {"metric": benefit_metric, "value": benefit_value}
    try:
        res = compare(ref, cand, benefit)
    except ReportError as e:
        raise click.ClickException(str(e))
    click.echo(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
