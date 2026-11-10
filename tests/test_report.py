import copy
import json

import pytest

from torch_spots.detection.eval import PROTOCOL, REFERENCE_OPERATING_POINT
from torch_spots.detection import report as report_mod
from torch_spots.detection.report import ReportError, build_report, compare

TEST_DS = {"dataset_version": "SpotNet-v1_1", "split": "test", "source_crc32": "36c922d7",
           "n_images": 94, "n_spots": 995}
TRAIN_DS = {"train": {"source_crc32": "4cb57ba8"}, "val": {"source_crc32": "29836378"}}


def write_run(root, name, seed, f1, **over):
    d = root / name
    d.mkdir(parents=True)
    s = {
        "protocol": PROTOCOL, "n_images_scored": 94, "operating_point": dict(REFERENCE_OPERATING_POINT),
        "reference_metrics": {"f1": f1}, "unknown_provenance": False, "seed": seed,
        "training": {"seed": seed, "dataset": TRAIN_DS, "lr": 1e-3, "epochs": 20},
        "test_dataset": TEST_DS, "model_path": "m.pth", "evaluation_config": {},
        "code": {"commit": "abc123", "dirty": False},
    }
    s.update(over)
    json.dump(s, open(d / "summary.json", "w"))
    return d


def runs(tmp_path, tag, f1s, seeds=(0, 1, 2)):
    return [write_run(tmp_path, f"{tag}{s}", s, f) for s, f in zip(seeds, f1s)]


@pytest.fixture(autouse=True)
def recorded_published_score(monkeypatch):
    """A stand-in for the recorded published-model score, so the plumbing tests do not depend on it."""
    monkeypatch.setattr(report_mod, "PUBLISHED_MODEL_F1", 0.82)
    monkeypatch.setattr(report_mod, "PUBLISHED_MODEL_PROVENANCE", {"operating_point": dict(REFERENCE_OPERATING_POINT)})


PUBLISHED = {"f1": 0.842, "provenance": {"model": "SpotDetection-8",
                                         "operating_point": dict(REFERENCE_OPERATING_POINT)}}


def test_reference_matches_or_beats_the_published_model(tmp_path):
    rep = build_report(runs(tmp_path, "r", [0.86, 0.88, 0.89]), published=PUBLISHED)
    assert rep["seeds"] == [0, 1, 2] and rep["per_seed_f1"] == [0.86, 0.88, 0.89]
    assert rep["mean_f1"] == pytest.approx(0.876667, abs=1e-5) and rep["range_f1"] == pytest.approx(0.03)
    assert rep["reproduces_published_model"] and rep["verdict"].startswith("PASS")
    assert rep["published_model"]["f1"] == 0.842
    assert all(r["code"]["commit"] and r["training"] for r in rep["runs"])


def test_reference_falls_short_of_the_published_model(tmp_path):
    rep = build_report(runs(tmp_path, "r", [0.82, 0.83, 0.84]), published=PUBLISHED)
    assert rep["mean_f1"] == pytest.approx(0.83)
    assert not rep["reproduces_published_model"] and rep["verdict"].startswith("FAIL")


@pytest.mark.parametrize("f1s,ok", [([0.842] * 3, True), ([0.90] * 3, True), ([0.841] * 3, False)])
def test_one_sided_boundary(tmp_path, f1s, ok):
    rep = build_report(runs(tmp_path, "r", f1s), published=PUBLISHED)
    assert rep["reproduces_published_model"] is ok


def test_report_uses_the_recorded_score_by_default(tmp_path):
    rep = build_report(runs(tmp_path, "r", [0.83] * 3))              # recorded stand-in: 0.82
    assert rep["published_model"]["f1"] == 0.82 and rep["reproduces_published_model"]


def test_report_stops_while_the_score_is_unset(tmp_path, monkeypatch):
    monkeypatch.setattr(report_mod, "PUBLISHED_MODEL_F1", None)
    with pytest.raises(ReportError, match="not recorded"):
        build_report(runs(tmp_path, "r", [0.83] * 3))


def test_published_score_at_another_operating_point_is_refused(tmp_path):
    other = {"f1": 0.842, "provenance": {"operating_point": {"threshold": 0.5, "min_distance": 1, "tau": 1.0}}}
    with pytest.raises(ReportError, match="operating point"):
        build_report(runs(tmp_path, "r", [0.86] * 3), published=other)


def test_summary_at_another_threshold_is_refused(tmp_path):
    rs = [write_run(tmp_path, f"t{s}", s, 0.9, operating_point={"threshold": 0.5, "min_distance": 1, "tau": 1.0})
          for s in (0, 1, 2)]
    with pytest.raises(ReportError, match="operating point"):
        build_report(rs, published=PUBLISHED)


@pytest.fixture
def ref_report(tmp_path):
    return build_report(runs(tmp_path, "ref", [0.82, 0.83, 0.84]))


def cand(tmp_path, f1s):
    return build_report(runs(tmp_path, "cand", f1s))


def test_clear_improvement(tmp_path, ref_report):
    assert compare(ref_report, cand(tmp_path, [0.86] * 3))["accuracy_supported"]


def test_within_noise(tmp_path, ref_report):
    assert not compare(ref_report, cand(tmp_path, [0.845] * 3))["accuracy_supported"]


def test_faster_and_non_inferior(tmp_path, ref_report):
    v = compare(ref_report, cand(tmp_path, [0.815] * 3), {"metric": "training_time_reduction", "value": 0.30})
    assert v["quality_of_life_adoptable"]


def test_benefit_not_measured(tmp_path, ref_report):
    c = cand(tmp_path, [0.815] * 3)
    assert not compare(ref_report, c)["quality_of_life_adoptable"]
    assert not compare(ref_report, c, {"metric": "speed"})["quality_of_life_adoptable"]


def test_too_much_worse_not_adopted(tmp_path, ref_report):
    v = compare(ref_report, cand(tmp_path, [0.80] * 3), {"metric": "x", "value": 1.0})
    assert not v["quality_of_life_adoptable"]


def test_no_divergence_adopted_against_failing_reference(tmp_path):
    bad_ref = build_report(runs(tmp_path, "ref", [0.70, 0.71, 0.72]))
    v = compare(bad_ref, cand(tmp_path, [0.99] * 3), {"metric": "x", "value": 1.0})
    assert not v["accuracy_supported"] and not v["quality_of_life_adoptable"]


def test_refuses_unknown_provenance(tmp_path):
    rs = runs(tmp_path, "r", [0.8, 0.8, 0.8])
    s = json.load(open(rs[1] / "summary.json"))
    s["unknown_provenance"] = True
    json.dump(s, open(rs[1] / "summary.json", "w"))
    with pytest.raises(ReportError, match="unknown_provenance"):
        build_report(rs)


def test_refuses_mismatched_seeds(tmp_path):
    with pytest.raises(ReportError, match="seeds"):
        build_report(runs(tmp_path, "r", [0.8] * 3, seeds=(0, 1, 5)))


def test_refuses_wrong_number_of_runs(tmp_path):
    with pytest.raises(ReportError, match="expected 3"):
        build_report(runs(tmp_path, "r", [0.8] * 2, seeds=(0, 1)))


def test_refuses_mismatched_protocol_and_operating_point(tmp_path):
    rs = runs(tmp_path, "r", [0.8] * 3)
    s = json.load(open(rs[0] / "summary.json"))
    s["protocol"] = "other"
    json.dump(s, open(rs[0] / "summary.json", "w"))
    with pytest.raises(ReportError, match="protocol"):
        build_report(rs)
    s["protocol"] = PROTOCOL
    s["operating_point"] = {"threshold": 0.99, "min_distance": 1, "tau": 1.0}
    json.dump(s, open(rs[0] / "summary.json", "w"))
    with pytest.raises(ReportError, match="operating point"):
        build_report(rs)


def test_refuses_mixed_configurations(tmp_path):
    rs = runs(tmp_path, "r", [0.8] * 3)
    s = json.load(open(rs[2] / "summary.json"))
    s["training"]["lr"] = 5e-3
    json.dump(s, open(rs[2] / "summary.json", "w"))
    with pytest.raises(ReportError, match="configuration"):
        build_report(rs)


def test_compare_refuses_mismatched_seeds(tmp_path, ref_report):
    c = copy.deepcopy(cand(tmp_path, [0.9] * 3))
    c["seeds"] = [0, 1, 3]
    with pytest.raises(ReportError, match="seeds"):
        compare(ref_report, c)


def test_per_seed_results_in_training_block_do_not_count_as_configuration(tmp_path):
    rs = runs(tmp_path, "r", [0.8] * 3)
    for i, d in enumerate(rs):
        s = json.load(open(d / "summary.json"))
        s["training"].update(best_epoch=i, best_val_objective=0.5 + i)
        json.dump(s, open(d / "summary.json", "w"))
    assert build_report(rs)["seeds"] == [0, 1, 2]


def test_validation_result_is_not_a_benchmark(tmp_path):
    ds_val = {"dataset_version": "SpotNet-v1_1", "split": "val", "source_crc32": "29836378",
              "n_images": 95, "n_spots": 1957}
    rs = [write_run(tmp_path, f"v{s}", s, 0.8, split="val", n_images_scored=95, val_dataset=ds_val)
          for s in (0, 1, 2)]
    with pytest.raises(ReportError, match="val"):
        build_report(rs)


# ---- published-model scoring ---------------------------------------------------------------------

def _published_fixture(tmp_path, monkeypatch, points=None, crc="36c922d7"):
    import numpy as np
    import zarr
    from torch_spots import dataset as ds
    from tests.synthetic import make_store
    monkeypatch.setitem(ds.PINNED_SPLITS, "test", {"crc32": "36c922d7", "n_images": 4, "n_spots": 12})
    make_store(tmp_path / "test.zarr", n_images=4, n_spots_per_image=3, size=32)
    z = zarr.open(str(tmp_path / "test.zarr"), mode="r")
    truth = [np.asarray(z["y"][i, :3]).tolist() for i in range(4)]
    pts = points or {"0.9": truth,                                   # perfect
                     "0.99": [[] for _ in range(4)]}                 # nothing found
    rec = {"provenance": {"model": "SpotDetection-8", "test_split": {"crc32": crc, "n_images": 4}},
           "points": pts}
    path = tmp_path / "points.json"
    json.dump(rec, open(path, "w"))
    return path, tmp_path / "test.zarr"


def test_score_published_scores_every_threshold(tmp_path, monkeypatch):
    from torch_spots.detection.report import score_published
    path, store = _published_fixture(tmp_path, monkeypatch)
    s = score_published(path, store)
    assert s["n_images_scored"] == 4 and s["test_dataset"]["source_crc32"] == "36c922d7"
    assert s["per_threshold"]["0.9"]["f1"] == pytest.approx(1.0)
    assert s["per_threshold"]["0.99"]["f1"] == pytest.approx(0.0)
    assert s["provenance"]["model"] == "SpotDetection-8"


def test_published_cli_writes_summary(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from torch_spots.detection import report as report_mod
    path, store = _published_fixture(tmp_path, monkeypatch)
    r = CliRunner().invoke(report_mod.main, ["published", str(path), "--data-path", str(store)])
    assert r.exit_code == 0, r.output
    assert json.load(open(tmp_path / "summary.json"))["per_threshold"]["0.9"]["f1"] == pytest.approx(1.0)


def test_published_points_from_another_split_refused(tmp_path, monkeypatch):
    from torch_spots.detection.report import score_published
    path, store = _published_fixture(tmp_path, monkeypatch, crc="29836378")
    with pytest.raises(ReportError, match="CRC-32"):
        score_published(path, store)


def test_published_wrong_image_count_refused(tmp_path, monkeypatch):
    from torch_spots.detection.report import score_published
    path, store = _published_fixture(tmp_path, monkeypatch, points={"0.9": [[], []]})
    with pytest.raises(ReportError, match="expected 4"):
        score_published(path, store)


def test_report_cli_runs_on_synthetic_runs(tmp_path):
    from click.testing import CliRunner
    rs = runs(tmp_path, "c", [0.83, 0.84, 0.85])
    out = tmp_path / "report.json"
    r = CliRunner().invoke(report_mod.main, ["report", *map(str, rs), "--output", str(out)])
    assert r.exit_code == 0, r.output
    assert "PASS" in r.output and json.load(open(out))["reproduces_published_model"] is True
