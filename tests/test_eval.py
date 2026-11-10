import json

import pandas as pd
import pytest
import torch

from torch_spots import dataset as ds
from torch_spots.detection.dotnet import SpotNet
from torch_spots.detection.eval import PROTOCOL, run_evaluation
from torch_spots.detection.reference_metrics import REFERENCE_FIELDS
from tests.synthetic import make_store, write_checkpoint


@pytest.fixture
def ckpt(tmp_path):
    d = tmp_path / "model"
    write_checkpoint(d)
    return d


@pytest.fixture
def synth_test_pin(monkeypatch):
    monkeypatch.setitem(ds.PINNED_SPLITS, "test", {"crc32": ds.PINNED_SPLITS["test"]["crc32"],
                                                   "n_images": 4, "n_spots": 12})


def test_end_to_end_unknown_provenance(tmp_path, ckpt, synth_test_pin):
    make_store(tmp_path / "test.zarr", n_images=4, n_spots_per_image=3, size=128)
    out = tmp_path / "run"
    s = run_evaluation(ckpt / "m.pth", tmp_path / "test.zarr", device="cpu", output_dir=out)
    df = pd.read_csv(out / "per_image.csv")
    assert len(df) == 4 and set(REFERENCE_FIELDS) <= set(df.columns)
    j = json.load(open(out / "summary.json"))
    assert j["protocol"] == PROTOCOL
    assert j["n_images_scored"] == 4
    assert j["operating_point"] == {"threshold": 0.993, "min_distance": 1, "tau": 1.0}
    assert j["unknown_provenance"] is True and j["benchmark_eligible"] is False
    assert j["test_dataset"]["source_crc32"] == "36c922d7"
    for k in ("f1", "precision", "recall", "jac", "rmse", "rmse_n_images", "d_md_n_images"):
        assert k in j["reference_metrics"]
    assert set(j["supplementary_micro"]) == {"tp", "fp", "fn"}
    assert "code" in j and "commit" in j["code"]
    assert s["output_dir"] == str(out)


def test_known_provenance_recorded(tmp_path, ckpt, synth_test_pin):
    make_store(tmp_path / "test.zarr", n_images=4, n_spots_per_image=3, size=128)
    prov = {"train": {"source_crc32": "x"}, "val": {"source_crc32": "y"}}
    write_checkpoint(ckpt, record={"seed": 1, "dataset": prov, "lr": 1e-3})
    s = run_evaluation(ckpt / "m.pth", tmp_path / "test.zarr", device="cpu", output_dir=tmp_path / "r")
    assert s["unknown_provenance"] is False and s["benchmark_eligible"] is True and s["seed"] == 1


def test_non_reference_operating_point_not_eligible(tmp_path, ckpt, synth_test_pin):
    make_store(tmp_path / "test.zarr", n_images=4, n_spots_per_image=3, size=128)
    write_checkpoint(ckpt, record={"seed": 1, "dataset": {"train": {}, "val": {}}})
    s = run_evaluation(ckpt / "m.pth", tmp_path / "test.zarr", device="cpu", tau=2.0, output_dir=tmp_path / "r")
    assert s["benchmark_eligible"] is False


def test_refuses_100_image_store(tmp_path, ckpt):
    make_store(tmp_path / "test.zarr", n_images=100, n_spots_per_image=1, size=8)
    with pytest.raises(ds.DatasetMismatchError, match="images"):
        run_evaluation(ckpt / "m.pth", tmp_path / "test.zarr", device="cpu", output_dir=tmp_path / "r")
    assert not (tmp_path / "r").exists()


# ---- checkpoint compatibility (benchmark-evaluation: Checkpoint model compatibility) ----------

from torch_spots.detection.dotnet import MODEL_ARCHITECTURE_VERSION, IncompatibleCheckpointError


def test_matching_version_scored_and_recorded(tmp_path, ckpt, synth_test_pin):
    make_store(tmp_path / "test.zarr", n_images=4, n_spots_per_image=3, size=128)
    s = run_evaluation(ckpt / "m.pth", tmp_path / "test.zarr", device="cpu", output_dir=tmp_path / "r")
    assert s["model_architecture_version"] == MODEL_ARCHITECTURE_VERSION
    assert json.load(open(tmp_path / "r" / "summary.json"))["model_architecture_version"] == \
        MODEL_ARCHITECTURE_VERSION


def test_checkpoint_from_before_versions_is_refused(tmp_path, synth_test_pin):
    make_store(tmp_path / "test.zarr", n_images=4, n_spots_per_image=3, size=128)
    old = write_checkpoint(tmp_path / "old", version=None)
    with pytest.raises(IncompatibleCheckpointError, match="missing"):
        run_evaluation(old, tmp_path / "test.zarr", device="cpu", output_dir=tmp_path / "r")
    assert not (tmp_path / "r").exists()           # no score written


def test_checkpoint_without_any_training_record_is_refused(tmp_path, synth_test_pin):
    make_store(tmp_path / "test.zarr", n_images=4, n_spots_per_image=3, size=128)
    old = write_checkpoint(tmp_path / "old")
    (tmp_path / "old" / "training_config.json").unlink()
    with pytest.raises(IncompatibleCheckpointError, match="missing"):
        run_evaluation(old, tmp_path / "test.zarr", device="cpu", output_dir=tmp_path / "r")


def test_different_version_names_both_versions(tmp_path, synth_test_pin):
    make_store(tmp_path / "test.zarr", n_images=4, n_spots_per_image=3, size=128)
    other = write_checkpoint(tmp_path / "other", version=MODEL_ARCHITECTURE_VERSION + 1)
    with pytest.raises(IncompatibleCheckpointError) as err:
        run_evaluation(other, tmp_path / "test.zarr", device="cpu", output_dir=tmp_path / "r")
    msg = str(err.value)
    assert str(MODEL_ARCHITECTURE_VERSION + 1) in msg and f"current version is {MODEL_ARCHITECTURE_VERSION}" in msg
    assert not (tmp_path / "r").exists()


# ---- scoring the validation split ----------------------------------------------------------------

@pytest.fixture
def synth_val_pin(monkeypatch):
    """The real validation pin has 95 images; spots per image are set so the counts agree."""
    monkeypatch.setitem(ds.PINNED_SPLITS, "val", {"crc32": ds.PINNED_SPLITS["val"]["crc32"],
                                                  "n_images": 95, "n_spots": 190})


def test_scoring_the_validation_split(tmp_path, ckpt, synth_val_pin):
    make_store(tmp_path / "val.zarr", n_images=95, n_spots_per_image=2, size=64, split="val")
    s = run_evaluation(ckpt / "m.pth", tmp_path / "val.zarr", device="cpu", output_dir=tmp_path / "v",
                       split="val")
    assert s["n_images_scored"] == 95
    assert s["split"] == "val"
    assert s["val_dataset"]["source_crc32"] == "29836378" and s["val_dataset"]["n_images"] == 95
    assert "test_dataset" not in s
    assert s["benchmark_eligible"] is False
    assert json.load(open(tmp_path / "v" / "summary.json"))["split"] == "val"


def test_known_provenance_validation_is_still_not_eligible(tmp_path, ckpt, synth_val_pin):
    make_store(tmp_path / "val.zarr", n_images=95, n_spots_per_image=2, size=64, split="val")
    write_checkpoint(ckpt, record={"seed": 1, "dataset": {"train": {}, "val": {}}})
    s = run_evaluation(ckpt / "m.pth", tmp_path / "val.zarr", device="cpu", output_dir=tmp_path / "v",
                       split="val")
    assert s["unknown_provenance"] is False and s["benchmark_eligible"] is False


def test_mismatched_validation_store_is_refused(tmp_path, ckpt):
    make_store(tmp_path / "val.zarr", n_images=4, n_spots_per_image=3, size=64, split="val")
    with pytest.raises(ds.DatasetMismatchError):
        run_evaluation(ckpt / "m.pth", tmp_path / "val.zarr", device="cpu", output_dir=tmp_path / "v",
                       split="val")


def test_test_store_is_refused_as_validation(tmp_path, ckpt, synth_test_pin):
    make_store(tmp_path / "test.zarr", n_images=4, n_spots_per_image=3, size=64, split="test")
    with pytest.raises(ds.DatasetMismatchError):
        run_evaluation(ckpt / "m.pth", tmp_path / "test.zarr", device="cpu", split="val",
                       output_dir=tmp_path / "v")


def test_unknown_split_rejected(tmp_path, ckpt):
    with pytest.raises(ValueError, match="split"):
        run_evaluation(ckpt / "m.pth", tmp_path, device="cpu", split="train")


def test_cli_split_option_runs(tmp_path, ckpt, synth_val_pin):
    """The README command: ``eval --split val`` (store path given explicitly for the synthetic store)."""
    from click.testing import CliRunner
    from torch_spots.detection import eval as eval_mod
    make_store(tmp_path / "val.zarr", n_images=95, n_spots_per_image=2, size=64, split="val")
    r = CliRunner().invoke(eval_mod.main, ["--split", "val", "--device", "cpu", "--model-path",
                                           str(ckpt / "m.pth"), "--data-path", str(tmp_path / "val.zarr"),
                                           "--output-dir", str(tmp_path / "o")])
    assert r.exit_code == 0, r.output
    assert "95 val images" in r.output and "not a benchmark result" in r.output
