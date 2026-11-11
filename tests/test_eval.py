import json

import pandas as pd
import pytest

from torch_spots import dataset as ds
from torch_spots.detection.eval import run_evaluation
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


def test_end_to_end(tmp_path, ckpt, synth_test_pin):
    make_store(tmp_path / "test.zarr", n_images=4, n_spots_per_image=3, size=128)
    out = tmp_path / "run"
    s = run_evaluation(ckpt / "m.pth", tmp_path / "test.zarr", device="cpu", output_dir=out)
    df = pd.read_csv(out / "per_image.csv")
    assert len(df) == 4 and set(REFERENCE_FIELDS) <= set(df.columns)
    j = json.load(open(out / "summary.json"))
    assert j["n_images_scored"] == 4 and j["split"] == "test"
    assert j["operating_point"] == {"threshold": 0.993, "min_distance": 1, "tau": 1.0}
    assert j["dataset"]["source_crc32"] == "36c922d7"
    for k in ("f1", "precision", "recall", "jac", "rmse", "rmse_n_images", "d_md_n_images"):
        assert k in j["reference_metrics"]
    assert set(j["supplementary_micro"]) == {"tp", "fp", "fn"}
    assert s["output_dir"] == str(out)


def test_refuses_100_image_store(tmp_path, ckpt):
    make_store(tmp_path / "test.zarr", n_images=100, n_spots_per_image=1, size=8)
    with pytest.raises(ds.DatasetMismatchError, match="images"):
        run_evaluation(ckpt / "m.pth", tmp_path / "test.zarr", device="cpu", output_dir=tmp_path / "r")
    assert not (tmp_path / "r").exists()


# ---- checkpoint loading ------------------------------------------------------------------------

def test_norm_method_is_passed_to_the_model(tmp_path, synth_test_pin, monkeypatch):
    import torch_spots.detection.eval as ev
    make_store(tmp_path / "test.zarr", n_images=4, n_spots_per_image=3, size=128)
    std = write_checkpoint(tmp_path / "std", norm_method="std", training_record=False)
    built = {}
    real = ev.SpotDetection

    def spy(*args, **kwargs):
        built.update(kwargs)
        return real(*args, **kwargs)
    monkeypatch.setattr(ev, "SpotDetection", spy)
    s = run_evaluation(std, tmp_path / "test.zarr", device="cpu", output_dir=tmp_path / "r",
                       norm_method="std")
    assert built["norm_method"] == "std" and s["norm_method"] == "std"


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
    assert s["dataset"]["source_crc32"] == "29836378" and s["dataset"]["n_images"] == 95
    assert json.load(open(tmp_path / "v" / "summary.json"))["split"] == "val"


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
    assert "95 val images" in r.output
