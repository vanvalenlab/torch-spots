import json
import numpy as np
import pytest
import torch

from torch_spots import dataset as ds
from torch_spots.detection.dotnet import SpotNet
from torch_spots.detection.eval_sweep import build_grid, run_sweep
from tests.synthetic import make_store, write_checkpoint

from torch_spots.detection.inference import EVAL_THRESHOLD

REF = (EVAL_THRESHOLD, 1, 1.0)


def _has_ref(grid):
    return any(np.isclose(t, EVAL_THRESHOLD) and m == 1 and np.isclose(tau, 1.0) for t, m, tau in grid)


def test_default_grid_contains_reference_point():
    assert _has_ref(build_grid())


def test_user_grid_omitting_reference_gets_it():
    grid = build_grid(thresholds=[0.5, 0.99], min_distances=[2, 3], taus=[2.0, 5.0])
    assert _has_ref(grid)
    assert len(grid) == 3 * 3 * 3


def test_float_near_reference_not_duplicated():
    grid = build_grid(thresholds=np.linspace(0.9, 1.0, 11), min_distances=[1], taus=[1.0])
    assert sum(np.isclose(t, EVAL_THRESHOLD) for t, _, _ in grid) == 1


def test_sweep_reports_reference_point_separately(tmp_path, monkeypatch):
    monkeypatch.setitem(ds.PINNED_SPLITS, "test", {"crc32": "36c922d7", "n_images": 2, "n_spots": 4})
    make_store(tmp_path / "test.zarr", n_images=2, n_spots_per_image=2, size=128)
    write_checkpoint(tmp_path)
    df = run_sweep(tmp_path / "m.pth", tmp_path / "test.zarr", device="cpu",
                   grid=([0.5], [2], [2.0]), output_dir=tmp_path / "out")
    assert df["is_reference_point"].sum() == 1
    assert (tmp_path / "out" / "reference_point.json").is_file()


def test_sweep_on_validation_split_records_split(tmp_path, monkeypatch):
    monkeypatch.setitem(ds.PINNED_SPLITS, "val", {"crc32": "29836378", "n_images": 2, "n_spots": 4})
    make_store(tmp_path / "val.zarr", n_images=2, n_spots_per_image=2, size=128, split="val")
    write_checkpoint(tmp_path)
    df = run_sweep(tmp_path / "m.pth", tmp_path / "val.zarr", device="cpu",
                   grid=([0.5], [1], [1.0]), output_dir=tmp_path / "out", split="val")
    assert (df["n_images"] == 2).all()
    rec = json.load(open(tmp_path / "out" / "reference_point.json"))
    assert rec["split"] == "val" and rec["benchmark_eligible"] is False
    assert rec["dataset"]["source_crc32"] == "29836378"


def test_sweep_refuses_test_store_as_validation(tmp_path, monkeypatch):
    monkeypatch.setitem(ds.PINNED_SPLITS, "test", {"crc32": "36c922d7", "n_images": 2, "n_spots": 4})
    make_store(tmp_path / "test.zarr", n_images=2, n_spots_per_image=2, size=128)
    write_checkpoint(tmp_path)
    with pytest.raises(ds.DatasetMismatchError):
        run_sweep(tmp_path / "m.pth", tmp_path / "test.zarr", device="cpu", grid=([0.5], [1], [1.0]),
                  output_dir=tmp_path / "out", split="val")
