import json

import numpy as np
import pytest
import torch

from torch_spots import dataset as ds
from torch_spots.detection import train as train_mod
from torch_spots.detection.loader import spot_dataloader
from tests.synthetic import make_store

N_IMG, N_SPOTS = 4, 3


def _stores(tmp_path, split_attrs=True):
    for split in ("train", "val"):
        make_store(tmp_path / f"{split}.zarr", n_images=N_IMG, n_spots_per_image=N_SPOTS, size=128,
                   seed=7, split=split)
    return tmp_path


@pytest.fixture
def pinned_synth(monkeypatch):
    for split in ("train", "val"):
        monkeypatch.setitem(ds.PINNED_SPLITS, split, {
            "crc32": ds.PINNED_SPLITS[split]["crc32"], "n_images": N_IMG, "n_spots": N_IMG * N_SPOTS})


def _first_epoch(path, seed, num_workers=0, augmentation=None):
    """Every batch of the first epoch, concatenated (images and both label maps)."""
    import zarr
    z = zarr.open(str(path / "train.zarr"), mode="r")
    train_mod.seed_everything(seed)
    dl = spot_dataloader(z["X"], z["y"], z["y_inds"], batch_size=1, num_workers=num_workers,
                         augment=True, augmentation=augmentation, seed=seed)
    batches = list(dl)
    x = torch.cat([b[0] for b in batches])
    labels = {k: torch.cat([b[1][k] for b in batches]) for k in batches[0][1]}
    return x, labels


@pytest.mark.parametrize("num_workers", [0, 2])
def test_same_seed_same_first_epoch(tmp_path, num_workers):
    _stores(tmp_path)
    a = _first_epoch(tmp_path, 0, num_workers)
    b = _first_epoch(tmp_path, 0, num_workers)
    c = _first_epoch(tmp_path, 1, num_workers)
    assert torch.equal(a[0], b[0])
    for k in a[1]:
        assert torch.equal(a[1][k], b[1][k])
    # Flips alone give few possible variants, so compare the whole epoch, not one batch.
    assert not torch.equal(a[0], c[0])


@pytest.mark.parametrize("num_workers", [0, 2])
def test_validation_is_not_augmented(tmp_path, num_workers):
    import zarr
    from torch_spots.detection.loader import SpotDataset, preprocess_image, point_list_to_annotations
    _stores(tmp_path)
    z = zarr.open(str(tmp_path / "val.zarr"), mode="r")
    # Even with every augmentation turned on, augment=False must return the data untouched.
    aug = {"flip": True, "rotation_range": 180.0, "zoom_range": [0.8, 1.25]}
    dl = spot_dataloader(z["X"], z["y"], z["y_inds"], batch_size=1, shuffle=False,
                         num_workers=num_workers, augment=False, augmentation=aug, seed=0)
    for _ in range(2):                                     # same on every epoch
        for i, (x, labels) in enumerate(dl):
            expect_x = torch.from_numpy(preprocess_image(z["X"][i].copy())).float()
            assert torch.equal(x[0], expect_x)
            pts = z["y"][i][: z["y_inds"][i]]
            ann = point_list_to_annotations(pts, image_shape=(128, 128))
            assert torch.equal(labels["detections"][0], torch.tensor(ann["detections"]))
            assert torch.equal(labels["offsets"][0], torch.tensor(ann["offsets"]))


def test_mismatched_store_raises_before_training(tmp_path, monkeypatch):
    _stores(tmp_path)  # synthetic counts differ from the real pins
    called = []
    monkeypatch.setattr(train_mod, "train", lambda *a, **k: called.append(1))
    cfg = {**train_mod.DEFAULT_CONFIG, "data_path": str(tmp_path), "device": "cpu", "num_workers": 0,
           "model_path": str(tmp_path / "models"), "run_info": str(tmp_path / "logs")}
    with pytest.raises(ds.DatasetMismatchError):
        train_mod.run_training(cfg)
    assert not called
    assert not (tmp_path / "models").exists()


def test_short_run_writes_seed_and_provenance(tmp_path, pinned_synth):
    _stores(tmp_path)
    cfg = {**train_mod.DEFAULT_CONFIG, "data_path": str(tmp_path), "device": "cpu", "num_workers": 0,
           "batch_size": 2, "epochs": 1, "seed": 2,
           "model_path": str(tmp_path / "models"), "run_info": str(tmp_path / "logs")}
    model_dir = train_mod.run_training(cfg)
    rec = json.load(open(f"{model_dir}/training_config.json"))
    assert rec["seed"] == 2
    for split, n in (("train", N_IMG), ("val", N_IMG)):
        d = rec["dataset"][split]
        assert d["dataset_version"] == ds.DATASET_VERSION
        assert d["source_crc32"] == ds.PINNED_SPLITS[split]["crc32"]
        assert d["n_images"] == n and d["n_spots"] == n * N_SPOTS


def test_config_override_of_one_nested_key_keeps_the_rest(tmp_path):
    path = tmp_path / "ablation.json"
    path.write_text(json.dumps({"augmentation": {"rotation_range": 45}}))
    cfg = train_mod.load_config(path)
    assert cfg["augmentation"]["rotation_range"] == 45
    ref = train_mod.DEFAULT_CONFIG
    assert cfg["augmentation"]["flip"] == ref["augmentation"]["flip"]
    assert cfg["augmentation"]["zoom_range"] == ref["augmentation"]["zoom_range"]
    assert cfg["optimizer"] == ref["optimizer"] and cfg["loss"] == ref["loss"]
    assert cfg["epochs"] == ref["epochs"]


def test_unknown_config_key_raises(tmp_path):
    path = tmp_path / "typo.json"
    path.write_text(json.dumps({"optimizer": {"learning_rate": 1}}))
    with pytest.raises(ValueError):
        train_mod.load_config(path)


def _short_cfg(tmp_path, **over):
    cfg = {**train_mod.DEFAULT_CONFIG, "data_path": str(tmp_path), "device": "cpu", "num_workers": 0,
           "batch_size": 2, "epochs": 1,
           "model_path": str(tmp_path / "models"), "run_info": str(tmp_path / "logs")}
    return train_mod.merge_config(cfg, over)


def test_reference_run_lists_no_non_reference_settings(tmp_path, pinned_synth):
    _stores(tmp_path)
    cfg = _short_cfg(tmp_path)
    # batch_size/epochs are shortened only to keep the test fast; restore them for the comparison
    assert train_mod.non_reference_settings(
        {**cfg, "epochs": train_mod.REFERENCE_TRAINING["epochs"],
         "batch_size": train_mod.REFERENCE_TRAINING["batch_size"]}) == {}
    rec = json.load(open(f"{train_mod.run_training(cfg)}/training_config.json"))
    assert set(rec["non_reference"]) == {"epochs", "batch_size"}


def test_ablation_run_lists_rotation_and_loss_weights(tmp_path, pinned_synth):
    _stores(tmp_path)
    ref = train_mod.REFERENCE_TRAINING
    cfg = _short_cfg(tmp_path, augmentation={"rotation_range": 0.0},
                     loss={"regression_weight": 1.0, "classification_weight": 1.0},
                     epochs=ref["epochs"], batch_size=ref["batch_size"])
    nr = train_mod.non_reference_settings(cfg)
    assert nr == {"augmentation.rotation_range": 0.0, "loss.regression_weight": 1.0,
                  "loss.classification_weight": 1.0}


def test_longer_training_lists_only_epochs(tmp_path, pinned_synth):
    ref = train_mod.REFERENCE_TRAINING
    cfg = _short_cfg(tmp_path, epochs=50, batch_size=ref["batch_size"])
    assert train_mod.non_reference_settings(cfg) == {"epochs": 50}


def test_cli_lists_config_option():
    from click.testing import CliRunner
    out = CliRunner().invoke(train_mod.main, ["--help"]).output
    assert "--config" in out


def test_readme_ablation_example_runs(tmp_path, pinned_synth):
    """The JSON example in the README loads through --config and trains, with its non-reference list."""
    import re
    from pathlib import Path
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    block = re.search(r"```json\n(.*?)```", readme, re.S).group(1)
    cfg_path = tmp_path / "ablation.json"
    cfg_path.write_text(block)
    _stores(tmp_path)
    cfg = train_mod.load_config(cfg_path)
    cfg.update(data_path=str(tmp_path), device="cpu", num_workers=0, batch_size=2, epochs=1,
               model_path=str(tmp_path / "models"), run_info=str(tmp_path / "logs"))
    rec = json.load(open(f"{train_mod.run_training(cfg)}/training_config.json"))
    assert rec["augmentation"]["rotation_range"] == 0
    assert {"augmentation.rotation_range", "loss.regression_weight",
            "loss.classification_weight"} <= set(rec["non_reference"])
    assert "best_epoch" in rec and "best_val_objective" in rec


def test_reference_run_records_no_normalisation(tmp_path, pinned_synth):
    _stores(tmp_path)
    cfg = _short_cfg(tmp_path)
    rec = json.load(open(f"{train_mod.run_training(cfg)}/training_config.json"))
    assert rec["model"]["norm_method"] is None
    assert "model.norm_method" not in rec["non_reference"]
    ref = train_mod.REFERENCE_TRAINING
    assert train_mod.non_reference_settings(
        {**cfg, "epochs": ref["epochs"], "batch_size": ref["batch_size"]}) == {}


def test_normalisation_ablation_is_listed_as_non_reference(tmp_path, pinned_synth):
    _stores(tmp_path)
    cfg = _short_cfg(tmp_path, model={"norm_method": "std"})
    rec = json.load(open(f"{train_mod.run_training(cfg)}/training_config.json"))
    assert rec["non_reference"]["model.norm_method"] == "std"
    assert rec["model"]["norm_method"] == "std"


def test_unknown_norm_method_rejected_at_load(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"model": {"norm_method": "max"}}))
    with pytest.raises(ValueError, match="norm_method"):
        train_mod.load_config(path)
