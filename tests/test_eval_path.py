import numpy as np
import pytest
import torch

from torch_spots.detection.inference import EVAL_MIN_DISTANCE, EVAL_THRESHOLD, SpotDetection
from torch_spots.detection.loader import SpotDataset, preprocess_image
from tests.synthetic import write_checkpoint


@pytest.fixture(scope="module")
def detector(tmp_path_factory):
    path = write_checkpoint(tmp_path_factory.mktemp("ckpt"))
    return SpotDetection(model_path=str(path), device="cpu")


def _images(n=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.gamma(2.0, 300.0, size=(n, 1, 128, 128)).astype(np.float32)


def test_same_image_two_paths(detector):
    X = _images()
    y = np.zeros((len(X), 1, 2), dtype=np.float32)
    ds = SpotDataset(X, y, np.zeros(len(X), dtype=int), augment=False)
    captured = []
    h = detector.model.register_forward_hook(lambda m, inp, out: captured.append(inp[0].clone()))
    try:
        for i in range(len(X)):
            detector._eval_predict(X[i:i + 1])
    finally:
        h.remove()
    for i in range(len(X)):
        train_x = ds[i][0].numpy()
        assert np.array_equal(captured[i][0].numpy(), train_x)


def test_preprocess_matches_documented_transform():
    x = _images(1)[0]
    assert np.allclose(preprocess_image(x), (x - x.min()) / (x.max() - x.min()))


def test_operating_point_defaults():
    assert (EVAL_THRESHOLD, EVAL_MIN_DISTANCE) == (0.993, 1)
    import inspect
    sig = inspect.signature(SpotDetection._eval_predict)
    assert sig.parameters["threshold"].default == EVAL_THRESHOLD
    assert sig.parameters["min_distance"].default == 1


def _synthetic_maps(rng, h=40, w=40, n_peaks=8):
    prob = rng.uniform(0, 0.3, size=(1, h, w))
    ys, xs = rng.integers(0, h, n_peaks), rng.integers(0, w, n_peaks)
    prob[0, ys, xs] = rng.uniform(0.9, 1.0, n_peaks)
    prob = np.stack([1 - prob, prob], axis=1).astype(np.float32)  # (1, 2, H, W)
    offsets = rng.normal(0, 0.5, size=(1, 2, h, w)).astype(np.float32)
    return {"detections": prob, "offsets": offsets}


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("min_distance", [1, 2, 3])
def test_peaks_identical_to_oracle(oracle, detector, seed, min_distance):
    _, post = oracle
    rng = np.random.default_rng(seed)
    maps = _synthetic_maps(rng)
    ours = detector._y_annotations_to_point_list_max(maps, threshold=0.95, min_distance=min_distance)
    theirs = post.y_annotations_to_point_list_max(
        {"classification": np.moveaxis(maps["detections"], 1, -1),
         "offset_regression": np.moveaxis(maps["offsets"], 1, -1)},
        threshold=0.95, min_distance=min_distance)
    theirs = np.asarray(theirs[0], dtype=np.float32).reshape(-1, 2)
    assert ours.shape == theirs.shape
    assert np.array_equal(ours, theirs)


def test_all_preprocessing_paths_agree(detector):
    """One raw image through the dataset, the eval path and every inference entry point
    reaches the model as the same array (spec: "All paths agree")."""
    X = _images(1)
    y = np.zeros((1, 1, 2), dtype=np.float32)
    captured = {}

    def run(name, fn):
        seen = []
        h = detector.model.register_forward_hook(lambda m, inp, out: seen.append(inp[0].clone()))
        try:
            fn()
        finally:
            h.remove()
        captured[name] = torch.cat(seen).numpy()

    run("dataset", lambda: detector.model(SpotDataset(X, y, np.zeros(1, dtype=int), augment=False)[0][0][None]))
    run("eval", lambda: detector._eval_predict(X))
    run("transforms", lambda: detector.predict_transforms(X))
    run("points", lambda: detector.predict_points(X))
    for name, arr in captured.items():
        assert np.array_equal(arr, captured["dataset"]), name


# ---- loading ----------------------------------------------------------------------------------

def test_bare_state_dict_loads_with_defaults(tmp_path):
    path = write_checkpoint(tmp_path, training_record=False)
    det = SpotDetection(model_path=str(path), device="cpu")
    assert det.model.backbone.norm_method is None


def test_norm_method_is_passed_to_the_model(tmp_path):
    path = write_checkpoint(tmp_path, norm_method="std", training_record=False)
    det = SpotDetection(model_path=str(path), device="cpu", norm_method="std")
    assert det.model.backbone.norm_method == "std"


def test_mismatched_settings_fail_strictly(tmp_path):
    """Weights always match the architecture for any norm_method, but a state dict with an
    unexpected parameter must not load silently."""
    path = write_checkpoint(tmp_path)
    state = torch.load(path)
    state["backbone.subnets.0.stale.weight"] = torch.zeros(1)
    torch.save(state, path)
    with pytest.raises(RuntimeError):
        SpotDetection(model_path=str(path), device="cpu")


def test_default_model_downloads_only_the_weights(tmp_path, monkeypatch):
    import torch_spots.detection.inference as inf
    requested = []

    def fake_download(repo_id, filename, local_dir):
        requested.append(filename)
        write_checkpoint(local_dir, name=filename, training_record=False)
    monkeypatch.setattr(inf.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(inf, "hf_hub_download", fake_download)
    det = SpotDetection()
    assert requested == [inf.DEFAULT_PRETRAINED]
    assert det.model_path == tmp_path / ".deepcell/models" / inf.DEFAULT_PRETRAINED
