"""Input preprocessing (spec: detection-model, Reference input preprocessing)."""
from pathlib import Path

import numpy as np
import pytest

from torch_spots.detection.loader import preprocess_image

SPOTS_DIR = Path.home() / ".deepcell" / "spots"


def test_scaling_is_plain_min_max_without_clipping():
    rng = np.random.default_rng(0)
    x = rng.uniform(100, 1100, size=(1, 64, 64))
    x[0, 0, 0], x[0, 1, 1] = 100.0, 1100.0
    x[0, 5, 5] = 1099.9                           # a bright pixel just below the max
    out = preprocess_image(x)
    assert np.allclose(out, (x - 100) / 1000)
    assert out[0, 5, 5] == pytest.approx(0.9999) and out.max() == 1.0 and out.min() == 0.0


def test_channels_are_scaled_separately():
    x = np.stack([np.linspace(0, 10, 64).reshape(8, 8), np.linspace(50, 52, 64).reshape(8, 8)])
    out = preprocess_image(x)
    for c in range(2):
        assert out[c].min() == 0 and out[c].max() == 1


def test_clip_requested_flattens_a_single_hot_pixel():
    x = np.full((1, 100, 100), 10.0)
    x[0, 50, 50] = 10_000.0
    out = preprocess_image(x, clip=True)
    assert (out == 0).all()


def test_clip_not_requested_is_the_reference_preprocessing():
    x = np.full((1, 100, 100), 10.0)
    x[0, 50, 50] = 10_000.0
    assert np.array_equal(preprocess_image(x), preprocess_image(x, clip=False))
    assert preprocess_image(x)[0, 50, 50] == 1.0 and preprocess_image(x)[0, 0, 0] == 0.0


def test_constant_image_maps_to_zero():
    assert (preprocess_image(np.full((1, 32, 32), 500, dtype=np.uint16)) == 0).all()


@pytest.mark.skipif(not any(SPOTS_DIR.glob("*.npz")),
                    reason=f"SpotNet v1.1 npz files not found under {SPOTS_DIR}")
def test_spotnet_v1_1_images_unchanged():
    n = 0
    for split in ("train", "val", "test"):
        X = np.load(SPOTS_DIR / f"{split}.npz", allow_pickle=True)["X"]
        for img in X:                                   # (H, W, 1)
            chw = np.moveaxis(img, -1, 0)
            assert np.array_equal(preprocess_image(chw), chw)
            n += 1
    assert n == 1038
