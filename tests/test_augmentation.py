import numpy as np
import pytest
import torch

from torch_spots.detection.augmentation import SpotAugmentation


def _gauss(cy, cx, size=128, sigma=3.0):
    yy, xx = np.mgrid[:size, :size]
    return torch.from_numpy(np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))).float()[None]


def _com(img):
    a = img[0].numpy().astype(np.float64)
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    return (a * yy).sum() / a.sum(), (a * xx).sum() / a.sum()


def _seed_for(aug_fn, predicate, tries=200):
    for s in range(tries):
        np.random.seed(s)
        out = aug_fn()
        if predicate(out):
            np.random.seed(s)
            return s
    raise AssertionError("no seed found")


def test_horizontal_flip_moves_pixels_and_points_together():
    X = torch.arange(128 * 128, dtype=torch.float32).reshape(1, 128, 128)
    y = np.array([[10.3, 20.7]], dtype=np.float32)
    aug = SpotAugmentation(flip=True, rotation_range=0.0, zoom_range=None)
    # find a seed whose first draw flips horizontally and second does not
    for s in range(100):
        np.random.seed(s)
        a, b = np.random.random() < 0.5, np.random.random() < 0.5
        if a and not b:
            break
    np.random.seed(s)
    Xo, yo = aug(X, y)
    assert np.allclose(yo, [[10.3, 106.3]], atol=1e-4)
    assert torch.equal(Xo[0, :, 127 - 5], X[0, :, 5])


def test_vertical_flip_moves_pixels_and_points_together():
    X = torch.arange(128 * 128, dtype=torch.float32).reshape(1, 128, 128)
    y = np.array([[10.3, 20.7]], dtype=np.float32)
    aug = SpotAugmentation(flip=True, rotation_range=0.0, zoom_range=None)
    for s in range(100):
        np.random.seed(s)
        a, b = np.random.random() < 0.5, np.random.random() < 0.5
        if b and not a:
            break
    np.random.seed(s)
    Xo, yo = aug(X, y)
    assert np.allclose(yo, [[116.7, 20.7]], atol=1e-4)
    assert torch.equal(Xo[0, 127 - 5, :], X[0, 5, :])


def test_flip_frequency_and_independence():
    aug = SpotAugmentation(flip=True, rotation_range=0.0, zoom_range=None)
    X = torch.zeros(1, 128, 128)
    X[0, 0, 0] = 1.0                     # marks the corner so we can see which flips happened
    y = np.array([[0.0, 0.0]], dtype=np.float32)
    np.random.seed(0)
    n = 2500
    h = v = both = 0
    for _ in range(n):
        Xo, yo = aug(X, y)
        flipped_h = yo[0, 1] == 127
        flipped_v = yo[0, 0] == 127
        assert Xo[0, int(yo[0, 0]), int(yo[0, 1])] == 1.0      # pixels and label agree
        h += flipped_h; v += flipped_v; both += flipped_h and flipped_v
    assert abs(h / n - 0.5) < 0.05 and abs(v / n - 0.5) < 0.05
    assert abs(both / n - (h / n) * (v / n)) < 0.05             # independent
    # no affine: every output value is exactly 0 or 1 (no interpolation)
    assert set(torch.unique(Xo).tolist()) <= {0.0, 1.0}


def test_default_ranges():
    """Defaults are the reference: flips, rotation +-180 degrees, zoom 0.8-1.2."""
    aug = SpotAugmentation()
    assert aug.flip and aug.uses_affine
    assert aug.rotation_range == 180.0 and aug.zoom_range == (0.8, 1.2)
    np.random.seed(0)
    draws = np.array([aug._draw_affine() for _ in range(2000)])
    assert draws[:, 0].min() >= -180.0 and draws[:, 0].max() <= 180.0
    assert draws[:, 0].min() < -170.0 and draws[:, 0].max() > 170.0       # spread over the range
    assert draws[:, 1].min() >= 0.8 and draws[:, 1].max() <= 1.2
    assert draws[:, 1].min() < 0.81 and draws[:, 1].max() > 1.19


def test_geometric_augmentation_turned_off():
    aug = SpotAugmentation(flip=False, rotation_range=0.0, zoom_range=None)
    assert not aug.uses_affine
    X = torch.arange(128 * 128, dtype=torch.float32).reshape(1, 128, 128)
    y = np.array([[10.3, 20.7]], dtype=np.float32)
    Xo, yo = aug(X, y)
    assert torch.equal(Xo, X) and np.allclose(yo, y)


@pytest.mark.parametrize("angle,zoom,expected", [
    (180, 1.0, (97.0, 27.0)),
    (45, 1.0, None),
    (180, 1.2, None),
])
def test_rotation_keeps_labels_on_spots(angle, zoom, expected):
    """Fixed angles (the spec's spot at (30, 100) leaves the image at some random angles)."""
    aug = SpotAugmentation(flip=False, rotation_range=angle, zoom_range=(zoom, zoom))
    M = aug._forward_matrix(angle, zoom)
    pt = aug._affine_points(np.array([[30.0, 100.0]]), M)[0]
    if expected is not None:
        assert np.allclose(pt, expected, atol=1e-9)
    assert 10 < pt[0] < 117 and 10 < pt[1] < 117      # spot stays well inside, so COM is unbiased
    Xo = aug._affine_image(_gauss(30, 100), M)
    cy, cx = _com(Xo)
    assert abs(cy - pt[0]) <= 0.05 and abs(cx - pt[1]) <= 0.05


def test_random_rotation_and_zoom_keep_labels_on_spots():
    """End to end through forward(): a spot near the centre stays in view for every draw."""
    aug = SpotAugmentation(flip=False, rotation_range=180.0, zoom_range=(0.8, 1.25))
    for s in range(20):
        np.random.seed(s)
        Xo, yo = aug(_gauss(50, 70), np.array([[50.0, 70.0]], dtype=np.float32))
        assert len(yo) == 1
        cy, cx = _com(Xo)
        assert abs(cy - yo[0, 0]) <= 0.05 and abs(cx - yo[0, 1]) <= 0.05


def test_fill_does_not_create_artificial_edges():
    aug = SpotAugmentation(flip=False, rotation_range=45.0, zoom_range=(0.8, 0.8))
    rng = np.random.RandomState(0)
    X = torch.from_numpy(rng.uniform(-0.5, 0.5, (1, 128, 128))).float()
    np.random.seed(1)
    Xo, _ = aug(X, np.zeros((0, 2), dtype=np.float32))
    assert Xo.min() >= X.min() - 1e-6 and Xo.max() <= X.max() + 1e-6
    # zoom-out exposes corners: they must take edge values, not a constant fill like -1
    assert Xo.min() > -0.9
