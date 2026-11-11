"""Hand-off from detection to decoding (spec: decoding-input), on synthetic detection output."""
import numpy as np

from torch_spots.detection.utils import extract_spots_prob_from_coords_maxpool

from torch_spots.detection.inference import EVAL_THRESHOLD, SpotDetection

H = W = 60
N = 20


def _detector():
    # get_spot_probabilities does not use the model, so skip loading one.
    return object.__new__(SpotDetection)


def _transforms(n_images=N, h=H, w=W):
    """Detection output with channel 1 (spot class) all zero; tests paint peaks into it."""
    return {'detections': np.zeros((n_images, 2, h, w), dtype=np.float32),
            'offsets': np.zeros((n_images, 2, h, w), dtype=np.float32)}


def _paint(transforms, image, y, x, value):
    transforms['detections'][image, 1, y, x] = value


def test_merfish_example_stack_column_i_is_image_i():
    t = _transforms()
    for i in range(N):  # a distinct pixel per image, with a distinct value
        _paint(t, i, 1 + 2 * i, 3 + i, 0.91 + 0.004 * i)
    out = _detector().get_spot_probabilities(t)
    assert out['probabilities'].shape == (N, N)
    assert out['locations'].shape == (N, 2)
    for y, x, row in zip(out['locations'][:, 0], out['locations'][:, 1], out['probabilities']):
        i = (y - 1) // 2
        assert x == 3 + i
        assert np.isclose(row[i], 0.91 + 0.004 * i)
        assert np.count_nonzero(row) == 1  # every other image is 0 here


def test_peak_in_one_image_only():
    t = _transforms()
    _paint(t, 3, 10, 20, 0.97)
    for i in range(N):
        if i != 3:
            _paint(t, i, 10, 20, 0.05)
    out = _detector().get_spot_probabilities(t)
    assert out['locations'].tolist() == [[10, 20]]
    assert np.isclose(out['probabilities'][0, 3], 0.97)
    assert np.allclose(np.delete(out['probabilities'][0], 3), 0.05)


def test_integer_locations():
    t = _transforms()
    _paint(t, 0, 5, 7, 0.95)
    _paint(t, 1, 30, 12, 0.99)
    loc = _detector().get_spot_probabilities(t)['locations']
    assert np.issubdtype(loc.dtype, np.integer)
    assert loc.min() >= 0 and loc[:, 0].max() < H and loc[:, 1].max() < W


def test_default_threshold_is_0_9_and_independent_of_eval_threshold():
    t = _transforms()
    _paint(t, 0, 5, 7, 0.95)    # above 0.9: candidate
    _paint(t, 1, 20, 20, 0.85)  # below 0.9: not a candidate
    out = _detector().get_spot_probabilities(t)
    assert out['locations'].tolist() == [[5, 7]]
    assert out['settings']['threshold'] == 0.9
    assert EVAL_THRESHOLD != out['settings']['threshold']


def test_tuned_threshold_includes_peaks_between_0_7_and_0_9():
    t = _transforms()
    _paint(t, 0, 5, 7, 0.95)
    _paint(t, 1, 20, 20, 0.75)
    out = _detector().get_spot_probabilities(t, threshold=0.7)
    assert sorted(map(tuple, out['locations'].tolist())) == [(5, 7), (20, 20)]


def test_default_pool_reads_the_pixel_itself():
    t = _transforms()
    for i in range(N):
        t['detections'][i, 1, 10, 20] = 0.5 + 0.02 * i
    _paint(t, 0, 10, 20, 0.95)
    # a larger neighbour that a pooled read would pick up, but is not a separate candidate
    t['detections'][:, 1, 10, 21] = 0.3
    out = _detector().get_spot_probabilities(t)
    row = out['probabilities'][out['locations'].tolist().index([10, 20])]
    expected = np.array([0.95] + [0.5 + 0.02 * i for i in range(1, N)])
    assert np.allclose(row, expected)


def test_pool_at_the_border_uses_only_rows_0_1_and_cols_0_1():
    # The hand-off itself never proposes a candidate on the outermost pixels (see the next
    # test), so the clipping is checked on the extraction step with the location given.
    maps = np.zeros((N, 1, H, W), dtype=np.float32)
    maps[0, 0, 0, 0] = 0.95
    maps[1, 0, 1, 1] = 0.4   # inside the window rows 0-1, columns 0-1
    maps[2, 0, 2, 2] = 0.8   # outside the window; must not be read
    maps[3, 0, 0, 2] = 0.8   # outside the window; must not be read
    out = extract_spots_prob_from_coords_maxpool(maps, np.array([[0, 0]]), pool_size=1)
    row = np.concatenate(out, axis=1)[0]
    assert np.isclose(row[0], 0.95)
    assert np.isclose(row[1], 0.4)
    assert row[2] == 0 and row[3] == 0


def test_candidates_are_never_on_the_outermost_pixels():
    # Baseline behaviour kept as is: peak_local_max excludes a border of min_distance pixels.
    t = _transforms()
    for y, x in [(0, 0), (H - 1, W - 1), (5, 0), (0, 9)]:
        _paint(t, 0, y, x, 0.99)
    _paint(t, 1, 1, 1, 0.99)
    out = _detector().get_spot_probabilities(t)
    assert out['locations'].tolist() == [[1, 1]]


def test_settings_travel_with_the_result():
    t = _transforms()
    _paint(t, 0, 5, 7, 0.85)
    out = _detector().get_spot_probabilities(t, threshold=0.8)
    assert out['settings'] == {'threshold': 0.8, 'min_distance': 1, 'pool_size': 0}
    assert set(out) == {'locations', 'probabilities', 'settings'}
