"""Test-only loader for the original deepcell_spots 0.4.2 metric code.

``deepcell_spots/__init__.py`` imports TensorFlow, so the package cannot be
imported. ``point_metrics.py`` and ``utils/postprocessing_utils.py`` only need
numpy, scipy and skimage, so they are loaded here by file path, with the
``deepcell_spots`` and ``deepcell_spots.utils`` package modules stubbed
(design D2).
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ORACLE_ROOT = Path(__file__).resolve().parents[1] / "deepcell_spots"

_cache = {}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_oracle():
    """Return ``(point_metrics, postprocessing_utils)`` modules of the original.

    Skips the calling test with a clear message when ``deepcell_spots/`` is absent.
    """
    if "mods" in _cache:
        return _cache["mods"]
    if not (ORACLE_ROOT / "point_metrics.py").is_file():
        pytest.skip(
            f"original deepcell_spots 0.4.2 not found at {ORACLE_ROOT}; "
            "oracle comparison tests need the vendored copy"
        )
    for pkg, path in (("deepcell_spots", ORACLE_ROOT), ("deepcell_spots.utils", ORACLE_ROOT / "utils")):
        stub = types.ModuleType(pkg)
        stub.__path__ = [str(path)]
        sys.modules[pkg] = stub
    post = _load("deepcell_spots.utils.postprocessing_utils", ORACLE_ROOT / "utils" / "postprocessing_utils.py")
    pm = _load("deepcell_spots.point_metrics", ORACLE_ROOT / "point_metrics.py")
    _cache["mods"] = (pm, post)
    return pm, post


def deepcell_std_normalize(x, filter_size=13, epsilon=1e-7):
    """NumPy transcription of deepcell's ``norm_method='std'``.

    Source: ``deepcell/layers/normalization.py:164-191`` (deepcell-tf 0.12.10):
    ``_average_filter`` is a depthwise ``'SAME'`` convolution with a kernel of ones
    divided by ``filter_size * filter_size * C`` (lines 130-142), and

        outputs = inputs - average(inputs)
        outputs = outputs / (sqrt(average(outputs**2) - average(outputs)**2) + epsilon)

    No clamp before the square root, exactly as in the original.

    Args:
        x: (C, H, W) array.  Computed in float64.
    """
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    x = np.asarray(x, dtype=np.float64)
    channels = x.shape[0]
    pad = filter_size // 2

    def average(t):
        padded = np.pad(t, ((0, 0), (pad, pad), (pad, pad)))   # zeros: 'SAME' at stride 1
        windows = sliding_window_view(padded, (filter_size, filter_size), axis=(1, 2))
        return windows.sum(axis=(-2, -1)) / (filter_size * filter_size * channels)

    o = x - average(x)
    return o / (np.sqrt(average(o * o) - average(o) * average(o)) + epsilon)
