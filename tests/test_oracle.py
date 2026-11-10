import sys

import numpy as np


def test_oracle_is_tf_free_and_matches_spec_example(oracle):
    pm, _ = oracle
    assert "tensorflow" not in sys.modules
    true = np.array([[0.0, 0.0], [10.0, 0.0]])
    pred = np.array([[1.0, 0.0], [-8.5, 0.0]])
    s = pm.stats_points(true, pred, 1)
    assert s["precision"] == 0.5 and s["recall"] == 0.5
