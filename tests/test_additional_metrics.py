import numpy as np

from torch_spots.detection.metrics import PointMetrics
from torch_spots.detection.reference_metrics import REFERENCE_FIELDS


def test_vertical_offset_only():
    true = np.array([[3.0, 4.0], [10.0, 20.0]])
    pred = true + np.array([0.5, 0.0])  # 0.5 px larger row index
    r = PointMetrics(true, pred).evaluate(threshold=2)
    assert r.bias_y == 0.5 and r.bias_x == 0.0


def test_no_additional_field_collides_with_reference_name():
    fields = set(PointMetrics._empty_result(threshold=1).return_fields())
    assert not fields & set(REFERENCE_FIELDS)
    assert len(fields) == len(PointMetrics._empty_result(threshold=1).return_fields())


def test_chamfer_distinct_from_dmd():
    fields = set(PointMetrics._empty_result(threshold=1).return_fields())
    assert "chamfer_distance" in fields and "d_md" not in fields


def test_fields_match_numpy_length():
    r = PointMetrics._empty_result(threshold=1)
    assert len(r.return_fields()) == len(r.return_numpy())
