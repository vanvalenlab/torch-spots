import pytest

from torch_spots import dataset as ds
from tests.synthetic import make_store


@pytest.fixture
def small_pins(monkeypatch):
    """Pin a 5-image / 15-spot synthetic 'test' split."""
    monkeypatch.setitem(ds.PINNED_SPLITS, "test", {"crc32": "deadbeef", "n_images": 5, "n_spots": 15})


def _good(path, **kw):
    return make_store(path, n_images=5, n_spots_per_image=3, attrs={
        ds.ATTR_VERSION: ds.DATASET_VERSION, ds.ATTR_CRC32: "deadbeef",
        ds.ATTR_N_IMAGES: 5, ds.ATTR_N_SPOTS: 15, **kw})


def test_pass(tmp_path, small_pins):
    _good(tmp_path / "t.zarr")
    prov = ds.verify_store(tmp_path / "t.zarr", "test")
    assert prov["source_crc32"] == "deadbeef" and prov["n_images"] == 5 and prov["n_spots"] == 15


def test_wrong_image_count(tmp_path, small_pins):
    make_store(tmp_path / "t.zarr", n_images=4, n_spots_per_image=3, attrs={
        ds.ATTR_VERSION: ds.DATASET_VERSION, ds.ATTR_CRC32: "deadbeef",
        ds.ATTR_N_IMAGES: 5, ds.ATTR_N_SPOTS: 15})  # attributes lie
    with pytest.raises(ds.DatasetMismatchError, match="images"):
        ds.verify_store(tmp_path / "t.zarr", "test")


def test_wrong_spot_count(tmp_path, small_pins):
    make_store(tmp_path / "t.zarr", n_images=5, n_spots_per_image=2, attrs={
        ds.ATTR_VERSION: ds.DATASET_VERSION, ds.ATTR_CRC32: "deadbeef",
        ds.ATTR_N_IMAGES: 5, ds.ATTR_N_SPOTS: 15})
    with pytest.raises(ds.DatasetMismatchError, match="spots"):
        ds.verify_store(tmp_path / "t.zarr", "test")


def test_wrong_checksum(tmp_path, small_pins):
    _good(tmp_path / "t.zarr", **{ds.ATTR_CRC32: "00000000"})
    with pytest.raises(ds.DatasetMismatchError, match=ds.ATTR_CRC32):
        ds.verify_store(tmp_path / "t.zarr", "test")


def test_missing_attributes(tmp_path, small_pins):
    make_store(tmp_path / "t.zarr", n_images=5, n_spots_per_image=3, attrs={})
    with pytest.raises(ds.DatasetMismatchError, match="missing attribute"):
        ds.verify_store(tmp_path / "t.zarr", "test")


def test_wrong_version(tmp_path, small_pins):
    _good(tmp_path / "t.zarr", **{ds.ATTR_VERSION: "SpotNet-v1_0"})
    with pytest.raises(ds.DatasetMismatchError, match=ds.ATTR_VERSION):
        ds.verify_store(tmp_path / "t.zarr", "test")


def test_pinned_table_values():
    assert ds.PINNED_SPLITS["train"] == {"crc32": "4cb57ba8", "n_images": 849, "n_spots": 17410}
    assert ds.PINNED_SPLITS["val"] == {"crc32": "29836378", "n_images": 95, "n_spots": 1957}
    assert ds.PINNED_SPLITS["test"] == {"crc32": "36c922d7", "n_images": 94, "n_spots": 995}
