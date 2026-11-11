import numpy as np
import zarr

from torch_spots import dataset as ds
from torch_spots.preprocess import convert_to_zarr, file_crc32


def test_convert_records_provenance_and_is_clean_v3(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.random((3, 16, 16, 1)).astype(np.float32)
    y = np.empty(3, dtype=object)
    for i, n in enumerate((2, 0, 4)):
        y[i] = rng.uniform(0, 16, size=(n, 2))
    npz = tmp_path / "test.npz"
    np.savez(npz, X=X, y=y)
    crc_before = file_crc32(npz)
    # a stale v2 marker must not survive conversion
    (tmp_path / "test.zarr").mkdir()
    (tmp_path / "test.zarr" / ".zgroup").write_text("{}")

    store = convert_to_zarr(npz)
    g = zarr.open_group(store, mode="r")
    assert g.attrs[ds.ATTR_CRC32] == crc_before
    assert g.attrs[ds.ATTR_N_IMAGES] == 3 and g.attrs[ds.ATTR_N_SPOTS] == 6
    assert g.attrs[ds.ATTR_VERSION] == "unknown"  # not the pinned file
    assert g["X"].shape == (3, 1, 16, 16)
    assert list(g["y_inds"][:]) == [2, 0, 4]
    assert not (tmp_path / "test.zarr" / ".zgroup").exists()
    assert file_crc32(npz) == crc_before
