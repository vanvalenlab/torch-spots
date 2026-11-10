"""Convert the SpotNet v1.1 ``.npz`` files into zarr stores.

The ``.npz`` files are canonical and are never modified; the zarr stores are a cache that
records the source file's CRC-32, dataset version, image count and spot count as store
attributes (see :mod:`torch_spots.dataset`). Stores are written as clean zarr v3 (an existing
store is removed first, so no stale zarr v2 metadata is left behind).
"""
import os
import shutil
import zlib
from pathlib import Path

import click
import numpy as np
import zarr

from torch_spots import dataset as ds


def file_crc32(path, chunk=1 << 20) -> str:
    """CRC-32 of a file as 8 lowercase hex digits."""
    crc = 0
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            crc = zlib.crc32(block, crc)
    return f"{crc:08x}"


def convert_to_zarr(filename, out_dir=None):
    """Convert one ``<split>.npz`` to ``<out_dir>/<split>.zarr`` (default: next to the npz).

    Returns the path of the written store.
    """
    filename = os.fspath(filename)
    split = os.path.splitext(os.path.basename(filename))[0]
    out_dir = os.path.dirname(filename) if out_dir is None else os.fspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    crc = file_crc32(filename)
    pin = ds.PINNED_SPLITS.get(split)
    version = ds.DATASET_VERSION if (pin is not None and pin["crc32"] == crc) else "unknown"
    if version == "unknown":
        print(f"    WARNING: {split}.npz (CRC-32 {crc}) is not the pinned {ds.DATASET_VERSION} split; "
              "the store will fail benchmark verification.")

    print(f"    Loading {split}.")
    data = np.load(filename, allow_pickle=True)
    X = data["X"]
    y = data["y"]

    # Make it channels first like PyTorch is expecting
    X = np.moveaxis(X, -1, 1).astype(np.float32)
    B, C, H, W = X.shape

    store_path = os.path.join(out_dir, f"{split}.zarr")
    if os.path.exists(store_path):
        shutil.rmtree(store_path)  # derived cache; also clears any leftover zarr v2 metadata
    store = zarr.open_group(store_path, mode="w", zarr_format=3)

    print(f"    Writing {split}.")
    images = store.create_array("X", shape=(B, C, H, W), chunks=(1, C, H, W), dtype="float32")
    images[:] = X

    N_max = max(len(pts) for pts in y)
    points = store.create_array("y", shape=(B, N_max, 2), chunks=(1, N_max, 2), dtype="float32",
                                fill_value=0.0)
    lengths = store.create_array("y_inds", shape=(B,), dtype="int32")
    for i, pts in enumerate(y):
        points[i, : len(pts)] = pts
        lengths[i] = len(pts)

    store.attrs.update({
        ds.ATTR_VERSION: version,
        ds.ATTR_CRC32: crc,
        ds.ATTR_N_IMAGES: int(B),
        ds.ATTR_N_SPOTS: int(sum(len(pts) for pts in y)),
    })
    return store_path


@click.command()
@click.option("--data-dir", default=str(ds.DEFAULT_DATA_DIR), show_default=True,
              help="Directory holding the SpotNet v1.1 train/val/test .npz files; zarr stores are written here.")
def main(data_dir):
    names = [Path(data_dir) / f"{s}.npz" for s in ds.PINNED_SPLITS]
    missing = [str(p) for p in names if not p.is_file()]
    if missing:
        raise click.ClickException(f"SpotNet v1.1 data not found: {', '.join(missing)}")
    for path in names:
        print(f"Converting {path.name}")
        store = convert_to_zarr(path)
        ds.verify_store(store, path.stem)
        print(f"    Verified {store}")


if __name__ == "__main__":
    main()
