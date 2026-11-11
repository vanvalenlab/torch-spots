"""Pinned SpotNet v1.1 datasets and store verification.

The v1.1 ``.npz`` files are canonical; zarr stores are a verified cache.
Conversion (``torch_spots.preprocess``) records the source checksum, version, image count
and spot count as store attributes. :func:`verify_store` is shared by training and
evaluation: it checks those attributes against the pinned table below, and also the actual
array shape and ``y_inds`` sum, so a store with hand-edited attributes still fails.

A future dataset version is a one-line change to :data:`PINNED_SPLITS`.
"""
from pathlib import Path

import numpy as np
import zarr

DATASET_VERSION = "SpotNet-v1_1"

#: Default location of the v1.1 data (npz and converted zarr stores).
DEFAULT_DATA_DIR = Path.home() / ".deepcell" / "spots"

#: Source: ``~/.deepcell/spots/README.md`` (counts) and CRC-32 of the distributed npz files.
PINNED_SPLITS = {
    "train": {"crc32": "4cb57ba8", "n_images": 849, "n_spots": 17410},
    "val": {"crc32": "29836378", "n_images": 95, "n_spots": 1957},
    "test": {"crc32": "36c922d7", "n_images": 94, "n_spots": 995},
}

#: Store attribute names written by conversion and checked by verification.
ATTR_VERSION = "dataset_version"
ATTR_CRC32 = "source_crc32"
ATTR_N_IMAGES = "n_images"
ATTR_N_SPOTS = "n_spots"


class DatasetMismatchError(ValueError):
    """A zarr store does not match the pinned SpotNet v1.1 split."""


def verify_store(store, split: str) -> dict:
    """Verify ``store`` (a path or an open zarr group) is the pinned v1.1 ``split``.

    Returns:
        Provenance dict: ``dataset_version``, ``split``, ``source_crc32``, ``n_images``, ``n_spots``.

    Raises:
        DatasetMismatchError: naming each mismatching property.
    """
    if split not in PINNED_SPLITS:
        raise DatasetMismatchError(f"unknown split {split!r}; expected one of {sorted(PINNED_SPLITS)}")
    pin = PINNED_SPLITS[split]
    where = str(store) if isinstance(store, (str, Path)) else "<zarr group>"
    group = zarr.open_group(str(store), mode="r") if isinstance(store, (str, Path)) else store

    problems = []
    attrs = dict(group.attrs)
    for name in (ATTR_VERSION, ATTR_CRC32, ATTR_N_IMAGES, ATTR_N_SPOTS):
        if name not in attrs:
            problems.append(f"missing attribute {name!r}")
    if not problems:
        expected = {ATTR_VERSION: DATASET_VERSION, ATTR_CRC32: pin["crc32"],
                    ATTR_N_IMAGES: pin["n_images"], ATTR_N_SPOTS: pin["n_spots"]}
        for name, want in expected.items():
            if attrs[name] != want:
                problems.append(f"attribute {name} is {attrs[name]!r}, expected {want!r}")

    # Actual contents, independent of the attributes.
    try:
        n_images = int(group["X"].shape[0])
        n_spots = int(np.sum(np.asarray(group["y_inds"][:], dtype=np.int64)))
    except KeyError as e:
        problems.append(f"missing array {e}")
    else:
        if n_images != pin["n_images"]:
            problems.append(f"X has {n_images} images, expected {pin['n_images']}")
        if n_spots != pin["n_spots"]:
            problems.append(f"y_inds sums to {n_spots} spots, expected {pin['n_spots']}")

    if problems:
        raise DatasetMismatchError(
            f"{where} is not the pinned {DATASET_VERSION} {split} split: " + "; ".join(problems)
        )
    return {
        "dataset_version": DATASET_VERSION,
        "split": split,
        "source_crc32": pin["crc32"],
        "n_images": pin["n_images"],
        "n_spots": pin["n_spots"],
    }
