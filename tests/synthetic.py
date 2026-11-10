"""Helpers building tiny synthetic zarr stores in the converted-store layout."""
import numpy as np
import zarr

from torch_spots import dataset as ds


def make_store(path, n_images=4, n_spots_per_image=3, size=32, seed=0, attrs=None, split="test",
               extra_attrs=None):
    """Write a store shaped like a converted split. ``attrs=None`` writes attrs matching the actual data
    but with the *pinned* crc/version, which still fails pin verification on counts (fine for plumbing tests)."""
    rng = np.random.default_rng(seed)
    X = rng.gamma(2.0, 100.0, size=(n_images, 1, size, size)).astype(np.float32)
    y = rng.uniform(2, size - 2, size=(n_images, n_spots_per_image, 2)).astype(np.float32)
    y_inds = np.full(n_images, n_spots_per_image, dtype=np.int32)

    g = zarr.open_group(str(path), mode="w")
    g.create_array("X", data=X, chunks=(1, 1, size, size))
    g.create_array("y", data=y, chunks=(1, n_spots_per_image, 2))
    g.create_array("y_inds", data=y_inds)
    if attrs is None:
        attrs = {
            ds.ATTR_VERSION: ds.DATASET_VERSION,
            ds.ATTR_CRC32: ds.PINNED_SPLITS[split]["crc32"],
            ds.ATTR_N_IMAGES: n_images,
            ds.ATTR_N_SPOTS: int(y_inds.sum()),
        }
    g.attrs.update(attrs)
    return g


def write_checkpoint(directory, name="m.pth", norm_method=None, record=None, seed=0,
                     training_record=True):
    """Save a randomly initialised SpotNet and, unless ``training_record=False``, a
    ``training_config.json`` next to it recording its model settings.

    ``record`` is merged into the training record (e.g. to add a seed and dataset provenance).
    Returns the checkpoint path.
    """
    import json
    from pathlib import Path

    import torch

    from torch_spots.detection.dotnet import SpotNet

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.save(SpotNet(norm_method=norm_method).state_dict(), directory / name)
    if training_record:
        rec = {"model": {"norm_method": norm_method}, **(record or {})}
        with open(directory / "training_config.json", "w") as fh:
            json.dump(rec, fh)
    return directory / name
