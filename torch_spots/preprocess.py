from pathlib import Path
import os
import zarr
import glob
import numpy as np
import pandas as pd

def convert_to_zarr(filename, out_dir=None):

    # there are four very poorly segmented images in the beginnign of the test split
    # remove those

    if out_dir is None:
        file_dir = os.path.dirname(filename)
        split = os.path.splitext(os.path.basename(filename))[0]

    print(f"    Loading {split}.")
    data = np.load(os.path.join(filename), allow_pickle=True)
    X = data['X']
    y = data['y']
    
    ## header is repeated 3 additional times in test split 
    #throwing off cropping X and y
    if split == 'test':
        offset = 3 

    # Make it channels first like PyTorch is expecting
    X = np.moveaxis(X, -1, 1)
    
    X = X[:].astype(np.float32)

    B, C, H, W = X.shape

    # Create a Zarr store
    store = zarr.open(f"{file_dir}/{split}.zarr", mode="w")

    print(f"    Writing {split}.")

    images = store.create_dataset(
        "X",
        shape=(B, C, H, W),
        chunks=(1, C, H, W),  # One sample per chunk — common for ML
        dtype="float32",
    )
    images[:] = X

    N_max = max(len(pts) for pts in y)

    points = store.create_dataset(
        "y",
        shape=(B, N_max, 2),
        chunks=(1, N_max, 2),
        dtype="float32",
        fill_value=0.0,
    )
    lengths = store.create_dataset("y_inds", shape=(B,), dtype="int32")

    for i, pts in enumerate(y):
        points[i, :len(pts)] = pts
        lengths[i] = len(pts)

if __name__ == "__main__":

    data_directory = Path.home() / ".deepcell/spotnet/*.npz"

    if not (fnames := list(glob.glob(str(data_directory)))):
        raise ValueError("Tissuenet data not found at {data_directory}")

    for filename in fnames:
        print(f"Converting {os.path.basename(filename)}")
        convert_to_zarr(filename)