"""Spot detection image datasets and DataLoaders (PyTorch)"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import distance_transform_edt

from .augmentation import SpotAugmentation


def _one_hot_2d(contains_point):
    """Convert a 2-D binary map to a one-hot encoded array of shape (H, W, 2)."""
    no_point = 1 - contains_point
    return np.stack([no_point, contains_point], axis=0).astype(np.float32)


def point_list_to_annotations(points, image_shape, dy=1, dx=1):
    """Generate label images used in loss calculation from point labels.

    Args:
        points (np.array): Array of size (N, 2) containing points as [y, x].
        image_shape (tuple): Shape of the 2-D image (H, W).
        dy (float): Pixel height.
        dx (float): Pixel width.

    Returns:
        dict: Dictionary with keys ``'detections'`` and ``'offset'``.
            - ``'detections'``: float32 array of shape (H, W, 2), one-hot
              encoding of spot locations.
            - ``'offset'``: float32 array of shape (H, W, 2), signed
              sub-pixel distance to the nearest spot in (y, x) order.
    """
    contains_point = np.zeros(image_shape, dtype=np.float32)
    for y, x in points:
        nearest_pixel_y = int(round(y / dy))
        nearest_pixel_x = int(round(x / dx))
        contains_point[nearest_pixel_y, nearest_pixel_x] = 1.0

    delta_y, delta_x, _ = subpixel_distance_transform(points, image_shape, dy=dy, dx=dx)
    offset = np.stack([delta_y, delta_x], axis=0).astype(np.float32)   # (2, H, W)
    detections = _one_hot_2d(contains_point)                              # (2, H, W)

    return {'detections': detections, 'offsets': offset}

def subpixel_distance_transform(point_list, image_shape, dy=1, dx=1):

    Ly, Lx = image_shape
    nearest_point = np.full(image_shape, np.nan)
    delta_y = np.full(image_shape, fill_value=image_shape[1], dtype=np.float32)
    delta_x = np.full(image_shape, fill_value=image_shape[0], dtype=np.float32)

    if len(point_list) == 0:
        return delta_y, delta_x, nearest_point

    # Map each occupied pixel -> last point index assigned to it
    point_array = np.asarray(point_list)
    pixel_y = np.round(point_array[:, 0] / dy).astype(int)
    pixel_x = np.round(point_array[:, 1] / dx).astype(int)

    contains_point = np.ones(image_shape, dtype=bool)
    contains_point[pixel_y, pixel_x] = False

    # For each occupied pixel, store one representative point index
    pixel_to_point = np.full(image_shape, -1, dtype=int)
    pixel_to_point[pixel_y, pixel_x] = np.arange(len(point_list))

    _, inds = distance_transform_edt(contains_point, return_indices=True, sampling=[dy, dx])

    # inds[0/1] give the y/x pixel index of the nearest occupied pixel for every pixel
    nearest_py = inds[0]  # (Ly, Lx)
    nearest_px = inds[1]  # (Ly, Lx)

    nearest_point = pixel_to_point[nearest_py, nearest_px].astype(float)

    # Pixel centers in real coordinates
    grid_y = dy * np.arange(Ly)[:, None]  # (Ly, 1)
    grid_x = dx * np.arange(Lx)[None, :]  # (1, Lx)

    chosen = nearest_point.astype(int)
    delta_y = point_array[chosen, 0] - grid_y   # real y of point minus pixel center y
    delta_x = point_array[chosen, 1] - grid_x   # real x of point minus pixel center x

    return delta_y.astype(np.float32), delta_x.astype(np.float32), nearest_point


class SpotDataset(Dataset):
    """PyTorch Dataset for fully-convolutional spot detection.

    Accepts the same ``train_dict`` format as the original
    ``ImageFullyConvDotIterator``:
        - ``train_dict['X']``: float array of shape (N, H, W, C) –
          *channels-last*.
        - ``train_dict['y']``: object array of length N where each element
          is a float array of shape (n_spots, 2) containing [y, x] spot
          coordinates.

    Augmentation is controlled by the keyword arguments below, which mirror
    the most commonly used parameters of the old ``ImageFullyConvDotDataGenerator``.

    Args:
        train_dict (dict): Dictionary with keys ``'X'`` and ``'y'``.
        rotation_range (float): Max rotation in degrees.
        width_shift_range (float): Fraction of width for horizontal shift.
        height_shift_range (float): Fraction of height for vertical shift.
        shear_range (float): Shear angle in degrees.
        zoom_range (tuple[float, float]): ``(min_zoom, max_zoom)`` range.
            Pass ``(1.0, 1.0)`` to disable zoom.
        horizontal_flip (bool): Randomly flip images horizontally.
        vertical_flip (bool): Randomly flip images vertically.
        fill_mode (str): One of ``'constant'``, ``'nearest'``,
            ``'reflect'``, ``'wrap'``.
        cval (float): Fill value used when ``fill_mode='constant'``.
        rescale (float | None): Multiplicative rescaling factor applied to
            raw pixel values before any other transform.
        samplewise_center (bool): Subtract per-sample mean.
        samplewise_std_normalization (bool): Divide by per-sample std.
        augment (bool): Whether to apply random augmentation (set to
            ``False`` for validation / test sets).
        seed (int | None): Base random seed for reproducibility.
        save_to_dir (str | None): Optional directory to save visualisations.
        save_prefix (str): Filename prefix for saved images.
        save_format (str): File extension for saved images.

    Raises:
        ValueError: If ``X`` and ``y`` have different first dimensions.
        ValueError: If ``X`` does not have rank 4.
    """

    def __init__(self,
                 X,
                 y,
                 y_inds,
                 normalize=True,
                 augment=True,
                 image_shape = (128,128)):


        self.x = X
        self.y = y
        self.y_inds = y_inds
        self.image_shape = image_shape

        # Augmentation parameters
        self.transform = SpotAugmentation()

        self.normalize = normalize
        self.augment = augment

        # Seeded RNG – each worker will fork this with its own worker seed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _minmax(self,x):

        xmin = np.min(x)
        xmax = np.max(x)
        x = (x - xmin) / (xmax - xmin)

        return x
    
    def _perc_clip(self, x):

        p1, p99 = np.percentile(x, (0.1, 99.9))
        return np.clip(x, p1, p99)

    def _standardize(self, x):
        """Apply percentile clipping, min-max normalization, then subtraction at 0.5."""
        x = self._perc_clip(x)
        x = self._minmax(x)
        x = x - 0.5

        return x

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        """Return a single (image, labels) pair.

        Returns:
            tuple:
                - **x** (torch.Tensor): Float tensor of shape (C, H, W).
                - **labels** (dict): Dictionary with two float tensors:
                    - ``'detections'``: (2, H, W) one-hot spot map.
                    - ``'offset'``: (2, H, W) sub-pixel offset map.
        """
        
        x = self.x[idx].copy()   # (C, H, W)
        y = self.y[idx]           # (N_spots, 2) with zero padding
        y = y[0:self.y_inds[idx]]  # get rid of padding

        if self.normalize:
            x = self._standardize(x)

        x = torch.from_numpy(x).type(torch.float32)

        if self.augment:
            x, y = self.transform(x, y)

        if y.shape[0] > 0:
            annotations = point_list_to_annotations(y, image_shape=self.image_shape)
        else:
            annotations = point_list_to_annotations(np.empty((0, 2)), image_shape=self.image_shape)


        # Convert to (C, H, W) tensors (channels-first for PyTorch)
        
        det_t = torch.tensor(annotations['detections'])
        off_t = torch.tensor(annotations['offsets'])

        return x, {'detections': det_t, 'offsets': off_t}


def spot_dataloader(X,
                    y,
                    y_inds,
                    batch_size=1,
                    shuffle=True,
                    num_workers=0,
                    augment=True):
    """Build a :class:`torch.utils.data.DataLoader` for spot detection.

    This is a convenience wrapper that constructs a :class:`SpotDataset` and
    wraps it in a :class:`~torch.utils.data.DataLoader`.  It is the direct
    replacement for ``ImageFullyConvDotDataGenerator.flow()``.

    Args:
        train_dict (dict): Dictionary with ``'X'`` (N, H, W, C) float array
            and ``'y'`` object array of (n_spots, 2) point arrays.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data each epoch.
        seed (int | None): Random seed for the dataset RNG.
        num_workers (int): Number of subprocesses for data loading.
            ``0`` means data is loaded in the main process.
        rotation_range (float): See :class:`SpotDataset`.
        width_shift_range (float): See :class:`SpotDataset`.
        height_shift_range (float): See :class:`SpotDataset`.
        shear_range (float): See :class:`SpotDataset`.
        zoom_range (tuple[float, float]): See :class:`SpotDataset`.
        horizontal_flip (bool): See :class:`SpotDataset`.
        vertical_flip (bool): See :class:`SpotDataset`.
        fill_mode (str): See :class:`SpotDataset`.
        cval (float): See :class:`SpotDataset`.
        rescale (float | None): See :class:`SpotDataset`.
        samplewise_center (bool): See :class:`SpotDataset`.
        samplewise_std_normalization (bool): See :class:`SpotDataset`.
        augment (bool): See :class:`SpotDataset`.
        save_to_dir (str | None): See :class:`SpotDataset`.
        save_prefix (str): See :class:`SpotDataset`.
        save_format (str): See :class:`SpotDataset`.

    Returns:
        torch.utils.data.DataLoader: Yields ``(x, labels)`` tuples where
            ``x`` is a float tensor of shape (B, C, H, W) and ``labels`` is
            a dict with keys ``'detections'`` (B, 2, H, W) and
            ``'offset'`` (B, 2, H, W).

    Example::

        loader = spot_dataloader(
            train_dict,
            batch_size=4,
            shuffle=True,
            rotation_range=15,
            horizontal_flip=True,
        )
        for x_batch, label_batch in loader:
            detections = label_batch['detections']   # (B, 2, H, W)
            offsets    = label_batch['offset']       # (B, 2, H, W)
            ...
    """

    dataset = SpotDataset(
        X,
        y,
        y_inds,
        augment=augment)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers)
