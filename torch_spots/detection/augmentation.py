"""Image augmentation that moves pixels and spot coordinates together.

Follows the reference training recipe (``deepcell_spots/training.py:38-42,144-145``,
``deepcell_spots/utils/augmentation_utils.py:285-309``): by default random
horizontal and vertical flips, rotation in +-180 degrees and zoom in [0.8, 1.2] (paper Methods).
Each can be turned off for ablations.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class SpotAugmentation(nn.Module):
    """Randomly augments an image and its spot coordinates with one shared transform.

    Order of operations (as in the original): optional rotation/zoom first, then flips.

    Rotation and zoom are done with one 2x2 matrix about the image centre ``((H-1)/2, (W-1)/2)``
    in pixel-index coordinates. The matrix is applied to the spot coordinates directly, and its
    inverse is used to build the sampling grid for the image, so the two cannot disagree about
    where the centre is. Pixels that fall outside the source image take the nearest edge value.

    Flips are exact pixel re-orderings. A flipped coordinate ``c`` on an axis of length ``L``
    becomes ``L - 1 - c``.

    Args:
        flip (bool): Draw independent horizontal and vertical flips, each with probability 0.5.
        rotation_range (float): Rotation angle is drawn uniformly from
            ``[-rotation_range, +rotation_range]`` degrees. ``0`` turns rotation off.
        zoom_range (tuple[float, float] | None): ``(low, high)`` zoom factor drawn uniformly.
            A factor above 1 magnifies. ``None`` turns zoom off.
        image_shape (tuple[int, int]): ``(H, W)`` of the input images.
    """

    def __init__(self, flip=True, rotation_range=180.0, zoom_range=(0.8, 1.2), image_shape=(128, 128)):
        super().__init__()
        self.flip = bool(flip)
        self.rotation_range = float(rotation_range or 0.0)
        self.zoom_range = None if zoom_range is None else (float(zoom_range[0]), float(zoom_range[1]))
        self.image_shape = tuple(image_shape)

    @property
    def uses_affine(self):
        return self.rotation_range != 0.0 or self.zoom_range is not None

    def _draw_affine(self):
        theta = float(np.random.uniform(-self.rotation_range, self.rotation_range)) \
            if self.rotation_range != 0.0 else 0.0
        zoom = float(np.random.uniform(*self.zoom_range)) if self.zoom_range is not None else 1.0
        return theta, zoom

    def _forward_matrix(self, theta_deg, zoom):
        """2x2 matrix mapping (y, x) offsets from the centre in the input to the output."""
        t = np.deg2rad(theta_deg)
        c, s = np.cos(t), np.sin(t)
        return zoom * np.array([[c, -s], [s, c]], dtype=np.float64)

    def _affine_image(self, X, M):
        h, w = self.image_shape
        centre = np.array([(h - 1) / 2.0, (w - 1) / 2.0])
        M_inv = np.linalg.inv(M)
        ys, xs = torch.meshgrid(
            torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing='ij')
        out = torch.stack([ys - centre[0], xs - centre[1]], dim=-1)             # (H, W, 2) as (y, x)
        src = out @ torch.from_numpy(M_inv.T) + torch.from_numpy(centre)        # source index of each output pixel
        # align_corners=True: -1 and +1 are the centres of pixels 0 and L-1
        gx = src[..., 1] / (w - 1) * 2 - 1
        gy = src[..., 0] / (h - 1) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).to(X.dtype)
        out_img = F.grid_sample(X.unsqueeze(0), grid, mode='bilinear', padding_mode='border',
                                align_corners=True)
        return out_img[0]

    def _affine_points(self, points, M):
        h, w = self.image_shape
        centre = np.array([(h - 1) / 2.0, (w - 1) / 2.0])
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return (pts - centre) @ M.T + centre

    def forward(self, X, y):
        """Augment an image and its spot coordinates with the same transform.

        Args:
            X (torch.Tensor): Image tensor of shape ``(C, H, W)``.
            y (np.ndarray): Spot coordinates of shape ``(N, 2)`` in ``[y, x]`` order.

        Returns:
            tuple[torch.Tensor, np.ndarray]: Augmented image ``(C, H, W)`` and coordinates
                ``(N', 2)`` in ``[y, x]`` order (points moved outside the image are dropped).
        """
        h, w = self.image_shape
        pts = np.asarray(y, dtype=np.float64).reshape(-1, 2)

        if self.uses_affine:
            theta, zoom = self._draw_affine()
            M = self._forward_matrix(theta, zoom)
            X = self._affine_image(X, M)
            pts = self._affine_points(pts, M)
            keep = ((pts[:, 0] >= -0.5) & (pts[:, 0] <= h - 0.5) &
                    (pts[:, 1] >= -0.5) & (pts[:, 1] <= w - 0.5))
            pts = pts[keep]

        if self.flip:
            if np.random.random() < 0.5:     # horizontal: flips the x axis
                X = torch.flip(X, dims=[-1])
                pts = pts.copy()
                pts[:, 1] = (w - 1) - pts[:, 1]
            if np.random.random() < 0.5:     # vertical: flips the y axis
                X = torch.flip(X, dims=[-2])
                pts = pts.copy()
                pts[:, 0] = (h - 1) - pts[:, 0]

        return X, pts.astype(np.float32)
