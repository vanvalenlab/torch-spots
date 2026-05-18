"""Functions for image augmentation"""

import numpy as np
from torch import nn
import torchvision.transforms.v2.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import _get_inverse_affine_matrix


class SpotAugmentation(nn.Module):
    """Randomly augments an image and its associated spot coordinates.

    Applies a consistent affine transformation to both the image (via
    ``torchvision.transforms.v2.functional.affine``) and the point labels,
    using the same matrix that torchvision computes internally so that image
    pixels and spot coordinates always stay aligned.

    Args:
        aug_params (dict): Ranges for each augmentation parameter:
            - ``'theta'`` (float): max rotation in degrees, sampled from
              ``[-theta, +theta]``.
            - ``'tx'`` (float): max horizontal translation in pixels, sampled
              from ``[-tx, +tx]``.
            - ``'ty'`` (float): max vertical translation in pixels, sampled
              from ``[-ty, +ty]``.
            - ``'shear'`` (float): max shear angle in degrees, sampled from
              ``[-shear, +shear]``.
            - ``'zoom'`` (float): zoom base value in ``(0, 1]``, scale is
              sampled uniformly from ``[1/zoom, zoom]``.
        image_shape (tuple[int, int]): ``(H, W)`` of the input images.
    """

    def __init__(
            self,
            aug_params=None,
            image_shape=(128, 128)
    ):
        super().__init__()

        if aug_params is None:
            aug_params = {'theta': 180, 'tx': 0, 'ty': 0, 'shear': 0, 'zoom': 0.8}

        self.aug_params = aug_params
        self.image_shape = image_shape

    def _gen_aug_params(self):
        """Sample a fresh set of augmentation parameters."""
        self.theta = float(np.random.uniform(-self.aug_params['theta'], self.aug_params['theta']))
        self.tx = float(np.random.uniform(-self.aug_params['tx'], self.aug_params['tx']))
        self.ty = float(np.random.uniform(-self.aug_params['ty'], self.aug_params['ty']))
        self.shear = float(np.random.uniform(-self.aug_params['shear'], self.aug_params['shear']))
        self.zoom = float(np.random.uniform(1 / self.aug_params['zoom'], self.aug_params['zoom']))

    def _transform_points(self, points):
        h, w = self.image_shape
        center = [w * 0.5, h * 0.5]

        inv_matrix = _get_inverse_affine_matrix(
            center=center,
            angle=self.theta,
            translate=[self.tx, self.ty],
            scale=self.zoom,
            shear=[0.0, 0.0]
        )

        # Promote the 2x3 inverse matrix to 3x3 homogeneous form and invert it
        # to get the forward transform (input -> output)
        M_inv = np.array(inv_matrix).reshape(2, 3).astype(np.float32)
        M_inv_h = np.vstack([M_inv, [0, 0, 1]])   # (3, 3)
        M_fwd = np.linalg.inv(M_inv_h)             # forward: input -> output
        A, b = M_fwd[:2, :2], M_fwd[:2, 2]

        # points are [y, x], torchvision matrix is in [x, y] space — flip, transform, flip back
        pts_xy = np.array(points, dtype=np.float32)[:, ::-1]           # (N, 2) as [x, y]
        transformed_xy = (A @ pts_xy.T + b[:, None]).T                 # (N, 2) as [x, y]
        transformed_yx = transformed_xy[:, ::-1]                       # (N, 2) as [y, x]

        in_bounds = (
            (transformed_yx[:, 0] >= -0.5) & (transformed_yx[:, 0] <= h - 0.5) &
            (transformed_yx[:, 1] >= -0.5) & (transformed_yx[:, 1] <= w - 0.5)
        )
        return transformed_yx[in_bounds]

    def forward(self, X, y):
        """Augment an image and its spot coordinates with the same transform.

        Args:
            X (torch.Tensor): Image tensor of shape ``(C, H, W)``.
            y (np.ndarray): Spot coordinates of shape ``(N, 2)`` in
                ``[y, x]`` order.

        Returns:
            tuple[torch.Tensor, np.ndarray]: Augmented image ``(C, H, W)``
                and transformed coordinates ``(N', 2)`` in ``[y, x]`` order.
        """
        self._gen_aug_params()

        h, w = self.image_shape
        center = [w * 0.5, h * 0.5]

        X = F.affine(
            X,
            angle=self.theta,
            translate=[self.tx, self.ty],
            scale=self.zoom,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            center=center,
            fill=-1
        )

        y = self._transform_points(y).astype(np.float32)

        return X, y