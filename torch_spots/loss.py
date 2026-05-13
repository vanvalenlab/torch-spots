"""Custom loss functions for DeepCell spots"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_l1(y_true, y_pred, sigma=3.0):
    """Compute the smooth L1 loss of `y_pred` w.r.t. `y_true`.

    Args:
        y_true: Tensor of shape `(B, ?, ?)`.
        y_pred: Tensor of shape `(B, ?, ?)`. Same shape as `y_true`.
        sigma: The point where the loss changes from L2 to L1.

    Returns:
        The pixelwise smooth L1 loss of `y_pred` w.r.t. `y_true`.
        Has same shape as each of the inputs: `(B, ?, ?)`.
    """
    sigma_squared = sigma ** 2

    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma^2
    #        |x| - 0.5 / sigma^2            otherwise
    regression_diff = torch.abs(y_true - y_pred)

    regression_loss = torch.where(
        regression_diff < (1.0 / sigma_squared),
        0.5 * sigma_squared * regression_diff ** 2,
        regression_diff - 0.5 / sigma_squared
    )
    return regression_loss


def weighted_categorical_crossentropy(y_true, y_pred, n_classes=2, eps=1e-7):
    """Weighted categorical cross-entropy matching DeepCell's formulation.

    Args:
        y_true: Tensor of shape `(B, C, H, W)`, one-hot encoded.
        y_pred: Tensor of shape `(B, C, H, W)`, predicted probabilities.
        n_classes: Number of classes.
        eps: Small value to avoid log(0).

    Returns:
        Per-pixel weighted loss tensor of shape `(B, H, W)`.
    """
    y_pred = torch.clamp(y_pred, eps, 1.0 - eps)

    # Class weights: inverse of class frequency across batch and spatial dims
    # Sum over (B, H, W), keeping C, shape: (C,)
    class_counts = y_true.sum(dim=(0, 2, 3)) + eps
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * n_classes  # (C,)

    # Broadcast weights over (B, H, W): reshape to (1, C, 1, 1)
    weights = weights.view(1, -1, 1, 1)

    # Cross-entropy per pixel: sum over C, shape: (B, H, W)
    ce = -(weights * y_true * torch.log(y_pred)).sum(-1)
    return ce


def weighted_focal_loss(y_true, y_pred, gamma=2.0, n_classes=2, eps=1e-7):
    """Weighted focal loss matching DeepCell's formulation.

    Args:
        y_true: Tensor of shape `(B, C, H, W)`, one-hot encoded.
        y_pred: Tensor of shape `(B, C, H, W)`, predicted probabilities.
        gamma: Focusing parameter.
        n_classes: Number of classes.
        eps: Small value to avoid log(0).

    Returns:
        Scalar focal loss.
    """
    y_pred = torch.clamp(y_pred, eps, 1.0 - eps)

    class_counts = y_true.sum(dim=(0, 2, 3)) + eps
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * n_classes  # (C,)
    weights = weights.view(1, -1, 1, 1)

    focal_weight = (1.0 - y_pred) ** gamma
    ce = -(weights * focal_weight * y_true * torch.log(y_pred)).sum(dim=1)
    return ce.mean()


class DotNetLosses(nn.Module):
    def __init__(self,
                 gamma=2.0,
                 sigma=3.0,
                 n_classes=2,
                 focal=False,
                 d_pixels=1,
                 mu=0,
                 beta=0):
        super().__init__()
        self.gamma = gamma
        self.sigma = sigma
        self.n_classes = n_classes
        self.focal = focal
        self.d_pixels = d_pixels
        self.mu = mu
        self.beta = beta

    def regression_loss(self, y_true, y_pred):
        """
        Calculates the regression loss of the shift from pixel center, only
        for pixels containing a dot (true regression shifts smaller in
        absolute value than 0.5).

        Args:
            y_true: tensor of shape `(batch, 2, Ly, Lx)`.
            y_pred: tensor of shape `(batch, 2, Ly, Lx)`.
                Channel 0 contains `delta_y`, channel 1 contains `delta_x`.

        Returns:
            float: the normalized smooth L1 loss over all input pixels with
            regressed point within the same pixel.
        """
        d_pixels = self.d_pixels
        sigma = self.sigma

        y_offset_true = y_true[:, 0, :, :]  # (batch, Ly, Lx)
        x_offset_true = y_true[:, 1, :, :]
        y_offset_pred = y_pred[:, 0, :, :]
        x_offset_pred = y_pred[:, 1, :, :]

        d = 0.5 + d_pixels

        near_pt_y = (-d <= y_offset_true) & (y_offset_true < d)
        near_pt_x = (-d <= x_offset_true) & (x_offset_true < d)
        near_pt_mask = near_pt_y & near_pt_x  # (batch, Ly, Lx)

        y_offset_true_cp = y_offset_true[near_pt_mask]
        x_offset_true_cp = x_offset_true[near_pt_mask]
        y_offset_pred_cp = y_offset_pred[near_pt_mask]
        x_offset_pred_cp = x_offset_pred[near_pt_mask]

        pixelwise_loss_y = smooth_l1(y_offset_true_cp, y_offset_pred_cp, sigma=sigma)
        pixelwise_loss_x = smooth_l1(x_offset_true_cp, x_offset_pred_cp, sigma=sigma)

        normalizer = max(1, near_pt_mask.sum().item())

        loss = torch.cat([pixelwise_loss_y, pixelwise_loss_x], dim=-1)
        return loss.sum() / normalizer

    def classification_loss(self, y_true, y_pred):
        """
        Args:
            y_true: tensor of size `(batch, 2, Ly, Lx)`, one-hot encoded.
            y_pred: tensor of size `(batch, 2, Ly, Lx)`, predicted probabilities.

        Returns:
            float: focal / weighted categorical cross entropy loss.
        """
        if self.focal:
            loss = weighted_focal_loss(
                y_true, y_pred, gamma=self.gamma, n_classes=self.n_classes)
        else:
            loss = weighted_categorical_crossentropy(
                y_true, y_pred, n_classes=self.n_classes)
            loss = loss.mean()

        return loss

    def classification_loss_regularized(self, y_true, y_pred):
        """Regularized classification loss.

        Args:
            y_true: tensor of size `(batch, 2, Ly, Lx)`, one-hot encoded.
            y_pred: tensor of size `(batch, 2, Ly, Lx)`, predicted probabilities.

        Returns:
            float: focal / weighted categorical cross entropy loss.
        """
        mu = self.mu
        beta = self.beta

        if self.focal:
            loss = weighted_focal_loss(
                y_true, y_pred, gamma=self.gamma, n_classes=self.n_classes)
        else:
            loss = weighted_categorical_crossentropy(
                y_true, y_pred, n_classes=self.n_classes)
            loss = loss.mean()

        # L2 penalty on difference in mean spot density per image in the batch
        # Spot channel is index 1; mean over spatial dims (H, W)
        N_diff = y_pred[:, 1, :, :].mean(dim=(1, 2)) - y_true[:, 1, :, :].mean(dim=(1, 2))
        N_loss = (N_diff ** 2).mean()

        # Interaction term: penalizes adjacent positive predictions
        # Spot channel is index 1, shape: (batch, Ly, Lx)
        spot_pred = y_pred[:, 1, :, :]

        # Pad H and W dims (last two): (left, right, top, bottom)
        spot_padded = F.pad(spot_pred, (1, 1, 1, 1))  # (batch, Ly+2, Lx+2)

        inter_loss = (
            (spot_padded[:, 1:, :] * spot_padded[:, :-1, :]).sum() +
            (spot_padded[:, :, 1:] * spot_padded[:, :, :-1]).sum()
        )

        B, _, Ly, Lx = y_pred.shape
        normalizer = float(B * Ly * Lx)
        inter_loss = inter_loss / normalizer

        return loss + mu * N_loss + beta * inter_loss
    
    def forward(self, outputs, labels):

        class_weight = 5
        reg_weight = 1

        class_loss = self.classification_loss(labels['detections'], outputs['detections'])

        reg_loss = self.regression_loss(labels['offsets'], outputs['offsets'])

        loss = (reg_loss*reg_weight + class_loss*class_weight)/(class_weight+reg_weight)
        
        return loss
    
class LossTracker:
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_loss = 0.0
        self.total_samples = 0

    def update(self, loss, batch_size):
        """
        Args:
            loss: scalar loss value
            batch_size: batch size
        """

        self.total_loss += loss.item() * batch_size
        self.total_samples += batch_size
    
    def get_loss(self):
        """Compute and return current loss."""
        avg_loss = self.total_loss / max(self.total_samples, 1)

        return avg_loss 
    

if __name__ == '__main__':

    print()

    test_class = torch.rand((10, 2, 128, 128))
    test_reg = torch.rand((10, 2, 128, 128))

    output = {
        'detections': test_class,
        'offsets': test_reg
    }

    true_class = torch.rand((10, 2, 128, 128))
    true_reg = torch.rand((10, 2, 128, 128))

    labels = {
        'detections': true_class,
        'offsets': true_reg
    }

    loss = DotNetLosses(gamma=3.0, sigma=0.5, focal=False)

    curr_loss = loss(labels, output)
    print(curr_loss.item())