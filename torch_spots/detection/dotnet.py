import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_spots.detection.modules import BNFeatureNetSkip2D, _bn


# ---------------------------------------------------------------------------
# Classification head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """Per-pixel binary classification head.

    Mirrors the TensorFlow ``classification_head``:
        TensorProduct(n_dense_filters) → BN → ReLU →
        TensorProduct(n_features)      → Softmax

    TensorProduct is a 1×1 Conv2D (no spatial context, channel mixing only).

    Shape flow:
        Input  : (B, in_channels, H, W)   e.g. (B, 128, H, W)
        Hidden : (B, n_dense_filters, H, W)   (B, 128, H, W)
        Output : (B, n_features, H, W)    e.g. (B,   2, H, W)

    Args:
        in_channels    (int): Input channel depth.  Default 128.
        n_dense_filters(int): Hidden channel width.  Default 128.
        n_features     (int): Number of output classes per pixel.  Default 2.
    """

    def __init__(self,
                 in_channels:     int = 128,
                 n_dense_filters: int = 128,
                 n_features:      int = 2):
        super().__init__()

        self.net = nn.Sequential(
            # TensorProduct(n_dense_filters)  → BN → ReLU
            # (B, 128, H, W)  →  (B, 128, H, W)
            nn.Conv2d(in_channels, n_dense_filters, kernel_size=1, bias=False),
            _bn(n_dense_filters),
            nn.ReLU(inplace=True),

            # TensorProduct(n_features)
            # (B, 128, H, W)  →  (B, 2, H, W)
            nn.Conv2d(n_dense_filters, n_features, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)

        Returns:
            (B, n_features, H, W)  – softmax probabilities across channel dim.
        """
        logits = self.net(x)                           # (B, 2, H, W)
        return F.softmax(logits, dim=1)                # (B, 2, H, W)


# ---------------------------------------------------------------------------
# Offset regression head
# ---------------------------------------------------------------------------

class OffsetRegressionHead(nn.Module):
    """Per-pixel sub-pixel offset regression head.

    Mirrors the TensorFlow ``offset_regression_head``:
        4 × [Conv2D(regression_feature_size, 3×3) → ReLU]
        → Conv2D(2, 3×3)   (no activation – raw regression output)

    Shape flow:
        Input   : (B, in_channels,          H, W)   e.g. (B, 128, H, W)
        Hidden  : (B, regression_feature_size, H, W)   (B, 256, H, W)  × 4
        Output  : (B, 2,                    H, W)

    Args:
        in_channels            (int): Input channel depth.  Default 128.
        regression_feature_size(int): Hidden channel width.  Default 256.
    """

    def __init__(self,
                 in_channels:             int = 128,
                 regression_feature_size: int = 256):
        super().__init__()

        layers = []
        ch = in_channels

        for i in range(4):
            # Conv2D(256, 3×3, same) → ReLU
            # (B, ch, H, W)  →  (B, 256, H, W)
            layers.append(nn.Conv2d(ch, regression_feature_size,
                                    kernel_size=3, padding=1, bias=True))
            layers.append(nn.ReLU(inplace=True))
            ch = regression_feature_size

        # Final conv: (B, 256, H, W)  →  (B, 2, H, W), no activation
        layers.append(nn.Conv2d(ch, 2, kernel_size=3, padding=1, bias=True))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)

        Returns:
            (B, 2, H, W)  – (Δy, Δx) sub-pixel offsets, unbounded floats.
        """
        return self.net(x)   # (B, 2, H, W)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_spotnet(checkpoint_path, device='cpu', **model_kwargs) -> 'SpotNet':
    """Build a SpotNet and load a saved state dict into it.

    ``model_kwargs`` are passed to :class:`SpotNet`; the defaults build the published model.
    Weights load strictly, so a checkpoint whose parameters do not match the network fails.
    """
    model = SpotNet(**model_kwargs)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

# Standard deviation of a unit normal truncated at +/- 2 sigma.  Keras divides
# by this so the truncated distribution ends up with the intended std.
_TRUNC2_STD = 0.87962566103423978


def init_reference_weights(model: nn.Module) -> None:
    """Initialise a SpotNet the way deepcell_spots 0.4.2 does.

    * Backbone and classification-head kernels: Keras ``he_normal`` -- a normal
      distribution cut off at 2 standard deviations, scaled so that its
      standard deviation is sqrt(2 / fan_in).
    * Regression-head kernels: normal with std 0.01.
    * Every bias: 0.  BatchNorm scale 1 and shift 0.

    Sources: ``deepcell/model_zoo/featurenet.py:58,150,199`` and
    ``deepcell_spots/dotnet.py:39,61,65,89-90``.
    """
    def he_normal(conv: nn.Conv2d) -> None:
        fan_in = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
        s = math.sqrt(2.0 / fan_in) / _TRUNC2_STD
        nn.init.trunc_normal_(conv.weight, mean=0.0, std=s, a=-2 * s, b=2 * s)

    for part in (model.backbone, model.classification_head):
        for m in part.modules():
            if isinstance(m, nn.Conv2d):
                he_normal(m)
    for m in model.offset_regression_head.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)

    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.bias is not None:
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# SpotNet  –  full model
# ---------------------------------------------------------------------------

class SpotNet(nn.Module):
    """Fully-convolutional spot detection network.

    Combines a BNFeatureNetSkip2D backbone with a classification head and an
    offset-regression head to jointly predict:

        1. A per-pixel probability that each pixel contains a spot centre.
        2. A per-pixel (Δy, Δx) sub-pixel offset to the nearest centre.

    End-to-end shape flow
    ---------------------
    Input raw image:
        (B, input_channels, H, W)        e.g. (B, 1, 256, 256)

    After BNFeatureNetSkip2D backbone (4 sub-networks, sequential refinement):
        (B, 128, H, W)

        Iteration 0:  img(B,1,H,W)          → BNFeatureNet2D → feat(B,128,H,W)
        Iteration 1:  cat(img, feat)(B,129,H,W) → BNFeatureNet2D → feat(B,128,H,W)
        Iteration 2:  cat(img, feat)(B,129,H,W) → BNFeatureNet2D → feat(B,128,H,W)
        Iteration 3:  cat(img, feat)(B,129,H,W) → BNFeatureNet2D → feat(B,128,H,W)

    After ClassificationHead:
        (B, 2, H, W)   – softmax(p_background, p_spot)

    After OffsetRegressionHead:
        (B, 2, H, W)   – (Δy, Δx) sub-pixel offset map

    Args:
        input_channels         (int):  Channels in input image.  Default 1.
        receptive_field        (int):  RF of each BNFeatureNet2D.  Default 13.
        n_skips                (int):  Skip iterations (total subnets =
                                       n_skips + 1).  Default 3.
        n_conv_filters         (int):  Conv filters in backbone subnets.
                                       Default 32.
        n_dense_filters        (int):  Dense-conv output channels in backbone,
                                       and input channels to both heads.
                                       Default 128.
        num_classes            (int):  Number of classification outputs.
                                       Default 2 (background / spot).
        regression_feature_size(int):  Hidden channels in regression head.
                                       Default 256.
        norm_method            (str):  Input normalisation: None (default, as in the
                                       published model), 'std' or 'whole_image'.
    """

    def __init__(self,
                 input_channels:          int = 1,
                 receptive_field:         int = 13,
                 n_skips:                 int = 3,
                 n_conv_filters:          int = 32,
                 n_dense_filters:         int = 128,
                 num_classes:             int = 2,
                 regression_feature_size: int = 256,
                 norm_method:             str = None):
        super().__init__()

        # ---- Backbone ----
        # Input  : (B, input_channels, H, W)
        # Output : (B, n_dense_filters, H, W)  =  (B, 128, H, W)
        self.backbone = BNFeatureNetSkip2D(
            input_channels  = input_channels,
            receptive_field = receptive_field,
            n_skips         = n_skips,
            n_conv_filters  = n_conv_filters,
            n_dense_filters = n_dense_filters,
            norm_method     = norm_method,
            last_only       = True,
        )

        # ---- Classification head ----
        # Input  : (B, n_dense_filters, H, W)  =  (B, 128, H, W)
        # Output : (B, num_classes,     H, W)  =  (B,   2, H, W)
        self.classification_head = ClassificationHead(
            in_channels     = n_dense_filters,
            n_dense_filters = n_dense_filters,
            n_features      = num_classes,
        )

        # ---- Offset regression head ----
        # Input  : (B, n_dense_filters,         H, W)  =  (B, 128, H, W)
        # Output : (B, 2,                       H, W)
        self.offset_regression_head = OffsetRegressionHead(
            in_channels             = n_dense_filters,
            regression_feature_size = regression_feature_size,
        )

        init_reference_weights(self)

    def forward(self, x: torch.Tensor):
        """Full forward pass.

        Args:
            x: (B, input_channels, H, W)  – raw fluorescence image(s).

        Returns:
            dict with keys:
                'detections'     : (B, 2, H, W)
                    Softmax probability map; channel 1 is p(spot centre).

                'offset_regression'  : (B, 2, H, W)
                    Sub-pixel offsets (Δy, Δx) from each pixel to the
                    nearest spot centre.  Meaningful only at pixels where
                    classification[:, 1] is high.
        """
        # ------------------------------------------------------------------
        # 1. Backbone  –  sequential-refinement feature extraction
        # ------------------------------------------------------------------
        # (B, input_channels, H, W)  →  (B, 128, H, W)
        features = self.backbone(x)

        # ------------------------------------------------------------------
        # 2. Classification head
        # ------------------------------------------------------------------
        # (B, 128, H, W)  →  (B, 2, H, W)
        classification = self.classification_head(features)

        # ------------------------------------------------------------------
        # 3. Offset regression head
        # ------------------------------------------------------------------
        # (B, 128, H, W)  →  (B, 2, H, W)
        offset_regression = self.offset_regression_head(features)

        return {
            'detections':    classification,    # (B, 2, H, W)
            'offsets': offset_regression, # (B, 2, H, W)
        }
