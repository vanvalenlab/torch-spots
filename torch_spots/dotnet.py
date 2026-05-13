import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import BNFeatureNetSkip2D


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
            nn.BatchNorm2d(n_dense_filters),
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
        norm_method            (str):  Input normalisation.  Default 'std'.
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


# ===========================================================================
# Smoke tests
# ===========================================================================

if __name__ == '__main__':
    import sys

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running spotnet.py tests on: {device}\n')

    B, C, H, W = 2, 1, 64, 64   # 64×64 keeps memory low; model is fully-conv so any size works

    # ------------------------------------------------------------------
    # Test 1: ClassificationHead
    # ------------------------------------------------------------------
    print('=' * 60)
    print('Test 1: ClassificationHead')
    print('=' * 60)
    cls_head = ClassificationHead(in_channels=128, n_dense_filters=128,
                                  n_features=2).to(device)
    feat = torch.randn(B, 128, H, W, device=device)
    print(f'  Input  shape : {tuple(feat.shape)}')
    with torch.no_grad():
        cls_out = cls_head(feat)
    print(f'  Output shape : {tuple(cls_out.shape)}')
    assert cls_out.shape == (B, 2, H, W)
    # Softmax: probabilities sum to 1 over channel dim
    prob_sum = cls_out.sum(dim=1)
    assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5), \
        'Softmax probabilities do not sum to 1'
    print('  Softmax sums  : all close to 1.0 ✓')
    print('  PASSED ✓\n')

    # ------------------------------------------------------------------
    # Test 2: OffsetRegressionHead
    # ------------------------------------------------------------------
    print('=' * 60)
    print('Test 2: OffsetRegressionHead')
    print('=' * 60)
    reg_head = OffsetRegressionHead(in_channels=128,
                                    regression_feature_size=256).to(device)
    print(f'  Input  shape : {tuple(feat.shape)}')
    with torch.no_grad():
        reg_out = reg_head(feat)
    print(f'  Output shape : {tuple(reg_out.shape)}')
    assert reg_out.shape == (B, 2, H, W)
    print('  PASSED ✓\n')

    # ------------------------------------------------------------------
    # Test 3: Full SpotNet – default config
    # ------------------------------------------------------------------
    print('=' * 60)
    print('Test 3: SpotNet – default config (256×256, 1-channel input)')
    print('=' * 60)
    model = SpotNet(
        input_channels          = C,
        receptive_field         = 13,
        n_skips                 = 3,
        n_conv_filters          = 32,
        n_dense_filters         = 128,
        num_classes             = 2,
        regression_feature_size = 256,
    ).to(device)

    x = torch.randn(B, C, H, W, device=device)
    print(f'  Input  shape : {tuple(x.shape)}')
    with torch.no_grad():
        out = model(x)

    cls = out['detections']
    reg = out['offsets']
    print(f'  detections    shape : {tuple(cls.shape)}')
    print(f'  offsets shape : {tuple(reg.shape)}')
    assert cls.shape == (B, 2, H, W)
    assert reg.shape == (B, 2, H, W)
    print('  PASSED ✓\n')

    # ------------------------------------------------------------------
    # Test 4: Spatial size preservation
    # ------------------------------------------------------------------
    print('=' * 60)
    print('Test 4: Spatial size preservation')
    print('=' * 60)
    for h, w in [(32, 32), (128, 128)]:
        x_sz = torch.randn(1, C, h, w, device=device)
        with torch.no_grad():
            out_sz = model(x_sz)
        cls_sz = out_sz['detections']
        reg_sz = out_sz['offsets']
        print(f'  Input ({h}×{w}):  cls {tuple(cls_sz.shape)},  '
              f'reg {tuple(reg_sz.shape)}')
        assert cls_sz.shape == (1, 2, h, w)
        assert reg_sz.shape == (1, 2, h, w)
    print('  PASSED ✓\n')

    # ------------------------------------------------------------------
    # Test 5: Multi-channel input (e.g. 3-channel fluorescence)
    # ------------------------------------------------------------------
    print('=' * 60)
    print('Test 5: Multi-channel input (C=3)')
    print('=' * 60)
    model_mc = SpotNet(input_channels=3).to(device)
    x_mc = torch.randn(B, 3, H, W, device=device)
    print(f'  Input  shape : {tuple(x_mc.shape)}')
    with torch.no_grad():
        out_mc = model_mc(x_mc)
    assert out_mc['detections'].shape    == (B, 2, H, W)
    assert out_mc['offsets'].shape == (B, 2, H, W)
    print(f'  detections    shape : {tuple(out_mc["detections"].shape)}')
    print(f'  offsets shape : {tuple(out_mc["offsets"].shape)}')
    print('  PASSED ✓\n')

    # ------------------------------------------------------------------
    # Test 6: Output value ranges
    # ------------------------------------------------------------------
    print('=' * 60)
    print('Test 6: Output value ranges')
    print('=' * 60)
    with torch.no_grad():
        out6 = model(torch.randn(B, C, H, W, device=device))
    cls6 = out6['detections']
    reg6 = out6['offsets']
    assert cls6.min() >= 0.0 and cls6.max() <= 1.0, \
        'detections output outside [0, 1]'
    print(f'  detections  range : [{cls6.min():.4f}, {cls6.max():.4f}]  '
          f'(expected [0, 1]) ✓')
    print(f'  offsets range: [{reg6.min():.4f}, {reg6.max():.4f}]  '
          f'(unbounded floats, no constraint) ✓')
    print('  PASSED ✓\n')

    # ------------------------------------------------------------------
    # Test 7: Backward pass (gradient flow)
    # ------------------------------------------------------------------
    print('=' * 60)
    print('Test 7: Backward pass (gradient flow)')
    print('=' * 60)
    model_grad = SpotNet(
        input_channels  = C,
        receptive_field = 13,
        n_skips         = 3,
        n_conv_filters  = 32,
        n_dense_filters = 128,
    ).to(device)
    x_grad = torch.randn(B, C, H, W, device=device)
    out_grad = model_grad(x_grad)
    # Combine both heads into a scalar loss
    loss = out_grad['detections'].sum() + out_grad['offsets'].sum()
    loss.backward()
    # Verify that all parameters received a gradient
    no_grad = [n for n, p in model_grad.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not no_grad, f'Parameters with no gradient: {no_grad}'
    print('  All parameters received gradients ✓')
    print('  PASSED ✓\n')

    # ------------------------------------------------------------------
    # Test 8: Parameter count
    # ------------------------------------------------------------------
    print('=' * 60)
    print('Test 8: Parameter counts')
    print('=' * 60)
    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    total      = count_params(model)
    backbone   = count_params(model.backbone)
    cls_params = count_params(model.classification_head)
    reg_params = count_params(model.offset_regression_head)
    print(f'  Backbone                : {backbone:>10,}')
    print(f'  ClassificationHead      : {cls_params:>10,}')
    print(f'  OffsetRegressionHead    : {reg_params:>10,}')
    print(f'  Total                   : {total:>10,}')
    print()

    print('All spotnet.py tests passed.')
    sys.exit(0)