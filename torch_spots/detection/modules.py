import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# In-model input normalisation
# ---------------------------------------------------------------------------

_NORM_EPSILON = 1e-7        # Keras K.epsilon()


def _box(t: torch.Tensor, filter_size: int) -> torch.Tensor:
    """Mean over a ``filter_size`` square window, with zero padding outside the image.

    Matches deepcell's depthwise convolution with ``padding='SAME'`` and a kernel
    of ones divided by ``filter_size**2 * C`` (``deepcell/layers/normalization.py:130-142``).
    Written with ``avg_pool2d`` so it never runs in reduced (TF32) precision, which
    the subtraction in :func:`local_std_normalize` could not tolerate.
    """
    channels = t.shape[1]
    pooled = F.avg_pool2d(t, filter_size, stride=1, padding=filter_size // 2,
                          count_include_pad=True)       # window sum / filter_size**2
    return pooled / channels                            # deepcell's extra 1 / C


def local_std_normalize(x: torch.Tensor, filter_size: int = 13) -> torch.Tensor:
    """deepcell's ``norm_method='std'``: local mean removal and local std scaling.

    ``o = x - box(x)``, then ``o / (sqrt(box(o**2) - box(o)**2) + 1e-7)``, where
    ``box`` is a ``filter_size`` x ``filter_size`` window mean with zero padding
    (source: ``deepcell/layers/normalization.py:164-178,189-191``).

    The variance is clamped at 0 before the square root.  In float32 it can come
    out a few ULP below 0, where TensorFlow would return NaN.

    Args:
        x: (B, C, H, W) raw images.
    """
    o = x - _box(x, filter_size)
    var = (_box(o * o, filter_size) - _box(o, filter_size) ** 2).clamp(min=0)
    return o / (var.sqrt() + _NORM_EPSILON)


def whole_image_normalize(x: torch.Tensor) -> torch.Tensor:
    """deepcell's ``norm_method='whole_image'``: per image and channel, subtract the
    mean and divide by the population std plus 1e-7
    (source: ``deepcell/layers/normalization.py:180-183``).

    Args:
        x: (B, C, H, W) raw images.
    """
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True, unbiased=False)
    return (x - mean) / (std + _NORM_EPSILON)


# ---------------------------------------------------------------------------
# Helper: BatchNorm with Keras's constants
# ---------------------------------------------------------------------------

def _bn(n: int) -> nn.BatchNorm2d:
    """BatchNorm2d matching Keras's defaults (momentum 0.99, epsilon 1e-3).

    PyTorch's ``momentum`` is the weight given to the *new* batch, so Keras's 0.99
    on the old value is 0.01 here: ``running = 0.99 * running + 0.01 * batch``.
    Sources: ``deepcell/model_zoo/featurenet.py:153,202``, ``deepcell_spots/dotnet.py:63``.
    """
    return nn.BatchNorm2d(n, momentum=0.01, eps=1e-3)


# ---------------------------------------------------------------------------
# Helper: conv block  Conv2D (no padding) → BN → ReLU
# ---------------------------------------------------------------------------

def conv_bn_relu(in_ch: int, out_ch: int,
                 kernel_size: int = 3,
                 dilation: int = 1) -> nn.Sequential:
    """Return an unpadded Conv2d → BatchNorm2d → ReLU block.

    The convolution shrinks its input by ``dilation * (kernel_size - 1)`` pixels
    per axis.  The caller pads the image once up front (see ``BNFeatureNet2D``).

    Args:
        in_ch:       Number of input channels.
        out_ch:      Number of output channels.
        kernel_size: Spatial extent of the filter.
        dilation:    Dilation factor for the convolution.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                  dilation=dilation, padding=0, bias=False),
        _bn(out_ch),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
# BNFeatureNet2D  –  single fully-convolutional sub-network
# ---------------------------------------------------------------------------

class BNFeatureNet2D(nn.Module):
    """Single 2-D fully-convolutional feature network.

    Mirrors ``bn_feature_net_2D`` from deepcell-tf called with
    ``dilated=True, padding=True, include_top=False``.

    The input is mirror-padded once by ``(receptive_field - 1) // 2`` pixels on
    every side.  After that no layer pads anything: each convolution and the
    pooling step shrink the image, and the total shrinkage is exactly the
    padding, so the output is the same height and width as the input and each
    output pixel sees the centred ``receptive_field`` × ``receptive_field``
    window.

    With receptive_field=13 the layers are:
        4×4 conv (32) → BN → ReLU → 3×3 conv (32) → BN → ReLU →
        2×2 max-pool, stride 1 → 4×4 conv, dilation 2 (128) → BN → ReLU
    Spatial size for a 128-px input: 140 → 137 → 135 → 134 → 128.

    Input / output shapes:
        Input:  (B, in_channels, H, W)
        Output: (B, n_dense_filters, H, W)   [128 by default]

    Args:
        in_channels     (int):  Number of input channels.
        receptive_field (int):  Controls the depth of the conv stack.  Default 13.
        n_conv_filters  (int):  Number of filters in the conv stack.  Default 32.
        n_dense_filters (int):  Number of filters in the final dense conv.
                                This is the output channel depth.  Default 128.
        reg             (float): L2 weight-decay (applied in the training loop;
                                 stored as an attribute for reference).
    """

    def __init__(self,
                 in_channels: int,
                 receptive_field: int = 13,
                 n_conv_filters: int = 32,
                 n_dense_filters: int = 128,
                 reg: float = 1e-5):
        super().__init__()
        self.in_channels     = in_channels
        self.receptive_field = receptive_field
        self.n_conv_filters  = n_conv_filters
        self.n_dense_filters = n_dense_filters
        self.reg             = reg

        # ------------------------------------------------------------------
        # Build the conv stack by following deepcell's while-loop.  It runs
        # until rf_counter <= 4, using a 4×4 kernel when rf_counter is odd and
        # 3×3 when even.  After every 2nd block it applies a stride-1 2×2
        # max-pool with the *current* dilation, then doubles the dilation.
        #
        # For receptive_field=13:
        #   iter 1: rf=13 (odd)  → k=4, rf_counter = 10, block=1
        #   iter 2: rf=10 (even) → k=3, rf_counter = 8,  block=2
        #           pool (dilation 1), d: 1→2, rf_counter = 8//2 = 4
        # Final dense conv: kernel = rf_counter = 4, dilation = d = 2.
        # ------------------------------------------------------------------
        self.pad = (receptive_field - 1) // 2

        conv_layers = []
        rf_counter  = receptive_field
        block_counter = 0
        d = 1               # current dilation factor
        ch = in_channels    # running channel count

        while rf_counter > 4:
            k = 4 if rf_counter % 2 != 0 else 3   # original: odd→4, even→3
            conv_layers.append(conv_bn_relu(ch, n_conv_filters,
                                            kernel_size=k, dilation=d))
            ch = n_conv_filters
            block_counter += 1
            rf_counter -= (k - 1)

            if block_counter % 2 == 0:
                # deepcell DilatedMaxPool2D: 2×2, stride 1, current dilation
                conv_layers.append(nn.MaxPool2d(2, stride=1, dilation=d))
                d *= 2
                rf_counter = rf_counter // 2

        self.conv_stack = nn.Sequential(*conv_layers)

        # Final "dense" conv: kernel=rf_counter, dilation=d.
        dense_k = rf_counter          # 4 for receptive_field=13
        self.dense_conv = conv_bn_relu(ch, n_dense_filters,
                                       kernel_size=dense_k, dilation=d)

        # Record final dilation for inspection / tests
        self._final_dilation = d
        self._final_k        = dense_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, in_channels, H, W)

        Returns:
            (B, n_dense_filters, H, W)   e.g. (B, 128, H, W)
        """
        # One mirror reflection (edge pixel not repeated); nothing else pads.
        p = self.pad
        x = F.pad(x, (p, p, p, p), mode='reflect')   # (B, C, H+2p, W+2p)
        x = self.conv_stack(x)   # shrinks by the pool/conv geometry
        x = self.dense_conv(x)   # (B, n_dense_filters, H, W)
        return x


# ---------------------------------------------------------------------------
# BNFeatureNetSkip2D  –  sequential-refinement skip network
# ---------------------------------------------------------------------------

class BNFeatureNetSkip2D(nn.Module):
    """2-D feature network with sequential-refinement skip connections.

    Mirrors ``bn_feature_net_skip_2D`` from deepcell-tf.

    Architecture overview
    ---------------------
    A normalised copy of the input image (``img``) is computed once and reused
    across all iterations.  The network then runs ``n_skips + 1`` sub-networks
    (``BNFeatureNet2D`` instances) sequentially.

    Iteration 0:
        sub-net input = img                        (B, C,       H, W)
        sub-net output = feat_0                    (B, 128,     H, W)

    Iteration k > 0:
        sub-net input = Concat([img, feat_{k-1}])  (B, C+128,   H, W)
        sub-net output = feat_k                    (B, 128,     H, W)

    The concatenation means every refinement sub-network always has direct
    access to the original image signal alongside the accumulated prediction,
    preventing the network from "forgetting" raw pixel evidence.

    With n_skips=3 there are 4 sub-networks total.  ``last_only=True`` (the
    default, matching dot_net_2D's usage) returns only the final output.

    Final output shape: (B, n_dense_filters, H, W)  →  (B, 128, H, W)

    Args:
        input_channels  (int):  Channels in the raw input image.  Default 1.
        receptive_field (int):  Passed to every BNFeatureNet2D.  Default 13.
        n_skips         (int):  Number of skip iterations (total sub-nets =
                                n_skips + 1).  Default 3.
        n_conv_filters  (int):  Filters in conv stack of each sub-net.
                                Default 32.
        n_dense_filters (int):  Output channels of each sub-net.  Default 128.
        norm_method     (str):  Normalisation applied to the raw input: None
                                (the default, as in the published model, whose
                                normalisation layers are pass-through), 'std'
                                (local window) or 'whole_image'.
        last_only       (bool): If True, return only the final sub-net output.
                                If False, return a list of all sub-net outputs.
    """

    def __init__(self,
                 input_channels: int  = 1,
                 receptive_field: int = 13,
                 n_skips: int         = 3,
                 n_conv_filters: int  = 32,
                 n_dense_filters: int = 128,
                 norm_method: str     = None,
                 last_only: bool      = True):
        super().__init__()
        self.input_channels  = input_channels
        self.receptive_field = receptive_field
        self.n_skips         = n_skips
        self.n_dense_filters = n_dense_filters
        self.norm_method     = norm_method
        self.last_only       = last_only

        # ------------------------------------------------------------------
        # Build n_skips+1 sub-networks.
        # Sub-net 0 receives input_channels channels.
        # Sub-nets 1..n_skips receive input_channels + n_dense_filters channels
        # (original image concatenated with previous output).
        # ------------------------------------------------------------------
        self.subnets = nn.ModuleList()

        for i in range(n_skips + 1):
            in_ch = input_channels if i == 0 else (input_channels + n_dense_filters)
            # Shapes comment:
            #   i=0:  in_ch = C              (e.g. 1)
            #   i>0:  in_ch = C + 128        (e.g. 129)
            self.subnets.append(
                BNFeatureNet2D(
                    in_channels     = in_ch,
                    receptive_field = receptive_field,
                    n_conv_filters  = n_conv_filters,
                    n_dense_filters = n_dense_filters,
                )
            )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply input normalisation according to self.norm_method."""
        if self.norm_method == 'std':
            return local_std_normalize(x)
        if self.norm_method == 'whole_image':
            return whole_image_normalize(x)
        if self.norm_method is None:
            return x
        raise ValueError(f"Unsupported norm_method: '{self.norm_method}'")

    def forward(self, x: torch.Tensor):
        """Sequential-refinement forward pass.

        Args:
            x: (B, input_channels, H, W)  – raw input image.

        Returns:
            If last_only=True:
                (B, n_dense_filters, H, W)   e.g. (B, 128, H, W)
            If last_only=False:
                List of (B, n_dense_filters, H, W) tensors, one per sub-net.
        """
        # Normalise once; img is reused in every iteration
        img = self._normalize(x)         # (B, C, H, W)

        outputs = []
        feat = None                      # previous sub-net output

        for i, subnet in enumerate(self.subnets):
            if feat is None:
                # Iteration 0: only the normalised image
                # subnet input: (B, C, H, W)
                subnet_input = img
            else:
                # Iterations 1..n_skips: concatenate img + previous output
                # subnet input: (B, C + n_dense_filters, H, W)
                #                e.g. (B, 1 + 128, H, W) = (B, 129, H, W)
                subnet_input = torch.cat([img, feat], dim=1)

            feat = subnet(subnet_input)  # (B, n_dense_filters, H, W)
            outputs.append(feat)

        if self.last_only:
            return outputs[-1]           # (B, 128, H, W)
        return outputs                   # list of 4 × (B, 128, H, W)
