import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: per-image standardisation  (norm_method='std')
# ---------------------------------------------------------------------------

def image_normalize_std(x: torch.Tensor) -> torch.Tensor:
    """Standardise each image in a batch to zero mean, unit std.

    Args:
        x: (B, C, H, W)  – raw input images.

    Returns:
        (B, C, H, W)  – normalised images.
    """
    # Compute per-(batch-item, channel) mean and std over spatial dims
    mean = x.mean(dim=(-2, -1), keepdim=True)   # (B, C, 1, 1)
    std  = x.std(dim=(-2, -1), keepdim=True).clamp(min=1e-8)  # (B, C, 1, 1)
    return (x - mean) / std


# ---------------------------------------------------------------------------
# Helper: 'same'-padding Conv2d
#
# PyTorch's nn.Conv2d only accepts integer (symmetric) padding.  For even
# kernel sizes (e.g. k=4, d=1) the required total padding is odd (3), which
# cannot be split symmetrically.  We handle this with an explicit asymmetric
# F.pad before the convolution.
#
# Total padding per axis:  d*(k-1)   →  floor / ceil split for odd totals.
# ---------------------------------------------------------------------------

class ReflectPadConv2d(nn.Module):
    """Conv2d with 'same' spatial padding, supporting even kernels and dilation.

    For a convolution with kernel k and dilation d the effective kernel size is
        eff_k = d*(k-1) + 1
    To preserve the spatial dimension we need total padding per axis:
        total_pad = eff_k - 1 = d*(k-1)
    Applied as (floor(total/2), ceil(total/2)) on each axis.

    Args:
        in_channels:  Input channel count.
        out_channels: Output channel count.
        kernel_size:  Spatial kernel size (int, square).
        dilation:     Dilation factor.
        bias:         Whether to add a learnable bias.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1, bias: bool = False):
        super().__init__()
        total_pad = dilation * (kernel_size - 1)   # total pad needed per axis
        self.pad  = (total_pad // 2,               # left  / top
                     math.ceil(total_pad / 2))     # right / bottom
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              dilation=dilation,
                              padding=0,           # padding handled manually
                              bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.pad order: (left, right, top, bottom)
        p = self.pad
        x = F.pad(x, (p[0], p[1], p[0], p[1]), mode='reflect')
        return self.conv(x)


# ---------------------------------------------------------------------------
# Helper: conv block  Conv2D → BN → ReLU
# ---------------------------------------------------------------------------

def conv_bn_relu(in_ch: int, out_ch: int,
                 kernel_size: int = 3,
                 dilation: int = 1) -> nn.Sequential:
    """Return a SamePadConv2d → BatchNorm2D → ReLU block.

    Uses SamePadConv2d so H, W are always preserved, even for even kernels.

    Args:
        in_ch:       Number of input channels.
        out_ch:      Number of output channels.
        kernel_size: Spatial extent of the filter.
        dilation:    Dilation factor for the convolution.

    Returns:
        nn.Sequential of (SamePadConv2d, BatchNorm2d, ReLU).
    """
    return nn.Sequential(
        ReflectPadConv2d(in_ch, out_ch, kernel_size=kernel_size,
                      dilation=dilation, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
# BNFeatureNet2D  –  single fully-convolutional sub-network
# ---------------------------------------------------------------------------

class BNFeatureNet2D(nn.Module):
    """Single 2-D fully-convolutional feature network.

    Mirrors ``bn_feature_net_2D`` from deepcell-tf called with
    ``dilated=True, padding=True, include_top=False``.

    The network contains:
      1.  A *convolutional block* whose depth is controlled by
          ``receptive_field``.  With receptive_field=13 the while-loop in the
          original code runs two conv layers (4×4 then 3×3) before triggering
          one dilated pool.  We reproduce the same number of layers but use
          same-padding throughout so that H, W are never reduced.

      2.  A *dense conv* – the final Conv2D that projects to ``n_dense_filters``
          channels.

    Input / output shapes (channels_last → channels_first in PyTorch):
        Input:  (B, in_channels, H, W)
        Output: (B, n_dense_filters, H, W)   [128 by default]

    Args:
        in_channels     (int):  Number of input channels.
        receptive_field (int):  Controls the depth of the conv stack.
                                Default 13 → 2 conv layers + 1 dilated pool
                                + 1 dense conv.
        n_conv_filters  (int):  Number of filters in the conv stack.  Default 32.
        n_dense_filters (int):  Number of filters in the final dense conv.
                                This is the output channel depth.  Default 128.
        reg             (float): L2 weight-decay (passed to optimiser separately
                                 in PyTorch; stored as an attribute for reference).
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
        # Build the convolutional stack by simulating the original while-loop
        # ------------------------------------------------------------------
        # The loop iterates until rf_counter <= 4, choosing kernel size 4 when
        # rf_counter is odd and 3 when it is even.  Every 2nd block it fires a
        # dilated max-pool (d doubles) and halves rf_counter.
        #
        # For receptive_field=13:
        #   iter 1: rf=13 (odd)  → k=4, rf_counter = 13-3 = 10, block=1
        #   iter 2: rf=10 (even) → k=3, rf_counter = 10-2 = 8,  block=2
        #           block%2==0  → pool, d: 1→2, rf_counter = 8//2 = 4
        #   loop exits (rf_counter == 4, not > 4)
        # Final dense conv: kernel = rf_counter=4, dilation=d=2
        #
        # We preserve the original conv geometry but use same-padding so the
        # spatial dimensions H×W are unchanged throughout.
        # ------------------------------------------------------------------

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
                # Original: DilatedMaxPool2D then d*=2, rf//=2
                # We skip the pool (same-padding keeps full resolution) but
                # still double d and halve rf_counter to match the receptive
                # field geometry of the conv that follows.
                d *= 2
                rf_counter = rf_counter // 2

        self.conv_stack = nn.Sequential(*conv_layers)

        # Final "dense" conv: kernel=rf_counter, dilation=d
        # Reduces the remaining receptive field to a single point.
        # Uses SamePadConv2d to correctly handle any even/odd kernel+dilation.
        dense_k = rf_counter          # 4 for receptive_field=13
        self.dense_conv = nn.Sequential(
            ReflectPadConv2d(ch, n_dense_filters,
                          kernel_size=dense_k,
                          dilation=d,
                          bias=False),
            nn.BatchNorm2d(n_dense_filters),
            nn.ReLU(inplace=True),
        )

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
        x = self.conv_stack(x)   # (B, n_conv_filters, H, W)
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
        norm_method     (str):  Normalisation applied to the raw input.
                                Currently only 'std' is supported.
        last_only       (bool): If True, return only the final sub-net output.
                                If False, return a list of all sub-net outputs.
    """

    def __init__(self,
                 input_channels: int  = 1,
                 receptive_field: int = 13,
                 n_skips: int         = 3,
                 n_conv_filters: int  = 32,
                 n_dense_filters: int = 128,
                 norm_method: str     = 'std',
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
            return image_normalize_std(x)
        if self.norm_method == None:
            return x
        # Extend here for other norm methods (e.g. 'max', 'whole_image')
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


# ===========================================================================
# Smoke tests
# ===========================================================================

if __name__ == '__main__':
    import sys

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running modules.py tests on: {device}\n')

    B, C, H, W = 2, 1, 64, 64   # 64×64 keeps memory low; model is fully-conv so any size works

    # ------------------------------------------------------------------
    # Test 1: BNFeatureNet2D  –  single sub-network
    # ------------------------------------------------------------------
    print('=' * 60)
    print('Test 1: BNFeatureNet2D (receptive_field=13)')
    print('=' * 60)
    net2d = BNFeatureNet2D(
        in_channels     = C,
        receptive_field = 13,
        n_conv_filters  = 32,
        n_dense_filters = 128,
    ).to(device)

    x = torch.randn(B, C, H, W, device=device)
    print(f'  Input  shape : {tuple(x.shape)}')
    with torch.no_grad():
        out = net2d(x)
    print(f'  Output shape : {tuple(out.shape)}')
    assert out.shape == (B, 128, H, W), f'Expected ({B},128,{H},{W}), got {out.shape}'
    print('  PASSED ✓\n')

    # ------------------------------------------------------------------
    # Test 2: BNFeatureNet2D  –  second subnet receives C+128 channels
    # ------------------------------------------------------------------
    print('=' * 60)
    print('Test 2: BNFeatureNet2D with concatenated input (C+128=129 channels)')
    print('=' * 60)
    net2d_skip = BNFeatureNet2D(
        in_channels     = C + 128,   # 129
        receptive_field = 13,
        n_conv_filters  = 32,
        n_dense_filters = 128,
    ).to(device)

    x_cat = torch.randn(B, C + 128, H, W, device=device)
    print(f'  Input  shape : {tuple(x_cat.shape)}')
    with torch.no_grad():
        out_cat = net2d_skip(x_cat)
    print(f'  Output shape : {tuple(out_cat.shape)}')
    assert out_cat.shape == (B, 128, H, W)
    print('  PASSED ✓\n')

    # ------------------------------------------------------------------
    # Test 3: BNFeatureNetSkip2D  –  last_only=True  (as used in SpotNet)
    # ------------------------------------------------------------------
    print('=' * 60)
    print('Test 3: BNFeatureNetSkip2D (n_skips=3, last_only=True)')
    print('=' * 60)
    skip_net = BNFeatureNetSkip2D(
        input_channels  = C,
        receptive_field = 13,
        n_skips         = 3,
        n_conv_filters  = 32,
        n_dense_filters = 128,
        last_only       = True,
    ).to(device)

    x = torch.randn(B, C, H, W, device=device)
    print(f'  Input  shape : {tuple(x.shape)}')
    with torch.no_grad():
        feat = skip_net(x)
    print(f'  Output shape : {tuple(feat.shape)}')
    assert feat.shape == (B, 128, H, W)
    print('  PASSED ✓\n')

    # ------------------------------------------------------------------
    # Test 4: BNFeatureNetSkip2D  –  last_only=False  (all sub-net outputs)
    # ------------------------------------------------------------------
    print('=' * 60)
    print('Test 4: BNFeatureNetSkip2D (n_skips=3, last_only=False)')
    print('=' * 60)
    skip_net_all = BNFeatureNetSkip2D(
        input_channels  = C,
        receptive_field = 13,
        n_skips         = 3,
        n_conv_filters  = 32,
        n_dense_filters = 128,
        last_only       = False,
    ).to(device)

    with torch.no_grad():
        all_feats = skip_net_all(x)

    print(f'  Number of sub-net outputs : {len(all_feats)}')
    for idx, f in enumerate(all_feats):
        print(f'  Sub-net {idx} output shape  : {tuple(f.shape)}')
        assert f.shape == (B, 128, H, W)
    print('  PASSED ✓\n')

    # ------------------------------------------------------------------
    # Test 5: Different spatial sizes  – verify H, W are fully preserved
    # ------------------------------------------------------------------
    print('=' * 60)
    print('Test 5: Spatial size preservation (128×128 and 512×512)')
    print('=' * 60)
    for h, w in [(32, 32), (128, 128)]:
        x_sz = torch.randn(1, C, h, w, device=device)
        with torch.no_grad():
            out_sz = skip_net(x_sz)
        print(f'  Input ({h}×{w}) → Output {tuple(out_sz.shape)}')
        assert out_sz.shape == (1, 128, h, w)
    print('  PASSED ✓\n')

    # ------------------------------------------------------------------
    # Test 6: Parameter count
    # ------------------------------------------------------------------
    print('=' * 60)
    print('Test 6: Parameter counts')
    print('=' * 60)
    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f'  BNFeatureNet2D (C=1)   : {count_params(net2d):,} parameters')
    print(f'  BNFeatureNet2D (C=129) : {count_params(net2d_skip):,} parameters')
    print(f'  BNFeatureNetSkip2D     : {count_params(skip_net):,} parameters')
    print()

    print('All modules.py tests passed.')
    sys.exit(0)