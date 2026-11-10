"""Tests for the SpotNet architecture (spec: detection-model)."""
import torch

from torch_spots.detection.dotnet import SpotNet


def _first_129_channel_conv(model):
    for m in model.backbone.subnets[1].modules():
        if isinstance(m, torch.nn.Conv2d) and m.in_channels == 129:
            return m
    raise AssertionError("no 129-channel conv found")


def test_backbone_kernel_statistics():
    torch.manual_seed(0)
    w = _first_129_channel_conv(SpotNet()).weight.detach()
    assert tuple(w.shape) == (32, 129, 4, 4)       # fan_in 2,064
    assert abs(w.std().item() - 0.03113) / 0.03113 < 0.03
    assert abs(w.mean().item()) < 0.002
    assert w.abs().max().item() <= 0.07078 + 1e-6


def test_regression_head_kernel_statistics():
    torch.manual_seed(0)
    model = SpotNet()
    first = next(m for m in model.offset_regression_head.modules()
                 if isinstance(m, torch.nn.Conv2d))
    assert (first.in_channels, first.out_channels) == (128, 256)
    assert abs(first.weight.std().item() - 0.01) / 0.01 < 0.03


def test_zero_biases_and_unit_batchnorm():
    torch.manual_seed(0)
    model = SpotNet()
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) and m.bias is not None:
            assert torch.count_nonzero(m.bias) == 0
        if isinstance(m, torch.nn.BatchNorm2d):
            assert torch.all(m.weight == 1) and torch.all(m.bias == 0)


def test_seeded_initialisation_identical():
    torch.manual_seed(7)
    a = SpotNet()
    torch.manual_seed(7)
    b = SpotNet()
    for (na, pa), (nb, pb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert na == nb
        assert torch.equal(pa, pb), na


# ---------------------------------------------------------------------------
# Backbone geometry (spec: Backbone sub-network layers, Symmetric receptive
# field, Border padding once per sub-network)
# ---------------------------------------------------------------------------
import torch.nn.functional as F

from torch_spots.detection.modules import BNFeatureNet2D, BNFeatureNetSkip2D


def _influence_window(fn, size=128, centre=64, n_inputs=8, seed=0):
    """Rows/columns of the input that influence ``fn``'s output at (centre, centre).

    Measured as the union of gradient support over ``n_inputs`` seeded random
    inputs, so a ReLU or max-pool that happens to block one path for one input
    does not shrink the window.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_inputs, 1, size, size, generator=g, requires_grad=True)
    fn(x)[..., centre, centre].sum().backward()
    support = (x.grad != 0).any(dim=0).any(dim=0)          # (H, W)
    rows = torch.nonzero(support.any(dim=1)).flatten()
    cols = torch.nonzero(support.any(dim=0)).flatten()
    return (rows.min().item(), rows.max().item(), cols.min().item(), cols.max().item())


def test_max_pool_present_between_second_relu_and_dilated_conv():
    net = BNFeatureNet2D(1)
    layers = list(net.conv_stack) + list(net.dense_conv)
    kinds = [type(m).__name__ for blk in layers
             for m in (blk if isinstance(blk, torch.nn.Sequential) else [blk])]
    assert kinds == ['Conv2d', 'BatchNorm2d', 'ReLU',
                     'Conv2d', 'BatchNorm2d', 'ReLU',
                     'MaxPool2d',
                     'Conv2d', 'BatchNorm2d', 'ReLU']
    pool = net.conv_stack[2]
    assert (pool.kernel_size, pool.stride) == (2, 1)
    convs = [m for m in net.modules() if isinstance(m, torch.nn.Conv2d)]
    assert [(c.out_channels, c.kernel_size, c.dilation, c.padding) for c in convs] == [
        (32, (4, 4), (1, 1), (0, 0)), (32, (3, 3), (1, 1), (0, 0)),
        (128, (4, 4), (2, 2), (0, 0))]


def test_skip_wiring_unchanged():
    net = BNFeatureNetSkip2D(input_channels=1, norm_method=None)
    assert [s.in_channels for s in net.subnets] == [1, 129, 129, 129]
    seen = []
    for s in net.subnets:
        s.register_forward_pre_hook(lambda m, a: seen.append(a[0].clone()))
    x = torch.randn(1, 1, 32, 32)
    net.eval()
    with torch.no_grad():
        net.last_only = False
        outs = net(x)
    assert torch.equal(seen[0], x)
    for i in (1, 2, 3):
        assert torch.equal(seen[i][:, :1], x)                      # image first
        assert torch.equal(seen[i][:, 1:], outs[i - 1])            # then previous output


def test_single_subnetwork_window():
    torch.manual_seed(0)
    net = BNFeatureNet2D(1).eval()
    assert _influence_window(net) == (58, 70, 58, 70)


def test_full_backbone_window():
    torch.manual_seed(0)
    model = SpotNet(norm_method=None).eval()
    assert _influence_window(lambda x: model(x)['detections'][:, 1]) == (40, 88, 40, 88)


def test_equivalent_to_one_upfront_reflection():
    torch.manual_seed(0)
    net = BNFeatureNet2D(3).eval()
    x = torch.randn(2, 3, 40, 52)
    padded = F.pad(x, (6, 6, 6, 6), mode='reflect')
    with torch.no_grad():
        expected = net.dense_conv(net.conv_stack(padded))
        assert torch.allclose(net(x), expected, atol=1e-6)


def test_shapes_and_softmax():
    model = SpotNet(norm_method=None).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 1, 128, 128))
    assert out['detections'].shape == (2, 2, 128, 128)
    assert out['offsets'].shape == (2, 2, 128, 128)
    assert torch.allclose(out['detections'].sum(1), torch.ones(2, 128, 128), atol=1e-6)


def test_non_square_input():
    model = SpotNet(norm_method=None).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 1, 96, 160))
    assert out['detections'].shape == (1, 2, 96, 160)
    assert out['offsets'].shape == (1, 2, 96, 160)


# ---------------------------------------------------------------------------
# BatchNorm constants (spec: Normalisation-layer constants)
# ---------------------------------------------------------------------------

def test_running_mean_update():
    from torch_spots.detection.modules import _bn
    bn = _bn(1).train()
    x = torch.randn(64, 1, 8, 8)
    x = x - x.mean() + 2.0                      # batch mean exactly 2.0
    bn(x)
    assert abs(bn.running_mean.item() - 0.02) < 1e-6


def test_epsilon_in_eval_mode():
    from torch_spots.detection.modules import _bn
    bn = _bn(1).eval()
    bn.running_mean.zero_(); bn.running_var.zero_()
    out = bn(torch.full((1, 1, 1, 1), 0.1))
    assert abs(out.item() - 3.1623) < 1e-4


def test_every_batchnorm_in_default_spotnet_uses_keras_constants():
    bns = [m for m in SpotNet().modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert len(bns) == 4 * 3 + 1                # 3 per sub-network x 4, plus the head
    assert all(m.momentum == 0.01 and m.eps == 1e-3 for m in bns)


# ---------------------------------------------------------------------------
# In-model local normalisation (spec: In-model local normalisation)
# ---------------------------------------------------------------------------
import numpy as np

from torch_spots.detection.modules import local_std_normalize, whole_image_normalize
from tests.oracle import deepcell_std_normalize


def _random_image(rng, size):
    """Noisy image with an uneven (ramp) background and a few bright spots."""
    yy, xx = np.mgrid[:size, :size] / size
    img = rng.uniform(0.0, 0.5) * yy + rng.uniform(0.0, 0.5) * xx   # ramp
    img += rng.normal(0.0, 0.05, size=(size, size))                 # sensor noise
    for _ in range(rng.integers(3, 12)):
        cy, cx = rng.integers(0, size, 2)
        img += rng.uniform(0.5, 1.0) * np.exp(-(((yy * size - cy) ** 2 + (xx * size - cx) ** 2) / 4.0))
    return img.astype(np.float32)


def test_agreement_with_deepcell_formula():
    rng = np.random.default_rng(0)
    n = 0
    for size in (32, 64, 128):
        for _ in range(18):                                          # 54 images in all
            img = _random_image(rng, size)
            got = local_std_normalize(torch.from_numpy(img)[None, None])[0].numpy()
            want = deepcell_std_normalize(img[None])
            assert np.abs(got - want).max() <= 1e-5, (size, np.abs(got - want).max())
            n += 1
    assert n >= 50


def test_multichannel_divisor_matches_deepcell():
    rng = np.random.default_rng(1)
    img = np.stack([_random_image(rng, 32) for _ in range(3)])
    got = local_std_normalize(torch.from_numpy(img)[None])[0].numpy()
    assert np.abs(got - deepcell_std_normalize(img)).max() <= 1e-5


def test_default_normalisation_is_off():
    """Default is off: the first sub-network receives the input unchanged (forward hook)."""
    model = SpotNet().eval()
    assert model.backbone.norm_method is None
    seen = []
    model.backbone.subnets[0].register_forward_hook(lambda m, inp, out: seen.append(inp[0]))
    x = torch.rand(1, 1, 32, 32)
    with torch.no_grad():
        model(x)
    assert torch.equal(seen[0], x)


def test_selected_normalisation_is_applied():
    model = SpotNet(norm_method='std').eval()
    assert model.backbone.norm_method == 'std'
    x = torch.rand(1, 1, 32, 32)
    assert torch.allclose(model.backbone._normalize(x), local_std_normalize(x))
    seen = []
    model.backbone.subnets[0].register_forward_hook(lambda m, inp, out: seen.append(inp[0]))
    with torch.no_grad():
        model(x)
    assert torch.allclose(seen[0], local_std_normalize(x))


def test_constant_images_stay_finite():
    for value in (0.0, 0.7):
        out = local_std_normalize(torch.full((1, 1, 40, 40), value))
        assert torch.isfinite(out).all()
        if value == 0.0:
            assert torch.count_nonzero(out) == 0


def test_whole_image_uses_population_std_plus_epsilon():
    x = torch.rand(2, 1, 16, 16)
    want = (x - x.mean((-2, -1), keepdim=True)) / (
        torch.from_numpy(np.std(x.numpy(), axis=(-2, -1), keepdims=True)) + 1e-7)
    assert torch.allclose(whole_image_normalize(x), want, atol=1e-5)
