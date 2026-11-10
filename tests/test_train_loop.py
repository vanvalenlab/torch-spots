"""Training-loop behaviour (optimizer, schedule, penalty, checkpoints) on a tiny stub model."""
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_spots.detection import train as train_mod
from torch_spots.detection.loader import spot_dataloader
from torch_spots.detection.loss import kernel_l2_penalty
from torch_spots.detection.dotnet import SpotNet

H = W = 8


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU(),
                                 nn.Conv2d(4, 2, 1, bias=True))


class Stub(nn.Module):
    """Same attribute layout as SpotNet where it matters: ``classification_head.net``."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(1, 3, 3, padding=1)
        self.classification_head = Head()
        self.offset_regression_head = nn.Conv2d(3, 2, 3, padding=1)

    def forward(self, x):
        f = F.relu(self.backbone(x))
        return {'detections': F.softmax(self.classification_head.net(f), dim=1),
                'offsets': self.offset_regression_head(f)}


def batches(n=3, scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        x = torch.randn(1, 1, H, W, generator=g) * scale
        spot = (torch.rand(1, H, W, generator=g) < 0.2).float()
        det = torch.stack([1 - spot, spot], dim=1)
        off = torch.randn(1, 2, H, W, generator=g) * 0.3 * scale
        out.append((x, {'detections': det, 'offsets': off}))
    return out


def run(model, data, epochs=1, opt=None, loss=None, **kw):
    cfg = {**train_mod.REFERENCE_TRAINING['optimizer'], **(opt or {})}
    lcfg = {**train_mod.REFERENCE_TRAINING['loss'], **(loss or {})}
    return train_mod.train(data, data, model=model, optimizer_config=cfg, loss_config=lcfg, epochs=epochs,
                           save_path_prefix=kw.pop('prefix', '.'), writer=None, write=False, device='cpu', **kw)


# ---- 4.1 head L2 penalty ---------------------------------------------------------------------

def test_penalty_value():
    from torch_spots.detection.loss import penalized_kernels
    m = Stub()
    ks = penalized_kernels(m)
    with torch.no_grad():
        for k in ks:
            k.zero_()
        ks[0].view(-1)[0] = 10.0                 # backbone kernel: 100
        ks[-1].view(-1)[0] = np.sqrt(150.0)      # classification-head kernel: 150
    assert kernel_l2_penalty(m, 1e-5).item() == pytest.approx(0.0025, rel=1e-5)


def test_only_kernels_are_penalized():
    from torch_spots.detection.loss import penalized_kernels
    m = Stub()
    base = kernel_l2_penalty(m, 1e-5).item()
    kernels = {id(k) for k in penalized_kernels(m)}
    n_checked = 0
    for name, p in m.named_parameters():
        if id(p) in kernels:
            continue
        with torch.no_grad():
            old = p.clone(); p.add_(5.0)
        assert kernel_l2_penalty(m, 1e-5).item() == base, name
        with torch.no_grad():
            p.copy_(old)
        n_checked += 1
    assert n_checked >= 5      # backbone bias, BN weight/bias, head bias, regression head ...


def test_backbone_and_head_kernels_are_penalized():
    from torch_spots.detection.loss import penalized_kernels
    m = Stub()
    for k in penalized_kernels(m):
        before = (k ** 2).sum().item()
        base = kernel_l2_penalty(m, 1e-5).item()
        with torch.no_grad():
            k.mul_(2.0)
        assert kernel_l2_penalty(m, 1e-5).item() - base == pytest.approx(1e-5 * 3 * before, rel=1e-4)
        with torch.no_grad():
            k.div_(2.0)
    assert len(penalized_kernels(m)) == 3        # 1 backbone conv + 2 head kernels in the stub


def test_real_spotnet_penalized_set():
    from torch_spots.detection.loss import penalized_kernels
    net = SpotNet(input_channels=1, receptive_field=13, n_skips=3, n_conv_filters=32,
                  n_dense_filters=128, num_classes=2, regression_feature_size=256, norm_method=None)
    ks = penalized_kernels(net)
    ids = {id(k) for k in ks}
    # 4 sub-networks x (2 conv blocks + 1 dense conv) in the backbone, plus 2 head kernels
    assert len(ks) == 4 * 3 + 2
    assert all(id(m.weight) not in ids for m in net.offset_regression_head.modules()
               if isinstance(m, torch.nn.Conv2d))
    head = net.classification_head.net
    assert id(head[0].weight) in ids and id(head[3].weight) in ids
    assert not any(id(p) in ids for n, p in net.named_parameters() if p.dim() == 1)   # no bias/BN


def test_optimizer_weight_decay_is_zero(monkeypatch):
    seen = {}
    real = torch.optim.SGD

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)
    monkeypatch.setattr(torch.optim, "SGD", spy)
    run(Stub(), batches())
    assert seen["weight_decay"] == 0 and seen["nesterov"] is True and seen["momentum"] == 0.9


def test_penalty_enters_validation_objective():
    # same weights, large penalty -> larger validation objective
    torch.manual_seed(0); a = Stub(); torch.manual_seed(0); b = Stub()
    _, sa = run(a, batches(), opt={"kernel_l2": 0.0, "lr": 0.0})
    _, sb = run(b, batches(), opt={"kernel_l2": 1.0, "lr": 0.0})
    assert sb["best_val_objective"] > sa["best_val_objective"] + 0.5 * kernel_l2_penalty(a, 1.0).item()


# ---- 4.2 learning rate, clipping, divergence -----------------------------------------------------

def test_lr_mid_training():
    assert train_mod.reference_lr(2, 2000, 0.01, 0.99, 1e-6) == pytest.approx(0.0097814, abs=1e-7)


def test_lr_last_step_of_default_run():
    # epoch 9, global step 849 * 10 - 1 = 8,489
    assert train_mod.reference_lr(9, 8489, 0.01, 0.99, 1e-6) == pytest.approx(0.0090583, abs=1e-7)


def test_lr_first_step():
    assert train_mod.reference_lr(0, 0, 0.01, 0.99, 1e-6) == 0.01


def test_lr_is_set_before_every_step(monkeypatch):
    lrs = []
    real = torch.optim.SGD

    class Spy(real):
        def step(self, *a, **k):
            lrs.append(self.param_groups[0]["lr"])
            return super().step(*a, **k)
    monkeypatch.setattr(torch.optim, "SGD", Spy)
    run(Stub(), batches(3), epochs=2)
    expected = [train_mod.reference_lr(e, e * 3 + i, 0.01, 0.99, 1e-6) for e in range(2) for i in range(3)]
    assert lrs == pytest.approx(expected)


def test_no_clipping_applies_raw_gradient():
    torch.manual_seed(0); m = Stub(); torch.manual_seed(0); ref = Stub()
    data = batches(1, scale=40.0)
    opt = {"kernel_l2": 0.0, "lr": 0.01}
    x, labels = data[0]
    # reference: raw gradient of the same objective, first SGD-Nesterov step = lr * (1 + momentum) * g
    from torch_spots.detection.loss import DotNetLosses
    ref.train()
    l = DotNetLosses(sigma=3.0, gamma=0.5, **train_mod.REFERENCE_TRAINING['loss'])(ref(x), labels)
    l.backward()
    norm = torch.sqrt(sum((p.grad ** 2).sum() for p in ref.parameters()))
    assert norm > 1.0
    expected = {n: p.detach() - 0.01 * 1.9 * p.grad for n, p in ref.named_parameters()}
    run(m, data, opt=opt, epochs=1)
    # parameters other than BN running stats: compare after exactly one step (validation does not update)
    for n, p in m.named_parameters():
        assert torch.allclose(p, expected[n], atol=1e-6), n


def test_nan_loss_raises():
    class Bad(Stub):
        def forward(self, x):
            out = super().forward(x)
            out['offsets'] = out['offsets'] * float('nan')
            return out
    with pytest.raises(FloatingPointError):
        run(Bad(), batches())


# ---- 4.3 defaults and step counts ----------------------------------------------------------------

def test_reference_defaults():
    r = train_mod.REFERENCE_TRAINING
    assert r['optimizer'] == {'lr': 1e-2, 'momentum': 0.9, 'step_decay': 1e-6, 'epoch_decay': 0.99,
                              'kernel_l2': 1e-5, 'grad_clip': None}
    assert r['loss'] == {'regression_weight': 1 / 6, 'classification_weight': 5 / 6, 'focal': False}
    assert r['epochs'] == 10 and r['batch_size'] == 1
    assert r['augmentation'] == {'flip': True, 'rotation_range': 180.0, 'zoom_range': (0.8, 1.2)}
    assert r['model'] == {'norm_method': None}
    assert train_mod.non_reference_settings(train_mod.DEFAULT_CONFIG) == {}


def _loader_len(n, bs):
    X = np.zeros((n, 128, 128), dtype=np.uint8)     # only its length is used
    return len(spot_dataloader(X, None, None, batch_size=bs, drop_last=True))


def test_default_step_count():
    steps_per_epoch = _loader_len(849, train_mod.REFERENCE_TRAINING['batch_size'])
    assert steps_per_epoch == 849
    assert steps_per_epoch * train_mod.REFERENCE_TRAINING['epochs'] == 8490


def test_partial_batch_dropped():
    assert _loader_len(849, 10) == 84


# ---- 5.1 best checkpoint --------------------------------------------------------------------------

def test_best_is_not_last(tmp_path, monkeypatch):
    seq = [0.50, 0.40, 0.45]
    snaps = []

    class Stub2(Stub):
        calls = 0

        def forward(self, x):
            out = super().forward(x)
            if not self.training:
                snaps.append({k: v.clone() for k, v in self.state_dict().items()})
            return out

    class FixedLoss(nn.Module):
        def __init__(self, **kw):
            super().__init__()
            self.n_val = 0
            self.model = None

        def forward(self, outputs, labels):
            if outputs['offsets'].requires_grad:                 # training batch
                return (outputs['offsets'] ** 2).mean()
            v = seq[len(snaps) - 1]
            return torch.tensor(v)
    monkeypatch.setattr(train_mod, "DotNetLosses", FixedLoss)
    model = Stub2()
    train_mod.train(batches(2), batches(1), model=model,
                    optimizer_config={**train_mod.REFERENCE_TRAINING['optimizer'], "kernel_l2": 0.0},
                    loss_config={}, epochs=3, save_path_prefix=str(tmp_path), writer=None, write=True,
                    device='cpu')
    best = torch.load(tmp_path / "saved_model_best_dict.pth")
    last = torch.load(tmp_path / "last_model_dict.pth")
    assert len(snaps) == 3
    for k in best:
        assert torch.equal(best[k], snaps[1][k]), k
        assert torch.equal(last[k], snaps[2][k]), k
    assert not all(torch.equal(best[k], last[k]) for k in best)


# ---- 5.2 constant images --------------------------------------------------------------------------

def test_blank_image_preprocesses_to_finite_constant():
    from torch_spots.detection.loader import preprocess_image
    out = preprocess_image(np.full((1, 128, 128), 500, dtype=np.uint16))
    assert np.isfinite(out).all() and (out == 0).all()
