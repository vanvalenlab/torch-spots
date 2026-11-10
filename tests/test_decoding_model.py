"""Mixture model and fitting settings (spec: spot-decoding), on a small synthetic codebook."""
import numpy as np
import pandas as pd
import pyro
import pytest
import torch
from pyro.distributions import constraints

from torch_spots.decoding import decoding_utils
from torch_spots.decoding.decode import SpotDecoding
from torch_spots.decoding.decoding_utils import instantiate_rb_params, reshape_torch_array
from tests.test_spot_decoding import make_codebook

ROUNDS, CHANNELS = 4, 2
N_BITS = ROUNDS * CHANNELS


def make_probabilities(codebook, n_spots, seed=0):
    """Noisy copies of the codebook rows, as detection probabilities in [0, 1]."""
    rng = np.random.default_rng(seed)
    bits = codebook.iloc[:, 1:].values
    rows = bits[rng.integers(0, len(bits), n_spots)].astype(float)
    return np.clip(rows * 0.8 + 0.1 + rng.normal(0, 0.05, rows.shape), 0, 1)


class StopFit(Exception):
    pass


def test_initial_location():
    pyro.clear_param_store()
    codes = torch.tensor([[0., 1., 0., 1.]])
    scaled_sigma, _ = instantiate_rb_params(2, 2, codes, '2*R*C')
    # sigma starts at 0.5: location b + (-1)^b * 0.3 * sigma
    assert torch.allclose(scaled_sigma, torch.tensor([[0.15, 0.85, 0.15, 0.85]]))


def test_default_mode_has_40_values_for_20_bits():
    pyro.clear_param_store()
    codes = torch.zeros(3, 20)
    instantiate_rb_params(10, 2, codes, '2*R*C')
    assert pyro.param('sigma').numel() == 40
    assert pyro.param('temperature').numel() == 40
    assert torch.allclose(pyro.param('sigma'), torch.full((2, 20), 0.5))
    assert torch.allclose(pyro.param('temperature'), torch.full((2, 20), 0.5))


def test_default_mode_is_2_r_c():
    assert SpotDecoding(make_codebook(), ROUNDS, CHANNELS).params_mode == '2*R*C'


def test_inherited_grouping_in_the_2_r_mode():
    """Records what the baseline does (rounds=10, channels=2, mode '2*R').

    A model column is shared with the column 10 places away, but the model's columns are in a
    different order from the codebook's bit columns (a known limitation, out of scope here).
    In codebook bit-column terms the pairs are (0, 5), (1, 6), ..., not (0, 10), (1, 11), ...
    """
    R, C = 10, 2
    pyro.clear_param_store()
    values = torch.arange(2 * R).float().reshape(2, R) / 40 + 0.01
    pyro.param('sigma', values.clone(), constraint=constraints.unit_interval)
    pyro.param('temperature', values.clone(), constraint=constraints.unit_interval)
    codes = reshape_torch_array(torch.zeros(1, C, R))  # same path the model's data takes
    _, aug_temperature = instantiate_rb_params(R, C, codes, '2*R')
    by_model_column = aug_temperature[0].detach().numpy()
    assert np.allclose(by_model_column[:R], by_model_column[R:])  # m and m + R share a value
    # Map back to codebook bit columns: bit j = c * R + r sits at model column m = r * C + c.
    by_bit = np.zeros(R * C)
    for r in range(R):
        for c in range(C):
            by_bit[c * R + r] = by_model_column[r * C + c]
    assert by_bit[0] == by_bit[5] and by_bit[1] == by_bit[6]
    assert by_bit[0] != by_bit[10]


def _capture_fit(monkeypatch):
    """Record the optimizer settings and the fitting call, then stop before fitting."""
    seen = {}
    real_adam = decoding_utils.Adam

    def adam(config, *args, **kwargs):
        seen['adam'] = config
        return real_adam(config, *args, **kwargs)

    def train(svi, num_iter, data, codes, c, r, batch_size, params_mode):
        seen.update(num_iter=num_iter, batch_size=batch_size, n_spots=data.shape[0],
                    params_mode=params_mode)
        raise StopFit

    monkeypatch.setattr(decoding_utils, 'Adam', adam)
    monkeypatch.setattr(decoding_utils, 'train', train)
    return seen


def test_defaults(monkeypatch):
    seen = _capture_fit(monkeypatch)
    cb = make_codebook()
    with pytest.raises(StopFit):
        SpotDecoding(cb, ROUNDS, CHANNELS).predict(make_probabilities(cb, 3000))
    assert seen['adam'] == {'lr': 0.085, 'betas': [0.85, 0.99]}
    assert seen['num_iter'] == 500
    assert seen['batch_size'] == 1000  # min(3000, 1000)
    assert seen['params_mode'] == '2*R*C'


def test_fewer_spots_than_the_minibatch(monkeypatch):
    seen = _capture_fit(monkeypatch)
    cb = make_codebook()
    with pytest.raises(StopFit):
        SpotDecoding(cb, ROUNDS, CHANNELS).predict(make_probabilities(cb, 600))
    assert seen['n_spots'] == 600 and seen['batch_size'] == 600


def test_repeat_run_gives_identical_tables_on_cpu():
    cb = make_codebook()
    probs = make_probabilities(cb, 300)
    decoder = SpotDecoding(cb, ROUNDS, CHANNELS)
    first = pd.DataFrame(decoder.predict(probs, num_iter=20, seed=0))
    second = pd.DataFrame(decoder.predict(probs, num_iter=20, seed=0))
    pd.testing.assert_frame_equal(first, second)
