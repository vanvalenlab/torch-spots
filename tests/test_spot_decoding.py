"""Decoding module (spec: spot-decoding): codebook and input checks, settings, seed."""
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from torch_spots.decoding import decode as decode_module
from torch_spots.decoding.decode import SpotDecoding

ROOT = Path(__file__).resolve().parents[1]
MERFISH_CODEBOOK = ROOT / 'SpotNetExampleData-v1_0' / 'MERFISH_codebook.csv'


def make_codebook(n_genes=6, rounds=4, channels=2, seed=0):
    """A small valid codebook: 1-based index, 'Gene' first, binary bit columns."""
    rng = np.random.default_rng(seed)
    n_bits = rounds * channels
    codes = set()
    while len(codes) < n_genes:
        codes.add(tuple(rng.permutation([1, 1, 1, 0, 0, 0, 0, 0][:n_bits] + [0] * max(0, n_bits - 8))))
    df = pd.DataFrame(sorted(codes), columns=[f'b{i}' for i in range(n_bits)])
    df.insert(0, 'Gene', [f'g{i}' for i in range(n_genes)])
    df.index = range(1, n_genes + 1)
    return df


class Recorder:
    """Stands in for the model fit: records what it was given, returns a one-hot posterior."""

    def __init__(self):
        self.calls = []

    def __call__(self, spots, barcodes, **kwargs):
        self.calls.append({'spots': spots, 'barcodes': barcodes, **kwargs})
        n, k = spots.shape[0], barcodes.shape[0]
        class_probs = np.zeros((n, k))
        class_probs[np.arange(n), np.arange(n) % k] = 1.0
        return {'class_probs': class_probs, 'params': {}}


@pytest.fixture
def recorder(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(decode_module, 'decoding_function', rec)
    return rec


@pytest.mark.skipif(not MERFISH_CODEBOOK.exists(), reason='example data not present')
def test_valid_merfish_codebook():
    codebook = pd.read_csv(MERFISH_CODEBOOK, index_col=0)
    decoder = SpotDecoding(codebook, rounds=10, channels=2)
    # The pinned file has 265 rows (241 genes + 24 Blank-*); the change's spec text says 264.
    assert len(codebook) == 265
    assert codebook['Gene'].str.contains('Blank').sum() == 24
    assert codebook.shape[1] - 1 == 20
    assert decoder.rounds * decoder.channels == 20


def test_non_binary_value():
    cb = make_codebook()
    cb.iloc[0, 1] = 2
    with pytest.raises(ValueError, match='0 or 1'):
        SpotDecoding(cb, rounds=4, channels=2)


@pytest.mark.parametrize('name', ['Background', 'Unknown'])
def test_reserved_name_in_the_gene_column(name):
    cb = make_codebook()
    cb.loc[3, 'Gene'] = name
    with pytest.raises(ValueError, match='Background.*Unknown'):
        SpotDecoding(cb, rounds=4, channels=2)


def test_reserved_names_as_column_headers_are_not_the_check():
    # The old check looked at the column headers; a bit column may be called anything.
    cb = make_codebook().rename(columns={'b0': 'Background'})
    SpotDecoding(cb, rounds=4, channels=2)


def test_added_classes():
    cb = make_codebook(n_genes=6)
    decoder = SpotDecoding(cb, rounds=4, channels=2)
    aug = decoder.df_barcodes
    assert len(aug) == 8
    assert aug.loc[7, 'Gene'] == 'Background' and aug.loc[8, 'Gene'] == 'Unknown'
    assert (aug.loc[7].iloc[1:] == 0).all() and (aug.loc[8].iloc[1:] == -1).all()


def test_column_count_mismatch():
    decoder = SpotDecoding(make_codebook(), rounds=4, channels=2)
    with pytest.raises(ValueError) as err:
        decoder.predict(np.full((10, 6), 0.5))
    assert '8' in str(err.value) and '6' in str(err.value)


def test_value_out_of_range():
    decoder = SpotDecoding(make_codebook(), rounds=4, channels=2)
    probs = np.full((10, 8), 0.5)
    probs[2, 3] = 1.2
    with pytest.raises(ValueError, match=r'\[0, 1\]'):
        decoder.predict(probs)


def test_component_count(recorder):
    cb = make_codebook(n_genes=6)
    SpotDecoding(cb, rounds=4, channels=2).predict(np.full((10, 8), 0.5))
    # 6 codebook rows plus Background; Unknown is not a component.
    assert recorder.calls[0]['barcodes'].shape[0] == 7


@pytest.mark.skipif(not MERFISH_CODEBOOK.exists(), reason='example data not present')
def test_component_count_merfish(recorder):
    codebook = pd.read_csv(MERFISH_CODEBOOK, index_col=0)
    decoder = SpotDecoding(codebook, rounds=10, channels=2)
    decoder.predict(np.full((10, 20), 0.5))
    n = len(codebook)
    assert recorder.calls[0]['barcodes'].shape[0] == n + 1  # codebook rows plus Background
    assert decoder.df_barcodes.loc[n + 1, 'Gene'] == 'Background'
    assert decoder.df_barcodes.loc[n + 2, 'Gene'] == 'Unknown'


def test_invalid_mode():
    decoder = SpotDecoding(make_codebook(), rounds=4, channels=2, params_mode='3')
    with pytest.raises(ValueError, match=r"2\*R\*C"):
        decoder.predict(np.full((10, 8), 0.5))


def test_no_distribution_setting():
    cb = make_codebook()
    with pytest.raises(TypeError):
        SpotDecoding(cb, rounds=4, channels=2, distribution='Gaussian')
    from torch_spots.decoding import decoding_utils
    for fn in (decoding_utils.decoding_function, decoding_utils.model_constrained_tensor,
               decoding_utils.train):
        assert 'distribution' not in inspect.signature(fn).parameters


def test_default_seed(recorder):
    assert inspect.signature(SpotDecoding.predict).parameters['seed'].default == 1
    decoder = SpotDecoding(make_codebook(), rounds=4, channels=2)
    decoder.predict(np.full((10, 8), 0.5))
    decoder.predict(np.full((10, 8), 0.5), seed=0)
    assert [c['seed'] for c in recorder.calls] == [1, 0]
