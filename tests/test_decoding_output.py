"""Output spot table and settings (spec: spot-decoding "Output spot table", "Locations follow the
spot index", "Genes-only output", "Decoding settings reported")."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from torch_spots.decoding import decode as decode_module
from torch_spots.decoding.decode import SpotDecoding
from tests.test_spot_decoding import make_codebook
from tests.test_decoding_model import make_probabilities

MERFISH_CODEBOOK = Path(__file__).resolve().parents[1] / 'SpotNetExampleData-v1_0' / 'MERFISH_codebook.csv'

# gene_alpha and gene_beta are disjoint 3-bit barcodes; Blank-01 is a blank control.
BARCODES = {'gene_alpha': '11100000', 'gene_beta': '00011100', 'gene_gamma': '10000011',
            'Blank-01': '01010101'}
NAMED_BITS = [f'RS{i:04d}' for i in range(8)]
SEVEN = ['spot_index', 'y', 'x', 'predicted_id', 'predicted_name', 'probability', 'source']


def decoder_with_blank():
    df = pd.DataFrame([[int(b) for b in bits] for bits in BARCODES.values()],
                      columns=NAMED_BITS, index=range(1, len(BARCODES) + 1))
    df.insert(0, 'Gene', list(BARCODES))
    return SpotDecoding(df, 4, 2)


def stub_fit(monkeypatch, class_probs):
    def fit(spots, barcodes, **kwargs):
        return {'class_probs': np.asarray(class_probs, dtype=float), 'params': {}}
    monkeypatch.setattr(decode_module, 'decoding_function', fit)


def bits(s):
    return [float(c) for c in s]


def test_columns(monkeypatch):
    dec = decoder_with_blank()
    stub_fit(monkeypatch, [[0.97, 0.01, 0.01, 0.0, 0.01]] * 3)
    locs = np.array([[5, 6], [7, 8], [9, 10]])
    table = dec.predict(np.full((3, 8), 0.5), locs)
    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == SEVEN + [f'prob_{n}' for n in NAMED_BITS]
    assert table['y'].tolist() == [5, 7, 9] and table['x'].tolist() == [6, 8, 10]


@pytest.mark.skipif(not MERFISH_CODEBOOK.exists(), reason='example data not present')
def test_columns_merfish_names(monkeypatch):
    cb = pd.read_csv(MERFISH_CODEBOOK, index_col=0)
    dec = SpotDecoding(cb, 10, 2)
    k = len(cb) + 1
    stub_fit(monkeypatch, np.eye(k)[[0, 1, 2]])
    table = dec.predict(np.full((3, 20), 0.5), np.zeros((3, 2), dtype=int))
    assert list(table.columns[:7]) == SEVEN
    assert table.columns[7] == 'prob_RS0015' and table.columns[-1] == 'prob_RS0707'
    assert len(table.columns) == 7 + 20


def test_no_locations_given(monkeypatch):
    dec = decoder_with_blank()
    stub_fit(monkeypatch, [[0.97, 0.01, 0.01, 0.0, 0.01]] * 2)
    table = dec.predict(np.full((2, 8), 0.5))
    assert 'y' not in table.columns and 'x' not in table.columns
    assert list(table.columns) == [c for c in SEVEN if c not in ('y', 'x')] + \
        [f'prob_{n}' for n in NAMED_BITS]


def test_locations_of_wrong_shape_are_rejected(monkeypatch):
    dec = decoder_with_blank()
    stub_fit(monkeypatch, [[0.97, 0.01, 0.01, 0.0, 0.01]] * 2)
    with pytest.raises(ValueError, match='locations'):
        dec.predict(np.full((2, 8), 0.5), np.zeros((3, 2), dtype=int))


def test_added_row_carries_its_spots_location_and_probabilities(monkeypatch):
    dec = decoder_with_blank()
    # Spot 1 is gene_alpha at 0.9; zeroing alpha's bits leaves beta minus one bit -> mixed rescue.
    probs = np.array([bits('00000000'), bits('11111000'), bits('00000000')])
    stub_fit(monkeypatch, [[0.01, 0.01, 0.01, 0.01, 0.96],
                           [0.90, 0.04, 0.02, 0.02, 0.02],
                           [0.01, 0.01, 0.01, 0.01, 0.96]])
    locs = np.array([[1, 2], [120, 45], [3, 4]])
    table = dec.predict(probs, locs, pred_prob_thresh=0.5, rescue_errors=False, rescue_mixed=True)
    assert len(table) == 4
    added = table.iloc[-1]
    original = table[(table.spot_index == 1) & (table.source == 'prediction')].iloc[0]
    assert added['source'] == 'mixed rescue' and added['predicted_name'] == 'gene_beta'
    assert added['probability'] == -1
    assert added['spot_index'] == 1 and added['y'] == 120 and added['x'] == 45
    prob_cols = [f'prob_{n}' for n in NAMED_BITS]
    assert added[prob_cols].tolist() == original[prob_cols].tolist() == bits('11111000')
    # rows for the input spots come first, in input order
    assert table['spot_index'].tolist()[:3] == [0, 1, 2]


def _mixed_posteriors():
    return [[0.97, 0.01, 0.01, 0.0, 0.01],   # gene_alpha
            [0.01, 0.01, 0.01, 0.96, 0.01],  # Blank-01
            [0.01, 0.01, 0.01, 0.01, 0.96],  # Background
            [0.30, 0.20, 0.20, 0.20, 0.10]]  # Unknown (error rescue off below)


def test_default_keeps_all_rows_and_switch_on_removes_the_non_genes(monkeypatch):
    dec = decoder_with_blank()
    probs = np.array([bits('11100000'), bits('01010101'), bits('00000000'), bits('00100000')])
    locs = np.arange(8).reshape(4, 2)
    stub_fit(monkeypatch, _mixed_posteriors())
    full = dec.predict(probs, locs, rescue_errors=False)
    assert set(full['predicted_name']) == {'gene_alpha', 'Blank-01', 'Background', 'Unknown'}
    genes = dec.predict(probs, locs, rescue_errors=False, genes_only=True)
    expected = full[~full.predicted_name.isin(['Background', 'Unknown', 'Blank-01'])]
    pd.testing.assert_frame_equal(genes, expected.reset_index(drop=True))
    assert genes['predicted_name'].tolist() == ['gene_alpha']


def test_genes_only_does_not_change_which_spots_are_rescued(monkeypatch):
    dec = decoder_with_blank()
    probs = np.array([bits('11100000'), bits('01010101'), bits('00000000'), bits('11000000')])
    stub_fit(monkeypatch, _mixed_posteriors())
    full = dec.predict(probs, rescue_errors=True)
    genes = dec.predict(probs, rescue_errors=True, genes_only=True)
    keep = ~full.predicted_name.isin(['Background', 'Unknown']) & ~full.predicted_name.str.contains('Blank')
    pd.testing.assert_frame_equal(genes, full[keep].reset_index(drop=True))
    assert (full['source'] == 'error rescue').any()


def test_settings_and_fit_available():
    cb = make_codebook(n_genes=6)
    dec = SpotDecoding(cb, 4, 2)
    dec.predict(make_probabilities(cb, 200), num_iter=5, seed=3, rescue_mixed=True, genes_only=True)
    run = dec.last_run
    assert run['settings'] == {'params_mode': '2*R*C', 'num_iter': 5, 'batch_size': 1000, 'seed': 3,
                               'pred_prob_thresh': 0.95, 'rescue_errors': True,
                               'rescue_mixed': True, 'genes_only': True}
    assert run['weights'].shape == (7,)            # 6 genes + Background
    assert run['sigma'].shape == (2, 8) and run['temperature'].shape == (2, 8)
    assert len(run['losses']) == 5
