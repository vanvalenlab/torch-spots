"""Assignment, Unknown threshold and rescues (spec: spot-decoding), on fixed posteriors and
probability vectors. No model is fitted here."""
import numpy as np
import pandas as pd

from torch_spots.decoding import decode as decode_module
from torch_spots.decoding.decode import SpotDecoding

ROUNDS, CHANNELS = 4, 2
# Eight bits, three 1-bits per gene. Gene names are long on purpose (see notes.md: names are
# stored in a fixed-width array, so names shorter than "Unknown" can be cut short).
BARCODES = {
    'gene_alpha': '11100000',
    'gene_beta':  '00011100',
    'gene_gamma': '10000011',
    'gene_delta': '01010101',
}
N_GENES = len(BARCODES)
BACKGROUND, UNKNOWN = N_GENES + 1, N_GENES + 2


def make_decoder():
    df = pd.DataFrame([[int(b) for b in bits] for bits in BARCODES.values()],
                      columns=[f'b{i}' for i in range(8)], index=range(1, N_GENES + 1))
    df.insert(0, 'Gene', list(BARCODES))
    return SpotDecoding(df, ROUNDS, CHANNELS)


def vec(bits):
    return np.array([float(b) for b in bits])


def posterior(*rows):
    """class_probs rows over (genes..., Background), each row padded to sum to 1."""
    return np.array(rows)


def assign(decoder, class_probs, thresh=0.95):
    d = decoder._decoding_output_to_dict({'class_probs': class_probs})
    return decoder._threshold_unknown_by_prob(d, UNKNOWN, pred_prob_thresh=thresh)


def test_clear_barcode():
    decoder = make_decoder()
    out = assign(decoder, posterior([0.003, 0.99, 0.004, 0.002, 0.001]))
    assert out['predicted_id'][0] == 2
    assert out['predicted_name'][0] == 'gene_beta'
    assert np.isclose(out['probability'][0], 0.99)
    assert out['source'][0] == 'prediction'


def test_low_confidence_spot():
    decoder = make_decoder()
    out = assign(decoder, posterior([0.80, 0.05, 0.05, 0.05, 0.05]))
    assert out['predicted_name'][0] == 'Unknown'
    assert out['predicted_id'][0] == UNKNOWN
    assert np.isclose(out['probability'][0], 0.80)


def _rescue(decoder, probabilities, names, ids):
    n = len(names)
    d = {'spot_index': np.arange(n), 'predicted_id': np.array(ids),
         'predicted_name': np.array(names, dtype='U32'), 'probability': np.full(n, 0.5),
         'source': np.repeat('prediction', n).astype('U25')}
    return decoder._rescue_errors(d, probabilities)


def test_one_bit_missing():
    decoder = make_decoder()
    probs = np.array([vec('11000000')])  # gene_alpha with its third bit missing
    out = _rescue(decoder, probs, ['Unknown'], [UNKNOWN])
    assert out['predicted_name'][0] == 'gene_alpha'
    assert out['predicted_id'][0] == 1
    assert out['source'][0] == 'error rescue'
    assert np.isclose(out['probability'][0], 0.5)  # unchanged


def test_single_lit_bit():
    decoder = make_decoder()
    probs = np.array([vec('00000100')])  # distance 2 from every gene, 1 from all-zero Background
    out = _rescue(decoder, probs, ['Unknown'], [UNKNOWN])
    assert out['predicted_name'][0] == 'Background'
    assert out['predicted_id'][0] == BACKGROUND
    assert out['source'][0] == 'error rescue'


def test_half_rounds_to_zero():
    decoder = make_decoder()
    # 0.5 rounds to 0, so this is 11000000 and is rescued to gene_alpha
    probs = np.array([[1, 1, 0.5, 0, 0, 0, 0, 0.0]])
    out = _rescue(decoder, probs, ['Unknown'], [UNKNOWN])
    assert out['predicted_name'][0] == 'gene_alpha'


def test_rescue_off(monkeypatch):
    decoder = make_decoder()

    def fake_fit(spots, barcodes, **kwargs):
        n, k = spots.shape[0], barcodes.shape[0]
        return {'class_probs': np.full((n, k), 1.0 / k), 'params': {}}  # all below 0.95

    monkeypatch.setattr(decode_module, 'decoding_function', fake_fit)
    probs = np.array([vec('11000000'), vec('00000100')])
    off = pd.DataFrame(decoder.predict(probs, rescue_errors=False))
    on = pd.DataFrame(decoder.predict(probs, rescue_errors=True))
    assert not (off['source'] == 'error rescue').any()
    assert (off['predicted_name'] == 'Unknown').all()
    assert (on['source'] == 'error rescue').all()


def test_two_overlapping_barcodes():
    decoder = make_decoder()
    # alpha (bits 0-2) plus two of beta's three bits (3, 4): zeroing alpha leaves beta minus a bit
    probs = np.array([vec('00000000'), vec('11111000')])
    d = {'spot_index': np.arange(2), 'predicted_id': np.array([BACKGROUND, 1]),
         'predicted_name': np.array(['Background', 'gene_alpha'], dtype='U32'),
         'probability': np.array([0.99, 0.9]),
         'source': np.repeat('prediction', 2).astype('U25')}
    out = decoder._rescue_mixed_spots(d, probs)
    assert out['spot_index'].tolist() == [0, 1, 1]
    assert out['predicted_name'].tolist() == ['Background', 'gene_alpha', 'gene_beta']
    assert out['predicted_id'].tolist() == [BACKGROUND, 1, 2]
    assert out['probability'].tolist() == [0.99, 0.9, -1]
    assert out['source'].tolist() == ['prediction', 'prediction', 'mixed rescue']


# ---- gene names of any length (the names array used to be fixed-width) ----

def make_short_name_decoder():
    """Gene names shorter than 'Unknown' (7 characters)."""
    df = pd.DataFrame([[int(b) for b in bits] for bits in ('11100000', '00011100')],
                      columns=[f'b{i}' for i in range(8)], index=[1, 2])
    df.insert(0, 'Gene', ['a', 'b'])
    return SpotDecoding(df, ROUNDS, CHANNELS)


def test_unknown_label_is_not_cut_short_when_gene_names_are_short():
    decoder = make_short_name_decoder()
    out = assign(decoder, posterior([0.5, 0.3, 0.2], [0.97, 0.02, 0.01]))
    assert out['predicted_name'].tolist() == ['Unknown', 'a']


def test_rescued_name_longer_than_every_predicted_name_is_kept():
    df = pd.DataFrame([[int(b) for b in bits] for bits in ('11100000', '00011100')],
                      columns=[f'b{i}' for i in range(8)], index=[1, 2])
    df.insert(0, 'Gene', ['a', 'a_much_longer_gene_name'])
    decoder = SpotDecoding(df, ROUNDS, CHANNELS)
    # spot 0 is confidently 'a'; spot 1 is Unknown and one bit from the long-named gene
    out = assign(decoder, posterior([0.97, 0.02, 0.01], [0.4, 0.4, 0.2]))
    probs = np.array([vec('11100000'), vec('00011000')])
    out = decoder._rescue_errors(out, probs)
    assert out['predicted_name'].tolist() == ['a', 'a_much_longer_gene_name']
