"""Decoding evaluation module (spec: decoding-evaluation), on synthetic tables and tiny fixtures."""
import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from torch_spots.decoding import evaluate as ev

GENES = ['gene_alpha', 'gene_beta', 'gene_gamma']
CODEBOOK = pd.DataFrame({'Gene': GENES + ['Blank-01']})


def make_table(genes=0, blanks=0, unknown=0, background=0, rescue=0):
    """A spot table with the given counts of each kind of row."""
    names = [GENES[i % 3] for i in range(genes)] + ['Blank-01'] * blanks \
        + ['Unknown'] * unknown + ['Background'] * background
    sources = ['prediction'] * len(names)
    for i in range(rescue):  # turn the last `rescue` gene rows into error-rescue rows
        sources[genes - 1 - i] = 'error rescue'
    return pd.DataFrame({'spot_index': range(len(names)), 'predicted_name': names,
                         'source': sources, 'probability': 0.99})


# ---------------------------------------------------------------- fixtures for run/sweep

def md5_of(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


@pytest.fixture
def pins(tmp_path):
    files = {}
    for role, content in (('image', b'image'), ('codebook', b'codebook'), ('checkpoint', b'weights')):
        f = tmp_path / 'pinned' / role
        f.parent.mkdir(exist_ok=True)
        f.write_bytes(content)
        files[role] = (f, md5_of(f))
    return files


def fake_handoff(n=5, threshold=0.9):
    rng = np.random.default_rng(0)
    return {'locations': rng.integers(1, 20, (n, 2)), 'probabilities': rng.random((n, 8)),
            'settings': {'threshold': threshold, 'min_distance': 1, 'pool_size': 0}}


def fake_decode(probabilities, locations, seed):
    n = len(probabilities)
    return make_table(genes=n - 1 - seed % 2, blanks=1 + seed % 2, unknown=0)


def run(tmp_path, pins, name='baseline', **kwargs):
    kwargs.setdefault('decode_fn', fake_decode)
    kwargs.setdefault('codebook', CODEBOOK)
    return ev.run_configuration(name, root=tmp_path / 'out', pins=pins, **kwargs)


# ---------------------------------------------------------------- 6.1 protocol

def test_checksum_mismatch(tmp_path, pins):
    pins['image'][0].write_bytes(b'tampered')
    with pytest.raises(ev.EvaluationError) as err:
        run(tmp_path, pins, detect_fn=lambda s: fake_handoff())
    message = str(err.value.message)
    assert str(pins['image'][0]) in message
    assert pins['image'][1] in message and md5_of(pins['image'][0]) in message
    assert not (tmp_path / 'out').exists()  # no result written


def test_decoding_only_change_reads_the_stored_handoff(tmp_path, pins):
    stored = tmp_path / 'stored' / 'handoff.npz'
    md5 = ev.save_handoff(stored, fake_handoff())
    summary = run(tmp_path, pins, handoff_path=stored)
    assert summary['provenance']['checksums']['handoff'] == md5


def test_handoff_change_produces_and_stores_a_new_handoff(tmp_path, pins):
    calls = []

    def detect(settings):
        calls.append(settings)
        return fake_handoff(threshold=settings['threshold'])

    summary = run(tmp_path, pins, detect_fn=detect, handoff_settings={'threshold': 0.7})
    assert calls[0]['threshold'] == 0.7
    stored = tmp_path / 'out' / 'handoff' / 'thr0.7_pool0.npz'
    assert stored.is_file()
    assert summary['provenance']['checksums']['handoff'] == ev.read_md5_sidecar(stored)
    assert summary['provenance']['handoff_settings']['threshold'] == 0.7


def test_a_changed_stored_handoff_is_refused(tmp_path):
    stored = tmp_path / 'handoff.npz'
    ev.save_handoff(stored, fake_handoff())
    stored.write_bytes(stored.read_bytes() + b'x')
    with pytest.raises(ev.EvaluationError, match='checksum'):
        ev.load_handoff(stored)


def test_reporting_a_configuration(tmp_path, pins):
    stored = tmp_path / 'h.npz'
    ev.save_handoff(stored, fake_handoff())
    summary = run(tmp_path, pins, handoff_path=stored)
    assert sorted(summary['seeds']) == ['0', '1', '2']
    for seed in summary['seeds'].values():
        assert 'gene_calls' in seed and 'blank_rate' in seed
    assert summary['mean']['gene_calls'] == np.mean([s['gene_calls'] for s in summary['seeds'].values()])
    assert summary['range']['gene_calls'] == 1  # seeds give 3, 3, 4 gene calls... see fake_decode
    out = tmp_path / 'out' / 'baseline'
    assert (out / 'summary.json').is_file()
    assert all((out / f'seed{k}.csv').is_file() for k in (0, 1, 2))
    assert json.loads((out / 'summary.json').read_text())['mean'] == summary['mean']


def test_counting():
    table = make_table(genes=1000, blanks=20, unknown=300, background=500)
    result = ev.measure_table(table, GENES)
    assert result['gene_calls'] == 1000
    assert result['blank_calls'] == 20
    assert result['blank_rate'] == pytest.approx(20 / 1020)
    assert round(result['blank_rate'], 4) == 0.0196


def test_breakdown_present(tmp_path, pins):
    stored = tmp_path / 'h.npz'
    ev.save_handoff(stored, fake_handoff(n=9))
    summary = run(tmp_path, pins, handoff_path=stored,
                  decode_fn=lambda p, l, s: make_table(genes=6, blanks=1, unknown=1, background=1, rescue=2))
    seed = summary['seeds']['1']
    assert seed['candidates'] == 9
    assert seed['background_rows'] == 1 and seed['unknown_rows'] == 1
    assert seed['by_source']['prediction']['gene_calls'] == 4
    assert seed['by_source']['error rescue']['gene_calls'] == 2
    assert seed['by_source']['mixed rescue'] == {'gene_calls': 0, 'blank_calls': 0}
    assert seed['seconds'] >= 0 and seed['peak_memory_mb'] > 0


def test_reproducing_a_result(tmp_path, pins):
    stored = tmp_path / 'h.npz'
    ev.save_handoff(stored, fake_handoff())
    prov = run(tmp_path, pins, handoff_path=stored)['provenance']
    assert ev.missing_provenance(prov) == []
    for key in ('commit', 'handoff_settings', 'decoding_settings', 'seeds', 'decoding_device',
                'threads', 'torch', 'pyro'):
        assert prov[key] not in (None, '', {}, [])
    assert set(prov['checksums']) == {'image', 'codebook', 'checkpoint', 'handoff'}
    assert prov['threads'] == 4 and prov['seeds'] == [0, 1, 2]
    broken = copy.deepcopy(prov)
    del broken['checksums']['image']
    assert ev.missing_provenance(broken) == ['checksums.image']


# ---------------------------------------------------------------- 6.2 decision rules

def summary_for(gene, rate, margins=(15, 0.002), seconds=100.0, machine='A', device='cpu',
                memory=2000.0, label='release', threads=4, handoff='h1'):
    return {'protocol': ev.PROTOCOL,
            'mean': {'gene_calls': gene, 'blank_rate': rate, 'seconds': seconds, 'peak_memory_mb': memory},
            'range': {'gene_calls': margins[0], 'blank_rate': margins[1], 'seconds': 1.0,
                      'peak_memory_mb': 10.0},
            'margins': {'gene_calls': margins[0], 'blank_rate': margins[1]},
            'provenance': {'commit': 'abc', 'handoff_settings': {'threshold': 0.9},
                           'decoding_settings': {'num_iter': 500}, 'seeds': [0, 1, 2],
                           'checksums': {'image': 'a', 'codebook': 'b', 'checkpoint': 'c', 'handoff': handoff},
                           'decoding_device': device, 'detection_device': 'cuda:0', 'threads': threads,
                           'torch': '2', 'pyro': '1', 'machine': machine,
                           'checkpoint_label': label, 'pinned_checkpoint': label == 'release'}}


def test_zero_range():
    results = {s: {'gene_calls': 1200, 'blank_calls': 24, 'blank_rate': 24 / 1224} for s in (0, 1, 2)}
    _, _, margins = ev.aggregate_seeds(results)
    assert margins['gene_calls'] == 1
    assert margins['blank_rate'] == pytest.approx(1 / 1224)


def test_margin_is_the_baseline_range_when_not_zero():
    results = {0: {'gene_calls': 1190, 'blank_calls': 24, 'blank_rate': 0.019},
               1: {'gene_calls': 1200, 'blank_calls': 24, 'blank_rate': 0.020},
               2: {'gene_calls': 1205, 'blank_calls': 24, 'blank_rate': 0.021}}
    _, rng, margins = ev.aggregate_seeds(results)
    assert margins['gene_calls'] == 15 == rng['gene_calls']
    assert margins['blank_rate'] == pytest.approx(0.002)


def test_fewer_false_calls():
    verdict = ev.judge_accuracy(summary_for(1200, 0.020), summary_for(1190, 0.015))
    assert verdict['supported']


def test_fewer_calls_overall():
    verdict = ev.judge_accuracy(summary_for(1200, 0.020), summary_for(1000, 0.012))
    assert not verdict['supported']


def test_faster_and_non_inferior():
    verdict = ev.judge_qol(summary_for(1200, 0.020), summary_for(1195, 0.021, seconds=60.0), benefit='time')
    assert verdict['supported']


def test_benefit_not_measured():
    verdict = ev.judge_qol(summary_for(1200, 0.020), summary_for(1195, 0.021))
    assert not verdict['supported']
    assert any('no measurement' in r for r in verdict['reasons'])


def test_qol_refused_when_a_measure_worsens_beyond_its_margin():
    verdict = ev.judge_qol(summary_for(1200, 0.020), summary_for(1000, 0.020, seconds=10.0), benefit='time')
    assert not verdict['supported']


def test_time_claim_on_another_machine():
    with pytest.raises(ev.EvaluationError, match='same machine'):
        ev.judge_qol(summary_for(1200, 0.020, machine='A'),
                     summary_for(1200, 0.020, seconds=10.0, machine='B'), benefit='time')


def test_device_change_is_judged_under_the_quality_of_life_rule():
    verdict = ev.judge_accuracy(summary_for(1200, 0.020, device='cpu'),
                                summary_for(1195, 0.021, seconds=10.0, device='cuda:0'))
    assert verdict['rule'] == 'quality-of-life' and verdict['supported']
    assert 'quality-of-life' in verdict['note']


def test_results_without_complete_provenance_are_not_evidence():
    broken = summary_for(1190, 0.015)
    del broken['provenance']['checksums']['handoff']
    with pytest.raises(ev.EvaluationError, match='provenance'):
        ev.judge_accuracy(summary_for(1200, 0.020), broken)


def test_another_checkpoint_is_not_evidence():
    with pytest.raises(ev.EvaluationError, match='not evidence'):
        ev.judge_accuracy(summary_for(1200, 0.020), summary_for(1190, 0.015, label='round8_seed0'))


def test_a_different_handoff_needs_the_handoff_change_flag():
    base, change = summary_for(1200, 0.020), summary_for(1190, 0.015, handoff='h2')
    with pytest.raises(ev.EvaluationError, match='hand-off'):
        ev.judge_accuracy(base, change)
    assert ev.judge_accuracy(base, change, handoff_change=True)['supported']


def write_seed1(directory, table):
    directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(directory / 'seed1.csv', index=False)
    return directory


def test_rename(tmp_path):
    table = make_table(genes=6, blanks=1)
    base = write_seed1(tmp_path / 'base', table)
    change = write_seed1(tmp_path / 'change', table.rename(columns={'probability': 'posterior'}))
    verdict = ev.judge_identical(base, change, summary_for(1, 0.1), summary_for(1, 0.1),
                                 renames={'posterior': 'probability'})
    assert verdict['supported']
    # without the documented rename, the column is missing
    assert not ev.judge_identical(base, change, summary_for(1, 0.1), summary_for(1, 0.1))['supported']


def test_added_columns_do_not_count_as_a_difference(tmp_path):
    table = make_table(genes=6, blanks=1)
    base = write_seed1(tmp_path / 'base', table)
    change = write_seed1(tmp_path / 'change', table.assign(prob_RS0015=0.5))
    verdict = ev.judge_identical(base, change, summary_for(1, 0.1), summary_for(1, 0.1))
    assert verdict['supported'] and any('added columns' in r for r in verdict['reasons'])


def test_any_difference(tmp_path):
    table = make_table(genes=6, blanks=1)
    other = table.copy()
    other.loc[3, 'predicted_name'] = 'gene_beta'
    verdict = ev.judge_identical(write_seed1(tmp_path / 'base', table), write_seed1(tmp_path / 'change', other),
                                 summary_for(1, 0.1), summary_for(1, 0.1))
    assert not verdict['supported']
    assert any("'predicted_name' differs in 1 rows" in r for r in verdict['reasons'])


# ---------------------------------------------------------------- 6.3 sweep

def sweep_maps():
    """Detection maps with one peak per level, so the candidate count falls as the threshold rises."""
    maps = np.zeros((4, 2, 30, 30), dtype=np.float32)
    for i, value in enumerate([0.55, 0.75, 0.85, 0.92, 0.96, 0.985, 0.995]):
        maps[i % 4, 1, 3 + 3 * i, 5] = value
    return maps


def fake_decode_factory(codebook, settings):
    return lambda probabilities, locations, seed: make_table(genes=len(probabilities), blanks=seed)


def test_sweep_output(tmp_path, pins):
    sweep = ev.run_sweep('sweep_release', maps=sweep_maps(), root=tmp_path / 'out', pins=pins,
                         codebook=CODEBOOK, decode_fn_factory=fake_decode_factory)
    rows = sweep['thresholds']
    assert [r['threshold'] for r in rows] == list(ev.SWEEP_THRESHOLDS) and len(rows) == 8
    for row in rows:
        assert sorted(row['seeds']) == ['0', '1', '2']
        for seed in row['seeds'].values():
            assert {'gene_calls', 'blank_rate', 'candidates'} <= set(seed)
    assert [r['threshold'] for r in rows if r['is_baseline']] == [0.9]
    candidates = [r['mean']['candidates'] for r in rows]
    assert candidates == sorted(candidates, reverse=True) and candidates[0] > candidates[-1]
    out = tmp_path / 'out' / 'sweep_release'
    assert (out / 'sweep.json').is_file() and (out / 'thr0.9' / 'summary.json').is_file()
    assert (out / 'handoff' / 'thr0.9.npz.md5').is_file()
    assert (out / 'prob_maps.npz.md5').is_file()


def test_seed0_detection_checkpoint_is_kept_apart(tmp_path, pins):
    other = tmp_path / 'round8_seed0.pth'
    other.write_bytes(b'other weights')
    pins = {**pins, 'checkpoint': (other, md5_of(other))}
    root = tmp_path / 'out'
    sweep = ev.run_sweep('sweep_seed0', maps=sweep_maps(), root=root, pins=pins, codebook=CODEBOOK,
                         decode_fn_factory=fake_decode_factory, checkpoint_label='round8_seed0')
    assert sweep['checkpoint_label'] == 'round8_seed0' and not sweep['pinned_checkpoint']
    assert (root / 'other' / 'round8_seed0' / 'sweep_seed0' / 'sweep.json').is_file()
    assert not (root / 'sweep_seed0').exists()  # not in the pinned-checkpoint report
    summary = json.loads((root / 'other' / 'round8_seed0' / 'sweep_seed0' / 'thr0.9' / 'summary.json').read_text())
    assert summary['provenance']['checkpoint_label'] == 'round8_seed0'
    with pytest.raises(ev.EvaluationError, match='not evidence'):
        ev.judge_accuracy(summary_for(1200, 0.02), summary)


def test_a_file_that_is_not_a_current_format_handoff_is_refused(tmp_path):
    old_style = tmp_path / 'handoff.npz'
    np.savez(old_style, point_coord=np.zeros((2, 2), dtype=int), intensities=np.zeros((2, 4)))
    ev.write_md5_sidecar(old_style)
    with pytest.raises(ev.EvaluationError, match='not a stored hand-off'):
        ev.load_handoff(old_style)
