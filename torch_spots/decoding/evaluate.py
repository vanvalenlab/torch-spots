"""Decoding evaluation (decoding-evaluation capability).

Scores a decoding configuration on the pinned MERFISH example, from a stored detection hand-off,
under the seed protocol (seeds 0, 1, 2), and judges changes against a recorded baseline.

Three commands:

``run``      one configuration, seeds 0-2  ->  ``<root>/<name>/summary.json`` + ``seed<k>.csv``
``sweep``    candidate-threshold grid (detection runs once)  ->  ``<root>/<name>/sweep.json``
``compare``  apply the accuracy, quality-of-life or output-identical rule to two recorded runs

Results for the pinned release checkpoint are kept under ``data/eval/decoding/``. Results for any
other checkpoint go to ``data/eval/decoding/other/<label>/`` and are never used as evidence.

Shared-server rules: thread counts are capped at import (4). Launch long runs with ``nice -n 10``
in a named tmux session, and pick the detection GPU with ``CUDA_VISIBLE_DEVICES``.
"""
import os

# Cap threads before numpy / torch are imported (shared server).
THREADS = 4
for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_var] = str(THREADS)

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(THREADS)

ROOT = Path(__file__).resolve().parents[2]
DECODING_ROOT = ROOT / 'data' / 'eval' / 'decoding'

#: Identifies the scoring protocol. Compare refuses to mix values.
PROTOCOL = 'decoding-eval/1'

#: decoding-evaluation, "Pinned decoding example" and "Pinned detection checkpoint".
PINNED = {
    'image': (ROOT / 'SpotNetExampleData-v1_0' / 'MERFISH_example.npz', '2cd7ce177b503fd0873125784097622b'),
    'codebook': (ROOT / 'SpotNetExampleData-v1_0' / 'MERFISH_codebook.csv', '56f107ed562b73dc0037174c18f6f4dd'),
    'checkpoint': (ROOT / 'data' / 'release' / 'saved_model_best_dict.pth', 'f4713d8a7c552e68eecf63a880259ef1'),
}
PINNED_CHECKPOINT_LABEL = 'release'

SEEDS = (0, 1, 2)
SWEEP_THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.993)
BASELINE_THRESHOLD = 0.9
SOURCES = ('prediction', 'error rescue', 'mixed rescue')

DEFAULT_HANDOFF_SETTINGS = {'threshold': 0.9, 'min_distance': 1, 'pool_size': 0}
DEFAULT_DECODING_SETTINGS = {'rounds': 10, 'channels': 2, 'params_mode': '2*R*C', 'num_iter': 500,
                             'batch_size': 1000, 'pred_prob_thresh': 0.95, 'rescue_errors': True,
                             'rescue_mixed': False}

#: A result missing any of these cannot be used for a decision.
REQUIRED_PROVENANCE = ('commit', 'handoff_settings', 'decoding_settings', 'seeds', 'checksums',
                       'decoding_device', 'detection_device', 'threads', 'torch', 'pyro', 'machine')
REQUIRED_CHECKSUMS = ('image', 'codebook', 'checkpoint', 'handoff')


class EvaluationError(click.ClickException):
    """A rule of the protocol stops the evaluation."""


# --------------------------------------------------------------------------------------
# Checksums and stored hand-offs
# --------------------------------------------------------------------------------------

def md5_file(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_checksums(pins=None):
    """Check the pinned image, codebook and checkpoint. Returns ``{role: md5}``.

    Raises EvaluationError naming the file and both checksums when one differs, so no result is
    reported.
    """
    checksums = {}
    for role, (path, expected) in (pins or PINNED).items():
        path = Path(path)
        if not path.is_file():
            raise EvaluationError(f'Pinned {role} file is missing: {path}')
        got = md5_file(path)
        if got != expected:
            raise EvaluationError(f'Checksum mismatch for the pinned {role} {path}: '
                                  f'expected {expected}, got {got}. No result reported.')
        checksums[role] = got
    return checksums


def write_md5_sidecar(path):
    digest = md5_file(path)
    Path(str(path) + '.md5').write_text(f'{digest}  {Path(path).name}\n')
    return digest


def read_md5_sidecar(path):
    sidecar = Path(str(path) + '.md5')
    if not sidecar.is_file():
        raise EvaluationError(f'{path} has no MD5 sidecar ({sidecar.name}); a stored hand-off needs one.')
    expected = sidecar.read_text().split()[0]
    got = md5_file(path)
    if got != expected:
        raise EvaluationError(f'Stored file {path} does not match its checksum: '
                              f'expected {expected}, got {got}.')
    return got


def save_handoff(path, handoff):
    """Store a hand-off (``get_spot_probabilities`` output) with an MD5 sidecar. Returns the MD5."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, locations=handoff['locations'], probabilities=handoff['probabilities'],
             settings=json.dumps(handoff['settings']))
    return write_md5_sidecar(path)


def load_handoff(path):
    """Read a stored hand-off after checking its sidecar. Returns ``(handoff, md5)``."""
    path = Path(path)
    digest = read_md5_sidecar(path)
    data = np.load(path)
    missing = [k for k in ('locations', 'probabilities', 'settings') if k not in data]
    if missing:
        raise EvaluationError(f'{path} is not a stored hand-off: missing {missing}.')
    return {'locations': data['locations'], 'probabilities': data['probabilities'],
            'settings': json.loads(str(data['settings']))}, digest


# --------------------------------------------------------------------------------------
# Measures
# --------------------------------------------------------------------------------------

def _is_blank(names):
    return names.astype(str).str.contains('Blank', regex=False)


def measure_table(table, gene_names):
    """Counts for one full spot table (``genes_only`` off): every row, rescue rows included.

    A gene call is a row named for a codebook gene (blank controls excluded), a blank call a row
    whose name contains ``Blank``. Blank rate = blank / (gene + blank).
    """
    names = table['predicted_name'].astype(str)
    blank = _is_blank(names)
    gene = names.isin(set(gene_names)) & ~blank
    total = int(gene.sum() + blank.sum())
    out = {
        'gene_calls': int(gene.sum()),
        'blank_calls': int(blank.sum()),
        'blank_rate': (float(blank.sum()) / total) if total else None,
        'background_rows': int((names == 'Background').sum()),
        'unknown_rows': int((names == 'Unknown').sum()),
        'rows': int(len(table)),
        'by_source': {},
    }
    for source in SOURCES:
        from_source = table['source'].astype(str) == source
        out['by_source'][source] = {'gene_calls': int((gene & from_source).sum()),
                                    'blank_calls': int((blank & from_source).sum())}
    return out


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _range(values):
    values = [v for v in values if v is not None]
    return float(max(values) - min(values)) if values else None


def aggregate_seeds(seed_results):
    """Mean and range over seeds of each measure, and the noise margins.

    The margin of a measure is the baseline's range over the seeds. If that range is 0, it is the
    effect of one call: 1 for the gene-call count, and 1 / (mean gene + mean blank calls) for the
    blank rate.
    """
    keys = ('gene_calls', 'blank_calls', 'blank_rate', 'candidates', 'seconds', 'peak_memory_mb')
    mean = {k: _mean([r.get(k) for r in seed_results.values()]) for k in keys}
    rng = {k: _range([r.get(k) for r in seed_results.values()]) for k in keys}
    margins = {'gene_calls': rng['gene_calls'], 'blank_rate': rng['blank_rate']}
    if not margins['gene_calls']:
        margins['gene_calls'] = 1.0
    if not margins['blank_rate']:
        denom = (mean['gene_calls'] or 0) + (mean['blank_calls'] or 0)
        margins['blank_rate'] = 1.0 / denom if denom else None
    return mean, rng, margins


# --------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------

def git_state():
    def run(*args):
        return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True).stdout.strip()
    return {'commit': run('rev-parse', 'HEAD') or None,
            'dirty': bool(run('status', '--short', '--untracked-files=no', '--', 'torch_spots'))}


def peak_memory_mb():
    """Peak resident memory of this process since the last reset, in MB (Linux)."""
    try:
        for line in open('/proc/self/status'):
            if line.startswith('VmHWM:'):
                return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    import resource
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def reset_peak_memory():
    try:
        with open('/proc/self/clear_refs', 'w') as f:
            f.write('5')
    except OSError:
        pass


def build_provenance(*, checksums, handoff_md5, handoff_settings, decoding_settings, seeds,
                     detection_device, checkpoint_label=PINNED_CHECKPOINT_LABEL):
    try:
        import pyro
        pyro_version = pyro.__version__
    except ImportError:  # pragma: no cover
        pyro_version = None
    git = git_state()
    return {
        'commit': git['commit'], 'dirty': git['dirty'],
        'handoff_settings': dict(handoff_settings), 'decoding_settings': dict(decoding_settings),
        'seeds': list(seeds),
        'checksums': {**checksums, 'handoff': handoff_md5},
        'checkpoint_label': checkpoint_label,
        'pinned_checkpoint': checkpoint_label == PINNED_CHECKPOINT_LABEL,
        'decoding_device': 'cpu', 'detection_device': detection_device,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'threads': THREADS, 'torch': torch.__version__, 'pyro': pyro_version,
        'machine': platform.node(),
    }


def missing_provenance(provenance):
    """Names of required provenance entries that are absent (empty list = complete)."""
    missing = [k for k in REQUIRED_PROVENANCE if provenance.get(k) in (None, '', {}, [])]
    sums = provenance.get('checksums') or {}
    missing += [f'checksums.{k}' for k in REQUIRED_CHECKSUMS if not sums.get(k)]
    return missing


# --------------------------------------------------------------------------------------
# Running a configuration
# --------------------------------------------------------------------------------------

def make_decode_fn(codebook, settings):
    """Default decode function: the real ``SpotDecoding`` on the CPU."""
    from torch_spots.decoding.decode import SpotDecoding
    decoder = SpotDecoding(codebook, rounds=settings['rounds'], channels=settings['channels'],
                           params_mode=settings['params_mode'])

    def decode(probabilities, locations, seed):
        return decoder.predict(probabilities, locations, num_iter=settings['num_iter'],
                               batch_size=settings['batch_size'], seed=seed,
                               pred_prob_thresh=settings['pred_prob_thresh'],
                               rescue_errors=settings['rescue_errors'],
                               rescue_mixed=settings['rescue_mixed'], genes_only=False)
    return decode


def evaluate_configuration(handoff, codebook, decode_fn, seeds=SEEDS):
    """Decode one stored hand-off at every seed. Returns ``({seed: result}, {seed: table})``."""
    gene_names = codebook['Gene'].tolist()
    results, tables = {}, {}
    for seed in seeds:
        reset_peak_memory()
        t0 = time.perf_counter()
        table = decode_fn(handoff['probabilities'], handoff['locations'], seed)
        seconds = time.perf_counter() - t0
        result = measure_table(table, gene_names)
        result.update({'candidates': int(len(handoff['locations'])), 'seconds': round(seconds, 2),
                       'peak_memory_mb': peak_memory_mb()})
        results[seed], tables[seed] = result, table
    return results, tables


def write_run(out_dir, name, results, tables, provenance):
    """Write ``summary.json`` and one ``seed<k>.csv`` per seed. Returns the summary dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mean, rng, margins = aggregate_seeds(results)
    summary = {'protocol': PROTOCOL, 'name': name,
               'seeds': {str(k): v for k, v in results.items()},
               'mean': mean, 'range': rng, 'margins': margins, 'provenance': provenance}
    for seed, table in tables.items():
        table.to_csv(out_dir / f'seed{seed}.csv', index=False)
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def eval_root(checkpoint_label, root=None):
    """Pinned-checkpoint results live at the root; any other checkpoint's go to ``other/<label>``."""
    root = Path(root or DECODING_ROOT)
    return root if checkpoint_label == PINNED_CHECKPOINT_LABEL else root / 'other' / checkpoint_label


def run_configuration(name, *, handoff_path=None, detect_fn=None, handoff_settings=None,
                      decoding_settings=None, codebook=None, decode_fn=None, root=None,
                      checkpoint_label=PINNED_CHECKPOINT_LABEL, detection_device='cuda:0',
                      pins=None, seeds=SEEDS):
    """One configuration × seeds. Reads a stored hand-off, or produces and stores one.

    handoff_path: an existing stored hand-off (with MD5 sidecar) to decode.
    detect_fn:    otherwise, ``detect_fn(handoff_settings) -> hand-off dict``; the result is stored
                  under ``<root>/handoff/`` and its checksum recorded.
    """
    checksums = verify_checksums(pins)  # stops here, before any result, on a mismatch
    decoding_settings = {**DEFAULT_DECODING_SETTINGS, **(decoding_settings or {})}
    out_root = eval_root(checkpoint_label, root)

    if handoff_path is not None:
        handoff, handoff_md5 = load_handoff(handoff_path)
    else:
        settings = {**DEFAULT_HANDOFF_SETTINGS, **(handoff_settings or {})}
        handoff = detect_fn(settings)
        stored = out_root / 'handoff' / f"thr{settings['threshold']}_pool{settings['pool_size']}.npz"
        handoff_md5 = save_handoff(stored, handoff)

    if codebook is None:
        codebook = pd.read_csv((pins or PINNED)['codebook'][0], index_col=0)
    decode_fn = decode_fn or make_decode_fn(codebook, decoding_settings)

    results, tables = evaluate_configuration(handoff, codebook, decode_fn, seeds)
    provenance = build_provenance(checksums=checksums, handoff_md5=handoff_md5,
                                  handoff_settings=handoff['settings'],
                                  decoding_settings=decoding_settings, seeds=seeds,
                                  detection_device=detection_device, checkpoint_label=checkpoint_label)
    return write_run(out_root / name, name, results, tables, provenance)


def detect_maps(checkpoint, device):
    """Run detection once on the pinned image. Returns the detection output maps ``[N, 2, H, W]``."""
    from torch_spots.detection.inference import SpotDetection
    image = np.load(PINNED['image'][0], allow_pickle=True)['spots_image']
    image = np.moveaxis(image, -1, 0)
    detector = SpotDetection(model_path=str(checkpoint), device=device)
    return detector.predict_transforms(image)['detections']


def handoff_from_maps(maps, threshold, min_distance=1, pool_size=0):
    """The hand-off for one candidate threshold, from stored detection maps."""
    from torch_spots.detection.inference import SpotDetection
    # get_spot_probabilities only reads the maps, so no model needs to be loaded for it.
    return SpotDetection.get_spot_probabilities(None, {'detections': maps}, threshold=threshold,
                                                min_distance=min_distance, pool_size=pool_size)


def run_sweep(name, *, maps=None, detect_maps_fn=None, thresholds=SWEEP_THRESHOLDS,
              decoding_settings=None, codebook=None, decode_fn_factory=None, root=None,
              checkpoint_label=PINNED_CHECKPOINT_LABEL, detection_device='cuda:0', pins=None,
              seeds=SEEDS, min_distance=1, pool_size=0):
    """Candidate-threshold sweep. Detection runs once; each threshold's hand-off is derived from
    the stored probability maps, then decoded at every seed."""
    checksums = verify_checksums(pins)
    out_root = eval_root(checkpoint_label, root)
    sweep_dir = out_root / name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    decoding_settings = {**DEFAULT_DECODING_SETTINGS, **(decoding_settings or {})}
    if codebook is None:
        codebook = pd.read_csv((pins or PINNED)['codebook'][0], index_col=0)

    if maps is None:
        maps = detect_maps_fn()
    np.savez(sweep_dir / 'prob_maps.npz', detections=maps)
    maps_md5 = write_md5_sidecar(sweep_dir / 'prob_maps.npz')

    rows = []
    for threshold in thresholds:
        handoff = handoff_from_maps(maps, threshold, min_distance, pool_size)
        handoff_md5 = save_handoff(sweep_dir / 'handoff' / f'thr{threshold}.npz', handoff)
        decode_fn = (decode_fn_factory or make_decode_fn)(codebook, decoding_settings)
        results, tables = evaluate_configuration(handoff, codebook, decode_fn, seeds)
        provenance = build_provenance(checksums=checksums, handoff_md5=handoff_md5,
                                      handoff_settings=handoff['settings'],
                                      decoding_settings=decoding_settings, seeds=seeds,
                                      detection_device=detection_device,
                                      checkpoint_label=checkpoint_label)
        summary = write_run(sweep_dir / f'thr{threshold}', f'{name}/thr{threshold}', results, tables,
                            provenance)
        rows.append({'threshold': threshold,
                     'is_baseline': bool(np.isclose(threshold, BASELINE_THRESHOLD)),
                     'seeds': {k: {m: v[m] for m in ('candidates', 'gene_calls', 'blank_calls', 'blank_rate')}
                               for k, v in summary['seeds'].items()},
                     'mean': summary['mean'], 'range': summary['range'],
                     'by_source': {k: v['by_source'] for k, v in summary['seeds'].items()}})
    sweep = {'protocol': PROTOCOL, 'name': name, 'checkpoint_label': checkpoint_label,
             'pinned_checkpoint': checkpoint_label == PINNED_CHECKPOINT_LABEL,
             'baseline_threshold': BASELINE_THRESHOLD, 'prob_maps_md5': maps_md5,
             'thresholds': rows}
    (sweep_dir / 'sweep.json').write_text(json.dumps(sweep, indent=2) + '\n')
    return sweep


# --------------------------------------------------------------------------------------
# Judging changes
# --------------------------------------------------------------------------------------

def _usable(summary, role):
    """A result can support a decision only with complete provenance and the pinned checkpoint."""
    prov = summary.get('provenance', {})
    missing = missing_provenance(prov)
    if missing:
        raise EvaluationError(f'The {role} result is missing provenance ({", ".join(missing)}) and '
                              'cannot be used for a decision.')
    if not prov.get('pinned_checkpoint'):
        raise EvaluationError(f'The {role} result used checkpoint "{prov.get("checkpoint_label")}", not '
                              'the pinned release checkpoint; it is reported apart and is not evidence.')


def _check_same_handoff(base, change, handoff_change):
    same = base['provenance']['checksums']['handoff'] == change['provenance']['checksums']['handoff']
    if not same and not handoff_change:
        raise EvaluationError('The two results used different stored hand-offs. A decoding-only change '
                              'must be decoded from the baseline hand-off; for a hand-off change, '
                              'pass --handoff-change.')
    return same


def _moves(base, change, margins):
    """How far each measure moved, in the direction 'better' (positive) or 'worse' (negative)."""
    return {'gene_calls': change['mean']['gene_calls'] - base['mean']['gene_calls'],
            'blank_rate': base['mean']['blank_rate'] - change['mean']['blank_rate'],
            'margins': margins}


def judge_accuracy(base, change, handoff_change=False):
    """Accuracy rule: one measure improves by more than its margin and the other does not worsen
    by more than its margin. Lower mean blank rate and higher mean gene-call count are better."""
    _usable(base, 'baseline'), _usable(change, 'change')
    _check_same_handoff(base, change, handoff_change)
    if base['provenance']['decoding_device'] != change['provenance']['decoding_device']:
        verdict = judge_qol(base, change, benefit='time', handoff_change=handoff_change)
        verdict['note'] = ('The decoding device differs from the baseline, so the change is judged under '
                           'the quality-of-life rule.')
        return verdict
    m = _moves(base, change, base['margins'])
    gm, bm = base['margins']['gene_calls'], base['margins']['blank_rate']
    improved = m['gene_calls'] > gm or m['blank_rate'] > bm
    worsened = m['gene_calls'] < -gm or m['blank_rate'] < -bm
    reasons = [f"gene calls {m['gene_calls']:+.1f} (margin {gm:g}), blank rate improvement "
               f"{m['blank_rate']:+.5f} (margin {bm:.5f})"]
    if worsened:
        reasons.append('a measure worsened by more than its margin')
    elif not improved:
        reasons.append('no measure improved by more than its margin')
    return {'rule': 'accuracy', 'supported': bool(improved and not worsened), 'reasons': reasons}


def judge_qol(base, change, benefit=None, benefit_note=None, handoff_change=False):
    """Quality-of-life rule: neither measure worsens by more than its margin, and the benefit is
    measured and reported. Time and memory need the same machine and thread limits as the baseline."""
    _usable(base, 'baseline'), _usable(change, 'change')
    _check_same_handoff(base, change, handoff_change)
    m = _moves(base, change, base['margins'])
    gm, bm = base['margins']['gene_calls'], base['margins']['blank_rate']
    non_inferior = m['gene_calls'] >= -gm and m['blank_rate'] >= -bm
    reasons = [f"gene calls {m['gene_calls']:+.1f} (margin {gm:g}), blank rate improvement "
               f"{m['blank_rate']:+.5f} (margin {bm:.5f})"]
    if not non_inferior:
        reasons.append('a measure worsened by more than its margin')

    benefit_measured = False
    if benefit in ('time', 'memory'):
        bp, cp = base['provenance'], change['provenance']
        if bp['machine'] != cp['machine'] or bp['threads'] != cp['threads']:
            raise EvaluationError(
                f"The baseline was timed on {bp['machine']} with {bp['threads']} threads and the change on "
                f"{cp['machine']} with {cp['threads']} threads. The claim is refused until the baseline is "
                'also timed on the same machine with the same thread limits.')
        key = 'seconds' if benefit == 'time' else 'peak_memory_mb'
        saved = base['mean'][key] - change['mean'][key]
        noise = base['range'][key] or 0.0
        benefit_measured = saved > noise and saved > 0
        pct = 100.0 * saved / base['mean'][key] if base['mean'][key] else 0.0
        reasons.append(f'{benefit}: {base["mean"][key]:.1f} -> {change["mean"][key]:.1f} '
                       f'({pct:+.0f}% shorter/smaller; baseline range {noise:.1f})')
        if not benefit_measured:
            reasons.append(f'the {benefit} benefit is not larger than the baseline\'s own seed-to-seed range')
    elif benefit == 'other' and benefit_note:
        benefit_measured = True
        reasons.append(f'reported benefit: {benefit_note}')
    else:
        reasons.append('the benefit has no measurement')
    return {'rule': 'quality-of-life', 'supported': bool(non_inferior and benefit_measured),
            'reasons': reasons}


def judge_identical(base_dir, change_dir, base, change, renames=None):
    """Output-identical rule: at seed 1, on the baseline hand-off, every row and value equals the
    baseline's, after the change's documented column renames. Extra columns in the change are
    reported as added and do not count as a difference."""
    _usable(base, 'baseline'), _usable(change, 'change')
    _check_same_handoff(base, change, handoff_change=False)  # must be the baseline hand-off
    a = pd.read_csv(Path(base_dir) / 'seed1.csv', float_precision='round_trip')
    b = pd.read_csv(Path(change_dir) / 'seed1.csv', float_precision='round_trip').rename(columns=renames or {})
    reasons, identical = [], True
    missing = [c for c in a.columns if c not in b.columns]
    added = [c for c in b.columns if c not in a.columns]
    if missing:
        identical = False
        reasons.append(f'columns missing from the change: {missing}')
    if len(a) != len(b):
        identical = False
        reasons.append(f'row count differs: {len(a)} vs {len(b)}')
    if identical:
        for col in a.columns:
            if not np.array_equal(a[col].astype(str).to_numpy(), b[col].astype(str).to_numpy()):
                identical = False
                n = int((a[col].astype(str).to_numpy() != b[col].astype(str).to_numpy()).sum())
                reasons.append(f'column {col!r} differs in {n} rows')
    if added:
        reasons.append(f'added columns (not compared): {added}')
    if identical:
        reasons.insert(0, f'all {len(a)} rows and {len(a.columns)} columns equal')
    else:
        reasons.append('not output-identical: it needs accuracy or quality-of-life evidence')
    return {'rule': 'output-identical', 'supported': bool(identical), 'reasons': reasons}


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------

def _checkpoint_options(checkpoint, label):
    pinned_path = PINNED['checkpoint'][0]
    if checkpoint is None or Path(checkpoint).resolve() == Path(pinned_path).resolve():
        return pinned_path, PINNED_CHECKPOINT_LABEL
    if not label:
        raise click.UsageError('A checkpoint other than the pinned release one needs --checkpoint-label.')
    return Path(checkpoint), label


def _pins_for(checkpoint, label):
    """Pins to verify: the default set for the pinned checkpoint, else the image and codebook plus
    the given checkpoint's own checksum (recorded, since there is nothing to pin it against)."""
    if label == PINNED_CHECKPOINT_LABEL:
        return None
    return {**{k: v for k, v in PINNED.items() if k != 'checkpoint'},
            'checkpoint': (checkpoint, md5_file(checkpoint))}


@click.group()
def cli():
    """Decoding evaluation: run, sweep, compare."""


@cli.command()
@click.option('--name', required=True, help='Run name; results go to <root>/<name>/.')
@click.option('--handoff', 'handoff_path', type=click.Path(exists=True, dir_okay=False),
              help='A stored hand-off (.npz with an .md5 sidecar). Without it, detection runs once '
                   'and its hand-off is stored.')
@click.option('--checkpoint', type=click.Path(exists=True, dir_okay=False),
              help='Detection checkpoint (default: the pinned release checkpoint).')
@click.option('--checkpoint-label', help='Label for a non-pinned checkpoint; its results are kept apart.')
@click.option('--device', default='cuda:0', show_default=True, help='Detection device.')
@click.option('--threshold', default=0.9, show_default=True)
@click.option('--min-distance', default=1, show_default=True)
@click.option('--pool-size', default=0, show_default=True)
@click.option('--num-iter', default=500, show_default=True)
@click.option('--batch-size', default=1000, show_default=True)
@click.option('--pred-prob-thresh', default=0.95, show_default=True)
@click.option('--params-mode', default='2*R*C', show_default=True)
@click.option('--rescue-errors/--no-rescue-errors', default=True, show_default=True)
@click.option('--rescue-mixed/--no-rescue-mixed', default=False, show_default=True)
def run(name, handoff_path, checkpoint, checkpoint_label, device, threshold, min_distance, pool_size,
        num_iter, batch_size, pred_prob_thresh, params_mode, rescue_errors, rescue_mixed):
    """One configuration at seeds 0, 1 and 2."""
    checkpoint, label = _checkpoint_options(checkpoint, checkpoint_label)
    pins = _pins_for(checkpoint, label)
    handoff_settings = {'threshold': threshold, 'min_distance': min_distance, 'pool_size': pool_size}

    def detect(settings):
        return handoff_from_maps(detect_maps(checkpoint, device), **settings)

    summary = run_configuration(
        name, handoff_path=handoff_path, detect_fn=detect, handoff_settings=handoff_settings,
        decoding_settings={'num_iter': num_iter, 'batch_size': batch_size, 'params_mode': params_mode,
                           'pred_prob_thresh': pred_prob_thresh, 'rescue_errors': rescue_errors,
                           'rescue_mixed': rescue_mixed},
        checkpoint_label=label, detection_device=device, pins=pins)
    click.echo(json.dumps({'mean': summary['mean'], 'range': summary['range'],
                           'margins': summary['margins']}, indent=2))


@cli.command()
@click.option('--name', required=True)
@click.option('--checkpoint', type=click.Path(exists=True, dir_okay=False))
@click.option('--checkpoint-label')
@click.option('--device', default='cuda:0', show_default=True, help='Detection device.')
@click.option('--thresholds', default=','.join(map(str, SWEEP_THRESHOLDS)), show_default=True)
@click.option('--num-iter', default=500, show_default=True)
def sweep(name, checkpoint, checkpoint_label, device, thresholds, num_iter):
    """Candidate-threshold grid; detection runs once."""
    checkpoint, label = _checkpoint_options(checkpoint, checkpoint_label)
    pins = _pins_for(checkpoint, label)
    result = run_sweep(name, detect_maps_fn=lambda: detect_maps(checkpoint, device),
                       thresholds=tuple(float(t) for t in thresholds.split(',')),
                       decoding_settings={'num_iter': num_iter}, checkpoint_label=label,
                       detection_device=device, pins=pins)
    for row in result['thresholds']:
        mark = ' (baseline)' if row['is_baseline'] else ''
        click.echo(f"thr {row['threshold']}{mark}: mean gene calls {row['mean']['gene_calls']:.1f}, "
                   f"blank rate {row['mean']['blank_rate']:.4f}, candidates {row['mean']['candidates']:.0f}")


@cli.command()
@click.argument('baseline_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('change_dir', type=click.Path(exists=True, file_okay=False))
@click.option('--rule', type=click.Choice(['accuracy', 'qol', 'identical']), required=True)
@click.option('--benefit', type=click.Choice(['time', 'memory', 'other']),
              help='For --rule qol: the benefit claimed.')
@click.option('--benefit-note', help='For --benefit other: the reported evidence.')
@click.option('--rename', multiple=True, help='old=new column rename the change documents (identical).')
@click.option('--handoff-change', is_flag=True, help='The change alters the hand-off (a new stored hand-off).')
def compare(baseline_dir, change_dir, rule, benefit, benefit_note, rename, handoff_change):
    """Apply a decision rule to two recorded runs. Exits 0 if the claim is supported, 1 if not."""
    base = json.loads((Path(baseline_dir) / 'summary.json').read_text())
    change = json.loads((Path(change_dir) / 'summary.json').read_text())
    if base.get('protocol') != change.get('protocol'):
        raise EvaluationError('The two runs used different protocols; they cannot be compared.')
    if rule == 'accuracy':
        verdict = judge_accuracy(base, change, handoff_change)
    elif rule == 'qol':
        verdict = judge_qol(base, change, benefit, benefit_note, handoff_change)
    else:
        renames = dict(r.split('=', 1) for r in rename)
        verdict = judge_identical(baseline_dir, change_dir, base, change, renames)
    click.echo(json.dumps(verdict, indent=2))
    raise SystemExit(0 if verdict['supported'] else 1)


if __name__ == '__main__':
    cli()
