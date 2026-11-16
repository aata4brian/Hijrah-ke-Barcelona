"""Search mixture/decoder only on external calibration matches, never target GT."""
import argparse
from dataclasses import asdict
from importlib.metadata import version
import gc
import json
from pathlib import Path
import time
import platform
import joblib
import numpy as np
from .artifact_checks import validate_probabilities
from .data import DEV, SCENARIOS, discover, fingerprint, load, split, views
from .decode import Settings, decode
from .evaluate import aggregate, atomic_json, score, write_csv
from .features import pair
from .registry import sync_registry
from evaluation import possession_metrics as pm
from night_benchmark import combine
from .mixtures import Mixture, blend
from .experiment_io import output_lock, code_fingerprints, validate_rows, verified_complete, finish


def candidate_grid(tree_settings, temporal_settings, count=48, seed=731):
    if not 2 <= count <= 4096:
        raise ValueError('Candidate count must be between 2 and 4096')
    entries = []
    def add(a, b, setting):
        item = dict(state_weight=float(a), edge_weight=float(b), decoder=asdict(setting))
        if item not in entries:
            entries.append(item)
    add(.5, .5, tree_settings)
    add(0, 0, temporal_settings)
    for weight in (0., .25, .5, .75, 1.):
        add(weight, weight, tree_settings)
    rng = np.random.default_rng(seed)
    while len(entries) < count:
        add(rng.choice([0., .25, .5, .75, 1.]), rng.choice([0., .5, 1.]),
            Settings(team=float(rng.choice([.5, 1., 2., 4.])),
                     dead=float(rng.choice([1., 2., 4., 6.])),
                     boundary_gain=float(rng.choice([0., .5, 1., 2., 4.])),
                     emission_scale=.2,
                     dead_bias=float(rng.choice([-.35, -.18, 0., .18])),
                     proposal_threshold=float(rng.choice([.025, .075, .15, .3]))))
    return entries[:count]


def predictions(cases, candidate):
    setting = Settings(**candidate['decoder'])
    return [decode(*blend(arrays, candidate['state_weight'], candidate['edge_weight']), setting)
            for arrays, _ in cases]


def calibration_value(scenarios, candidate):
    """Weighted scenario mean; halves separated by X, never concatenated alive."""
    total = 0.
    for scenario, cases in scenarios.items():
        truth = combine([y for _, y in cases])
        pred = combine(predictions(cases, candidate))
        runs = pm.valid_runs(truth)
        total += SCENARIOS[scenario]*pm.spell_scores(truth, pred, runs)['mean_f1']
    return float(total / sum(SCENARIOS[s] for s in scenarios))


def select(scenarios, grid, existing=None, progress=None):
    rows = list(existing or [])
    if not scenarios or not grid:
        raise ValueError('Calibration scenarios and candidates are required')
    if any(row.get('candidate') != i or not isinstance(row.get('calibration_spellF1'), (int,float))
           or not np.isfinite(row['calibration_spellF1']) or not 0 <= row['calibration_spellF1'] <= 1
           for i, row in enumerate(rows)) or len(rows) > len(grid):
        raise ValueError('Invalid partial calibration search')
    for index in range(len(rows), len(grid)):
        value = calibration_value(scenarios, grid[index])
        rows.append(dict(candidate=index, calibration_spellF1=value))
        if progress:
            progress(rows)
    # First candidate wins exact ties, favouring the frozen baseline.
    best = max(range(len(rows)), key=lambda i: (rows[i]['calibration_spellF1'], -i))
    return grid[best], rows, best


def oracle_diagnostics(truth, prediction, probabilities):
    """External-data-only counterfactuals; not deployable scores or upper bounds."""
    if not len(truth) == len(prediction) == len(probabilities):
        raise ValueError('Diagnostic half counts differ')
    perfect_boundaries, segment_relabel = [], []
    for y, pred, p in zip(truth, prediction, probabilities):
        if len(y) != len(pred) or p.shape != (len(y),3):
            raise ValueError('Diagnostic frame counts differ')
        a = np.full(len(y), 3, np.int8)
        b = pred.copy()
        # Truth holes split diagnostics; do not leak a majority vote across X.
        masked_pred = pred.copy(); masked_pred[y >= 3] = 3
        b[y >= 3] = 3
        for source, destination, use_prob in ((y, a, True), (masked_pred, b, False)):
            bounds = np.r_[0, np.flatnonzero(source[1:] != source[:-1])+1, len(source)]
            for first, last in zip(bounds[:-1], bounds[1:]):
                if use_prob:
                    if y[first] < 3:
                        destination[first:last] = p[first:last].mean(axis=0).argmax()
                else:
                    valid = y[first:last][y[first:last] < 3]
                    if len(valid):
                        destination[first:last] = np.bincount(valid, minlength=3).argmax()
        perfect_boundaries.append(a)
        segment_relabel.append(b)
    return dict(oracle_true_boundaries=perfect_boundaries,
                oracle_relabel_predicted_segments=segment_relabel)


def probability_cases(path, scenario, seed, trees, temporal, folder):
    file = folder / (path.stem+'_'+scenario+'.npz')
    if file.exists():
        with np.load(file, allow_pickle=False) as saved:
            result = [(tuple(saved[f'{i}_{key}'] for key in ('p','q','tp','tq')), saved[f'{i}_y'])
                      for i in range(int(saved['halves']))]
        if len(result) != 2:
            raise ValueError('Probability cache must contain both halves')
        for arrays, y in result:
            if y.ndim != 1 or not np.isin(y, [0,1,2,3]).all():
                raise ValueError('Malformed cached labels')
            validate_probabilities(arrays[0], arrays[1], len(y))
            validate_probabilities(arrays[2], arrays[3], len(y))
        return result
    result, arrays_to_save = [], {}
    for i, (data, y) in enumerate(views(load(path), scenario, seed)):
        features, _ = pair(data, True)
        p, q = trees.predict(features)
        tp, tq = temporal.predict(features)
        validate_probabilities(p, q, len(y)); validate_probabilities(tp, tq, len(y))
        arrays = (p, q, tp, tq)
        result.append((arrays, y))
        arrays_to_save.update({f'{i}_{k}': v for k,v in zip(('p','q','tp','tq','y'), (*arrays,y))})
        del features
        gc.collect()
    folder.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, halves=len(result), **arrays_to_save)
    temporary.replace(file)
    return result


def run(args):
    with output_lock(args.output):
        return _run_locked(args)


def _run_locked(args):
    started = time.monotonic()
    train, calibration, tests = split(args.fold)
    paths = discover(args.cache)
    source = args.source / f'fold_{args.fold}'
    recorded_split = json.loads((source/'split.json').read_text(encoding='utf-8'))
    if recorded_split != dict(train=list(train), calibration=calibration, test=list(tests)):
        raise ValueError('Source checkpoint split differs from declared external split')
    base_manifest = json.loads((args.source/'manifest.json').read_text(encoding='utf-8'))
    for match, path in paths.items():
        if fingerprint(path) != base_manifest['cache_sha256'][match]:
            raise ValueError('Source data fingerprint mismatch: '+match)
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    models = [source/'tree_candidate.joblib', source/'tcn_candidate.joblib']
    manifest = dict(version='overnight_decoder_v1', folds=[args.fold],
        seed=args.seed, test_seed=base_manifest['seed']+101, calibration_seed=base_manifest['seed']+17,
        candidates=args.candidates, source_models={p.name:fingerprint(p) for p in models},
        python=platform.python_version(),
        packages={n:version(n) for n in ('numpy','scipy','scikit-learn','joblib','numba','torch')},
        source_manifest_sha256=fingerprint(args.source/'manifest.json'),
        cache_sha256={m:fingerprint(p) for m,p in paths.items()},
        code_sha256=code_fingerprints(['__init__.py','search_decoder.py','mixtures.py','experiment_io.py',
            'data.py','features.py','models.py','tcn.py','labels.py','decode.py','evaluate.py','artifact_checks.py']),
        selection_source='external calibration match only; target public/private GT never read',
        test_scenarios=list(SCENARIOS),
        note='Known seven-match controlled retest, not a new holdout. Calibration search is optimistic.')
    file = root/'manifest.json'
    if file.exists() and json.loads(file.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Output has another code/config/data version; use a new output')
    if not file.exists():
        atomic_json(file, manifest)
    if verified_complete(root, len(tests)*len(SCENARIOS)*2):
        print('Decoder search complete; hashes verified, no artifacts overwritten', flush=True)
        sync_registry(strict=True)
        return
    trees, temporal = (joblib.load(p) for p in models)
    grid = candidate_grid(trees.decoder, temporal.decoder, args.candidates, args.seed)
    gridpath = root/'search_grid.json'
    if gridpath.exists() and json.loads(gridpath.read_text(encoding='utf-8')) != grid:
        raise ValueError('Saved search grid differs from declared configuration')
    if not gridpath.exists():
        atomic_json(gridpath, grid)
    print(json.dumps(dict(stage='calibration_probabilities', match=calibration, fold=args.fold)), flush=True)
    calibration_cases = {s:probability_cases(paths[calibration], s, manifest['calibration_seed'],
        trees, temporal, root/'probability_cache') for s in SCENARIOS}
    partial_file = root/'calibration_search.json'
    previous = json.loads(partial_file.read_text()) if partial_file.exists() else []
    def progress(rows):
        atomic_json(partial_file, rows)
        print(json.dumps(dict(stage='calibration_search', done=len(rows), total=len(grid),
                             elapsed_seconds=round(time.monotonic()-started,1))), flush=True)
    chosen, rows, index = select(calibration_cases, grid, previous, progress)
    atomic_json(root/'selection.json', dict(candidate=chosen, candidate_index=index,
        calibration_match=calibration, selection_source='calibration only, never test',
        baseline_calibration_spellF1=rows[0]['calibration_spellF1'],
        selected_calibration_spellF1=rows[index]['calibration_spellF1']))
    mixed = Mixture(trees, temporal, chosen['state_weight'], chosen['edge_weight'], Settings(**chosen['decoder']))
    temporary = root/'candidate.tmp'
    joblib.dump(mixed, temporary, compress=3); temporary.replace(root/'candidate.joblib')
    del calibration_cases
    gc.collect()
    all_rows, diagnostics = [], []
    for match in tests:
        for scenario in SCENARIOS:
            rowpath = root/f'rows_{match}_{scenario}.json'
            diagpath = root/f'diagnostics_{match}_{scenario}.json'
            if rowpath.exists() and diagpath.exists():
                all_rows.extend(validate_rows(json.loads(rowpath.read_text()),
                    ['frozen_v44_ensemble','calibration_selected_decoder'],args.fold,match,scenario,manifest['test_seed']))
                diagnostics.extend(validate_rows(json.loads(diagpath.read_text()),
                    ['oracle_true_boundaries','oracle_relabel_predicted_segments'],args.fold,match,scenario,manifest['test_seed']))
            else:
                print(json.dumps(dict(stage='external_evaluation', fold=args.fold, match=match, scenario=scenario)), flush=True)
                cases = probability_cases(paths[match], scenario, manifest['test_seed'], trees, temporal, root/'probability_cache')
                truth = [y for _,y in cases]
                meta = dict(fold=args.fold, split='development' if match in DEV else 'known_retest',
                            match=match, scenario=scenario, seed=manifest['test_seed'])
                base, improved = predictions(cases, grid[0]), predictions(cases, chosen)
                records = [score(truth, p, method=name, **meta) for name,p in
                    [('frozen_v44_ensemble',base), ('calibration_selected_decoder',improved)]]
                counterfactuals = oracle_diagnostics(truth, base, [blend(a,.5,.5)[0] for a,_ in cases])
                diag = [score(truth, p, method=name, **meta) for name,p in counterfactuals.items()]
                atomic_json(rowpath, records); atomic_json(diagpath, diag)
                all_rows.extend(records); diagnostics.extend(diag)
                del cases
                gc.collect()
            write_csv(root/'performa_detail.csv', all_rows)
            write_csv(root/'diagnostik_bottleneck.csv', diagnostics)
            write_csv(root/'ringkasan_performa.csv', aggregate(all_rows))
            sync_registry()
    if len(all_rows) != len(tests)*len(SCENARIOS)*2:
        raise ValueError('Evaluation matrix is incomplete')
    checkpoint_names = [f'{prefix}_{match}_{scenario}.json' for match in tests
        for scenario in SCENARIOS for prefix in ('rows','diagnostics')]
    finish(root, ['manifest.json','search_grid.json','selection.json','calibration_search.json',
        *checkpoint_names,
        'candidate.joblib','performa_detail.csv','diagnostik_bottleneck.csv','ringkasan_performa.csv'], rows=len(all_rows),
        summary=aggregate(all_rows), selected_on='external calibration match only',
        oracle_diagnostics_not_deployable=True,
        elapsed_seconds=time.monotonic()-started)
    sync_registry(strict=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/riset_v44_full'))
    parser.add_argument('--cache', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fold', type=int, choices=range(5), required=True)
    parser.add_argument('--seed', type=int, default=731)
    parser.add_argument('--candidates', type=int, default=48)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
