"""Bounded V4.6 ablation: frozen deep + old/new state features, no target GT."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import platform
import time
from importlib.metadata import version
import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
import torch
from .artifact_checks import validate_probabilities
from .data import DEV, SCENARIOS, discover, fingerprint, load, split, views
from .decode import decode
from .evaluate import aggregate, atomic_json, score, write_csv
from .experiment_io import code_fingerprints, finish, output_lock, validate_rows, verified_complete
from .features import pair
from .fusion_features import SCHEMA, extra_pair
from .fusion_model import combine_probabilities, specialist_probability
from .labels import CLASS_SWAP, sample_weights, train_indices
from .registry import sync_registry
from .run import save_model

METHODS = ['frozen_deep_v45', 'frozen_ensemble_deep_v45', 'control_old_features',
           'expanded_tracking_features', 'calibration_selected_v46']
TRAIN_SCENARIOS = ['original', 'camera_gaps', 'drift']
PARAMETERS = dict(max_iter=60, max_leaf_nodes=20, min_samples_leaf=80,
                  learning_rate=.08, l2_regularization=2., early_stopping=False, random_state=331)
TREE_WEIGHTS = (0., .25, .5)
SPECIALIST_WEIGHTS = (0., .15, .35)


def training_samples(paths, matches, report):
    xs, ys, ws = [], [], []
    rng = np.random.default_rng(331)
    for match in matches:
        halves = load(paths[match])
        for scenario in TRAIN_SCENARIOS:
            for half, (data, y) in enumerate(views(halves, scenario, 331)):
                features, _ = pair(data); added, _ = extra_pair(data)
                idx = train_indices(y, step=15)
                if len(idx) > 6000:
                    idx = np.sort(rng.choice(idx, 6000, replace=False))
                weights = sample_weights(y)[idx]*SCENARIOS[scenario]
                for side in range(2):
                    xs.append(np.column_stack([features[side][idx], added[side][idx]]))
                    ys.append(y[idx] if side == 0 else CLASS_SWAP[y[idx]])
                    ws.append(weights)
                report('training_features', match=match, scenario=scenario, half=half, samples=len(idx)*2)
                del features, added; gc.collect()
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)


def specialist_temperature(probabilities, truth):
    p, y = np.concatenate(probabilities), np.concatenate(truth)
    valid = y < 3; logs = np.log(np.clip(p[valid], 1e-7, 1)); y = y[valid]
    def loss(t):
        p = softmax(logs/t, axis=1)
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean()
    return float(minimize_scalar(loss, bounds=(.5, 3), method='bounded').x)


def predict_case(deep, tree, specialists, data):
    features, _ = pair(data); added, _ = extra_pair(data)
    # Large inference chunks avoid repeated GPU launch/halo overhead. No network training.
    p, q = deep.raw(features, chunk=8192)
    dp, dq = deep.calibration.state(p), deep.calibration.edge(q)
    tp, tq = tree.predict(features)
    sp = {}
    for name, model in specialists.items():
        inputs = features if name == 'old' else tuple(np.column_stack([f, a]) for f, a in zip(features, added))
        sp[name] = specialist_probability(model, inputs)
    for pp, qq in ((dp, dq), (tp, tq)):
        validate_probabilities(pp, qq, len(data.ball))
    return dict(deep=(dp, dq), tree=(tp, tq), specialists=sp)


def calibrated_specialist(case, name, temperatures):
    p = case['specialists'][name]
    return softmax(np.log(np.clip(p, 1e-7, 1))/temperatures[name], axis=1).astype(np.float32)


def select_on_calibration(cases, truth, scenarios, temperatures, deep, tree):
    candidates = []
    for name in ('old', 'expanded'):
        for tw in TREE_WEIGHTS:
            # Decoder comes with the chosen frozen base; no new test-driven tuning.
            setting = deep.decoder if tw == 0 else tree.decoder
            for sw in SPECIALIST_WEIGHTS:
                scores = []
                for case, y in zip(cases, truth):
                    p, q = combine_probabilities(case['deep'], case['tree'],
                        calibrated_specialist(case, name, temperatures), tw, sw)
                    scores.append(score([y], [decode(p, q, setting)])['spellF1'])
                # Equal weighting of halves, then declared scenario weights (same halves per scenario).
                value = float(np.average(scores, weights=[SCENARIOS[s] for s in scenarios]))
                candidates.append(dict(specialist=name, tree_weight=tw, specialist_weight=sw,
                    decoder=asdict(setting), calibration_spellF1=value))
    # Stable order prefers deep-only/zero extra weight on exact ties.
    best = {name: max((c for c in candidates if c['specialist'] == name),
                     key=lambda c: c['calibration_spellF1']) for name in ('old', 'expanded')}
    chosen = max(candidates, key=lambda c: c['calibration_spellF1'])
    return dict(chosen=chosen, by_ablation=best, candidates=candidates,
                selection_source='calibration match only; no test labels used')


def apply_choice(case, choice, temperatures, deep, tree):
    p, q = combine_probabilities(case['deep'], case['tree'],
        calibrated_specialist(case, choice['specialist'], temperatures),
        choice['tree_weight'], choice['specialist_weight'])
    setting = deep.decoder if choice['tree_weight'] == 0 else tree.decoder
    return decode(p, q, setting)


def run(args):
    with output_lock(args.output):
        return run_locked(args)


def run_locked(args):
    root = args.output; start = time.monotonic(); torch.set_num_threads(4)
    paths = discover(args.cache); train, calibration, tests = split(args.fold)
    deep_folder = args.deep_root/f'local_deep_v45_fold_{args.fold}'
    tree_folder = args.source/f'fold_{args.fold}'
    expected_split = dict(train=list(train), calibration=calibration, test=list(tests))
    dm = json.loads((deep_folder/'manifest.json').read_text())
    if dm['split'] != expected_split or json.loads((tree_folder/'split.json').read_text()) != expected_split:
        raise ValueError('Frozen model splits differ')
    for match, path in paths.items():
        if fingerprint(path) != dm['cache_sha256'][match]:
            raise ValueError('External cache differs from frozen baseline')
    model_paths = [deep_folder/'variant_candidate.joblib', tree_folder/'tree_candidate.joblib']
    manifest = dict(version='fusion_v46', folds=[args.fold], split=expected_split,
        train_seed=331, calibration_seed=348, test_seed=432, train_scenarios=TRAIN_SCENARIOS,
        test_scenarios=list(SCENARIOS), feature_schema=SCHEMA, old_columns=203, extra_columns=78,
        parameters=PARAMETERS, cap_train_rows_per_half_scenario_side=6000,
        grid=dict(tree_weights=TREE_WEIGHTS, specialist_weights=SPECIALIST_WEIGHTS),
        cache_sha256={m:fingerprint(p) for m,p in paths.items()},
        source_models_sha256={str(p):fingerprint(p) for p in model_paths},
        code_sha256=code_fingerprints(['fusion_experiment.py','fusion_features.py','fusion_model.py',
            'features.py','models.py','deep_temporal.py','decode.py','labels.py','data.py','evaluate.py',
            'experiment_io.py','run.py','tcn.py']), python=platform.python_version(),
        packages={n:version(n) for n in ('numpy','scipy','scikit-learn','joblib','torch','numba')},
        target_gt_read=False, cue_modalities_trained=False, no_new_deep_training=True,
        note='Known seven matches controlled retest, not new holdout. Cue sidecars excluded: no aligned external audio/flag labels.')
    # JSON canonicalization makes resume comparison independent of tuple/list representation.
    manifest = json.loads(json.dumps(manifest))
    path = root/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Changed code/data/config; choose a new output directory')
    atomic_json(path, manifest)
    if verified_complete(root, 75):
        return dict(complete=True, reused=True)
    def report(stage, **kw):
        event = dict(stage=stage, fold=args.fold, elapsed_seconds=time.monotonic()-start, **kw)
        atomic_json(root/'status.json', event); print(json.dumps(event), flush=True)
    specialists = {}
    missing = [name for name in ('old','expanded') if not (root/f'{name}_specialist.joblib').exists()]
    if missing:
        x, y, w = training_samples(paths, train, report)
        report('training_sample_matrix', rows=len(y), columns=x.shape[1], matrix_mib=x.nbytes/1024**2)
        for name in missing:
            report('fit_specialist', name=name)
            with threadpool_limits(limits=4):
                model = HistGradientBoostingClassifier(**PARAMETERS).fit(
                    x[:, :203] if name == 'old' else x, y, sample_weight=w)
            save_model(root/f'{name}_specialist.joblib', model)
        del x, y, w; gc.collect()
    for name in ('old','expanded'):
        specialists[name] = joblib.load(root/f'{name}_specialist.joblib')
    deep, tree = (joblib.load(p) for p in model_paths)
    selection_path = root/'calibration_selection.json'
    if not selection_path.exists():
        cases, truth, scenarios = [], [], []
        for scenario in SCENARIOS:
            for half, (data, y) in enumerate(views(load(paths[calibration]), scenario, 348)):
                report('calibration_predictions', match=calibration, scenario=scenario, half=half)
                cases.append(predict_case(deep, tree, specialists, data)); truth.append(y); scenarios.append(scenario)
        temperatures = {name:specialist_temperature([c['specialists'][name] for c in cases], truth)
                        for name in specialists}
        report('select_calibration_only')
        selection = select_on_calibration(cases, truth, scenarios, temperatures, deep, tree)
        selection.update(temperatures=temperatures, calibration_match=calibration)
        atomic_json(selection_path, selection)
        del cases, truth; gc.collect()
    selection = json.loads(selection_path.read_text()); temperatures = selection['temperatures']
    report('selection_frozen_before_test', chosen=selection['chosen'])
    rows = []; checkpoints = []
    for match in tests:
        for scenario in SCENARIOS:
            file = root/f'rows_{match}_{scenario}.json'; checkpoints.append(file.name)
            if file.exists():
                records = validate_rows(json.loads(file.read_text()), METHODS, args.fold, match, scenario, 432)
            else:
                report('external_evaluation', match=match, scenario=scenario)
                truth = []; predictions = {name:[] for name in METHODS}
                for data, y in views(load(paths[match]), scenario, 432):
                    case = predict_case(deep, tree, specialists, data)
                    p,q = case['deep']; tp,tq = case['tree']
                    values = [decode(p,q,deep.decoder), decode((p+tp)/2,(q+tq)/2,tree.decoder),
                        apply_choice(case, selection['by_ablation']['old'], temperatures, deep, tree),
                        apply_choice(case, selection['by_ablation']['expanded'], temperatures, deep, tree),
                        apply_choice(case, selection['chosen'], temperatures, deep, tree)]
                    truth.append(y)
                    for name,value in zip(METHODS,values): predictions[name].append(value)
                meta = dict(fold=args.fold, split='development' if match in DEV else 'known_retest',
                            match=match, scenario=scenario, seed=432)
                records = [score(truth,predictions[name],method=name,**meta) for name in METHODS]
                # Reproduce two already measured baselines, do not compare incompatible runs.
                frozen = json.loads((deep_folder/file.name).read_text())
                for name, old in [('frozen_deep_v45','deep_v45'),('frozen_ensemble_deep_v45','ensemble_deep_v45')]:
                    baseline = next(r for r in frozen if r['method'] == old)
                    current = next(r for r in records if r['method'] == name)
                    for metric in ('spellF1','macroF1','deadF1','predicted_spells','reference_spells'):
                        if abs(current[metric]-baseline[metric]) > 1e-12:
                            raise ValueError('Frozen baseline reproduction differs: '+name+' '+metric)
                validate_rows(records,METHODS,args.fold,match,scenario,432); atomic_json(file,records)
            rows.extend(records); write_csv(root/'performa_detail.csv',rows)
            write_csv(root/'ringkasan_performa.csv',aggregate(rows))
    finish(root,['manifest.json','calibration_selection.json','old_specialist.joblib','expanded_specialist.joblib',
        'performa_detail.csv','ringkasan_performa.csv',*checkpoints], rows=len(rows), elapsed_seconds=time.monotonic()-start,
        summary=aggregate(rows), frozen_baselines_reproduced=True)
    sync_registry(); report('complete', rows=len(rows), summary=aggregate(rows))
    return dict(complete=True, summary=aggregate(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('outputs/riset_v44_full'))
    parser.add_argument('--deep-root',type=Path,default=Path('outputs'))
    parser.add_argument('--cache',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fold',type=int,choices=range(5),default=0)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
