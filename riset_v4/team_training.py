"""Paired overnight variants; never modify the frozen V4.4 implementation."""
import argparse
from dataclasses import asdict
import gc
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import time
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from .artifact_checks import validate_probabilities
from .data import DEV, SCENARIOS, discover, fingerprint, load, split, views
from .decode import decode
from .evaluate import aggregate, atomic_json, score, write_csv
from .experiment_io import code_fingerprints, finish, output_lock, validate_rows, verified_complete
from .features import pair
from .labels import CLASS_SWAP, EDGE_SWAP, boundary_targets, sample_weights, train_indices
from .mixtures import Mixture
from .models import Heads, switch_context
from .registry import sync_registry
from .run import calibration_cases, fit_calibration, log, save_model

BASE_TREE = dict(max_iter=90, max_leaf_nodes=25, min_samples_leaf=64,
                 learning_rate=.09, l2_regularization=1.3, early_stopping=False, random_state=331)
CAPACITY_TREE = dict(BASE_TREE, max_iter=240, max_leaf_nodes=41, min_samples_leaf=40,
                     learning_rate=.06, l2_regularization=2.)
BASE_SCENARIOS = ['original', 'camera_gaps', 'drift']


class ResearchHeads(Heads):
    """Explicit configurable heads, leaving models.Heads unchanged."""
    def fit(self, cases, parameters, work_dir):
        work_dir = Path(work_dir); work_dir.mkdir(parents=True, exist_ok=True)
        self.parameters = dict(parameters)
        specification = work_dir/'parameters.json'
        if specification.exists() and json.loads(specification.read_text(encoding='utf-8')) != parameters:
            raise ValueError('Tree checkpoint parameters changed')
        if not specification.exists(): atomic_json(specification,parameters)
        self.n_features = cases[0][0][0].shape[1]
        selections = [train_indices(y, step=10) for _, y, _ in cases]
        n = sum(2*len(idx) for idx in selections)
        for name in ('state', 'edge'):
            checkpoint = work_dir/(name+'.joblib')
            if checkpoint.exists():
                setattr(self, name+'_model', joblib.load(checkpoint)); continue
            file = work_dir/(name+'_samples.npy')
            width = self.n_features*(2 if name == 'edge' else 1)
            x = np.lib.format.open_memmap(file, mode='w+', dtype=np.float32, shape=(n,width))
            target = np.empty(n, np.int8); weights = np.empty(n, np.float32); first = 0
            try:
                for (features,y,scenario_weight),idx in zip(cases,selections):
                    if name == 'state':
                        labels, w = y, sample_weights(y)
                    else:
                        edges = boundary_targets(y,radius=2)
                        labels = np.where(edges.max(axis=1)>0,edges.argmax(axis=1)+1,0)
                        w = np.where(labels>0,15.,1.)
                    for side in range(2):
                        last = first+len(idx)
                        x[first:last] = switch_context(features[side],idx) if name == 'edge' else features[side][idx]
                        yy = labels[idx].copy()
                        if side:
                            if name == 'state': yy = CLASS_SWAP[yy]
                            else:
                                positive = yy>0; yy[positive] = EDGE_SWAP[yy[positive]-1]+1
                        target[first:last] = yy; weights[first:last] = w[idx]*scenario_weight; first = last
                x.flush()
                with threadpool_limits(limits=4):
                    model = HistGradientBoostingClassifier(**parameters).fit(x,target,sample_weight=weights)
                setattr(self,name+'_model',model); save_model(checkpoint,model)
            finally:
                x._mmap.close(); del x
            file.unlink()  # Only this runner's reproducible temporary sample matrix.
            del target,weights; gc.collect()
        return self


def training_cases(paths, train, scenarios, seed, root):
    """Disk-backed features, no legacy feature duplication in RAM."""
    folder = root/'feature_cache'; folder.mkdir(exist_ok=True); cases = []
    for match in train:
        for scenario in scenarios:
            log(root,'training_features',match=match,scenario=scenario)
            for half,(data,y) in enumerate(views(load(paths[match]),scenario,seed)):
                names = [folder/f'{match}_{scenario}_{seed}_{half}_{side}.npy' for side in range(2)]
                if not all(p.is_file() for p in names):
                    features,_ = pair(data,True)
                    for file,values in zip(names,features):
                        temp = file.with_suffix('.tmp')
                        with temp.open('wb') as handle: np.save(handle,values)
                        temp.replace(file)
                    del features,values; gc.collect()
                disk = tuple(np.load(file,mmap_mode='r') for file in names)
                if any(a.shape[0] != len(y) for a in disk): raise ValueError('Feature cache shape mismatch')
                cases.append((disk,y,SCENARIOS[scenario]))
    return cases


def close_features(cases):
    for features,_,_ in cases:
        for array in features: array._mmap.close()


def required_training_bytes(paths, train, scenarios):
    # Full direct+mirrored features and the largest edge training matrix.
    # Deliberately conservative: all frames may become training samples.
    frames = sum(sum(len(y) for _,y in load(paths[m])) for m in train)*len(scenarios)
    return frames*203*4*6 + 512*1024**2


def methods_for(track):
    return ['frozen_v44_'+('tcn' if track=='tcn' else 'tree'),
            'long_tcn' if track=='tcn' else 'capacity_tree' if track=='capacity' else 'augment5_tree',
            'frozen_v44_ensemble', 'ensemble_'+track]


def run(args):
    with output_lock(args.output):
        return _run_locked(args)


def _run_locked(args):
    started = time.monotonic(); root = args.output; train,calibration,tests = split(args.fold)
    paths = discover(args.cache); source = args.source/f'fold_{args.fold}'
    base = json.loads((args.source/'manifest.json').read_text(encoding='utf-8'))
    if base['train_scenarios'] != BASE_SCENARIOS or base['seed'] != 331 or base['iterations'] != 90:
        raise ValueError('Expected frozen baseline protocol (seed331, 90 iterations, three train scenarios)')
    recorded = json.loads((source/'split.json').read_text(encoding='utf-8'))
    if recorded != dict(train=list(train),calibration=calibration,test=list(tests)):
        raise ValueError('Frozen source split mismatch')
    for match,path in paths.items():
        if fingerprint(path) != base['cache_sha256'][match]: raise ValueError('Cache fingerprint mismatch: '+match)
    if args.track=='tcn' and args.epochs <= base['epochs']:
        raise ValueError('Long TCN epochs must exceed frozen baseline epochs')
    scenarios = list(SCENARIOS) if args.track=='augment' else BASE_SCENARIOS
    parameters = CAPACITY_TREE if args.track=='capacity' else BASE_TREE
    source_files = ['tree_candidate.joblib','tcn_candidate.joblib']
    if args.track=='tcn': source_files.append('tcn_training.pt')
    manifest = dict(version='team_overnight_v1',track=args.track,folds=[args.fold],
        seed=331,test_seed=432,calibration_seed=348,train_scenarios=scenarios,
        calibration_scenarios=BASE_SCENARIOS,test_scenarios=list(SCENARIOS),
        epochs=args.epochs if args.track=='tcn' else 0,steps=base['steps'],
        tree_parameters=parameters,device=args.device,python=platform.python_version(),
        packages={n:version(n) for n in ('numpy','scipy','scikit-learn','joblib','numba','torch')},
        source_manifest_sha256=fingerprint(args.source/'manifest.json'),
        source_models={n:fingerprint(source/n) for n in source_files},
        cache_sha256={m:fingerprint(p) for m,p in paths.items()},
        code_sha256=code_fingerprints(['__init__.py','team_training.py','experiment_io.py','mixtures.py',
            'run.py','models.py','tcn.py','data.py','features.py','labels.py','decode.py','evaluate.py','artifact_checks.py']),
        selection_source='Calibration match only. Target GT not read. No automatic promotion.',
        note='Known seven-match controlled retest, not a new holdout; paired baseline with identical test corruptions.')
    manifest_path = root/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Different code/data/config/environment: choose a new output')
    if not manifest_path.exists(): atomic_json(manifest_path,manifest)
    methods = methods_for(args.track); expected = len(tests)*len(SCENARIOS)*len(methods)
    if verified_complete(root,expected):
        sync_registry(strict=True); print('Verified complete; not overwritten',flush=True); return
    variant_path = root/'variant_candidate.joblib'
    if variant_path.exists(): variant = joblib.load(variant_path)
    else:
        uncalibrated = root/'variant_uncalibrated.joblib'
        if uncalibrated.exists(): variant = joblib.load(uncalibrated)
        else:
            needed = required_training_bytes(paths,train,scenarios)+args.min_free_gib*1024**3
            log(root,'disk_preflight',required_free_gib=needed/1024**3,free_gib=shutil.disk_usage(root).free/1024**3)
            if shutil.disk_usage(root).free < needed:
                raise RuntimeError('Insufficient free disk for training; no existing data deleted')
            cases = training_cases(paths,train,scenarios,331,root)
            try:
                log(root,'training_variant',track=args.track,fold=args.fold)
                if args.track=='tcn':
                    import torch
                    from .tcn import Temporal
                    checkpoint = root/'tcn_training.pt'
                    if not checkpoint.exists(): shutil.copyfile(source/'tcn_training.pt',checkpoint)
                    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
                    if not base['epochs'] <= saved['epoch'] <= args.epochs:
                        raise ValueError('Unexpected TCN checkpoint epoch')
                    if len(saved['history']) != saved['epoch']:
                        raise ValueError('Incomplete TCN checkpoint history')
                    del saved
                    variant = Temporal().fit(cases,epochs=args.epochs,steps=base['steps'],device=args.device,seed=331,
                        checkpoint=checkpoint,progress=lambda v:log(root,'training_tcn',fold=args.fold,**v))
                else:
                    variant = ResearchHeads().fit(cases,parameters,root/'tree_work')
                save_model(uncalibrated,variant)
            finally: close_features(cases)
            del cases; gc.collect()
            # Clean only generated feature arrays AFTER fitted model was saved.
            for file in (root/'feature_cache').glob('*.npy'): file.unlink()
        log(root,'calibrating',match=calibration,fold=args.fold)
        variant = fit_calibration(variant,calibration_cases(paths[calibration],BASE_SCENARIOS,348),root,'variant')
        save_model(variant_path,variant)
    trees = joblib.load(source/'tree_candidate.joblib'); temporal = joblib.load(source/'tcn_candidate.joblib')
    ensemble_path = root/'ensemble_candidate.joblib'
    improved_ensemble = Mixture(trees if args.track=='tcn' else variant,
        variant if args.track=='tcn' else temporal,.5,.5,trees.decoder if args.track=='tcn' else variant.decoder)
    save_model(ensemble_path,improved_ensemble); rows = []; rowfiles = []
    for match in tests:
        for scenario in SCENARIOS:
            rowpath = root/f'rows_{match}_{scenario}.json'; rowfiles.append(rowpath.name)
            if rowpath.exists():
                records = validate_rows(json.loads(rowpath.read_text()),methods,args.fold,match,scenario,432)
            else:
                log(root,'evaluating',track=args.track,fold=args.fold,match=match,scenario=scenario)
                truth = []; predictions = {m:[] for m in methods}
                for data,y in views(load(paths[match]),scenario,432):
                    features,_ = pair(data,True)
                    p,q = trees.predict(features); tp,tq = temporal.predict(features); vp,vq = variant.predict(features)
                    for pp,qq in ((p,q),(tp,tq),(vp,vq)): validate_probabilities(pp,qq,len(y))
                    bp,bq,bd = (tp,tq,temporal.decoder) if args.track=='tcn' else (p,q,trees.decoder)
                    ep,eq = ((p+vp)/2,(q+vq)/2) if args.track=='tcn' else ((vp+tp)/2,(vq+tq)/2)
                    values = [decode(bp,bq,bd),decode(vp,vq,variant.decoder),
                              decode((p+tp)/2,(q+tq)/2,trees.decoder),decode(ep,eq,improved_ensemble.decoder)]
                    truth.append(y)
                    for name,value in zip(methods,values): predictions[name].append(value)
                    del features; gc.collect()
                meta = dict(fold=args.fold,split='development' if match in DEV else 'known_retest',
                            match=match,scenario=scenario,seed=432)
                records = [score(truth,predictions[name],method=name,**meta) for name in methods]
                validate_rows(records,methods,args.fold,match,scenario,432); atomic_json(rowpath,records)
            rows.extend(records)
            write_csv(root/'performa_detail.csv',rows); write_csv(root/'ringkasan_performa.csv',aggregate(rows))
            sync_registry()
    if len(rows) != expected: raise ValueError('Evaluation matrix incomplete')
    finish(root,['manifest.json','variant_candidate.joblib','ensemble_candidate.joblib',
        'calibration_variant.json','decoder_variant.json','performa_detail.csv','ringkasan_performa.csv',*rowfiles],
        rows=len(rows),summary=aggregate(rows),elapsed_seconds=time.monotonic()-started)
    sync_registry(strict=True); log(root,'complete',rows=len(rows),track=args.track)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True); parser.add_argument('--cache',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--fold',type=int,choices=range(5),required=True)
    parser.add_argument('--track',choices=['tcn','capacity','augment'],required=True)
    parser.add_argument('--epochs',type=int,default=20); parser.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
    parser.add_argument('--min-free-gib',type=float,default=6.)
    args = parser.parse_args()
    # CLI execution must serialize ResearchHeads under its stable import path.
    from riset_v4.team_training import run as canonical_run
    with threadpool_limits(limits=4): canonical_run(args)


if __name__=='__main__': main()
