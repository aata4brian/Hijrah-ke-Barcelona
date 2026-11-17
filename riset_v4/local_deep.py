"""Independent local V4.5 experiment; target labels never enter this module."""
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
import torch
from .artifact_checks import validate_probabilities
from .data import DEV, SCENARIOS, discover, fingerprint, load, split, views
from .decode import Settings, decode
from .deep_temporal import DeepConfig, DeepTemporal
from .evaluate import aggregate, atomic_json, score, write_csv
from .experiment_io import code_fingerprints, finish, output_lock, validate_rows, verified_complete
from .features import pair
from .mixtures import Mixture
from .registry import sync_registry
from .run import fit_calibration, save_model

METHODS = ['frozen_v44_tcn','frozen_v44_ensemble','deep_v45','ensemble_deep_v45']


class StreamingCalibrationCases:
    """Recompute one calibration half at a time, matching disk float16 bytes."""
    def __init__(self, paths, matches, scenarios, seed):
        self.paths = paths
        self.matches = tuple(matches)
        self.scenarios = tuple(scenarios)
        self.seed = seed

    def __iter__(self):
        for match in self.matches:
            halves = load(self.paths[match])
            for scenario in self.scenarios:
                for data, y in views(halves, scenario, self.seed):
                    features, _ = pair(data, True)
                    compact = []
                    for x in features:
                        if np.any(np.abs(x[np.isfinite(x)]) > 60000):
                            raise ValueError('Float16 feature overflow risk')
                        compact.append(x.astype(np.float16))
                    del features, x
                    yield tuple(compact), y, SCENARIOS[scenario]
                    del compact, data, y


def feature_cases(paths, matches, scenarios, seed, folder, report):
    folder.mkdir(parents=True,exist_ok=True); result=[]
    try:
        for match in matches:
            halves=load(paths[match])
            for scenario in scenarios:
                for half,(data,y) in enumerate(views(halves,scenario,seed)):
                    names=[folder/f'{match}_{scenario}_{seed}_{half}_{side}.npy' for side in range(2)]
                    receipt=folder/f'{match}_{scenario}_{seed}_{half}.json'
                    if receipt.exists():
                        hashes=json.loads(receipt.read_text())
                        if any(not p.exists() or fingerprint(p)!=hashes.get(p.name) for p in names):
                            raise ValueError('Feature cache modified; preserve evidence and choose new output')
                    else:
                        report('feature_cache',match=match,scenario=scenario,half=half)
                        features,_=pair(data,True)
                        for p,x in zip(names,features):
                            if np.any(np.abs(x[np.isfinite(x)])>60000): raise ValueError('Float16 feature overflow risk')
                            temp=p.with_suffix('.tmp')
                            with temp.open('wb') as handle: np.save(handle,x.astype(np.float16))
                            temp.replace(p)
                        atomic_json(receipt,{p.name:fingerprint(p) for p in names})
                        del features,x; gc.collect()
                    disk=tuple(np.load(p,mmap_mode='r') for p in names)
                    if any(x.dtype!=np.float16 or x.shape!=(len(y),203) for x in disk):
                        raise ValueError('Unexpected disk feature schema')
                    result.append((disk,y,SCENARIOS[scenario]))
        return result
    except BaseException:
        close_cases(result); raise


def close_cases(cases):
    if isinstance(cases, StreamingCalibrationCases):
        return  # No open files or persistent feature arrays.
    for arrays,_,_ in cases:
        for array in arrays: array._mmap.close()


def calibration_value(model,cases,setting):
    values=[]; weights=[]
    for features,y,weight in cases:
        p,q=model.predict(features)
        validate_probabilities(p,q,len(y))
        values.append(score([y],[decode(p,q,setting)])['spellF1']); weights.append(weight)
    return float(np.average(values,weights=weights))


def run(args):
    with output_lock(args.output): return run_locked(args)


def run_locked(args):
    root=args.output; started=time.monotonic(); config=DeepConfig(epochs=args.epochs,steps=args.steps,
        batch=args.batch,validation_every=args.validation_every)
    calibration_storage = getattr(args, 'calibration_storage', 'disk')
    if calibration_storage not in ('disk', 'stream'):
        raise ValueError('Unknown calibration storage')
    train,calibration,tests=split(args.fold); paths=discover(args.cache)
    source=args.source/f'fold_{args.fold}'
    base=json.loads((args.source/'manifest.json').read_text())
    recorded=json.loads((source/'split.json').read_text())
    if recorded!=dict(train=list(train),calibration=calibration,test=list(tests)): raise ValueError('Baseline split differs')
    if base['seed']!=331 or base['epochs']!=5: raise ValueError('Expected frozen V4.4 training protocol')
    if any(fingerprint(p)!=base['cache_sha256'][m] for m,p in paths.items()): raise ValueError('Cache changed')
    manifest=dict(version='local_deep_v45',folds=[args.fold],seed=331,test_seed=432,calibration_seed=348,
        train_scenarios=list(SCENARIOS),test_scenarios=list(SCENARIOS),config=asdict(config),
        split=recorded,feature_storage='float16, training and calibration cache only; evaluation float32',
        architecture='warm-start 8-block dilated TCN + residual 6-block probability refiner, halo636',
        python=platform.python_version(),packages={n:version(n) for n in ('numpy','scipy','scikit-learn','numba','joblib','torch')},
        source_models={p.name:fingerprint(p) for p in (source/'tree_candidate.joblib',source/'tcn_candidate.joblib')},
        source_manifest_sha256=fingerprint(args.source/'manifest.json'),cache_sha256={m:fingerprint(p) for m,p in paths.items()},
        code_sha256=code_fingerprints(['__init__.py','local_deep.py','deep_temporal.py','experiment_io.py',
            'tcn.py','models.py','features.py','labels.py','data.py','decode.py','mixtures.py','evaluate.py','run.py']),
        selection_source='calibration match only; repeated scenarios not independent matches',
        target_gt_read=False,note='Controlled retest on known seven matches, not new holdout; no automatic promotion')
    manifest['calibration_storage'] = calibration_storage
    if calibration_storage == 'stream':
        manifest['feature_storage'] = 'float16 training disk cache and calibration streaming RAM; evaluation float32'
    file=root/'manifest.json'
    if file.exists() and json.loads(file.read_text())!=manifest: raise ValueError('Code/data/config changed: use new output')
    if not file.exists(): atomic_json(file,manifest)
    if verified_complete(root,60): sync_registry(); return dict(complete=True)
    def report(stage,**details):
        record=dict(details)
        if 'elapsed_seconds' in record:
            record['stage_elapsed_seconds']=record.pop('elapsed_seconds')
        record.update(stage=stage,fold=args.fold,elapsed_seconds=time.monotonic()-started)
        atomic_json(root/'status.json',record); print(json.dumps(record),flush=True)
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('NVIDIA CUDA required')
    train_cases=[]; cal_cases=[]
    candidate=root/'variant_candidate.joblib'
    if not candidate.exists():
        # Allocation guard counts final compact files and retains 3 GiB reserve.
        disk_matches = (*train, calibration) if calibration_storage == 'disk' else train
        frames=sum(sum(len(y) for _,y in load(paths[m])) for m in disk_matches)
        required=frames*len(SCENARIOS)*203*2*2
        existing_folders = [root/'compact_features/train']
        if calibration_storage == 'disk':
            existing_folders.append(root/'compact_features/calibration')
        existing=sum(p.stat().st_size for folder in existing_folders for p in folder.rglob('*.npy'))
        if shutil.disk_usage(root).free < max(0,required-existing)+3*1024**3:
            raise RuntimeError('Insufficient safe disk; existing evidence left untouched')
        try:
            train_cases=feature_cases(paths,train,list(SCENARIOS),331,root/'compact_features/train',report)
            if calibration_storage == 'stream':
                report('calibration_streaming_enabled', match=calibration, seed=348)
                cal_cases=StreamingCalibrationCases(paths,[calibration],list(SCENARIOS),348)
            else:
                cal_cases=feature_cases(paths,[calibration],list(SCENARIOS),348,root/'compact_features/calibration',report)
            baseline=joblib.load(source/'tcn_candidate.joblib')
            torch.manual_seed(config.seed); model=DeepTemporal(baseline)
            def validate(current,epoch):
                report('calibration_epoch',epoch=epoch,match=calibration)
                value=calibration_value(current,cal_cases,baseline.decoder)
                atomic_json(root/f'calibration_epoch_{epoch}.json',dict(epoch=epoch,spellF1=value,
                    match=calibration,purpose='checkpoint_selection_calibration_only'))
                return value
            def progress(event):
                stage=event.pop('stage'); report(stage,**event)
            model.fit(train_cases,config,root/'training.pt',validate,progress,args.device,args.stop_after_epoch)
            if len(model.history)<config.epochs:
                report('pilot_checkpoint_saved',epoch=len(model.history),history=model.history)
                return dict(complete=False,pilot=True,history=model.history)
            # Persist last model for audit; best epoch was selected only by calibration.
            save_model(root/'last_uncalibrated.joblib',model)
            model=joblib.load(root/'best_uncalibrated.joblib')
            report('calibrating_best',best_epoch=model.best_epoch,match=calibration)
            model=fit_calibration(model,(([(features,y)],'calibration') for features,y,_ in cal_cases),root,'deep')
            save_model(candidate,model)
        finally:
            close_cases(train_cases); close_cases(cal_cases)
        del train_cases,cal_cases,baseline; gc.collect()
    model=joblib.load(candidate); trees=joblib.load(source/'tree_candidate.joblib'); temporal=joblib.load(source/'tcn_candidate.joblib')
    ensemble=Mixture(trees,model,.5,.5,trees.decoder)
    save_model(root/'ensemble_candidate.joblib',ensemble)
    rows=[]; checkpoints=[]
    for match in tests:
        for scenario in SCENARIOS:
            rowfile=root/f'rows_{match}_{scenario}.json'; checkpoints.append(rowfile.name)
            if rowfile.exists(): records=validate_rows(json.loads(rowfile.read_text()),METHODS,args.fold,match,scenario,432)
            else:
                report('external_evaluation',match=match,scenario=scenario)
                truth=[]; predictions={m:[] for m in METHODS}
                for data,y in views(load(paths[match]),scenario,432):
                    features,_=pair(data,True)
                    p,q=trees.predict(features); tp,tq=temporal.predict(features); vp,vq=model.predict(features)
                    for pp,qq in ((p,q),(tp,tq),(vp,vq)): validate_probabilities(pp,qq,len(y))
                    values=[decode(tp,tq,temporal.decoder),decode((p+tp)/2,(q+tq)/2,trees.decoder),
                        decode(vp,vq,model.decoder),decode((p+vp)/2,(q+vq)/2,trees.decoder)]
                    truth.append(y)
                    for name,value in zip(METHODS,values): predictions[name].append(value)
                    del features; gc.collect()
                meta=dict(fold=args.fold,split='development' if match in DEV else 'known_retest',match=match,scenario=scenario,seed=432)
                records=[score(truth,predictions[m],method=m,**meta) for m in METHODS]
                validate_rows(records,METHODS,args.fold,match,scenario,432); atomic_json(rowfile,records)
            rows.extend(records); write_csv(root/'performa_detail.csv',rows); write_csv(root/'ringkasan_performa.csv',aggregate(rows))
            sync_registry()
    if len(rows)!=60: raise ValueError('Incomplete evaluation')
    finish(root,['manifest.json','variant_candidate.joblib','ensemble_candidate.joblib','training.pt',
        'calibration_deep.json','decoder_deep.json','performa_detail.csv','ringkasan_performa.csv',*checkpoints],
        rows=60,summary=aggregate(rows),elapsed_seconds=time.monotonic()-started)
    sync_registry(); report('complete',rows=60,summary=aggregate(rows))
    return dict(complete=True,summary=aggregate(rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('outputs/riset_v44_full'))
    parser.add_argument('--cache',type=Path,nargs='+',required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fold',type=int,choices=range(5),default=0); parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--steps',type=int,default=160); parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--validation-every',type=int,default=3); parser.add_argument('--stop-after-epoch',type=int)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--calibration-storage', choices=['disk','stream'], default='disk')
    args=parser.parse_args()
    if min(args.epochs,args.steps,args.batch,args.validation_every)<1: parser.error('Positive training configuration required')
    run(args)


if __name__=='__main__': main()
