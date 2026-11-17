"""Eksperimen terpisah dari paket tim 1--15, dengan checkpoint dan CSV."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import os
import time
from types import SimpleNamespace
from importlib.metadata import version
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from . import VERSION
from .data import DEV,RETEST,SCENARIOS,discover,load,split,views,fingerprint
from .features import pair
from .models import Heads,calibrate,calibration_report,aligned
from .decode import decode,Settings
from .evaluate import atomic_json,write_csv,score,aggregate
from night_model import feature_pair,STRIDE
from explore_model import expand,fit_head,adaptive,Settings as LegacySettings
from transition_model import fit_switch,switch_probability,decode_context,gap_lengths


def log(root,stage,**details):
    event=dict(time=datetime.now(timezone.utc).isoformat(),stage=stage,**details)
    print(json.dumps(event),flush=True);atomic_json(root/'status.json',event)


def save_model(path,model):
    temp=path.with_suffix('.tmp');joblib.dump(model,temp,compress=3);temp.replace(path)


def legacy_features(d):return expand(d,feature_pair(d,True))


def legacy_prob(model,features):
    return (aligned(model,features[0],3)+aligned(model,features[1],3)[:,[1,0,2]])/2


def build_training(paths,train,scenarios,seed,root):
    cases=[];lx=[];ly=[];lw=[]
    cache=root/'feature_cache';cache.mkdir(exist_ok=True)
    for match in train:
        halves=load(paths[match])
        for scenario in scenarios:
            log(root,'training_features',match=match,scenario=scenario)
            for h,(d,y) in enumerate(views(halves,scenario,seed)):
                names=[cache/f'{match}_{scenario}_{seed}_{h}_{side}.npy' for side in range(2)]
                if not all(path.exists() for path in names):
                    features,_=pair(d,True)
                    for path,x in zip(names,features):
                        temp=path.with_suffix('.tmp')
                        with temp.open('wb') as f:np.save(f,x)
                        temp.replace(path)
                    del features,x
                disk=tuple(np.load(path,mmap_mode='r') for path in names)
                cases.append((disk,y,SCENARIOS[scenario]))
                old=legacy_features(d)
                grid=np.minimum(np.arange(0,len(y),STRIDE)+STRIDE//2,len(y)-1)
                idx=np.flatnonzero(y[grid]<3)[::2];truth=y[grid[idx]]
                for side in range(2):
                    lx.append(old[side][idx]);ly.append(truth if side==0 else np.array([1,0,2])[truth])
                    lw.append(np.full(len(idx),SCENARIOS[scenario]))
    return cases,(np.concatenate(lx),np.concatenate(ly),np.concatenate(lw))


def calibration_cases(path,scenarios,seed):
    halves=load(path)
    for scenario in scenarios:
        for d,y in views(halves,scenario,seed):
            yield [(pair(d,True)[0],y)],scenario


def fit_calibration(model,cases,root,name):
    rawp,rawq,truth=[],[],[]
    for halves,_ in cases:
        for features,y in halves:
            p,q=model.raw(features);rawp.append(p);rawq.append(q);truth.append(y)
    model.calibration=calibrate(rawp,rawq,truth)
    p=[model.calibration.state(v) for v in rawp];q=[model.calibration.edge(v) for v in rawq]
    report=calibration_report((rawp,rawq),(p,q),truth)
    atomic_json(root/f'calibration_{name}.json',report)
    # Tiny predeclared search, confined to the calibration match.
    from night_benchmark import combine
    from evaluation.possession_metrics import evaluate
    best=None; candidates=[]
    for threshold in (.025,.075,.15):
        for gain in (1.,2.):
            setting=Settings(boundary_gain=gain,proposal_threshold=threshold)
            pred=[decode(a,b,setting) for a,b in zip(p,q)]
            value=evaluate(combine(truth),combine(pred),taus=(),bootstrap_n=0)['spell_f1_mean']
            candidates.append(dict(**asdict(setting),calibration_spellF1=value))
            if best is None or value>best[0]:best=(value,setting)
    atomic_json(root/f'decoder_{name}.json',{'chosen':asdict(best[1]),'candidates':candidates,
        'selection_source':'calibration match only; repeated scenarios are not independent matches'})
    model.decoder=best[1]
    return model


def benchmark(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    run_lock=(root/'benchmark.lock').open('a+b')
    if run_lock.seek(0,2)==0:run_lock.write(b'0');run_lock.flush()
    run_lock.seek(0)
    if os.name=='nt':
        import msvcrt
        try:msvcrt.locking(run_lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise RuntimeError('A benchmark is already using this output directory')
    else:
        import fcntl
        fcntl.flock(run_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    paths=discover(args.cache)
    files=[Path(__file__).parent/n for n in ('__init__.py','data.py','features.py','labels.py','models.py','decode.py','tcn.py','evaluate.py','run.py')]
    files+=sorted((Path(__file__).resolve().parents[1]/'eksperimen'/'pipeline').glob('*.py'))
    files+=[Path(__file__).resolve().parents[1]/'evaluation'/'possession_metrics.py']
    codehash=hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in files)).hexdigest()
    manifest=dict(version=VERSION,code_sha256=codehash,cache_sha256={m:fingerprint(p) for m,p in paths.items()},
                  folds=args.folds,seed=args.seed,train_scenarios=args.train_scenarios,
                  test_scenarios=list(SCENARIOS),iterations=args.iterations,epochs=args.epochs,steps=args.steps,
                  device=args.device,python=platform.python_version(),
                  packages={n:version(n) for n in ('numpy','scipy','scikit-learn','joblib','numba','torch')},
                  note='All seven matches were inspected in earlier research. This is a controlled retest, not a new independent holdout.')
    existing=root/'manifest.json'
    if existing.exists() and json.loads(existing.read_text(encoding='utf-8'))!=manifest:
        raise ValueError('Output belongs to a different code/data/config version; use a new --output directory')
    atomic_json(existing,manifest)
    all_rows=[]
    def refresh():
        if all_rows:
            write_csv(root/'performa_detail.csv',all_rows)
            write_csv(root/'ringkasan_performa.csv',aggregate(all_rows))
    for fold in args.folds:
        train,cal,tests=split(fold);folder=root/f'fold_{fold}';folder.mkdir(exist_ok=True)
        atomic_json(folder/'split.json',dict(train=train,calibration=cal,test=tests))
        model_path=folder/'models.joblib'
        if model_path.exists():
            log(root,'loading_checkpoint',fold=fold);models=joblib.load(model_path)
        else:
            checkpoint=folder/'trained_before_calibration.joblib'
            if checkpoint.exists():
                baseline,plain,weighted,temporal,switch=joblib.load(checkpoint)
            else:
                cases,old=build_training(paths,train,args.train_scenarios,args.seed,root)
                def staged(name,fit):
                    path=folder/(name+'.joblib')
                    if path.exists():return joblib.load(path)
                    log(root,'fitting_'+name,fold=fold)
                    result=fit();save_model(path,result);return result
                baseline=staged('legacy_baseline',lambda:fit_head(*old,max_iter=args.iterations))
                del old;gc.collect()
                # Legacy switch uses exactly the same training matches and scenarios.
                # The original 110-column representation is preserved at stride five.
                items=[SimpleNamespace(features=[tuple(x[::STRIDE,:110] for x in features)],
                       truth=[y],scenario=i) for i,(features,y,w) in enumerate(cases)]
                switch=staged('switch_baseline',lambda:fit_switch(items,{i:c[2] for i,c in enumerate(cases)}))
                del items;gc.collect()
                plain=staged('relative_unweighted',lambda:Heads(weighted=False).fit(cases,args.iterations,
                    work_dir=folder/'plain_work',fit_edges=False))
                weighted=staged('relative_weighted',lambda:Heads(weighted=True).fit(cases,args.iterations,
                    work_dir=folder/'weighted_work'))
                temporal=None
                if args.epochs:
                    from .tcn import Temporal
                    temporal=staged('temporal_uncalibrated',lambda:Temporal().fit(cases,
                        epochs=args.epochs,steps=args.steps,device=args.device,seed=args.seed,
                        checkpoint=folder/'tcn_training.pt',
                        progress=lambda v:log(root,'training_tcn',fold=fold,**v)))
                save_model(checkpoint,(baseline,plain,weighted,temporal,switch))
                for features,_,_ in cases:
                    for x in features:x._mmap.close()
                del cases;gc.collect()
            log(root,'calibrating',fold=fold,match=cal)
            weighted=fit_calibration(weighted,calibration_cases(paths[cal],args.train_scenarios,args.seed+17),folder,'trees')
            if temporal is not None:
                temporal=fit_calibration(temporal,calibration_cases(paths[cal],args.train_scenarios,args.seed+17),folder,'tcn')
            gc.collect()
            models={'legacy_refit':baseline,'switch_refit':switch,'relative':plain,'weighted':weighted,'tcn':temporal}
            save_model(model_path,models)
            save_model(folder/'tree_candidate.joblib',weighted)
            if temporal is not None:
                from .models import Ensemble
                save_model(folder/'tcn_candidate.joblib',temporal)
                save_model(folder/'ensemble_candidate.joblib',Ensemble(weighted,temporal))
            # These are reproducible intermediate arrays created by this runner.
            # Finished model checkpoints survive; later folds need no stale mapped arrays.
            for cachefile in (root/'feature_cache').glob('*.npy'):cachefile.unlink()
        baseline,plain,weighted,temporal,switch=(models[k] for k in ('legacy_refit','relative','weighted','tcn','switch_refit'))
        for match in tests:
            path=folder/f'{match}_rows.json'
            if path.exists():
                all_rows.extend(json.loads(path.read_text()));refresh();continue
            halves=load(paths[match]);rows=[]
            for scenario in SCENARIOS:
                rowpath=folder/f'rows_{match}_{scenario}.json'
                if rowpath.exists():
                    rows.extend(json.loads(rowpath.read_text()));continue
                log(root,'evaluating',fold=fold,match=match,scenario=scenario)
                truth=[];predictions={};arrays={}
                for h,(d,y) in enumerate(views(halves,scenario,args.seed+101)):
                    truth.append(y);features,_=pair(d,True)
                    old=tuple(x[::STRIDE,:110] for x in features)
                    lp=legacy_prob(baseline,old)
                    sp=switch_probability(switch,old)
                    p0=plain.state_probability(features);p,q=weighted.raw(features)
                    cp,cq=weighted.calibration.state(p),weighted.calibration.edge(q)
                    pred={'legacy_refit':adaptive(lp,len(y),LegacySettings(2,4,-.18)),
                          'switch_refit':decode_context(lp,gap_lengths(d.ball),'switch',sp),
                          'relative_25fps':decode(p0), 'episode_weighted':decode(p),
                          'boundary_raw':decode(p,q,Settings()),
                          'boundary_calibrated':decode(cp,cq,weighted.decoder)}
                    if temporal is not None:
                        tp,tq=temporal.predict(features)
                        pred['tcn_dual']=decode(tp,tq,temporal.decoder)
                        pred['ensemble']=decode((cp+tp)/2,(cq+tq)/2,weighted.decoder)
                    for name,value in pred.items():
                        predictions.setdefault(name,[]).append(value);arrays[f'{name}_{h}']=value
                    arrays[f'truth_{h}']=y
                    del features,old;gc.collect()
                target=folder/f'predictions_{match}_{scenario}.npz'
                temp=target.with_suffix('.tmp')
                with temp.open('wb') as f:np.savez_compressed(f,**arrays)
                temp.replace(target)
                scenario_rows=[]
                for name,pred in predictions.items():
                    row=score(truth,pred,fold=fold,split='development' if match in DEV else 'known_retest',
                              match=match,scenario=scenario,seed=args.seed+101,method=name)
                    scenario_rows.append(row);print(json.dumps(row),flush=True)
                atomic_json(rowpath,scenario_rows);rows.extend(scenario_rows)
                write_csv(root/'performa_detail.csv',all_rows+rows)
                write_csv(root/'ringkasan_performa.csv',aggregate(all_rows+rows))
            atomic_json(path,rows);all_rows.extend(rows);refresh()
        del models;gc.collect()
    summary=aggregate(all_rows)
    atomic_json(root/'result.json',{'complete':True,'summary':summary,'rows':len(all_rows),
        'development':aggregate([r for r in all_rows if r['split']=='development']),
        'known_retest':aggregate([r for r in all_rows if r['split']=='known_retest']),
        'promoted':False,'reason':'External research candidates; target score remains unknown.'})
    log(root,'complete',rows=len(all_rows),best=summary[0])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--folds',type=int,nargs='+',default=[0])
    parser.add_argument('--seed',type=int,default=331)
    parser.add_argument('--train-scenarios',nargs='+',choices=list(SCENARIOS),default=['original','camera_gaps','drift'])
    parser.add_argument('--iterations',type=int,default=90)
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--steps',type=int,default=180)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=parser.parse_args()
    if any(f<0 or f>4 for f in args.folds) or len(set(args.folds))!=len(args.folds):parser.error('folds must be distinct values 0..4')
    if min(args.iterations,args.steps)<1 or args.epochs<0:parser.error('Invalid training budget')
    with threadpool_limits(limits=4):benchmark(args)


if __name__=='__main__':main()

