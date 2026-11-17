"""Isolated target candidates from completed folds. Never promote or read target labels."""
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
import torch
from .artifact_checks import validate_probabilities
from .data import fingerprint
from .decode import Settings, decode
from .evaluate import atomic_json
from .experiment_io import verified_complete
from .fusion_model import combine_probabilities, specialist_probability
from .fusion_validation import source_model_digest, validate_bundle


def predict(bundle, runs, output, deep_root, source):
    """Fixed equal-weight fold ensemble; not itself a measured external holdout score."""
    start=time.monotonic(); bundle=Path(bundle); output=Path(output)
    m=validate_bundle(bundle)
    specifications=[]; source_code_audit={}
    for run in runs:
        run=Path(run)
        if not verified_complete(run,75): raise ValueError('Fold incomplete/modified: '+str(run))
        rm=json.loads((run/'manifest.json').read_text())
        fold=rm['folds'][0]
        for name,digest in rm['code_sha256'].items():
            path=Path(__file__).resolve().parents[1]/name
            if not path.is_file() or fingerprint(path)!=digest:
                raise ValueError('Frozen experiment source differs: '+name)
        source_code_audit[str(fold)]=True
        selection=json.loads((run/'calibration_selection.json').read_text())
        dp=Path(deep_root)/f'local_deep_v45_fold_{fold}/variant_candidate.joblib'
        tp=Path(source)/f'fold_{fold}/tree_candidate.joblib'
        if (fingerprint(dp)!=source_model_digest(rm,'variant_candidate.joblib')
                or fingerprint(tp)!=source_model_digest(rm,'tree_candidate.joblib')):
            raise ValueError('Frozen source model differs')
        specifications.append((fold,run,selection,dp,tp))
    if sorted(s[0] for s in specifications)!=list(range(5)):
        raise ValueError('Fixed ensemble requires exactly folds 0..4, no test-winning fold selection')
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    methods=('deep_v45_equal5','calibration_fusion_v46_equal5')
    n=m['frames']; predictions={name:np.full(n,2,np.int8) for name in methods}
    probabilities={name:np.tile([0.,0.,1.],(n,1)).astype(np.float32) for name in methods}
    boundaries={name:np.zeros((n,6),np.float32) for name in methods}
    receipt=dict(complete=False,frames=n,fps=m['fps'],target_labels_used=False,
        submission_changed=False,promoted=False,cue_weight=None,
        feature_bundle_sha256=fingerprint(bundle/'manifest.json'),
        input_precision='float16 bundle converted to float32; external experiment evaluated float32 extraction',
        ensemble='fixed arithmetic mean of 5 calibration-selected folds; not evaluated as an ensemble on external labels',
        cue_columns_used_for_prediction=False,
        experiment_source_hashes_verified=source_code_audit,
        outside_half_policy='D placeholder outside official playing halves; not model-inferred',
        bundle_source_code_matches_current={name:fingerprint(Path(__file__).with_name(name))==digest
            for name,digest in m.get('code_sha256',{}).items()
            if Path(name).name==name and Path(__file__).with_name(name).is_file()},
        note='Cue columns joined but excluded from learned prediction pending external aligned validation.',
        folds=[dict(fold=f,run=str(r),result_sha256=fingerprint(r/'result.json'),
                    choice=s['chosen']) for f,r,s,_,_ in specifications])
    atomic_json(output/'manifest.json',receipt)
    for half in m['halves']:
        first,end=half['first_frame'],half['end_frame_exclusive']
        arrays=[np.load(bundle/f['name'],mmap_mode='r',allow_pickle=False) for f in half['files']]
        if len(arrays)!=2 or any(a.shape!=(end-first,290) for a in arrays):
            raise ValueError('Expected two 290-column feature sides')
        features=tuple(np.asarray(a[:,:203],dtype=np.float32) for a in arrays)
        expanded=tuple(np.asarray(a[:,:281],dtype=np.float32) for a in arrays)
        sums={name:[np.zeros((end-first,3),np.float32),np.zeros((end-first,6),np.float32)] for name in methods}
        # Decode averaged probabilities with one fixed decoder, not test-tuned voting/spell editing.
        for fold,run,selection,dp,tp in specifications:
            deep,tree=joblib.load(dp),joblib.load(tp)
            rawp,rawq=deep.raw(features,chunk=8192)
            p,q=deep.calibration.state(rawp),deep.calibration.edge(rawq)
            choice=selection['chosen']; tw=choice['tree_weight']; sw=choice['specialist_weight']
            if tw>0: tpv,tqv=tree.predict(features)
            else: tpv,tqv=p,q
            if sw>0:
                name=choice['specialist']; specialist=joblib.load(run/f'{name}_specialist.joblib')
                sp=specialist_probability(specialist,features if name=='old' else expanded,
                                           selection['temperatures'][name])
            else: sp=p
            cp,cq=combine_probabilities((p,q),(tpv,tqv),sp,tw,sw)
            for name,pp,qq in [(methods[0],p,q),(methods[1],cp,cq)]:
                sums[name][0]+=pp/5; sums[name][1]+=qq/5
            print(json.dumps(dict(stage='target_half_fold',half=half['half'],fold=fold,
                                  elapsed_seconds=time.monotonic()-start)),flush=True)
        # Prespecified neutral decoder. Per-fold decoder search is NOT re-fit on target.
        for name in methods:
            p,q=sums[name]; validate_probabilities(p,q,end-first)
            probabilities[name][first:end]=p; boundaries[name][first:end]=q
            predictions[name][first:end]=decode(p,q,Settings())
        for a in arrays: a._mmap.close()
    from starter.baseline import write_submission
    from evaluation.possession_metrics import read_label_csv,validate_submission
    for name in methods:
        path=output/(name+'.csv')
        write_submission(str(path),np.asarray(['H','A','D'])[predictions[name]])
        error=validate_submission(read_label_csv(str(path)),n)
        if error: raise ValueError(error)
        np.savez_compressed(output/(name+'.probabilities.npz'),probabilities=probabilities[name],boundaries=boundaries[name])
    receipt.update(complete=True,elapsed_seconds=time.monotonic()-start,decoder=vars(Settings()),
        metrics=dict(spellF1=None,macroF1=None),
        outputs_sha256={p.name:fingerprint(p) for p in output.iterdir() if p.suffix in ('.csv','.npz')},
        code_sha256={name:fingerprint(Path(__file__).with_name(name)) for name in
            ('predict_fusion.py','fusion_validation.py','fusion_model.py','decode.py')})
    atomic_json(output/'manifest.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--deep-root',type=Path,default=Path('outputs'))
    parser.add_argument('--source',type=Path,default=Path('outputs/riset_v44_full'))
    args=parser.parse_args(); predict(args.bundle,args.runs,args.output,args.deep_root,args.source)


if __name__=='__main__': main()
