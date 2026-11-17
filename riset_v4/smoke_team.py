"""End-to-end synthetic smoke for all four teammate tracks; never a score claim."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch
import joblib
import numpy as np
from .data import DEV, RETEST, fingerprint, split
from .decode import Settings
from .evaluate import atomic_json
from .experiment_io import verified_complete
from .features import pair
from .models import Heads
from .tcn import Temporal
from .test_v4 import fixture
from . import search_decoder, team_training
from .team_results import export_results, import_results


def interrupted_resume(module, args):
    original=module.score; calls=0
    def interrupt(*values, **kwargs):
        nonlocal calls
        calls+=1
        if calls==5: raise InterruptedError('Synthetic interruption after first saved scenario')
        return original(*values, **kwargs)
    with patch.object(module,'score',side_effect=interrupt):
        try: module.run(args)
        except InterruptedError: pass
        else: raise AssertionError('Expected partial-run interruption')
    checkpoints=sorted(args.output.glob('rows_*.json'))
    assert len(checkpoints)==1 and not (args.output/'result.json').exists()
    before={p.name:p.read_bytes() for p in checkpoints}
    module.run(args)
    assert all((args.output/name).read_bytes()==data for name,data in before.items())
    module.run(args)


def main():
    started=time.monotonic(); checks=[]
    with tempfile.TemporaryDirectory(prefix='team_smoke_') as temporary:
        root=Path(temporary); cache=root/'cache'; cache.mkdir(); source=root/'source'; source.mkdir()
        data,y=fixture(600); features,_=pair(data,True)
        for match in DEV+RETEST:
            arrays={'pitch':data.pitch}
            for half in (0,1):
                arrays.update({f'ball{half}':data.ball,f'home{half}':data.players[0],
                    f'away{half}':data.players[1],f'truth{half}':y})
            np.savez(cache/(match+'.npz'),**arrays)
        fold=source/'fold_0'; fold.mkdir(); train,calibration,tests=split(0)
        atomic_json(fold/'split.json',dict(train=list(train),calibration=calibration,test=list(tests)))
        tree=Heads().fit([(features,y,1.)],iterations=2); tree.decoder=Settings()
        temporal=Temporal().fit([(features,y,1.)],epochs=1,steps=1,device='cpu',seed=331,
            checkpoint=fold/'tcn_training.pt',progress=lambda event:None); temporal.decoder=Settings()
        joblib.dump(tree,fold/'tree_candidate.joblib'); joblib.dump(temporal,fold/'tcn_candidate.joblib')
        atomic_json(source/'manifest.json',dict(seed=331,iterations=90,epochs=1,steps=1,
            train_scenarios=team_training.BASE_SCENARIOS,
            cache_sha256={m:fingerprint(cache/(m+'.npz')) for m in DEV+RETEST},
            synthetic_test_only=True,actual_synthetic_tree_iterations=2,
            note='Reduced synthetic fixture, NOT a frozen benchmark or measured research result'))
        common=dict(source=source,cache=[cache],fold=0,device='cpu',min_free_gib=0.)
        with patch.object(search_decoder,'sync_registry'), patch.object(team_training,'sync_registry'), \
             patch.object(team_training,'BASE_TREE',dict(team_training.BASE_TREE,max_iter=2)), \
             patch.object(team_training,'CAPACITY_TREE',dict(team_training.CAPACITY_TREE,max_iter=3)):
            args=SimpleNamespace(**common,output=root/'outputs/rekan_1_smoke_fold_0',seed=731,candidates=2)
            interrupted_resume(search_decoder,args)
            assert verified_complete(args.output,30)
            checks.append('decoder_partial_and_complete_resume')
            for member,track in enumerate(('tcn','capacity','augment'),2):
                args=SimpleNamespace(**common,output=root/f'outputs/rekan_{member}_smoke_fold_0',track=track,epochs=2)
                interrupted_resume(team_training,args)
                assert verified_complete(args.output,60)
                candidate=joblib.load(args.output/'variant_candidate.joblib')
                assert type(candidate).__module__!='__main__'
                # A fresh interpreter must load the trained candidate, not just metadata.
                command='import joblib,sys; m=joblib.load(sys.argv[1]); assert type(m).__module__!="__main__"; print("LOAD_OK")'
                result=subprocess.run([sys.executable,'-c',command,str(args.output/'variant_candidate.joblib')],
                    cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,check=True)
                assert 'LOAD_OK' in result.stdout
                checks.append(track+'_partial_complete_resume_fresh_process_load')
            with patch('riset_v4.team_results.sync_registry'):
                for member in range(1,5):
                    archive=root/f'member_{member}.zip'
                    export_results(root,archive,member,'smoke')
                    first=import_results(archive,root/'receiver')
                    second=import_results(archive,root/'receiver')
                    assert first==second
            checks.append('four_real_output_structures_transfer_and_idempotent_import')
            # A code/config mismatch must fail, not overwrite completed outputs.
            args.epochs=99; args.device='auto'
            try: team_training.run(args)
            except ValueError: checks.append('changed_configuration_rejected')
            else: raise AssertionError('Changed completed configuration was accepted')
    print(json.dumps(dict(complete=True,synthetic_test_only=True,checks=checks,
        target_gt_read=False,elapsed_seconds=time.monotonic()-started),indent=2))


if __name__=='__main__': main()
