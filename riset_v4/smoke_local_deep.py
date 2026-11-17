"""Full synthetic local-deep pipeline, isolated from competition results."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
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
from . import local_deep


def main(calibration_storage='disk'):
    with tempfile.TemporaryDirectory(prefix='local_deep_smoke_') as temp:
        root=Path(temp); cache=root/'cache'; cache.mkdir(); source=root/'source'; source.mkdir()
        data,y=fixture(600); features,_=pair(data,True)
        for match in DEV+RETEST:
            arrays={'pitch':data.pitch}
            for half in (0,1):
                arrays.update({f'ball{half}':data.ball,f'home{half}':data.players[0],
                    f'away{half}':data.players[1],f'truth{half}':y})
            np.savez(cache/(match+'.npz'),**arrays)
        folder=source/'fold_0'; folder.mkdir(); train,cal,tests=split(0)
        atomic_json(folder/'split.json',dict(train=list(train),calibration=cal,test=list(tests)))
        tree=Heads().fit([(features,y,1.)],iterations=2); tree.decoder=Settings()
        temporal=Temporal().fit([(features,y,1.)],epochs=1,steps=1,device='cpu',progress=lambda e:None)
        temporal.decoder=Settings()
        joblib.dump(tree,folder/'tree_candidate.joblib'); joblib.dump(temporal,folder/'tcn_candidate.joblib')
        atomic_json(source/'manifest.json',dict(seed=331,epochs=5,synthetic_test_only=True,
            actual_synthetic_epochs=1,cache_sha256={m:fingerprint(cache/(m+'.npz')) for m in DEV+RETEST}))
        args=SimpleNamespace(source=source,cache=[cache],output=root/'output',fold=0,epochs=1,
            steps=1,batch=1,validation_every=1,device='cuda',stop_after_epoch=None,
            calibration_storage=calibration_storage)
        with patch.object(local_deep,'sync_registry'):
            result=local_deep.run(args); assert result['complete']; assert verified_complete(args.output,60)
            local_deep.run(args)
        code='import joblib,sys; x=joblib.load(sys.argv[1]); assert type(x).__module__=="riset_v4.deep_temporal"; print("FRESH_LOAD_OK")'
        proc=subprocess.run([sys.executable,'-c',code,str(args.output/'variant_candidate.joblib')],
            cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,check=True)
        assert 'FRESH_LOAD_OK' in proc.stdout
        if calibration_storage == 'stream':
            assert not (args.output/'compact_features/calibration').exists()
        print(json.dumps(dict(complete=True,synthetic_test_only=True,rows=60,
            cuda_training=True,calibration=True,fresh_process_load=True,complete_resume=True,
            real_ledger_touched=False,target_gt_read=False,calibration_storage=calibration_storage)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration-storage',choices=['disk','stream'],default='disk')
    main(parser.parse_args().calibration_storage)
