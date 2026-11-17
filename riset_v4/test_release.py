"""Synthetic release safety checks; no target annotations or licensed assets."""
import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import joblib
import numpy as np
from .decode import Settings
from .release import run, trusted_model


class ReleaseTests(unittest.TestCase):
    def test_hash_checked_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); payload = io.BytesIO()
            joblib.dump(dict(synthetic=True), payload); raw = payload.getvalue()
            (folder/'model.b64').write_bytes(base64.b64encode(raw)+b'\n')
            spec = dict(file='model.b64', bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            self.assertEqual(trusted_model(folder,spec),dict(synthetic=True))
            spec['sha256'] = '0'*64
            with patch('riset_v4.release.joblib.load') as loader:
                with self.assertRaisesRegex(ValueError,'integrity'): trusted_model(folder,spec)
                loader.assert_not_called()

    def test_existing_csv_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory); target=folder/'original.csv'; target.write_bytes(b'preserve')
            with patch('riset_v4.release.trusted_model') as loader:
                with self.assertRaises(FileExistsError): run(folder,target)
                loader.assert_not_called()
            self.assertEqual(target.read_bytes(),b'preserve')

    def synthetic_run(self, folder, wrong_hash=False, pending=False):
        import games
        from starter.baseline import write_submission
        folder=Path(folder); models=folder/'models/equal5'; models.mkdir(parents=True)
        expected=folder/'expected.csv'; write_submission(str(expected),np.array(list('DHHHDDHHDD')))
        digest=hashlib.sha256(expected.read_bytes()).hexdigest()
        manifest=dict(folds=[dict(fold=i,tree_weight=0,deep={}) for i in range(5)],
            target_manual_labels_used=False,specialist_weight=0,feature_columns=203,
            decoder=vars(Settings()),expected_csv_sha256='bad' if wrong_hash else digest)
        (models/'manifest.json').write_text(json.dumps(manifest))
        target=folder/'new.csv'
        if pending: target.with_name('new.csv.pending').write_bytes(b'preserve-pending')
        class Identity:
            def state(self,x): return x
            def edge(self,x): return x
        class Deep:
            calibration=Identity()
            def raw(self,features,chunk,device):
                n=len(features[0]); return np.tile([.99,.005,.005],(n,1)).astype(np.float32),np.zeros((n,6),np.float32)
        halves=[(1,1,4,3,dict(atd_meta_sha256='synthetic',atd_raw_sha256='synthetic')),
                (2,6,8,2,dict(atd_meta_sha256='synthetic',atd_raw_sha256='synthetic'))]
        def features(n): return (np.zeros((n,203),np.float32),)*2,list(range(203))
        with patch('riset_v4.release.ROOT',folder), \
             patch.object(games,'GAMES',{games.GAME:dict(n_video_frames=10)}), \
             patch('riset_v4.release.trusted_model',return_value=Deep()), \
             patch('riset_v4.release.local_tracking',return_value=halves), \
             patch('riset_v4.release.pair',side_effect=features), \
             contextlib.redirect_stdout(io.StringIO()):
            if wrong_hash or pending:
                with self.assertRaises((ValueError,FileExistsError)): run(folder,target)
                self.assertFalse(target.exists())
            else:
                receipt=run(folder,target)
                self.assertTrue(receipt['byte_identical']); self.assertEqual(target.read_bytes(),expected.read_bytes())
                self.assertFalse(receipt['smart_tagging_read'])
                self.assertFalse(receipt['target_manual_labels_used'])
        return target

    def test_synthetic_end_to_end_two_halves(self):
        with tempfile.TemporaryDirectory() as directory: self.synthetic_run(directory)

    def test_mismatching_output_not_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            target=self.synthetic_run(directory,wrong_hash=True)
            self.assertTrue(target.with_name('new.csv.pending').exists())

    def test_existing_pending_file_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            target=self.synthetic_run(directory,pending=True)
            self.assertEqual(target.with_name('new.csv.pending').read_bytes(),b'preserve-pending')


if __name__=='__main__': unittest.main()
