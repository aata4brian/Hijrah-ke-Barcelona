import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from .cue_features import CUE_NAMES
from .data import fingerprint
from .fusion_features import SCHEMA
from .fusion_validation import source_model_digest, validate_bundle, expected_feature_names
from .predict_fusion import predict
from .report_fusion import report


class FusionArtifactTests(unittest.TestCase):
    def test_source_hash_does_not_depend_on_dictionary_order(self):
        m = dict(source_models_sha256={r'C:\models\tree_candidate.joblib':'tree',
                                      'other/variant_candidate.joblib':'deep'})
        self.assertEqual(source_model_digest(m,'variant_candidate.joblib'),'deep')
        m['source_models_sha256']['duplicate/variant_candidate.joblib']='other'
        with self.assertRaises(ValueError): source_model_digest(m,'variant_candidate.joblib')

    def bundle(self, folder):
        files=[]
        for side in (0,1):
            p=folder/f'half_1_side_{side}.npy'
            np.save(p,np.zeros((8,290),np.float16))
            files.append(dict(name=p.name,sha256=fingerprint(p),shape=[8,290]))
        m=dict(complete=True,target_labels_used=False,smart_tagging_read=False,schema=SCHEMA,
            frames=10,fps=25,feature_names=expected_feature_names(),
            halves=[dict(half=1,first_frame=1,end_frame_exclusive=9,files=files)])
        (folder/'manifest.json').write_text(json.dumps(m))
        return m

    def test_bundle_contract_rejects_bad_alignment_schema_and_hash(self):
        import games
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(games,'GAMES',{games.GAME:dict(n_video_frames=10,fps=25)}), \
             patch.object(games,'half_video_blocks',return_value=[(1,1,8)]):
            folder=Path(directory); original=self.bundle(folder)
            self.assertTrue(validate_bundle(folder)['complete'])
            for fault in ('empty_half','shifted','swapped_side','schema','missing_label_gate','modified'):
                m=json.loads(json.dumps(original))
                if fault=='empty_half': m['halves']=[]
                elif fault=='shifted': m['halves'][0]['first_frame']=0
                elif fault=='swapped_side': m['halves'][0]['files'].reverse()
                elif fault=='schema': m['feature_names'].reverse()
                elif fault=='missing_label_gate': del m['target_labels_used']
                elif fault=='modified': m['halves'][0]['files'][0]['sha256']='bad'
                (folder/'manifest.json').write_text(json.dumps(m))
                with self.subTest(fault=fault), self.assertRaises(ValueError): validate_bundle(folder)

    def test_predict_end_to_end_synthetic_and_no_overwrite(self):
        from .decode import Settings
        class Calibration:
            def state(self,x): return x
            def edge(self,x): return x
        class Deep:
            calibration=Calibration()
            def raw(self,features,chunk):
                n=len(features[0]); return np.tile([.8,.1,.1],(n,1)),np.zeros((n,6))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); bundle=root/'bundle'; bundle.mkdir(); m=self.bundle(bundle)
            output=root/'candidate'; runs=[]
            for fold in range(5):
                run=root/f'run_{fold}';run.mkdir();runs.append(run)
                for name in ('manifest.json','calibration_selection.json','result.json'):
                    data=(dict(folds=[fold],code_sha256={},source_models_sha256={
                        'tree_candidate.joblib':'fake','variant_candidate.joblib':'fake'}) if name=='manifest.json'
                        else dict(chosen=dict(tree_weight=0,specialist_weight=0,specialist='old'))
                        if name=='calibration_selection.json' else dict(complete=True))
                    (run/name).write_text(json.dumps(data))
            with patch('riset_v4.predict_fusion.validate_bundle',return_value=m), \
                 patch('riset_v4.predict_fusion.verified_complete',return_value=True), \
                 patch('riset_v4.predict_fusion.fingerprint',return_value='fake'), \
                 patch('riset_v4.predict_fusion.joblib.load',return_value=Deep()), \
                 contextlib.redirect_stdout(io.StringIO()):
                receipt=predict(bundle,runs,output,root,root)
                self.assertTrue(receipt['complete'])
                self.assertFalse(receipt['cue_columns_used_for_prediction'])
                self.assertIsNone(receipt['metrics']['spellF1'])
                for name in ('deep_v45_equal5','calibration_fusion_v46_equal5'):
                    with np.load(output/(name+'.probabilities.npz')) as z:
                        np.testing.assert_allclose(z['probabilities'][0],[0,0,1])
                        np.testing.assert_allclose(z['probabilities'][1:9],np.tile([.8,.1,.1],(8,1)))
                with self.assertRaises(FileExistsError): predict(bundle,runs,output,root,root)
                with self.assertRaises(ValueError): predict(bundle,runs[:-1],root/'missing',root,root)
                self.assertFalse((root/'missing').exists())

    def test_report_real_completed_pilot_and_duplicates(self):
        run=Path(__file__).resolve().parents[1]/'outputs/fusion_v46_fold_0'
        if not (run/'result.json').exists(): self.skipTest('Local measured pilot not available')
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output=Path(directory)/'report'
            report([run],output)
            m=json.loads((output/'verification.json').read_text())
            self.assertEqual(m['rows'],75)
            for name,digest in m['artifacts_sha256'].items():
                self.assertEqual(fingerprint(output/name),digest)
            with self.assertRaises(FileExistsError): report([run],output)
            with self.assertRaises(ValueError): report([run,run],Path(directory)/'duplicate')
            self.assertFalse((Path(directory)/'duplicate').exists())


if __name__=='__main__': unittest.main()
