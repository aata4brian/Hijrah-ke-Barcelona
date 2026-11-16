"""Fast synthetic teammate tests; no real/public target labels read."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
from .data import fingerprint
from .decode import Settings
from .evaluate import atomic_json, write_csv
from .experiment_io import finish
from .models import Heads
from .team_training import BASE_TREE, CAPACITY_TREE, ResearchHeads, methods_for
from .team_queue import job_arguments, launch_allowed
from .team_results import export_results, import_results
from .team_public_audit import run as audit
from .test_v4 import fixture
from .features import pair


class TeamTests(unittest.TestCase):
    def test_four_assignments_disjoint_and_shared_protocol(self):
        commands=[job_arguments(i,0,Path('source'),[Path('cache')],Path('out'),'cpu') for i in range(1,5)]
        self.assertIn('riset_v4.search_decoder',commands[0])
        for command,track in zip(commands[1:],('tcn','capacity','augment')):
            self.assertEqual(command[command.index('--track')+1],track)
        self.assertEqual(len({tuple(c) for c in commands}),4)
        self.assertEqual(BASE_TREE['random_state'],CAPACITY_TREE['random_state'])
        self.assertEqual(len(methods_for('augment')),4)

    def test_launch_budget_and_stop_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.assertTrue(launch_allowed(root,10.,8.))
            self.assertFalse(launch_allowed(root,8*3600,8.))
            (root/'STOP_AFTER_JOB').touch()
            self.assertFalse(launch_allowed(root,0.,8.))

    def test_configurable_heads_match_unchanged_baseline_and_resume(self):
        d,y=fixture(600); features,_=pair(d); cases=[(features,y,1.)]
        baseline=Heads().fit(cases,iterations=2)
        with tempfile.TemporaryDirectory() as folder:
            parameters=dict(BASE_TREE,max_iter=2)
            model=ResearchHeads().fit(cases,parameters,folder)
            for a,b in zip(baseline.raw(features),model.raw(features)):
                np.testing.assert_allclose(a,b,atol=0,rtol=0)
            with patch('riset_v4.team_training.HistGradientBoostingClassifier',side_effect=AssertionError('Retrained')):
                resumed=ResearchHeads().fit(cases,parameters,folder)
            np.testing.assert_array_equal(resumed.raw(features)[0],model.raw(features)[0])
            self.assertEqual(ResearchHeads.__module__,'riset_v4.team_training')

    def make_run(self,project):
        folder=project/'outputs'/'rekan_1_test_fold_0'; folder.mkdir(parents=True)
        atomic_json(folder/'manifest.json',dict(synthetic_test_only=True,folds=[0]))
        rows=[dict(method=m,fold=0,match=match,scenario=scenario,seed=432,
            split='development' if match=='J03WMX' else 'known_retest',spellF1=.5,macroF1=.7)
            for match in ('J03WMX','J03WQQ','J03WR9')
            for scenario in ('original','short_gaps','long_gaps','camera_gaps','drift')
            for m in ('frozen_v44_ensemble','calibration_selected_decoder')]
        write_csv(folder/'performa_detail.csv',rows)
        finish(folder,['manifest.json','performa_detail.csv'],rows=30)
        (folder/'feature_cache').mkdir(); (folder/'feature_cache'/'do_not_send.npy').write_bytes(b'cache')
        return folder

    def test_transfer_roundtrip_idempotent_and_excludes_caches(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); self.make_run(root/'sender'); archive=root/'results.zip'
            export_results(root/'sender',archive,1,'test')
            with zipfile.ZipFile(archive) as zipped:
                self.assertFalse(any('feature_cache' in n for n in zipped.namelist()))
            with patch('riset_v4.team_results.sync_registry') as sync:
                a=import_results(archive,root/'receiver'); b=import_results(archive,root/'receiver')
            self.assertEqual(a,b); self.assertEqual(sync.call_count,2)
            target=root/'receiver'/'outputs'/'rekan_1_test_fold_0'/'performa_detail.csv'
            target.write_text('existing user changes')
            with self.assertRaises(FileExistsError): import_results(archive,root/'receiver')
            self.assertEqual(target.read_text(),'existing user changes')

    def test_transfer_rejects_tampering_and_path_traversal(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); self.make_run(root/'sender'); good=root/'good.zip'
            export_results(root/'sender',good,1,'test')
            with zipfile.ZipFile(good) as zipped: contents={n:zipped.read(n) for n in zipped.namelist()}
            for name in ('rekan_1_test_fold_0/manifest.json','../escape.txt'):
                values=dict(contents)
                if name.startswith('..'):
                    index=json.loads(values['transfer_manifest.json']); index['sha256'][name]='0'*64
                    values['transfer_manifest.json']=json.dumps(index).encode(); values[name]=b'unsafe'
                else: values[name]=b'tampered'
                bad=root/'bad.zip'
                with zipfile.ZipFile(bad,'w') as zipped:
                    for key,value in values.items(): zipped.writestr(key,value)
                with self.assertRaises(ValueError): import_results(bad,root/'receiver')
                self.assertFalse((root/'escape.txt').exists())

    def test_audit_frozen_inputs_once_no_candidate_search(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); assets=root/'assets'; assets.mkdir(); output=root/'audit'
            y=['H']*25+['A']*25+['X']*50; pred=['H']*25+['A']*25+['D']*50
            names=['BAR-ATH_possession_gt_public.csv','frozen_candidate.csv']
            for name,labels in zip(names,[y,pred]):
                write_csv(assets/name,[dict(frame=i,label=v) for i,v in enumerate(labels)])
            spec=dict(purpose='pipeline_check_only_not_model_selection',frames=100,
                      sha256={name:fingerprint(assets/name) for name in names})
            atomic_json(assets/'audit_spec.json',spec)
            first=audit(assets,output); second=audit(assets,output)
            self.assertEqual(first,second); self.assertEqual(first['graded_frames'],50)
            self.assertFalse(first['target_gt_used_for_training_or_selection'])
            (assets/'frozen_candidate.csv').write_text('changed')
            with self.assertRaises(ValueError): audit(assets,output)


if __name__=='__main__': unittest.main()
