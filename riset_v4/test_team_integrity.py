"""Synthetic queue/transfer regression checks, never the real result ledger."""
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
from . import team_queue as queue
from .data import fingerprint
from .evaluate import write_csv
from .team_results import export_results, import_results
from . import test_team as fixtures


class TeamIntegrityTests(unittest.TestCase):
    def queue_case(self, root, existing=False, returncode=0):
        source=root/'source'; source.mkdir()
        (source/'manifest.json').write_text('{}')
        cache=root/'cache.npz'; cache.write_bytes(b'synthetic')
        output=root/'outputs'; output.mkdir()
        q=output/'queue'; q.mkdir()
        args=SimpleNamespace(member=1,device='cpu',allow_cpu_tcn=False,source=source,
            cache=[root],session='test',folds=[0],candidates=192,epochs=20,
            hours=8.,min_free_gib=0.)
        stack=ExitStack()
        stack.enter_context(patch.object(queue,'discover',return_value={'fixture':cache}))
        stack.enter_context(patch.object(queue,'sync_registry'))
        verified=stack.enter_context(patch.object(queue,'verified_complete',side_effect=[existing,True]))
        process=stack.enter_context(patch.object(queue.subprocess,'Popen'))
        process.return_value.wait.return_value=returncode
        process.return_value.pid=12345
        return args,output,q,stack,process,verified

    def test_queue_launches_and_checks_child_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            args,out,q,stack,process,verified=self.queue_case(Path(temp))
            with stack: queue._run(args,Path(temp),out,q)
            self.assertEqual(process.call_count,1)
            self.assertEqual(verified.call_count,2)
            self.assertEqual(json.loads((q/'queue_status.json').read_text())['stage'],'complete')

    def test_complete_run_still_rechecks_child_manifest_after_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            args,out,q,stack,process,_=self.queue_case(Path(temp),existing=True)
            (q/'STOP_AFTER_JOB').touch()
            with stack: queue._run(args,Path(temp),out,q)
            self.assertEqual(process.call_count,1)

    def test_stop_prevents_new_fold_and_child_failure_is_reported(self):
        for stopped in (True,False):
            with self.subTest(stopped=stopped), tempfile.TemporaryDirectory() as temp:
                args,out,q,stack,process,_=self.queue_case(Path(temp),returncode=7)
                if stopped: (q/'STOP_AFTER_JOB').touch()
                with stack:
                    if stopped: queue._run(args,Path(temp),out,q)
                    else:
                        with self.assertRaises(RuntimeError): queue._run(args,Path(temp),out,q)
                status=json.loads((q/'queue_status.json').read_text())
                self.assertEqual(status['stage'],'paused' if stopped else 'failed')
                self.assertEqual(process.call_count,0 if stopped else 1)

    def test_complete_transfer_rejects_missing_keys_even_with_matching_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); run=fixtures.TeamTests().make_run(root/'sender')
            detail=run/'performa_detail.csv'
            with detail.open(encoding='utf-8-sig',newline='') as handle: rows=list(csv.DictReader(handle))
            write_csv(detail,rows[:-1])
            result=json.loads((run/'result.json').read_text())
            result['artifacts_sha256']['performa_detail.csv']=fingerprint(detail)
            (run/'result.json').write_text(json.dumps(result))
            archive=root/'bad.zip'; export_results(root/'sender',archive,1,'test')
            with patch('riset_v4.team_results.sync_registry') as sync:
                with self.assertRaisesRegex(ValueError,'missing keys'): import_results(archive,root/'receiver')
            sync.assert_not_called()
            self.assertFalse((root/'receiver/outputs'/run.name).exists())

    def test_partial_snapshot_conflict_preserves_previous_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); run=fixtures.TeamTests().make_run(root/'sender')
            (run/'result.json').unlink()
            first=root/'first.zip'; export_results(root/'sender',first,1,'test')
            with patch('riset_v4.team_results.sync_registry'):
                import_results(first,root/'receiver')
            old=(root/'receiver/outputs'/run.name/'manifest.json').read_bytes()
            manifest=json.loads((run/'manifest.json').read_text()); manifest['additional_evidence']=True
            (run/'manifest.json').write_text(json.dumps(manifest))
            second=root/'second.zip'; export_results(root/'sender',second,1,'test')
            with self.assertRaises(FileExistsError): import_results(second,root/'receiver')
            self.assertEqual((root/'receiver/outputs'/run.name/'manifest.json').read_bytes(),old)


if __name__=='__main__': unittest.main()
