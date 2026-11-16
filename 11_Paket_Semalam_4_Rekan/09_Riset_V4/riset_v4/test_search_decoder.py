"""Synthetic tests only: no target labels, no experiment scores."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import joblib
import numpy as np
from .decode import Settings
from .experiment_io import output_lock, finish, verified_complete, validate_rows
from .mixtures import Mixture, blend
from . import search_decoder as search


class SearchDecoderTests(unittest.TestCase):
    def cases(self):
        y = np.array([0]*75+[1]*40+[2]*20+[0]*50, dtype=np.int8)
        p = np.eye(3, dtype=np.float32)[y]*.97+.01
        q = np.zeros((len(y),6), np.float32)
        return {'original': [((p,q,p.copy(),q.copy()), y)]}

    def test_blend_endpoints_independent_weights_and_invalid_inputs(self):
        arrays = (np.ones((4,3)),np.zeros((4,6)),np.zeros((4,3)),np.ones((4,6)))
        p,q = blend(arrays,1,0)
        np.testing.assert_array_equal(p,arrays[0]); np.testing.assert_array_equal(q,arrays[3])
        for weight in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError): blend(arrays,weight,.5)
        with self.assertRaises(ValueError): blend((*arrays[:2],np.zeros((5,3)),arrays[3]),.5,.5)

    def test_grid_reproducible_unique_and_baseline_first(self):
        a = search.candidate_grid(Settings(),Settings(team=1.),48,731)
        self.assertEqual(a,search.candidate_grid(Settings(),Settings(team=1.),48,731))
        self.assertEqual(len({json.dumps(x,sort_keys=True) for x in a}),48)
        self.assertEqual(a[0],dict(state_weight=.5,edge_weight=.5,decoder=asdict(Settings())))
        self.assertNotEqual(a,search.candidate_grid(Settings(),Settings(team=1.),48,732))
        for count in (1,4097):
            with self.assertRaises(ValueError): search.candidate_grid(Settings(),Settings(),count)

    def test_selection_uses_only_supplied_calibration_and_ties_favour_baseline(self):
        scenarios = self.cases(); grid = search.candidate_grid(Settings(),Settings(),4)
        with patch.object(search,'calibration_value',return_value=.7) as scorer:
            chosen,rows,index = search.select(scenarios,grid)
        self.assertEqual(index,0); self.assertEqual(chosen,grid[0])
        self.assertEqual(scorer.call_count,4)
        self.assertTrue(all(call.args[0] is scenarios for call in scorer.call_args_list))
        with patch.object(search,'calibration_value',return_value=.8) as scorer:
            _,resumed,index = search.select(scenarios,grid,rows[:2])
        self.assertEqual(scorer.call_count,2); self.assertEqual(resumed[:2],rows[:2]); self.assertEqual(index,2)
        for bad in ([dict(candidate=1,calibration_spellF1=.9)],
                    [dict(candidate=0,calibration_spellF1=float('nan'))]):
            with self.assertRaises(ValueError): search.select(scenarios,grid,bad)

    def test_calibration_and_predictions_keep_halves_separate(self):
        scenarios=self.cases(); case=scenarios['original'][0]
        self.assertAlmostEqual(search.calibration_value(scenarios,search.candidate_grid(Settings(),Settings(),2)[0]),1.)
        candidate=search.candidate_grid(Settings(),Settings(),2)[0]
        self.assertEqual(len(search.predictions([case,case],candidate)),2)
        with patch.object(search,'combine',wraps=search.combine) as combine:
            search.calibration_value({'original':[case,case]},candidate)
        self.assertTrue(all(len(c.args[0])==2 for c in combine.call_args_list))

    def test_diagnostic_holes_and_half_boundaries_do_not_leak(self):
        y=np.array([0,0,3,1,1],dtype=np.int8); pred=np.zeros(5,dtype=np.int8)
        p=np.eye(3)[[0,0,2,1,1]]
        result=search.oracle_diagnostics([y,y],[pred,pred],[p,p])
        self.assertEqual(len(result['oracle_true_boundaries']),2)
        for values in result.values():
            for value in values: np.testing.assert_array_equal(value,y)
        with self.assertRaises(ValueError): search.oracle_diagnostics([y],[],[])

    def test_serialization_loads_in_fresh_process(self):
        self.assertEqual(Mixture.__module__,'riset_v4.mixtures')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'candidate.joblib'
            joblib.dump(Mixture(None,None,.25,.75,Settings()),path)
            result=subprocess.run([sys.executable,'-c',
                'import joblib,sys; m=joblib.load(sys.argv[1]); assert type(m).__module__=="riset_v4.mixtures"; assert m.state_weight==.25; print("LOAD_OK")',
                str(path)],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,check=True)
            self.assertIn('LOAD_OK',result.stdout)

    def test_output_lock_rejects_concurrent_owner_then_releases(self):
        with tempfile.TemporaryDirectory() as folder:
            with output_lock(folder):
                with self.assertRaises(RuntimeError):
                    with output_lock(folder): pass
            with output_lock(folder): pass

    def test_completion_rejects_missing_modified_artifacts_and_wrong_row_count(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); path=root/'candidate.joblib'; path.write_bytes(b'synthetic')
            self.assertFalse(verified_complete(root,30))
            finish(root,['candidate.joblib'],rows=30)
            self.assertTrue(verified_complete(root,30))
            with self.assertRaises(ValueError): verified_complete(root,29)
            path.write_bytes(b'corrupted')
            with self.assertRaises(ValueError): verified_complete(root,30)
            path.unlink()
            with self.assertRaises(ValueError): verified_complete(root,30)

    def test_partial_evaluation_rows_validated(self):
        rows=[dict(method='a',fold=0,match='match',scenario='original',seed=432,split='known_retest',spellF1=.5,macroF1=.8)]
        self.assertEqual(validate_rows(rows,['a'],0,'match','original',432),rows)
        for bad in ([],rows*2,[dict(rows[0],seed=433)],[dict(rows[0],spellF1=float('nan'))]):
            with self.assertRaises(ValueError): validate_rows(bad,['a'],0,'match','original',432)


if __name__=='__main__': unittest.main()
