from dataclasses import asdict
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from .decode import Settings
from .fusion_experiment import select_on_calibration, specialist_temperature, apply_choice
from .fusion_model import specialist_probability
from .fusion_bundle import local_tracking
from .test_v4 import fixture


class FusionExperimentTests(unittest.TestCase):
    def test_selection_tie_prefers_no_extra_weight(self):
        p = np.tile([.7,.2,.1],(100,1)).astype(np.float32)
        q = np.zeros((100,6),np.float32)
        case = dict(deep=(p,q),tree=(p,q),specialists=dict(old=p,expanded=p))
        frozen = SimpleNamespace(decoder=Settings())
        selection = select_on_calibration([case],[np.zeros(100,np.int8)],['original'],
            dict(old=1,expanded=1),frozen,frozen)
        self.assertEqual(selection['chosen']['tree_weight'],0)
        self.assertEqual(selection['chosen']['specialist_weight'],0)
        self.assertEqual(len(selection['candidates']),18)
        self.assertEqual(selection['by_ablation']['expanded']['specialist_weight'],0)
        pred = apply_choice(case,selection['chosen'],dict(old=1,expanded=1),frozen,frozen)
        np.testing.assert_array_equal(pred,np.zeros(100,np.int8))

    def test_temperature_ignores_x_labels(self):
        p = np.tile([.8,.1,.1],(100,1))
        y = np.r_[np.zeros(50,np.int8),np.full(50,3,np.int8)]
        a = specialist_temperature([p],[y])
        changed = p.copy(); changed[50:] = [.001,.001,.998]
        b = specialist_temperature([changed],[y])
        self.assertAlmostEqual(a,b,places=12)
        self.assertTrue(.5 <= a <= 3)

    def test_probability_classes_and_team_symmetry(self):
        class Head:
            classes_ = np.array([0,1,2])
            def predict_proba(self,x):
                return x
        forward=np.tile([.6,.1,.3],(20,1)).astype(np.float32)
        mirror=forward[:,[1,0,2]]
        result=specialist_probability(Head(),(forward,mirror))
        swapped=specialist_probability(Head(),(mirror,forward))
        np.testing.assert_allclose(result,swapped[:,[1,0,2]],atol=1e-7)

    def test_tracking_only_loader_never_requests_annotations(self):
        import games
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); meta=root/'meta.xml'; meta.write_text('<unused/>')
            raw=root/'raw.txt'
            # Two slots, two frames, coordinates normalized to pitch.
            raw.write_text('1:0.1,0.2,0;0.8,0.7,0:0.4,0.5,0\n'
                           '2:0.2,0.2,0;0.7,0.7,0:NaN,NaN,0\n')
            table={games.GAME:dict(n_video_frames=2,pitch=(100,50))}
            with patch.object(games,'require',return_value=[str(meta),str(raw)]) as require, \
                 patch.object(games,'GAMES',table), \
                 patch.object(games,'half_video_blocks',return_value=[(1,0,1)]), \
                 patch('starter.baseline.team_of_each_track',return_value=np.array([0,1])):
                halves=list(local_tracking(root))
            self.assertEqual(len(halves),1)
            self.assertEqual(require.call_args.args[0],['atd_meta','atd_raw'])
            data=halves[0][3]
            np.testing.assert_allclose(data.ball[0],[40,25])
            self.assertTrue(np.isnan(data.ball[1]).all())
            self.assertEqual(data.players[0].shape,(2,1,2))


if __name__ == '__main__':
    unittest.main()
