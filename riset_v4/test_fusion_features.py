import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .features import pair
from .fusion_features import extra_pair, shifted
from .cue_features import load_cues, merge_cues, export_cues
from .test_v4 import fixture
from .data import fingerprint
from .fusion_model import combine_probabilities


class FeatureTests(unittest.TestCase):
    def test_symmetry_and_frozen_schema(self):
        data, _ = fixture(900)
        old, old_names = pair(data)
        arrays, names = extra_pair(data)
        swapped = type(data)(data.ball, data.players[::-1], data.pitch)
        mirrored, other = extra_pair(swapped)
        np.testing.assert_allclose(arrays[0], mirrored[1], equal_nan=True)
        np.testing.assert_allclose(arrays[1], mirrored[0], equal_nan=True)
        self.assertEqual(names, other)
        self.assertEqual(old[0].shape[1], 203)
        np.testing.assert_allclose(old[0], pair(data)[0][0], equal_nan=True)
        self.assertEqual(len(names), arrays[0].shape[1])
        self.assertFalse(np.isinf(arrays[0]).any())

    def test_missing_motion_is_not_static(self):
        data, _ = fixture(900)
        data = type(data)(np.full_like(data.ball, np.nan),
                          tuple(np.full_like(p, np.nan) for p in data.players), data.pitch)
        arrays, names = extra_pair(data)
        for name in ['collective_static', 'ball_static_run', 'home_static_share', 'contested_contact', 'home_contact_density']:
            self.assertTrue(np.isnan(arrays[0][:, names.index(name)]).all(), name)

    def test_shift_has_no_half_wrap(self):
        np.testing.assert_allclose(shifted(np.arange(3, dtype=float), 1), [np.nan, 0, 1], equal_nan=True)
        self.assertTrue(np.isnan(shifted(np.arange(3, dtype=float), -25)).all())

    def test_slot_teleport_rejected(self):
        data, _ = fixture(900)
        players = tuple(p.copy() for p in data.players)
        for p in players:
            p[450:] += 100
        arrays, names = extra_pair(type(data)(data.ball, players, data.pitch))
        self.assertEqual(arrays[0][450, names.index('home_motion_coverage')], 0)

    def test_merge_missing_not_zero_and_reject_overlap(self):
        a = np.full((5, 9), np.nan, np.float32); a[:, [0, 3]] = 0
        b = a.copy(); b[2, 0:2] = [1, 0]
        out = merge_cues([a, b])
        self.assertEqual(out[2, 1], 0)
        self.assertTrue(np.isnan(out[1, 1]))
        c = b.copy(); c[2, 1] = .8
        with self.assertRaises(ValueError): merge_cues([b, c])

    def test_zero_weight_exactly_preserves_frozen_probabilities(self):
        rng = np.random.default_rng(14)
        p = rng.dirichlet([2,2,2], 100).astype(np.float32)
        q = rng.random((100,6)).astype(np.float32)
        tp = p[:, [1,0,2]].copy(); tq = q*.5
        output = combine_probabilities((p,q),(tp,tq),tp,0,0)
        np.testing.assert_array_equal(output[0],p)
        np.testing.assert_array_equal(output[1],q)
        with self.assertRaises(ValueError): combine_probabilities((p,q),(tp,tq),tp,0,1.1)

    def test_cue_coverage_sparse_flags_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'whistles.csv').write_text('start_s,end_s,score\n0.1,0.3,0.8\n')
            (root/'visual_samples.csv').write_text('frame,time_s,homography_valid\n10,0.4,1\n20,0.8,0\n')
            (root/'flags.csv').write_text('frame,state,score\n10,LOWERED_CANDIDATE,0.7\n')
            m = dict(status='complete',metadata=dict(fps=25,frames=50,width=160,height=120),audio_status='processed',
                visual_status='processed',audio_audit=dict(start_s=0,end_s=1.5,decoded_samples=24000),
                whistle_config=dict(sample_rate=16000),source_name='test.mp4',source_bytes=123,
                output_sha256={p.name:fingerprint(p) for p in root.glob('*.csv')})
            (root/'manifest.json').write_text(json.dumps(m))
            features, audit = load_cues(root,50)
            self.assertEqual(features[10,1],np.float32(.8))
            self.assertTrue(np.isnan(features[9,2]))  # closure context not yet elapsed
            self.assertEqual(features[10,3],1)
            self.assertEqual(features[10,7],np.float32(.7))
            self.assertTrue(np.isnan(features[11,7]))  # no forward fill
            self.assertEqual(features[20,3],0)  # invalid homography
            self.assertEqual(features[40,0],0)
            self.assertTrue(np.isnan(features[40,1]))
            result = root/'export'
            summary = export_cues([root], result)
            self.assertEqual(summary['visual_valid_sample_frames'],1)
            self.assertEqual(summary['associated_flag_sample_frames'],1)
            self.assertIsNone(summary['learned_weight'])
            with np.load(result/'cue_features.npz', allow_pickle=False) as archive:
                np.testing.assert_allclose(archive['features'],features,equal_nan=True)
            with self.assertRaises(FileExistsError): export_cues([root],result)
            m['source_bytes'] = 124
            other = root/'other'; other.mkdir()
            (other/'manifest.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError): export_cues([root,other],root/'invalid')
            self.assertFalse((root/'invalid').exists())
            (root/'flags.csv').write_text('corrupted')
            with self.assertRaises(ValueError): load_cues(root,50)


if __name__ == '__main__':
    unittest.main()
