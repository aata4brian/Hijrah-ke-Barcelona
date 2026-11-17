import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from deadball.flags import color_candidates, classify, touchline_distance, near_touchline_proposals, TemporalFlags, HomographyReader
from deadball.pose import decode


def person(raised=True):
    k = np.zeros((17,3),np.float32)
    for i,p in {5:(90,100),7:(90,70),9:(90,45),15:(95,200),16:(105,200)}.items():
        k[i] = (*p,.95)
    if not raised:
        k[7] = (75,135,.95)
        k[9] = (60,165,.95)
    return dict(box=[70.,40.,60.,160.],score=.92,keypoints=k)


H = np.array([[1000.,0,0],[0,1000.,200],[0,0,1]])


class FlagTests(unittest.TestCase):
    def test_vectorized_gate_matches_scalar_geometry(self):
        rng = np.random.default_rng(19)
        candidates = [dict(x=float(x),y=float(y)) for x,y in rng.uniform(-150,1300,(200,2))]
        expected = [any((d := touchline_distance(H,(c['x'],c['y']+dy))) is not None
            and d <= .07 for dy in (0,30,60,100)) for c in candidates]
        np.testing.assert_array_equal(near_touchline_proposals(H,candidates),expected)
        self.assertFalse(near_touchline_proposals(np.zeros((3,3)),candidates).any())

    def test_color_blob_and_green_rejection(self):
        image = np.zeros((100,100,3),np.uint8)
        image[:] = (0,140,0)
        cv2.rectangle(image,(15,20),(22,28),(0,220,255),-1)
        found = color_candidates(image)
        self.assertEqual(len(found),1)
        self.assertAlmostEqual(found[0]['x'],18.5)

    def test_raised_and_lowered_geometry(self):
        for raised,x,y,expected in [(True,92,28,'RAISED_CANDIDATE'),(False,48,175,'LOWERED_CANDIDATE')]:
            state,score,_,_ = classify(person(raised),[dict(x=x,y=y,area=40)],H)
            self.assertEqual(state,expected)
            self.assertGreater(score,0)

    def test_no_flag_no_pose_or_no_homography_abstains(self):
        flag = [dict(x=92.,y=28.,area=40)]
        self.assertEqual(classify(person(),[],H)[0],'UNKNOWN')
        self.assertEqual(classify(person(),flag,None)[0],'UNKNOWN')
        p = person(); p['keypoints'][9,2] = .1
        self.assertEqual(classify(p,flag,H)[0],'UNKNOWN')

    def test_middle_of_pitch_not_assistant(self):
        self.assertEqual(classify(person(),[dict(x=92.,y=28.,area=40)],np.eye(3)*1000)[0],'UNKNOWN')
        self.assertIsNone(touchline_distance(np.zeros((3,3)),[2,3]))

    def test_persistence_and_occlusion_reset(self):
        tracker = TemporalFlags()
        d = dict(person=person(),state='RAISED_CANDIDATE',score=.8,reason='test',flag=dict(x=92.,y=28.))
        self.assertEqual(tracker.update(0,[d])[0]['state'],'RAISED_CANDIDATE')
        self.assertEqual(tracker.update(.5,[d])[0]['state'],'RAISED_CANDIDATE')
        self.assertEqual(tracker.update(1,[d])[0]['state'],'RAISED_SUPPORTED')
        self.assertEqual(tracker.update(1.5,[]),[])
        self.assertEqual(tracker.update(2,[d])[0]['state'],'RAISED_CANDIDATE')
        with self.assertRaises(ValueError): tracker.update(2,[])

    def test_colored_boot_is_not_a_flag(self):
        state,_,_,_ = classify(person(False),[dict(x=95.,y=195.,area=40)],H)
        self.assertEqual(state,'UNKNOWN')

    def test_flag_identity_jump_resets_persistence(self):
        tracker = TemporalFlags()
        d = dict(person=person(),state='RAISED_CANDIDATE',score=.8,reason='test',flag=dict(x=92.,y=28.))
        tracker.update(0,[d]); tracker.update(.5,[d])
        switched = dict(d,flag=dict(x=210.,y=28.))
        self.assertEqual(tracker.update(1,[switched])[0]['state'],'RAISED_CANDIDATE')

    def test_pose_decode_contract_and_coordinates(self):
        data = np.zeros((1,56,8400),np.float32)
        data[0,:5,0] = [200,240,80,160,.9]
        data[0,5:,0] = np.tile([200,240,.8],17)
        out = decode(data,2,0,0,640,480)
        self.assertEqual(len(out),1)
        np.testing.assert_allclose(out[0]['box'],[80,80,40,80])
        np.testing.assert_allclose(out[0]['keypoints'][0],[100,120,.8],atol=1e-6)
        with self.assertRaises(ValueError): decode(np.zeros((1,57,8400)),1,0,0,640,480)

    def test_homography_frame_keys_and_invalid_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'h.csv'
            path.write_text('frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n0,1,0,0,0,1,0,0,0,1,.2,1\n2,1,0,0,0,1,0,0,0,1,8,1\n',encoding='utf-8')
            reader = HomographyReader(path)
            try:
                np.testing.assert_array_equal(reader.at(0),np.eye(3))
                self.assertIsNone(reader.at(1))
                self.assertIsNone(reader.at(2))
                with self.assertRaises(ValueError): reader.at(0)
            finally: reader.close()


if __name__ == '__main__': unittest.main()
