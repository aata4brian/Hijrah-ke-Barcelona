import unittest
import cv2
import numpy as np
from .video import refine_anchor,recover_gap


class VideoRefinementTests(unittest.TestCase):
    def image(self,centers):
        image=np.full((200,300),70,np.uint8)
        for center in centers:cv2.circle(image,center,3,230,-1)
        return image

    def test_find_offset_ball(self):
        center,why=refine_anchor(self.image([(100,100)]),[108,100])
        self.assertEqual(why,'')
        np.testing.assert_allclose(center,[100,100],atol=.2)

    def test_flat_background_is_rejected(self):
        self.assertIsNone(refine_anchor(self.image([]),[100,100])[0])

    def test_ambiguous_candidates_are_rejected(self):
        self.assertEqual(refine_anchor(self.image([(94,100),(106,100)]),[100,100])[1],
                         'ambiguous_candidates')

    def test_pitch_line_is_not_a_ball(self):
        image=self.image([]);cv2.line(image,(0,100),(299,100),230,2)
        self.assertIsNone(refine_anchor(image,[100,100])[0])

    def test_large_player_region_is_rejected(self):
        image=self.image([]);cv2.rectangle(image,(95,85),(105,115),230,-1)
        self.assertIsNone(refine_anchor(image,[100,100])[0])

    def test_invalid_anchor_is_rejected(self):
        for point in ([np.nan,20],[1,20],[-2,10],[1000,20]):
            self.assertIsNone(refine_anchor(self.image([]),point)[0])

    def test_corrected_tracking_preserves_anchors_and_rejects_occlusion(self):
        pitch=np.array([15.,10.]);h=np.tile(np.diag([300.,200.,1.]),(9,1,1))
        frames=[self.image([(100+i,100)]) for i in range(9)]
        start=np.array([108.,100.])/20;end=np.array([116.,100.])/20
        xy,confidence,why=recover_gap(frames,h,np.ones(9,bool),np.zeros(9),start,end,pitch,refine=True)
        self.assertEqual(why,'accepted')
        np.testing.assert_array_equal(xy[[0,-1]],[start,end])
        truth=np.array([[100+i,100] for i in range(9)])/20
        self.assertLess(np.max(np.linalg.norm(xy[1:-1]-truth[1:-1],axis=1)),.05)
        self.assertGreaterEqual(confidence[1:-1].min(),.2)
        frames[4]=self.image([])
        self.assertIsNone(recover_gap(frames,h,np.ones(9,bool),np.zeros(9),start,end,pitch,refine=True)[0])


if __name__=='__main__':unittest.main()
