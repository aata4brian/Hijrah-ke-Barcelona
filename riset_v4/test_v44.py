import tempfile
import unittest
from pathlib import Path
import numpy as np
from .test_v4 import fixture
from .features import pair
from .models import Heads,switch_context
from .run import legacy_features


class ContinuationTests(unittest.TestCase):
    def test_sparse_context_matches_dense_and_legacy_grid(self):
        d,y=fixture(900);features,_=pair(d)
        idx=np.array([0,1,24,25,26,500,875,899])
        for x,old in zip(features,legacy_features(d)):
            np.testing.assert_allclose(switch_context(x,idx),switch_context(x)[idx],equal_nan=True)
            np.testing.assert_allclose(x[::5,:110],old,equal_nan=True)

    def test_disk_training_and_chunked_inference(self):
        d,y=fixture(1600);features,_=pair(d);cases=[(features,y,1.)]
        ram=Heads().fit(cases,iterations=3)
        with tempfile.TemporaryDirectory() as folder:
            disk=Heads().fit(cases,iterations=3,work_dir=folder)
            for a,b in zip(ram.raw(features,chunk=1600),disk.raw(features,chunk=71)):
                np.testing.assert_allclose(a,b,atol=1e-7)
            self.assertEqual(list(Path(folder).glob('*.npy')),[])
            resumed=Heads().fit(cases,iterations=3,work_dir=folder)
            np.testing.assert_allclose(resumed.raw(features)[0],disk.raw(features)[0])

    def test_video_buffer_above_old_memory_limit(self):
        import cv2
        from .video import read_frames
        class Capture:
            def set(self,key,value):self.index=int(value);return True
            def get(self,key):return self.index
            def read(self):
                image=np.zeros((1500,1500,3),np.uint8)
                cv2.circle(image,(750,750),5,(240,240,240),-1)
                cv2.circle(image,(748,749),2,(100,100,100),-1)
                self.index+=1;return True,image
        with tempfile.TemporaryDirectory() as folder:
            frames=read_frames(Capture(),0,50,np.array([750.,750.]),Path(folder)/'buffer.npy')
            self.assertIsInstance(frames,np.memmap)
            self.assertGreater(frames.nbytes,96*1024*1024)
            np.testing.assert_equal(frames[0],frames[-1])
            frames._mmap.close()

    def test_tcn_resumes_identical_training(self):
        import torch
        from .tcn import Temporal
        d,y=fixture(900);features,_=pair(d);cases=[(features,y,1.)]
        full=Temporal().fit(cases,epochs=2,steps=1,device='cpu',seed=13,progress=lambda x:None)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'resume.pt'
            Temporal().fit(cases,epochs=1,steps=1,device='cpu',seed=13,progress=lambda x:None,checkpoint=path)
            resumed=Temporal().fit(cases,epochs=2,steps=1,device='cpu',seed=13,progress=lambda x:None,checkpoint=path)
            for name,value in full.net.state_dict().items():
                torch.testing.assert_close(value,resumed.net.state_dict()[name],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
