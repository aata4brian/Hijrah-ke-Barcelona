"""Synthetic validation of local deep model, not competition results."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from .deep_temporal import DeepConfig, DeepTemporal, RefinedNetwork, temporal_loss
from .decode import Settings
from .labels import CLASS_SWAP, EDGE_SWAP
from .models import Calibration
from .tcn import Network
from types import SimpleNamespace


def baseline(width=4):
    torch.manual_seed(331)
    return SimpleNamespace(net=Network(width*2,channels=8),center=np.zeros(width,np.float32),
        scale=np.ones(width,np.float32),calibration=Calibration(),decoder=Settings())


class DeepTemporalTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(2)

    def test_warm_start_identical_and_new_context_learns(self):
        base=baseline(); model=RefinedNetwork(base.net).eval(); base.net.eval()
        x=torch.randn(1,8,1600)
        with torch.no_grad():
            expected=base.net(x); stages=model(x)
        for stage in stages:
            for a,b in zip(stage,expected): torch.testing.assert_close(a,b,atol=0,rtol=0)
        model.train(); model(x)[1][0].square().mean().backward()
        self.assertGreater(float(model.blocks[6].net[3].weight.grad.abs().sum()),0)
        self.assertGreater(float(model.refine_state.weight.grad.abs().sum()),0)

    def test_loss_ignores_holes_and_does_not_smooth_real_boundaries(self):
        target=torch.tensor([[0,1,3,3]])
        edges=torch.zeros(1,6,4); weights=torch.tensor([[1.,1.,0.,0.]])
        state=torch.randn(1,3,4,requires_grad=True); edge=torch.randn(1,6,4,requires_grad=True)
        cfg=DeepConfig()
        actual=temporal_loss([(state,edge)],target,edges,weights,cfg)
        plain=temporal_loss([(state,edge)],target,edges,weights,replace(cfg,smoothing_weight=0))
        torch.testing.assert_close(actual,plain,atol=0,rtol=0)
        actual.backward()
        self.assertEqual(float(state.grad[:,:,2:].abs().sum()),0.)
        self.assertEqual(float(edge.grad[:,:,2:].abs().sum()),0.)

    def test_chunked_inference_and_team_swap(self):
        model=DeepTemporal(baseline()); rng=np.random.default_rng(4)
        arrays=tuple(rng.normal(size=(1450,4)).astype(np.float32) for _ in range(2))
        arrays[0][30:45,0]=np.nan
        p,q=model.raw(arrays,chunk=1450,device='cpu')
        pp,qq=model.raw(arrays,chunk=131,device='cpu')
        np.testing.assert_allclose(p,pp,atol=2e-6,rtol=0)
        np.testing.assert_allclose(q,qq,atol=2e-6,rtol=0)
        sp,sq=model.raw(arrays[::-1],device='cpu')
        np.testing.assert_allclose(sp,p[:,[1,0,2]],atol=2e-6,rtol=0)
        np.testing.assert_allclose(sq,q[:,EDGE_SWAP],atol=2e-6,rtol=0)
        self.assertTrue(np.isfinite(p).all()); np.testing.assert_allclose(p.sum(1),1.,atol=2e-6)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_mixed_precision_windows_targets_and_resume(self):
        rng=np.random.default_rng(8); y=(np.arange(170)//30%3).astype(np.int32)
        features=tuple(rng.normal(size=(170,4)).astype(np.float16) for _ in range(2))
        torch.manual_seed(4); model=DeepTemporal(baseline())
        cfg=DeepConfig(epochs=2,steps=2,batch=1,core=64,validation_every=1)
        def validate(current, epoch):
            p,q=current.raw(features,device='cuda')
            self.assertTrue(np.isfinite(p).all()); self.assertTrue(np.isfinite(q).all())
            return .5
        with tempfile.TemporaryDirectory() as temp:
            checkpoint=Path(temp)/'training.pt'
            model.fit([(features,y,1.)],cfg,checkpoint,validate=validate,device='cuda',
                stop_after_epoch=1,progress=lambda e:None)
            resumed=DeepTemporal(baseline()).fit([(features,y,1.)],cfg,checkpoint,
                validate=validate,device='cuda',progress=lambda e:None)
            self.assertEqual(len(resumed.history),2)
            self.assertTrue(all(r['optimizer_updates']>0 for r in resumed.history))

    def test_checkpoint_resume_matches_uninterrupted_cpu(self):
        rng=np.random.default_rng(2); y=(np.arange(170)//30%3).astype(np.int8)
        features=tuple(rng.normal(size=(170,4)).astype(np.float32) for _ in range(2))
        cases=[(features,y,1.)]; base=baseline()
        torch.manual_seed(9); original=DeepTemporal(base)
        a=deepcopy(original); b=deepcopy(original); cfg=DeepConfig(epochs=2,steps=1,batch=1,core=64,mixed_precision=False)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            a.fit(cases,cfg,root/'full.pt',device='cpu',progress=lambda e:None)
            b.fit(cases,cfg,root/'resume.pt',device='cpu',stop_after_epoch=1,progress=lambda e:None)
            resumed=deepcopy(original).fit(cases,cfg,root/'resume.pt',device='cpu',progress=lambda e:None)
            for name,value in a.net.state_dict().items():
                torch.testing.assert_close(value,resumed.net.state_dict()[name],atol=0,rtol=0)
            self.assertEqual([r['loss'] for r in a.history],[r['loss'] for r in resumed.history])
            with self.assertRaisesRegex(ValueError,'protocol changed'):
                deepcopy(original).fit(cases,replace(cfg,batch=2),root/'resume.pt',device='cpu',progress=lambda e:None)


if __name__=='__main__': unittest.main()
