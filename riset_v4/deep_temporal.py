"""Local V4.5 research: warm-start long-context TCN and residual refinement."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import gc
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .labels import CLASS_SWAP, EDGE_SWAP, boundary_targets, sample_weights
from .models import Calibration
from .tcn import Residual


@dataclass(frozen=True)
class DeepConfig:
    epochs: int = 12
    steps: int = 160
    batch: int = 4
    core: int = 512
    learning_rate: float = .0002
    refinement_learning_rate: float = .0005
    boundary_weight: float = .4
    smoothing_weight: float = .15
    seed: int = 331
    validation_every: int = 3
    mixed_precision: bool = True


class RefinedNetwork(nn.Module):
    # Six pretrained blocks (126), extra dilation 64/128 (384), refiner (126).
    halo = 636

    def __init__(self, baseline):
        super().__init__()
        self.input = deepcopy(baseline.input)
        blocks = list(deepcopy(baseline.blocks).children())
        channels = baseline.input.out_channels
        for dilation in (64, 128):
            block = Residual(channels, dilation)
            nn.init.zeros_(block.net[3].weight); nn.init.zeros_(block.net[3].bias)
            block.net[4] = nn.Identity()  # Zero-start branch must retain a nonzero derivative.
            blocks.append(block)
        self.blocks = nn.Sequential(*blocks)
        self.state = deepcopy(baseline.state); self.edge = deepcopy(baseline.edge)
        self.refine = nn.Sequential(nn.Conv1d(9, 32, 1), nn.ReLU(),
            *[Residual(32, d) for d in (1, 2, 4, 8, 16, 32)])
        self.refine_state = nn.Conv1d(32, 3, 1); self.refine_edge = nn.Conv1d(32, 6, 1)
        for head in (self.refine_state, self.refine_edge):
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)

    def forward(self, x):
        z = self.blocks(self.input(x))
        state, edge = self.state(z), self.edge(z)
        refined = self.refine(torch.cat([torch.softmax(state, 1), torch.sigmoid(edge)], 1))
        return (state, edge), (state+self.refine_state(refined), edge+self.refine_edge(refined))


def temporal_loss(outputs, target, edges, weights, config):
    loss = weights.sum()*0.
    valid_pair = (target[:, 1:] == target[:, :-1]) & (target[:, 1:] < 3) & (target[:, :-1] < 3)
    pair_weights = torch.minimum(weights[:, 1:], weights[:, :-1])*valid_pair
    for state, boundary in outputs:
        state, boundary = state.float(), boundary.float()
        ce = F.cross_entropy(state, target, ignore_index=3, reduction='none')
        bce = F.binary_cross_entropy_with_logits(boundary, edges,
            pos_weight=torch.full((1, 6, 1), 15., device=state.device), reduction='none').mean(1)
        supervised = ((ce+config.boundary_weight*bce)*weights).sum()/weights.sum().clamp_min(1)
        logs = F.log_softmax(state, 1)
        difference = ((logs[:, :, 1:]-logs[:, :, :-1].detach())**2).clamp(max=16.).mean(1)
        smooth = (difference*pair_weights).sum()/pair_weights.sum().clamp_min(1)
        loss = loss+supervised+config.smoothing_weight*smooth
    return loss/len(outputs)


class DeepTemporal:
    def __init__(self, baseline):
        self.center = baseline.center.copy(); self.scale = baseline.scale.copy()
        self.net = RefinedNetwork(baseline.net.cpu())
        self.calibration = deepcopy(baseline.calibration); self.decoder = baseline.decoder
        self.history = []; self.best_value = -1.; self.best_epoch = 0

    def transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        observed = np.isfinite(x)
        z = np.where(observed, (x-self.center)/self.scale, 0)
        return np.concatenate([np.clip(z, -10, 10), observed.astype(np.float32)], 1).astype(np.float32)

    def raw(self, pair, chunk=2048, device='auto'):
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.net.to(device).eval(); result = []; boundary = []
        try:
            with torch.inference_mode():
                for features in pair:
                    ps = []; qs = []; n = len(features)
                    for first in range(0, n, chunk):
                        last = min(n, first+chunk); a = max(0, first-self.net.halo); b = min(n, last+self.net.halo)
                        x = torch.from_numpy(self.transform(features[a:b]).T.copy())[None].to(device)
                        _, (p, q) = self.net(x); section = slice(first-a, last-a)
                        ps.append(torch.softmax(p.float(), 1)[0, :, section].T.cpu().numpy())
                        qs.append(torch.sigmoid(q.float())[0, :, section].T.cpu().numpy())
                    result.append(np.concatenate(ps)); boundary.append(np.concatenate(qs))
        finally: self.net.cpu()
        return ((result[0]+result[1][:, [1, 0, 2]])/2,
                (boundary[0]+boundary[1][:, EDGE_SWAP])/2)

    def predict(self, pair):
        p, q = self.raw(pair)
        return self.calibration.state(p), self.calibration.edge(q)

    def fit(self, cases, config, checkpoint, validate=None, progress=print, device='cuda', stop_after_epoch=None):
        checkpoint = Path(checkpoint); torch.set_num_threads(4)
        random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
        if device == 'cuda':
            if not torch.cuda.is_available(): raise RuntimeError('CUDA required for this local experiment')
            torch.cuda.manual_seed_all(config.seed); torch.backends.cudnn.benchmark = False
            torch.cuda.reset_peak_memory_stats()
        self.net.to(device)
        refine_params = []; pretrained_params = []
        for name, parameter in self.net.named_parameters():
            (refine_params if name.startswith('refine') or name.startswith(('blocks.6.', 'blocks.7.')) else pretrained_params).append(parameter)
        optimizer = torch.optim.AdamW([dict(params=pretrained_params, lr=config.learning_rate),
            dict(params=refine_params, lr=config.refinement_learning_rate)], weight_decay=.001)
        scaler = torch.amp.GradScaler('cuda', enabled=device=='cuda' and config.mixed_precision)
        rng = np.random.default_rng(config.seed); start = 0
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if saved['config'] != asdict(config): raise ValueError('Checkpoint training protocol changed')
            self.net.load_state_dict(saved['network']); optimizer.load_state_dict(saved['optimizer'])
            scaler.load_state_dict(saved['scaler']); self.history = saved['history']; start = saved['epoch']
            if len(self.history) != start or start > config.epochs: raise ValueError('Invalid checkpoint epoch/history')
            self.best_value = saved['best_value']; self.best_epoch = saved['best_epoch']
            rng.bit_generator.state = saved['numpy_rng']; torch.set_rng_state(saved['torch_rng'])
            if device=='cuda' and saved['cuda_rng'] is not None: torch.cuda.set_rng_state_all(saved['cuda_rng'])
            del saved
        arrays = []
        for features, y, scenario_weight in cases:
            edge = boundary_targets(y)
            candidates = np.flatnonzero(edge.max(1) > .9)
            arrays.append((features, y, edge, sample_weights(y), candidates, scenario_weight))
        selection = np.array([len(y)*w for _, y, _, _, _, w in arrays]); selection /= selection.sum()
        for epoch in range(start, config.epochs):
            self.net.to(device).train(); epoch_start = time.monotonic(); total = 0.; successful_steps = 0
            for step in range(config.steps):
                xx = []; yy = []; ee = []; ww = []
                for _ in range(config.batch):
                    pair, y, edge, weight, boundaries, _ = arrays[rng.choice(len(arrays), p=selection)]
                    first = int(rng.choice(boundaries)-config.core//2) if len(boundaries) and rng.random()<.5 else int(rng.integers(max(1,len(y)-config.core+1)))
                    first = max(0, min(first, max(0,len(y)-config.core))); side = int(rng.integers(2))
                    left = first-self.net.halo; right = first+config.core+self.net.halo
                    a, b = max(0,left), min(len(y),right); count = min(config.core,len(y)-first)
                    x = np.pad(self.transform(pair[side][a:b]), ((a-left,right-b),(0,0)))
                    target = np.full(config.core,3,np.int64); target[:count] = y[first:first+count]
                    e = np.zeros((config.core,6),np.float32); e[:count] = edge[first:first+count]
                    w = np.zeros(config.core,np.float32); w[:count] = weight[first:first+count]
                    if side: target=CLASS_SWAP[target]; e=e[:,EDGE_SWAP]
                    xx.append(x.T); yy.append(target); ee.append(e.T); ww.append(w)
                x = torch.from_numpy(np.stack(xx)).to(device)
                # CLASS_SWAP uses platform-native NumPy integers (int32 on Windows).
                # Cross entropy requires int64, regardless of which sides were sampled.
                target = torch.from_numpy(np.stack(yy).astype(np.int64)).to(device=device, dtype=torch.long)
                e = torch.from_numpy(np.stack(ee)).to(device); w = torch.from_numpy(np.stack(ww)).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=device=='cuda' and config.mixed_precision):
                    outputs = [(p[:,:,self.net.halo:self.net.halo+config.core], q[:,:,self.net.halo:self.net.halo+config.core]) for p,q in self.net(x)]
                    loss = temporal_loss(outputs,target,e,w,config)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss; preserve checkpoint')
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),3.)
                old_scale=scaler.get_scale(); scaler.step(optimizer); scaler.update()
                successful_steps += scaler.get_scale() >= old_scale
                total += float(loss.detach().cpu())
                if (step+1)%40==0 or step+1==config.steps:
                    progress(dict(stage='train_step',epoch=epoch+1,step=step+1,steps=config.steps,
                        loss=total/(step+1),elapsed_seconds=time.monotonic()-epoch_start))
            record=dict(epoch=epoch+1,loss=total/config.steps,optimizer_updates=int(successful_steps),
                seconds=time.monotonic()-epoch_start,peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2 if device=='cuda' else 0.)
            if successful_steps == 0: raise FloatingPointError('AMP skipped all optimizer steps')
            self.history.append(record)
            if validate is not None and ((epoch+1)%config.validation_every==0 or epoch+1==config.epochs):
                value = float(validate(self,epoch+1))
                if not np.isfinite(value) or not 0<=value<=1: raise ValueError('Invalid calibration-only score')
                record['calibration_spellF1'] = value
                if value>self.best_value:
                    self.best_value=value; self.best_epoch=epoch+1
                    from .run import save_model
                    self.net.cpu().eval(); save_model(checkpoint.parent/'best_uncalibrated.joblib',self); self.net.to(device)
            temp=checkpoint.with_suffix('.tmp')
            torch.save(dict(config=asdict(config),network=self.net.state_dict(),optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(),history=self.history,epoch=epoch+1,best_value=self.best_value,
                best_epoch=self.best_epoch,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                cuda_rng=[x.cpu() for x in torch.cuda.get_rng_state_all()] if device=='cuda' else None),temp)
            temp.replace(checkpoint); progress(dict(stage='epoch_complete',**record))
            if stop_after_epoch is not None and epoch+1>=stop_after_epoch: break
        self.net.cpu().eval(); del optimizer,scaler,arrays; gc.collect()
        if device=='cuda': torch.cuda.empty_cache()
        return self
