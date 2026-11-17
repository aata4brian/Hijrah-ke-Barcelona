"""Model status dan enam arah batas; kalibrasi memakai pertandingan terpisah."""
from dataclasses import dataclass
from pathlib import Path
import gc
import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit, softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from .labels import CLASS_SWAP, EDGE_SWAP, boundary_targets, sample_weights, train_indices


def aligned(model, x, classes):
    result = np.zeros((len(x),classes),dtype=np.float32)
    with threadpool_limits(limits=4):
        result[:,model.classes_.astype(int)] = model.predict_proba(x)
    return result


def head(x,y,w,iterations=90):
    model = HistGradientBoostingClassifier(max_iter=iterations, max_leaf_nodes=25,
        min_samples_leaf=64,learning_rate=.09,l2_regularization=1.3,
        early_stopping=False,random_state=331)
    with threadpool_limits(limits=4):
        model.fit(x,y,sample_weight=w)
    return model


def switch_context(x, indices=None):
    # The original switch head used +/-5 steps on a five-frame grid.
    # Keep that +/-1 second context at the new full-frame resolution.
    t=np.arange(len(x)) if indices is None else np.asarray(indices)
    before=x[np.maximum(t-25,0)];after=x[np.minimum(t+25,len(x)-1)]
    return np.column_stack([x[t],after-before]).astype(np.float32)


@dataclass
class Calibration:
    temperature: float = 1.
    slopes: object = None
    intercepts: object = None

    def state(self,p):
        return softmax(np.log(np.clip(p,1e-7,1))/self.temperature,axis=1).astype(np.float32)

    def edge(self,q):
        if self.slopes is None:
            return q
        return expit(logit(np.clip(q,1e-6,1-1e-6))*self.slopes+self.intercepts).astype(np.float32)


def calibrate(prob, edge, truth):
    p,q,y = np.concatenate(prob),np.concatenate(edge),np.concatenate(truth)
    valid = y < 3
    logs = np.log(np.clip(p[valid],1e-7,1)); actual=y[valid]
    def loss(t):
        z = softmax(logs/t,axis=1)
        return -np.log(np.clip(z[np.arange(len(z)),actual],1e-9,1)).mean()
    t=float(minimize_scalar(loss,bounds=(.4,3.),method='bounded').x)
    # Build targets per half, never invent an edge across half time.
    target = np.concatenate([boundary_targets(v,radius=2)>0 for v in truth])
    slopes, offsets = np.zeros(6),np.zeros(6)
    # Tie home/away counterparts so calibration preserves team symmetry.
    for group in ((0,1),(2,4),(3,5)):
        yy=target[valid][:,group].reshape(-1).astype(int)
        xx=logit(np.clip(q[valid][:,group].reshape(-1),1e-6,1-1e-6))[:,None]
        if len(np.unique(yy)) == 2:
            model=LogisticRegression(C=1.,max_iter=150).fit(xx,yy)
            slopes[list(group)]=max(0.,float(model.coef_[0,0]))
            offsets[list(group)]=float(model.intercept_[0])
        else:
            offsets[list(group)]=float(logit((yy.sum()+1)/(len(yy)+2)))
    return Calibration(t,np.asarray(slopes),np.asarray(offsets))


class Heads:
    def __init__(self,relative=True,weighted=True):
        self.relative=relative; self.weighted=weighted
        self.calibration=Calibration()

    def fit(self,cases,iterations=90,work_dir=None,fit_edges=True):
        # Build one head at a time; full dense context is never materialised.
        selections=[train_indices(y,step=10) for _,y,_ in cases]
        n=sum(2*len(idx) for idx in selections)
        self.n_features=cases[0][0][0].shape[1]
        self.edge_model=None
        work_dir=Path(work_dir) if work_dir is not None else None
        if work_dir is not None:work_dir.mkdir(parents=True,exist_ok=True)
        for name in ('state','edge') if fit_edges else ('state',):
            checkpoint=work_dir/f'{name}.joblib' if work_dir is not None else None
            if checkpoint is not None and checkpoint.exists():
                setattr(self,name+'_model',joblib.load(checkpoint));continue
            width=self.n_features*(2 if name=='edge' else 1)
            matrix_path=work_dir/f'{name}_samples.npy' if work_dir is not None else None
            x=(np.lib.format.open_memmap(matrix_path,mode='w+',dtype=np.float32,shape=(n,width))
               if matrix_path else np.empty((n,width),np.float32))
            target=np.empty(n,np.int8);weights=np.empty(n,np.float32);first=0
            for (pair,y,scenario_weight),idx in zip(cases,selections):
                if name=='state':
                    labels=y;w=sample_weights(y) if self.weighted else (y<3).astype(float)
                else:
                    edges=boundary_targets(y,radius=2)
                    labels=np.where(edges.max(axis=1)>0,edges.argmax(axis=1)+1,0)
                    w=np.where(labels>0,15.,1.)
                for side in range(2):
                    last=first+len(idx)
                    x[first:last]=switch_context(pair[side],idx) if name=='edge' else pair[side][idx]
                    yy=labels[idx].copy()
                    if side:
                        if name=='state':yy=CLASS_SWAP[yy]
                        else:
                            positive=yy>0;yy[positive]=EDGE_SWAP[yy[positive]-1]+1
                    target[first:last]=yy;weights[first:last]=w[idx]*scenario_weight
                    first=last
            if matrix_path:x.flush()
            model=head(x,target,weights,iterations)
            setattr(self,name+'_model',model)
            if checkpoint is not None:
                tmp=checkpoint.with_suffix('.tmp');joblib.dump(model,tmp,compress=3);tmp.replace(checkpoint)
            if matrix_path:
                x._mmap.close();del x;matrix_path.unlink()
            else:del x
            del target,weights;gc.collect()
        return self

    def state_probability(self,pair,chunk=8192):
        out=np.empty((len(pair[0]),3),np.float32)
        for first in range(0,len(out),chunk):
            last=min(len(out),first+chunk)
            a,b=(x[first:last] for x in pair)
            out[first:last]=(aligned(self.state_model,a,3)+aligned(self.state_model,b,3)[:,[1,0,2]])/2
        return out

    def raw(self,pair,chunk=8192):
        p=self.state_probability(pair,chunk);q=np.zeros((len(p),6),np.float32)
        if self.edge_model is not None:
            for first in range(0,len(p),chunk):
                idx=np.arange(first,min(len(p),first+chunk))
                a=aligned(self.edge_model,switch_context(pair[0],idx),7)[:,1:]
                b=aligned(self.edge_model,switch_context(pair[1],idx),7)[:,1:][:,EDGE_SWAP]
                q[idx]=(a+b)/2
        return p,q

    def predict(self,pair,calibrated=True):
        p,q=self.raw(pair)
        return (self.calibration.state(p),self.calibration.edge(q)) if calibrated else (p,q)


def calibration_report(raw,calibrated,truth):
    p,q=np.concatenate(raw[0]),np.concatenate(raw[1])
    cp,cq=np.concatenate(calibrated[0]),np.concatenate(calibrated[1])
    y=np.concatenate(truth); valid=y<3
    target=np.concatenate([boundary_targets(t,radius=2)>0 for t in truth])
    report={}
    for name,pp,qq in [('raw',p,q),('calibrated',cp,cq)]:
        report[name]={'state_nll':float(-np.log(np.clip(pp[valid][np.arange(valid.sum()),y[valid]],1e-8,1)).mean()),
                      'edge_brier':float(((qq[valid]-target[valid])**2).mean()),
                      'edge_prevalence':float(target[valid].mean()),'edge_mean_probability':float(qq[valid].mean())}
    return report



class Ensemble:
    def __init__(self,trees,temporal):
        self.trees=trees;self.temporal=temporal;self.decoder=trees.decoder

    def predict(self,pair):
        p,q=self.trees.predict(pair);tp,tq=self.temporal.predict(pair)
        return (p+tp)/2,(q+tq)/2
