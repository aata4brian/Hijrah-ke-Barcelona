"""Strict local checks; never upload tracking or target labels."""
from pathlib import Path
import numpy as np


def validate_probabilities(probabilities, boundaries, frames):
    p, q = np.asarray(probabilities), np.asarray(boundaries)
    if p.shape != (frames, 3) or q.shape != (frames, 6):
        raise ValueError('Model probability shape mismatch')
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Model probabilities must be finite')
    if np.any((p < 0) | (p > 1)) or np.any((q < 0) | (q > 1)):
        raise ValueError('Model probabilities must lie in [0, 1]')
    if not np.allclose(p.sum(axis=1), 1., atol=1e-5, rtol=0):
        raise ValueError('State probabilities must sum to one')


def load_recovery(path, frames, fps=25):
    with np.load(Path(path), allow_pickle=False) as archive:
        required = ('ball', 'original_ball', 'recovered', 'confidence', 'fps')
        if any(key not in archive.files for key in required):
            raise ValueError('Recovery sidecar is missing required arrays')
        sidecar = {key: archive[key] for key in required}
    if sidecar['fps'].shape != () or sidecar['fps'].item() != fps:
        raise ValueError('Recovery frame rate mismatch')
    for name in ('ball', 'original_ball'):
        if sidecar[name].shape != (frames, 2):
            raise ValueError('Recovery coordinate shape mismatch')
        if sidecar[name].dtype.kind not in 'fi':
            raise ValueError('Recovery coordinates must be numeric')
        if np.isinf(sidecar[name]).any():
            raise ValueError('Recovery coordinates contain infinity')
    mask, confidence = sidecar['recovered'], sidecar['confidence']
    if mask.shape != (frames,) or mask.dtype.kind != 'b':
        raise ValueError('Recovery mask must be a boolean vector')
    if confidence.shape != (frames,) or confidence.dtype.kind not in 'fi':
        raise ValueError('Recovery confidence shape or type mismatch')
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError('Recovery confidence must be finite and lie in [0, 1]')
    observed = np.isfinite(sidecar['original_ball']).all(axis=1)
    if np.any(mask & observed):
        raise ValueError('Recovery mask attempts to overwrite observed tracking')
    if not np.array_equal(sidecar['ball'][observed], sidecar['original_ball'][observed]):
        raise ValueError('Recovery sidecar changed observed tracking')
    if not np.isfinite(sidecar['ball'][mask]).all():
        raise ValueError('Recovered positions must be finite')
    if not np.allclose(sidecar['ball'][~mask], sidecar['original_ball'][~mask],
                       equal_nan=True, atol=0, rtol=0):
        raise ValueError('Recovery sidecar changed positions outside its mask')
    return sidecar
