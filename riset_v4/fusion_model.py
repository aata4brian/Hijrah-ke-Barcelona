"""Small calibrated state specialist around frozen V4.5 probabilities."""
import numpy as np
from scipy.special import softmax
from .artifact_checks import validate_probabilities
from .labels import CLASS_SWAP
from .models import aligned


def specialist_probability(model, features, temperature=1.):
    p = (aligned(model, features[0], 3)+aligned(model, features[1], 3)[:, CLASS_SWAP[:3]])/2
    return softmax(np.log(np.clip(p, 1e-7, 1))/temperature, axis=1).astype(np.float32)


def combine_probabilities(deep, tree, specialist, tree_weight, specialist_weight):
    if not all(np.isfinite(w) and 0 <= w <= 1 for w in (tree_weight, specialist_weight)):
        raise ValueError('Weights must be in [0,1]')
    p, q = deep; tp, tq = tree
    validate_probabilities(p, q, len(p)); validate_probabilities(tp, tq, len(p))
    if specialist.shape != p.shape or not np.isfinite(specialist).all() or np.any(specialist < 0) or not np.allclose(specialist.sum(1), 1, atol=1e-5):
        raise ValueError('Malformed specialist probabilities')
    # A zero weight preserves the frozen baseline exactly, not a log-domain approximation.
    bp = p if tree_weight == 0 else (1-tree_weight)*p+tree_weight*tp
    bq = q if tree_weight == 0 else (1-tree_weight)*q+tree_weight*tq
    return (bp if specialist_weight == 0 else
            (1-specialist_weight)*bp+specialist_weight*specialist), bq
