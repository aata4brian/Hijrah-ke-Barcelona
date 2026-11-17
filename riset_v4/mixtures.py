"""Stable import path for deployment-safe, calibrated mixtures."""
import numpy as np


def blend(arrays, state_weight, edge_weight):
    if not all(np.isfinite(w) and 0 <= w <= 1 for w in (state_weight, edge_weight)):
        raise ValueError('Mixture weights must lie in [0, 1]')
    p, q, tp, tq = arrays
    if p.shape != tp.shape or q.shape != tq.shape:
        raise ValueError('Mixture probability shapes differ')
    return (state_weight*p + (1-state_weight)*tp,
            edge_weight*q + (1-edge_weight)*tq)


class Mixture:
    """Never defined in __main__; safe to load in another Python process."""
    def __init__(self, trees, temporal, state_weight, edge_weight, decoder):
        if not all(np.isfinite(w) and 0 <= w <= 1 for w in (state_weight, edge_weight)):
            raise ValueError('Mixture weights must lie in [0, 1]')
        self.trees, self.temporal = trees, temporal
        self.state_weight, self.edge_weight = state_weight, edge_weight
        self.decoder = decoder

    def predict(self, features):
        p, q = self.trees.predict(features)
        tp, tq = self.temporal.predict(features)
        return blend((p, q, tp, tq), self.state_weight, self.edge_weight)
