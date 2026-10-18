import numpy as np

from NeuroEA_GEA.Problems.PROBLEM import Problem
from ._platemo import (
    dimension_ratio,
    expand_xopt,
    platemo_rotation,
    tosz,
)


class BBOB_F10(Problem):
    """Rotated ellipsoidal function, ported from PlatEMO 2026."""

    def __init__(self, **kwargs):
        self.xopt_val = kwargs.get("xopt", 0)
        super().__init__(**kwargs)

    def setting(self):
        self.M = 1
        self.D = 30 if self.D is None else self.D
        self.x_opt = expand_xopt(self.xopt_val, self.D)
        self.lower = np.full(self.D, -5.0)
        self.upper = np.full(self.D, 5.0)
        self.R = platemo_rotation(2, self.D)

    def cal_obj(self, X):
        X = np.asarray(X, dtype=float)
        z = tosz((X - self.x_opt) @ self.R)
        weights = 10.0 ** (6.0 * dimension_ratio(self.D))
        return np.sum(weights * z**2, axis=1, keepdims=True)

    Tosz = staticmethod(tosz)
