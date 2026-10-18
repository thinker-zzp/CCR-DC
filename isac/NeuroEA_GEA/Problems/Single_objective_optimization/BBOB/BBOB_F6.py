import numpy as np

from NeuroEA_GEA.Problems.PROBLEM import Problem
from ._platemo import (
    dimension_ratio,
    expand_xopt,
    platemo_rotation,
    tosz,
)


class BBOB_F6(Problem):
    """Attractive sector function, ported from PlatEMO 2026."""

    def __init__(self, **kwargs):
        self.xopt_val = kwargs.get("xopt", 0)
        super().__init__(**kwargs)

    def setting(self):
        self.M = 1
        self.D = 30 if self.D is None else self.D
        self.x_opt = expand_xopt(self.xopt_val, self.D)
        self.lower = np.full(self.D, -5.0)
        self.upper = np.full(self.D, 5.0)
        self.Q = platemo_rotation(1, self.D)
        self.R = platemo_rotation(2, self.D)

    def cal_obj(self, X):
        X = np.asarray(X, dtype=float)
        scaling = np.diag(10.0 ** (0.5 * dimension_ratio(self.D)))
        z = (X - self.x_opt) @ (self.Q @ scaling @ self.R)
        sector = np.ones_like(z)
        sector[z * self.x_opt > 0] = 100.0
        value = tosz(np.sum((sector * z) ** 2, axis=1)) ** 0.9
        return value.reshape(-1, 1)

    Tosz = staticmethod(tosz)
