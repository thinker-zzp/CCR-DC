import numpy as np

from NeuroEA_GEA.Problems.PROBLEM import Problem
from ._platemo import (
    dimension_ratio,
    expand_xopt,
    fpen,
    platemo_rotation,
)


class BBOB_F7(Problem):
    """Step ellipsoidal function, ported from PlatEMO 2026."""

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
        z = (X - self.x_opt) @ (scaling @ self.R)
        zh = np.floor(0.5 + 10.0 * z) / 10.0
        mask = np.abs(z) > 0.5
        zh[mask] = np.floor(0.5 + z[mask])
        first = np.abs(zh[:, 0]) / 1.0e4
        ellipsoid = np.sum(
            10.0 ** (2.0 * dimension_ratio(self.D)) * (z @ self.Q) ** 2,
            axis=1,
        )
        value = 0.1 * np.maximum(first, ellipsoid) + fpen(X)
        return value.reshape(-1, 1)

    fpen = staticmethod(fpen)
