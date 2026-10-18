import numpy as np

from NeuroEA_GEA.Problems.PROBLEM import Problem
from ._platemo import expand_xopt


class BBOB_F1(Problem):
    """Sphere function, ported from PlatEMO 2026 BBOB_F1.m."""

    def __init__(self, **kwargs):
        self.xopt_val = kwargs.get("xopt", 0)
        super().__init__(**kwargs)

    def setting(self):
        self.M = 1
        self.D = 30 if self.D is None else self.D
        self.x_opt = expand_xopt(self.xopt_val, self.D)
        self.lower = np.full(self.D, -5.0)
        self.upper = np.full(self.D, 5.0)

    def cal_obj(self, X):
        z = np.asarray(X, dtype=float) - self.x_opt
        return np.sum(z**2, axis=1, keepdims=True)
