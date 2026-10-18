import numpy as np

from NeuroEA_GEA.Problems.PROBLEM import Problem
from ._platemo import expand_xopt


class BBOB_F8(Problem):
    """Rosenbrock function, ported from PlatEMO 2026."""

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
        X = np.asarray(X, dtype=float)
        z = max(1.0, np.sqrt(self.D) / 8.0) * (X - self.x_opt) + 1.0
        value = np.sum(
            100.0 * (z[:, :-1] ** 2 - z[:, 1:]) ** 2
            + (z[:, :-1] - 1.0) ** 2,
            axis=1,
        )
        return value.reshape(-1, 1)
