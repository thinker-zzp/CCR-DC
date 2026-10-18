import numpy as np

from NeuroEA_GEA.Problems.PROBLEM import Problem
from ._platemo import dimension_ratio, platemo_f5_xopt


class BBOB_F5(Problem):
    """Linear slope function, ported from PlatEMO 2026."""

    def setting(self):
        self.M = 1
        self.D = 30 if self.D is None else self.D
        self.x_opt = platemo_f5_xopt(self.D)
        self.lower = np.full(self.D, -5.0)
        self.upper = np.full(self.D, 5.0)

    def cal_obj(self, X):
        X = np.asarray(X, dtype=float)
        z = X.copy()
        xopt = np.broadcast_to(self.x_opt, X.shape)
        mask = xopt * X >= 25.0
        z[mask] = xopt[mask]
        scale = np.sign(self.x_opt) * 10.0 ** dimension_ratio(self.D)
        value = np.sum(5.0 * np.abs(scale) - scale * z, axis=1)
        return value.reshape(-1, 1)
