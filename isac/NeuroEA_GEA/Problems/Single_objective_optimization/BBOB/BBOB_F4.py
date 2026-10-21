# PlatEMO-derived benchmark material; Copyright (c) 2026 BIMK Group.
# Research-use and publication citation terms: see licenses/PlatEMO-NOTICE.txt.
# Excluded from the root MIT grant. Preserve inherited NeuroEA_RL notices too.
# See LICENSING.md for scope.

import numpy as np

from NeuroEA_GEA.Problems.PROBLEM import Problem
from ._platemo import dimension_ratio, expand_xopt, fpen, tosz


class BBOB_F4(Problem):
    """Bueche-Rastrigin function, ported from PlatEMO 2026."""

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
        z = tosz(X - self.x_opt)
        scale = np.tile(
            10.0 ** (0.5 * dimension_ratio(self.D)), (len(X), 1)
        )
        columns = np.arange(0, self.D, 2)
        scale[:, columns] *= 10.0 ** (z[:, columns] > 0)
        z *= scale
        value = (
            10.0 * (self.D - np.sum(np.cos(2.0 * np.pi * z), axis=1))
            + np.sum(z**2, axis=1)
            + 100.0 * fpen(X)
        )
        return value.reshape(-1, 1)

    Tosz = staticmethod(tosz)
    fpen = staticmethod(fpen)
