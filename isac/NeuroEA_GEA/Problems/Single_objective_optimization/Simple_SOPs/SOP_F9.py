import numpy as np

from NeuroEA_GEA.Problems import Problem


class SOP_F9(Problem):
    """Rastrigin function."""

    def setting(self):
        self.M = 1
        self.lower = np.full(self.D, -5.12)
        self.upper = np.full(self.D, 5.12)
        self.encoding = 1

    def cal_obj(self, pop_dec):
        x = np.asarray(pop_dec)
        values = np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x) + 10.0, axis=1)
        return values[:, None]

