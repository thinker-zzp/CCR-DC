import numpy as np

from NeuroEA_GEA.Problems import Problem


class SOP_F6(Problem):
    """Step function."""

    def setting(self):
        self.M = 1
        self.lower = np.full(self.D, -100.0)
        self.upper = np.full(self.D, 100.0)
        self.encoding = 1

    def cal_obj(self, pop_dec):
        return np.sum(np.floor(np.asarray(pop_dec) + 0.5) ** 2, axis=1, keepdims=True)

