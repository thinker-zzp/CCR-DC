import numpy as np

from NeuroEA_GEA.Problems import Problem


class SOP_F4(Problem):
    """Schwefel's function 2.21."""

    def setting(self):
        self.M = 1
        self.lower = np.full(self.D, -100.0)
        self.upper = np.full(self.D, 100.0)
        self.encoding = 1

    def cal_obj(self, pop_dec):
        return np.max(np.abs(np.asarray(pop_dec)), axis=1, keepdims=True)

