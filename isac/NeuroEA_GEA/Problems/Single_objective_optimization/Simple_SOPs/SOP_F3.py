import numpy as np

from NeuroEA_GEA.Problems import Problem


class SOP_F3(Problem):
    """Schwefel's function 1.2."""

    def setting(self):
        self.M = 1
        self.lower = np.full(self.D, -100.0)
        self.upper = np.full(self.D, 100.0)
        self.encoding = 1

    def cal_obj(self, pop_dec):
        cumulative = np.cumsum(np.asarray(pop_dec), axis=1)
        return np.sum(cumulative ** 2, axis=1, keepdims=True)

