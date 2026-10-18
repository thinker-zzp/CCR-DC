import numpy as np

from NeuroEA_GEA.Problems import Problem


class SOP_F2(Problem):
    """Schwefel's function 2.22."""

    def setting(self):
        self.M = 1
        self.lower = np.full(self.D, -10.0)
        self.upper = np.full(self.D, 10.0)
        self.encoding = 1

    def cal_obj(self, pop_dec):
        absolute = np.abs(np.asarray(pop_dec))
        values = np.sum(absolute, axis=1) + np.prod(absolute, axis=1)
        return values[:, None]

