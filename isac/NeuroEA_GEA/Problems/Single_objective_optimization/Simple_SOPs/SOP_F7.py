import numpy as np

from NeuroEA_GEA.Problems import Problem


class SOP_F7(Problem):
    """Quartic function with uniform random noise."""

    def setting(self):
        self.M = 1
        self.lower = np.full(self.D, -1.28)
        self.upper = np.full(self.D, 1.28)
        self.encoding = 1

    def cal_obj(self, pop_dec):
        x = np.asarray(pop_dec)
        weights = np.arange(1, self.D + 1, dtype=np.float64)
        values = np.sum(weights * x ** 4, axis=1) + np.random.rand(x.shape[0])
        return values[:, None]

