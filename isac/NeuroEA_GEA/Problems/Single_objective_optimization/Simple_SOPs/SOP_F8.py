import numpy as np

from NeuroEA_GEA.Problems import Problem


class SOP_F8(Problem):
    """Generalized Schwefel's function 2.26."""

    def setting(self):
        self.M = 1
        self.lower = np.full(self.D, -500.0)
        self.upper = np.full(self.D, 500.0)
        self.encoding = 1

    def cal_obj(self, pop_dec):
        x = np.asarray(pop_dec)
        values = -np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)
        return values[:, None]

    def get_optimum(self, N):
        return np.full((int(N), 1), -418.9829 * self.D)

