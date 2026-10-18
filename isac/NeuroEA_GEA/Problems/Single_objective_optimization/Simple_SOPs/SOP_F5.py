import numpy as np

from NeuroEA_GEA.Problems import Problem


class SOP_F5(Problem):
    """Generalized Rosenbrock function."""

    def setting(self):
        self.M = 1
        self.lower = np.full(self.D, -30.0)
        self.upper = np.full(self.D, 30.0)
        self.encoding = 1

    def cal_obj(self, pop_dec):
        x = np.asarray(pop_dec)
        values = np.sum(
            100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2
            + (x[:, :-1] - 1.0) ** 2,
            axis=1,
        )
        return values[:, None]

