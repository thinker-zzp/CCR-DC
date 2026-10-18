import numpy as np

from NeuroEA_GEA.Problems import Problem


class SOP_F1(Problem):
    """Sphere function from the latest PlatEMO Simple SOP suite."""

    def setting(self):
        self.M = 1
        if self.D is None:
            self.D = 30
        self.lower = np.full(self.D, -100.0)
        self.upper = np.full(self.D, 100.0)
        self.encoding = 1

    def cal_obj(self, pop_dec):
        return np.sum(np.asarray(pop_dec) ** 2, axis=1, keepdims=True)

