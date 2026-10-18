import numpy as np

from NeuroEA_GEA.Problems import Problem


class SOP_F10(Problem):
    """Shifted Ackley function from the current Trainable-EA benchmark code."""

    def setting(self):
        self.M = 1
        self.lower = np.full(self.D, -32.0)
        self.upper = np.full(self.D, 32.0)
        self.encoding = 1

    def cal_obj(self, pop_dec):
        x = np.asarray(pop_dec, dtype=np.float64)
        offset = np.linspace(0.0, float(self.upper[0]), self.D)
        shifted = x - offset
        term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.mean(shifted ** 2, axis=1)))
        term2 = -np.exp(np.mean(np.cos(2.0 * np.pi * shifted), axis=1))
        values = term1 + term2 + 20.0 + np.e
        return values[:, None]
