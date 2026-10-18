import numpy as np

from NeuroEA_GEA.Algorithms.NEUROEA.BLOCK import Block


class Block_Exchange(Block):
    """
    交换模块：参数化离散重组
    """

    def __init__(self, n_parents=2):
        super().__init__()
        self.n_parents = n_parents
        self.lower = np.zeros(n_parents) + 1e-20
        self.upper = np.ones(n_parents)

        # 参数代表每个父代被选中的原始权重
        self.parameter = np.random.uniform(self.lower, self.upper)
        self.fitness = None
        self.parameter_assign()

    def parameter_assign(self):
        # 计算累积概率，用于决定每个基因位取自哪一个父代
        self.fitness = np.cumsum(self.parameter)
        self.fitness /= self.fitness[-1]

    def main(self, problem, precursors, ratio):
        # 收集父代决策变量，确保数量是 n_parents 的倍数
        parent_dec = self.gather(problem, precursors, ratio, 2, self.n_parents)
        if parent_dec.size == 0:
            self.output = np.array([])
            return

        n_offspring = parent_dec.shape[0] // self.n_parents
        n_vars = parent_dec.shape[1]

        # 1. 为子代的每一个基因位(n_offspring, n_vars) 采样一个父代索引
        rand_vals = np.random.rand(n_offspring, n_vars)
        # parent_selector 的值在 0 ~ n_parents-1 之间
        parent_selector = np.searchsorted(self.fitness, rand_vals)

        # 2. 计算在平铺的 parent_dec 中的索引位置
        # MATLAB 逻辑：selected = selected + (0:n_off-1)*n_parents + (0:n_vars-1)*total_parents

        # 生成子代决策变量矩阵
        off_dec = np.zeros((n_offspring, n_vars))

        # 将父代组织为 (n_offspring, n_parents, n_vars) 方便索引
        reshaped_parents = parent_dec.reshape(n_offspring, self.n_parents, n_vars)

        # 高级索引：从每个 offspring 对应的 n_parents 个候选中，
        # 根据 parent_selector 选出一个
        off_idx = np.arange(n_offspring)[:, None]
        var_idx = np.arange(n_vars)
        off_dec = reshaped_parents[off_idx, parent_selector, var_idx]

        self.output = off_dec