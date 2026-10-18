import numpy as np

from NeuroEA_GEA.Algorithms.NEUROEA.BLOCK import Block


class Block_Crossover(Block):
    """
    统一交叉模块：基于参数化权重的多父代交叉
    """

    def __init__(self, n_parents=2, n_sets=5):
        super().__init__()
        self.n_parents = n_parents
        self.n_sets = n_sets

        # 参数结构：每个 parent(除第一个) 对应 3 个值 [std, mean, selection_prob]
        # 总参数量 = (n_parents - 1) * n_sets * 3
        param_dim = (self.n_parents - 1) * self.n_sets * 3
        self.lower = np.tile([0.0, -1.0, 1e-20], (self.n_parents - 1) * self.n_sets)
        self.upper = np.ones(param_dim)

        # 初始化参数
        self.parameter = np.random.uniform(self.lower, self.upper)
        self.weight = None
        self.fit = None
        self.parameter_assign()

    def parameter_assign(self):
        """将一维参数数组重构为权重矩阵和累积概率"""
        # reshape 为 (n_sets, (n_parents-1)*3)
        reshaped = self.parameter.reshape(self.n_sets, -1)
        self.weight = reshaped

        # 提取每个 parent 的第 3 个参数作为选择概率 (Fit)
        # 对应 MATLAB 的 obj.Weight(:,3:3:end)
        probs = reshaped[:, 2::3]
        # 计算归一化的累积概率，用于后续的权重集抽样
        cumsum_probs = np.cumsum(probs, axis=0)
        self.fit = cumsum_probs / cumsum_probs[-1, :]

    def main(self, problem, precursors, ratio):
        # 1. 收集父代决策变量 (in_type=2, multiple=n_parents)
        parent_dec = self.gather(problem, precursors, ratio, 2, self.n_parents)
        if parent_dec.size == 0:
            self.output = np.array([])
            return

        n_offspring = parent_dec.shape[0] // self.n_parents
        n_vars = parent_dec.shape[1]

        # 2. 为每个父代（除第一个外）采样权重 R
        # R 的形状将是 (n_offspring * (n_parents-1), n_vars)
        r_list = []
        for i in range(self.n_parents - 1):
            # 获取对应的 [std, mean] 组
            w_sub = self.weight[:, i * 3: i * 3 + 2]
            f_sub = self.fit[:, i]
            r_sub = self.para_sampling((n_offspring, n_vars), w_sub, f_sub)
            r_list.append(r_sub)

        # R_all 对应 MATLAB 的 R，通过交错存放
        r_weights = np.stack(r_list, axis=1)  # (n_offspring, n_parents-1, n_vars)

        # 3. 计算子代：Offspring = (1 - sum(r)) * Parent1 + r2*Parent2 + r3*Parent3...
        # 将父代重塑为 (n_offspring, n_parents, n_vars)
        parents_reshaped = parent_dec.reshape(n_offspring, self.n_parents, n_vars)

        # 计算所有父代的组合权重
        # w_first: (n_offspring, 1, n_vars)
        w_first = 1.0 - np.sum(r_weights, axis=1, keepdims=True)
        # combined_weights: (n_offspring, n_parents, n_vars)
        combined_weights = np.concatenate([w_first, r_weights], axis=1)

        # # 元素级相乘并求和
        off_dec = np.sum(combined_weights * parents_reshaped, axis=1)

        # 强力修正：确保形状是 (N, D) 而不是 (1, N) 或 (N,)
        if off_dec.ndim == 1:
            off_dec = off_dec.reshape(-1, n_vars)
        elif off_dec.shape[1] != n_vars:
            # 如果发生了奇怪的转置，强制转回来
            off_dec = off_dec.reshape(-1, n_vars)

        self.output = off_dec

    def para_sampling(self, shape, weight, fit):
        """参数采样逻辑"""
        n_samples, n_vars = shape
        r = np.random.standard_normal(shape)

        # 决定每个基因位使用哪一组 [std, mean]
        rand_vals = np.random.rand(n_samples, n_vars)
        # 使用 searchsorted 替代 MATLAB 的 arrayfun(find) 提高效率
        types = np.array([np.searchsorted(fit, rv) for rv in rand_vals.flat]).reshape(shape)

        for i in range(len(fit)):
            mask = (types == i)
            if np.any(mask):
                # r = normal * std + mean
                r[mask] = r[mask] * weight[i, 0] + weight[i, 1]
        return r