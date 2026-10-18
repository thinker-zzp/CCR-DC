import numpy as np

from NeuroEA_GEA.Algorithms.NEUROEA.BLOCK import Block


class Block_Mutation(Block):
    """
    统一变异模块：基于学习权重的参数化扰动
    """

    def __init__(self, n_sets=5):
        super().__init__()
        self.n_sets = n_sets
        self.n_dec = 1  # 决策变量维度，会在运行时更新

        # 每个参数集包含 2 个值：[扰动强度 sigma, 扰动权重/概率]
        self.lower = np.tile([0.0, 1e-20], self.n_sets)
        self.upper = np.tile([1.0, 2.0], self.n_sets)

        # 随机初始化参数 (MATLAB 中 upper 设置为 1 而非 self.upper 的原始定义，此处遵循逻辑)
        self.parameter = np.random.uniform(self.lower, np.ones(2 * self.n_sets))

        self.weight = None
        self.fit = None
        self.parameter_assign()

    def parameter_assign(self):
        """
        重构参数矩阵并计算变异概率分布
        """
        # 将参数重塑为 (n_sets, 2)
        weight = self.parameter.reshape(self.n_sets, 2).copy()

        # 1. 调整概率项：将原始权重除以变量维度 (Weight(:,end) / nDec)
        # 这是为了控制每个基因位发生变异的期望总数
        weight[:, 1] = weight[:, 1] / self.n_dec

        # 2. 计算“不做变异”的剩余概率 (1 - sum(probabilities))
        remain_prob = max(0.0, 1.0 - np.sum(weight[:, 1]))

        # 3. 构建最终权重矩阵：包含 n_sets 个变异器 + 1 个零扰动器
        # 最后一行为 [扰动强度=0, 概率=remain_prob]
        self.weight = np.vstack([weight, [0.0, remain_prob]])

        # 4. 计算累积概率 (Fit)，用于轮盘赌采样
        cumsum_probs = np.cumsum(self.weight[:, 1])
        self.fit = cumsum_probs / max(cumsum_probs)

    def main(self, problem, precursors, ratio):
        # 1. 收集父代决策变量 (in_type=2, multiple=1)
        parent_dec = self.gather(problem, precursors, ratio, 2, 1)
        if parent_dec.size == 0:
            self.output = np.array([])
            return

        # print(f"\n[Mutation Debug] Gen {getattr(problem, 'current_gen', 'unknown')}:")
        # for i in range(self.n_sets):
        #     sigma = self.weight[i, 0]
        #     prob = self.weight[i, 1]
        #     print(f"  Set {i}: Sigma={sigma:.6f}, Prob={prob:.6f}")
        # print(f"  Set {self.n_sets} (No Mutation): Prob={self.weight[-1, 1]:.6f}")

        # 2. 如果决策变量维度发生变化（首次运行或动态问题），重新计算权重
        if parent_dec.shape[1] != self.n_dec:
            self.n_dec = parent_dec.shape[1]
            self.parameter_assign()

        # 3. 采样扰动量 r
        # r 的维度与 parent_dec 一致 (N, D)
        r = self.para_sampling(parent_dec.shape, self.weight[:, 0], self.fit)

        # 4. 执行变异：Offspring = Parent + (Upper - Lower) * r
        # 这里的 r 已经包含了强度和概率的综合影响
        range_dec = problem.upper - problem.lower
        off_dec = parent_dec + range_dec * r

        # 5. 边界约束
        self.output = np.clip(off_dec, problem.lower, problem.upper)

    def para_sampling(self, shape, weights, fit):
        """
        参数化采样逻辑：决定每个基因位使用哪种强度的正态扰动
        """
        # 初始采样标准正态分布
        r = np.random.standard_normal(shape)

        # 为每个基因位分配一个变异器类型 (基于 fit 概率)
        rand_vals = np.random.rand(*shape)
        types = np.searchsorted(fit, rand_vals)

        # 根据选定的类型乘以对应的强度 (weight)
        for i in range(len(fit)):
            mask = (types == i)
            if np.any(mask):
                r[mask] = r[mask] * weights[i]

        return r