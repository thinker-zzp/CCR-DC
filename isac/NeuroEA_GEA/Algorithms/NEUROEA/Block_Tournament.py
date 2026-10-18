import numpy as np
from NeuroEA_GEA.Algorithms.NEUROEA.BLOCK import Block
from NeuroEA_GEA.utils.get_fitness_single import get_fitness_single
from NeuroEA_GEA.utils.nd_sort import nd_sort


class Block_Tournament(Block):
    """
    锦标赛选择模块：参数化 K-Tournament 选择
    """

    def __init__(self, n_parents=100, upper_k=2):
        super().__init__()
        self.n_parents = n_parents
        self.lower = np.array([1.0])
        self.upper = np.array([float(upper_k)])
        self.parameter = np.random.uniform(self.lower, self.upper)
        self.n_tournament = 1
        self.parameter_assign()

    def parameter_assign(self):
        self.n_tournament = int(np.round(self.parameter[0]))

    def main(self, problem, precursors, ratio):
        population = self.gather(problem, precursors, ratio, 1, 1)
        if not population:
            self.output = []
            return

        n_pop = len(population)

        # 处理锦标赛规模大于种群数量的边界情况
        k = min(self.n_tournament, n_pop)

        # 计算所有个体的排名 (Rank)
        if problem.m == 1:
            # 单目标：适应度越小，Rank 越小
            fitness = get_fitness_single(population)
            # argsort 的 argsort 得到的是该元素在数组中的“排名位次”
            rank = np.argsort(np.argsort(fitness))
        else:
            # 多目标：非支配层级作为 Rank
            pop_objs = np.array([p.obj for p in population])
            pop_cons = np.array([p.con for p in population])
            # rank 即为 FrontNo，第一层为 1，第二层为 2...
            rank, _ = nd_sort(pop_objs, pop_cons, n_pop)

        # 随机挑选候选者
        # candidates 形状: (k, n_parents)
        candidates = np.random.randint(0, n_pop, size=(k, self.n_parents))

        # 获取对应的排名
        candidate_ranks = rank[candidates]

        # 找到每组锦标赛（每一列）中 Rank 最小的行索引
        best_row_indices = np.argmin(candidate_ranks, axis=0)

        # 提取最终选中的个体索引
        final_indices = candidates[best_row_indices, np.arange(self.n_parents)]

        self.output = [population[i] for i in final_indices]