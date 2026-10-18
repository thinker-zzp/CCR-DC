import numpy as np

from NeuroEA_GEA.utils.nd_sort import nd_sort


class Solution:
    """
    SOLUTION 类 - 存储单个解的所有属性
    """

    def __init__(self, dec=None, obj=None, con=None, add=None):
        self.dec = dec  # 决策变量 (1D array)
        self.obj = obj  # 目标值 (1D array)
        self.con = con  # 约束违反 (1D array)
        self.add = add  # 附加属性 (如速度、历史记录)

    # 以下静态方法用于处理 Solution 对象列表（模拟 MATLAB 的批量获取方法）
    @staticmethod
    def get_decs(population):
        return np.vstack([p.dec for p in population]) if population else np.array([])

    @staticmethod
    def get_objs(population):
        return np.vstack([p.obj for p in population]) if population else np.array([])

    @staticmethod
    def get_cons(population):
        return np.vstack([p.con for p in population]) if population else np.array([])

    @staticmethod
    def get_best(population):
        """
        获取种群中的最佳解（可行且非支配）
        """
        if not population:
            return []

        # 1. 筛选可行解
        cons = Solution.get_cons(population)
        feasible_idx = np.where(np.all(cons <= 0, axis=1))[0]

        if len(feasible_idx) == 0:
            return []

        feasible_pop = [population[i] for i in feasible_idx]
        objs = Solution.get_objs(feasible_pop)

        # 2. 根据目标数量判断
        if objs.shape[1] > 1:
            # 多目标：调用非支配排序获取第一层 (FrontNo == 1)
            # from selection_utils import nd_sort
            front_no, _ = nd_sort(objs, np.zeros((len(feasible_pop), 1)), 1)
            best_idx = np.where(front_no == 1)[0]
            return [feasible_pop[i] for i in best_idx]
        else:
            # 单目标：取最小值
            best_idx = np.argmin(objs.flatten())
            return [feasible_pop[best_idx]]