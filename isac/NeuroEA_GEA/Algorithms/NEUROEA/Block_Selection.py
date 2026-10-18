import numpy as np


from NeuroEA_GEA.Algorithms.NEUROEA.BLOCK import Block
from NeuroEA_GEA.utils.nd_sort import nd_sort


class Block_Selection(Block):
    """
    环境选择模块：支持单目标适应度选择和多目标非支配排序选择
    """

    def __init__(self, n_solutions=100):
        super().__init__()
        self.n_solutions = n_solutions

    def main(self, problem, precursors, ratio):
        # 1. 收集所有前驱节点的 SOLUTION 对象 (in_type=1)
        population = self.gather(problem, precursors, ratio, 1, 1)
        if not population:
            self.output = []
            return

        n_pop = len(population)
        num_to_keep = min(n_pop, self.n_solutions)

        # 2. 根据目标维度执行选择策略
        if problem.m == 1:
            # 单目标优化：按适应度（通常是函数值）升序排列
            # 假设 Solution 对象有 fitness 属性或通过 problem 计算
            fitness = self._get_fitness_single(population)
            rank = np.argsort(fitness)
            self.output = [population[i] for i in rank[:num_to_keep]]
        else:
            # 多目标优化：非支配排序 + 截断操作
            pop_objs = np.array([p.obj for p in population])
            pop_cons = np.array([p.con for p in population])  # 约束冲突值

            # 执行非支配排序（需外部实现 nd_sort 函数）
            front_no, max_f_no = nd_sort(pop_objs, pop_cons, num_to_keep)

            next_idx = np.where(front_no < max_f_no)[0]
            last_idx = np.where(front_no == max_f_no)[0]

            # 如果最后的一层需要截断
            k = len(next_idx) + len(last_idx) - num_to_keep
            if k > 0:
                # 仅针对最后一层的个体进行截断
                del_mask = self._truncation(pop_objs[last_idx], pop_objs, k)
                selected_last = [population[last_idx[i]] for i, is_del in enumerate(del_mask) if not is_del]
                self.output = [population[i] for i in next_idx] + selected_last
            else:
                self.output = [population[i] for i in next_idx] + [population[i] for i in last_idx]

    def _get_fitness_single(self, population):
        """单目标适应度获取（处理约束：若违反约束则加上惩罚项）"""
        objs = np.array([p.obj for p in population]).flatten()
        cons = np.array([p.con for p in population]).flatten()
        # 简单的约束处理策略：适应度 = 目标值 + 约束违反量
        return objs + np.maximum(0, cons)

    def _truncation(self, pop_obj_last, pop_obj_all, k):
        """
        截断操作：删除距离最近（最拥挤）的个体
        """
        n_last = pop_obj_last.shape[0]
        del_mask = np.zeros(n_last, dtype=bool)

        # 计算最后一层与全体的距离矩阵
        distance = np.sqrt(np.sum((pop_obj_last[:, np.newaxis, :] - pop_obj_all[np.newaxis, :, :]) ** 2, axis=2))
        # 排除掉自己与自己的距离（设为无穷大）
        # 这里需要找到 pop_obj_last 在 pop_obj_all 中的对应索引，或者简化处理：
        for i in range(n_last):
            # 找到在 all 中与自己坐标完全相同的索引并排除
            self_idx = np.where(np.all(pop_obj_all == pop_obj_last[i], axis=1))[0]
            distance[i, self_idx] = np.inf

        while np.sum(del_mask) < k:
            remain = np.where(~del_mask)[0]
            # 找到剩余个体到其他所有个体的最小距离排序
            temp_dist = np.sort(distance[remain][:, np.concatenate([remain, np.arange(n_last, pop_obj_all.shape[0])])],
                                axis=1)

            # 选择“最近邻距离”最小的个体删除（如果距离相同，对比次近邻）
            # argpartition 或 argsort 模拟 MATLAB 的 sortrows
            worst_idx = self._sort_rows_idx(temp_dist)[0]
            del_mask[remain[worst_idx]] = True

        return del_mask

    def _sort_rows_idx(self, matrix):
        """模拟 MATLAB 的 sortrows，按列优先级排序"""
        return np.lexsort(matrix.T[::-1])