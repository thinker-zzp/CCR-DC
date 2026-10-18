# utils/selection_utils.py
import numpy as np


def get_fitness_single(population):
    """
    获取单目标下的适应度排名参考值
    计算方式：目标值 + 约束惩罚项
    """
    objs = np.array([p.obj for p in population]).flatten()
    cons = np.array([p.con for p in population]).flatten()

    # 约束处理：如果 con > 0 (违反约束)，则将违反量作为一个巨大的惩罚
    # 加上 objs 的最大值是为了确保任何可行解都优于不可行解
    penalty = np.max(objs) if len(objs) > 0 else 0
    fitness = objs + np.maximum(0, cons) * (penalty + 1e6)

    return fitness


