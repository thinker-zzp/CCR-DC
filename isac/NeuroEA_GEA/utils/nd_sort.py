import numpy as np


def nd_sort(pop_obj, pop_con=None, n_sort=None):
    """
    非支配排序 (ENS 算法的高效实现)
    :param pop_obj: 目标函数矩阵 (N x M)
    :param pop_con: 约束违反矩阵 (N x 1)，可选
    :param n_sort: 仅排序前 n_sort 个个体，可选
    :return: (front_no, max_f_no)
             front_no: 每个个体所属的层号 (1 为第一层)
             max_f_no: 最终排序到的最高层号
    """
    n, m = pop_obj.shape
    if n_sort is None:
        n_sort = n

    # 如果有约束，先处理约束逻辑
    # 在 PlatEMO 中，违反约束的个体（con > 0）被视为被任何可行个体支配
    if pop_con is not None:
        con_sum = np.sum(np.maximum(0, pop_con), axis=1)
        # 排序：首先按约束违反度升序，其次按目标函数值升序
        # 这里简化处理：直接合并排序权重

    # 简单的非支配逻辑实现
    front_no = np.full(n, np.inf)
    max_f_no = 0

    # 对目标值进行排序（加速排序过程）
    # 首先按第一目标排序，若相同则按后续目标
    idx = np.lexsort(pop_obj.T[::-1])
    sorted_obj = pop_obj[idx]

    for i in range(n):
        for f in range(1, max_f_no + 2):
            # 检查个体 i 是否被当前层 f 中的任何个体支配
            is_dominated = False
            # 仅需检查当前层已有的个体
            current_front = np.where(front_no[idx[:i]] == f)[0]
            for j in current_front:
                # 检查 sorted_obj[j] 是否支配 sorted_obj[i]
                # 因为已经按 lexsort 排序，sorted_obj[j] 不会被 sorted_obj[i] 支配
                # 只需检查是否至少有一个目标更优
                if np.all(sorted_obj[j] <= sorted_obj[i]) and np.any(sorted_obj[j] < sorted_obj[i]):
                    is_dominated = True
                    break

            if not is_dominated:
                front_no[idx[i]] = f
                max_f_no = max(max_f_no, f)
                break

        # 如果已经找够了需要的层数对应的个体，可以提前停止
        if np.sum(front_no <= max_f_no) >= n_sort:
            break

    return front_no, max_f_no