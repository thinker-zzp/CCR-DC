import numpy as np
from abc import ABC, abstractmethod


class Block(ABC):
    """
    BLOCK 基类 - NeuroEA 的核心构建块（类似于神经网络的 Layer）
    """

    def __init__(self):
        # 对应 MATLAB 的 properties
        self.parameter = np.array([])  # 块内的参数
        self.lower = np.array([])  # 参数下界
        self.upper = np.array([])  # 参数上界
        self.output = None  # 块的当前输出（Population 或矩阵）
        self.next_out = 0  # 下一个输出解的索引 (Python 从0开始)
        self.train_time = 0  # 训练次数

    @abstractmethod
    def main(self, problem, precursors, ratio):
        """
        块的主要执行逻辑，由子类实现
        :param problem: PROBLEM 对象
        :param precursors: 前驱 Block 对象列表
        :param ratio: 来自各个前驱的解流动比例
        """
        pass

    def parameter_assign(self):
        """
        当参数更新时，将 self.parameter 分配给具体变量
        子类根据需要覆盖此方法
        """
        pass

    def set_parameters(self, value):
        """
        对应 MATLAB 的 ParameterSet (处理数组片段赋值)
        """
        # 确保值在上下界内
        self.parameter = np.clip(value, self.lower, self.upper)
        self.parameter_assign()
        self.train_time += 1


    def gather(self, problem, predecessors, ratios, in_type, multiple):
        """
        从多个前驱节点收集解并混合
        in_type: 1 为需要 SOLUTION 对象列表, 2 为需要决策变量数值矩阵 (np.ndarray)
        multiple: 确保输出数量是该值的整数倍
        """
        valid_indices = [i for i, r in enumerate(ratios) if r > 0]
        temp_outputs = []
        lens = []

        for i in valid_indices:
            pred = predecessors[i]
            ratio = ratios[i]

            # --- 1. 数据形态标准化 ---
            # 情况 A: 下游需要 SOLUTION 对象 (in_type=1)
            if in_type == 1:
                # 如果当前是矩阵(如来自 Crossover/Mutation), 则必须评价并包装成对象
                if isinstance(pred.output, np.ndarray):
                    pred.output = problem.evaluation(pred.output)

                # 现在确保是 list 类型
                out = pred.output if isinstance(pred.output, list) else []
                size_n = len(out)

            # 情况 B: 下游需要数值矩阵 (in_type=2)
            else:
                # 如果当前是对象列表, 提取其决策变量矩阵
                if isinstance(pred.output, list):
                    if len(pred.output) > 0:
                        out = np.array([p.dec for p in pred.output])
                    else:
                        out = np.empty((0, problem.D))
                else:
                    out = pred.output

                # 强制确保数值矩阵是 2D (N, D)
                if out.ndim == 1 and out.size > 0:
                    out = out.reshape(-1, problem.D)
                size_n = out.shape[0]

            # --- 2. 循环取样 (MATLAB Mod 逻辑) ---
            if size_n == 0:
                continue

            num_to_gather = int(np.floor(size_n * ratio))
            # 即使比例很小，如果 ratio > 0，至少取 1 个
            num_to_gather = max(1, num_to_gather)

            # 生成循环索引指针
            indices = np.mod(np.arange(pred.next_out, pred.next_out + num_to_gather), size_n)

            if in_type == 1:
                temp_outputs.append([out[idx] for idx in indices])
            else:
                temp_outputs.append(out[indices, :])

            lens.append(len(indices))
            # 更新前驱节点的指针偏移
            pred.next_out = (indices[-1] + 1) % size_n if len(indices) > 0 else pred.next_out

        # --- 3. 判空处理 ---
        if not temp_outputs:
            return [] if in_type == 1 else np.empty((0, problem.D))

        # --- 4. 交错混合 (Interleave Logic) ---
        # 模拟 MATLAB 的交错提取，保证种群分布的均匀性
        max_l = max(lens)
        final_indices = []
        offsets = np.cumsum([0] + lens[:-1])

        for j in range(max_l):
            for i in range(len(lens)):
                if j < lens[i]:
                    final_indices.append(offsets[i] + j)

        # --- 5. 格式化最终输出并进行 Multiple 截断 ---
        if in_type == 1:
            # 合并所有列表
            flat_list = [item for sublist in temp_outputs for item in sublist]
            output = [flat_list[idx] for idx in final_indices]
            # 截断为 multiple 的倍数
            limit = (len(output) // multiple) * multiple
            output = output[:limit]
        else:
            # 垂直堆叠矩阵
            flat_mat = np.vstack(temp_outputs)
            output = flat_mat[final_indices, :]
            # 截断为 multiple 的倍数
            limit = (output.shape[0] // multiple) * multiple
            output = output[:limit, :]

        return output

