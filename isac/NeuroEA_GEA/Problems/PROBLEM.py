import numpy as np
from abc import ABC, abstractmethod

from NeuroEA_GEA.Problems.SOLUTION import Solution




class Problem(ABC):
    """
    PROBLEM 基类 - 定义问题的属性、评价逻辑和初始化方法
    """

    def __init__(self, N=100, M=1, D=10, max_fe=10000, max_runtime=float('inf'), parameter=None):
        # 基础设置
        self.N = N  # 种群大小
        self.M = M  # 目标数量
        self.D = D  # 变量维度
        self.max_fe = max_fe  # 最大评价次数
        self.fe = 0  # 当前评价次数计数器
        self.max_runtime = max_runtime

        # 变量范围与编码
        self.encoding = 1  # 1.real 2.integer 3.label 4.binary 5.permutation
        self.lower = np.zeros(D)  # 下界
        self.upper = np.ones(D)  # 上界

        # 理论最优值
        self.optimum = None
        self.pf = None
        self.parameter = parameter if parameter is not None else []

        # 调用子类的 Setting 方法进行个性化配置
        self.setting()

        # 初始化最优解信息 (示例简化，实际可根据 M 设定)
        if self.optimum is None:
            self.optimum = self.get_optimum(10000)

    # 添加以下代码，确保访问小写字母时不会报错
    @property
    def m(self):
        return self.M

    @property
    def d(self):
        return self.D

    @abstractmethod
    def setting(self):
        """子类需实现此方法来设置 M, D, lower, upper 等"""
        pass

    def initialization(self, N=None):
        """
        生成初始种群
        """
        if N is None:
            N = self.N

        # 根据编码方式生成随机决策变量 (PopDec)
        # 这里以实数编码 (encoding=1) 为例，其他类型可按需扩展
        pop_dec = np.random.uniform(
            np.tile(self.lower, (N, 1)),
            np.tile(self.upper, (N, 1))
        )

        # 如果是整数编码，进行取整
        if self.encoding in [2, 3]:
            pop_dec = np.round(pop_dec)

        return self.evaluation(pop_dec)

    def evaluation(self, pop_dec, *args):
        """
        评价多组解。增加 FE 计数，并返回 SOLUTION 对象列表。
        """

        # print(f"DEBUG: Evaluation called. Current FE: {self.fe}, PopSize: {len(pop_dec)}")
        # 1. 修复越界解 (Repair)
        pop_dec = self.cal_dec(pop_dec)

        # 2. 计算目标值和约束违反
        pop_obj = self.cal_obj(pop_dec)
        pop_con = self.cal_con(pop_dec)

        # 3. 封装为 SOLUTION 对象 (假设你已实现 SOLUTION 类)
        # from algorithms.solution import Solution
        population = []
        for i in range(len(pop_dec)):
            # 这里对应 MATLAB 的 SOLUTION(PopDec, PopObj, PopCon, ...)
            sol = Solution(pop_dec[i], pop_obj[i], pop_con[i], *args)
            population.append(sol)

        # 4. 更新消耗的评价次数
        self.fe += len(population)
        return population

    # def cal_dec(self, pop_dec):
    #
    #     pop_dec = np.maximum(np.minimum(pop_dec, self.upper), self.lower)
    #     if self.encoding in [2, 3, 4]:
    #         pop_dec = np.round(pop_dec)
    #     return pop_dec
    # Problems/PROBLEM.py
    def cal_dec(self, pop_dec):
        pop_dec = np.asarray(pop_dec)

        # 针对 (1, 100) 这种错误的维度进行复原
        if pop_dec.ndim == 2 and pop_dec.shape[0] == 1 and pop_dec.shape[1] != self.D:
            # 认为这其实是 (N,) 被错误提升成了 (1, N)，现在转为 (N/D, D)
            pop_dec = pop_dec.reshape(-1, self.D)
        elif pop_dec.ndim == 1:
            pop_dec = pop_dec.reshape(-1, self.D)

        # 使用 reshape(1, -1) 确保边界数组在第二维上广播
        low = self.lower.reshape(1, -1)
        up = self.upper.reshape(1, -1)

        return np.maximum(np.minimum(pop_dec, up), low)

    @abstractmethod
    def cal_obj(self, pop_dec):
        """子类实现：计算目标函数值矩阵 (N x M)"""
        return np.zeros((len(pop_dec), self.M))

    def cal_con(self, pop_dec):
        """子类实现：计算约束违反矩阵 (N x 1)，无约束则返回 0"""
        return np.zeros((len(pop_dec), 1))

    def get_optimum(self, N):
        """获取参考最优值"""
        return np.ones((1, self.M)) if self.M > 1 else np.zeros(1)

    def parameter_set(self, *default_values):
        """对应 MATLAB 的 ParameterSet"""
        results = list(default_values)
        for i, p in enumerate(self.parameter):
            if p is not None:
                results[i] = p
        return results

    def cal_metric(self, metric_name, result):
        """
        计算指标。
        result 格式通常为 [[FE, Population], [FE, Population], ...]
        """
        if not result:
            return np.inf

        # 获取最后一代的种群
        last_pop = result[-1][1]

        if metric_name == 'Min_value':
            # 提取所有个体的目标值 (obj)
            objs = np.array([p.obj for p in last_pop])
            return np.min(objs)

        elif metric_name == 'HV':
            # 如果是多目标，这里需要调用 HV 计算逻辑
            return HV(np.array([p.obj for p in last_pop]), self.optimum)

        return 0

def HV(param, optimum):
    pass
