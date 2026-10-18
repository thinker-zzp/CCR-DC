from NeuroEA_GEA.Algorithms.NEUROEA.BLOCK import Block


class Block_Population(Block):
    """
    种群模块：作为算法图架构中的存储节点
    负责持有当前的种群并在每一代演化中更新
    """
    def __init__(self):
        super().__init__()
        # self.output 在基类中定义，用于存储当前的 Population (List of Solution objects)

    def main(self, problem, precursors, ratio):
        """
        更新种群：从前驱节点（如 Selection 或 Mutation 块）收集解
        """
        # in_type=1 表示收集的是 SOLUTION 对象
        # multiple=1 表示不需要对解的数量做倍数限制
        self.output = self.gather(problem, precursors, ratio, 1, 1)

    def initialization(self, population):
        """
        初始化种群：在算法开始的第一代，由 NeuroEA.py 调用
        """
        self.output = population