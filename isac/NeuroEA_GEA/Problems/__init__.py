import importlib
import inspect

from NeuroEA_GEA.Problems.PROBLEM import Problem

import importlib
import inspect
import pkgutil

from NeuroEA_GEA import Problems  # 确保 Problems 文件夹下有 __init__.py


def get_problem_instance(problem_name, N=100, M=1, D=10, max_fe=10000, **kwargs):
    """
    动态扫描 Problems 包下所有子模块，查找并实例化匹配的问题类。
    """
    # 获取 Problems 包的根路径
    pkg_path = Problems.__path__
    pkg_name = Problems.__name__

    # walk_packages 会递归遍历 Problems 文件夹下所有的 .py 文件
    for loader, module_full_name, is_pkg in pkgutil.walk_packages(pkg_path, pkg_name + "."):
        try:
            # 导入模块
            module = importlib.import_module(module_full_name)

            # 获取模块中所有的类
            classes = inspect.getmembers(module, inspect.isclass)

            for name, cls in classes:
                # 忽略大小写匹配类名 (例如输入 'sop_f1' 匹配 'SOP_F1')
                if name.lower() == problem_name.lower():
                    # 确保它是 Problem 的子类，且不是基类本身
                    # 注意：这里需要你已经从某个地方导入了 Problem 基类
                    from NeuroEA_GEA.Problems.PROBLEM import Problem
                    if issubclass(cls, Problem) and cls is not Problem:
                        # print(f"成功找到匹配类: {name} 来自 {module_full_name}")
                        return cls(N=N, M=M, D=D, max_fe=max_fe, **kwargs)
        except Exception as e:
            # 忽略某些可能导致导入失败的无关模块
            continue

    raise ValueError(f"未能在 Problems 目录下找到问题类: {problem_name}。")