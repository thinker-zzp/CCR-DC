from __future__ import annotations

import numpy as np

from .core import normalized_to_parameters
from .scaled_residual_env import ScaledResidualCounterfactualEnv


class Continuous27ResidualCounterfactualEnv(ScaledResidualCounterfactualEnv):
    """仅动态控制27个连续参数，前三个整数参数固定为Static Prior。

    状态仍为44维：14个搜索特征与30个上一时刻残差。由于整数参数不再
    属于动作空间，状态中的前三个残差始终为0。SAC直接输出27维连续动作，
    不再产生或量化任何整数动作。
    """

    action_dim = 27

    def __init__(self, *args, **kwargs):
        if kwargs.get("integer_discrete", False):
            raise ValueError("27-D continuous-only mode cannot enable integer_discrete.")
        kwargs["integer_discrete"] = False
        kwargs["freeze_integer"] = True
        super().__init__(*args, **kwargs)

    def _parameters(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_dim,):
            raise ValueError(
                f"Expected a {self.action_dim}-D continuous action, got {action.shape}."
            )
        effective_action = np.clip(action, -1.0, 1.0)
        if self.continuous_budget is not None:
            norm = float(np.linalg.norm(effective_action))
            if norm > self.continuous_budget:
                effective_action *= self.continuous_budget / max(norm, 1e-12)
        effective_action *= self.continuous_scale

        residual = np.zeros(30, dtype=np.float64)
        residual[3:] = self.radii[3:] * effective_action
        normalized = np.clip(self.static_normalized + residual, -1.0, 1.0)
        parameters = normalized_to_parameters(normalized)

        # 前三个Tournament参数严格继承静态配置，不进入策略动作空间。
        parameters[:3] = self.static_parameters[:3]
        residual[:3] = 0.0
        parameters[residual == 0.0] = self.static_parameters[residual == 0.0]
        return parameters, residual, effective_action
