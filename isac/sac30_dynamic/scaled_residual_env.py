from __future__ import annotations

import numpy as np

from .advanced_env import AdvancedCounterfactualEnv
from .core import normalized_to_parameters, parameters_to_normalized


class ScaledResidualCounterfactualEnv(AdvancedCounterfactualEnv):
    """Advanced controller with an explicit executed-residual scale.

    The scale is applied after the continuous l2 projection. The calibrated
    base radii remain unchanged in state normalization, so the 30 residual
    state features describe the actual executed displacement. Integer
    parameters can either retain the original -1/0/+1 mapping or remain fixed
    at the Static Prior.
    """

    def __init__(
        self,
        *args,
        continuous_scale: float = 1.0,
        freeze_integer: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.continuous_scale = float(continuous_scale)
        self.freeze_integer = bool(freeze_integer)
        if self.continuous_scale < 0.0:
            raise ValueError("continuous_scale must be nonnegative")

    def _parameters(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        effective_action = action.copy()
        if self.continuous_budget is not None:
            norm = float(np.linalg.norm(effective_action[3:]))
            if norm > self.continuous_budget:
                effective_action[3:] *= self.continuous_budget / max(norm, 1e-12)

        # Apply scale after projection; scaling before projection would often
        # be cancelled whenever the scaled vector still had norm above one.
        effective_action[3:] *= self.continuous_scale
        residual = self.radii * effective_action
        normalized = np.clip(self.static_normalized + residual, -1.0, 1.0)
        parameters = normalized_to_parameters(normalized)

        if self.freeze_integer:
            parameters[:3] = self.static_parameters[:3]
            residual[:3] = 0.0
            effective_action[:3] = 0.0
        elif self.integer_discrete:
            offsets = np.where(
                action[:3] > 1.0 / 3.0,
                1.0,
                np.where(action[:3] < -1.0 / 3.0, -1.0, 0.0),
            )
            parameters[:3] = np.clip(
                self.static_parameters[:3] + offsets, 1.0, 10.0
            )
            residual[:3] = (
                parameters_to_normalized(parameters)[:3]
                - self.static_normalized[:3]
            )
            effective_action[:3] = offsets

        parameters[residual == 0.0] = self.static_parameters[residual == 0.0]
        return parameters, residual, effective_action
