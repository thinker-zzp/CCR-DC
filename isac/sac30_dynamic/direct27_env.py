from __future__ import annotations

import numpy as np

from .advanced_env import AdvancedCounterfactualEnv
from .core import normalized_to_parameters, parameters_to_normalized


class Continuous27DirectControlEnv(AdvancedCounterfactualEnv):
    """Directly control the 27 continuous parameters in normalized space.

    This is an ablation of persistent continuous anchoring.  Parameters 1--3
    remain fixed at the Static Prior, while action components 1--27 map
    naturally to normalized parameter coordinates 4--30 in [-1, 1].  The
    previous-control part of the 44-D state stores the last normalized direct
    action (with three leading zeros), rather than a displacement from the
    Static Prior.
    """

    action_dim = 27

    def __init__(self, *args, **kwargs):
        if kwargs.get("integer_discrete", False):
            raise ValueError("27-D direct control cannot enable integer_discrete.")
        # Accepted by the shared runner for residual environments but not used
        # by direct normalized control.
        kwargs.pop("continuous_scale", None)
        kwargs.pop("freeze_integer", None)
        kwargs["integer_discrete"] = False
        super().__init__(*args, **kwargs)
        self.initial_actor_action = np.clip(
            parameters_to_normalized(self.static_parameters)[3:], -1.0, 1.0
        ).astype(np.float64)

    def _parameters(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_dim,):
            raise ValueError(
                f"Expected a {self.action_dim}-D direct action, got {action.shape}."
            )
        effective_action = np.clip(action, -1.0, 1.0)

        normalized = self.static_normalized.copy()
        normalized[3:] = effective_action
        parameters = normalized_to_parameters(normalized)
        parameters[:3] = self.static_parameters[:3]

        # AdvancedCounterfactualEnv normalizes this vector by radii when it
        # builds the state.  Encoding radii * action therefore stores the last
        # direct normalized action in the same 30-entry state block.
        state_encoding = np.zeros(30, dtype=np.float64)
        state_encoding[3:] = self.radii[3:] * effective_action
        return parameters, state_encoding, effective_action

    def step_training(self, action, reward_mode=None):
        next_state, reward, done, details = super().step_training(action, reward_mode)
        # The shadow executes the Static Prior, whose direct action is not zero.
        static_action = self.initial_actor_action.copy()
        details["static_action"] = static_action.astype(np.float32)
        details["zero_next_state"][-self.action_dim:] = static_action
        details["zero_reward"] -= (
            self.movement_penalty * float(np.mean(static_action ** 2))
        )
        return next_state, reward, done, details
