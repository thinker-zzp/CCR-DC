from __future__ import annotations

import numpy as np

from .scaled_residual_env import ScaledResidualCounterfactualEnv


class TrustRegionCounterfactualEnv(ScaledResidualCounterfactualEnv):
    """Residual controller with an adaptive Lagrangian trust region.

    The continuous residual remains inside the calibrated unit l2 ball, but
    unlike a fixed global multiplier it may use the whole ball when useful.
    A dual variable controls the average normalized intervention energy.
    Discrete Tournament changes consume one full unit each, encouraging the
    online controller to retain the structural choices made by static SAC
    unless their counterfactual benefit justifies a switch.
    """

    def __init__(
        self,
        *args,
        trust_region_target: float = 0.5,
        dual_learning_rate: float = 1e-3,
        dual_initial: float = 0.05,
        dual_max: float = 5.0,
        integer_switch_cost: float = 1.0,
        **kwargs,
    ):
        # The adaptive Lagrangian replaces the legacy fixed movement penalty.
        kwargs["movement_penalty"] = 0.0
        super().__init__(*args, **kwargs)
        self.trust_region_target = float(trust_region_target)
        self.dual_learning_rate = float(dual_learning_rate)
        self.dual_initial = float(dual_initial)
        self.dual_max = float(dual_max)
        self.integer_switch_cost = float(integer_switch_cost)
        if not (0.0 <= self.trust_region_target <= 1.0):
            raise ValueError("trust_region_target must be within [0, 1]")
        if self.dual_learning_rate <= 0.0:
            raise ValueError("dual_learning_rate must be positive")
        if self.dual_initial < 0.0 or self.dual_max <= 0.0:
            raise ValueError("dual bounds must be positive")
        if self.dual_initial > self.dual_max:
            raise ValueError("dual_initial cannot exceed dual_max")
        if self.integer_switch_cost < 0.0:
            raise ValueError("integer_switch_cost must be nonnegative")
        self.dual_variable = self.dual_initial

    def trust_region_state_dict(self) -> dict[str, float]:
        return {
            "dual_variable": float(self.dual_variable),
            "trust_region_target": self.trust_region_target,
            "dual_learning_rate": self.dual_learning_rate,
            "dual_max": self.dual_max,
            "integer_switch_cost": self.integer_switch_cost,
        }

    def load_trust_region_state_dict(self, state: dict) -> None:
        for key in (
            "trust_region_target",
            "dual_learning_rate",
            "dual_max",
            "integer_switch_cost",
        ):
            if not np.isclose(float(state[key]), float(getattr(self, key))):
                raise ValueError(f"Checkpoint trust-region setting differs: {key}")
        self.dual_variable = float(state["dual_variable"])

    def step_training(self, action, reward_mode=None):
        next_state, base_reward, done, info = super().step_training(
            action, reward_mode=reward_mode
        )
        effective = np.asarray(info["effective_action"], dtype=np.float64)
        continuous_energy = float(np.dot(effective[3:], effective[3:]))
        integer_switches = int(np.count_nonzero(effective[:3]))
        cost = float(
            continuous_energy + self.integer_switch_cost * integer_switches
        )

        dual_before = float(self.dual_variable)
        constraint_penalty = dual_before * cost
        constrained_reward = float(base_reward - constraint_penalty)
        violation = float(cost - self.trust_region_target)
        self.dual_variable = float(np.clip(
            dual_before + self.dual_learning_rate * violation,
            0.0,
            self.dual_max,
        ))

        # The cloned zero-action transition has zero intervention cost.
        info["base_normalized_reward"] = float(base_reward)
        info["normalized_reward"] = constrained_reward
        info["zero_base_reward"] = float(info["zero_reward"])
        info["trust_region_cost"] = cost
        info["trust_region_target"] = self.trust_region_target
        info["trust_region_violation"] = violation
        info["continuous_action_energy"] = continuous_energy
        info["integer_switches"] = integer_switches
        info["dual_variable_before"] = dual_before
        info["dual_variable"] = float(self.dual_variable)
        info["constraint_penalty"] = float(constraint_penalty)
        return next_state, constrained_reward, done, info
