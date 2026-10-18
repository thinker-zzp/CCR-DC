from __future__ import annotations

import copy

import numpy as np

from .core import (
    NeuroEASession, normalized_to_parameters, objective_gap,
    parameters_to_normalized,
)


class AdvancedCounterfactualEnv:
    """30-D control held for K generations with a paired static counterfactual."""

    action_dim = 30
    state_dim = 44

    def __init__(self, static_parameters, radii, problem="SOP_F1", protocol="paper",
                 horizon=1, dimension=10, population_size=100, max_fe=10000,
                 median_weight=0.3, movement_penalty=0.005,
                 continuous_budget=None, integer_discrete=False,
                 reward_mode="counterfactual", hybrid_cf_weight=0.25):
        self.static_parameters = np.asarray(static_parameters, dtype=np.float64)
        self.static_normalized = parameters_to_normalized(self.static_parameters)
        self.radii = np.asarray(radii, dtype=np.float64)
        if self.radii.shape != (30,):
            raise ValueError("Expected 30 calibrated radii.")
        self.problem_name, self.protocol = problem, protocol
        self.horizon = int(horizon)
        self.dimension, self.population_size, self.max_fe = dimension, population_size, max_fe
        self.median_weight, self.movement_penalty = median_weight, movement_penalty
        self.continuous_budget = (None if continuous_budget is None
                                  else float(continuous_budget))
        self.integer_discrete = bool(integer_discrete)
        if reward_mode not in {"counterfactual", "ordinary", "hybrid"}:
            raise ValueError(
                "reward_mode must be 'counterfactual', 'ordinary', or 'hybrid'."
            )
        self.reward_mode = reward_mode
        self.hybrid_cf_weight = float(hybrid_cf_weight)
        if self.hybrid_cf_weight < 0.0:
            raise ValueError("hybrid_cf_weight must be nonnegative.")
        # This history is deliberately shared across episodes of one training
        # run, matching the normalization used by the original controller.
        self.reward_history = []
        # Hybrid mode must not normalize two differently scaled targets with a
        # shared history.  Legacy modes keep using reward_history unchanged.
        self.counterfactual_reward_history = []
        self.ordinary_reward_history = []
        # Backwards-compatible alias used by older diagnostics.
        self.advantage_history = self.reward_history

    def reset(self, seed):
        self.session = NeuroEASession(
            self.problem_name, self.protocol, self.dimension, self.population_size,
            self.max_fe, int(seed), self.static_parameters,
        )
        stats = self.session.population_statistics()
        self.initial_stats = stats.copy()
        self.history = {key: [stats[key]] for key in ["best", "q25", "median", "q75", "diversity"]}
        self.current_residual = np.zeros(30, dtype=np.float64)
        self.stagnation = 0
        return self._state(), {"initial_best": stats["best"]}

    def _parameters(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        effective_action = action.copy()
        if self.continuous_budget is not None:
            norm = float(np.linalg.norm(effective_action[3:]))
            if norm > self.continuous_budget:
                effective_action[3:] *= self.continuous_budget / max(norm, 1e-12)
        residual = self.radii * effective_action
        normalized = np.clip(self.static_normalized + residual, -1.0, 1.0)
        parameters = normalized_to_parameters(normalized)
        if self.integer_discrete:
            offsets = np.where(action[:3] > 1.0 / 3.0, 1.0,
                               np.where(action[:3] < -1.0 / 3.0, -1.0, 0.0))
            parameters[:3] = np.clip(self.static_parameters[:3] + offsets, 1.0, 10.0)
            residual[:3] = parameters_to_normalized(parameters)[:3] - self.static_normalized[:3]
            effective_action[:3] = offsets
        parameters[residual == 0.0] = self.static_parameters[residual == 0.0]
        return parameters, residual, effective_action

    @staticmethod
    def _append(history, stats):
        result = {key: list(values) for key, values in history.items()}
        for key in result:
            result[key].append(stats[key])
            if len(result[key]) > 6:
                result[key] = result[key][-6:]
        return result

    def _compose_state(self, session, history, residual, stagnation):
        progress = session.problem.fe / self.max_fe
        levels = [
            np.log10(
                objective_gap(self.problem_name, self.initial_stats[key], self.dimension)
                / objective_gap(self.problem_name, history[key][-1], self.dimension)
            ) / 20.0
            for key in ["best", "q25", "median", "q75"]
        ]
        changes = []
        for key in ["best", "median"]:
            for lag in [1, 3, 5]:
                old_index = max(0, len(history[key]) - 1 - lag)
                changes.append(np.log10(
                    objective_gap(self.problem_name, history[key][old_index], self.dimension)
                    / objective_gap(self.problem_name, history[key][-1], self.dimension)
                ) / 5.0)
        diversity = history["diversity"][-1]
        previous_diversity = history["diversity"][-2] if len(history["diversity"]) > 1 else diversity
        global_features = np.array([
            2.0 * progress - 1.0,
            *np.clip(levels, -1.0, 1.0),
            *np.clip(changes, -1.0, 1.0),
            np.clip(diversity * 20.0, 0.0, 1.0),
            np.clip((diversity - previous_diversity) * 100.0, -1.0, 1.0),
            np.clip(stagnation / 20.0, 0.0, 1.0),
        ], dtype=np.float32)
        scaled_residual = residual / np.maximum(self.radii, 1e-12)
        return np.concatenate([global_features, np.clip(scaled_residual, -1, 1).astype(np.float32)])

    def _state(self):
        return self._compose_state(self.session, self.history, self.current_residual, self.stagnation)

    def step_training(self, action, reward_mode=None):
        """Advance paired dynamic/static rollouts and return one training sample.

        Both reward modes execute exactly the same dynamic rollout and cloned
        static shadow rollout.  This makes the counterfactual-vs-ordinary
        ablation evaluation- and compute-matched; only the scalar reward target
        changes.  ``zero_reward`` is the logically corresponding target for the
        cloned zero-action replay transition.
        """
        mode = self.reward_mode if reward_mode is None else reward_mode
        if mode not in {"counterfactual", "ordinary", "hybrid"}:
            raise ValueError(
                "reward_mode must be 'counterfactual', 'ordinary', or 'hybrid'."
            )
        parameters, residual, effective_action = self._parameters(action)
        previous_best = self.history["best"][-1]
        previous_median = self.history["median"][-1]
        random_before = np.random.get_state()
        remaining = max(1, int((self.max_fe - self.session.problem.fe) // self.population_size))
        horizon = min(self.horizon, remaining)

        static_branch = copy.deepcopy(self.session)
        static_fe_before = static_branch.problem.fe
        np.random.set_state(random_before)
        static_sequence = []
        for _ in range(horizon):
            static_branch.step(self.static_parameters)
            static_sequence.append(static_branch.population_statistics())
        static_end = static_sequence[-1]

        np.random.set_state(random_before)
        dynamic_fe_before = self.session.problem.fe
        dynamic_sequence = []
        for _ in range(horizon):
            self.session.step(parameters)
            dynamic_sequence.append(self.session.population_statistics())
        dynamic_end = dynamic_sequence[-1]

        best_advantage = np.log10(
            objective_gap(self.problem_name, static_end["best"], self.dimension)
            / objective_gap(self.problem_name, dynamic_end["best"], self.dimension)
        )
        median_advantage = np.log10(
            objective_gap(self.problem_name, static_end["median"], self.dimension)
            / objective_gap(self.problem_name, dynamic_end["median"], self.dimension)
        )
        raw_counterfactual = float(best_advantage + self.median_weight * median_advantage)

        dynamic_best_progress = np.log10(
            objective_gap(self.problem_name, previous_best, self.dimension)
            / objective_gap(self.problem_name, dynamic_end["best"], self.dimension)
        )
        dynamic_median_progress = np.log10(
            objective_gap(self.problem_name, previous_median, self.dimension)
            / objective_gap(self.problem_name, dynamic_end["median"], self.dimension)
        )
        raw_ordinary = float(
            dynamic_best_progress + self.median_weight * dynamic_median_progress
        )
        static_best_progress = np.log10(
            objective_gap(self.problem_name, previous_best, self.dimension)
            / objective_gap(self.problem_name, static_end["best"], self.dimension)
        )
        static_median_progress = np.log10(
            objective_gap(self.problem_name, previous_median, self.dimension)
            / objective_gap(self.problem_name, static_end["median"], self.dimension)
        )
        raw_static_ordinary = float(
            static_best_progress + self.median_weight * static_median_progress
        )

        movement_penalty_value = (
            self.movement_penalty * float(np.mean(effective_action ** 2))
        )
        if mode == "hybrid":
            self.counterfactual_reward_history.append(abs(raw_counterfactual))
            self.ordinary_reward_history.append(abs(raw_ordinary))
            counterfactual_scale = max(
                1.4826 * float(np.median(
                    self.counterfactual_reward_history[-2000:]
                )),
                1e-5,
            )
            ordinary_scale = max(
                1.4826 * float(np.median(self.ordinary_reward_history[-2000:])),
                1e-5,
            )
            normalized_counterfactual = float(np.clip(
                raw_counterfactual / counterfactual_scale, -5.0, 5.0
            ))
            normalized_ordinary = float(np.clip(
                raw_ordinary / ordinary_scale, -5.0, 5.0
            ))
            normalized_static_ordinary = float(np.clip(
                raw_static_ordinary / ordinary_scale, -5.0, 5.0
            ))
            denominator = 1.0 + self.hybrid_cf_weight
            normalized_reward = float(
                (normalized_ordinary
                 + self.hybrid_cf_weight * normalized_counterfactual)
                / denominator
            )
            # For the cloned zero-action transition, the counterfactual term
            # is exactly zero and there is no movement penalty.
            normalized_zero_reward = float(
                normalized_static_ordinary / denominator
            )
            raw_reward = float(
                (raw_ordinary + self.hybrid_cf_weight * raw_counterfactual)
                / denominator
            )
            raw_zero_reward = float(raw_static_ordinary / denominator)
            scale = float(
                (ordinary_scale
                 + self.hybrid_cf_weight * counterfactual_scale)
                / denominator
            )
        else:
            raw_reward = (
                raw_counterfactual if mode == "counterfactual" else raw_ordinary
            )
            raw_zero_reward = (
                0.0 if mode == "counterfactual" else raw_static_ordinary
            )
            self.reward_history.append(abs(raw_reward))
            scale = max(
                1.4826 * float(np.median(self.reward_history[-2000:])), 1e-5
            )
            normalized_reward = float(np.clip(raw_reward / scale, -5.0, 5.0))
            normalized_zero_reward = float(np.clip(
                raw_zero_reward / scale, -5.0, 5.0
            ))
            normalized_counterfactual = float(np.clip(
                raw_counterfactual / scale, -5.0, 5.0
            ))
            normalized_ordinary = float(np.clip(
                raw_ordinary / scale, -5.0, 5.0
            ))
            normalized_static_ordinary = float(np.clip(
                raw_static_ordinary / scale, -5.0, 5.0
            ))
            counterfactual_scale = scale if mode == "counterfactual" else np.nan
            ordinary_scale = scale if mode == "ordinary" else np.nan
        reward = normalized_reward - movement_penalty_value

        static_history = {key: list(values) for key, values in self.history.items()}
        static_stagnation, static_previous = self.stagnation, previous_best
        for stats in static_sequence:
            static_history = self._append(static_history, stats)
            static_stagnation = 0 if stats["best"] < static_previous else static_stagnation + 1
            static_previous = stats["best"]
        zero_next_state = self._compose_state(
            static_branch,
            static_history, np.zeros(30), static_stagnation,
        )

        dynamic_previous = previous_best
        for stats in dynamic_sequence:
            self.history = self._append(self.history, stats)
            self.stagnation = 0 if stats["best"] < dynamic_previous else self.stagnation + 1
            dynamic_previous = stats["best"]
        self.current_residual = residual
        details = {
            "fe": self.session.problem.fe, "best": dynamic_end["best"],
            # Legacy fields retain their old meaning for existing callers.
            "raw_advantage": raw_counterfactual,
            "normalized_advantage": float(np.clip(raw_counterfactual / scale, -5.0, 5.0)),
            "reward_mode": mode, "raw_reward": raw_reward,
            "normalized_reward": normalized_reward,
            "zero_reward": normalized_zero_reward,
            "raw_counterfactual": raw_counterfactual,
            "raw_ordinary": raw_ordinary,
            "raw_static_ordinary": raw_static_ordinary,
            "normalized_counterfactual": normalized_counterfactual,
            "normalized_ordinary": normalized_ordinary,
            "normalized_static_ordinary": normalized_static_ordinary,
            "counterfactual_scale": float(counterfactual_scale),
            "ordinary_scale": float(ordinary_scale),
            "counterfactual_clip_positive": bool(
                raw_counterfactual / counterfactual_scale >= 5.0
            ) if np.isfinite(counterfactual_scale) else False,
            "counterfactual_clip_negative": bool(
                raw_counterfactual / counterfactual_scale <= -5.0
            ) if np.isfinite(counterfactual_scale) else False,
            "ordinary_clip_positive": bool(
                raw_ordinary / ordinary_scale >= 5.0
            ) if np.isfinite(ordinary_scale) else False,
            "ordinary_clip_negative": bool(
                raw_ordinary / ordinary_scale <= -5.0
            ) if np.isfinite(ordinary_scale) else False,
            "movement_penalty": float(movement_penalty_value),
            "hybrid_cf_weight": self.hybrid_cf_weight,
            "reward_scale": scale, "best_advantage": float(best_advantage),
            "median_advantage": float(median_advantage), "parameters": parameters.copy(),
            "zero_next_state": zero_next_state, "effective_action": effective_action.copy(),
            "dynamic_fe": int(self.session.problem.fe - dynamic_fe_before),
            "shadow_fe": int(static_branch.problem.fe - static_fe_before),
        }
        return self._state(), reward, self.session.done, details

    def step_counterfactual(self, action):
        """Backwards-compatible counterfactual-reward training step."""
        return self.step_training(action, reward_mode="counterfactual")

    def step_deploy(self, action, execute_dynamic):
        parameters, residual, effective_action = self._parameters(action) if execute_dynamic else (
            self.static_parameters.copy(), np.zeros(30, dtype=np.float64),
            np.zeros(30, dtype=np.float64))
        remaining = max(1, int((self.max_fe - self.session.problem.fe) // self.population_size))
        steps = min(self.horizon, remaining)
        previous_best = self.history["best"][-1]
        for _ in range(steps):
            self.session.step(parameters)
            stats = self.session.population_statistics()
            self.history = self._append(self.history, stats)
            self.stagnation = 0 if stats["best"] < previous_best else self.stagnation + 1
            previous_best = stats["best"]
        self.current_residual = residual
        return self._state(), self.session.done, {
            "fe": self.session.problem.fe, "best": stats["best"],
            "parameters": parameters.copy(), "executed_dynamic": bool(execute_dynamic),
            "effective_action": effective_action.copy(),
        }
