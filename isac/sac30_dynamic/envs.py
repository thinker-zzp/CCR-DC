from __future__ import annotations

import copy

import numpy as np

from .core import (
    F1_REFERENCE, NeuroEASession, normalized_to_parameters,
    objective_gap, parameters_to_normalized, run_fixed,
)


class StaticConfigurationBandit:
    """One-step 30-D SAC environment used to produce the initial configuration."""

    state_dim = 1
    action_dim = 30

    def __init__(self, problem_name="SOP_F1", protocol="paper", repeats=3,
                 dimension=10, population_size=100, max_fe=10000):
        self.problem_name, self.protocol = problem_name, protocol
        self.repeats = repeats
        self.dimension, self.population_size, self.max_fe = dimension, population_size, max_fe

    def evaluate(self, action, seed):
        parameters = normalized_to_parameters(action)
        values = [run_fixed(parameters, seed + index, self.problem_name, self.protocol,
                            self.dimension, self.population_size, self.max_fe)
                  for index in range(self.repeats)]
        robust = float(np.mean(values))
        gap = float(objective_gap(self.problem_name, robust, self.dimension))
        reward = float(-np.log10(gap))
        return reward, robust, parameters, values


class DynamicConfigurationEnv:
    """Every step controls all 30 NeuroEA parameters for one generation."""

    action_dim = 30
    state_dim = 37

    def __init__(self, static_parameters=F1_REFERENCE, problem_name="SOP_F1", protocol="paper",
                 dimension=10, population_size=100, max_fe=10000, continuous_radius=0.10,
                 discrete_radius=0.25, smoothing=0.35, movement_penalty=0.002,
                 terminal_bonus=5.0, relative_weight=3.0):
        self.static_parameters = np.asarray(static_parameters, dtype=np.float64)
        self.static_normalized = parameters_to_normalized(self.static_parameters)
        self.problem_name, self.protocol = problem_name, protocol
        self.dimension, self.population_size, self.max_fe = dimension, population_size, max_fe
        self.radius = np.full(30, float(continuous_radius), dtype=np.float64)
        self.radius[:3] = float(discrete_radius)
        self.smoothing = float(smoothing)
        self.movement_penalty = float(movement_penalty)
        self.terminal_bonus = float(terminal_bonus)
        self.relative_weight = float(relative_weight)
        self._baseline_cache = {}

    def reset(self, seed):
        seed = int(seed)
        if seed not in self._baseline_cache:
            baseline = NeuroEASession(
                self.problem_name, self.protocol, self.dimension, self.population_size,
                self.max_fe, seed, self.static_parameters)
            trace = {baseline.problem.fe: baseline.best()}
            while not baseline.done:
                point = baseline.step(self.static_parameters)
                trace[point.fe] = point.best
            self._baseline_cache[seed] = (baseline.best(), trace)
        self.static_final, self.static_trace = self._baseline_cache[seed]
        self.session = NeuroEASession(
            self.problem_name, self.protocol, self.dimension, self.population_size,
            self.max_fe, seed, self.static_parameters,
        )
        self.current_normalized = self.static_normalized.copy()
        self.current_residual = np.zeros(30, dtype=np.float64)
        self.initial_best = self.previous_best = self.session.best()
        self.last_improvement = self.ema_improvement = 0.0
        self.current_advantage = 0.0
        self.stagnation = 0
        return self._state(), {"static_final": self.static_final, "initial_best": self.initial_best}

    def _state(self):
        tiny = 1e-300
        progress = self.session.problem.fe / self.max_fe
        total_gain = np.log10((self.initial_best + tiny) / (self.previous_best + tiny)) / 20.0
        residual = self.current_residual / np.maximum(self.radius, 1e-12)
        features = np.array([
            2.0 * progress - 1.0,
            np.clip(total_gain, -1.0, 1.0),
            np.clip(self.last_improvement / 5.0, -1.0, 1.0),
            np.clip(self.ema_improvement / 5.0, -1.0, 1.0),
            np.clip(self.current_advantage / 5.0, -1.0, 1.0),
            np.clip(self.session.diversity() * 20.0, 0.0, 1.0),
            np.clip(self.stagnation / 20.0, 0.0, 1.0),
        ], dtype=np.float32)
        return np.concatenate([features, np.clip(residual, -1.0, 1.0).astype(np.float32)])

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target_residual = self.radius * action
        self.current_residual = (
            (1.0 - self.smoothing) * self.current_residual + self.smoothing * target_residual
        )
        self.current_normalized = np.clip(
            self.static_normalized + self.current_residual, -1.0, 1.0
        )
        parameters = normalized_to_parameters(self.current_normalized)
        # A zero residual must be a bitwise-exact no-op. At ~1e-23 fitness,
        # a normalization round trip of only 1e-16 can change the whole trace.
        parameters[self.current_residual == 0.0] = self.static_parameters[self.current_residual == 0.0]
        previous = self.previous_best
        previous_fe = self.session.problem.fe
        info = self.session.step(parameters)
        current = info.best
        tiny = 1e-300
        improvement = float(np.log10((previous + tiny) / (current + tiny)))
        static_improvement = float(np.log10(
            (self.static_trace[previous_fe] + tiny) / (self.static_trace[info.fe] + tiny)))
        relative_improvement = improvement - static_improvement
        self.current_advantage = float(np.log10(
            (self.static_trace[info.fe] + tiny) / (current + tiny)))
        self.last_improvement = improvement
        self.ema_improvement = 0.8 * self.ema_improvement + 0.2 * improvement
        self.stagnation = 0 if improvement > 1e-8 else self.stagnation + 1
        self.previous_best = current
        reward = float(np.clip(
            0.1 * improvement + self.relative_weight * relative_improvement, -5.0, 5.0)
            - self.movement_penalty * np.mean(action ** 2))
        terminated = self.session.done
        bonus = 0.0
        if terminated:
            bonus = self.terminal_bonus * float(np.clip(self.current_advantage, -5.0, 5.0))
            reward += bonus
        details = {
            "fe": info.fe, "best": current, "static_final": self.static_final,
            "improvement": improvement, "terminal_bonus": bonus,
            "static_improvement": static_improvement,
            "relative_improvement": relative_improvement,
            "advantage": self.current_advantage,
            "parameters": parameters.copy(), "action_rms": float(np.sqrt(np.mean(action ** 2))),
        }
        return self._state(), reward, terminated, details


class SameStateCounterfactualEnv(DynamicConfigurationEnv):
    """Dynamic MDP with an exact same-state, common-random-number baseline.

    The cloned static transition is used only to shape the training reward. The
    actual environment still advances exactly once with the dynamic action.
    """

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target_residual = self.radius * action
        self.current_residual = (
            (1.0 - self.smoothing) * self.current_residual + self.smoothing * target_residual
        )
        self.current_normalized = np.clip(
            self.static_normalized + self.current_residual, -1.0, 1.0
        )
        parameters = normalized_to_parameters(self.current_normalized)
        parameters[self.current_residual == 0.0] = self.static_parameters[self.current_residual == 0.0]

        previous = self.previous_best
        previous_fe = self.session.problem.fe
        random_state = np.random.get_state()
        static_clone = copy.deepcopy(self.session)
        np.random.set_state(random_state)
        static_info = static_clone.step(self.static_parameters)
        np.random.set_state(random_state)
        info = self.session.step(parameters)

        tiny = 1e-300
        current = info.best
        improvement = float(np.log10((previous + tiny) / (current + tiny)))
        counterfactual_improvement = float(
            np.log10((previous + tiny) / (static_info.best + tiny))
        )
        relative_improvement = improvement - counterfactual_improvement

        # Retain the deployment-state definition used by DynamicConfigurationEnv.
        self.current_advantage = float(np.log10(
            (self.static_trace[info.fe] + tiny) / (current + tiny)
        ))
        self.last_improvement = improvement
        self.ema_improvement = 0.8 * self.ema_improvement + 0.2 * improvement
        self.stagnation = 0 if improvement > 1e-8 else self.stagnation + 1
        self.previous_best = current
        reward = float(
            np.clip(0.1 * improvement + self.relative_weight * relative_improvement, -5.0, 5.0)
            - self.movement_penalty * np.mean(action ** 2)
        )
        terminated = self.session.done
        bonus = 0.0
        if terminated:
            bonus = self.terminal_bonus * float(np.clip(self.current_advantage, -5.0, 5.0))
            reward += bonus
        details = {
            "fe": info.fe,
            "previous_fe": previous_fe,
            "best": current,
            "static_final": self.static_final,
            "improvement": improvement,
            "counterfactual_best": static_info.best,
            "counterfactual_improvement": counterfactual_improvement,
            "relative_improvement": relative_improvement,
            "terminal_bonus": bonus,
            "advantage": self.current_advantage,
            "parameters": parameters.copy(),
            "action_rms": float(np.sqrt(np.mean(action ** 2))),
        }
        return self._state(), reward, terminated, details


class CriticGatedCounterfactualEnv(DynamicConfigurationEnv):
    """Exact static fallback plus a same-state transition for critic grounding."""

    def reset(self, seed):
        self.session = NeuroEASession(
            self.problem_name, self.protocol, self.dimension, self.population_size,
            self.max_fe, int(seed), self.static_parameters,
        )
        self.current_normalized = self.static_normalized.copy()
        self.current_residual = np.zeros(30, dtype=np.float64)
        self.initial_best = self.previous_best = self.session.best()
        self.last_improvement = self.ema_improvement = 0.0
        self.current_advantage = 0.0
        self.stagnation = 0
        return self._state(), {"initial_best": self.initial_best}

    def _state_for(self, session, best, last_improvement, ema_improvement,
                   advantage, stagnation, residual):
        tiny = 1e-300
        progress = session.problem.fe / self.max_fe
        total_gain = np.log10((self.initial_best + tiny) / (best + tiny)) / 20.0
        features = np.array([
            2.0 * progress - 1.0,
            np.clip(total_gain, -1.0, 1.0),
            np.clip(last_improvement / 5.0, -1.0, 1.0),
            np.clip(ema_improvement / 5.0, -1.0, 1.0),
            0.0,  # Reserved: deployment cannot observe counterfactual advantage.
            np.clip(session.diversity() * 20.0, 0.0, 1.0),
            np.clip(stagnation / 20.0, 0.0, 1.0),
        ], dtype=np.float32)
        scaled = residual / np.maximum(self.radius, 1e-12)
        return np.concatenate([features, np.clip(scaled, -1.0, 1.0).astype(np.float32)])

    def _state(self):
        return self._state_for(
            self.session, self.previous_best, self.last_improvement,
            self.ema_improvement, 0.0, self.stagnation, self.current_residual,
        )

    def step_gated(self, action, execute_dynamic):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        previous = self.previous_best
        random_state = np.random.get_state()

        # Ground Q(s, 0) with the exact same population and random numbers.
        static_clone = copy.deepcopy(self.session)
        np.random.set_state(random_state)
        static_info = static_clone.step(self.static_parameters)
        static_random_after = np.random.get_state()
        tiny = 1e-300
        static_improvement = float(np.log10((previous + tiny) / (static_info.best + tiny)))
        static_ema = 0.8 * self.ema_improvement + 0.2 * static_improvement
        static_stagnation = 0 if static_improvement > 1e-8 else self.stagnation + 1
        static_advantage = self.current_advantage
        zero_residual = np.zeros(30, dtype=np.float64)
        zero_next_state = self._state_for(
            static_clone, static_info.best, static_improvement, static_ema,
            static_advantage, static_stagnation, zero_residual,
        )
        zero_reward = float(np.clip(0.05 * static_improvement, -1.0, 1.0))

        if execute_dynamic:
            target_residual = self.radius * action
            candidate_residual = (
                (1.0 - self.smoothing) * self.current_residual
                + self.smoothing * target_residual
            )
            candidate_normalized = np.clip(
                self.static_normalized + candidate_residual, -1.0, 1.0
            )
            parameters = normalized_to_parameters(candidate_normalized)
            parameters[candidate_residual == 0.0] = self.static_parameters[candidate_residual == 0.0]
            np.random.set_state(random_state)
            info = self.session.step(parameters)
            current = info.best
            improvement = float(np.log10((previous + tiny) / (current + tiny)))
            relative = improvement - static_improvement
            reward = float(
                np.clip(0.05 * improvement + self.relative_weight * relative, -5.0, 5.0)
                - self.movement_penalty * np.mean(action ** 2)
            )
            self.current_residual = candidate_residual
            self.current_normalized = candidate_normalized
        else:
            self.session = static_clone
            np.random.set_state(static_random_after)
            info = static_info
            current = info.best
            improvement = static_improvement
            relative = 0.0
            reward = zero_reward
            parameters = self.static_parameters.copy()
            self.current_residual = zero_residual
            self.current_normalized = self.static_normalized.copy()

        self.current_advantage += relative
        self.last_improvement = improvement
        self.ema_improvement = 0.8 * self.ema_improvement + 0.2 * improvement
        self.stagnation = 0 if improvement > 1e-8 else self.stagnation + 1
        self.previous_best = current
        details = {
            "fe": info.fe,
            "best": current,
            "executed_dynamic": bool(execute_dynamic),
            "improvement": improvement,
            "static_improvement": static_improvement,
            "relative_improvement": relative,
            "parameters": parameters.copy(),
            "counterfactual_next_state": zero_next_state,
            "counterfactual_reward": zero_reward,
        }
        return self._state(), reward, self.session.done, details

    def step_deploy(self, action, execute_dynamic):
        """Advance only one NeuroEA generation; no training counterfactual is run."""
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if execute_dynamic:
            target_residual = self.radius * action
            self.current_residual = (
                (1.0 - self.smoothing) * self.current_residual
                + self.smoothing * target_residual
            )
            self.current_normalized = np.clip(
                self.static_normalized + self.current_residual, -1.0, 1.0
            )
            parameters = normalized_to_parameters(self.current_normalized)
            parameters[self.current_residual == 0.0] = self.static_parameters[self.current_residual == 0.0]
        else:
            self.current_residual = np.zeros(30, dtype=np.float64)
            self.current_normalized = self.static_normalized.copy()
            parameters = self.static_parameters.copy()
        previous = self.previous_best
        info = self.session.step(parameters)
        tiny = 1e-300
        improvement = float(np.log10((previous + tiny) / (info.best + tiny)))
        self.last_improvement = improvement
        self.ema_improvement = 0.8 * self.ema_improvement + 0.2 * improvement
        self.stagnation = 0 if improvement > 1e-8 else self.stagnation + 1
        self.previous_best = info.best
        details = {
            "fe": info.fe, "best": info.best, "executed_dynamic": bool(execute_dynamic),
            "parameters": parameters.copy(),
        }
        return self._state(), self.session.done, details
