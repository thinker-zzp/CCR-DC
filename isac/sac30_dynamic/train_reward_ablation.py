from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .advanced_env import AdvancedCounterfactualEnv
from .core import objective_gap, run_fixed
from .continuous27_env import Continuous27ResidualCounterfactualEnv
from .direct27_env import Continuous27DirectControlEnv
from .sac_agent import SACAgent
from .scaled_residual_env import ScaledResidualCounterfactualEnv
from .trust_region_env import TrustRegionCounterfactualEnv


def paired_validation_scores(env, actor, seeds, static_parameters, static_cache=None):
    """Return deterministic paired Dynamic/Static log-gap diagnostics."""
    ratios = []
    static_cache = {} if static_cache is None else static_cache
    for seed in seeds:
        seed = int(seed)
        if seed not in static_cache:
            static_cache[seed] = run_fixed(
                static_parameters, seed, env.problem_name, env.protocol,
                env.dimension, env.population_size, env.max_fe,
            )
        state, _ = env.reset(seed)
        while True:
            action = actor.act(state, deterministic=True)
            state, done, info = env.step_deploy(action, True)
            if done:
                ratios.append(np.log10(
                    objective_gap(env.problem_name, info["best"], env.dimension)
                    / objective_gap(
                        env.problem_name, static_cache[seed], env.dimension
                    )
                ))
                break
    values = np.asarray(ratios, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "q90": float(np.quantile(values, 0.9)),
        "win_rate": float(np.mean(values < -1e-12)),
        "tie_rate": float(np.mean(np.abs(values) <= 1e-12)),
    }


def save_resume_checkpoint(path, actor, episode, rows, best_actor_score,
                           env, exploration_rng):
    data = {
        "format_version": 1,
        "episode": int(episode),
        "rows": rows,
        "best_actor_score": float(best_actor_score),
        "actor": actor.training_state_dict(),
        "numpy_global_state": np.random.get_state(),
        "exploration_rng_state": copy.deepcopy(exploration_rng.bit_generator.state),
        "reward_history": list(env.reward_history),
        "counterfactual_reward_history": list(
            env.counterfactual_reward_history
        ),
        "ordinary_reward_history": list(env.ordinary_reward_history),
        "trust_region_state": (
            env.trust_region_state_dict()
            if hasattr(env, "trust_region_state_dict") else None
        ),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    os.replace(temporary, path)


def load_resume_checkpoint(path, actor, env, exploration_rng, device):
    # The checkpoint is generated locally by this script and intentionally
    # contains NumPy RNG/replay objects in addition to tensors. PyTorch >=2.6
    # therefore requires the explicit trusted-local ``weights_only=False``.
    data = torch.load(path, map_location=device, weights_only=False)
    if int(data.get("format_version", 0)) != 1:
        raise ValueError("Unsupported reward-ablation checkpoint format.")
    actor.load_training_state_dict(data["actor"])
    np.random.set_state(data["numpy_global_state"])
    exploration_rng.bit_generator.state = copy.deepcopy(data["exploration_rng_state"])
    env.reward_history[:] = [float(value) for value in data["reward_history"]]
    env.counterfactual_reward_history[:] = [
        float(value)
        for value in data.get("counterfactual_reward_history", [])
    ]
    env.ordinary_reward_history[:] = [
        float(value) for value in data.get("ordinary_reward_history", [])
    ]
    if data.get("trust_region_state") is not None:
        if not hasattr(env, "load_trust_region_state_dict"):
            raise ValueError("Checkpoint requires a trust-region environment.")
        env.load_trust_region_state_dict(data["trust_region_state"])
    return int(data["episode"]), list(data["rows"]), float(data["best_actor_score"])


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Gate-free, compute-matched Counterfactual-vs-Ordinary reward "
            "ablation for the 44-D residual NeuroEA controller."
        )
    )
    parser.add_argument("--problem", required=True)
    parser.add_argument("--protocol", choices=["platemo", "isac", "paper", "matlab", "senior"],
                        default="platemo")
    parser.add_argument(
        "--reward-mode",
        choices=["counterfactual", "ordinary", "hybrid"],
        required=True,
    )
    parser.add_argument("--hybrid-cf-weight", type=float, default=0.25)
    parser.add_argument("--actor-episodes", type=int, default=1500)
    parser.add_argument("--actor-validation-every", type=int, default=100)
    parser.add_argument("--actor-validation-runs", type=int, default=30)
    parser.add_argument("--selection-tail-weight", type=float, default=0.5)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--training-replication", type=int, required=True)
    parser.add_argument("--validation-seed-base", type=int, required=True)
    parser.add_argument("--continuous-budget", type=float, default=1.0)
    parser.add_argument("--integer-discrete", action="store_true")
    parser.add_argument("--continuous-scale", type=float, default=1.0)
    parser.add_argument("--freeze-integer", action="store_true")
    parser.add_argument(
        "--continuous-only-27d",
        action="store_true",
        help="Use a true 27-D action space and keep parameters 1--3 fixed.",
    )
    parser.add_argument(
        "--direct-continuous-control",
        action="store_true",
        help=(
            "Use a 27-D direct normalized continuous action; parameters 1--3 "
            "remain fixed at the Static Prior."
        ),
    )
    parser.add_argument(
        "--no-l2-projection",
        action="store_true",
        help="Disable only the global l2 projection while retaining the scale.",
    )
    parser.add_argument("--trust-region", action="store_true")
    parser.add_argument("--trust-region-target", type=float, default=0.5)
    parser.add_argument("--dual-learning-rate", type=float, default=1e-3)
    parser.add_argument("--dual-initial", type=float, default=0.05)
    parser.add_argument("--dual-max", type=float, default=5.0)
    parser.add_argument("--integer-switch-cost", type=float, default=1.0)
    parser.add_argument("--target-entropy", type=float, default=-5.0)
    parser.add_argument("--actor-log-std", type=float, default=-4.0)
    parser.add_argument(
        "--terminal-static-weight",
        type=float,
        default=0.0,
        help=(
            "Add a robust-normalized terminal log-gap advantage against a "
            "full Static Prior run on the same training seed."
        ),
    )
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--static-parameters", required=True)
    parser.add_argument("--radii", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.actor_episodes < 1 or args.actor_validation_every < 1:
        raise ValueError("Episode and validation intervals must be positive.")
    if args.actor_validation_runs < 1 or args.checkpoint_every < 1:
        raise ValueError("Validation runs and checkpoint interval must be positive.")
    if args.terminal_static_weight < 0.0:
        raise ValueError("terminal-static-weight must be nonnegative.")
    if args.continuous_only_27d and (args.integer_discrete or args.trust_region):
        raise ValueError(
            "continuous-only-27d cannot be combined with integer-discrete or trust-region."
        )
    if args.direct_continuous_control and (
        args.continuous_only_27d or args.integer_discrete or args.trust_region
    ):
        raise ValueError(
            "direct-continuous-control cannot be combined with other action modes."
        )

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.direct_continuous_control and (output / "resume_checkpoint.pt").exists():
        previous_manifest = json.loads(
            (output / "training_manifest.json").read_text(encoding="utf-8")
        )
        if previous_manifest.get("static_replay_semantics") != "direct_prior_v1":
            raise RuntimeError(
                "Legacy Direct replay is incompatible with the corrected mapping; "
                "start a fresh training output directory."
            )
    completion_marker = output / ".complete"
    if completion_marker.exists() and (output / "selected_actor.pt").exists():
        checkpoint_path = output / "resume_checkpoint.pt"
        completed = 0
        if checkpoint_path.exists():
            completed = int(torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            ).get("episode", 0))
        if completed >= args.actor_episodes:
            print(f"[{args.problem} {args.reward_mode}] already complete", flush=True)
            return
        completion_marker.unlink()
        print(
            f"[{args.problem} {args.reward_mode}] extending "
            f"{completed} -> {args.actor_episodes} episodes",
            flush=True,
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    exploration_rng = np.random.default_rng(args.seed + 101)
    replay_rng = np.random.default_rng(args.seed + 202)

    static_parameters = np.load(args.static_parameters)
    radii = np.load(args.radii)
    env_kwargs = dict(
        static_parameters=static_parameters,
        radii=radii,
        problem=args.problem,
        protocol=args.protocol,
        horizon=args.horizon,
        continuous_budget=(None if args.no_l2_projection else args.continuous_budget),
        integer_discrete=args.integer_discrete,
        hybrid_cf_weight=args.hybrid_cf_weight,
    )
    env_kwargs.update(
        continuous_scale=args.continuous_scale,
        freeze_integer=args.freeze_integer,
    )
    if args.direct_continuous_control:
        env = Continuous27DirectControlEnv(
            **env_kwargs, reward_mode=args.reward_mode
        )
    elif args.continuous_only_27d:
        env = Continuous27ResidualCounterfactualEnv(
            **env_kwargs, reward_mode=args.reward_mode
        )
    elif args.trust_region:
        env = TrustRegionCounterfactualEnv(
            **env_kwargs,
            reward_mode=args.reward_mode,
            trust_region_target=args.trust_region_target,
            dual_learning_rate=args.dual_learning_rate,
            dual_initial=args.dual_initial,
            dual_max=args.dual_max,
            integer_switch_cost=args.integer_switch_cost,
        )
    else:
        env = ScaledResidualCounterfactualEnv(
            **env_kwargs, reward_mode=args.reward_mode
        )
    validation_env_class = (
        Continuous27DirectControlEnv
        if args.direct_continuous_control
        else (
            Continuous27ResidualCounterfactualEnv
            if args.continuous_only_27d
            else ScaledResidualCounterfactualEnv
        )
    )
    validation_env = validation_env_class(**env_kwargs)
    actor = SACAgent(
        env.state_dim, env.action_dim, args.device, gamma=0.95, alpha=0.05,
        batch_size=256, capacity=200_000, target_entropy=args.target_entropy,
        replay_rng=replay_rng,
    )
    initial_actor_action = np.asarray(
        getattr(env, "initial_actor_action", np.zeros(env.action_dim)),
        dtype=np.float32,
    )
    actor.initialize_actor_mean(initial_actor_action, log_std=args.actor_log_std)

    manifest = {
        "experiment": (
            "terminal_static_regret_reward"
            if args.terminal_static_weight > 0.0
            else (
                "ordinary_dominant_hybrid_reward"
                if args.reward_mode == "hybrid"
                else "counterfactual_vs_ordinary_reward_ablation"
            )
        ),
        "problem": args.problem,
        "protocol": args.protocol,
        "reward_mode": args.reward_mode,
        "hybrid_cf_weight": args.hybrid_cf_weight,
        "state_dim": env.state_dim,
        "action_dim": env.action_dim,
        "action_mapping": (
            "27-D direct normalized continuous control; parameters 1--3 fixed"
            if args.direct_continuous_control
            else (
                "27-D continuous residual; parameters 1--3 fixed at Static Prior"
                if args.continuous_only_27d
                else "bounded residual around the fixed Static Prior"
            )
        ),
        "continuous_residual_scale": (
            None if args.direct_continuous_control else args.continuous_scale
        ),
        "integer_parameters_frozen": bool(
            args.freeze_integer or args.continuous_only_27d
        ),
        "continuous_only_27d": args.continuous_only_27d,
        "direct_continuous_control": args.direct_continuous_control,
        "static_replay_semantics": (
            "direct_prior_v1" if args.direct_continuous_control else "zero_residual"
        ),
        "l2_projection": (
            None if args.direct_continuous_control else not args.no_l2_projection
        ),
        "continuous_budget": (
            None if args.no_l2_projection else args.continuous_budget
        ),
        "terminal_static_weight": args.terminal_static_weight,
        "terminal_reward": (
            "clipped final log10(Static gap / Dynamic gap)"
            if args.terminal_static_weight > 0.0 else None
        ),
        "adaptive_trust_region": args.trust_region,
        "trust_region_target": args.trust_region_target if args.trust_region else None,
        "dual_learning_rate": args.dual_learning_rate if args.trust_region else None,
        "dual_initial": args.dual_initial if args.trust_region else None,
        "dual_max": args.dual_max if args.trust_region else None,
        "integer_switch_cost": args.integer_switch_cost if args.trust_region else None,
        "horizon_generations": args.horizon,
        "actor_episodes": args.actor_episodes,
        "actor_validation_every": args.actor_validation_every,
        "actor_validation_runs": args.actor_validation_runs,
        "selection_tail_weight": args.selection_tail_weight,
        "training_seed": args.seed,
        "training_replication": args.training_replication,
        "validation_seed_range": [
            args.validation_seed_base,
            args.validation_seed_base + args.actor_validation_runs - 1,
        ],
        "static_parameters": str(Path(args.static_parameters).resolve()),
        "radii": str(Path(args.radii).resolve()),
        "device": str(args.device),
        "gate_training": False,
        "rng_isolation": {
            "evolution": "NumPy global RNG reset by each NeuroEA episode seed",
            "exploration": "dedicated NumPy Generator(seed+101)",
            "replay_sampling": "dedicated NumPy Generator(seed+202)",
            "policy_sampling": "PyTorch RNG",
        },
        "compute_matching": {
            "dynamic_rollouts_per_decision": 1,
            "static_shadow_rollouts_per_decision": 1,
            "replay_transitions_per_decision": 2,
            "SAC_updates_per_decision": 1,
            "checkpoint_selection": "same paired validation seeds and score",
        },
        "ordinary_reward_note": (
            "The static shadow is retained solely to match compute and provide "
            "the logically corresponding zero-action replay transition. The "
            "ordinary actor reward uses only within-trajectory progress."
        ),
        "hybrid_reward_note": (
            "Hybrid separately robust-normalizes Ordinary and Counterfactual "
            "targets, forms (Ordinary + lambda*Counterfactual)/(1+lambda), "
            "and subtracts the movement penalty exactly once."
        ),
    }
    (output / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    checkpoint = output / "resume_checkpoint.pt"
    rows, best_actor_score, completed_episode = [], float("inf"), 0
    if checkpoint.exists() and not args.no_resume:
        completed_episode, rows, best_actor_score = load_resume_checkpoint(
            checkpoint, actor, env, exploration_rng, args.device
        )
        print(
            f"[{args.problem} {args.reward_mode}] resumed after episode "
            f"{completed_episode}",
            flush=True,
        )

    validation_seeds = [
        args.validation_seed_base + index for index in range(args.actor_validation_runs)
    ]
    static_cache = {}
    for episode in range(completed_episode + 1, args.actor_episodes + 1):
        terminal_static_objective = np.nan
        if args.terminal_static_weight > 0.0:
            terminal_static_objective = run_fixed(
                static_parameters,
                args.seed + episode,
                args.problem,
                args.protocol,
                dimension=10,
                population_size=100,
                max_fe=10_000,
            )
        state, _ = env.reset(args.seed + episode)
        reward_sum = raw_reward_sum = zero_reward_sum = 0.0
        raw_counterfactual_sum = raw_ordinary_sum = 0.0
        raw_values, normalized_values, scales, action_norms = [], [], [], []
        normalized_counterfactual_values = []
        normalized_ordinary_values = []
        counterfactual_scales, ordinary_scales = [], []
        counterfactual_clip_positive = []
        counterfactual_clip_negative = []
        ordinary_clip_positive = []
        ordinary_clip_negative = []
        movement_penalties = []
        trust_region_costs, trust_region_violations = [], []
        continuous_action_energies, integer_switch_counts = [], []
        dual_variables, constraint_penalties = [], []
        dynamic_fe = shadow_fe = decisions = 0
        raw_terminal_advantage = normalized_terminal_advantage = 0.0
        terminal_reward_scale = terminal_bonus = 0.0
        while True:
            if episode <= 10:
                action = np.clip(
                    initial_actor_action
                    + exploration_rng.normal(0.0, 0.25, env.action_dim),
                    -1.0,
                    1.0,
                ).astype(np.float32)
            else:
                action = actor.act(state, deterministic=False).astype(np.float32)
            next_state, reward, done, info = env.step_training(action)
            if done and args.terminal_static_weight > 0.0:
                raw_terminal_advantage = float(np.log10(
                    objective_gap(
                        args.problem, terminal_static_objective, 10
                    )
                    / objective_gap(args.problem, info["best"], 10)
                ))
                # The advantage is already measured in interpretable log10
                # orders of magnitude.  A fixed scale avoids exact-hit values
                # near +/-300 distorting an adaptive history-based scale.
                terminal_reward_scale = 1.0
                normalized_terminal_advantage = float(np.clip(
                    raw_terminal_advantage,
                    -5.0,
                    5.0,
                ))
                terminal_bonus = float(
                    args.terminal_static_weight
                    * normalized_terminal_advantage
                )
                reward += terminal_bonus
            actor.replay.add(state, action, reward, next_state, done)
            actor.replay.add(
                state, info.get("static_action", np.zeros(env.action_dim, dtype=np.float32)),
                info["zero_reward"],
                info["zero_next_state"], done,
            )
            actor.update(1)
            state = next_state
            reward_sum += reward
            raw_reward_sum += info["raw_reward"]
            zero_reward_sum += info["zero_reward"]
            raw_counterfactual_sum += info["raw_counterfactual"]
            raw_ordinary_sum += info["raw_ordinary"]
            raw_values.append(info["raw_reward"])
            normalized_values.append(info["normalized_reward"])
            scales.append(info["reward_scale"])
            normalized_counterfactual_values.append(
                info["normalized_counterfactual"]
            )
            normalized_ordinary_values.append(info["normalized_ordinary"])
            counterfactual_scales.append(info["counterfactual_scale"])
            ordinary_scales.append(info["ordinary_scale"])
            counterfactual_clip_positive.append(
                info["counterfactual_clip_positive"]
            )
            counterfactual_clip_negative.append(
                info["counterfactual_clip_negative"]
            )
            ordinary_clip_positive.append(info["ordinary_clip_positive"])
            ordinary_clip_negative.append(info["ordinary_clip_negative"])
            movement_penalties.append(info["movement_penalty"])
            if args.trust_region:
                trust_region_costs.append(info["trust_region_cost"])
                trust_region_violations.append(info["trust_region_violation"])
                continuous_action_energies.append(info["continuous_action_energy"])
                integer_switch_counts.append(info["integer_switches"])
                dual_variables.append(info["dual_variable"])
                constraint_penalties.append(info["constraint_penalty"])
            action_norms.append(float(np.sqrt(np.mean(info["effective_action"] ** 2))))
            dynamic_fe += info["dynamic_fe"]
            shadow_fe += info["shadow_fe"]
            decisions += 1
            if done:
                break

        validation_mean = validation_q90 = selection_score = np.nan
        if episode % args.actor_validation_every == 0 or episode == args.actor_episodes:
            metrics = paired_validation_scores(
                validation_env, actor, validation_seeds, static_parameters, static_cache
            )
            validation_mean, validation_q90 = metrics["mean"], metrics["q90"]
            selection_score = (
                validation_mean
                + args.selection_tail_weight * max(validation_q90, 0.0)
            )
            if selection_score < best_actor_score:
                best_actor_score = selection_score
                actor.save(output / "selected_actor.pt")

        rows.append({
            "reward_mode": args.reward_mode,
            "episode": episode,
            "reward_sum": reward_sum,
            "raw_reward_sum": raw_reward_sum,
            "zero_reward_sum": zero_reward_sum,
            "raw_counterfactual_sum": raw_counterfactual_sum,
            "raw_ordinary_sum": raw_ordinary_sum,
            "raw_mean": float(np.mean(raw_values)),
            "raw_std": float(np.std(raw_values)),
            "raw_positive_rate": float(np.mean(np.asarray(raw_values) > 0.0)),
            "normalized_mean": float(np.mean(normalized_values)),
            "normalized_std": float(np.std(normalized_values)),
            "reward_scale_mean": float(np.mean(scales)),
            "normalized_counterfactual_mean": float(np.mean(
                normalized_counterfactual_values
            )),
            "normalized_ordinary_mean": float(np.mean(
                normalized_ordinary_values
            )),
            "counterfactual_scale_mean": float(np.nanmean(
                counterfactual_scales
            )) if np.isfinite(counterfactual_scales).any() else np.nan,
            "ordinary_scale_mean": float(np.nanmean(
                ordinary_scales
            )) if np.isfinite(ordinary_scales).any() else np.nan,
            "counterfactual_clip_positive_rate": float(np.mean(
                counterfactual_clip_positive
            )),
            "counterfactual_clip_negative_rate": float(np.mean(
                counterfactual_clip_negative
            )),
            "ordinary_clip_positive_rate": float(np.mean(
                ordinary_clip_positive
            )),
            "ordinary_clip_negative_rate": float(np.mean(
                ordinary_clip_negative
            )),
            "movement_penalty_mean": float(np.mean(movement_penalties)),
            "terminal_static_objective": terminal_static_objective,
            "raw_terminal_advantage": raw_terminal_advantage,
            "normalized_terminal_advantage": normalized_terminal_advantage,
            "terminal_reward_scale": terminal_reward_scale,
            "terminal_bonus": terminal_bonus,
            "trust_region_cost_mean": (
                float(np.mean(trust_region_costs)) if args.trust_region else np.nan
            ),
            "trust_region_violation_mean": (
                float(np.mean(trust_region_violations)) if args.trust_region else np.nan
            ),
            "continuous_action_energy_mean": (
                float(np.mean(continuous_action_energies))
                if args.trust_region else np.nan
            ),
            "integer_switch_rate": (
                float(np.mean(np.asarray(integer_switch_counts) > 0))
                if args.trust_region else np.nan
            ),
            "dual_variable_mean": (
                float(np.mean(dual_variables)) if args.trust_region else np.nan
            ),
            "dual_variable_final": (
                float(env.dual_variable) if args.trust_region else np.nan
            ),
            "constraint_penalty_mean": (
                float(np.mean(constraint_penalties)) if args.trust_region else np.nan
            ),
            "episode_final_dynamic_gap": float(objective_gap(
                args.problem, info["best"], 10
            )),
            "action_rms_mean": float(np.mean(action_norms)),
            "decisions": decisions,
            "dynamic_fe": dynamic_fe,
            "shadow_fe": shadow_fe,
            "actor_validation_mean": validation_mean,
            "actor_validation_q90": validation_q90,
            "actor_selection_score": selection_score,
            "best_actor_score": best_actor_score,
        })

        if episode % 10 == 0 or episode == 1:
            print(
                f"[{args.problem} {args.reward_mode}] actor episode "
                f"{episode}/{args.actor_episodes}",
                flush=True,
            )
        if (episode % args.checkpoint_every == 0
                or episode == args.actor_episodes
                or episode % args.actor_validation_every == 0):
            actor.save(output / "latest_actor.pt")
            save_resume_checkpoint(
                checkpoint, actor, episode, rows, best_actor_score, env,
                exploration_rng,
            )

    if not (output / "selected_actor.pt").exists():
        actor.save(output / "selected_actor.pt")
    actor.save(output / "latest_actor.pt")

    with (output / "training_log.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    log = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True)
    axes[0].plot(
        log.episode,
        log.reward_sum.rolling(20, min_periods=1).median(),
        label="20-episode median training reward",
    )
    validation = log.dropna(subset=["actor_validation_mean"])
    axes[0].scatter(
        validation.episode, validation.actor_validation_mean,
        color="tab:red", label="validation log10(Dynamic/Static)",
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Training / validation score")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(log.episode, log.raw_positive_rate, label="positive reward fraction")
    axes[1].plot(log.episode, log.action_rms_mean, label="action RMS", alpha=0.8)
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel("Reward/action diagnostics")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "training_convergence.png", dpi=220)
    plt.close(fig)

    completion_marker.write_text("complete\n", encoding="utf-8")
    print(f"[{args.problem} {args.reward_mode}] complete: {output}", flush=True)


if __name__ == "__main__":
    main()
