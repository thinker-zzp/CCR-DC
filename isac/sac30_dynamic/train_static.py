from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .core import (
    DEFAULT_REFERENCE,
    F1_REFERENCE,
    parameters_to_normalized,
    static_search_reference,
)
from .envs import StaticConfigurationBandit
from .sac_agent import SACAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="SOP_F1")
    parser.add_argument(
        "--protocol",
        choices=["platemo", "senior", "isac", "paper", "matlab"],
        default="paper",
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--validation-runs", type=int, default=30)
    parser.add_argument("--validation-every", type=int, default=50)
    parser.add_argument("--continuous-radius", type=float, default=0.08)
    parser.add_argument("--discrete-radius", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--reference", choices=["auto", "default", "f1-specialized"], default="auto",
        help="Static search center. Auto uses F1-specialized only for SOP_F1.",
    )
    parser.add_argument("--reference-parameters", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    slug = args.problem.lower().removeprefix("sop_")
    output = Path(args.output or f"results/sac30_static_{slug}_robust")
    output.mkdir(parents=True, exist_ok=True)
    env = StaticConfigurationBandit(args.problem, args.protocol, args.repeats)
    validation_env = StaticConfigurationBandit(args.problem, args.protocol, args.validation_runs)
    agent = SACAgent(env.state_dim, env.action_dim, args.device, batch_size=128, capacity=20_000)
    if args.reference_parameters:
        reference_parameters = np.load(args.reference_parameters).astype(np.float64)
        reference_source = str(args.reference_parameters)
    elif args.reference == "default":
        reference_parameters = DEFAULT_REFERENCE.astype(np.float64)
        reference_source = "DEFAULT_REFERENCE"
    elif args.reference == "f1-specialized":
        reference_parameters = F1_REFERENCE.astype(np.float64)
        reference_source = "F1_REFERENCE"
    else:
        reference_parameters = static_search_reference(args.problem)
        reference_source = ("F1_REFERENCE" if args.problem.upper() == "SOP_F1"
                            else "DEFAULT_REFERENCE")
    if reference_parameters.shape != (30,):
        raise ValueError("Static reference must contain exactly 30 parameters.")
    reference_action = parameters_to_normalized(reference_parameters)
    agent.initialize_actor_mean(np.zeros(30))
    radius = np.full(30, args.continuous_radius)
    radius[:3] = args.discrete_radius
    compose = lambda residual: np.clip(reference_action + radius * residual, -1.0, 1.0)
    state = np.zeros(1, dtype=np.float32)
    validation_seed = args.seed + 9_000_000
    best_reward, best_fitness, best_parameters, _ = validation_env.evaluate(reference_action, validation_seed)
    np.save(output / "reference_parameters.npy", reference_parameters)
    np.save(output / "best_parameters.npy", best_parameters)
    agent.save(output / "best_agent.pt")
    (output / "training_config.json").write_text(json.dumps({
        "problem": args.problem,
        "protocol": args.protocol,
        "episodes": args.episodes,
        "repeats": args.repeats,
        "validation_runs": args.validation_runs,
        "validation_every": args.validation_every,
        "continuous_radius": args.continuous_radius,
        "discrete_radius": args.discrete_radius,
        "seed": args.seed,
        "reference_source": reference_source,
    }, indent=2), encoding="utf-8")
    most_promising_action, most_promising_reward = np.zeros(30), -np.inf
    rows = []

    for episode in range(1, args.episodes + 1):
        if episode <= 100:
            action = np.clip(np.random.normal(0.0, 0.35, 30), -1.0, 1.0)
        else:
            action = agent.act(state, deterministic=False)
        reward, fitness, parameters, individual = env.evaluate(
            compose(action), args.seed + episode * args.repeats,
        )
        agent.replay.add(state, action.astype(np.float32), reward, state, True)
        agent.update(1)
        if reward > most_promising_reward:
            most_promising_reward, most_promising_action = reward, action.copy()
        validation_fitness = np.nan
        if episode % args.validation_every == 0:
            candidates = [agent.act(state, deterministic=True), most_promising_action]
            for candidate in candidates:
                _, candidate_fitness, candidate_parameters, _ = validation_env.evaluate(
                    compose(candidate), validation_seed)
                if candidate_fitness < best_fitness:
                    best_fitness, best_parameters = candidate_fitness, candidate_parameters.copy()
                    np.save(output / "best_parameters.npy", best_parameters)
                    agent.save(output / "best_agent.pt")
            validation_fitness = best_fitness
        rows.append({"episode": episode, "reward": reward, "fitness_mean": fitness,
                     "fitness_std": float(np.std(individual)), "validation_best": validation_fitness,
                     "best_fitness": best_fitness})
        if episode % 25 == 0 or episode == 1:
            print(f"[{args.problem}] static episode {episode}/{args.episodes}", flush=True)
        if episode % 100 == 0:
            agent.save(output / "latest_agent.pt")
            with (output / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)

    agent.save(output / "latest_agent.pt")
    np.save(output / "best_parameters.npy", best_parameters)
    with (output / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(f"best static mean fitness: {best_fitness:.8e}")


if __name__ == "__main__":
    main()
