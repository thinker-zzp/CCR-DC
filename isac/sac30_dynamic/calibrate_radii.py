from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from .core import (
    NeuroEASession, normalized_to_parameters, objective_gap,
    parameters_to_normalized,
)


def response(reference, normalized, index, radius, session, random_state):
    tiny = 1e-300
    values = []
    for sign in (-1.0, 1.0):
        candidate_n = normalized.copy()
        candidate_n[index] = np.clip(candidate_n[index] + sign * radius, -1.0, 1.0)
        candidate = normalized_to_parameters(candidate_n)
        if candidate[index] == reference[index]:
            continue
        trial = copy.deepcopy(session)
        np.random.set_state(random_state)
        trial.step(candidate)
        values.append(trial.population_statistics())
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-parameters", default=None)
    parser.add_argument("--problem", default="SOP_F1")
    parser.add_argument("--protocol", default="paper")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--checkpoints", default="20,50,80")
    parser.add_argument("--target", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=606000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    slug = args.problem.lower().removeprefix("sop_")
    static_parameters = (args.static_parameters
                         or f"results/sac30_static_{slug}_robust/best_parameters.npy")
    output = Path(args.output or f"results/sac30_parameter_sensitivity_{slug}")
    output.mkdir(parents=True, exist_ok=True)
    reference = np.load(static_parameters).astype(np.float64)
    reference_n = parameters_to_normalized(reference)
    checkpoints = {int(item) for item in args.checkpoints.split(",")}
    continuous = np.array([3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    # One integer tournament step is about 2/9 in normalized coordinates.
    discrete = np.array([0.06, 0.12, 0.25])
    observations = {(i, float(r)): [] for i in range(30)
                    for r in (discrete if i < 3 else continuous)}

    for seed_index in range(args.seeds):
        seed = args.seed + seed_index
        print(f"[{args.problem}] sensitivity seed {seed_index + 1}/{args.seeds}", flush=True)
        session = NeuroEASession(args.problem, args.protocol, seed=seed, parameters=reference)
        generation = 0
        while not session.done:
            generation += 1
            if generation in checkpoints:
                random_state = np.random.get_state()
                baseline = copy.deepcopy(session)
                np.random.set_state(random_state); baseline.step(reference)
                base = baseline.population_statistics()
                for index in range(30):
                    candidates = discrete if index < 3 else continuous
                    for radius in candidates:
                        for stats in response(reference, reference_n, index, radius,
                                              session, random_state):
                            best_effect = abs(np.log10(
                                objective_gap(args.problem, base["best"], session.problem.D)
                                / objective_gap(args.problem, stats["best"], session.problem.D)
                            ))
                            median_effect = abs(np.log10(
                                objective_gap(args.problem, base["median"], session.problem.D)
                                / objective_gap(args.problem, stats["median"], session.problem.D)
                            ))
                            observations[(index, float(radius))].append(
                                best_effect + 0.3 * median_effect)
                np.random.set_state(random_state)
            session.step(reference)

    rows, selected = [], np.empty(30, dtype=np.float64)
    for index in range(30):
        candidates = discrete if index < 3 else continuous
        scores = []
        for radius in candidates:
            values = observations[(index, float(radius))]
            score = float(np.mean(values)) if values else 0.0
            rows.append({"parameter": index + 1, "radius": radius,
                         "mean_absolute_effect": score, "samples": len(values)})
            scores.append(score)
        scores = np.asarray(scores)
        valid = np.where(scores > 0)[0]
        choice = valid[np.argmin(np.abs(scores[valid] - args.target))] if len(valid) else 0
        selected[index] = candidates[choice]
    np.save(output / "radii.npy", selected)
    pd.DataFrame(rows).to_csv(output / "sensitivity.csv", index=False)
    pd.DataFrame({"parameter": np.arange(1, 31), "selected_radius": selected}).to_csv(
        output / "selected_radii.csv", index=False)
    print("Selected radii:", np.array2string(selected, precision=5))


if __name__ == "__main__":
    main()
