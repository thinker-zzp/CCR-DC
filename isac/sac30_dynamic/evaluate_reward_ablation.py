from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .advanced_env import AdvancedCounterfactualEnv
from .core import objective_gap, run_fixed
from .continuous27_env import Continuous27ResidualCounterfactualEnv
from .direct27_env import Continuous27DirectControlEnv
from .sac_agent import SACAgent
from .scaled_residual_env import ScaledResidualCounterfactualEnv


def atomic_csv(frame, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def safe_label(label):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label)


def parse_model(text):
    if "=" not in text:
        raise argparse.ArgumentTypeError("Model must use LABEL=CHECKPOINT syntax.")
    label, checkpoint = text.split("=", 1)
    if not label or not checkpoint:
        raise argparse.ArgumentTypeError("Model must use LABEL=CHECKPOINT syntax.")
    return label, Path(checkpoint).resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Paired held-out evaluation for reward-ablation actors."
    )
    parser.add_argument("--problem", required=True)
    parser.add_argument("--protocol", choices=["platemo", "isac", "paper", "matlab", "senior"],
                        default="platemo")
    parser.add_argument("--static-parameters", required=True)
    parser.add_argument("--radii", required=True)
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--runs", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--continuous-budget", type=float, default=1.0)
    parser.add_argument("--integer-discrete", action="store_true")
    parser.add_argument("--continuous-scale", type=float, default=1.0)
    parser.add_argument("--freeze-integer", action="store_true")
    parser.add_argument(
        "--continuous-only-27d",
        action="store_true",
        help="Evaluate a true 27-D actor with parameters 1--3 fixed.",
    )
    parser.add_argument(
        "--direct-continuous-control",
        action="store_true",
        help="Evaluate a 27-D direct normalized continuous-control actor.",
    )
    parser.add_argument(
        "--no-l2-projection",
        action="store_true",
        help="Disable only the global l2 projection while retaining the scale.",
    )
    parser.add_argument("--target-entropy", type=float, default=-5.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("runs must be positive.")
    if args.continuous_only_27d and args.integer_discrete:
        raise ValueError(
            "continuous-only-27d cannot be combined with integer-discrete."
        )
    if args.direct_continuous_control and (
        args.continuous_only_27d or args.integer_discrete
    ):
        raise ValueError(
            "direct-continuous-control cannot be combined with other action modes."
        )
    output = Path(args.output).resolve()
    parts = output / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    static_parameters = np.load(args.static_parameters)
    radii = np.load(args.radii)
    seeds = [args.seed_base + index for index in range(args.runs)]

    static_path = parts / "static_prior.csv"
    if static_path.exists():
        static_frame = pd.read_csv(static_path)
        if static_frame.eval_seed.astype(int).tolist() != seeds:
            raise ValueError("Existing static cache uses a different evaluation seed range.")
    else:
        rows = []
        for index, seed in enumerate(seeds, start=1):
            value = run_fixed(
                static_parameters, seed, args.problem, args.protocol,
                dimension=10, population_size=100, max_fe=10_000,
            )
            rows.append({
                "problem": args.problem,
                "eval_seed": seed,
                "method": "Static Prior",
                "reward_mode": "static",
                "training_seed": -1,
                "training_replication": -1,
                "final_objective": value,
                "gap": float(objective_gap(args.problem, value, 10)),
                "static_objective": value,
                "static_gap": float(objective_gap(args.problem, value, 10)),
                "log10_dynamic_static": 0.0,
            })
            if index % 10 == 0 or index == args.runs:
                print(f"[{args.problem}] Static {index}/{args.runs}", flush=True)
        static_frame = pd.DataFrame(rows)
        atomic_csv(static_frame, static_path)

    static_by_seed = static_frame.set_index("eval_seed")
    result_frames = [static_frame]
    for label, checkpoint in args.model:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        part_path = parts / f"{safe_label(label)}.csv"
        if part_path.exists():
            frame = pd.read_csv(part_path)
            if frame.eval_seed.astype(int).tolist() != seeds:
                raise ValueError(f"Existing {label} cache uses a different seed range.")
            result_frames.append(frame)
            print(f"[{args.problem}] {label}: already complete", flush=True)
            continue

        model_dir = checkpoint.parent
        manifest_path = model_dir / "training_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        env_class = (
            Continuous27DirectControlEnv
            if args.direct_continuous_control
            else (
                Continuous27ResidualCounterfactualEnv
                if args.continuous_only_27d
                else ScaledResidualCounterfactualEnv
            )
        )
        env = env_class(
            static_parameters, radii, args.problem, args.protocol,
            horizon=args.horizon,
            continuous_budget=(
                None if args.no_l2_projection else args.continuous_budget
            ),
            integer_discrete=args.integer_discrete,
            continuous_scale=args.continuous_scale,
            freeze_integer=args.freeze_integer,
        )
        actor = SACAgent(
            env.state_dim, env.action_dim, args.device, gamma=0.95, alpha=0.05,
            batch_size=256, capacity=200_000, target_entropy=args.target_entropy,
        )
        actor.load(checkpoint)
        if int(manifest.get("action_dim", env.action_dim)) != env.action_dim:
            raise ValueError(
                f"{label}: checkpoint action dimension does not match evaluation mode."
            )
        rows = []
        for index, seed in enumerate(seeds, start=1):
            state, _ = env.reset(seed)
            continuous_energies = []
            integer_switches = []
            action_norms = []
            while True:
                action = actor.act(state, deterministic=True)
                state, done, info = env.step_deploy(action, True)
                effective = np.asarray(info["effective_action"], dtype=np.float64)
                continuous = effective if args.continuous_only_27d else effective[3:]
                continuous_energies.append(float(np.dot(continuous, continuous)))
                integer_switches.append(
                    0 if args.continuous_only_27d
                    else int(np.count_nonzero(effective[:3]))
                )
                action_norms.append(float(np.linalg.norm(effective)))
                if done:
                    value = float(info["best"])
                    break
            static_value = float(static_by_seed.loc[seed, "final_objective"])
            dynamic_gap = float(objective_gap(args.problem, value, 10))
            static_gap = float(objective_gap(args.problem, static_value, 10))
            rows.append({
                "problem": args.problem,
                "eval_seed": seed,
                "method": label,
                "reward_mode": manifest["reward_mode"],
                "training_seed": int(manifest["training_seed"]),
                "training_replication": int(manifest["training_replication"]),
                "final_objective": value,
                "gap": dynamic_gap,
                "static_objective": static_value,
                "static_gap": static_gap,
                "log10_dynamic_static": float(np.log10(dynamic_gap / static_gap)),
                "continuous_action_energy_mean": float(np.mean(continuous_energies)),
                "continuous_action_energy_max": float(np.max(continuous_energies)),
                "integer_switch_rate": float(
                    np.mean(np.asarray(integer_switches) > 0)
                ),
                "integer_switches_mean": float(np.mean(integer_switches)),
                "effective_action_norm_mean": float(np.mean(action_norms)),
                "effective_action_norm_max": float(np.max(action_norms)),
            })
            if index % 10 == 0 or index == args.runs:
                print(f"[{args.problem}] {label} {index}/{args.runs}", flush=True)
        frame = pd.DataFrame(rows)
        atomic_csv(frame, part_path)
        result_frames.append(frame)

    raw = pd.concat(result_frames, ignore_index=True)
    atomic_csv(raw, output / "raw_results.csv")
    config = {
        "problem": args.problem,
        "protocol": args.protocol,
        "evaluation_seed_range": [seeds[0], seeds[-1]],
        "runs": args.runs,
        "models": [label for label, _ in args.model],
        "continuous_residual_scale": (
            None if args.direct_continuous_control else args.continuous_scale
        ),
        "integer_parameters_frozen": bool(
            args.freeze_integer or args.continuous_only_27d
        ),
        "continuous_only_27d": args.continuous_only_27d,
        "direct_continuous_control": args.direct_continuous_control,
        "l2_projection": (
            None if args.direct_continuous_control else not args.no_l2_projection
        ),
        "continuous_budget": (
            None if args.no_l2_projection else args.continuous_budget
        ),
        "action_dim": 27 if args.continuous_only_27d else 30,
        "paired_design": "All actors and Static Prior use identical held-out seeds.",
    }
    (output / "evaluation_manifest.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / ".complete").write_text("complete\n", encoding="utf-8")
    print(f"[{args.problem}] evaluation complete: {output}", flush=True)


if __name__ == "__main__":
    main()
