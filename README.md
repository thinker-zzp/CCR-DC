# CCR-DC

Research code for **Static Prior Based Dynamic Configuration with Constrained Continuous Residuals for Trainable Evolutionary Algorithms**.

CCR-DC searches for a static NeuroEA configuration, calibrates parameter sensitivity, and learns a constrained continuous residual controller. The controller observes 44 state features and adjusts 27 continuous parameters; the three integer tournament parameters remain fixed at the static prior. Training uses a paired static counterfactual.

本仓库提供精简核心代码：静态配置搜索、敏感度校准、残差控制器训练及配对评测。模型、训练状态、实验结果、论文文件和历史探索脚本不随仓库发布。

## Install

Tested on Windows with Python 3.9, PyTorch 2.8.0 CPU, and the versions in `requirements.txt`.

```sh
python -m venv .venv
# Activate the environment, then:
python -m pip install -r requirements.txt
cd isac
```

## Small end-to-end example

Run these commands from `isac/`. They generate their own static configuration, radii, and checkpoint. This is a smoke run with two training episodes and two evaluation seeds; its scores are not paper results. CPU execution is supported. To limit CPU use, set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` in your shell.

```sh
python -m sac30_dynamic.senior_gap_entry sac30_dynamic.train_static --problem SOP_F1 --protocol senior --episodes 2 --repeats 1 --validation-runs 1 --validation-every 1 --reference default --seed 20261008 --device cpu --output results/example/static
python -m sac30_dynamic.senior_gap_entry sac30_dynamic.calibrate_radii --problem SOP_F1 --protocol senior --static-parameters results/example/static/best_parameters.npy --seeds 1 --seed 6110800 --output results/example/calibration
python -m sac30_dynamic.senior_gap_entry sac30_dynamic.train_reward_ablation --problem SOP_F1 --protocol senior --static-parameters results/example/static/best_parameters.npy --radii results/example/calibration/radii.npy --continuous-only-27d --continuous-scale 0.75 --horizon 5 --reward-mode counterfactual --actor-episodes 2 --actor-validation-every 1 --actor-validation-runs 1 --seed 410000001 --training-replication 1 --validation-seed-base 250000001 --selection-tail-weight 0.5 --actor-log-std -4 --checkpoint-every 1 --device cpu --output results/example/dynamic
python -m sac30_dynamic.senior_gap_entry sac30_dynamic.evaluate_reward_ablation --problem SOP_F1 --protocol senior --static-parameters results/example/static/best_parameters.npy --radii results/example/calibration/radii.npy --continuous-only-27d --continuous-scale 0.75 --horizon 5 --model Dynamic=results/example/dynamic/selected_actor.pt --seed-base 1 --runs 2 --device cpu --output results/example/evaluation
```

The evaluation writes paired Static Prior / Dynamic measurements to `results/example/evaluation/raw_results.csv`. Generated files remain ignored by Git.

## Code map and settings

| File under `isac/sac30_dynamic/` | Purpose |
| --- | --- |
| `core.py` | NeuroEA session, parameter mapping, and protocol definitions |
| `sac_agent.py` | SAC actor, critic, replay buffer, and checkpoints |
| `train_static.py`, `calibrate_radii.py` | Generate the static prior and sensitivity radii |
| `advanced_env.py`, `scaled_residual_env.py`, `continuous27_env.py` | Counterfactual reward and constrained residual actions |
| `train_reward_ablation.py`, `evaluate_reward_ablation.py` | Train and evaluate the controller |
| `senior_gap_entry.py` | Install the shifted SOP F8 optimum correction before importing a training or evaluation entry |

Benchmark definitions are under `isac/NeuroEA_GEA/Problems/`. The small `.npz` file contains fixed benchmark rotations, not learned weights. Direct-control and trust-region classes are retained because the shared training entry imports them.

The main CCR-DC configuration uses dimension 10, population 100, 10,000 objective evaluations per search, 27 continuous actions, unit-L2 projection, residual scale 0.75, horizon 5, and SAC discount 0.95. Shared entries expose ablation options through `--help`. The local `senior` protocol includes specific coordinate shifts; it is not an official COCO/BBOB evaluation protocol.

For longer runs, increase static and dynamic training budgets to 1,000 episodes, use three static repeats, five calibration seeds, and larger validation and evaluation sets. The local main experiment selected a controller every 100 episodes using ten validation seeds and evaluated on seeds 1–30 and 260000001–260000030. This small example regenerates inputs and does not reproduce the exact published tables: pretrained policies, archived inputs, and raw experiment records are not distributed here.

## Attribution

The NeuroEA foundation includes code adapted from [NeuroEA_RL](https://github.com/liw770075-lgtm/NeuroEA_RL). Benchmark definitions include local Python ports of [PlatEMO](https://github.com/BIMK/PlatEMO). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/` for attribution and upstream terms. No single license is assigned to the entire mixed-source repository.
