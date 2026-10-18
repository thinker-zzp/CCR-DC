# Third-party notices

## NeuroEA_RL

- Upstream: https://github.com/liw770075-lgtm/NeuroEA_RL
- Upstream license: MIT; Copyright (c) 2026 liw770075-lgtm.
- License text: `licenses/NeuroEA_RL-MIT.txt`.
- Upstream default-branch revision observed on 2026-10-18: `5ae62255c7f66d214bf3da3fb8ffe5e493ab0f45`. This identifies the upstream inspected for attribution, not an exact source revision for every local file.

The project author confirms reuse of this earlier NeuroEA project. `isac/NeuroEA_GEA/` contains the local NumPy foundation and modified problem definitions; `isac/sac30_dynamic/` contains the local static-to-dynamic control implementation. The inherited foundation, operators, and problem definitions are not claimed as original CCR-DC contributions. Preserve the upstream MIT notice when redistributing substantial portions of that code.

## PlatEMO

- Upstream: https://github.com/BIMK/PlatEMO
- Copyright and citation notice: `licenses/PlatEMO-NOTICE.txt`, transcribed from the upstream README checked on 2026-10-18.
- Local adaptations include the BBOB F1–F10 Python definitions, `_platemo.py`, the fixed rotation fixture, and SOP benchmark definitions under `isac/NeuroEA_GEA/Problems/Single_objective_optimization/`.

These local adaptations include local protocol transformations. They do not establish official COCO/BBOB compliance. PlatEMO's upstream terms are separate from NeuroEA_RL's MIT license; the MIT license is not asserted to cover the PlatEMO adaptations.

Upstream reference: Ye Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin. “PlatEMO: A MATLAB Platform for Evolutionary Multi-Objective Optimization [Educational Forum].” IEEE Computational Intelligence Magazine, 12(4):73–87, 2017.

## Distribution scope

Complete upstream repositories, the LDE baseline, trained weights, checkpoints, raw results, paper submission files, and historical orchestration scripts are excluded. Python dependencies are installed separately and retain their respective licenses. This repository does not assign a single blanket license to its mixed-source contents.
