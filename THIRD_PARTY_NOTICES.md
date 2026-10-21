# Third-party notices

The root [LICENSE](LICENSE) applies only to contributions owned by thinker-zzp.
It does not relicense the material described here. [LICENSING.md](LICENSING.md)
records component and mixed-file boundaries; this is not an all-MIT repository.

## NeuroEA_RL

- Upstream: https://github.com/liw770075-lgtm/NeuroEA_RL
- Upstream license: MIT; Copyright (c) 2026 liw770075-lgtm.
- MIT text rechecked against upstream `main/LICENSE` on 2026-10-21; the existing local copyright and license text are retained.
- License text: `licenses/NeuroEA_RL-MIT.txt`.
- Upstream default-branch revision observed on 2026-10-18: `5ae62255c7f66d214bf3da3fb8ffe5e493ab0f45`. This identifies the upstream inspected for attribution, not an exact source revision for every local file.

The project author confirms substantial reuse of this earlier NeuroEA project, including benchmark problem definitions and multiple foundation/operator implementations. NeuroEA_RL is the direct source of inherited code; its role is not limited to an external runtime dependency. Local copies include modifications and extensions.

`isac/NeuroEA_GEA/` contains the local NumPy foundation and modified problem definitions; `isac/sac30_dynamic/` integrates the static-to-dynamic control implementation. These directory descriptions indicate functionality, not exclusive authorship; reuse and local modifications may cross these boundaries. The inherited foundation, operators, and problem definitions are attributed to their upstream sources. The exact historical upstream revision and a complete file-by-file modification map have not been established. Preserve the upstream MIT notice when redistributing substantial portions of that code.

作者确认：问题定义及多处基础代码来自 NeuroEA_RL，并在本项目中继续修改、扩展。论文方法的贡献与基础实现的代码来源应分别说明；不得将继承的基础框架、算子或基准问题定义作为本项目原创贡献。

## PlatEMO

- Upstream: https://github.com/BIMK/PlatEMO
- Copyright: BIMK Group; current local upstream source headers state Copyright (c) 2026 BIMK Group.
- Research-use and publication acknowledgment/citation statement rechecked against upstream `master/README.md` on 2026-10-21.
- Upstream revision inspected on 2026-10-21: `d25e65d1ffba58dbf4d7e1b5259786187d12968a` (notice verification, not an exact source revision for every local adaptation).
- Copyright and citation notice: `licenses/PlatEMO-NOTICE.txt`, transcribed from the upstream README checked on 2026-10-18.
- Additional PlatEMO-related adaptations are identified by the current BBOB F1–F10 definitions, `_platemo.py`, and fixed rotation fixture under `isac/NeuroEA_GEA/Problems/Single_objective_optimization/BBOB/`. This is additional benchmark provenance; it does not mean every problem definition was copied directly from PlatEMO or supersede NeuroEA_RL attribution for inherited files.

These local adaptations include local protocol transformations. They do not establish official COCO/BBOB compliance. PlatEMO's upstream terms are separate from NeuroEA_RL's MIT license; the MIT license is not asserted to cover the PlatEMO adaptations.

The source audit also identified explicit PlatEMO provenance in
`Simple_SOPs/SOP_F1.py`, inherited BBOB transformations in
`isac/sac30_dynamic/core.py`, and a PlatEMO sorting-behavior reference in
`isac/NeuroEA_GEA/utils/nd_sort.py`. The latter reference alone does not establish
exact derivation; its provenance remains unresolved. The root MIT grant does
not cover any PlatEMO-derived portions of these files. See LICENSING.md for the
specific mixed-file boundaries and unresolved-provenance treatment.

Upstream reference: Ye Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin. “PlatEMO: A MATLAB Platform for Evolutionary Multi-Objective Optimization [Educational Forum].” IEEE Computational Intelligence Magazine, 12(4):73–87, 2017.

## Distribution scope

Complete upstream repositories, the LDE baseline, trained weights, checkpoints, raw results, paper submission files, and historical orchestration scripts are excluded. Python dependencies are installed separately and retain their respective licenses. This repository does not assign a single blanket license to its mixed-source contents.
