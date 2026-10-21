# Licensing and reuse

CCR-DC contains material from several sources. **The repository as a whole is
not licensed under MIT.** The root [LICENSE](LICENSE) grants MIT permission for
the original code, documentation, and modifications owned by thinker-zzp. It
does not change any upstream license or remove third-party obligations.

## Scope by component

| Component | Applicable terms |
| --- | --- |
| Author-owned CCR-DC contributions, including static-prior search, sensitivity calibration, residual control, training/evaluation integration, and documentation | Root MIT grant, only to the extent owned by thinker-zzp; component names do not imply exclusive authorship |
| Code inherited or adapted from NeuroEA_RL, including NeuroEA foundations, operators, and problem definitions | Preserve Copyright (c) 2026 liw770075-lgtm and [NeuroEA_RL MIT text](licenses/NeuroEA_RL-MIT.txt); underlying PlatEMO portions remain subject to their own terms |
| `isac/NeuroEA_GEA/Problems/Single_objective_optimization/BBOB/` | PlatEMO-derived BBOB F1-F10 definitions, `_platemo.py` transformations, and `platemo_bbob_fixtures.npz` fixed rotations are excluded from the root MIT grant; preserve [PlatEMO notice](licenses/PlatEMO-NOTICE.txt) and any NeuroEA_RL attribution |
| `isac/NeuroEA_GEA/Problems/Single_objective_optimization/Simple_SOPs/SOP_F1.py` | Identifies PlatEMO Simple SOP provenance; its PlatEMO-derived portions are excluded from the root MIT grant |
| `isac/sac30_dynamic/core.py` | Mixed file: BBOB transformations in `_senior_tasy`, `_senior_bbob_f3`, `_bbob_tosz`, and BBOB-specific branches of `configure_protocol` retain inherited benchmark provenance and are excluded from the root MIT grant wherever derived from PlatEMO; author-owned controller/session contributions have the root MIT grant |
| `isac/NeuroEA_GEA/utils/nd_sort.py` | Inherited NeuroEA code referring to PlatEMO sorting behavior; the exact derivation is unresolved. Preserve NeuroEA_RL attribution and the PlatEMO notice for any derived portions; no blanket MIT claim is made for this file |
| Other files with mixed or incompletely mapped provenance | The root MIT grant covers only author-owned contributions. Preserve applicable upstream notices; this table does not establish exclusive authorship or clear undocumented third-party rights |
| Separately installed dependencies | Their respective licenses; the root MIT grant does not relicense dependencies |

The exact historical source revision and complete portion-by-portion provenance
map are not established. The exclusions above identify known or flagged areas;
they do not override another copyright holder's rights elsewhere.

## PlatEMO terms

PlatEMO's upstream statement permits research use and requires publications
using the platform or its code to acknowledge PlatEMO and cite its 2017 paper.
The BIMK Group retains copyright. See the preserved statement and reference in
[licenses/PlatEMO-NOTICE.txt](licenses/PlatEMO-NOTICE.txt) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

This is a separate upstream permission, not MIT or a grant of unrestricted
commercial use. The root MIT license does not authorize uses of PlatEMO-derived
portions beyond their upstream permission; obtain permission from the relevant
rights holder for uses not covered by it. Bundling the research workflow does
not make all bundled components permissively licensed.

## Redistributing or extracting code

- Include the root LICENSE when distributing substantial author-owned portions.
- Retain the NeuroEA_RL MIT copyright and permission notice for substantial
  inherited portions, including when those portions were locally modified.
- Retain the PlatEMO copyright, research-use statement, and publication
  acknowledgment/citation requirements for PlatEMO-derived portions and data.
- When extracting a mixed file, carry the relevant notices with the extracted
  code. Do not label the whole file or repository MIT merely because a root
  LICENSE or GitHub license badge is present.

作者拥有版权的原创代码、文档和修改采用 MIT；NeuroEA_RL 的 MIT 版权声明继续保留。
PlatEMO 衍生代码、变换及固定数据保留其研究用途许可和论文引用要求，不改标为 MIT。
完整训练评测流程包含这些第三方部分，因此不能把整个仓库描述为全部 MIT。
