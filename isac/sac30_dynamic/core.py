from __future__ import annotations

# Mixed-source file: author-owned contributions use the root MIT grant.
# Inherited NeuroEA_RL portions retain licenses/NeuroEA_RL-MIT.txt.
# PlatEMO-derived BBOB transformations/branches retain the separate research-use
# and citation terms in licenses/PlatEMO-NOTICE.txt; see LICENSING.md.

from dataclasses import dataclass
from types import MethodType

import numpy as np

from NeuroEA_GEA.Problems import get_problem_instance
from NeuroEA_GEA.Algorithms.NEUROEA.Block_Population import Block_Population
from NeuroEA_GEA.Algorithms.NEUROEA.Block_Tournament import Block_Tournament
from NeuroEA_GEA.Algorithms.NEUROEA.Block_Exchange import Block_Exchange
from NeuroEA_GEA.Algorithms.NEUROEA.Block_Crossover import Block_Crossover
from NeuroEA_GEA.Algorithms.NEUROEA.Block_Mutation import Block_Mutation
from NeuroEA_GEA.Algorithms.NEUROEA.Block_Selection import Block_Selection


LOWER = np.array([
    1, 1, 1, *([1e-20] * 12), 0, -1, 1e-20, 0, -1, 1e-20, 0, -1,
    1e-20, 0, 1e-20, 0, 1e-20, 0, 1e-20,
], dtype=np.float64)
UPPER = np.array([
    10, 10, 10, *([1.0] * 12), 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 2, 1, 2, 1, 2,
], dtype=np.float64)

# Table-III/F1-specialized configuration found in the MATLAB handoff code.
F1_REFERENCE = np.array([
    7.64748297733589, 9.70611837719899, 7.68069088069810,
    0.859708326492895, 0.731510675684283, 0.467425409016248,
    0.809344515651542, 0.969153985848438, 0.563708490176869,
    0.617793881964623, 0.973105345617609, 0.802047886084958,
    0.376906010050315, 0.989194244018413, 0.747613416288249,
    0.450274080464739, 0.497095571494421, 0.0168993296931074,
    0.499899997306222, -0.547031533890586, 0.190757710176318,
    0.268572786910929, 0.731834113289507, 0.600929098112029,
    0.00124293085727638, 0.00149715086076880, 0.302417191118496,
    0.0234774878134407, 0.0, 0.130821978145734,
], dtype=np.float64)

# Final 30-D NeuroEA configuration used by isac/evaluate_final.py.
DEFAULT_REFERENCE = np.array([
    7.328845916458512, 9.519452716194290, 6.699378970058081,
    0.504833860696245, 0.871154112250468, 0.451289503013029,
    0.277377781142785, 0.999699252289889, 0.777907300954096,
    0.0404806672600856, 0.368156506713865, 0.728384642089435,
    0.257697135783844, 0.854586114401756, 0.404668247736162,
    0.513535549249738, -0.316033255651616, 0.158972592688388,
    0.304425102233896, 0.854778369939324, 0.304690855887451,
    0.245240419885119, 0.190764560248536, 0.0911426508854076,
    0.0884953899671375, 0.0363972714169237, 0.0,
    0.200203031642368, 0.0327544992994410, 0.00772647381847839,
], dtype=np.float32)

TINY = 1e-300


def static_search_reference(problem_name: str) -> np.ndarray:
    """Return the documented starting configuration for static SAC-30."""
    return (F1_REFERENCE if problem_name.upper() == "SOP_F1"
            else DEFAULT_REFERENCE).astype(np.float64, copy=True)


def parameters_to_normalized(parameters: np.ndarray) -> np.ndarray:
    return np.clip(2.0 * (np.asarray(parameters) - LOWER) / (UPPER - LOWER) - 1.0, -1.0, 1.0)


def normalized_to_parameters(values: np.ndarray) -> np.ndarray:
    parameters = LOWER + 0.5 * (np.clip(values, -1.0, 1.0) + 1.0) * (UPPER - LOWER)
    parameters[:3] = np.rint(parameters[:3])
    return np.clip(parameters, LOWER, UPPER)


def objective_optimum(problem_name: str, dimension: int) -> float:
    """Return the known global optimum used to form a non-negative error."""
    return -418.9829 * int(dimension) if problem_name.upper() == "SOP_F8" else 0.0


def objective_gap(problem_name: str, values, dimension: int):
    """Distance to the known optimum, safe for ratios and logarithms.

    NeuroEA still minimizes the original objective. This transformation is
    only used by RL rewards, state features, and relative-performance tests.
    """
    gap = np.asarray(values, dtype=np.float64) - objective_optimum(problem_name, dimension)
    return np.maximum(gap, TINY)


def _base_objective(problem_name: str, values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if problem_name == "SOP_F1":
        y = np.sum(x ** 2, axis=1)
    elif problem_name == "SOP_F2":
        absolute = np.abs(x)
        y = np.sum(absolute, axis=1) + np.prod(absolute, axis=1)
    elif problem_name == "SOP_F3":
        y = np.sum(np.cumsum(x, axis=1) ** 2, axis=1)
    elif problem_name == "SOP_F4":
        y = np.max(np.abs(x), axis=1)
    elif problem_name == "SOP_F5":
        y = np.sum(
            100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2
            + (x[:, :-1] - 1.0) ** 2,
            axis=1,
        )
    elif problem_name == "SOP_F6":
        y = np.sum(np.floor(x + 0.5) ** 2, axis=1)
    elif problem_name == "SOP_F7":
        weights = np.arange(1, x.shape[1] + 1, dtype=np.float64)
        y = np.sum(weights * x ** 4, axis=1) + np.random.rand(x.shape[0])
    elif problem_name == "SOP_F8":
        y = -np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)
    elif problem_name == "SOP_F9":
        y = np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x) + 10.0, axis=1)
    elif problem_name == "SOP_F10":
        term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.mean(x ** 2, axis=1)))
        term2 = -np.exp(np.mean(np.cos(2.0 * np.pi * x), axis=1))
        y = term1 + term2 + 20.0 + np.e
    else:
        raise ValueError(f"Protocol override is not defined for {problem_name}.")
    return y[:, None]


def _senior_offset(instance) -> np.ndarray:
    """Affine coordinate offset used by the earlier Torch NeuroEA port."""
    return np.linspace(0.0, float(instance.upper[0]), instance.D)


def _senior_orthogonal(dimension: int, seed: int) -> np.ndarray:
    """Reproduce the senior Torch port's seeded QR construction on CPU."""
    base = np.random.RandomState(seed).rand(dimension, dimension)
    return np.linalg.qr(base)[0]


def _senior_tasy(beta: float, values: np.ndarray) -> np.ndarray:
    """Torch-port Tasy: transform both signs using sign(abs(x)^exponent)."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape[1] == 1:
        return values
    ratio = np.arange(values.shape[1], dtype=np.float64) / (values.shape[1] - 1)
    exponent = 1.0 + beta * ratio.reshape(1, -1) * np.sqrt(np.abs(values))
    return np.sign(values) * (np.abs(values) ** exponent)


def _senior_bbob_f3(instance, decisions: np.ndarray) -> np.ndarray:
    x = np.asarray(decisions, dtype=np.float64)
    d = instance.D
    offset = (np.asarray(instance.upper) - np.asarray(instance.lower)) / (2.0 * d)
    xopt = np.asarray(instance.x_opt, dtype=np.float64)
    z = _senior_tasy(0.2, _bbob_tosz(x + offset - xopt))
    z *= 10.0 ** (0.5 * np.arange(d, dtype=np.float64) / (d - 1))
    value = 10.0 * (d - np.sum(np.cos(2.0 * np.pi * z), axis=1))
    value += np.sum(z ** 2, axis=1)
    return value.reshape(-1, 1)


def _bbob_tosz(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    xh = np.zeros_like(values)
    nonzero = values != 0
    xh[nonzero] = np.log(np.abs(values[nonzero]))
    c1 = np.where(values > 0, 10.0, 5.5)
    c2 = np.where(values > 0, 7.9, 3.1)
    return np.sign(values) * np.exp(
        xh + 0.049 * (np.sin(c1 * xh) + np.sin(c2 * xh))
    )


def configure_protocol(problem, problem_name: str, protocol: str) -> None:
    """Select the exact decision-space translation without editing problem files."""
    if protocol in {"isac", "platemo"}:
        return
    if protocol == "senior":
        original = problem.cal_obj
        name = problem_name.upper()

        # The senior Torch port builds these rotations from RandomState(seed)
        # followed by QR, rather than using the latest PlatEMO fixtures.
        if name in {"BBOB_F6", "BBOB_F7"}:
            problem.Q = _senior_orthogonal(problem.D, 1)
            problem.R = _senior_orthogonal(problem.D, 2)
        elif name in {"BBOB_F9", "BBOB_F10"}:
            problem.R = _senior_orthogonal(problem.D, 2)
        elif name == "BBOB_F5":
            problem.x_opt = 5.0 * np.sign(
                np.random.RandomState(1).rand(problem.D) - 0.5
            )

        def senior_objective(instance, decisions):
            x = np.asarray(decisions, dtype=np.float64)
            if name.startswith("SOP_F"):
                # All ten senior SOP definitions use this affine shift,
                # including the Ackley-based SOP_F10.
                return _base_objective(name, x - _senior_offset(instance))
            if name == "BBOB_F3":
                return _senior_bbob_f3(instance, x)
            if name == "BBOB_F4" or name == "BBOB_F7":
                # The senior port uses c=(u-l)/(2D) for these two functions.
                c = (np.asarray(instance.upper) - np.asarray(instance.lower)) / (2.0 * instance.D)
                return original(x + c.reshape(1, -1))
            if name in {"BBOB_F1", "BBOB_F2", "BBOB_F6", "BBOB_F8", "BBOB_F9", "BBOB_F10"}:
                return original(x - _senior_offset(instance))
            if name == "BBOB_F5":
                return original(x)
            raise ValueError(f"Senior protocol is not defined for {problem_name}.")

        problem.cal_obj = MethodType(senior_objective, problem)
        return
    if protocol not in {"paper", "matlab"}:
        raise ValueError(
            "protocol must be one of: platemo, senior, isac, paper, matlab"
        )

    def objective(instance, decisions):
        x = np.asarray(decisions, dtype=np.float64)
        if protocol == "paper":
            index = np.arange(1, instance.D + 1, dtype=np.float64)
            shift = index * (instance.upper - instance.lower) / (2.0 * instance.D)
            x = x - shift
        return _base_objective(problem_name, x)

    problem.cal_obj = MethodType(objective, problem)


def build_blocks(parameters: np.ndarray | None = None):
    blocks = [
        Block_Population(),
        Block_Tournament(n_parents=200, upper_k=10),
        Block_Tournament(200, 10),
        Block_Tournament(200, 10),
        Block_Exchange(n_parents=3),
        Block_Exchange(3), Block_Exchange(3), Block_Exchange(3),
        Block_Crossover(n_parents=2, n_sets=3),
        Block_Mutation(n_sets=3),
        Block_Selection(n_solutions=100),
    ]
    graph = np.zeros((11, 11), dtype=np.float64)
    graph[0, [1, 2, 3, 10]] = 1.0
    graph[1:4, 4:8] = 0.25
    graph[4:8, 8] = 1.0
    graph[8, 9] = graph[9, 10] = graph[10, 0] = 1.0
    if parameters is not None:
        set_parameters(blocks, parameters)
    return blocks, graph


def set_parameters(blocks, parameters: np.ndarray) -> None:
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (30,):
        raise ValueError(f"Expected 30 parameters, got {values.shape}.")
    cursor = 0
    for block in blocks:
        count = len(block.parameter)
        if count:
            block.set_parameters(values[cursor:cursor + count])
            cursor += count
    if cursor != 30:
        raise RuntimeError(f"Block graph consumed {cursor} parameters instead of 30.")


@dataclass
class StepInfo:
    fe: int
    best: float
    diversity: float


class NeuroEASession:
    """Generation-steppable implementation of the original 11-block NeuroEA."""

    def __init__(self, problem_name="SOP_F1", protocol="paper", dimension=10,
                 population_size=100, max_fe=10000, seed=1, parameters=F1_REFERENCE):
        np.random.seed(int(seed))
        self.problem_name = problem_name
        self.problem = get_problem_instance(problem_name, population_size, 1, dimension, max_fe)
        configure_protocol(self.problem, problem_name, protocol)
        # Build before initialization to preserve the RNG order of evaluate_final.py.
        self.blocks, self.graph = build_blocks(parameters)
        self.is_population = np.array([isinstance(b, Block_Population) for b in self.blocks])
        initial = self.problem.initialization()
        for index in np.where(self.is_population)[0]:
            self.blocks[index].initialization(initial)

    @property
    def done(self) -> bool:
        return self.problem.fe >= self.problem.max_fe

    def best(self) -> float:
        return float(min(float(np.asarray(item.obj).reshape(-1)[0]) for item in self.blocks[0].output))

    def diversity(self) -> float:
        decisions = np.asarray([item.dec for item in self.blocks[0].output], dtype=np.float64)
        scale = np.maximum(self.problem.upper - self.problem.lower, 1e-12)
        return float(np.mean(np.std(decisions, axis=0) / scale))

    def objective_values(self) -> np.ndarray:
        return np.asarray(
            [float(np.asarray(item.obj).reshape(-1)[0]) for item in self.blocks[0].output],
            dtype=np.float64,
        )

    def population_statistics(self) -> dict[str, float]:
        values = self.objective_values()
        q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
        return {
            "best": float(values.min()), "q25": float(q25),
            "median": float(median), "q75": float(q75),
            "diversity": self.diversity(),
        }

    def info(self) -> StepInfo:
        return StepInfo(self.problem.fe, self.best(), self.diversity())

    def step(self, parameters: np.ndarray) -> StepInfo:
        if self.done:
            raise RuntimeError("The NeuroEA session has terminated.")
        set_parameters(self.blocks, parameters)
        activated = np.zeros(len(self.blocks), dtype=bool)
        while not np.all(activated[self.is_population]):
            progressed = False
            for index in range(len(self.blocks)):
                if activated[index]:
                    continue
                predecessors = self.graph[:, index] > 0
                if np.all(activated[predecessors] | self.is_population[predecessors]):
                    previous = [self.blocks[j] for j in range(len(self.blocks)) if predecessors[j]]
                    self.blocks[index].main(self.problem, previous, self.graph[predecessors, index])
                    activated[index] = True
                    progressed = True
            if not progressed:
                raise RuntimeError("The NeuroEA graph cannot be scheduled.")
        return self.info()


def run_fixed(parameters: np.ndarray, seed=1, problem_name="SOP_F1", protocol="paper",
              dimension=10, population_size=100, max_fe=10000) -> float:
    session = NeuroEASession(problem_name, protocol, dimension, population_size, max_fe, seed, parameters)
    while not session.done:
        session.step(parameters)
    return session.best()
