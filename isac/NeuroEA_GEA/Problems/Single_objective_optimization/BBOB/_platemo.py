# PlatEMO-derived benchmark material; Copyright (c) 2026 BIMK Group.
# Research-use and publication citation terms: see licenses/PlatEMO-NOTICE.txt.
# Excluded from the root MIT grant. Preserve inherited NeuroEA_RL notices too.
# See LICENSING.md for scope.

"""Shared transformations for the PlatEMO 2026 BBOB F1-F10 port."""

from pathlib import Path
import warnings

import numpy as np
from scipy.linalg import orth


_FIXTURE_PATH = Path(__file__).with_name("platemo_bbob_fixtures.npz")
_WARNED_FALLBACK = False


def dimension_ratio(dimension: int) -> np.ndarray:
    if dimension <= 1:
        return np.zeros(dimension, dtype=float)
    return np.arange(dimension, dtype=float) / (dimension - 1)


def expand_xopt(value, dimension: int) -> np.ndarray:
    if np.isscalar(value):
        return np.full(dimension, float(value))
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size != dimension:
        raise ValueError(
            f"xopt has {result.size} elements, expected {dimension}."
        )
    return result


def tosz(x) -> np.ndarray:
    """PlatEMO's oscillatory transformation Tosz."""
    x = np.asarray(x, dtype=float)
    xh = np.zeros_like(x)
    nonzero = x != 0
    xh[nonzero] = np.log(np.abs(x[nonzero]))
    c1 = np.where(x > 0, 10.0, 5.5)
    c2 = np.where(x > 0, 7.9, 3.1)
    return np.sign(x) * np.exp(
        xh + 0.049 * (np.sin(c1 * xh) + np.sin(c2 * xh))
    )


def tasy(beta: float, x) -> np.ndarray:
    """PlatEMO's asymmetric transform; negative entries stay unchanged."""
    x = np.asarray(x, dtype=float)
    result = x.copy()
    if x.ndim != 2:
        raise ValueError("Tasy expects a two-dimensional population matrix.")
    ratio = dimension_ratio(x.shape[1])
    positive = x > 0
    transformed = np.zeros_like(x)
    transformed[positive] = np.power(
        x[positive],
        np.broadcast_to(
            1.0 + beta * ratio * np.sqrt(np.maximum(x, 0.0)),
            x.shape,
        )[positive],
    )
    result[positive] = transformed[positive]
    return result


def fpen(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.sum(np.maximum(0.0, np.abs(x) - 5.0) ** 2, axis=1)


def _fixture(name: str, dimension: int):
    if not _FIXTURE_PATH.exists():
        return None
    with np.load(_FIXTURE_PATH) as data:
        key = f"{name}_{dimension}"
        return data[key].copy() if key in data else None


def platemo_rotation(seed: int, dimension: int) -> np.ndarray:
    """Return PlatEMO's fixed rotation, or a deterministic fallback.

    The fallback has the same mathematical role but is not elementwise equal
    to MATLAB because NumPy and MATLAB use different random streams.
    """
    name = "Q" if seed == 1 else "R"
    saved = _fixture(name, dimension)
    if saved is not None:
        return saved

    global _WARNED_FALLBACK
    if not _WARNED_FALLBACK:
        warnings.warn(
            "Official MATLAB BBOB rotation fixture is unavailable; using a "
            "deterministic orthogonal fallback. Run tools/export_platemo_"
            "bbob_fixtures.m and tools/convert_platemo_bbob_fixtures.py "
            "before exact MATLAB/Python parity experiments.",
            RuntimeWarning,
            stacklevel=2,
        )
        _WARNED_FALLBACK = True
    return orth(np.random.RandomState(seed).rand(dimension, dimension))


def platemo_f5_xopt(dimension: int) -> np.ndarray:
    saved = _fixture("F5_xopt", dimension)
    if saved is not None:
        return saved.reshape(-1)
    return 5.0 * np.sign(np.random.RandomState(1).rand(dimension) - 0.5)
