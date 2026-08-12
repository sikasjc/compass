from __future__ import annotations

from datetime import date
from decimal import Decimal
from math import isfinite
from numbers import Real

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


WEIGHT_TOLERANCE = 1e-12


def _validate_index(index: pd.Index) -> None:
    if not isinstance(index, pd.Index):
        raise TypeError("sleeve indexes must be pandas indexes")
    if isinstance(index, pd.DatetimeIndex):
        if index.hasnans:
            raise ValueError("sleeve indexes must not contain missing dates")
    elif len(index) == 0 or any(type(value) is not date for value in index):
        raise TypeError("sleeve indexes must contain dates")
    if index.has_duplicates:
        raise ValueError("sleeve indexes must be unique")
    if not index.is_monotonic_increasing:
        raise ValueError("sleeve indexes must be increasing")


def _finite_values(frame: pd.DataFrame, *, label: str) -> np.ndarray:
    rows: list[list[float]] = []
    for row in frame.to_numpy(dtype=object, copy=True):
        converted: list[float] = []
        for value in row:
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{label} must not contain booleans")
            if not isinstance(value, (Real, Decimal)):
                raise TypeError(f"{label} must contain numeric values")
            numeric = float(value)
            if not isfinite(numeric):
                raise ValueError(f"{label} must contain finite values")
            converted.append(numeric)
        rows.append(converted)
    if not rows:
        return np.empty((0, len(frame.columns)), dtype=float)
    return np.asarray(rows, dtype=float)


def attribute_sleeves(
    sleeve_returns: pd.DataFrame,
    sleeve_weights: pd.DataFrame,
) -> pd.DataFrame:
    """Return beginning-weight sleeve contributions without implicit alignment.

    Each supplied weight belongs to the return on the same row; no shift is
    introduced. Cash has zero contribution. The caller may compare
    ``total_contribution`` with actual portfolio returns to expose combined-
    broker fee and rounding residuals; this pure function does not fabricate it.
    """

    if not isinstance(sleeve_returns, pd.DataFrame) or not isinstance(
        sleeve_weights, pd.DataFrame
    ):
        raise TypeError("sleeve returns and weights must be DataFrames")
    if sleeve_returns.columns.has_duplicates or sleeve_weights.columns.has_duplicates:
        raise ValueError("sleeve columns must be unique")
    if "total_contribution" in sleeve_returns.columns or "total_contribution" in sleeve_weights.columns:
        raise ValueError("total_contribution is a reserved attribution output column")
    if any(type(column) is not str or not column for column in sleeve_returns.columns):
        raise TypeError("sleeve columns must be non-empty strings")
    if not sleeve_returns.columns.equals(sleeve_weights.columns):
        raise ValueError("sleeve columns must exactly match in the same order")
    _validate_index(sleeve_returns.index)
    _validate_index(sleeve_weights.index)
    if not sleeve_returns.index.equals(sleeve_weights.index):
        raise ValueError("sleeve indexes must exactly match")

    returns = _finite_values(sleeve_returns, label="sleeve returns")
    weights = _finite_values(sleeve_weights, label="sleeve weights")
    if np.any(weights < 0):
        raise ValueError("sleeve weights must be non-negative")
    if len(weights) and np.any(weights.sum(axis=1) > 1.0 + WEIGHT_TOLERANCE):
        raise ValueError("sleeve weights must sum to at most one")

    contributions = returns * weights
    result = pd.DataFrame(
        contributions,
        index=sleeve_returns.index.copy(),
        columns=sleeve_returns.columns.copy(),
        dtype=float,
    )
    result["total_contribution"] = contributions.sum(axis=1)
    return result
