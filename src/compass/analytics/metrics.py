from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from math import fsum, isfinite, sqrt
from numbers import Real
from types import MappingProxyType

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


MetricValue = float | None


def _finite_or_none(value: float) -> MetricValue:
    return value if isfinite(value) else None


def _validate_index(index: pd.Index, *, label: str, allow_empty: bool = False) -> None:
    if not isinstance(index, pd.Index):
        raise TypeError(f"{label} index must be a DatetimeIndex or an index of exact dates")
    if not allow_empty and len(index) == 0:
        raise ValueError(f"{label} must contain at least one observation")
    if isinstance(index, pd.DatetimeIndex):
        if index.hasnans:
            raise ValueError(f"{label} dates must not contain missing values")
    elif any(type(value) is not date for value in index):
        raise TypeError(f"{label} index must be a DatetimeIndex or an index of exact dates")
    if index.has_duplicates:
        raise ValueError(f"{label} dates must be unique")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{label} dates must be increasing")


def _to_finite_array(series: pd.Series, *, label: str, positive: bool = False) -> np.ndarray:
    values: list[float] = []
    for value in series.to_numpy(dtype=object, copy=True):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{label} must not contain booleans")
        if not isinstance(value, (Real, Decimal)):
            raise TypeError(f"{label} must contain numeric values")
        numeric = float(value)
        if not isfinite(numeric):
            raise ValueError(f"{label} must contain finite values")
        if positive and numeric <= 0:
            raise ValueError(f"{label} values must be positive")
        values.append(numeric)
    return np.asarray(values, dtype=float)


def _equity_input(
    equity_curve: pd.Series | pd.DataFrame,
) -> tuple[pd.Index, np.ndarray, MetricValue, MetricValue]:
    if isinstance(equity_curve, pd.Series):
        frame = equity_curve.copy(deep=True)
        _validate_index(frame.index, label="equity curve")
        return frame.index.copy(), _to_finite_array(frame, label="equity", positive=True), None, None
    if not isinstance(equity_curve, pd.DataFrame):
        raise TypeError("equity_curve must be a pandas Series or DataFrame")
    frame = equity_curve.copy(deep=True)
    _validate_index(frame.index, label="equity curve")
    if frame.columns.has_duplicates:
        raise ValueError("equity curve columns must be unique")
    allowed = {"equity", "turnover", "costs"}
    if "equity" not in frame.columns or not set(frame.columns).issubset(allowed):
        raise ValueError("equity curve columns must contain equity and only optional turnover/costs")
    if any(type(column) is not str for column in frame.columns):
        raise TypeError("equity curve columns must be strings")
    equity = _to_finite_array(frame["equity"], label="equity", positive=True)
    aggregates: dict[str, MetricValue] = {"turnover": None, "costs": None}
    for column in ("turnover", "costs"):
        if column not in frame:
            continue
        values = _to_finite_array(frame[column], label=column)
        if np.any(values < 0):
            raise ValueError(f"{column} values must be non-negative")
        try:
            aggregate = fsum(values)
        except OverflowError:
            raise ValueError(f"{column} aggregate must be finite") from None
        if not isfinite(aggregate):
            raise ValueError(f"{column} aggregate must be finite")
        aggregates[column] = aggregate
    return frame.index.copy(), equity, aggregates["turnover"], aggregates["costs"]


def _benchmark_input(
    benchmark: pd.Series | pd.DataFrame | None,
    expected_index: pd.Index,
) -> np.ndarray | None:
    if benchmark is None:
        return None
    if isinstance(benchmark, pd.Series):
        series = benchmark.copy(deep=True)
    elif isinstance(benchmark, pd.DataFrame):
        if len(benchmark.columns) != 1:
            raise ValueError("benchmark DataFrame must contain exactly one column")
        series = benchmark.iloc[:, 0].copy(deep=True)
    else:
        raise TypeError("benchmark must be a pandas Series, one-column DataFrame, or None")
    _validate_index(series.index, label="benchmark")
    if not series.index.equals(expected_index):
        raise ValueError("benchmark dates must exactly match equity curve dates")
    return _to_finite_array(series, label="benchmark", positive=True)


def _annualized_return(levels: np.ndarray, periods_per_year: int) -> MetricValue:
    periods = len(levels) - 1
    if periods < 1:
        return None
    try:
        value = float((levels[-1] / levels[0]) ** (periods_per_year / periods) - 1.0)
    except (OverflowError, ZeroDivisionError):
        return None
    return _finite_or_none(value)


def _total_return(levels: np.ndarray) -> MetricValue:
    try:
        value = float(levels[-1] / levels[0] - 1.0)
    except (OverflowError, ZeroDivisionError):
        return None
    return _finite_or_none(value)


def _month_key(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        return f"{value.year:04d}-{value.month:02d}"
    if type(value) is date:
        assert isinstance(value, date)
        return f"{value.year:04d}-{value.month:02d}"
    raise TypeError("equity curve index must contain dates")


def _monthly_returns(index: pd.Index, levels: np.ndarray) -> Mapping[str, MetricValue]:
    month_ends: list[tuple[str, float]] = []
    for position, index_value in enumerate(index):
        key = _month_key(index_value)
        value = float(levels[position])
        if month_ends and month_ends[-1][0] == key:
            month_ends[-1] = (key, value)
        else:
            month_ends.append((key, value))
    result: dict[str, MetricValue] = {}
    previous = float(levels[0])
    for key, month_end in month_ends:
        result[key] = _finite_or_none(float(month_end / previous - 1.0))
        previous = month_end
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Immutable metrics computed from validated period-end equity observations.

    Win rate uses only non-zero periodic returns. Turnover and costs are sums of
    caller-supplied per-period columns; neither is inferred from equity. Monthly
    returns use the first equity observation as the base for the first partial
    month and each prior month end thereafter.
    """

    total_return: MetricValue
    annualized_return: MetricValue
    annualized_volatility: MetricValue
    sharpe_ratio: MetricValue
    maximum_drawdown: MetricValue
    calmar_ratio: MetricValue
    win_rate: MetricValue
    total_turnover: MetricValue
    total_costs: MetricValue
    monthly_returns: Mapping[str, MetricValue]
    benchmark_total_return: MetricValue = None
    benchmark_annualized_return: MetricValue = None
    excess_total_return: MetricValue = None
    excess_annualized_return: MetricValue = None

    def __post_init__(self) -> None:
        monthly = dict(self.monthly_returns)
        if any(type(key) is not str for key in monthly):
            raise TypeError("monthly return keys must be strings")
        for value in (*self.numeric_values(include_monthly=False), *monthly.values()):
            if value is not None and (isinstance(value, bool) or not isfinite(value)):
                raise ValueError("metric values must be finite or None")
        object.__setattr__(self, "monthly_returns", MappingProxyType(monthly))

    def numeric_values(self, *, include_monthly: bool = True) -> tuple[MetricValue, ...]:
        values = (
            self.total_return,
            self.annualized_return,
            self.annualized_volatility,
            self.sharpe_ratio,
            self.maximum_drawdown,
            self.calmar_ratio,
            self.win_rate,
            self.total_turnover,
            self.total_costs,
            self.benchmark_total_return,
            self.benchmark_annualized_return,
            self.excess_total_return,
            self.excess_annualized_return,
        )
        if not include_monthly:
            return values
        return (*values, *self.monthly_returns.values())


def calculate_metrics(
    equity_curve: pd.Series | pd.DataFrame,
    benchmark: pd.Series | pd.DataFrame | None = None,
    periods_per_year: int = 252,
    risk_free_rate: float | Decimal = 0,
) -> PerformanceMetrics:
    """Calculate deterministic metrics without sorting, filling, or implicit alignment."""

    if type(periods_per_year) is not int:
        raise TypeError("periods_per_year must be an exact integer")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    if isinstance(risk_free_rate, bool) or not isinstance(risk_free_rate, (Real, Decimal)):
        raise TypeError("risk_free_rate must be numeric and not boolean")
    annual_risk_free = float(risk_free_rate)
    if not isfinite(annual_risk_free) or annual_risk_free <= -1:
        raise ValueError("risk_free_rate must be finite and greater than -1")

    index, levels, turnover, costs = _equity_input(equity_curve)
    benchmark_levels = _benchmark_input(benchmark, index)
    returns = levels[1:] / levels[:-1] - 1.0
    annualized = _annualized_return(levels, periods_per_year)
    total = _total_return(levels)
    volatility: MetricValue = None
    sharpe: MetricValue = None
    if len(returns) >= 2:
        sample_std = float(returns.std(ddof=1))
        volatility = _finite_or_none(sample_std * sqrt(periods_per_year))
        if sample_std > 0 and isfinite(sample_std):
            periodic_risk_free = (1.0 + annual_risk_free) ** (1.0 / periods_per_year) - 1.0
            sharpe = _finite_or_none(
                float((returns.mean() - periodic_risk_free) / sample_std * sqrt(periods_per_year))
            )

    running_peak = np.maximum.accumulate(levels)
    drawdowns = levels / running_peak - 1.0
    maximum_drawdown = _finite_or_none(float(drawdowns.min()))
    calmar: MetricValue = None
    if annualized is not None and maximum_drawdown not in (None, 0.0):
        calmar = _finite_or_none(annualized / abs(maximum_drawdown))
    non_zero_returns = returns[returns != 0.0]
    win_rate = (
        None
        if len(non_zero_returns) == 0
        else _finite_or_none(float(np.count_nonzero(non_zero_returns > 0) / len(non_zero_returns)))
    )

    benchmark_total = None if benchmark_levels is None else _total_return(benchmark_levels)
    benchmark_annualized = (
        None if benchmark_levels is None else _annualized_return(benchmark_levels, periods_per_year)
    )
    excess_total = (
        None if total is None or benchmark_total is None else _finite_or_none(total - benchmark_total)
    )
    excess_annualized = (
        None
        if annualized is None or benchmark_annualized is None
        else _finite_or_none(annualized - benchmark_annualized)
    )
    return PerformanceMetrics(
        total_return=total,
        annualized_return=annualized,
        annualized_volatility=volatility,
        sharpe_ratio=sharpe,
        maximum_drawdown=maximum_drawdown,
        calmar_ratio=calmar,
        win_rate=win_rate,
        total_turnover=turnover,
        total_costs=costs,
        monthly_returns=_monthly_returns(index, levels),
        benchmark_total_return=benchmark_total,
        benchmark_annualized_return=benchmark_annualized,
        excess_total_return=excess_total,
        excess_annualized_return=excess_annualized,
    )
