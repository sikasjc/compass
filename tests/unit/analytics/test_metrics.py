from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date
from math import sqrt
from types import MappingProxyType

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from compass.analytics.metrics import PerformanceMetrics, calculate_metrics


def test_metrics_match_explicit_hand_calculations_and_are_immutable() -> None:
    index = pd.to_datetime(["2026-01-29", "2026-02-02", "2026-02-03"])
    curve = pd.DataFrame(
        {
            "equity": [100.0, 110.0, 99.0],
            "turnover": [0.0, 0.20, 0.30],
            "costs": [0.0, 1.25, 2.75],
        },
        index=index,
    )
    annual_risk_free = 0.05

    metrics = calculate_metrics(
        curve,
        None,
        periods_per_year=252,
        risk_free_rate=annual_risk_free,
    )

    returns = np.array([0.10, -0.10])
    expected_annualized = 0.99 ** (252 / 2) - 1
    expected_volatility = float(returns.std(ddof=1) * sqrt(252))
    periodic_risk_free = (1 + annual_risk_free) ** (1 / 252) - 1
    expected_sharpe = float((returns.mean() - periodic_risk_free) / returns.std(ddof=1) * sqrt(252))
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.total_return == pytest.approx(-0.01)
    assert metrics.annualized_return == pytest.approx(expected_annualized)
    assert metrics.annualized_volatility == pytest.approx(expected_volatility)
    assert metrics.sharpe_ratio == pytest.approx(expected_sharpe)
    assert metrics.maximum_drawdown == pytest.approx(-0.10)
    assert metrics.calmar_ratio == pytest.approx(expected_annualized / 0.10)
    assert metrics.win_rate == pytest.approx(0.50)
    assert metrics.total_turnover == pytest.approx(0.50)
    assert metrics.total_costs == pytest.approx(4.00)
    assert isinstance(metrics.monthly_returns, MappingProxyType)
    with pytest.raises(FrozenInstanceError):
        metrics.total_return = 1.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        metrics.monthly_returns["2026-01"] = 1.0  # type: ignore[index]


def test_monthly_returns_include_first_partial_month_in_chronological_order() -> None:
    curve = pd.Series(
        [100.0, 105.0, 110.0, 99.0, 108.0],
        index=pd.to_datetime(
            ["2026-01-29", "2026-01-30", "2026-02-02", "2026-02-27", "2026-03-02"]
        ),
        name="equity",
    )

    metrics = calculate_metrics(curve, None)

    assert tuple(metrics.monthly_returns) == ("2026-01", "2026-02", "2026-03")
    assert metrics.monthly_returns["2026-01"] == pytest.approx(0.05)
    assert metrics.monthly_returns["2026-02"] == pytest.approx(99 / 105 - 1)
    assert metrics.monthly_returns["2026-03"] == pytest.approx(108 / 99 - 1)
    assert metrics.total_turnover is None
    assert metrics.total_costs is None


def test_benchmark_uses_exact_dates_and_reports_total_and_annualized_excess() -> None:
    index = pd.Index([date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6)])
    curve = pd.Series([100.0, 110.0, 121.0], index=index)
    benchmark = pd.DataFrame({"benchmark": [100.0, 105.0, 110.0]}, index=index)

    metrics = calculate_metrics(curve, benchmark, periods_per_year=2)

    assert metrics.benchmark_total_return == pytest.approx(0.10)
    assert metrics.benchmark_annualized_return == pytest.approx(0.10)
    assert metrics.excess_total_return == pytest.approx(0.11)
    assert metrics.excess_annualized_return == pytest.approx(0.11)


def test_insufficient_or_zero_denominators_return_none_never_nan_or_infinity() -> None:
    one = pd.Series([100.0], index=pd.to_datetime(["2026-01-02"]))
    flat = pd.Series([100.0, 100.0, 100.0], index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]))

    one_metrics = calculate_metrics(one, None)
    flat_metrics = calculate_metrics(flat, None)

    assert one_metrics.annualized_return is None
    assert one_metrics.annualized_volatility is None
    assert one_metrics.sharpe_ratio is None
    assert one_metrics.calmar_ratio is None
    assert one_metrics.win_rate is None
    assert flat_metrics.annualized_volatility == 0.0
    assert flat_metrics.sharpe_ratio is None
    assert flat_metrics.calmar_ratio is None
    assert flat_metrics.win_rate is None
    for metrics in (one_metrics, flat_metrics):
        for value in metrics.numeric_values():
            assert value is None or np.isfinite(value)


@pytest.mark.parametrize(
    ("curve", "message"),
    [
        (pd.Series([], dtype=float, index=pd.DatetimeIndex([])), "at least one"),
        (pd.Series([100.0, 101.0], index=pd.to_datetime(["2026-01-03", "2026-01-02"])), "increasing"),
        (pd.Series([100.0, 101.0], index=pd.to_datetime(["2026-01-02", "2026-01-02"])), "unique"),
        (pd.Series([100.0, 0.0], index=pd.to_datetime(["2026-01-02", "2026-01-03"])), "positive"),
        (pd.Series([100.0, np.inf], index=pd.to_datetime(["2026-01-02", "2026-01-03"])), "finite"),
        (pd.Series([True, False], index=pd.to_datetime(["2026-01-02", "2026-01-03"])), "booleans"),
        (pd.DataFrame({"equity": [100.0], "unknown": [1.0]}, index=pd.to_datetime(["2026-01-02"])), "columns"),
        (pd.DataFrame({"equity": [100.0], "turnover": [-0.1]}, index=pd.to_datetime(["2026-01-02"])), "non-negative"),
    ],
)
def test_equity_curve_rejects_noncanonical_input(curve: pd.Series | pd.DataFrame, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        calculate_metrics(curve, None)


@pytest.mark.parametrize("periods_per_year", [True, 0, -1, 2.5])
def test_periods_per_year_must_be_a_positive_integer(periods_per_year: object) -> None:
    curve = pd.Series([100.0], index=pd.to_datetime(["2026-01-02"]))

    with pytest.raises((TypeError, ValueError), match="periods_per_year"):
        calculate_metrics(curve, None, periods_per_year=periods_per_year)  # type: ignore[arg-type]


@pytest.mark.parametrize("risk_free_rate", [True, np.nan, np.inf, -1.0])
def test_risk_free_rate_must_be_finite_and_greater_than_negative_one(
    risk_free_rate: object,
) -> None:
    curve = pd.Series([100.0, 101.0], index=pd.to_datetime(["2026-01-02", "2026-01-05"]))

    with pytest.raises((TypeError, ValueError), match="risk_free_rate"):
        calculate_metrics(curve, None, risk_free_rate=risk_free_rate)  # type: ignore[arg-type]


def test_benchmark_must_have_one_positive_column_and_exactly_matching_index() -> None:
    index = pd.to_datetime(["2026-01-02", "2026-01-05"])
    curve = pd.Series([100.0, 101.0], index=index)

    with pytest.raises(ValueError, match="exactly match"):
        calculate_metrics(
            curve,
            pd.Series([100.0, 101.0], index=pd.to_datetime(["2026-01-02", "2026-01-06"])),
        )
    with pytest.raises(ValueError, match="one column"):
        calculate_metrics(curve, pd.DataFrame({"a": [1.0, 1.0], "b": [1.0, 1.0]}, index=index))
    with pytest.raises(ValueError, match="positive"):
        calculate_metrics(curve, pd.Series([100.0, -1.0], index=index))


def test_calculation_does_not_mutate_inputs() -> None:
    index = pd.to_datetime(["2026-01-02", "2026-01-05"])
    curve = pd.DataFrame({"equity": [100.0, 101.0], "costs": [0.0, 1.0]}, index=index)
    benchmark = pd.Series([100.0, 100.5], index=index)
    curve_before = curve.copy(deep=True)
    benchmark_before = benchmark.copy(deep=True)

    calculate_metrics(curve, benchmark)

    pd.testing.assert_frame_equal(curve, curve_before)
    pd.testing.assert_series_equal(benchmark, benchmark_before)


@pytest.mark.parametrize("column", ["turnover", "costs"])
def test_finite_metric_aggregate_overflow_is_rejected(column: str) -> None:
    curve = pd.DataFrame(
        {"equity": [100.0, 101.0], column: [1e308, 1e308]},
        index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
    )

    with pytest.raises(ValueError, match=rf"{column} aggregate must be finite"):
        calculate_metrics(curve, None)
