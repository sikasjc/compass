from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from compass.analytics.forecast_quality import calculate_forecast_quality
from compass.backtest.engine import ForecastEvaluation, ForecastTrace
from compass.domain.market import InstrumentId


ETF = InstrumentId.parse("SSE.510300")


def _trace(day: int, expected: float, weight: str) -> ForecastTrace:
    return ForecastTrace(
        decision_date=date(2026, 7, day),
        strategy_id="kronos-main",
        instrument=ETF,
        action="BUY" if Decimal(weight) > 0 else "CASH",
        expected_return=expected,
        path_positive_ratio=0.75,
        rank=1,
        close=10.0,
        trend_value=9.5,
        trend_passed=True,
        target_weight=Decimal(weight),
        reason_code=(
            "KRONOS_FORECAST_ENTRY"
            if Decimal(weight) > 0
            else "KRONOS_FORECAST_CASH"
        ),
        horizon=2,
    )


def _evaluation(day: int, realized: float, tradable: float) -> ForecastEvaluation:
    return ForecastEvaluation(
        decision_date=date(2026, 7, day),
        strategy_id="kronos-main",
        instrument=ETF,
        horizon=2,
        execution_date=date(2026, 7, day + 1),
        evaluation_date=date(2026, 7, day + 2),
        realized_close_return=realized,
        tradable_return=tradable,
    )


def test_forecast_quality_separates_close_and_tradable_returns() -> None:
    traces = (
        _trace(1, 0.10, "0.5"),
        _trace(4, -0.05, "0"),
        _trace(7, 0.03, "0.5"),
    )
    evaluations = (
        _evaluation(1, 0.08, 0.04),
        _evaluation(4, -0.02, -0.03),
    )

    metrics = calculate_forecast_quality(traces, evaluations)

    assert metrics.forecast_count == 3
    assert metrics.evaluated_count == 2
    assert metrics.selected_count == 1
    assert metrics.direction_accuracy == 1.0
    assert metrics.return_correlation == pytest.approx(1.0)
    assert metrics.mean_predicted_return == pytest.approx(0.025)
    assert metrics.mean_realized_close_return == pytest.approx(0.03)
    assert metrics.mean_tradable_return == pytest.approx(0.005)
    assert metrics.selected_mean_tradable_return == pytest.approx(0.04)


def test_forecast_quality_returns_none_for_unidentifiable_statistics() -> None:
    metrics = calculate_forecast_quality((_trace(1, 0.1, "0"),), ())

    assert metrics.forecast_count == 1
    assert metrics.evaluated_count == 0
    assert metrics.direction_accuracy is None
    assert metrics.return_correlation is None
    assert metrics.mean_tradable_return is None
    assert metrics.selected_mean_tradable_return is None
