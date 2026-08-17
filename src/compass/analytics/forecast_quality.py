from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np

from compass.backtest.engine import ForecastEvaluation, ForecastTrace


MetricValue = float | None


def _metric(value: float | None, *, label: str) -> MetricValue:
    if value is None:
        return None
    if type(value) is not float or not isfinite(value):
        raise ValueError(f"{label} must be a finite exact float or None")
    return value


def _mean(values: Sequence[float]) -> MetricValue:
    return None if not values else float(np.mean(np.asarray(values, dtype=float)))


def _correlation(left: Sequence[float], right: Sequence[float]) -> MetricValue:
    if len(left) < 2 or len(right) != len(left):
        return None
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if float(np.std(left_values)) == 0.0 or float(np.std(right_values)) == 0.0:
        return None
    value = float(np.corrcoef(left_values, right_values)[0, 1])
    return value if isfinite(value) else None


@dataclass(frozen=True, slots=True)
class ForecastQualityMetrics:
    forecast_count: int
    evaluated_count: int
    selected_count: int
    direction_accuracy: MetricValue
    return_correlation: MetricValue
    mean_predicted_return: MetricValue
    mean_realized_close_return: MetricValue
    mean_tradable_return: MetricValue
    selected_mean_tradable_return: MetricValue

    def __post_init__(self) -> None:
        for name in ("forecast_count", "evaluated_count", "selected_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if self.evaluated_count > self.forecast_count:
            raise ValueError("evaluated forecasts cannot exceed all forecasts")
        if self.selected_count > self.evaluated_count:
            raise ValueError("selected forecasts cannot exceed evaluated forecasts")
        for name in (
            "direction_accuracy",
            "return_correlation",
            "mean_predicted_return",
            "mean_realized_close_return",
            "mean_tradable_return",
            "selected_mean_tradable_return",
        ):
            _metric(getattr(self, name), label=name)
        if self.direction_accuracy is not None and not 0 <= self.direction_accuracy <= 1:
            raise ValueError("direction_accuracy must be between zero and one")
        if self.return_correlation is not None and not -1 <= self.return_correlation <= 1:
            raise ValueError("return_correlation must be between negative one and one")


def calculate_forecast_quality(
    traces: Sequence[ForecastTrace],
    evaluations: Sequence[ForecastEvaluation],
) -> ForecastQualityMetrics:
    checked_traces = tuple(traces)
    checked_evaluations = tuple(evaluations)
    if any(type(item) is not ForecastTrace for item in checked_traces):
        raise TypeError("traces must contain exact ForecastTrace values")
    if any(type(item) is not ForecastEvaluation for item in checked_evaluations):
        raise TypeError("evaluations must contain exact ForecastEvaluation values")
    by_key = {
        (item.decision_date, item.strategy_id, item.instrument): item
        for item in checked_traces
    }
    matched: list[tuple[ForecastTrace, ForecastEvaluation]] = []
    for evaluation in checked_evaluations:
        trace = by_key.get(
            (
                evaluation.decision_date,
                evaluation.strategy_id,
                evaluation.instrument,
            )
        )
        if trace is None or trace.horizon != evaluation.horizon:
            raise ValueError("forecast evaluation has no matching trace")
        matched.append((trace, evaluation))

    predicted = [trace.expected_return for trace, _ in matched]
    realized = [evaluation.realized_close_return for _, evaluation in matched]
    tradable = [evaluation.tradable_return for _, evaluation in matched]
    selected_tradable = [
        evaluation.tradable_return
        for trace, evaluation in matched
        if trace.target_weight > 0
    ]
    directional = [
        (prediction > 0 and actual > 0)
        or (prediction < 0 and actual < 0)
        or (prediction == 0 and actual == 0)
        for prediction, actual in zip(predicted, realized, strict=True)
    ]
    return ForecastQualityMetrics(
        forecast_count=len(checked_traces),
        evaluated_count=len(matched),
        selected_count=len(selected_tradable),
        direction_accuracy=(
            None if not directional else float(sum(directional) / len(directional))
        ),
        return_correlation=_correlation(predicted, realized),
        mean_predicted_return=_mean(predicted),
        mean_realized_close_return=_mean(realized),
        mean_tradable_return=_mean(tradable),
        selected_mean_tradable_return=_mean(selected_tradable),
    )
