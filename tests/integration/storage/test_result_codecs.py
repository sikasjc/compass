from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from compass.backtest.engine import ForecastEvaluation, ForecastTrace
from compass.domain.market import InstrumentId
from compass.storage.backtest_result_codec import (
    decode_backtest_result,
    encode_backtest_result,
)
from compass.storage.decision_result_codec import (
    decode_decision_result,
    encode_decision_result,
)
from tests.integration.services.test_decision_service import _audited_result
from tests.support.sleeve_results import multi_reallocation_result


def test_backtest_codec_round_trips_forecast_diagnostics() -> None:
    result = replace(
        multi_reallocation_result(),
        forecast_traces=(
            ForecastTrace(
                decision_date=date(2026, 7, 20),
                strategy_id="kronos-1",
                instrument=InstrumentId.parse("SSE.510300"),
                action="BUY",
                expected_return=0.031,
                path_positive_ratio=0.75,
                rank=1,
                close=4.12,
                trend_value=4.01,
                trend_passed=True,
                target_weight=Decimal("0.8"),
                reason_code="KRONOS_FORECAST_ENTRY",
                horizon=3,
            ),
        ),
        forecast_evaluations=(
            ForecastEvaluation(
                decision_date=date(2026, 7, 20),
                strategy_id="kronos-1",
                instrument=InstrumentId.parse("SSE.510300"),
                horizon=3,
                execution_date=date(2026, 7, 21),
                evaluation_date=date(2026, 7, 23),
                realized_close_return=0.02,
                tradable_return=0.01,
            ),
        ),
    )

    assert decode_backtest_result(encode_backtest_result(result)) == result


def test_backtest_codec_reads_pre_evaluation_forecast_payload() -> None:
    result = replace(
        multi_reallocation_result(),
        forecast_traces=(
            ForecastTrace(
                decision_date=date(2026, 7, 20),
                strategy_id="kronos-legacy",
                instrument=InstrumentId.parse("SSE.510300"),
                action="CASH",
                expected_return=-0.01,
                path_positive_ratio=0.25,
                rank=1,
                close=4.12,
                trend_value=4.20,
                trend_passed=False,
                target_weight=Decimal("0"),
                reason_code="KRONOS_FORECAST_CASH",
            ),
        ),
    )
    encoded = encode_backtest_result(result)
    encoded.pop("forecast_evaluations")
    encoded["forecast_traces"][0].pop("horizon")

    restored = decode_backtest_result(encoded)

    assert restored.forecast_traces[0].horizon == 1
    assert restored.forecast_evaluations == ()


def test_backtest_codec_rejects_unknown_nested_record_keys() -> None:
    encoded = encode_backtest_result(multi_reallocation_result())
    mutations = (
        lambda payload: payload["orders"][0].__setitem__("unexpected", True),
        lambda payload: payload["fills"][0].__setitem__("unexpected", True),
        lambda payload: payload["ledger"][0].__setitem__("unexpected", True),
        lambda payload: payload["ledger"][1]["positions"][0].__setitem__("unexpected", True),
    )
    for mutate in mutations:
        tampered = deepcopy(encoded)
        mutate(tampered)
        with pytest.raises(ValueError, match="BACKTEST_RESULT_INTEGRITY"):
            decode_backtest_result(tampered)


def test_decision_codec_rejects_unknown_nested_record_keys(tmp_path: Path) -> None:
    encoded = encode_decision_result(_audited_result(tmp_path))
    recommendation = encoded["recommendations"][0]
    mutations = (
        lambda payload: payload["strategy_decisions"][0].__setitem__("unexpected", True),
        lambda payload: payload["recommendations"][0]["raw_intents"][0].__setitem__(
            "unexpected", True
        ),
        lambda payload: payload["recommendations"][0]["allocation_trace"][0].__setitem__(
            "unexpected", True
        ),
        lambda payload: payload["recommendations"][0]["risk_adjustments"][0].__setitem__(
            "unexpected", True
        ),
        lambda payload: payload["recommendations"][0]["costs"].__setitem__("unexpected", True),
    )
    assert recommendation["allocation_trace"]
    assert recommendation["risk_adjustments"]
    for mutate in mutations:
        tampered = deepcopy(encoded)
        mutate(tampered)
        with pytest.raises(ValueError, match="DECISION_RESULT_INTEGRITY"):
            decode_decision_result(tampered)
