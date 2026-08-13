from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from compass.domain.market import AssetType, InstrumentId
from compass.strategies.base import HoldingSummary, StrategyContext, StrategyDecisionStatus
from compass.strategies.kronos_forecast import (
    KronosForecast,
    KronosForecastParameters,
    KronosForecastStrategy,
    KronosRuntimeStatus,
)


FIRST = InstrumentId.parse("SSE.510300")
SECOND = InstrumentId.parse("SZSE.159949")


def bars(*, rising: bool = True) -> pd.DataFrame:
    close = [3 + position * 0.01 for position in range(80)]
    if not rising:
        close = list(reversed(close))
    return pd.DataFrame(
        {
            "open": close,
            "high": [value + 0.02 for value in close],
            "low": [value - 0.02 for value in close],
            "close": close,
            "volume": [1000.0] * len(close),
            "amount": [3000.0] * len(close),
        },
        index=pd.bdate_range("2026-01-01", periods=len(close)),
    )


class ForecastStub:
    def __init__(self) -> None:
        self.calls = 0

    def forecast(self, histories, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        assert kwargs["seed"] == 7
        return tuple(
            KronosForecast(
                instrument=instrument,
                as_of=kwargs["as_of"],
                horizon=kwargs["horizon"],
                expected_return=0.05 if instrument == FIRST else -0.03,
                path_positive_ratio=0.8 if instrument == FIRST else 0.2,
                predicted_closes=tuple(
                    float(histories[instrument]["close"].iloc[-1]) * (1.01 + step * 0.01)
                    for step in range(kwargs["horizon"])
                ),
            )
            for instrument in histories
        )


def parameters() -> KronosForecastParameters:
    return KronosForecastParameters(
        model_size="mini",
        device="cpu",
        lookback=64,
        horizon=3,
        rebalance_interval=5,
        entry_return=Decimal("0.02"),
        exit_return=Decimal("-0.01"),
        minimum_path_positive_ratio=Decimal("0.6"),
        trend_window=20,
        top_n=1,
        target_weight=Decimal("0.8"),
        temperature=0.8,
        top_p=0.9,
        sample_count=2,
        seed=7,
    )


def context() -> StrategyContext:
    frames = {FIRST: bars(), SECOND: bars()}
    return StrategyContext(
        as_of=frames[FIRST].index[-1].date(),
        bars=frames,
        instruments=(FIRST, SECOND),
        account_equity=Decimal("100000"),
        cash=Decimal("50000"),
        holdings={
            SECOND: HoldingSummary(
                SECOND,
                quantity=1000,
                available_quantity=1000,
                average_cost=Decimal("3"),
                mark_price=Decimal("3.5"),
            )
        },
        asset_types={FIRST: AssetType.ETF, SECOND: AssetType.ETF},
    )


def test_kronos_strategy_converts_forecasts_to_entry_and_cash_targets() -> None:
    forecaster = ForecastStub()
    strategy = KronosForecastStrategy(parameters(), "kronos-test", forecaster)

    decision = strategy.generate_targets(context())

    assert decision.status is StrategyDecisionStatus.GENERATED
    assert tuple((item.instrument, item.target_weight, item.reason_code) for item in decision) == (
        (FIRST, Decimal("0.8"), "KRONOS_FORECAST_ENTRY"),
        (SECOND, Decimal("0"), "KRONOS_FORECAST_EXIT"),
    )
    assert forecaster.calls == 1


def test_kronos_strategy_skips_repeated_inference_inside_rebalance_interval() -> None:
    forecaster = ForecastStub()
    strategy = KronosForecastStrategy(parameters(), "kronos-test", forecaster)

    strategy.generate_targets(context())
    repeated = strategy.generate_targets(context())

    assert repeated.status is StrategyDecisionStatus.SKIPPED
    assert repeated.reason_code == "KRONOS_REBALANCE_NOT_DUE"
    assert forecaster.calls == 1


def test_kronos_parameters_enforce_model_context_and_hysteresis() -> None:
    with pytest.raises(ValueError, match="context"):
        KronosForecastParameters(model_size="small", lookback=513)
    with pytest.raises(ValueError, match="exit return"):
        KronosForecastParameters(
            entry_return=Decimal("0"),
            exit_return=Decimal("0"),
        )


def test_kronos_runtime_status_explains_cuda_and_cpu() -> None:
    cuda = KronosRuntimeStatus(True, "2.13.0+cu132", "13.2", True, "RTX 4070 Ti")
    cpu = KronosRuntimeStatus(True, "2.13.0+cpu", None, False, None)

    assert cuda.display_text == "CUDA 可用 · RTX 4070 Ti · PyTorch 2.13.0+cu132 / CUDA 13.2"
    assert cpu.display_text == "当前仅 CPU · PyTorch 2.13.0+cpu"
    assert cuda.action_text is None
    assert cpu.action_text is not None
