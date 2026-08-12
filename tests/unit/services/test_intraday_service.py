from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, timedelta
from decimal import Decimal, localcontext
from math import nan
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from compass.domain.market import AssetType, Exchange, InstrumentId
from compass.domain.trading import TargetIntent
from compass.services.intraday_service import (
    IntradayQuote,
    IntradayService,
    IntradaySignal,
    IntradayState,
)
from compass.strategies.base import StrategyContext
from compass.strategies.base import HoldingSummary
from compass.strategies.dual_ma import DualMaParameters, DualMaStrategy


SHANGHAI = ZoneInfo("Asia/Shanghai")
ETF = InstrumentId.parse("SSE.510300")
SECOND_ETF = InstrumentId.parse("SZSE.159915")


def shanghai_time(hour: int, minute: int, *, day: int = 21) -> datetime:
    return datetime(2026, 7, day, hour, minute, tzinfo=SHANGHAI)


def history(symbol: InstrumentId = ETF) -> pd.DataFrame:
    del symbol
    return pd.DataFrame(
        {
            "open": [9.8, 10.0],
            "high": [10.2, 10.4],
            "low": [9.7, 9.9],
            "close": [10.0, 10.2],
            "volume": [1000.0, 1200.0],
            "amount": [10000.0, 12240.0],
            "adjust_factor": [1.0, 1.1],
        },
        index=pd.DatetimeIndex(["2026-07-17", "2026-07-20"], name="date"),
    )


def snapshot(
    *symbols: InstrumentId,
    at: datetime | None = None,
    close: object = Decimal("11.00"),
    adjust_factor: object = Decimal("1.10"),
) -> pd.DataFrame:
    instruments = symbols or (ETF,)
    timestamp = at or shanghai_time(10, 5)
    return pd.DataFrame(
        {
            "instrument": [str(symbol) for symbol in instruments],
            "timestamp": [timestamp for _ in instruments],
            "open": [Decimal("10.80") for _ in instruments],
            "high": [Decimal("11.20") for _ in instruments],
            "low": [Decimal("10.70") for _ in instruments],
            "close": [close for _ in instruments],
            "volume": [Decimal("1500") for _ in instruments],
            "amount": [Decimal("16500") for _ in instruments],
            "adjust_factor": [adjust_factor for _ in instruments],
        }
    )


class FakeProvider:
    name = "fake"

    def __init__(self, responses: list[pd.DataFrame | Exception]) -> None:
        self.responses = responses
        self.calls: list[tuple[InstrumentId, ...]] = []

    def fetch_snapshot(self, instruments: tuple[InstrumentId, ...]) -> pd.DataFrame:
        self.calls.append(tuple(instruments))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response.copy(deep=True)


class Calculator:
    def __init__(
        self,
        weights: list[Decimal] | None = None,
        *,
        reverse: bool = False,
    ) -> None:
        self.weights = weights or [Decimal("0.50")]
        self.reverse = reverse
        self.contexts: list[StrategyContext] = []

    def __call__(self, context: StrategyContext) -> tuple[TargetIntent, ...]:
        self.contexts.append(context)
        weight = self.weights.pop(0) if len(self.weights) > 1 else self.weights[0]
        instruments = tuple(reversed(context.instruments)) if self.reverse else context.instruments
        return tuple(
            TargetIntent(
                strategy_id="intraday-demo",
                instrument=instrument,
                target_weight=weight,
                score=1.0,
                confidence=0.8,
                reason_code="PRICE_UPDATE",
                valid_until=context.as_of,
            )
            for instrument in instruments
        )


def create_service(
    provider: FakeProvider,
    calculator: Any,
    *,
    instruments: tuple[InstrumentId, ...] = (ETF,),
    is_trading_day: Any = lambda day: day == date(2026, 7, 21),
    freshness: timedelta = timedelta(minutes=10),
    cooldown: timedelta = timedelta(minutes=30),
    material_change: Decimal = Decimal("0.01"),
    stale_after_failures: int = 2,
    completion_clock: Callable[[], datetime] | None = None,
) -> IntradayService:
    arguments: dict[str, object] = {
        "instruments": instruments,
        "provider": provider,
        "daily_history": {
            instrument: history(instrument) for instrument in instruments
        },
        "calculator": calculator,
        "is_trading_day": is_trading_day,
        "account_equity": Decimal("100000"),
        "asset_types": {
            instrument: AssetType.ETF for instrument in instruments
        },
        "freshness": freshness,
        "cooldown": cooldown,
        "material_change": material_change,
        "stale_after_failures": stale_after_failures,
    }
    if completion_clock is not None:
        arguments["completion_clock"] = completion_clock
    return IntradayService(**arguments)  # type: ignore[arg-type]


def test_intraday_signal_is_temporary_and_merges_an_incomplete_daily_row_without_mutation() -> None:
    source_history = history()
    source_snapshot = snapshot()
    provider = FakeProvider([source_snapshot])
    calculator = Calculator()
    service = IntradayService(
        instruments=(ETF,),
        provider=provider,
        daily_history={ETF: source_history},
        calculator=calculator,
        is_trading_day=lambda _: True,
        account_equity=Decimal("100000"),
        asset_types={ETF: AssetType.ETF},
    )

    state = service.refresh(shanghai_time(10, 5))
    calculated = calculator.contexts[0].history(ETF)

    assert state.signals
    assert all(signal.status == "temporary" for signal in state.signals)
    assert all(signal.confirmed is False for signal in state.signals)
    assert state.persist_to_daily_results is False
    assert state.signals[0].source_at == shanghai_time(10, 5)
    assert state.signals[0].raw_price == Decimal("11.00")
    assert state.signals[0].comparable_price == Decimal("12.1000")
    assert calculated.index[-1] == pd.Timestamp("2026-07-21")
    assert Decimal(str(calculated.iloc[-1]["close"])) == Decimal("12.1000")
    assert source_history.index[-1] == pd.Timestamp("2026-07-20")
    assert source_snapshot.equals(snapshot())
    assert not hasattr(service, "persist")


def test_completion_clock_accepts_a_source_after_invocation_and_sets_state_observation() -> None:
    invoked_at = shanghai_time(10, 5)
    source_at = invoked_at + timedelta(microseconds=1)
    completed_at = invoked_at + timedelta(microseconds=2)
    calculator = Calculator()
    service = create_service(
        FakeProvider([snapshot(at=source_at)]),
        calculator,
        completion_clock=lambda: completed_at,
    )

    state = service.refresh(invoked_at)

    assert state.failure_code is None
    assert state.observed_at == completed_at
    assert state.source_at == source_at
    assert state.source_at <= state.observed_at
    assert calculator.contexts


@pytest.mark.parametrize(
    "completed_at",
    [
        datetime(2026, 7, 21, 10, 5),
        datetime(2026, 7, 21, 2, 5, tzinfo=ZoneInfo("UTC")),
        shanghai_time(10, 5) - timedelta(microseconds=1),
    ],
    ids=("naive", "non-shanghai", "before-invocation"),
)
def test_invalid_completion_clock_fails_closed_after_fetch(
    completed_at: datetime,
) -> None:
    provider = FakeProvider([snapshot()])
    calculator = Calculator()
    service = create_service(
        provider,
        calculator,
        completion_clock=lambda: completed_at,
    )

    state = service.refresh(shanghai_time(10, 5))

    assert provider.calls == [(ETF,)]
    assert calculator.contexts == []
    assert state.failure_code == "COMPLETION_CLOCK_FAILED"
    assert state.observed_at == shanghai_time(10, 5)
    assert state.quotes == ()
    assert state.signals == ()


def test_completion_clock_exception_fails_closed_without_leaking_details() -> None:
    def failing_clock() -> datetime:
        raise RuntimeError("sentinel-completion-clock-secret")

    calculator = Calculator()
    service = create_service(
        FakeProvider([snapshot()]),
        calculator,
        completion_clock=failing_clock,
    )

    state = service.refresh(shanghai_time(10, 5))

    assert calculator.contexts == []
    assert state.failure_code == "COMPLETION_CLOCK_FAILED"
    assert "sentinel-completion-clock-secret" not in repr(state)


@pytest.mark.parametrize(
    "second_invocation",
    [
        shanghai_time(10, 5),
        shanghai_time(10, 5) - timedelta(microseconds=1),
    ],
    ids=("same-invocation", "invocation-regressed"),
)
def test_completion_clock_exception_after_success_retains_monotonic_state(
    second_invocation: datetime,
) -> None:
    invoked_at = shanghai_time(10, 5)
    source_at = invoked_at + timedelta(microseconds=1)
    completed_at = invoked_at + timedelta(microseconds=2)
    completions: list[datetime | Exception] = [
        completed_at,
        RuntimeError("sentinel-completion-clock-secret"),
    ]

    def completion_clock() -> datetime:
        value = completions.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    provider = FakeProvider(
        [snapshot(at=source_at), snapshot(at=source_at)]
    )
    service = create_service(
        provider,
        Calculator(),
        completion_clock=completion_clock,
    )
    succeeded = service.refresh(invoked_at)

    failed = service.refresh(second_invocation)

    assert failed.failure_code == "COMPLETION_CLOCK_FAILED"
    assert failed.quotes == succeeded.quotes
    assert failed.signals == succeeded.signals
    assert failed.source_at == succeeded.source_at
    assert failed.observed_at >= succeeded.observed_at
    assert failed.observed_at >= failed.source_at
    assert failed.consecutive_failures == 1
    assert failed.stale is False
    assert failed.notifications == ()


@pytest.mark.parametrize(
    ("second_invocation", "second_completion"),
    [
        (
            shanghai_time(10, 5),
            shanghai_time(10, 5) - timedelta(microseconds=1),
        ),
        (
            shanghai_time(10, 5) - timedelta(microseconds=1),
            shanghai_time(10, 5) - timedelta(microseconds=1),
        ),
    ],
    ids=("completion-before-invocation", "completion-before-trusted-state"),
)
def test_completion_clock_regression_after_success_fails_closed_monotonically(
    second_invocation: datetime,
    second_completion: datetime,
) -> None:
    invoked_at = shanghai_time(10, 5)
    source_at = invoked_at + timedelta(microseconds=1)
    completed_at = invoked_at + timedelta(microseconds=2)
    completions = iter((completed_at, second_completion))
    service = create_service(
        FakeProvider([snapshot(at=source_at), snapshot(at=source_at)]),
        Calculator(),
        completion_clock=lambda: next(completions),
    )
    succeeded = service.refresh(invoked_at)

    failed = service.refresh(second_invocation)

    assert failed.failure_code == "COMPLETION_CLOCK_FAILED"
    assert failed.quotes == succeeded.quotes
    assert failed.source_at == source_at
    assert failed.observed_at >= succeeded.observed_at
    assert failed.observed_at >= source_at
    assert failed.consecutive_failures == 1
    assert failed.stale is False


def test_outside_session_regression_retains_last_success_with_monotonic_observation() -> None:
    invoked_at = shanghai_time(10, 5)
    source_at = invoked_at + timedelta(microseconds=1)
    completed_at = invoked_at + timedelta(microseconds=2)
    provider = FakeProvider([snapshot(at=source_at)])
    service = create_service(
        provider,
        Calculator(),
        completion_clock=lambda: completed_at,
    )
    succeeded = service.refresh(invoked_at)

    outside = service.refresh(shanghai_time(9, 29))

    assert outside.active_session is False
    assert outside.failure_code is None
    assert outside.quotes == succeeded.quotes
    assert outside.signals == succeeded.signals
    assert outside.source_at == source_at
    assert outside.observed_at >= succeeded.observed_at
    assert outside.observed_at >= source_at
    assert outside.consecutive_failures == 0
    assert outside.stale is False
    assert provider.calls == [(ETF,)]


def test_snapshot_later_than_completion_is_still_rejected_as_future_data() -> None:
    invoked_at = shanghai_time(10, 5)
    completed_at = invoked_at + timedelta(microseconds=1)
    calculator = Calculator()
    service = create_service(
        FakeProvider(
            [snapshot(at=completed_at + timedelta(microseconds=1))]
        ),
        calculator,
        completion_clock=lambda: completed_at,
    )

    state = service.refresh(invoked_at)

    assert calculator.contexts == []
    assert state.observed_at == completed_at
    assert state.failure_code == "SNAPSHOT_INVALID"


def test_snapshot_freshness_uses_completion_time_as_its_upper_bound() -> None:
    invoked_at = shanghai_time(10, 5)
    completed_at = invoked_at + timedelta(minutes=11)
    calculator = Calculator()
    service = create_service(
        FakeProvider([snapshot(at=invoked_at)]),
        calculator,
        completion_clock=lambda: completed_at,
    )

    state = service.refresh(invoked_at)

    assert calculator.contexts == []
    assert state.observed_at == completed_at
    assert state.failure_code == "SNAPSHOT_INVALID"
    assert state.stale is True


def test_notification_cooldown_uses_completion_time() -> None:
    invoked_at = shanghai_time(10, 5)
    completion_times = iter(
        (invoked_at, invoked_at + timedelta(minutes=30))
    )
    provider = FakeProvider(
        [
            snapshot(at=invoked_at),
            snapshot(at=invoked_at + timedelta(minutes=1)),
        ]
    )
    service = create_service(
        provider,
        Calculator(),
        freshness=timedelta(hours=1),
        completion_clock=lambda: next(completion_times),
    )

    first = service.refresh(invoked_at)
    cooled_down = service.refresh(invoked_at + timedelta(minutes=1))

    assert first.notifications == first.signals
    assert cooled_down.observed_at == invoked_at + timedelta(minutes=30)
    assert cooled_down.notifications == cooled_down.signals


def test_intraday_strategy_history_uses_one_comparable_price_space() -> None:
    class BreakoutCalculator:
        def __init__(self) -> None:
            self.closes: tuple[Decimal, ...] = ()

        def __call__(self, context: StrategyContext) -> tuple[TargetIntent, ...]:
            self.closes = tuple(
                Decimal(str(value)) for value in context.history(ETF)["close"]
            )
            if self.closes[-1] <= self.closes[-2] * Decimal("1.08"):
                return ()
            return (
                TargetIntent(
                    "breakout",
                    ETF,
                    Decimal("1"),
                    1.0,
                    1.0,
                    "BREAKOUT",
                    context.as_of,
                ),
            )

    calculator = BreakoutCalculator()
    service = create_service(FakeProvider([snapshot()]), calculator)

    state = service.refresh(shanghai_time(10, 5))

    assert calculator.closes[-2:] == (Decimal("11.22"), Decimal("12.1000"))
    assert state.signals == ()


def test_intraday_daily_history_requires_raw_adjustment_mode_when_declared() -> None:
    adjusted = history().assign(adjust_flag=["1", "1"])

    with pytest.raises(ValueError, match="raw.*adjust_flag"):
        IntradayService(
            instruments=(ETF,),
            provider=FakeProvider([snapshot()]),
            daily_history={ETF: adjusted},
            calculator=Calculator(),
            is_trading_day=lambda _: True,
            account_equity=Decimal("1"),
            asset_types={ETF: AssetType.ETF},
        )


def test_intraday_daily_history_without_factor_uses_identity_comparable_policy() -> None:
    raw = history().drop(columns="adjust_factor")
    calculator = Calculator()
    service = IntradayService(
        instruments=(ETF,),
        provider=FakeProvider([snapshot(adjust_factor=Decimal("1"))]),
        daily_history={ETF: raw},
        calculator=calculator,
        is_trading_day=lambda _: True,
        account_equity=Decimal("1"),
        asset_types={ETF: AssetType.ETF},
    )

    service.refresh(shanghai_time(10, 5))

    closes = calculator.contexts[0].history(ETF)["close"].tolist()
    assert closes[-3:] == [10.0, 10.2, Decimal("11.00")]


def test_intraday_passes_asset_metadata_to_a_builtin_strategy() -> None:
    rising = history().copy()
    rising.loc[:, ["open", "high", "low", "close"]] = [
        [9.0, 9.0, 9.0, 9.0],
        [10.0, 10.0, 10.0, 10.0],
    ]
    rising["adjust_factor"] = [Decimal("1"), Decimal("1")]
    strategy = DualMaStrategy(DualMaParameters(short_window=1, long_window=2))
    service = IntradayService(
        instruments=(ETF,),
        provider=FakeProvider(
            [snapshot(close=Decimal("11"), adjust_factor=Decimal("1"))]
        ),
        daily_history={ETF: rising},
        calculator=strategy.generate_targets,
        is_trading_day=lambda _: True,
        account_equity=Decimal("1000"),
        asset_types={ETF: AssetType.ETF},
    )

    state = service.refresh(shanghai_time(10, 5))

    assert state.failure_code is None
    assert state.signals[0].reason_code == "MA_BULL_CONFIRMED"


def test_intraday_passes_immutable_holdings_to_a_builtin_exit_strategy() -> None:
    falling = history().copy()
    falling.loc[:, ["open", "high", "low", "close"]] = [
        [12.0, 12.0, 12.0, 12.0],
        [11.0, 11.0, 11.0, 11.0],
    ]
    falling["adjust_factor"] = [Decimal("1"), Decimal("1")]
    holding = HoldingSummary(ETF, 100, 100, Decimal("11"), Decimal("10"))
    caller_holdings = [holding]
    strategy = DualMaStrategy(DualMaParameters(short_window=1, long_window=2))
    service = IntradayService(
        instruments=(ETF,),
        provider=FakeProvider(
            [snapshot(close=Decimal("10.80"), adjust_factor=Decimal("1"))]
        ),
        daily_history={ETF: falling},
        calculator=strategy.generate_targets,
        is_trading_day=lambda _: True,
        account_equity=Decimal("1000"),
        asset_types={ETF: AssetType.ETF},
        holdings=caller_holdings,
    )
    caller_holdings.clear()

    state = service.refresh(shanghai_time(10, 5))

    assert state.failure_code is None
    assert state.signals[0].reason_code == "MA_BEAR_EXIT"
    assert state.signals[0].target_weight == Decimal("0")


def test_duplicate_snapshot_columns_fail_closed_before_calculation() -> None:
    duplicated = pd.concat((snapshot(), snapshot()[["close"]]), axis=1)
    calculator = Calculator()
    service = create_service(FakeProvider([duplicated]), calculator)

    state = service.refresh(shanghai_time(10, 5))

    assert calculator.contexts == []
    assert state.failure_code == "SNAPSHOT_INVALID"
    assert state.signals == ()


@pytest.mark.parametrize(
    "now",
    [
        shanghai_time(9, 29),
        shanghai_time(11, 31),
        shanghai_time(12, 59),
        shanghai_time(15, 1),
    ],
)
def test_intraday_service_does_not_poll_outside_continuous_sessions(now: datetime) -> None:
    provider = FakeProvider([snapshot(at=now)])
    service = create_service(provider, Calculator())

    state = service.refresh(now)

    assert provider.calls == []
    assert state.active_session is False
    assert state.notifications == ()


@pytest.mark.parametrize("now", [shanghai_time(9, 30), shanghai_time(11, 30), shanghai_time(13, 0), shanghai_time(15, 0)])
def test_intraday_service_polls_at_continuous_session_boundaries(now: datetime) -> None:
    provider = FakeProvider([snapshot(at=now)])
    service = create_service(provider, Calculator())

    state = service.refresh(now)

    assert provider.calls == [(ETF,)]
    assert state.active_session is True


def test_intraday_service_trusts_the_injected_exchange_calendar() -> None:
    provider = FakeProvider([snapshot()])
    calendar_calls: list[date] = []

    def holiday_calendar(day: date) -> bool:
        calendar_calls.append(day)
        return False

    service = create_service(provider, Calculator(), is_trading_day=holiday_calendar)

    state = service.refresh(shanghai_time(10, 5))

    assert calendar_calls == [date(2026, 7, 21)]
    assert provider.calls == []
    assert state.active_session is False


def test_repeated_and_effectively_unchanged_signals_are_debounced_but_material_changes_notify() -> None:
    provider = FakeProvider(
        [
            snapshot(at=shanghai_time(10, 5)),
            snapshot(at=shanghai_time(10, 10)),
            snapshot(at=shanghai_time(10, 15)),
            snapshot(at=shanghai_time(10, 45)),
        ]
    )
    calculator = Calculator(
        [Decimal("0.500"), Decimal("0.505"), Decimal("0.515"), Decimal("0.519")]
    )
    service = create_service(provider, calculator)

    first = service.refresh(shanghai_time(10, 5))
    unchanged = service.refresh(shanghai_time(10, 10))
    material = service.refresh(shanghai_time(10, 15))
    cooled_down = service.refresh(shanghai_time(10, 45))

    assert first.notifications == first.signals
    assert unchanged.notifications == ()
    assert material.notifications == material.signals
    assert cooled_down.notifications == cooled_down.signals


def test_provider_failures_retain_display_signals_fail_closed_and_recover() -> None:
    provider = FakeProvider(
        [
            snapshot(at=shanghai_time(10, 5)),
            RuntimeError("upstream unavailable"),
            RuntimeError("upstream unavailable"),
            snapshot(at=shanghai_time(10, 20), close=Decimal("11.20")),
        ]
    )
    service = create_service(provider, Calculator())

    success = service.refresh(shanghai_time(10, 5))
    first_failure = service.refresh(shanghai_time(10, 10))
    stale_failure = service.refresh(shanghai_time(10, 15))
    recovered = service.refresh(shanghai_time(10, 20))

    assert first_failure.signals == success.signals
    assert first_failure.notifications == ()
    assert first_failure.failure_code == "SNAPSHOT_FETCH_FAILED"
    assert first_failure.consecutive_failures == 1
    assert first_failure.stale is False
    assert stale_failure.signals == success.signals
    assert stale_failure.notifications == ()
    assert stale_failure.consecutive_failures == 2
    assert stale_failure.stale is True
    assert recovered.failure_code is None
    assert recovered.consecutive_failures == 0
    assert recovered.stale is False


def test_calculation_failure_never_publishes_partially_computed_recommendations() -> None:
    class FailingCalculator:
        def __call__(self, context: StrategyContext) -> tuple[TargetIntent, ...]:
            del context
            raise ArithmeticError("cannot calculate")

    service = create_service(FakeProvider([snapshot()]), FailingCalculator())

    state = service.refresh(shanghai_time(10, 5))

    assert state.signals == ()
    assert state.notifications == ()
    assert state.failure_code == "SIGNAL_CALCULATION_FAILED"
    assert state.consecutive_failures == 1


def test_last_successful_quote_ages_to_stale_without_an_additional_provider_failure() -> None:
    provider = FakeProvider([snapshot(at=shanghai_time(11, 29))])
    service = create_service(provider, Calculator(), freshness=timedelta(minutes=10))
    fresh = service.refresh(shanghai_time(11, 29))

    stale = service.refresh(shanghai_time(11, 45))

    assert stale.signals == fresh.signals
    assert stale.notifications == ()
    assert stale.stale is True
    assert stale.failure_code == "LAST_QUOTE_STALE"
    assert stale.consecutive_failures == 0
    assert len(provider.calls) == 1


def test_multi_instrument_state_ages_from_the_oldest_display_quote() -> None:
    initial = snapshot(ETF, SECOND_ETF, at=shanghai_time(11, 29))
    initial.loc[1, "timestamp"] = shanghai_time(11, 25)
    provider = FakeProvider([initial])
    service = create_service(
        provider,
        Calculator(),
        instruments=(ETF, SECOND_ETF),
        freshness=timedelta(minutes=10),
    )

    fresh = service.refresh(shanghai_time(11, 29))
    stale = service.refresh(shanghai_time(11, 36))

    assert fresh.stale is False
    assert fresh.source_at == shanghai_time(11, 25)
    assert stale.stale is True
    assert stale.failure_code == "LAST_QUOTE_STALE"
    assert len(provider.calls) == 1


@pytest.mark.parametrize(
    ("bad_snapshot", "now", "stale"),
    [
        (snapshot(at=shanghai_time(10, 6)), shanghai_time(10, 5), False),
        (snapshot(at=shanghai_time(9, 50)), shanghai_time(10, 5), True),
        (snapshot(SECOND_ETF), shanghai_time(10, 5), False),
        (snapshot(ETF, ETF), shanghai_time(10, 5), False),
        (snapshot(close=nan), shanghai_time(10, 5), False),
        (snapshot(close=Decimal("0")), shanghai_time(10, 5), False),
        (snapshot(close=Decimal("11.30")), shanghai_time(10, 5), False),
        (snapshot(at=datetime(2026, 7, 21, 10, 5)), shanghai_time(10, 5), False),
    ],
)
def test_invalid_snapshots_are_rejected_before_strategy_evaluation(
    bad_snapshot: pd.DataFrame,
    now: datetime,
    stale: bool,
) -> None:
    calculator = Calculator()
    service = create_service(FakeProvider([bad_snapshot]), calculator)

    state = service.refresh(now)

    assert calculator.contexts == []
    assert state.signals == ()
    assert state.notifications == ()
    assert state.failure_code == "SNAPSHOT_INVALID"
    assert state.stale is stale


def test_intraday_outputs_are_symbol_sorted_and_deeply_immutable() -> None:
    provider = FakeProvider([snapshot(SECOND_ETF, ETF)])
    service = create_service(
        provider,
        Calculator(reverse=True),
        instruments=(SECOND_ETF, ETF),
    )

    state = service.refresh(shanghai_time(10, 5))

    assert tuple(str(signal.instrument) for signal in state.signals) == (
        "SSE.510300",
        "SZSE.159915",
    )
    assert isinstance(state, IntradayState)
    assert isinstance(state.signals, tuple)
    assert isinstance(state.notifications, tuple)
    with pytest.raises(FrozenInstanceError):
        state.stale = True  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        state.signals[0].target_weight = Decimal("0")  # type: ignore[misc]


StateMutation = Callable[[IntradayState], IntradayState]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda state: replace(state, source_at=None),
        lambda state: replace(state, quotes=(state.quotes[0], state.quotes[0])),
        lambda state: replace(state, signals=(state.signals[0], state.signals[0])),
        lambda state: replace(
            state,
            notifications=(state.notifications[0], state.notifications[0]),
        ),
        lambda state: replace(
            state,
            signals=(replace(state.signals[0], instrument=SECOND_ETF),),
            notifications=(),
        ),
        lambda state: replace(
            state,
            signals=(
                replace(
                    state.signals[0],
                    source_at=state.signals[0].source_at - timedelta(minutes=1),
                ),
            ),
            notifications=(),
        ),
        lambda state: replace(
            state,
            signals=(replace(state.signals[0], raw_price=Decimal("10.99")),),
            notifications=(),
        ),
        lambda state: replace(
            state,
            stale=True,
            failure_code="LAST_QUOTE_STALE",
        ),
        lambda state: replace(
            state,
            consecutive_failures=1,
            failure_code=None,
            notifications=(),
        ),
        lambda state: replace(
            state,
            stale=False,
            consecutive_failures=0,
            failure_code="LAST_QUOTE_STALE",
            notifications=(),
        ),
    ],
    ids=(
        "missing-source-for-visible-data",
        "duplicate-quote",
        "duplicate-signal",
        "duplicate-notification",
        "signal-without-quote",
        "signal-time-mismatch",
        "signal-price-mismatch",
        "stale-state-with-notification",
        "failure-count-without-code",
        "stale-code-with-fresh-state",
    ),
)
def test_intraday_state_rejects_internally_inconsistent_public_values(
    mutation: StateMutation,
) -> None:
    service = create_service(FakeProvider([snapshot()]), Calculator())
    valid = service.refresh(shanghai_time(10, 5))
    assert isinstance(valid.quotes[0], IntradayQuote)

    with pytest.raises((TypeError, ValueError)):
        mutation(valid)


def test_intraday_signal_strictly_rejects_invalid_public_values() -> None:
    values: dict[str, object] = {
        "strategy_id": "demo",
        "instrument": ETF,
        "target_weight": Decimal("0.5"),
        "score": 1.0,
        "confidence": 0.8,
        "reason_code": "PRICE_UPDATE",
        "source_at": shanghai_time(10, 5),
        "raw_price": Decimal("11.00"),
        "comparable_price": Decimal("12.10"),
    }
    IntradaySignal(**values)  # type: ignore[arg-type]

    for field, invalid in (
        ("target_weight", 0.5),
        ("target_weight", Decimal("NaN")),
        ("score", True),
        ("confidence", nan),
        ("reason_code", "not-stable"),
        ("source_at", datetime(2026, 7, 21, 10, 5)),
        ("raw_price", True),
        ("comparable_price", Decimal("0")),
    ):
        candidate = dict(values)
        candidate[field] = invalid
        with pytest.raises((TypeError, ValueError)):
            IntradaySignal(**candidate)  # type: ignore[arg-type]


def test_intraday_quote_rejects_noncanonical_instrument() -> None:
    quote = create_service(FakeProvider([snapshot()]), Calculator()).refresh(
        shanghai_time(10, 5)
    ).quotes[0]
    malformed = InstrumentId(Exchange.SSE, "not-six-digits")

    with pytest.raises(ValueError, match="canonical"):
        replace(quote, instrument=malformed)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda quote: replace(quote, comparable_price=Decimal("999")),
        lambda quote: replace(quote, adjust_factor=Decimal("2")),
    ],
    ids=("wrong-comparable-price", "factor-no-longer-matches-products"),
)
def test_intraday_quote_rejects_inconsistent_raw_factor_comparable_products(
    mutation: Callable[[IntradayQuote], IntradayQuote],
) -> None:
    quote = create_service(FakeProvider([snapshot()]), Calculator()).refresh(
        shanghai_time(10, 5)
    ).quotes[0]

    with pytest.raises(ValueError, match="comparable"):
        mutation(quote)


def test_intraday_quote_product_audit_is_independent_of_ambient_decimal_context() -> None:
    quote = create_service(FakeProvider([snapshot()]), Calculator()).refresh(
        shanghai_time(10, 5)
    ).quotes[0]

    with localcontext() as context:
        context.prec = 2
        reconstructed = replace(quote)

    assert reconstructed.comparable_price == Decimal("12.1000")


def test_intraday_quote_product_audit_ignores_ambient_decimal_exponent_limits() -> None:
    quote = create_service(FakeProvider([snapshot()]), Calculator()).refresh(
        shanghai_time(10, 5)
    ).quotes[0]

    with localcontext() as context:
        context.prec = 2
        context.Emax = 3
        context.Emin = -3
        reconstructed = replace(
            quote,
            raw_open=Decimal("1000"),
            raw_high=Decimal("1000"),
            raw_low=Decimal("1000"),
            raw_price=Decimal("1000"),
            comparable_open=Decimal("10000"),
            comparable_high=Decimal("10000"),
            comparable_low=Decimal("10000"),
            comparable_price=Decimal("10000"),
            adjust_factor=Decimal("10"),
        )

    assert reconstructed.comparable_price == Decimal("10000")


def test_intraday_signal_rejects_noncanonical_instrument() -> None:
    signal = create_service(FakeProvider([snapshot()]), Calculator()).refresh(
        shanghai_time(10, 5)
    ).signals[0]

    with pytest.raises(ValueError, match="canonical"):
        replace(signal, instrument=InstrumentId(Exchange.SSE, "BAD"))


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("freshness", timedelta(0)),
        ("cooldown", timedelta(0)),
        ("material_change", 0.01),
        ("material_change", Decimal("NaN")),
        ("stale_after_failures", True),
        ("stale_after_failures", 0),
    ],
)
def test_intraday_service_rejects_invalid_configuration(field: str, invalid: object) -> None:
    arguments: dict[str, object] = {
        "instruments": (ETF,),
        "provider": FakeProvider([snapshot()]),
        "daily_history": {ETF: history()},
        "calculator": Calculator(),
        "is_trading_day": lambda _: True,
        "account_equity": Decimal("1"),
        "asset_types": {ETF: AssetType.ETF},
        field: invalid,
    }

    with pytest.raises((TypeError, ValueError)):
        IntradayService(**arguments)  # type: ignore[arg-type]
