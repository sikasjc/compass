from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest
from pydantic import ValidationError

from compass.domain.market import AssetType, InstrumentId
from compass.domain.weights import weight_to_units
from compass.strategies.base import (
    HoldingSummary,
    StrategyContext,
    StrategyDecision,
    StrategyDecisionStatus,
    StrategyFrequency,
)
from compass.strategies.dual_ma import DualMaParameters, DualMaStrategy
from compass.strategies.etf_rotation import (
    EtfRotationParameters,
    EtfRotationStrategy,
    _score_weights,
)
from compass.strategies.mean_reversion import (
    MeanReversionParameters,
    MeanReversionStrategy,
)
from compass.strategies.momentum import (
    CrossSectionalMomentumParameters,
    CrossSectionalMomentumStrategy,
    RebalanceFrequency,
    _equal_weights,
    _prepare_context,
    _weighted_momentum,
)
from compass.strategies.registry import StrategyRegistry


ETF_A = InstrumentId.parse("SSE.510300")
ETF_B = InstrumentId.parse("SZSE.159915")
ETF_C = InstrumentId.parse("SSE.512100")
STOCK = InstrumentId.parse("SSE.600000")


def bars(closes: list[float], start: str = "2025-01-01") -> pd.DataFrame:
    index = pd.bdate_range(start, periods=len(closes))
    close = pd.Series(closes, index=index, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1000.0,
            "amount": close * 1000.0,
        },
        index=index,
    )


def context(
    frames: dict[InstrumentId, pd.DataFrame],
    *,
    as_of: date | None = None,
    holdings: tuple[HoldingSummary, ...] = (),
    include_asset_types: bool = True,
) -> StrategyContext:
    last = min(frame.index[-1].date() for frame in frames.values())
    return StrategyContext(
        as_of or last,
        frames,
        tuple(frames),
        Decimal("100000"),
        holdings=holdings,
        asset_types=(
            {
                instrument: AssetType.STOCK if instrument == STOCK else AssetType.ETF
                for instrument in frames
            }
            if include_asset_types
            else None
        ),
    )


def test_parameter_defaults_metadata_and_builtin_registration() -> None:
    rotation = EtfRotationParameters()
    dual = DualMaParameters()
    momentum = CrossSectionalMomentumParameters()
    mean = MeanReversionParameters()

    assert rotation.lookbacks == (20, 60, 120)
    assert (rotation.trend_window, rotation.volatility_window, rotation.top_n) == (120, 20, 3)
    assert (dual.short_window, dual.long_window) == (20, 60)
    assert momentum.lookbacks == (60, 120)
    assert (mean.rsi_window, mean.bollinger_window, mean.bollinger_std) == (14, 20, 2.0)

    registry = StrategyRegistry()
    for strategy in (
        EtfRotationStrategy,
        DualMaStrategy,
        CrossSectionalMomentumStrategy,
        MeanReversionStrategy,
    ):
        registry.register(strategy.strategy_type, strategy)
    assert registry.strategy_types() == (
        "cross_sectional_momentum",
        "dual_ma",
        "etf_rotation",
        "mean_reversion",
    )
    for metadata in registry.list_metadata():
        assert metadata.version == "1.0.0"
        assert metadata.supported_frequencies == frozenset({StrategyFrequency.DAILY})
        assert metadata.required_fields == frozenset({"close"})
        assert metadata.minimum_history > 0
        assert all(field.description for field in metadata.parameters_type.model_fields.values())
    assert EtfRotationStrategy.metadata.supported_asset_types == frozenset({AssetType.ETF})
    assert AssetType.STOCK in CrossSectionalMomentumStrategy.metadata.supported_asset_types
    assert EtfRotationStrategy.metadata.default_required_history == 121
    assert CrossSectionalMomentumStrategy.metadata.default_required_history == 121
    assert DualMaStrategy.metadata.default_required_history == 60
    assert MeanReversionStrategy.metadata.default_required_history == 20


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (EtfRotationParameters, {"lookbacks": (20, 20)}),
        (EtfRotationParameters, {"lookback_weights": (1.0,)}),
        (EtfRotationParameters, {"top_n": 0}),
        (EtfRotationParameters, {"top_n": True}),
        (EtfRotationParameters, {"lookbacks": (True, 60, 120)}),
        (EtfRotationParameters, {"lookback_weights": (float("inf"), 1.0, 1.0)}),
        (EtfRotationParameters, {"volatility_penalty": True}),
        (EtfRotationParameters, {"volatility_penalty": 100.1}),
        (DualMaParameters, {"short_window": 60, "long_window": 20}),
        (DualMaParameters, {"confirmation_days": True}),
        (DualMaParameters, {"target_weight": "0.5"}),
        (CrossSectionalMomentumParameters, {"lookbacks": (120, 60)}),
        (CrossSectionalMomentumParameters, {"turnover_buffer": False}),
        (CrossSectionalMomentumParameters, {"lookback_weights": (1e308, 1e308)}),
        (MeanReversionParameters, {"entry_rsi": 70.0, "exit_rsi": 60.0}),
        (MeanReversionParameters, {"max_holding_sessions": True}),
        (MeanReversionParameters, {"bollinger_std": "2.0"}),
        (MeanReversionParameters, {"bollinger_std": 100.1}),
        (MeanReversionParameters, {"stop_loss": 1.0}),
        (MeanReversionParameters, {"rsi_window": 10_001}),
    ],
)
def test_parameter_models_reject_invalid_boundaries(model: type, kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        model(**kwargs)


def test_etf_rotation_selects_top_two_and_normalizes_decimal_weights() -> None:
    frames = {
        ETF_A: bars([100 * (1.004**day) for day in range(130)]),
        ETF_B: bars([100 * (1.003**day) for day in range(130)]),
        ETF_C: bars([100 * (1.001**day) for day in range(130)]),
    }
    strategy = EtfRotationStrategy(
        EtfRotationParameters(
            top_n=2,
            lookbacks=(20, 60, 120),
            lookback_weights=(1.0, 1.0, 1.0),
            volatility_penalty=0.0,
            rebalance_frequency=RebalanceFrequency.DAILY,
        )
    )

    intents = strategy.generate_targets(context(frames))

    assert [intent.instrument for intent in intents] == [ETF_A, ETF_B]
    assert sum((intent.target_weight for intent in intents), Decimal("0")) == Decimal("1")
    assert all(intent.reason_code == "MOMENTUM_TOP_N" for intent in intents)
    assert intents[0].score > intents[1].score
    assert all(intent.valid_until == frames[ETF_A].index[-1].date() for intent in intents)


def test_etf_rotation_filters_below_trend_and_uses_valid_risk_alternative() -> None:
    falling = bars([200.0 - day for day in range(130)])
    alternative = bars([100.0] * 130)
    strategy = EtfRotationStrategy(
        EtfRotationParameters(
            top_n=1,
            rebalance_frequency=RebalanceFrequency.DAILY,
            risk_alternative="SSE.511010",
        )
    )
    alt = InstrumentId.parse("SSE.511010")

    intents = strategy.generate_targets(context({ETF_A: falling, alt: alternative}))

    assert [(intent.instrument, intent.reason_code, intent.target_weight) for intent in intents] == [
        (alt, "RISK_ALTERNATIVE", Decimal("1"))
    ]
    missing_alternative = strategy.generate_targets(context({ETF_A: falling}))
    assert tuple(missing_alternative) == ()
    assert missing_alternative.reason_code == "NO_CANDIDATES_CASH"


def test_rotation_and_momentum_return_empty_when_not_rebalance_or_history_is_short() -> None:
    frame = bars([100.0 + day for day in range(20)], start="2026-01-05")
    weekly = EtfRotationStrategy(EtfRotationParameters())
    assert tuple(weekly.generate_targets(context({ETF_A: frame}))) == ()

    short_with_alternative = EtfRotationStrategy(
        EtfRotationParameters(
            risk_alternative="SSE.511010", rebalance_frequency=RebalanceFrequency.DAILY
        )
    )
    alt = InstrumentId.parse("SSE.511010")
    assert tuple(short_with_alternative.generate_targets(
        context({ETF_A: frame, alt: bars([100.0] * 20, start="2026-01-05")})
    )) == ()

    longer = bars([100.0 + day for day in range(130)], start="2025-08-04")
    as_of = longer.index[-1].date()
    if longer.index[-1].isocalendar().week != longer.index[-2].isocalendar().week:
        as_of = longer.index[-2].date()
    assert tuple(weekly.generate_targets(context({ETF_A: longer}, as_of=as_of))) == ()


def test_dual_ma_requires_close_confirmation_and_emits_bull_exit_or_no_signal() -> None:
    rising = bars([100.0] * 8 + [101.0, 102.0, 103.0, 104.0])
    parameters = DualMaParameters(
        short_window=2, long_window=4, confirmation_days=2, target_weight=Decimal("0.8")
    )
    strategy = DualMaStrategy(parameters)

    bull = strategy.generate_targets(context({STOCK: rising}))
    assert [(intent.reason_code, intent.target_weight) for intent in bull] == [
        ("MA_BULL_CONFIRMED", Decimal("0.8"))
    ]

    unconfirmed = bars([100.0] * 9 + [101.0])
    assert tuple(strategy.generate_targets(context({STOCK: unconfirmed}))) == ()

    falling = bars([104.0, 103.0, 102.0, 101.0, 100.0, 99.0])
    holding = HoldingSummary(STOCK, 100, 100, Decimal("102"), Decimal("99"))
    exit_intent = strategy.generate_targets(context({STOCK: falling}, holdings=(holding,)))
    assert [(intent.reason_code, intent.target_weight) for intent in exit_intent] == [
        ("MA_BEAR_EXIT", Decimal("0"))
    ]
    assert tuple(strategy.generate_targets(context({STOCK: falling}))) == ()


def test_cross_sectional_momentum_uses_visible_trading_boundary_and_stable_ties() -> None:
    dates = pd.bdate_range("2025-11-03", periods=50)
    frames = {
        ETF_B: bars([100.0 + day for day in range(50)], start="2025-11-03"),
        ETF_A: bars([100.0 + day for day in range(50)], start="2025-11-03"),
        STOCK: bars([100.0 + day * 0.5 for day in range(50)], start="2025-11-03"),
    }
    parameters = CrossSectionalMomentumParameters(
        lookbacks=(10, 20),
        lookback_weights=(1.0, 1.0),
        top_n=2,
        rebalance_frequency=RebalanceFrequency.WEEKLY,
    )
    strategy = CrossSectionalMomentumStrategy(parameters)
    boundary_index = next(
        index
        for index in range(1, len(dates))
        if dates[index].isocalendar().week != dates[index - 1].isocalendar().week
        and index >= 20
    )

    intents = strategy.generate_targets(
        context(frames, as_of=dates[boundary_index].date())
    )
    assert [intent.instrument for intent in intents] == [ETF_A, ETF_B]
    assert [intent.target_weight for intent in intents] == [Decimal("0.5"), Decimal("0.5")]
    assert {intent.reason_code for intent in intents} == {"CROSS_SECTIONAL_TOP_N"}
    assert tuple(strategy.generate_targets(
        context(frames, as_of=dates[boundary_index + 1].date())
    )) == ()


def test_rebalance_boundary_uses_union_of_visible_sessions_not_first_instrument() -> None:
    complete = bars([100.0 + day for day in range(26)], start="2026-01-05")
    assert complete.index[-1].weekday() == 0
    first_is_stale = complete.iloc[:-1]
    strategy = CrossSectionalMomentumStrategy(
        CrossSectionalMomentumParameters(
            lookbacks=(2, 3),
            lookback_weights=(1.0, 1.0),
            top_n=1,
            rebalance_frequency=RebalanceFrequency.WEEKLY,
        )
    )

    intents = strategy.generate_targets(
        context({ETF_A: first_is_stale, ETF_B: complete}, as_of=complete.index[-1].date())
    )

    assert len(intents) == 1


def test_mean_reversion_has_stable_entry_hold_and_exit_priority_reason_codes() -> None:
    falling = bars([100.0] * 20 + [98.0, 95.0, 90.0])
    parameters = MeanReversionParameters(
        rsi_window=3,
        bollinger_window=5,
        bollinger_std=1.5,
        entry_rsi=30,
        exit_rsi=60,
        stop_loss=0.1,
        max_holding_sessions=3,
        target_weight=Decimal("0.4"),
    )
    strategy = MeanReversionStrategy(parameters)
    entry = strategy.generate_targets(context({STOCK: falling}))
    assert [(intent.reason_code, intent.target_weight) for intent in entry] == [
        ("MEAN_REVERSION_ENTRY", Decimal("0.4"))
    ]

    hold_frame = bars([100.0, 99.0, 98.0, 97.0, 96.0, 95.0])
    unknown_age = HoldingSummary(STOCK, 100, 100, Decimal("100"), Decimal("95"))
    hold = strategy.generate_targets(context({STOCK: hold_frame}, holdings=(unknown_age,)))
    assert [(intent.reason_code, intent.target_weight) for intent in hold] == [
        ("MEAN_REVERSION_HOLD", Decimal("0.4"))
    ]

    stopped = HoldingSummary(STOCK, 100, 100, Decimal("110"), Decimal("95"))
    stopped_intent = strategy.generate_targets(context({STOCK: hold_frame}, holdings=(stopped,)))
    assert stopped_intent[0].reason_code == "MEAN_REVERSION_STOP_LOSS"
    assert stopped_intent[0].target_weight == Decimal("0")

    old = HoldingSummary(
        STOCK,
        100,
        100,
        Decimal("100"),
        Decimal("95"),
        holding_since=hold_frame.index[-3].date(),
    )
    old_intent = strategy.generate_targets(context({STOCK: hold_frame}, holdings=(old,)))
    assert old_intent[0].reason_code == "MEAN_REVERSION_MAX_HOLDING"


def test_all_strategies_ignore_extreme_future_rows_byte_for_byte() -> None:
    base = bars([100.0 + day * 0.3 for day in range(130)])
    as_of = base.index[-1].date()
    future = base.copy()
    future_day = future.index[-1] + pd.offsets.BDay()
    future.loc[future_day] = [1_000_000.0, 1_010_000.0, 990_000.0, 1_000_000.0, 1.0, 1_000_000.0]
    strategies = (
        EtfRotationStrategy(EtfRotationParameters(rebalance_frequency=RebalanceFrequency.DAILY)),
        DualMaStrategy(),
        CrossSectionalMomentumStrategy(
            CrossSectionalMomentumParameters(rebalance_frequency=RebalanceFrequency.DAILY)
        ),
        MeanReversionStrategy(),
    )
    for strategy in strategies:
        before = tuple(strategy.generate_targets(context({ETF_A: base}, as_of=as_of)))
        after = tuple(strategy.generate_targets(context({ETF_A: future}, as_of=as_of)))
        assert before == after


def test_insufficient_history_never_creates_an_intent() -> None:
    short = bars([100.0] * 10)
    strategies = (
        EtfRotationStrategy(EtfRotationParameters(rebalance_frequency=RebalanceFrequency.DAILY)),
        DualMaStrategy(),
        CrossSectionalMomentumStrategy(
            CrossSectionalMomentumParameters(rebalance_frequency=RebalanceFrequency.DAILY)
        ),
        MeanReversionStrategy(),
    )
    assert all(
        tuple(strategy.generate_targets(context({ETF_A: short}))) == ()
        for strategy in strategies
    )


def test_empty_decisions_explain_no_instruments_metadata_history_and_stale_data() -> None:
    empty_context = StrategyContext(
        date(2026, 1, 2), {}, (), Decimal("1"), asset_types={}
    )
    no_instruments = DualMaStrategy().generate_targets(empty_context)
    assert isinstance(no_instruments, StrategyDecision)
    assert no_instruments.status is StrategyDecisionStatus.SKIPPED
    assert no_instruments.reason_code == "NO_INSTRUMENTS"

    frame = bars([100.0] * 130)
    missing_metadata = EtfRotationStrategy().generate_targets(
        context({ETF_A: frame}, include_asset_types=False)
    )
    assert missing_metadata.reason_code == "MISSING_ASSET_METADATA"

    insufficient = CrossSectionalMomentumStrategy(
        CrossSectionalMomentumParameters(rebalance_frequency=RebalanceFrequency.DAILY)
    ).generate_targets(context({ETF_A: bars([100.0] * 10)}))
    assert insufficient.reason_code == "INSUFFICIENT_HISTORY"
    assert insufficient.details["required_history"] == 121

    stale = DualMaStrategy().generate_targets(
        context({STOCK: frame}, as_of=frame.index[-1].date() + pd.Timedelta(days=1))
    )
    assert stale.reason_code == "STALE_DATA"


def test_stale_high_score_is_skipped_and_fresh_lower_score_wins() -> None:
    fresh = bars([100.0 + day * 0.2 for day in range(130)])
    stale_high = bars([100.0 + day for day in range(129)])
    strategy = CrossSectionalMomentumStrategy(
        CrossSectionalMomentumParameters(
            top_n=1, rebalance_frequency=RebalanceFrequency.DAILY
        )
    )

    decision = strategy.generate_targets(
        context({ETF_A: stale_high, ETF_B: fresh}, as_of=fresh.index[-1].date())
    )

    assert [intent.instrument for intent in decision] == [ETF_B]
    assert decision.details["skipped_stale"] == 1


def test_risk_alternative_requires_fresh_etf_and_usable_history() -> None:
    falling = bars([300.0 - day for day in range(130)])
    one_bar = bars([100.0], start=str(falling.index[-1].date()))
    alt = InstrumentId.parse("SSE.511010")
    strategy = EtfRotationStrategy(
        EtfRotationParameters(
            risk_alternative=str(alt), rebalance_frequency=RebalanceFrequency.DAILY
        )
    )

    decision = strategy.generate_targets(context({ETF_A: falling, alt: one_bar}))

    assert tuple(decision) == ()
    assert decision.status is StrategyDecisionStatus.CASH
    assert decision.reason_code == "NO_CANDIDATES_CASH"
    assert decision.details["risk_alternative_usable"] is False


def test_fresh_risk_alternative_does_not_mask_stale_primary_data() -> None:
    primary = bars([200.0 - day * 0.2 for day in range(129)])
    alternative = bars([100.0] * 130)
    alt = InstrumentId.parse("SSE.511010")
    strategy = EtfRotationStrategy(
        EtfRotationParameters(
            risk_alternative=str(alt), rebalance_frequency=RebalanceFrequency.DAILY
        )
    )

    decision = strategy.generate_targets(
        context({ETF_A: primary, alt: alternative}, as_of=alternative.index[-1].date())
    )

    assert tuple(decision) == ()
    assert decision.reason_code == "STALE_DATA"


def test_any_stale_primary_blocks_risk_alternative_for_incomplete_cross_section() -> None:
    stale = bars([100.0 + day for day in range(129)])
    fresh_but_falling = bars([300.0 - day for day in range(130)])
    alternative = bars([100.0] * 130)
    alt = InstrumentId.parse("SSE.511010")
    strategy = EtfRotationStrategy(
        EtfRotationParameters(
            risk_alternative=str(alt), rebalance_frequency=RebalanceFrequency.DAILY
        )
    )

    decision = strategy.generate_targets(
        context(
            {ETF_A: stale, ETF_B: fresh_but_falling, alt: alternative},
            as_of=alternative.index[-1].date(),
        )
    )

    assert tuple(decision) == ()
    assert decision.reason_code == "STALE_DATA"
    assert decision.details["stale_primary_instruments"] == 1


def test_unsupported_asset_is_configuration_skip_while_mixed_pool_is_evaluated() -> None:
    stock_only = bars([100.0 + day for day in range(130)])
    unsupported = EtfRotationStrategy().generate_targets(context({STOCK: stock_only}))
    assert unsupported.status is StrategyDecisionStatus.SKIPPED
    assert unsupported.reason_code == "UNSUPPORTED_ASSET"

    helper_decision = _prepare_context(
        context({ETF_A: stock_only}), frozenset()
    )
    assert isinstance(helper_decision, StrategyDecision)
    assert helper_decision.reason_code == "UNSUPPORTED_ASSET"

    etf = bars([100.0 * (1.004**day) for day in range(130)])
    mixed = EtfRotationStrategy(
        EtfRotationParameters(
            volatility_penalty=0.0, rebalance_frequency=RebalanceFrequency.DAILY
        )
    ).generate_targets(context({ETF_A: etf, STOCK: stock_only}))
    assert len(mixed) == 1
    assert mixed[0].instrument == ETF_A
    assert mixed.details["skipped_unsupported_asset"] == 1


def test_weighted_momentum_normalizes_large_finite_weights_before_multiplication() -> None:
    two_periods = pd.Series([1.0, 1.0, 3.0])
    one_period = pd.Series([1.0, 3.0])

    assert _weighted_momentum(two_periods, (1, 2), (5e307, 5e307)) == pytest.approx(2.0)
    assert _weighted_momentum(one_period, (1,), (1e308,)) == pytest.approx(2.0)
    assert _weighted_momentum(pd.Series([0.0, 1.0]), (1,), (1.0,)) is None


def test_required_history_is_parameter_derived_and_registry_visible() -> None:
    rotation = EtfRotationStrategy(
        EtfRotationParameters(
            lookbacks=(10, 50),
            lookback_weights=(1.0, 1.0),
            trend_window=80,
            volatility_window=30,
        )
    )
    momentum = CrossSectionalMomentumStrategy(
        CrossSectionalMomentumParameters(lookbacks=(5, 90), lookback_weights=(1.0, 1.0))
    )
    dual = DualMaStrategy(DualMaParameters(short_window=5, long_window=40, confirmation_days=3))
    mean = MeanReversionStrategy(
        MeanReversionParameters(
            rsi_window=10, bollinger_window=25, max_holding_sessions=45
        )
    )

    assert rotation.required_history == 80
    assert momentum.required_history == 91
    assert dual.required_history == 42
    assert mean.required_history == 45
    registry = StrategyRegistry()
    registry.register("dual_ma", DualMaStrategy)
    assert registry.create("dual_ma", dual.parameters).required_history == 42


def test_multi_instrument_positive_targets_share_the_strategy_sleeve_budget() -> None:
    rising_a = bars([100.0] * 60 + [101.0, 102.0])
    rising_b = bars([80.0] * 60 + [81.0, 82.0])
    dual = DualMaStrategy(DualMaParameters(target_weight=Decimal("0.8")))
    dual_decision = dual.generate_targets(context({ETF_A: rising_a, ETF_B: rising_b}))
    assert [intent.target_weight for intent in dual_decision] == [Decimal("0.4")] * 2
    assert sum((intent.target_weight for intent in dual_decision), Decimal("0")) == Decimal("0.8")
    assert dual_decision.details["active_count"] == 2

    entry_a = bars([100.0] * 20 + [98.0, 95.0, 90.0])
    entry_b = bars([80.0] * 20 + [78.0, 75.0, 70.0])
    mean = MeanReversionStrategy(
        MeanReversionParameters(
            rsi_window=3,
            bollinger_window=5,
            bollinger_std=1.5,
            max_holding_sessions=5,
            target_weight=Decimal("0.6"),
        )
    )
    mean_decision = mean.generate_targets(context({ETF_A: entry_a, ETF_B: entry_b}))
    assert [intent.target_weight for intent in mean_decision] == [Decimal("0.3")] * 2
    assert mean_decision.details["active_count"] == 2


def test_zero_quantity_holding_does_not_receive_turnover_buffer_protection() -> None:
    winner = bars([100.0 + day for day in range(30)])
    runner_up = bars([100.0 + day * 0.5 for day in range(30)])
    zero_holding = HoldingSummary(ETF_B, 0, 0, Decimal("0"), Decimal("0"))
    strategy = CrossSectionalMomentumStrategy(
        CrossSectionalMomentumParameters(
            lookbacks=(5, 10),
            top_n=1,
            turnover_buffer=1,
            rebalance_frequency=RebalanceFrequency.DAILY,
        )
    )

    decision = strategy.generate_targets(
        context({ETF_A: winner, ETF_B: runner_up}, holdings=(zero_holding,))
    )

    assert [intent.instrument for intent in decision] == [ETF_A]


def test_mean_reversion_holding_age_counts_visible_sessions_inclusively() -> None:
    frame = bars([100.0, 99.0, 98.0, 97.0, 96.0, 95.0])
    strategy = MeanReversionStrategy(
        MeanReversionParameters(
            rsi_window=3,
            bollinger_window=5,
            max_holding_sessions=3,
            stop_loss=0.5,
        )
    )
    two_sessions = HoldingSummary(
        STOCK, 1, 1, Decimal("100"), Decimal("95"), frame.index[-2].date()
    )
    three_sessions = HoldingSummary(
        STOCK, 1, 1, Decimal("100"), Decimal("95"), frame.index[-3].date()
    )

    assert strategy.generate_targets(
        context({STOCK: frame}, holdings=(two_sessions,))
    )[0].reason_code != "MEAN_REVERSION_MAX_HOLDING"
    assert strategy.generate_targets(
        context({STOCK: frame}, holdings=(three_sessions,))
    )[0].reason_code == "MEAN_REVERSION_MAX_HOLDING"

    before_history = HoldingSummary(
        STOCK,
        1,
        1,
        Decimal("100"),
        Decimal("95"),
        holding_since=date(2020, 1, 1),
    )
    assert strategy.generate_targets(
        context({STOCK: frame}, holdings=(before_history,))
    )[0].reason_code == "MEAN_REVERSION_MAX_HOLDING"


def test_strategy_ids_are_exact_normalized_nonempty_strings() -> None:
    assert DualMaStrategy(strategy_id="  sleeve-a  ").strategy_id == "sleeve-a"
    with pytest.raises(TypeError, match="exact str"):
        DualMaStrategy(strategy_id=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        MeanReversionStrategy(strategy_id="  ")


def test_multi_asset_future_mutation_and_output_attributes_are_stable() -> None:
    base_a = bars([100.0 + day * 0.3 for day in range(130)])
    base_b = bars([90.0 + day * 0.2 for day in range(130)])
    as_of = base_a.index[-1].date()
    future_a = base_a.copy()
    future_b = base_b.copy()
    future_day = base_a.index[-1] + pd.offsets.BDay()
    future_a.loc[future_day] = [1.0, 1.01, 0.99, 1.0, 1.0, 1.0]
    future_b.loc[future_day] = [1_000_000.0, 1_010_000.0, 990_000.0, 1_000_000.0, 1.0, 1.0]
    strategies = (
        EtfRotationStrategy(EtfRotationParameters(rebalance_frequency=RebalanceFrequency.DAILY)),
        DualMaStrategy(),
        CrossSectionalMomentumStrategy(
            CrossSectionalMomentumParameters(rebalance_frequency=RebalanceFrequency.DAILY)
        ),
        MeanReversionStrategy(),
    )

    for strategy in strategies:
        before = strategy.generate_targets(context({ETF_A: base_a, ETF_B: base_b}, as_of=as_of))
        after = strategy.generate_targets(
            context({ETF_A: future_a, ETF_B: future_b}, as_of=as_of)
        )
        assert before == after
        for intent in before:
            assert intent.target_weight.is_finite()
            weight_to_units(intent.target_weight, label="strategy intent weight")
            assert 0 <= intent.score < float("inf") or -float("inf") < intent.score < 0
            assert 0.0 <= intent.confidence <= 1.0
            assert intent.valid_until == as_of


def test_equal_weights_use_fixed_units_and_canonical_largest_remainders() -> None:
    assert _equal_weights(3) == (
        Decimal("0.333333333334"),
        Decimal("0.333333333333"),
        Decimal("0.333333333333"),
    )
    assert sum(_equal_weights(3), Decimal("0")) == Decimal("1")
    assert _equal_weights(3, Decimal("0.2")) == (
        Decimal("0.066666666667"),
        Decimal("0.066666666667"),
        Decimal("0.066666666666"),
    )


def test_etf_score_weights_use_fixed_units_and_stable_score_remainders() -> None:
    selected = [(ETF_C, 1.0), (ETF_A, 1.0), (ETF_B, 1.0)]

    assert _score_weights(selected) == (
        Decimal("0.333333333333"),
        Decimal("0.333333333334"),
        Decimal("0.333333333333"),
    )
    assert sum(_score_weights(selected), Decimal("0")) == Decimal("1")


@pytest.mark.parametrize(
    ("parameters_type", "kwargs"),
    [
        (DualMaParameters, {"target_weight": Decimal("0.1234567890123")}),
        (MeanReversionParameters, {"target_weight": Decimal("0.1234567890123")}),
    ],
)
def test_strategy_target_weight_rejects_more_than_twelve_decimal_places(
    parameters_type: type[DualMaParameters] | type[MeanReversionParameters],
    kwargs: dict[str, Decimal],
) -> None:
    with pytest.raises(ValidationError, match="12 decimal places"):
        parameters_type(**kwargs)
