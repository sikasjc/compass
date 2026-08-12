from datetime import date, datetime
from decimal import Decimal

import pandas as pd
import pytest
from pydantic import Field, ValidationError

from compass.domain.market import AssetType, InstrumentId
from compass.domain.trading import TargetIntent
from compass.strategies.base import (
    HoldingSummary,
    Strategy,
    StrategyContext,
    StrategyDecision,
    StrategyDecisionStatus,
    StrategyFrequency,
    StrategyMetadata,
    StrategyParameters,
)
from compass.strategies.registry import StrategyRegistry


INSTRUMENT = InstrumentId.parse("SSE.510300")


class DemoParameters(StrategyParameters):
    lookback: int = Field(default=20, gt=0, description="Trailing daily-bar lookback.")


DEMO_METADATA = StrategyMetadata(
    strategy_type="dual_ma",
    version="1.0.0",
    display_name="Dual moving average",
    description="A daily dual-moving-average trend template.",
    supported_asset_types=frozenset({AssetType.ETF, AssetType.STOCK}),
    supported_frequencies=frozenset({StrategyFrequency.DAILY}),
    required_fields=frozenset({"close"}),
    minimum_history=20,
    default_required_history=20,
    parameters_type=DemoParameters,
)


class DemoStrategy:
    strategy_type = "dual_ma"
    parameters_type = DemoParameters
    minimum_history = 20
    required_history = 20
    metadata = DEMO_METADATA

    def generate_targets(self, context: StrategyContext) -> list[TargetIntent]:
        return []


def factory() -> DemoStrategy:
    return DemoStrategy()


factory.metadata = DEMO_METADATA  # type: ignore[attr-defined]


def daily_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
            "volume": [100.0, 100.0, 100.0],
            "amount": [1050.0, 1150.0, 1250.0],
        },
        index=pd.date_range("2026-01-01", periods=3),
    )


def test_registry_rejects_duplicate_strategy_type() -> None:
    registry = StrategyRegistry()
    registry.register("dual_ma", factory)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("dual_ma", factory)


def test_registry_has_deterministic_lookup_and_validates_created_strategy() -> None:
    registry = StrategyRegistry()
    registry.register("dual_ma", factory)

    assert registry.strategy_types() == ("dual_ma",)
    assert registry.lookup("dual_ma") is factory
    assert isinstance(registry.create("dual_ma"), Strategy)


def test_registry_rejects_invalid_and_unknown_types() -> None:
    registry = StrategyRegistry()

    with pytest.raises(ValueError, match="lower snake"):
        registry.register("Dual-MA", factory)
    with pytest.raises(KeyError, match="unknown strategy type"):
        registry.lookup("dual_ma")


def test_registry_fails_fast_when_factory_output_is_not_registered_strategy_type() -> None:
    registry = StrategyRegistry()

    with pytest.raises(ValueError, match="does not match"):
        registry.register("wrong", factory)


def test_registry_describes_metadata_without_instantiating_the_factory() -> None:
    calls = 0

    def counting_factory() -> DemoStrategy:
        nonlocal calls
        calls += 1
        return DemoStrategy()

    counting_factory.metadata = DEMO_METADATA  # type: ignore[attr-defined]
    registry = StrategyRegistry()
    registry.register("dual_ma", counting_factory)

    assert calls == 0
    assert registry.describe("dual_ma") == DEMO_METADATA
    assert registry.list_metadata() == (DEMO_METADATA,)
    assert calls == 0


def test_registry_rejects_missing_metadata_and_non_callable_generate_targets() -> None:
    registry = StrategyRegistry()

    with pytest.raises(TypeError, match="metadata"):
        registry.register("missing", lambda: DemoStrategy())

    class NonCallableStrategy:
        strategy_type = "dual_ma"
        parameters_type = DemoParameters
        minimum_history = 20
        required_history = 20
        metadata = DEMO_METADATA
        generate_targets = []

    def invalid_factory() -> NonCallableStrategy:
        return NonCallableStrategy()

    invalid_factory.metadata = DEMO_METADATA  # type: ignore[attr-defined]
    registry.register("dual_ma", invalid_factory)
    with pytest.raises(TypeError, match="generate_targets"):
        registry.create("dual_ma")


def test_registry_rejects_parameter_models_without_final_safe_documented_config() -> None:
    class UnsafeParameters(StrategyParameters):
        model_config = {"frozen": False, "extra": "allow"}
        lookback: int = 20

    unsafe_metadata = StrategyMetadata(
        strategy_type="unsafe",
        version="1.0.0",
        display_name="Unsafe",
        description="Unsafe parameters for a negative test.",
        supported_asset_types=frozenset({AssetType.ETF}),
        supported_frequencies=frozenset({StrategyFrequency.DAILY}),
        required_fields=frozenset({"close"}),
        minimum_history=20,
        default_required_history=20,
        parameters_type=UnsafeParameters,
    )

    def unsafe_factory() -> DemoStrategy:
        return DemoStrategy()

    unsafe_factory.metadata = unsafe_metadata  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="frozen"):
        StrategyRegistry().register("unsafe", unsafe_factory)


def test_registry_rejects_equality_lookalikes_before_comparing_strategy_contracts() -> None:
    one_metadata = StrategyMetadata(
        strategy_type="one",
        version="1.0.0",
        display_name="One",
        description="A single-period strategy used for contract testing.",
        supported_asset_types=frozenset({AssetType.ETF}),
        supported_frequencies=frozenset({StrategyFrequency.DAILY}),
        required_fields=frozenset({"close"}),
        minimum_history=1,
        default_required_history=1,
        parameters_type=DemoParameters,
    )

    class NameLookalike(str):
        pass

    class MetadataLookalike:
        def __eq__(self, other: object) -> bool:
            return True

    class FakeStrategy:
        strategy_type = NameLookalike("one")
        metadata = MetadataLookalike()
        parameters_type = DemoParameters
        minimum_history = True
        required_history = 1

        def generate_targets(self, context: StrategyContext) -> list[TargetIntent]:
            return []

    def fake_factory() -> FakeStrategy:
        return FakeStrategy()

    fake_factory.metadata = one_metadata  # type: ignore[attr-defined]
    registry = StrategyRegistry()
    registry.register("one", fake_factory)

    with pytest.raises(TypeError, match="strategy_type"):
        registry.create("one")

    FakeStrategy.strategy_type = "one"
    with pytest.raises(TypeError, match="metadata"):
        registry.create("one")

    FakeStrategy.metadata = one_metadata
    with pytest.raises(ValueError, match="minimum_history"):
        registry.create("one")


def test_registry_requires_exact_strategy_metadata_at_registration_and_creation() -> None:
    class EqualMetadataSubclass(StrategyMetadata):
        def __eq__(self, other: object) -> bool:
            return True

    subclass_metadata = EqualMetadataSubclass(
        strategy_type="dual_ma",
        version="1.0.0",
        display_name="Dual moving average",
        description="A daily dual-moving-average trend template.",
        supported_asset_types=frozenset({AssetType.ETF}),
        supported_frequencies=frozenset({StrategyFrequency.DAILY}),
        required_fields=frozenset({"close"}),
        minimum_history=20,
        default_required_history=20,
        parameters_type=DemoParameters,
    )

    def subclass_factory() -> DemoStrategy:
        return DemoStrategy()

    subclass_factory.metadata = subclass_metadata  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="exact StrategyMetadata"):
        StrategyRegistry().register("dual_ma", subclass_factory)

    class SubclassMetadataStrategy(DemoStrategy):
        metadata = subclass_metadata

    def instance_factory() -> SubclassMetadataStrategy:
        return SubclassMetadataStrategy()

    instance_factory.metadata = DEMO_METADATA  # type: ignore[attr-defined]
    registry = StrategyRegistry()
    registry.register("dual_ma", instance_factory)
    with pytest.raises(TypeError, match="exact StrategyMetadata"):
        registry.create("dual_ma")


def test_metadata_rejects_fields_not_preserved_by_the_daily_context() -> None:
    with pytest.raises(ValueError, match="required_fields"):
        StrategyMetadata(
            strategy_type="invalid_fields",
            version="1.0.0",
            display_name="Invalid fields",
            description="Reject fields that daily bars cannot retain.",
            supported_asset_types=frozenset({AssetType.ETF}),
            supported_frequencies=frozenset({StrategyFrequency.DAILY}),
            required_fields=frozenset({"unavailable_field"}),
            minimum_history=20,
            default_required_history=20,
            parameters_type=DemoParameters,
        )


def test_strategy_context_hides_future_data_and_defensively_copies_bars() -> None:
    source = daily_bars()
    context = StrategyContext(
        as_of=date(2026, 1, 2),
        bars={INSTRUMENT: source},
        instruments=[INSTRUMENT],
        account_equity=Decimal("10000"),
    )
    source.loc[:, "close"] = 0.0
    visible_bars = context.bars[INSTRUMENT]
    visible_bars.loc[:, "close"] = 0.0
    history = context.history(INSTRUMENT)
    history.loc[:, "close"] = 0.0

    assert context.history(INSTRUMENT).index.max() == pd.Timestamp("2026-01-02")
    assert context.history(INSTRUMENT)["close"].tolist() == [10.5, 11.5]
    assert context.bars[INSTRUMENT]["close"].tolist() == [10.5, 11.5]


def test_strategy_context_keeps_immutable_cash_and_holding_summaries() -> None:
    holding = HoldingSummary(
        instrument=INSTRUMENT,
        quantity=100,
        available_quantity=50,
        average_cost=Decimal("10.00"),
        mark_price=Decimal("12.00"),
        holding_since=date(2026, 1, 1),
    )
    source_holdings = {INSTRUMENT: holding}
    context = StrategyContext(
        date(2026, 1, 2),
        {INSTRUMENT: daily_bars()},
        [INSTRUMENT],
        Decimal("1200"),
        cash=Decimal("50"),
        holdings=source_holdings,
    )
    source_holdings.clear()

    assert context.cash == Decimal("50")
    assert context.holdings[INSTRUMENT] == holding
    with pytest.raises(TypeError):
        context.holdings[INSTRUMENT] = holding  # type: ignore[index]


def test_strategy_context_rejects_a_holding_start_date_after_as_of() -> None:
    future_holding = HoldingSummary(
        INSTRUMENT,
        1,
        1,
        Decimal("1"),
        Decimal("1"),
        holding_since=date(2026, 1, 3),
    )

    with pytest.raises(ValueError, match="holding_since"):
        StrategyContext(
            date(2026, 1, 2),
            {INSTRUMENT: daily_bars()},
            [INSTRUMENT],
            Decimal("1"),
            holdings={INSTRUMENT: future_holding},
        )


def test_strategy_context_ignores_invalid_future_rows_but_validates_visible_daily_rows() -> None:
    frame = pd.concat([daily_bars(), daily_bars().iloc[[-1]]])
    frame.iloc[2:, frame.columns.get_loc("high")] = 1.0
    context = StrategyContext(date(2026, 1, 2), {INSTRUMENT: frame}, [INSTRUMENT], Decimal("1"))

    assert len(context.history(INSTRUMENT)) == 2


def test_strategy_context_ignores_future_intraday_dates_but_rejects_visible_intraday_dates() -> None:
    future_intraday = daily_bars()
    future_intraday.index = pd.DatetimeIndex(
        ["2026-01-01", "2026-01-02", "2026-01-03 09:30"]
    )
    context = StrategyContext(
        date(2026, 1, 2), {INSTRUMENT: future_intraday}, [INSTRUMENT], Decimal("1")
    )
    assert len(context.history(INSTRUMENT)) == 2

    visible_intraday = future_intraday.copy()
    visible_intraday.index = pd.DatetimeIndex(
        ["2026-01-01", "2026-01-02 09:30", "2026-01-03"]
    )
    with pytest.raises(ValueError, match="midnight"):
        StrategyContext(
            date(2026, 1, 2), {INSTRUMENT: visible_intraday}, [INSTRUMENT], Decimal("1")
        )


def test_strategy_context_rejects_holdings_outside_its_instrument_universe() -> None:
    other = InstrumentId.parse("SSE.510301")
    holding = HoldingSummary(other, 1, 1, Decimal("1"), Decimal("1"))

    with pytest.raises(ValueError, match="holdings"):
        StrategyContext(
            date(2026, 1, 2),
            {INSTRUMENT: daily_bars()},
            [INSTRUMENT],
            Decimal("1"),
            holdings={other: holding},
        )


def test_strategy_context_requires_daily_bars_for_exact_unique_instruments() -> None:
    with pytest.raises(ValueError, match="unique"):
        StrategyContext(date(2026, 1, 2), {INSTRUMENT: daily_bars()}, [INSTRUMENT, INSTRUMENT], Decimal("1"))
    with pytest.raises(ValueError, match="exactly"):
        StrategyContext(date(2026, 1, 2), {}, [INSTRUMENT], Decimal("1"))


def test_strategy_context_defensively_copies_exact_asset_type_metadata() -> None:
    source = {INSTRUMENT: AssetType.ETF}
    context = StrategyContext(
        date(2026, 1, 2),
        {INSTRUMENT: daily_bars()},
        [INSTRUMENT],
        Decimal("1"),
        asset_types=source,
    )
    source.clear()

    assert context.asset_types == {INSTRUMENT: AssetType.ETF}
    with pytest.raises(TypeError):
        context.asset_types[INSTRUMENT] = AssetType.STOCK  # type: ignore[index]
    with pytest.raises(ValueError, match="asset_types"):
        StrategyContext(
            date(2026, 1, 2),
            {INSTRUMENT: daily_bars()},
            [INSTRUMENT],
            Decimal("1"),
            asset_types={},
        )


def test_strategy_context_rejects_datetime_unknown_instrument_and_invalid_equity() -> None:
    with pytest.raises(ValueError, match="as_of"):
        StrategyContext(datetime(2026, 1, 2), {INSTRUMENT: daily_bars()}, [INSTRUMENT], Decimal("1"))
    context = StrategyContext(date(2026, 1, 2), {INSTRUMENT: daily_bars()}, [INSTRUMENT], Decimal("1"))
    with pytest.raises(KeyError, match="unknown instrument"):
        context.history(InstrumentId.parse("SSE.510301"))
    with pytest.raises(ValueError, match="equity"):
        StrategyContext(date(2026, 1, 2), {INSTRUMENT: daily_bars()}, [INSTRUMENT], Decimal("NaN"))


def test_strategy_parameters_are_frozen_and_forbid_unknown_fields() -> None:
    parameters = DemoParameters()

    with pytest.raises(ValidationError):
        DemoParameters(extra_value=1)
    with pytest.raises(ValidationError):
        parameters.lookback = 30


def test_registry_accepts_dynamic_required_history_and_rejects_invalid_values() -> None:
    registry = StrategyRegistry()
    registry.register("dual_ma", factory)
    DemoStrategy.required_history = 30
    assert registry.create("dual_ma").required_history == 30

    for invalid in (True, 0, 19):
        DemoStrategy.required_history = invalid  # type: ignore[assignment]
        with pytest.raises(ValueError, match="required_history"):
            registry.create("dual_ma")
    DemoStrategy.required_history = 20


def test_strategy_decision_is_an_immutable_sequence_with_stable_explanation() -> None:
    intent = TargetIntent(
        "demo",
        INSTRUMENT,
        Decimal("0.5"),
        1.0,
        1.0,
        "DEMO",
        date(2026, 1, 2),
    )
    decision = StrategyDecision.generated((intent,), details={"active_count": 1})

    assert isinstance(decision, StrategyDecision)
    assert decision.status is StrategyDecisionStatus.GENERATED
    assert decision.reason_code == "GENERATED"
    assert tuple(decision) == (intent,)
    assert len(decision) == 1
    assert decision[0] == intent
    assert decision.details["active_count"] == 1
    with pytest.raises(TypeError):
        decision.details["active_count"] = 2  # type: ignore[index]


def test_empty_strategy_decision_preserves_distinct_reason_without_fake_intent() -> None:
    decision = StrategyDecision.empty(
        StrategyDecisionStatus.SKIPPED,
        "INSUFFICIENT_HISTORY",
        details={"required_history": 121},
    )

    assert tuple(decision) == ()
    assert decision.reason_code == "INSUFFICIENT_HISTORY"
    assert decision.details == {"required_history": 121}
