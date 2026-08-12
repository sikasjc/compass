from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta
from decimal import Decimal, localcontext
from types import MappingProxyType
from zoneinfo import ZoneInfo

import pytest

from compass.domain.market import AssetType, Exchange, Instrument, InstrumentId
from compass.domain.weights import units_to_weight, weight_to_units
from compass.risk.base import (
    RiskAdjustment,
    RiskContext,
    RiskSeverity,
    RiskStage,
    RiskTarget,
)
from compass.risk.engine import RiskEngine
from compass.risk.rules import (
    AvailableCashRule,
    AvailableSellQuantityRule,
    CashReserveRule,
    ConsecutiveLossCooldownRule,
    DataValidityRule,
    DrawdownRule,
    LiquidityRule,
    LotSizeRule,
    MinimumTradeAmountRule,
    PriceConstraintRule,
    SingleEtfCapRule,
    SingleStockCapRule,
    StaleDataRule,
    StopLossRule,
    StrategyBudgetRule,
    TakeProfitRule,
    TradabilityRule,
)


SHANGHAI = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 7, 21, 15, 0, tzinfo=SHANGHAI)


def _instrument(asset_type: AssetType = AssetType.STOCK) -> Instrument:
    code = "600000" if asset_type is AssetType.STOCK else "510300"
    return Instrument(InstrumentId.parse(f"SSE.{code}"), asset_type, 100, False)


def _target(
    weight: Decimal = Decimal("0.50"),
    *,
    current_weight: Decimal = Decimal("0"),
    asset_type: AssetType = AssetType.STOCK,
) -> RiskTarget:
    return RiskTarget(
        instrument=_instrument(asset_type),
        requested_weight=weight,
        current_weight=current_weight,
        strategy_id="main",
    )


def _context(**overrides: object) -> RiskContext:
    values: dict[str, object] = {
        "as_of": NOW,
        "data_as_of": NOW - timedelta(minutes=1),
        "data_valid": True,
        "tradable": True,
    }
    values.update(overrides)
    return RiskContext(**values)  # type: ignore[arg-type]


class _OneUnitOrderRule:
    code = "ONE_UNIT_STEP"
    stage = RiskStage.ORDER
    priority = 1
    enabled = True

    def evaluate(
        self, context: RiskContext, target: RiskTarget, current_weight: Decimal
    ) -> RiskAdjustment | None:
        del context
        current_units = weight_to_units(current_weight, label="current weight")
        reference_units = weight_to_units(target.current_weight, label="reference weight")
        if current_units == reference_units:
            return None
        after_units = current_units - 1 if current_units > reference_units else current_units + 1
        return RiskAdjustment(
            code=self.code,
            stage=self.stage,
            severity=RiskSeverity.ADJUST,
            before_weight=current_weight,
            after_weight=units_to_weight(after_units),
            reference_weight=target.current_weight,
            message="Take one fixed-point step toward the holding.",
        )


class _OrderInfoRule:
    code = "ORDER_NOTE"
    stage = RiskStage.ORDER
    priority = 1
    enabled = True

    def evaluate(
        self, context: RiskContext, target: RiskTarget, current_weight: Decimal
    ) -> RiskAdjustment:
        del context
        return RiskAdjustment(
            code=self.code,
            stage=self.stage,
            severity=RiskSeverity.INFO,
            before_weight=current_weight,
            after_weight=current_weight,
            reference_weight=target.current_weight,
            message="Order candidate inspected.",
        )


def test_risk_engine_keeps_exact_cap_chain_and_immutable_trace() -> None:
    engine = RiskEngine(
        (
            CashReserveRule(minimum_cash_weight=Decimal("0.10"), priority=20),
            SingleStockCapRule(maximum_weight=Decimal("0.20"), priority=10),
        )
    )

    result = engine.apply(_context(other_invested_weight=Decimal("0.80")), _target())

    assert result.requested_weight == Decimal("0.50")
    assert result.final_weight == Decimal("0.10")
    assert not result.blocked
    assert [item.code for item in result.adjustments] == [
        "SINGLE_STOCK_CAP",
        "CASH_RESERVE",
    ]
    assert all(item.before_weight >= item.after_weight for item in result.adjustments)
    assert result.adjustments[1].before_weight == result.adjustments[0].after_weight
    assert isinstance(result.adjustments, tuple)
    with pytest.raises(FrozenInstanceError):
        result.final_weight = Decimal("1")  # type: ignore[misc]


def test_cap_prevents_increase_when_current_holding_is_already_over_limit() -> None:
    result = RiskEngine((SingleStockCapRule(maximum_weight=Decimal("0.50")),)).apply(
        _context(),
        _target(Decimal("0.80"), current_weight=Decimal("0.60")),
    )

    assert result.final_weight == Decimal("0.60")
    assert [item.code for item in result.adjustments] == ["SINGLE_STOCK_CAP"]
    assert result.adjustments[0].before_weight == Decimal("0.80")
    assert result.adjustments[0].after_weight == Decimal("0.60")

    ordinary_sell = RiskEngine((SingleStockCapRule(maximum_weight=Decimal("0.50")),)).apply(
        _context(),
        _target(Decimal("0.40"), current_weight=Decimal("0.60")),
    )
    assert ordinary_sell.final_weight == Decimal("0.40")
    assert ordinary_sell.adjustments == ()

    ordinary_cap = RiskEngine((SingleStockCapRule(maximum_weight=Decimal("0.50")),)).apply(
        _context(),
        _target(Decimal("0.80"), current_weight=Decimal("0.20")),
    )
    assert ordinary_cap.final_weight == Decimal("0.50")
    assert [item.code for item in ordinary_cap.adjustments] == ["SINGLE_STOCK_CAP"]


def test_rules_sort_by_stage_priority_and_code_regardless_of_input_order() -> None:
    engine = RiskEngine(
        (
            PriceConstraintRule(priority=1),
            SingleEtfCapRule(maximum_weight=Decimal("0.20"), priority=20),
            StrategyBudgetRule(maximum_weight=Decimal("0.40"), priority=1),
            SingleStockCapRule(maximum_weight=Decimal("0.30"), priority=10),
            DataValidityRule(priority=20),
            TradabilityRule(priority=10),
        )
    )

    assert [rule.code for rule in engine.rules] == [
        "NON_TRADABLE",
        "DATA_INVALID",
        "STRATEGY_BUDGET",
        "SINGLE_STOCK_CAP",
        "SINGLE_ETF_CAP",
        "PRICE_CONSTRAINT",
    ]


def test_engine_rejects_duplicate_rule_codes() -> None:
    with pytest.raises(ValueError, match="duplicate risk rule"):
        RiskEngine(
            (
                SingleStockCapRule(maximum_weight=Decimal("0.20"), priority=1),
                SingleStockCapRule(maximum_weight=Decimal("0.10"), priority=2),
            )
        )


@pytest.mark.parametrize(
    "rule",
    [
        DrawdownRule(),
        StopLossRule(),
        TakeProfitRule(),
        ConsecutiveLossCooldownRule(),
        DrawdownRule(threshold=Decimal("0.10"), maximum_weight=Decimal("0.20")),
    ],
)
def test_optional_risk_rules_do_nothing_unless_explicitly_enabled(rule: object) -> None:
    result = RiskEngine((rule,)).apply(  # type: ignore[arg-type]
        _context(
            drawdown=Decimal("0.50"),
            unrealized_loss=Decimal("0.50"),
            unrealized_gain=Decimal("0.50"),
            consecutive_losses=10,
        ),
        _target(),
    )

    assert result.final_weight == Decimal("0.50")
    assert result.adjustments == ()


def test_enabling_optional_rule_requires_user_supplied_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        StopLossRule(enabled=True)
    with pytest.raises(ValueError, match="threshold"):
        DrawdownRule(enabled=True)


def test_stale_data_blocks_and_later_rules_cannot_restore_weight() -> None:
    engine = RiskEngine(
        (
            CashReserveRule(minimum_cash_weight=Decimal("0.10")),
            StaleDataRule(maximum_age=timedelta(minutes=5)),
        )
    )

    result = engine.apply(
        _context(data_as_of=NOW - timedelta(minutes=6)),
        _target(Decimal("0.20")),
    )

    assert result.blocked
    assert result.final_weight == Decimal("0")
    assert result.adjustments[0].code == "STALE_DATA"
    assert result.adjustments[0].severity is RiskSeverity.BLOCK
    assert result.adjustments[0].stage is RiskStage.DATA_INSTRUMENT


@pytest.mark.parametrize(
    ("rule", "context_overrides", "code"),
    [
        (DataValidityRule(), {"data_valid": False}, "DATA_INVALID"),
        (TradabilityRule(), {"tradable": False}, "NON_TRADABLE"),
    ],
)
def test_invalid_data_and_non_tradable_instrument_fail_closed(
    rule: object, context_overrides: dict[str, object], code: str
) -> None:
    result = RiskEngine((rule,)).apply(_context(**context_overrides), _target())  # type: ignore[arg-type]

    assert result.blocked
    assert result.final_weight == Decimal("0")
    assert result.adjustments[0].code == code


def test_liquidity_rule_scales_target_by_exact_volume_participation() -> None:
    result = RiskEngine((LiquidityRule(maximum_volume_participation=Decimal("0.25")),)).apply(
        _context(observed_volume=1000, expected_order_quantity=1000),
        _target(Decimal("0.40")),
    )

    assert result.final_weight == Decimal("0.10")
    assert result.adjustments[0].code == "LIQUIDITY_LIMIT"
    assert result.adjustments[0].severity is RiskSeverity.ADJUST


def test_liquidity_rule_constrains_sell_delta_toward_actual_holding() -> None:
    result = RiskEngine((LiquidityRule(maximum_volume_participation=Decimal("0.50")),)).apply(
        _context(observed_volume=200, expected_order_quantity=200),
        _target(Decimal("0.20"), current_weight=Decimal("0.60")),
    )

    assert result.final_weight == Decimal("0.40")
    assert result.adjustments[0].before_weight == Decimal("0.20")
    assert result.adjustments[0].after_weight == Decimal("0.40")


def test_zero_liquidity_blocks_a_nonzero_order() -> None:
    result = RiskEngine((LiquidityRule(maximum_volume_participation=Decimal("0.10")),)).apply(
        _context(observed_volume=0, expected_order_quantity=100),
        _target(Decimal("0.40")),
    )

    assert result.blocked
    assert result.final_weight == Decimal("0")


def test_lot_size_rule_constrains_sell_delta_toward_actual_holding() -> None:
    result = RiskEngine((LotSizeRule(),)).apply(
        _context(expected_order_quantity=150, lot_size=100),
        _target(Decimal("0.20"), current_weight=Decimal("0.50")),
    )

    assert result.final_weight == Decimal("0.30")
    assert result.adjustments[0].before_weight == Decimal("0.20")
    assert result.adjustments[0].after_weight == Decimal("0.30")


@pytest.mark.parametrize(
    ("rule", "target"),
    [
        (LiquidityRule(maximum_volume_participation=Decimal("0.10")), _target()),
        (AvailableCashRule(), _target()),
        (
            AvailableSellQuantityRule(),
            _target(Decimal("0.10"), current_weight=Decimal("0.30")),
        ),
        (LotSizeRule(), _target()),
        (MinimumTradeAmountRule(), _target()),
        (PriceConstraintRule(), _target()),
    ],
)
def test_enabled_order_rules_fail_closed_when_required_context_is_missing(
    rule: object, target: RiskTarget
) -> None:
    result = RiskEngine((rule,)).apply(_context(), target)  # type: ignore[arg-type]
    assert result.blocked
    assert result.final_weight == Decimal("0")

    zero_trade = _target(Decimal("0.30"), current_weight=Decimal("0.30"))
    no_op = RiskEngine((rule,)).apply(_context(), zero_trade)  # type: ignore[arg-type]
    assert not no_op.blocked
    assert no_op.adjustments == ()


def test_minimum_amount_uses_quantity_after_upstream_cash_adjustment() -> None:
    result = RiskEngine((AvailableCashRule(), MinimumTradeAmountRule())).apply(
        _context(
            available_cash=Decimal("1000"),
            account_equity=Decimal("10000"),
            expected_order_quantity=300,
            reference_price=Decimal("10"),
            minimum_trade_amount=Decimal("1200"),
        ),
        _target(Decimal("0.30")),
    )

    assert result.blocked
    assert [item.code for item in result.adjustments] == [
        "AVAILABLE_CASH",
        "MINIMUM_TRADE_AMOUNT",
    ]
    assert result.adjustments[1].before_weight == Decimal("0.10")


def test_order_stage_revalidates_minimum_amount_after_lower_priority_cash_cap() -> None:
    result = RiskEngine(
        (
            MinimumTradeAmountRule(priority=10),
            AvailableCashRule(priority=20),
        )
    ).apply(
        _context(
            available_cash=Decimal("1000"),
            account_equity=Decimal("10000"),
            expected_order_quantity=300,
            reference_price=Decimal("10"),
            minimum_trade_amount=Decimal("1200"),
        ),
        _target(Decimal("0.30")),
    )

    assert result.blocked
    assert [item.code for item in result.adjustments] == [
        "AVAILABLE_CASH",
        "MINIMUM_TRADE_AMOUNT",
    ]
    assert result.adjustments[1].before_weight == Decimal("0.10")


def test_order_stage_revalidates_lot_size_after_lower_priority_cash_cap() -> None:
    result = RiskEngine(
        (
            LotSizeRule(priority=10),
            AvailableCashRule(priority=20),
        )
    ).apply(
        _context(
            available_cash=Decimal("1500"),
            account_equity=Decimal("10000"),
            expected_order_quantity=300,
            lot_size=100,
        ),
        _target(Decimal("0.30")),
    )

    assert not result.blocked
    assert result.final_weight == Decimal("0.10")
    assert [item.code for item in result.adjustments] == ["AVAILABLE_CASH", "LOT_SIZE"]


def test_order_stage_bounds_non_converging_adjustments() -> None:
    with pytest.raises(RuntimeError, match="did not converge"):
        RiskEngine((_OneUnitOrderRule(),)).apply(_context(), _target(Decimal("0.30")))


def test_order_stage_does_not_duplicate_informational_trace_on_restart() -> None:
    result = RiskEngine((_OrderInfoRule(), AvailableCashRule(priority=20))).apply(
        _context(available_cash=Decimal("1000"), account_equity=Decimal("10000")),
        _target(Decimal("0.30")),
    )

    assert [item.code for item in result.adjustments] == ["ORDER_NOTE", "AVAILABLE_CASH"]


def test_available_cash_ratio_is_independent_of_decimal_context_precision() -> None:
    engine = RiskEngine((AvailableCashRule(),))
    context = _context(available_cash=Decimal("1"), account_equity=Decimal("3"))
    target = _target(Decimal("0.90"))

    with localcontext() as decimal_context:
        decimal_context.prec = 6
        low_precision = engine.apply(context, target).final_weight
    with localcontext() as decimal_context:
        decimal_context.prec = 50
        high_precision = engine.apply(context, target).final_weight

    assert low_precision == high_precision == Decimal("0.333333333333")


def test_order_stage_hooks_cover_cash_sell_lot_minimum_and_price_constraints() -> None:
    buy = RiskEngine(
        (
            AvailableCashRule(),
            LotSizeRule(),
            MinimumTradeAmountRule(),
            PriceConstraintRule(),
        )
    ).apply(
        _context(
            available_cash=Decimal("1000"),
            account_equity=Decimal("10000"),
            expected_order_quantity=300,
            lot_size=100,
            reference_price=Decimal("10"),
            minimum_trade_amount=Decimal("500"),
            price_constraint_ok=True,
        ),
        _target(Decimal("0.30")),
    )
    assert buy.final_weight == Decimal("0.10")
    assert [item.code for item in buy.adjustments] == ["AVAILABLE_CASH"]

    sell = RiskEngine((AvailableSellQuantityRule(),)).apply(
        _context(expected_order_quantity=200, available_sell_quantity=100),
        _target(Decimal("0.10"), current_weight=Decimal("0.30")),
    )
    assert sell.blocked
    assert sell.adjustments[0].code == "AVAILABLE_SELL_QUANTITY"

    below_minimum = RiskEngine((MinimumTradeAmountRule(),)).apply(
        _context(
            expected_order_quantity=10,
            reference_price=Decimal("10"),
            minimum_trade_amount=Decimal("500"),
        ),
        _target(Decimal("0.10")),
    )
    assert below_minimum.blocked

    price_block = RiskEngine((PriceConstraintRule(),)).apply(
        _context(price_constraint_ok=False), _target(Decimal("0.10"))
    )
    assert price_block.blocked


def test_context_defensively_copies_metadata() -> None:
    metadata = {"source": "fixture"}
    context = _context(metadata=metadata)
    metadata["source"] = "mutated"

    assert context.metadata == MappingProxyType({"source": "fixture"})
    with pytest.raises(TypeError):
        context.metadata["source"] = "mutated"  # type: ignore[index]


def test_public_weights_reject_more_than_twelve_decimal_places() -> None:
    with pytest.raises(ValueError, match="12 decimal places"):
        _target(Decimal("0.1234567890123"))
    with pytest.raises(TypeError, match="exact Decimal"):
        _target(0.1)  # type: ignore[arg-type]


def test_target_rejects_malformed_nested_instrument_metadata() -> None:
    malformed = Instrument(  # type: ignore[arg-type]
        instrument_id="SSE.600000", asset_type="stock", lot_size=100, same_day_sell=False
    )
    with pytest.raises(TypeError, match="instrument id"):
        RiskTarget(
            instrument=malformed,
            requested_weight=Decimal("0.10"),
            strategy_id="main",
        )


@pytest.mark.parametrize(
    "instrument_id",
    [
        InstrumentId(exchange="SSE", code="600000"),  # type: ignore[arg-type]
        InstrumentId.parse("SSE.600000").__class__(
            exchange=InstrumentId.parse("SSE.600000").exchange,
            code="60000X",
        ),
        InstrumentId(exchange=Exchange.SSE, code="１２３４５６"),
    ],
)
def test_target_rejects_malformed_nested_instrument_identifier(
    instrument_id: InstrumentId,
) -> None:
    malformed = Instrument(instrument_id, AssetType.STOCK, 100, False)
    with pytest.raises((TypeError, ValueError), match="instrument"):
        RiskTarget(
            instrument=malformed,
            requested_weight=Decimal("0.10"),
            strategy_id="main",
        )


def test_context_requires_timezone_aware_timestamps_and_exact_numbers() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _context(as_of=datetime(2026, 7, 21, 15, 0))
    with pytest.raises(TypeError, match="exact Decimal"):
        _context(other_invested_weight=0.2)


def test_context_rejects_zero_reference_price() -> None:
    with pytest.raises(ValueError, match="reference_price must be positive"):
        _context(reference_price=Decimal("0"))


def test_adjustment_rejects_a_no_op_audit_row() -> None:
    with pytest.raises(ValueError, match="must change"):
        RiskAdjustment(
            code="NO_OP",
            stage=RiskStage.ORDER,
            severity=RiskSeverity.ADJUST,
            before_weight=Decimal("0.20"),
            after_weight=Decimal("0.20"),
            reference_weight=Decimal("0"),
            message="This row changes nothing.",
        )
