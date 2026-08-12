from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from typing import ClassVar

from compass.domain.market import AssetType
from compass.domain.weights import WEIGHT_SCALE, units_to_weight, weight_to_units
from compass.risk.base import (
    RiskAdjustment,
    RiskContext,
    RiskSeverity,
    RiskStage,
    RiskTarget,
)


def _configuration_weight(value: object, *, label: str, positive: bool = False) -> Decimal:
    units = weight_to_units(value, label=label)
    if positive and units == 0:
        raise ValueError(f"{label} must be positive")
    assert isinstance(value, Decimal)
    return value


def _exact_priority(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("risk rule priority must be an exact integer")
    if value < 0:
        raise ValueError("risk rule priority must be non-negative")
    return value


def _adjust(
    rule: _Rule,
    target: RiskTarget,
    before: Decimal,
    after: Decimal,
    message: str,
    *,
    severity: RiskSeverity = RiskSeverity.ADJUST,
) -> RiskAdjustment:
    return RiskAdjustment(
        code=rule.code,
        stage=rule.stage,
        severity=severity,
        before_weight=before,
        after_weight=after,
        reference_weight=target.current_weight,
        message=message,
    )


def _block(
    rule: _Rule, target: RiskTarget, before: Decimal, message: str
) -> RiskAdjustment:
    return _adjust(
        rule,
        target,
        before,
        Decimal("0"),
        message,
        severity=RiskSeverity.BLOCK,
    )


def _cap(
    rule: _Rule,
    target: RiskTarget,
    current: Decimal,
    maximum: Decimal,
    message: str,
) -> RiskAdjustment | None:
    current_units = weight_to_units(current, label="current weight")
    reference_units = weight_to_units(target.current_weight, label="current holding weight")
    maximum_units = weight_to_units(maximum, label="maximum weight")
    if reference_units > maximum_units:
        if current_units <= reference_units:
            return None
        after_units = reference_units
    else:
        after_units = min(current_units, maximum_units)
    after = units_to_weight(after_units)
    if after == current:
        return None
    if abs(after_units - reference_units) > abs(current_units - reference_units):
        return None
    if (current_units - reference_units) * (after_units - reference_units) < 0:
        return None
    return _adjust(rule, target, current, after, message)


def _remaining(cap: Decimal, used: Decimal) -> Decimal:
    return units_to_weight(
        max(
            0,
            weight_to_units(cap, label="cap") - weight_to_units(used, label="used weight"),
        )
    )


def _candidate_order_quantity(
    context: RiskContext, target: RiskTarget, candidate_weight: Decimal
) -> int | None:
    """Map the current candidate delta to quantity using exact fixed-point ratios."""

    expected = context.expected_order_quantity
    if expected is None:
        return None
    holding_units = weight_to_units(target.current_weight, label="current holding weight")
    requested_units = weight_to_units(target.requested_weight, label="requested weight")
    candidate_units = weight_to_units(candidate_weight, label="candidate weight")
    candidate_delta = candidate_units - holding_units
    if candidate_delta == 0:
        return 0
    requested_delta = requested_units - holding_units
    if requested_delta == 0 or requested_delta * candidate_delta < 0:
        raise ValueError("candidate trade direction must match the requested trade")
    return abs(candidate_delta) * expected // abs(requested_delta)


def _candidate_for_quantity(
    context: RiskContext,
    target: RiskTarget,
    candidate_weight: Decimal,
    executable_quantity: int,
) -> Decimal:
    expected = context.expected_order_quantity
    if expected is None or expected <= 0:
        raise ValueError("expected order quantity must be positive")
    holding_units = weight_to_units(target.current_weight, label="current holding weight")
    requested_units = weight_to_units(target.requested_weight, label="requested weight")
    candidate_units = weight_to_units(candidate_weight, label="candidate weight")
    candidate_delta = candidate_units - holding_units
    requested_delta = requested_units - holding_units
    if candidate_delta == 0:
        return target.current_weight
    if requested_delta == 0 or requested_delta * candidate_delta < 0:
        raise ValueError("candidate trade direction must match the requested trade")
    executable_distance = abs(requested_delta) * executable_quantity // expected
    distance = min(abs(candidate_delta), executable_distance)
    signed_distance = distance if candidate_delta > 0 else -distance
    return units_to_weight(holding_units + signed_distance)


def _floor_decimal_ratio_units(numerator: Decimal, denominator: Decimal) -> int:
    numerator_value, numerator_scale = numerator.as_integer_ratio()
    denominator_value, denominator_scale = denominator.as_integer_ratio()
    return (
        numerator_value * denominator_scale * WEIGHT_SCALE
        // (numerator_scale * denominator_value)
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class _Rule:
    priority: int = 100
    enabled: bool = True

    code: ClassVar[str]
    stage: ClassVar[RiskStage]

    def __post_init__(self) -> None:
        _exact_priority(self.priority)
        if type(self.enabled) is not bool:
            raise TypeError("risk rule enabled state must be an exact bool")


@dataclass(frozen=True, slots=True, kw_only=True)
class DataValidityRule(_Rule):
    code: ClassVar[str] = "DATA_INVALID"
    stage: ClassVar[RiskStage] = RiskStage.DATA_INSTRUMENT

    def evaluate(
        self, context: RiskContext, target: RiskTarget, current_weight: Decimal
    ) -> RiskAdjustment | None:
        if context.data_valid:
            return None
        return _block(self, target, current_weight, "Market data failed its quality gate.")


@dataclass(frozen=True, slots=True, kw_only=True)
class TradabilityRule(_Rule):
    code: ClassVar[str] = "NON_TRADABLE"
    stage: ClassVar[RiskStage] = RiskStage.DATA_INSTRUMENT

    def evaluate(
        self, context: RiskContext, target: RiskTarget, current_weight: Decimal
    ) -> RiskAdjustment | None:
        if context.tradable:
            return None
        return _block(self, target, current_weight, "Instrument is not currently tradable.")


@dataclass(frozen=True, slots=True, kw_only=True)
class StaleDataRule(_Rule):
    maximum_age: timedelta

    code: ClassVar[str] = "STALE_DATA"
    stage: ClassVar[RiskStage] = RiskStage.DATA_INSTRUMENT

    def __post_init__(self) -> None:
        super(StaleDataRule, self).__post_init__()
        if type(self.maximum_age) is not timedelta:
            raise TypeError("maximum_age must be an exact timedelta")
        if self.maximum_age <= timedelta(0):
            raise ValueError("maximum_age must be positive")

    def evaluate(
        self, context: RiskContext, target: RiskTarget, current_weight: Decimal
    ) -> RiskAdjustment | None:
        if context.data_as_of is not None and context.as_of - context.data_as_of <= self.maximum_age:
            return None
        return _block(self, target, current_weight, "Market data is missing or stale.")


@dataclass(frozen=True, slots=True, kw_only=True)
class LiquidityRule(_Rule):
    maximum_volume_participation: Decimal

    code: ClassVar[str] = "LIQUIDITY_LIMIT"
    stage: ClassVar[RiskStage] = RiskStage.DATA_INSTRUMENT

    def __post_init__(self) -> None:
        super(LiquidityRule, self).__post_init__()
        _configuration_weight(
            self.maximum_volume_participation,
            label="maximum volume participation",
            positive=True,
        )

    def evaluate(
        self, context: RiskContext, target: RiskTarget, current_weight: Decimal
    ) -> RiskAdjustment | None:
        if current_weight == target.current_weight:
            return None
        volume = context.observed_volume
        quantity = _candidate_order_quantity(context, target, current_weight)
        if volume is None or quantity is None:
            return _block(
                self,
                target,
                current_weight,
                "Volume and expected order quantity are required for liquidity checks.",
            )
        if quantity <= 0:
            return _block(
                self, target, current_weight, "Candidate trade has no executable quantity."
            )
        participation_units = weight_to_units(
            self.maximum_volume_participation, label="maximum volume participation"
        )
        allowed = volume * participation_units // WEIGHT_SCALE
        if allowed <= 0:
            return _block(
                self, target, current_weight, "No executable quantity under the volume cap."
            )
        if quantity <= allowed:
            return None
        after = _candidate_for_quantity(context, target, current_weight, allowed)
        return _adjust(
            self,
            target,
            current_weight,
            after,
            "Trade delta reduced by volume participation.",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class StrategyBudgetRule(_Rule):
    maximum_weight: Decimal

    code: ClassVar[str] = "STRATEGY_BUDGET"
    stage: ClassVar[RiskStage] = RiskStage.STRATEGY

    def __post_init__(self) -> None:
        super(StrategyBudgetRule, self).__post_init__()
        _configuration_weight(self.maximum_weight, label="strategy maximum weight")

    def evaluate(
        self, context: RiskContext, target: RiskTarget, current_weight: Decimal
    ) -> RiskAdjustment | None:
        return _cap(
            self,
            target,
            current_weight,
            _remaining(self.maximum_weight, context.strategy_other_weight),
            "Target reduced to the remaining strategy budget.",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TurnoverCapRule(_Rule):
    maximum_turnover: Decimal

    code: ClassVar[str] = "TURNOVER_CAP"
    stage: ClassVar[RiskStage] = RiskStage.STRATEGY

    def __post_init__(self) -> None:
        super(TurnoverCapRule, self).__post_init__()
        _configuration_weight(self.maximum_turnover, label="maximum turnover")

    def evaluate(
        self, context: RiskContext, target: RiskTarget, current_weight: Decimal
    ) -> RiskAdjustment | None:
        if current_weight <= target.current_weight:
            return None
        remaining = _remaining(self.maximum_turnover, context.turnover_used_weight)
        maximum = units_to_weight(
            min(
                WEIGHT_SCALE,
                weight_to_units(target.current_weight, label="current holding weight")
                + weight_to_units(remaining, label="remaining turnover"),
            )
        )
        return _cap(self, target, current_weight, maximum, "Buy-side turnover cap applied.")


@dataclass(frozen=True, slots=True, kw_only=True)
class SingleStockCapRule(_Rule):
    maximum_weight: Decimal

    code: ClassVar[str] = "SINGLE_STOCK_CAP"
    stage: ClassVar[RiskStage] = RiskStage.PORTFOLIO

    def __post_init__(self) -> None:
        super(SingleStockCapRule, self).__post_init__()
        _configuration_weight(self.maximum_weight, label="single stock maximum weight")

    def evaluate(
        self, context: RiskContext, target: RiskTarget, current_weight: Decimal
    ) -> RiskAdjustment | None:
        if target.instrument.asset_type is not AssetType.STOCK:
            return None
        return _cap(
            self, target, current_weight, self.maximum_weight, "Single-stock cap applied."
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SingleEtfCapRule(_Rule):
    maximum_weight: Decimal

    code: ClassVar[str] = "SINGLE_ETF_CAP"
    stage: ClassVar[RiskStage] = RiskStage.PORTFOLIO

    def __post_init__(self) -> None:
        super(SingleEtfCapRule, self).__post_init__()
        _configuration_weight(self.maximum_weight, label="single ETF maximum weight")

    def evaluate(
        self, context: RiskContext, target: RiskTarget, current_weight: Decimal
    ) -> RiskAdjustment | None:
        if target.instrument.asset_type is not AssetType.ETF:
            return None
        return _cap(self, target, current_weight, self.maximum_weight, "Single-ETF cap applied.")


@dataclass(frozen=True, slots=True, kw_only=True)
class StockExposureCapRule(_Rule):
    maximum_weight: Decimal

    code: ClassVar[str] = "STOCK_EXPOSURE_CAP"
    stage: ClassVar[RiskStage] = RiskStage.PORTFOLIO

    def __post_init__(self) -> None:
        super(StockExposureCapRule, self).__post_init__()
        _configuration_weight(self.maximum_weight, label="stock exposure maximum weight")

    def evaluate(
        self, context: RiskContext, target: RiskTarget, current_weight: Decimal
    ) -> RiskAdjustment | None:
        if target.instrument.asset_type is not AssetType.STOCK:
            return None
        return _cap(
            self,
            target,
            current_weight,
            _remaining(self.maximum_weight, context.other_stock_weight),
            "Stock exposure cap applied.",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CashReserveRule(_Rule):
    minimum_cash_weight: Decimal

    code: ClassVar[str] = "CASH_RESERVE"
    stage: ClassVar[RiskStage] = RiskStage.PORTFOLIO

    def __post_init__(self) -> None:
        super(CashReserveRule, self).__post_init__()
        _configuration_weight(self.minimum_cash_weight, label="minimum cash weight")

    def evaluate(
        self, context: RiskContext, target: RiskTarget, current_weight: Decimal
    ) -> RiskAdjustment | None:
        invested_limit = units_to_weight(
            WEIGHT_SCALE
            - weight_to_units(self.minimum_cash_weight, label="minimum cash weight")
        )
        return _cap(
            self,
            target,
            current_weight,
            _remaining(invested_limit, context.other_invested_weight),
            "Target reduced to preserve the configured cash reserve.",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AvailableCashRule(_Rule):
    code: ClassVar[str] = "AVAILABLE_CASH"
    stage: ClassVar[RiskStage] = RiskStage.ORDER

    def evaluate(
        self, context: RiskContext, target: RiskTarget, current_weight: Decimal
    ) -> RiskAdjustment | None:
        if current_weight <= target.current_weight:
            return None
        if context.available_cash is None or context.account_equity is None:
            return _block(
                self,
                target,
                current_weight,
                "Available cash and account equity are required for buy checks.",
            )
        if context.account_equity == 0:
            return _block(
                self,
                target,
                current_weight,
                "Account equity is zero; buy target cannot be funded.",
            )
        additional_units = _floor_decimal_ratio_units(
            context.available_cash, context.account_equity
        )
        maximum = units_to_weight(
            min(
                WEIGHT_SCALE,
                weight_to_units(target.current_weight, label="current holding weight")
                + additional_units,
            )
        )
        return _cap(self, target, current_weight, maximum, "Target reduced to available cash.")


@dataclass(frozen=True, slots=True, kw_only=True)
class AvailableSellQuantityRule(_Rule):
    code: ClassVar[str] = "AVAILABLE_SELL_QUANTITY"
    stage: ClassVar[RiskStage] = RiskStage.ORDER

    def evaluate(
        self, context: RiskContext, target: RiskTarget, current_weight: Decimal
    ) -> RiskAdjustment | None:
        if current_weight >= target.current_weight:
            return None
        quantity = _candidate_order_quantity(context, target, current_weight)
        if quantity is None or context.available_sell_quantity is None:
            return _block(
                self,
                target,
                current_weight,
                "Expected and available sell quantities are required for sell checks.",
            )
        if quantity <= 0:
            return _block(
                self, target, current_weight, "Candidate sale has no executable quantity."
            )
        if quantity <= context.available_sell_quantity:
            return None
        return _block(
            self, target, current_weight, "Requested sale exceeds available quantity."
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class LotSizeRule(_Rule):
    code: ClassVar[str] = "LOT_SIZE"
    stage: ClassVar[RiskStage] = RiskStage.ORDER

    def evaluate(
        self, context: RiskContext, target: RiskTarget, current_weight: Decimal
    ) -> RiskAdjustment | None:
        if current_weight == target.current_weight:
            return None
        if current_weight < target.current_weight and context.allow_odd_lot_sell:
            return None
        quantity = _candidate_order_quantity(context, target, current_weight)
        lot_size = context.lot_size
        if quantity is None or lot_size is None:
            return _block(
                self,
                target,
                current_weight,
                "Expected order quantity and lot size are required for lot checks.",
            )
        if quantity <= 0:
            return _block(
                self, target, current_weight, "Candidate trade has no executable quantity."
            )
        executable = quantity // lot_size * lot_size
        if executable == quantity:
            return None
        if executable == 0:
            return _block(
                self, target, current_weight, "Order is smaller than one executable lot."
            )
        after = _candidate_for_quantity(
            context, target, current_weight, executable
        )
        return _adjust(
            self,
            target,
            current_weight,
            after,
            "Trade delta reduced to an executable lot size.",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class MinimumTradeAmountRule(_Rule):
    code: ClassVar[str] = "MINIMUM_TRADE_AMOUNT"
    stage: ClassVar[RiskStage] = RiskStage.ORDER

    def evaluate(
        self, context: RiskContext, target: RiskTarget, current_weight: Decimal
    ) -> RiskAdjustment | None:
        if current_weight == target.current_weight:
            return None
        quantity = _candidate_order_quantity(context, target, current_weight)
        if (
            context.minimum_trade_amount is None
            or context.reference_price is None
            or quantity is None
        ):
            return _block(
                self,
                target,
                current_weight,
                "Quantity, reference price, and minimum amount are required.",
            )
        price_numerator, price_denominator = context.reference_price.as_integer_ratio()
        minimum_numerator, minimum_denominator = (
            context.minimum_trade_amount.as_integer_ratio()
        )
        if (
            price_numerator * quantity * minimum_denominator
            >= minimum_numerator * price_denominator
        ):
            return None
        return _block(
            self, target, current_weight, "Order value is below the configured minimum."
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PriceConstraintRule(_Rule):
    code: ClassVar[str] = "PRICE_CONSTRAINT"
    stage: ClassVar[RiskStage] = RiskStage.ORDER

    def evaluate(
        self, context: RiskContext, target: RiskTarget, current_weight: Decimal
    ) -> RiskAdjustment | None:
        if current_weight == target.current_weight:
            return None
        if context.price_constraint_ok is None:
            return _block(
                self,
                target,
                current_weight,
                "Price-constraint state is required for order checks.",
            )
        if context.price_constraint_ok:
            return None
        return _block(
            self,
            target,
            current_weight,
            "Order price violates the configured market constraint.",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DrawdownRule(_Rule):
    threshold: Decimal | None = None
    maximum_weight: Decimal | None = None
    enabled: bool = False

    code: ClassVar[str] = "DRAWDOWN_CAP"
    stage: ClassVar[RiskStage] = RiskStage.PORTFOLIO

    def __post_init__(self) -> None:
        super(DrawdownRule, self).__post_init__()
        if self.threshold is not None:
            _configuration_weight(self.threshold, label="drawdown threshold", positive=True)
        if self.maximum_weight is not None:
            _configuration_weight(self.maximum_weight, label="drawdown maximum weight")
        if self.enabled and (self.threshold is None or self.maximum_weight is None):
            raise ValueError("enabled drawdown rule requires a threshold and maximum weight")

    def evaluate(
        self, context: RiskContext, target: RiskTarget, current_weight: Decimal
    ) -> RiskAdjustment | None:
        if (
            context.drawdown is None
            or self.threshold is None
            or self.maximum_weight is None
            or context.drawdown < self.threshold
        ):
            return None
        return _cap(
            self,
            target,
            current_weight,
            self.maximum_weight,
            "Drawdown exposure cap applied.",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class StopLossRule(_Rule):
    threshold: Decimal | None = None
    enabled: bool = False

    code: ClassVar[str] = "STOP_LOSS"
    stage: ClassVar[RiskStage] = RiskStage.STRATEGY

    def __post_init__(self) -> None:
        super(StopLossRule, self).__post_init__()
        if self.threshold is not None:
            _configuration_weight(self.threshold, label="stop-loss threshold", positive=True)
        if self.enabled and self.threshold is None:
            raise ValueError("enabled stop-loss rule requires a threshold")

    def evaluate(
        self, context: RiskContext, target: RiskTarget, current_weight: Decimal
    ) -> RiskAdjustment | None:
        if (
            context.unrealized_loss is None
            or self.threshold is None
            or context.unrealized_loss < self.threshold
        ):
            return None
        return _block(
            self, target, current_weight, "Configured stop-loss threshold reached."
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TakeProfitRule(_Rule):
    threshold: Decimal | None = None
    enabled: bool = False

    code: ClassVar[str] = "TAKE_PROFIT"
    stage: ClassVar[RiskStage] = RiskStage.STRATEGY

    def __post_init__(self) -> None:
        super(TakeProfitRule, self).__post_init__()
        if self.threshold is not None:
            _configuration_weight(self.threshold, label="take-profit threshold", positive=True)
        if self.enabled and self.threshold is None:
            raise ValueError("enabled take-profit rule requires a threshold")

    def evaluate(
        self, context: RiskContext, target: RiskTarget, current_weight: Decimal
    ) -> RiskAdjustment | None:
        if (
            context.unrealized_gain is None
            or self.threshold is None
            or context.unrealized_gain < self.threshold
        ):
            return None
        return _block(
            self, target, current_weight, "Configured take-profit threshold reached."
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ConsecutiveLossCooldownRule(_Rule):
    loss_count_threshold: int | None = None
    enabled: bool = False

    code: ClassVar[str] = "LOSS_COOLDOWN"
    stage: ClassVar[RiskStage] = RiskStage.STRATEGY

    def __post_init__(self) -> None:
        super(ConsecutiveLossCooldownRule, self).__post_init__()
        if self.loss_count_threshold is not None:
            if isinstance(self.loss_count_threshold, bool) or not isinstance(
                self.loss_count_threshold, int
            ):
                raise TypeError("loss-count threshold must be an exact integer")
            if self.loss_count_threshold <= 0:
                raise ValueError("loss-count threshold must be positive")
        if self.enabled and self.loss_count_threshold is None:
            raise ValueError("enabled loss-cooldown rule requires a threshold")

    def evaluate(
        self, context: RiskContext, target: RiskTarget, current_weight: Decimal
    ) -> RiskAdjustment | None:
        if (
            self.loss_count_threshold is None
            or context.consecutive_losses < self.loss_count_threshold
        ):
            return None
        return _block(
            self, target, current_weight, "Consecutive-loss cooldown is active."
        )
