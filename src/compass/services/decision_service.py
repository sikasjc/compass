from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, localcontext
from enum import StrEnum
from math import isfinite
import re
from types import MappingProxyType
from typing import Protocol
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from compass.backtest.market_rules import (
    MarketRuleBook,
    MarketRuleProfile,
    OddLotSellPolicy,
)
from compass.backtest.effective_rules import effective_price_limits
from compass.domain.market import AssetType, BarFrame, Instrument, InstrumentId
from compass.domain.trading import AccountSnapshot, Position, TargetIntent
from compass.domain.weights import WEIGHT_SCALE, units_to_weight, weight_to_units
from compass.portfolio.allocator import DeterministicAllocator
from compass.portfolio.trace import AllocationAdjustment, AllocationPolicy
from compass.risk.base import RiskAdjustment, RiskContext, RiskResult, RiskTarget
from compass.risk.engine import RiskEngine
from compass.storage.account_repository import AccountRepository, StoredAccountSnapshot
from compass.strategies.base import (
    HoldingSummary,
    StrategyContext,
    StrategyDecision,
    StrategyDecisionStatus,
)


SHANGHAI = ZoneInfo("Asia/Shanghai")
MONEY_QUANTUM = Decimal("0.01")
PRICE_QUANTUM = Decimal("0.0001")
_STABLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")
_REASON_CODE = re.compile(r"[A-Z][A-Z0-9_]*\Z")
_CONTENT_HASH = re.compile(r"[0-9a-f]{64}\Z")


class DecisionDataError(ValueError):
    """The formal close decision failed its market-data gate."""


class _DecisionStrategy(Protocol):
    strategy_id: str

    def generate_targets(self, context: StrategyContext) -> StrategyDecision: ...


class DecisionSide(StrEnum):
    SELL = "sell"
    BUY = "buy"
    NONE = "none"


def _aware_datetime(value: object, *, label: str) -> datetime:
    if type(value) is not datetime:
        raise TypeError(f"{label} must be an exact datetime")
    assert isinstance(value, datetime)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value


def _stable_id(value: object, *, label: str) -> str:
    if type(value) is not str or _STABLE_ID.fullmatch(value) is None:
        raise ValueError(f"{label} must be a stable non-empty identifier")
    assert isinstance(value, str)
    return value


def _exact_date(value: object, *, label: str) -> date:
    if type(value) is not date:
        raise TypeError(f"{label} must be an exact date")
    assert isinstance(value, date)
    return value


def _exact_decimal(value: object, *, label: str, positive: bool = False) -> Decimal:
    if type(value) is not Decimal:
        raise TypeError(f"{label} must be an exact Decimal")
    assert isinstance(value, Decimal)
    if not value.is_finite() or value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be finite and {qualifier}")
    return value


def _cell_decimal(value: object, *, label: str, positive: bool = False) -> Decimal:
    if type(value) is Decimal:
        result = value
    elif isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    elif isinstance(value, (int, float, np.integer, np.floating)):
        result = Decimal(str(value))
    else:
        raise TypeError(f"{label} must be numeric")
    return _exact_decimal(result, label=label, positive=positive)


def _round_money(value: Decimal) -> Decimal:
    _exact_decimal(value, label="money")
    with localcontext() as context:
        context.prec = 50
        return value.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)


def _round_price(value: Decimal) -> Decimal:
    _exact_decimal(value, label="price", positive=True)
    with localcontext() as context:
        context.prec = 50
        result = value.quantize(PRICE_QUANTUM, rounding=ROUND_HALF_UP)
    return _exact_decimal(result, label="rounded price", positive=True)


def _product(left: Decimal, right: Decimal | int) -> Decimal:
    with localcontext() as context:
        context.prec = 50
        return left * right


def _sum(*values: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = 50
        return sum(values, Decimal("0"))


def _ratio_weight(numerator: Decimal, denominator: Decimal) -> Decimal:
    if denominator == 0:
        return Decimal("0")
    numerator_value, numerator_scale = numerator.as_integer_ratio()
    denominator_value, denominator_scale = denominator.as_integer_ratio()
    units = (
        numerator_value * denominator_scale * WEIGHT_SCALE
        // (numerator_scale * denominator_value)
    )
    return units_to_weight(min(WEIGHT_SCALE, max(0, units)))


def _position_for(snapshot: AccountSnapshot, symbol: InstrumentId) -> Position | None:
    return next((item for item in snapshot.positions if item.instrument == symbol), None)


def _freeze_audit_detail(value: object) -> object:
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        assert isinstance(value, float)
        if not isfinite(value):
            raise ValueError("decision detail floats must be finite")
        return value
    if isinstance(value, Mapping):
        checked: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("decision detail mapping keys must be exact strings")
            checked[key] = _freeze_audit_detail(item)
        return MappingProxyType(dict(sorted(checked.items())))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_audit_detail(item) for item in value)
    raise TypeError("decision details must contain deterministic scalar, sequence, or mapping values")


@dataclass(frozen=True, slots=True)
class InstrumentRiskMetadata:
    data_valid: bool = True
    tradable: bool = True
    observed_volume: int | None = None
    minimum_trade_amount: Decimal = Decimal("0")
    price_constraint_ok: bool = True
    drawdown: Decimal | None = None
    unrealized_loss: Decimal | None = None
    unrealized_gain: Decimal | None = None
    consecutive_losses: int = 0
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label in ("data_valid", "tradable", "price_constraint_ok"):
            if type(getattr(self, label)) is not bool:
                raise TypeError(f"{label} must be an exact bool")
        if self.observed_volume is not None and (
            isinstance(self.observed_volume, bool)
            or not isinstance(self.observed_volume, int)
            or self.observed_volume < 0
        ):
            raise ValueError("observed_volume must be a non-negative integer")
        _exact_decimal(self.minimum_trade_amount, label="minimum_trade_amount")
        for label in ("drawdown", "unrealized_loss", "unrealized_gain"):
            value = getattr(self, label)
            if value is not None:
                _exact_decimal(value, label=label)
        if (
            isinstance(self.consecutive_losses, bool)
            or not isinstance(self.consecutive_losses, int)
            or self.consecutive_losses < 0
        ):
            raise ValueError("consecutive_losses must be a non-negative integer")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        checked: dict[str, str] = {}
        for key, value in self.metadata.items():
            _stable_id(key, label="risk metadata key")
            if type(value) is not str:
                raise TypeError("risk metadata values must be strings")
            checked[key] = value
        object.__setattr__(self, "metadata", MappingProxyType(dict(sorted(checked.items()))))


@dataclass(frozen=True, slots=True)
class StrategyDecisionTrace:
    strategy_id: str
    status: StrategyDecisionStatus
    reason_code: str
    details: Mapping[str, object]

    def __post_init__(self) -> None:
        _stable_id(self.strategy_id, label="strategy_id")
        if type(self.status) is not StrategyDecisionStatus:
            raise TypeError("status must be an exact StrategyDecisionStatus")
        if type(self.reason_code) is not str or _REASON_CODE.fullmatch(self.reason_code) is None:
            raise ValueError("reason_code must be upper snake case")
        if not isinstance(self.details, Mapping):
            raise TypeError("details must be a mapping")
        frozen = _freeze_audit_detail(self.details)
        assert isinstance(frozen, Mapping)
        object.__setattr__(self, "details", frozen)


@dataclass(frozen=True, slots=True)
class EstimatedCosts:
    commission: Decimal
    stamp_duty: Decimal
    transfer_fee: Decimal
    total: Decimal

    def __post_init__(self) -> None:
        for label in ("commission", "stamp_duty", "transfer_fee", "total"):
            value = _exact_decimal(getattr(self, label), label=label)
            if value != _round_money(value):
                raise ValueError(f"{label} must use the cent boundary")
        if self.total != _sum(self.commission, self.stamp_duty, self.transfer_fee):
            raise ValueError("total must equal itemized costs")


ZERO_COSTS = EstimatedCosts(
    commission=Decimal("0.00"),
    stamp_duty=Decimal("0.00"),
    transfer_fee=Decimal("0.00"),
    total=Decimal("0.00"),
)


@dataclass(frozen=True, slots=True)
class RebalanceRecommendation:
    instrument: InstrumentId
    raw_intents: tuple[TargetIntent, ...]
    strategy_decisions: tuple[StrategyDecisionTrace, ...]
    allocated_weight: Decimal
    allocation_trace: tuple[AllocationAdjustment, ...]
    pre_risk_weight: Decimal
    current_weight: Decimal
    final_weight: Decimal
    risk_adjustments: tuple[RiskAdjustment, ...]
    blocked: bool
    current_quantity: int
    target_quantity: int
    quantity_delta: int
    side: DecisionSide
    reference_price: Decimal
    estimated_execution_price: Decimal | None
    gross_amount: Decimal
    costs: EstimatedCosts
    profile_id: str
    market_data_source_at: datetime
    account_snapshot_row_id: int
    account_snapshot_hash: str
    decision_equity: Decimal
    decision_at: datetime
    decision_date: date
    valid_until: date
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.instrument) is not InstrumentId:
            raise TypeError("instrument must be an exact InstrumentId")
        for label, values, expected in (
            ("raw_intents", self.raw_intents, TargetIntent),
            ("strategy_decisions", self.strategy_decisions, StrategyDecisionTrace),
            ("allocation_trace", self.allocation_trace, AllocationAdjustment),
            ("risk_adjustments", self.risk_adjustments, RiskAdjustment),
        ):
            if type(values) is not tuple or any(type(item) is not expected for item in values):
                raise TypeError(f"{label} must be an exact immutable tuple")
        strategy_ids = tuple(item.strategy_id for item in self.strategy_decisions)
        if len(set(strategy_ids)) != len(strategy_ids) or strategy_ids != tuple(
            sorted(strategy_ids)
        ):
            raise ValueError("strategy decisions must be unique and sorted by strategy id")
        intent_keys = tuple((item.strategy_id, str(item.instrument)) for item in self.raw_intents)
        if len(set(intent_keys)) != len(intent_keys) or intent_keys != tuple(sorted(intent_keys)):
            raise ValueError("raw intents must be unique and deterministically sorted")
        if any(item.instrument != self.instrument for item in self.raw_intents):
            raise ValueError("raw intents must match the recommendation instrument")
        if any(item.strategy_id not in strategy_ids for item in self.raw_intents):
            raise ValueError("raw intent strategy must have decision trace provenance")
        trace_statuses = {item.strategy_id: item.status for item in self.strategy_decisions}
        if any(
            trace_statuses[item.strategy_id] is not StrategyDecisionStatus.GENERATED
            for item in self.raw_intents
        ):
            raise ValueError("raw intents require GENERATED strategy decision provenance")
        for label in ("allocated_weight", "pre_risk_weight", "current_weight", "final_weight"):
            weight_to_units(getattr(self, label), label=label)
        if self.pre_risk_weight != self.allocated_weight:
            raise ValueError("pre_risk_weight must equal allocated_weight")
        RiskResult(
            requested_weight=self.pre_risk_weight,
            final_weight=self.final_weight,
            blocked=self.blocked,
            adjustments=self.risk_adjustments,
        )
        if any(
            adjustment.reference_weight != self.current_weight
            for adjustment in self.risk_adjustments
        ):
            raise ValueError("risk adjustment references must equal current_weight")
        if type(self.blocked) is not bool:
            raise TypeError("blocked must be an exact bool")
        for label in ("current_quantity", "target_quantity", "quantity_delta"):
            if isinstance(getattr(self, label), bool) or not isinstance(getattr(self, label), int):
                raise TypeError(f"{label} must be an exact integer")
        if self.current_quantity < 0 or self.target_quantity < 0:
            raise ValueError("account quantities must be non-negative")
        if self.target_quantity != self.current_quantity + self.quantity_delta:
            raise ValueError("target_quantity must equal current_quantity plus quantity_delta")
        if type(self.side) is not DecisionSide:
            raise TypeError("side must be an exact DecisionSide")
        expected_side = (
            DecisionSide.BUY
            if self.quantity_delta > 0
            else DecisionSide.SELL
            if self.quantity_delta < 0
            else DecisionSide.NONE
        )
        if self.side is not expected_side:
            raise ValueError("side must match signed quantity_delta")
        if self.blocked and self.side is not DecisionSide.NONE:
            raise ValueError("blocked recommendations must be no-trade")
        _exact_decimal(self.reference_price, label="reference_price", positive=True)
        if self.estimated_execution_price is not None:
            price = _exact_decimal(
                self.estimated_execution_price, label="estimated_execution_price", positive=True
            )
            if price != _round_price(price):
                raise ValueError("estimated_execution_price must use four decimal places")
        gross = _exact_decimal(self.gross_amount, label="gross_amount")
        if gross != _round_money(gross):
            raise ValueError("gross_amount must use the cent boundary")
        if type(self.costs) is not EstimatedCosts:
            raise TypeError("costs must be exact EstimatedCosts")
        if self.side is DecisionSide.NONE:
            if (
                self.estimated_execution_price is not None
                or self.gross_amount != Decimal("0.00")
                or self.costs != ZERO_COSTS
                or self.target_quantity != self.current_quantity
            ):
                raise ValueError("no-trade recommendations cannot contain execution economics")
        else:
            if self.estimated_execution_price is None or self.quantity_delta == 0:
                raise ValueError("trade recommendations require price and non-zero quantity")
            expected_gross = _round_money(
                _product(self.estimated_execution_price, abs(self.quantity_delta))
            )
            if self.gross_amount != expected_gross:
                raise ValueError("gross_amount must equal estimated price times quantity")
            if self.side is DecisionSide.BUY and self.costs.stamp_duty != Decimal("0.00"):
                raise ValueError("buy recommendations cannot include stamp duty")
        _stable_id(self.profile_id, label="profile_id")
        if (
            isinstance(self.account_snapshot_row_id, bool)
            or not isinstance(self.account_snapshot_row_id, int)
            or self.account_snapshot_row_id <= 0
        ):
            raise ValueError("account_snapshot_row_id must be a positive integer")
        if (
            type(self.account_snapshot_hash) is not str
            or _CONTENT_HASH.fullmatch(self.account_snapshot_hash) is None
        ):
            raise ValueError("account_snapshot_hash must be a lowercase SHA-256 digest")
        equity = _exact_decimal(self.decision_equity, label="decision_equity")
        if equity != _round_money(equity):
            raise ValueError("decision_equity must use the cent boundary")
        source_at = _aware_datetime(self.market_data_source_at, label="market_data_source_at")
        decision_at = _aware_datetime(self.decision_at, label="decision_at")
        decision_date = _exact_date(self.decision_date, label="decision_date")
        valid_until = _exact_date(self.valid_until, label="valid_until")
        if decision_date != decision_at.astimezone(SHANGHAI).date():
            raise ValueError("decision_date must match decision_at in Asia/Shanghai")
        if source_at > decision_at or source_at.astimezone(SHANGHAI).date() != decision_date:
            raise ValueError("market data source must be fresh and not after decision_at")
        if valid_until < decision_date:
            raise ValueError("valid_until must not precede decision_date")
        expected_current_weight = _ratio_weight(
            _round_money(_product(self.reference_price, self.current_quantity)),
            self.decision_equity,
        )
        if self.current_weight != expected_current_weight:
            raise ValueError("current_weight must match current quantity and decision valuation")
        if type(self.reason_codes) is not tuple or any(
            type(code) is not str or _REASON_CODE.fullmatch(code) is None
            for code in self.reason_codes
        ):
            raise TypeError("reason_codes must be immutable upper-snake identifiers")
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise ValueError("reason_codes must be unique")
        risk_codes = tuple(item.code for item in self.risk_adjustments)
        if self.reason_codes[: len(risk_codes)] != risk_codes:
            raise ValueError("risk reason codes must be supported by the ordered risk trace")
        if self.blocked and not {"RISK_BLOCKED", "NO_TRADE"}.issubset(self.reason_codes):
            raise ValueError("blocked recommendations require explicit no-trade reason codes")
        if self.side is DecisionSide.NONE and "NO_TRADE" not in self.reason_codes:
            raise ValueError("no-trade recommendations require a stable NO_TRADE reason")


@dataclass(frozen=True, slots=True)
class DecisionResult:
    account_id: str
    account_snapshot_row_id: int
    account_snapshot_hash: str
    decision_equity: Decimal
    decision_at: datetime
    decision_date: date
    valid_until: date
    market_data_source_at: datetime
    strategy_decisions: tuple[StrategyDecisionTrace, ...]
    recommendations: tuple[RebalanceRecommendation, ...]
    remaining_cash: Decimal
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        _stable_id(self.account_id, label="account_id")
        if (
            isinstance(self.account_snapshot_row_id, bool)
            or not isinstance(self.account_snapshot_row_id, int)
            or self.account_snapshot_row_id <= 0
        ):
            raise ValueError("account_snapshot_row_id must be a positive integer")
        if (
            type(self.account_snapshot_hash) is not str
            or _CONTENT_HASH.fullmatch(self.account_snapshot_hash) is None
        ):
            raise ValueError("account_snapshot_hash must be a lowercase SHA-256 digest")
        equity = _exact_decimal(self.decision_equity, label="decision_equity")
        if equity != _round_money(equity):
            raise ValueError("decision_equity must use the cent boundary")
        decision_at = _aware_datetime(self.decision_at, label="decision_at")
        source_at = _aware_datetime(self.market_data_source_at, label="market_data_source_at")
        decision_date = _exact_date(self.decision_date, label="decision_date")
        valid_until = _exact_date(self.valid_until, label="valid_until")
        if decision_date != decision_at.astimezone(SHANGHAI).date():
            raise ValueError("decision_date must match decision_at in Asia/Shanghai")
        if source_at > decision_at or source_at.astimezone(SHANGHAI).date() != decision_date:
            raise ValueError("market data source must be fresh and not after decision_at")
        if valid_until < decision_date:
            raise ValueError("valid_until must not precede decision_date")
        if type(self.strategy_decisions) is not tuple or any(
            type(item) is not StrategyDecisionTrace for item in self.strategy_decisions
        ):
            raise TypeError("strategy_decisions must be an immutable tuple")
        strategy_ids = tuple(item.strategy_id for item in self.strategy_decisions)
        if len(set(strategy_ids)) != len(strategy_ids) or strategy_ids != tuple(
            sorted(strategy_ids)
        ):
            raise ValueError("strategy_decisions must be unique and sorted")
        if type(self.recommendations) is not tuple or any(
            type(item) is not RebalanceRecommendation for item in self.recommendations
        ):
            raise TypeError("recommendations must be an immutable tuple")
        symbols = tuple(item.instrument for item in self.recommendations)
        if len(set(symbols)) != len(symbols):
            raise ValueError("recommendations must be unique by instrument")
        expected_order = tuple(
            sorted(
                self.recommendations,
                key=lambda item: (
                    0
                    if item.side is DecisionSide.SELL
                    else 1
                    if item.side is DecisionSide.BUY
                    else 2,
                    str(item.instrument),
                ),
            )
        )
        if self.recommendations != expected_order:
            raise ValueError("recommendations must be sorted sells, buys, then no-trades")
        for recommendation in self.recommendations:
            if (
                recommendation.account_snapshot_row_id != self.account_snapshot_row_id
                or recommendation.account_snapshot_hash != self.account_snapshot_hash
            ):
                raise ValueError("recommendation account identity does not match result identity")
            if (
                recommendation.decision_equity != self.decision_equity
                or recommendation.decision_at != self.decision_at
                or recommendation.decision_date != self.decision_date
                or recommendation.market_data_source_at != self.market_data_source_at
                or recommendation.valid_until != self.valid_until
                or recommendation.strategy_decisions != self.strategy_decisions
            ):
                raise ValueError("recommendation audit identity does not match result identity")
        intent_strategy_ids = {
            intent.strategy_id
            for recommendation in self.recommendations
            for intent in recommendation.raw_intents
        }
        generated_strategy_ids = {
            trace.strategy_id
            for trace in self.strategy_decisions
            if trace.status is StrategyDecisionStatus.GENERATED
        }
        if intent_strategy_ids != generated_strategy_ids:
            raise ValueError(
                "GENERATED strategy decisions must have raw intents and non-GENERATED decisions must not"
            )
        cash = _exact_decimal(self.remaining_cash, label="remaining_cash")
        if cash != _round_money(cash):
            raise ValueError("remaining_cash must use the cent boundary")
        if type(self.warnings) is not tuple:
            raise TypeError("warnings must be an immutable tuple")
        expected_warnings = tuple(
            sorted({code for item in self.recommendations for code in item.reason_codes})
        )
        if self.warnings != expected_warnings:
            raise ValueError("warnings must be the exact deterministic child reason summary")


@dataclass(frozen=True, slots=True, init=False)
class CloseDecisionRequest:
    account_id: str
    decision_at: datetime
    valid_until: date
    data_accepted: bool
    daily_close_complete: bool
    market_data_source_at: datetime
    instruments: Mapping[InstrumentId, Instrument]
    strategies: tuple[_DecisionStrategy, ...]
    strategy_pools: Mapping[str, tuple[InstrumentId, ...]]
    allocation_policy: AllocationPolicy
    risk_engine: RiskEngine
    rule_book: MarketRuleBook
    risk_metadata: Mapping[InstrumentId, InstrumentRiskMetadata]
    _bars: Mapping[InstrumentId, pd.DataFrame]

    def __init__(
        self,
        *,
        account_id: str,
        decision_at: datetime,
        valid_until: date,
        data_accepted: bool,
        daily_close_complete: bool,
        market_data_source_at: datetime,
        instruments: Mapping[InstrumentId, Instrument],
        bars: Mapping[InstrumentId, pd.DataFrame],
        strategies: Sequence[_DecisionStrategy],
        allocation_policy: AllocationPolicy,
        risk_engine: RiskEngine,
        rule_book: MarketRuleBook,
        risk_metadata: Mapping[InstrumentId, InstrumentRiskMetadata] | None = None,
        strategy_pools: Mapping[str, Sequence[InstrumentId]] | None = None,
    ) -> None:
        checked_account = _stable_id(account_id, label="account_id")
        checked_decision_at = _aware_datetime(decision_at, label="decision_at")
        checked_source_at = _aware_datetime(
            market_data_source_at, label="market_data_source_at"
        )
        if type(valid_until) is not date or isinstance(valid_until, datetime):
            raise TypeError("valid_until must be an exact date")
        decision_date = checked_decision_at.astimezone(SHANGHAI).date()
        if valid_until < decision_date:
            raise ValueError("valid_until must not precede the decision date")
        if type(data_accepted) is not bool:
            raise TypeError("data_accepted must be an exact bool")
        if type(daily_close_complete) is not bool:
            raise TypeError("daily_close_complete must be an exact bool")
        if not isinstance(instruments, Mapping) or not instruments:
            raise ValueError("instruments must be a non-empty mapping")
        checked_instruments: dict[InstrumentId, Instrument] = {}
        for symbol, instrument in instruments.items():
            if type(symbol) is not InstrumentId or type(instrument) is not Instrument:
                raise TypeError("instruments must map exact InstrumentId to Instrument values")
            if instrument.instrument_id != symbol:
                raise ValueError("instrument key must match its identifier")
            checked_instruments[symbol] = instrument
        if not isinstance(bars, Mapping) or set(bars) != set(checked_instruments):
            raise DecisionDataError("DAILY_DATA_MISSING: bars must cover every instrument")
        checked_bars: dict[InstrumentId, pd.DataFrame] = {}
        for symbol in checked_instruments:
            frame = bars[symbol]
            if not isinstance(frame, pd.DataFrame):
                raise TypeError("bars must contain DataFrame values")
            checked_bars[symbol] = BarFrame.validate(frame)
        checked_strategies = tuple(strategies)
        strategy_ids: list[str] = []
        for strategy in checked_strategies:
            strategy_id = _stable_id(getattr(strategy, "strategy_id", None), label="strategy_id")
            if not callable(getattr(strategy, "generate_targets", None)):
                raise TypeError("strategies must provide generate_targets")
            strategy_ids.append(strategy_id)
        if len(set(strategy_ids)) != len(strategy_ids):
            raise ValueError("strategy instance ids must be unique")
        ordered_strategies = tuple(
            strategy
            for _, strategy in sorted(zip(strategy_ids, checked_strategies), key=lambda item: item[0])
        )
        if type(allocation_policy) is not AllocationPolicy:
            raise TypeError("allocation_policy must be an exact AllocationPolicy")
        if set(strategy_ids) != set(allocation_policy.strategy_budgets):
            raise ValueError("allocation policy must cover exactly the configured strategies")
        raw_pools = (
            {strategy_id: tuple(checked_instruments) for strategy_id in strategy_ids}
            if strategy_pools is None
            else strategy_pools
        )
        if not isinstance(raw_pools, Mapping) or set(raw_pools) != set(strategy_ids):
            raise ValueError("strategy pools must cover exactly the configured strategies")
        checked_pools: dict[str, tuple[InstrumentId, ...]] = {}
        for strategy_id in strategy_ids:
            raw_pool = raw_pools[strategy_id]
            if not isinstance(raw_pool, Sequence) or isinstance(
                raw_pool, (str, bytes, bytearray)
            ):
                raise TypeError("strategy pools must contain instrument sequences")
            pool = tuple(raw_pool)
            if (
                not pool
                or any(type(symbol) is not InstrumentId for symbol in pool)
                or len(set(pool)) != len(pool)
                or not set(pool).issubset(checked_instruments)
            ):
                raise ValueError("strategy pool instruments must be non-empty and configured")
            checked_pools[strategy_id] = tuple(sorted(pool, key=str))
        if allocation_policy.asset_types != {
            symbol: item.asset_type for symbol, item in checked_instruments.items()
        }:
            raise ValueError("allocation policy asset types must match configured instruments")
        if type(risk_engine) is not RiskEngine:
            raise TypeError("risk_engine must be an exact RiskEngine")
        if type(rule_book) is not MarketRuleBook:
            raise TypeError("rule_book must be an exact MarketRuleBook")
        raw_metadata = {} if risk_metadata is None else risk_metadata
        if not isinstance(raw_metadata, Mapping) or not set(raw_metadata).issubset(checked_instruments):
            raise ValueError("risk_metadata must contain only configured instruments")
        checked_metadata: dict[InstrumentId, InstrumentRiskMetadata] = {}
        for symbol in checked_instruments:
            value = raw_metadata.get(symbol, InstrumentRiskMetadata())
            if type(value) is not InstrumentRiskMetadata:
                raise TypeError("risk_metadata values must be exact InstrumentRiskMetadata")
            checked_metadata[symbol] = value

        object.__setattr__(self, "account_id", checked_account)
        object.__setattr__(self, "decision_at", checked_decision_at)
        object.__setattr__(self, "valid_until", valid_until)
        object.__setattr__(self, "data_accepted", data_accepted)
        object.__setattr__(self, "daily_close_complete", daily_close_complete)
        object.__setattr__(self, "market_data_source_at", checked_source_at)
        object.__setattr__(
            self,
            "instruments",
            MappingProxyType(dict(sorted(checked_instruments.items(), key=lambda item: str(item[0])))),
        )
        object.__setattr__(self, "_bars", MappingProxyType(checked_bars))
        object.__setattr__(self, "strategies", ordered_strategies)
        object.__setattr__(
            self,
            "strategy_pools",
            MappingProxyType(
                {strategy_id: checked_pools[strategy_id] for strategy_id in sorted(checked_pools)}
            ),
        )
        object.__setattr__(self, "allocation_policy", allocation_policy)
        object.__setattr__(self, "risk_engine", risk_engine)
        object.__setattr__(self, "rule_book", rule_book)
        object.__setattr__(
            self,
            "risk_metadata",
            MappingProxyType(dict(sorted(checked_metadata.items(), key=lambda item: str(item[0])))),
        )

    @property
    def bars(self) -> Mapping[InstrumentId, pd.DataFrame]:
        return MappingProxyType(
            {symbol: frame.copy(deep=True) for symbol, frame in self._bars.items()}
        )


def _current_bar(frame: pd.DataFrame, decision_date: date) -> pd.Series:
    timestamp = pd.Timestamp(decision_date)
    if timestamp not in frame.index:
        raise DecisionDataError(f"DAILY_CLOSE_MISSING:{decision_date.isoformat()}")
    row = frame.loc[timestamp]
    _cell_decimal(row["close"], label="raw close", positive=True)
    return row


def _strategy_bars(
    bars: Mapping[InstrumentId, pd.DataFrame],
) -> dict[InstrumentId, pd.DataFrame]:
    result: dict[InstrumentId, pd.DataFrame] = {}
    for symbol, source in bars.items():
        frame = source.copy(deep=True)
        if "adjust_flag" in frame.columns:
            flags = tuple(frame["adjust_flag"])
            if any(type(value) is not str or value != "3" for value in flags):
                raise DecisionDataError(f"RAW_PRICE_REQUIRED:{symbol}")
        if "adjust_factor" in frame.columns:
            factors = tuple(
                _cell_decimal(value, label="adjust_factor", positive=True)
                for value in frame["adjust_factor"]
            )
            for column in ("open", "high", "low", "close"):
                adjusted: list[Decimal | float] = []
                for value, factor in zip(frame[column], factors):
                    if type(value) is Decimal:
                        adjusted.append(_product(value, factor))
                    else:
                        adjusted.append(float(value) * float(factor))
                frame[column] = adjusted
        result[symbol] = frame
    return result


def _target_rational(
    weight: Decimal, equity: Decimal, price: Decimal
) -> tuple[int, int]:
    weight_units = weight_to_units(weight, label="target weight")
    equity_value, equity_scale = equity.as_integer_ratio()
    price_value, price_scale = price.as_integer_ratio()
    return (
        weight_units * equity_value * price_scale,
        WEIGHT_SCALE * equity_scale * price_value,
    )


def _directional_delta(
    weight: Decimal,
    equity: Decimal,
    price: Decimal,
    current_quantity: int,
    profile: MarketRuleProfile,
    *,
    available_quantity: int | None,
) -> tuple[int, tuple[str, ...]]:
    numerator, denominator = _target_rational(weight, equity, price)
    current_numerator = current_quantity * denominator
    warnings: list[str] = []
    if numerator > current_numerator:
        raw = (numerator - current_numerator) // denominator
        quantity = raw // profile.buy_lot_size * profile.buy_lot_size
        if quantity != raw:
            warnings.append("ROUNDED")
        return quantity, tuple(warnings)
    if numerator >= current_numerator:
        return 0, ()
    if weight == 0 and profile.odd_lot_sell_policy in (
        OddLotSellPolicy.ALLOWED,
        OddLotSellPolicy.POSITION_REMAINDER_ONLY,
    ):
        raw_sell = current_quantity
    else:
        raw_sell = (current_numerator - numerator) // denominator
    quantity = raw_sell
    if profile.odd_lot_sell_policy is OddLotSellPolicy.FORBIDDEN or (
        profile.odd_lot_sell_policy is OddLotSellPolicy.POSITION_REMAINDER_ONLY
        and not (weight == 0 and raw_sell == current_quantity)
    ):
        quantity = raw_sell // profile.buy_lot_size * profile.buy_lot_size
    if quantity != raw_sell:
        warnings.append("ROUNDED")
    if available_quantity is not None and quantity > available_quantity:
        quantity = available_quantity
        if profile.odd_lot_sell_policy is OddLotSellPolicy.FORBIDDEN or (
            profile.odd_lot_sell_policy is OddLotSellPolicy.POSITION_REMAINDER_ONLY
            and not (weight == 0 and quantity == current_quantity)
        ):
            quantity = quantity // profile.buy_lot_size * profile.buy_lot_size
        warnings.append("AVAILABILITY_LIMITED")
    return -quantity, tuple(warnings)


def _estimated_price(
    side: DecisionSide,
    reference_price: Decimal,
    profile: MarketRuleProfile,
    frame: pd.DataFrame,
    row: pd.Series,
    instrument: Instrument,
    decision_day: date,
) -> Decimal:
    limits = effective_price_limits(
        frame,
        pd.Timestamp(decision_day),
        row,
        decision_day,
        instrument,
        profile,
        has_corporate_action=False,
    )
    if not limits.state_known:
        raise DecisionDataError(
            f"PRICE_LIMIT_STATE_UNKNOWN:{instrument.instrument_id}"
        )
    with localcontext() as context:
        context.prec = 50
        direction = Decimal("1") if side is DecisionSide.BUY else Decimal("-1")
        slipped = reference_price * (
            Decimal("1") + direction * profile.slippage_bps / Decimal("10000")
        )
        if side is DecisionSide.BUY and limits.limit_up is not None:
            slipped = min(slipped, limits.limit_up)
        if side is DecisionSide.SELL and limits.limit_down is not None:
            slipped = max(slipped, limits.limit_down)
        rounded = _round_price(slipped)
        if (
            side is DecisionSide.BUY
            and limits.limit_up is not None
        ):
            if rounded > limits.limit_up:
                rounded = limits.limit_up.quantize(
                    PRICE_QUANTUM, rounding=ROUND_FLOOR
                )
        if (
            side is DecisionSide.SELL
            and limits.limit_down is not None
        ):
            if rounded < limits.limit_down:
                rounded = limits.limit_down.quantize(
                    PRICE_QUANTUM, rounding=ROUND_CEILING
                )
        return _exact_decimal(rounded, label="estimated price", positive=True)


def _costs(
    side: DecisionSide,
    quantity: int,
    price: Decimal,
    profile: MarketRuleProfile,
) -> tuple[Decimal, EstimatedCosts]:
    gross = _round_money(_product(price, quantity))
    commission = _round_money(max(profile.minimum_commission, _product(gross, profile.commission_rate)))
    stamp = (
        _round_money(_product(gross, profile.sell_stamp_duty_rate))
        if side is DecisionSide.SELL
        else Decimal("0.00")
    )
    transfer = _round_money(_product(gross, profile.transfer_fee_rate))
    total = _round_money(_sum(commission, stamp, transfer))
    return gross, EstimatedCosts(commission, stamp, transfer, total)


def _codes(*groups: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for group in groups:
        for code in group:
            if code not in result:
                result.append(code)
    return tuple(result)


def _allocation_for_symbol(
    adjustments: tuple[AllocationAdjustment, ...], symbol: InstrumentId
) -> tuple[AllocationAdjustment, ...]:
    canonical = str(symbol)
    suffix = f"/{symbol}"
    return tuple(
        adjustment
        for adjustment in adjustments
        if any(key == canonical or key.endswith(suffix) for key in adjustment.input_keys)
    )


class DecisionService:
    """Generate immutable close recommendations without order or broker side effects."""

    def __init__(self, accounts: AccountRepository) -> None:
        if type(accounts) is not AccountRepository:
            raise TypeError("accounts must be an exact AccountRepository")
        self._accounts = accounts

    def generate_close_decision(self, request: CloseDecisionRequest) -> DecisionResult:
        if type(request) is not CloseDecisionRequest:
            raise TypeError("request must be an exact CloseDecisionRequest")
        if request.account_id != self._accounts.account_id:
            raise ValueError("decision account does not match repository scope")
        account = self._accounts.latest()
        if account is None:
            raise LookupError(f"ACCOUNT_SNAPSHOT_MISSING:{request.account_id}")
        decision_date = request.decision_at.astimezone(SHANGHAI).date()
        self._validate_data(request, decision_date, account)
        snapshot = account.snapshot
        valuation = self._decision_valuation(request, snapshot, decision_date)
        holding_since = self._accounts.holding_since(decision_date)
        holdings = {
            position.instrument: HoldingSummary(
                position.instrument,
                position.quantity,
                position.available_quantity,
                position.average_cost,
                position.mark_price,
                holding_since.get(position.instrument, snapshot.as_of),
            )
            for position in valuation.positions
        }
        context = StrategyContext(
            as_of=decision_date,
            bars=_strategy_bars(request._bars),
            instruments=tuple(request.instruments),
            account_equity=valuation.equity,
            cash=valuation.cash,
            holdings=holdings,
            asset_types={symbol: item.asset_type for symbol, item in request.instruments.items()},
        )
        intents: list[TargetIntent] = []
        decision_traces: list[StrategyDecisionTrace] = []
        for strategy in request.strategies:
            decision = strategy.generate_targets(context)
            if type(decision) is not StrategyDecision:
                raise TypeError("strategy must return an exact StrategyDecision")
            strategy_id = strategy.strategy_id
            decision_traces.append(
                StrategyDecisionTrace(
                    strategy_id,
                    decision.status,
                    decision.reason_code,
                    decision.details,
                )
            )
            for intent in decision.intents:
                self._validate_intent(intent, strategy_id, decision_date, request)
                intents.append(intent)
        ordered_intents = tuple(sorted(intents, key=lambda item: (item.strategy_id, str(item.instrument))))
        portfolio = DeterministicAllocator().allocate(ordered_intents, request.allocation_policy)
        traces = tuple(decision_traces)
        allocated = dict(portfolio.weights)
        status_by_strategy = {item.strategy_id: item.status for item in traces}
        preserved = {
            symbol
            for strategy_id, status in status_by_strategy.items()
            if status is StrategyDecisionStatus.SKIPPED
            for symbol in request.strategy_pools[strategy_id]
        }
        current_weights = {
            position.instrument: _ratio_weight(
                position.market_value,
                valuation.equity,
            )
            for position in valuation.positions
        }
        for symbol in preserved:
            current = current_weights.get(symbol, Decimal("0"))
            allocated[str(symbol)] = max(allocated.get(str(symbol), Decimal("0")), current)
        recommendations, remaining_cash = self._recommendations(
            request,
            account,
            valuation,
            allocated,
            portfolio.adjustments,
            ordered_intents,
            traces,
        )
        warnings = tuple(
            sorted({code for recommendation in recommendations for code in recommendation.reason_codes})
        )
        return DecisionResult(
            account_id=request.account_id,
            account_snapshot_row_id=account.row_id,
            account_snapshot_hash=account.content_hash,
            decision_equity=valuation.equity,
            decision_at=request.decision_at,
            decision_date=decision_date,
            valid_until=request.valid_until,
            market_data_source_at=request.market_data_source_at,
            strategy_decisions=traces,
            recommendations=recommendations,
            remaining_cash=remaining_cash,
            warnings=warnings,
        )

    @staticmethod
    def _decision_valuation(
        request: CloseDecisionRequest,
        snapshot: AccountSnapshot,
        decision_date: date,
    ) -> AccountSnapshot:
        positions = tuple(
            Position(
                instrument=position.instrument,
                quantity=position.quantity,
                available_quantity=position.available_quantity,
                average_cost=position.average_cost,
                mark_price=_cell_decimal(
                    _current_bar(request._bars[position.instrument], decision_date)["close"],
                    label="raw close",
                    positive=True,
                ),
            )
            for position in snapshot.positions
        )
        return AccountSnapshot(snapshot.as_of, snapshot.cash, positions)

    @staticmethod
    def _validate_data(
        request: CloseDecisionRequest,
        decision_date: date,
        account: StoredAccountSnapshot,
    ) -> None:
        if not request.data_accepted:
            raise DecisionDataError("DAILY_DATA_NOT_ACCEPTED")
        if not request.daily_close_complete:
            raise DecisionDataError("DAILY_CLOSE_INCOMPLETE")
        if request.market_data_source_at > request.decision_at:
            raise DecisionDataError("MARKET_DATA_FROM_FUTURE")
        if request.market_data_source_at.astimezone(SHANGHAI).date() != decision_date:
            raise DecisionDataError("MARKET_DATA_STALE")
        if account.snapshot.as_of > decision_date:
            raise ValueError("ACCOUNT_SNAPSHOT_FROM_FUTURE")
        configured = set(request.instruments)
        held = {position.instrument for position in account.snapshot.positions}
        if not held.issubset(configured):
            raise DecisionDataError("HELD_INSTRUMENT_DATA_MISSING")
        for symbol, frame in request._bars.items():
            _current_bar(frame, decision_date)
            if frame.empty or frame.index[-1].date() < decision_date:
                raise DecisionDataError(f"MARKET_DATA_STALE:{symbol}")

    @staticmethod
    def _validate_intent(
        intent: TargetIntent,
        strategy_id: str,
        decision_date: date,
        request: CloseDecisionRequest,
    ) -> None:
        if type(intent) is not TargetIntent:
            raise TypeError("strategy intents must be exact TargetIntent values")
        if intent.strategy_id != strategy_id:
            raise ValueError("intent strategy id must match its configured strategy")
        if intent.instrument not in request.instruments:
            raise ValueError("intent instrument is not configured")
        if type(intent.valid_until) is not date or intent.valid_until < decision_date:
            raise ValueError("intent valid_until must not precede the decision date")
        if type(intent.reason_code) is not str or _REASON_CODE.fullmatch(intent.reason_code) is None:
            raise ValueError("intent reason_code must be upper snake case")
        if type(intent.score) is not float or not isfinite(intent.score):
            raise ValueError("intent score must be a finite float")

    def _recommendations(
        self,
        request: CloseDecisionRequest,
        account: StoredAccountSnapshot,
        valuation: AccountSnapshot,
        allocated: Mapping[str, Decimal],
        allocation_adjustments: tuple[AllocationAdjustment, ...],
        intents: tuple[TargetIntent, ...],
        strategy_traces: tuple[StrategyDecisionTrace, ...],
    ) -> tuple[tuple[RebalanceRecommendation, ...], Decimal]:
        target_symbols = {
            InstrumentId.parse(symbol) for symbol in allocated
        } | {position.instrument for position in valuation.positions}

        def direction_key(symbol: InstrumentId) -> tuple[int, str]:
            position = _position_for(valuation, symbol)
            current = (
                Decimal("0")
                if position is None
                else _ratio_weight(position.market_value, valuation.equity)
            )
            target = allocated.get(str(symbol), Decimal("0"))
            return (0 if target < current else 1, str(symbol))

        # One stable risk/funding order: sell-direction targets first, then buys,
        # with canonical instrument identity breaking ties in either group.
        running_cash = valuation.cash
        recommendations: list[RebalanceRecommendation] = []
        approved_strategy_units = 0
        approved_buy_turnover_units = 0
        total_allocated_units = sum(
            weight_to_units(value, label="allocated weight") for value in allocated.values()
        )
        for symbol in sorted(target_symbols, key=direction_key):
            instrument = request.instruments[symbol]
            profile = request.rule_book.profile_for(
                request.decision_at.astimezone(SHANGHAI).date(), instrument
            )
            row = _current_bar(request._bars[symbol], request.decision_at.astimezone(SHANGHAI).date())
            reference_price = _cell_decimal(row["close"], label="raw close", positive=True)
            position = _position_for(valuation, symbol)
            current_quantity = 0 if position is None else position.quantity
            available_quantity = 0 if position is None else position.available_quantity
            current_value = Decimal("0") if position is None else position.market_value
            current_weight = _ratio_weight(current_value, valuation.equity)
            allocated_weight = allocated.get(str(symbol), Decimal("0"))
            preliminary_delta, _ = _directional_delta(
                allocated_weight,
                valuation.equity,
                reference_price,
                current_quantity,
                profile,
                available_quantity=None,
            )
            metadata = request.risk_metadata[symbol]
            volume_value = row["volume"]
            if isinstance(volume_value, bool) or int(volume_value) != volume_value:
                raise DecisionDataError(f"VOLUME_INVALID:{symbol}")
            suspended_value = row.get("suspended", False)
            if pd.isna(suspended_value):
                suspended = False
            elif type(suspended_value) is bool or isinstance(suspended_value, np.bool_):
                suspended = bool(suspended_value)
            else:
                raise DecisionDataError(f"SUSPENDED_STATE_INVALID:{symbol}")
            other_units = max(
                0,
                total_allocated_units
                - weight_to_units(allocated_weight, label="allocated weight"),
            )
            other_stock_units = sum(
                weight_to_units(weight, label="other stock weight")
                for other_symbol, weight in allocated.items()
                if other_symbol != str(symbol)
                and request.instruments[InstrumentId.parse(other_symbol)].asset_type
                is AssetType.STOCK
            )
            risk_context = RiskContext(
                as_of=request.decision_at,
                data_as_of=request.market_data_source_at,
                data_valid=request.data_accepted and metadata.data_valid,
                tradable=metadata.tradable and not suspended,
                other_invested_weight=units_to_weight(min(WEIGHT_SCALE, other_units)),
                other_stock_weight=units_to_weight(min(WEIGHT_SCALE, other_stock_units)),
                strategy_other_weight=units_to_weight(approved_strategy_units),
                turnover_used_weight=units_to_weight(approved_buy_turnover_units),
                observed_volume=(
                    int(volume_value)
                    if metadata.observed_volume is None
                    else metadata.observed_volume
                ),
                expected_order_quantity=abs(preliminary_delta),
                available_cash=running_cash,
                account_equity=valuation.equity,
                available_sell_quantity=available_quantity,
                lot_size=profile.buy_lot_size,
                allow_odd_lot_sell=(
                    profile.odd_lot_sell_policy is not OddLotSellPolicy.FORBIDDEN
                ),
                reference_price=reference_price,
                minimum_trade_amount=metadata.minimum_trade_amount,
                price_constraint_ok=metadata.price_constraint_ok,
                drawdown=metadata.drawdown,
                unrealized_loss=metadata.unrealized_loss,
                unrealized_gain=metadata.unrealized_gain,
                consecutive_losses=metadata.consecutive_losses,
                metadata=metadata.metadata,
            )
            risk_result = request.risk_engine.apply(
                risk_context,
                RiskTarget(
                    instrument=instrument,
                    requested_weight=allocated_weight,
                    current_weight=current_weight,
                    strategy_id="portfolio",
                ),
            )
            outcome_codes: tuple[str, ...]
            if risk_result.blocked:
                delta = 0
                outcome_codes = ("RISK_BLOCKED", "NO_TRADE")
            else:
                delta, outcome_codes = _directional_delta(
                    risk_result.final_weight,
                    valuation.equity,
                    reference_price,
                    current_quantity,
                    profile,
                    available_quantity=available_quantity,
                )
                if delta == 0:
                    outcome_codes = _codes(outcome_codes, ("NO_TRADE",))
            side = (
                DecisionSide.BUY
                if delta > 0
                else DecisionSide.SELL
                if delta < 0
                else DecisionSide.NONE
            )
            if delta != 0 and not profile.fee_profile_confirmed:
                raise ValueError(f"FEE_PROFILE_UNCONFIRMED:{profile.profile_id}")
            if side is DecisionSide.NONE:
                estimated_price = None
                gross = Decimal("0.00")
                costs = ZERO_COSTS
            else:
                estimated_price = _estimated_price(
                    side,
                    reference_price,
                    profile,
                    request._bars[symbol],
                    row,
                    instrument,
                    request.decision_at.astimezone(SHANGHAI).date(),
                )
                gross, costs = _costs(side, abs(delta), estimated_price, profile)
            if side is DecisionSide.SELL:
                post_sell_cash = _sum(running_cash, gross, -costs.total)
                if post_sell_cash < 0:
                    delta = 0
                    side = DecisionSide.NONE
                    estimated_price = None
                    gross = Decimal("0.00")
                    costs = ZERO_COSTS
                    outcome_codes = _codes(
                        outcome_codes,
                        ("SELL_FEES_EXCEED_AVAILABLE_CASH", "NO_TRADE"),
                    )
                else:
                    running_cash = _round_money(post_sell_cash)
            effective_weight = _ratio_weight(
                _round_money(_product(reference_price, current_quantity + delta)),
                valuation.equity,
            )
            effective_units = weight_to_units(
                effective_weight, label="effective strategy weight"
            )
            approved_strategy_units = min(
                WEIGHT_SCALE, approved_strategy_units + effective_units
            )
            current_units = weight_to_units(current_weight, label="current weight")
            approved_buy_turnover_units = min(
                WEIGHT_SCALE,
                approved_buy_turnover_units + max(0, effective_units - current_units),
            )
            risk_codes = tuple(item.code for item in risk_result.adjustments)
            reason_codes = _codes(risk_codes, outcome_codes)
            recommendation = RebalanceRecommendation(
                instrument=symbol,
                raw_intents=tuple(item for item in intents if item.instrument == symbol),
                strategy_decisions=strategy_traces,
                allocated_weight=allocated_weight,
                allocation_trace=_allocation_for_symbol(allocation_adjustments, symbol),
                pre_risk_weight=allocated_weight,
                current_weight=current_weight,
                final_weight=risk_result.final_weight,
                risk_adjustments=risk_result.adjustments,
                blocked=risk_result.blocked,
                current_quantity=current_quantity,
                target_quantity=current_quantity + delta,
                quantity_delta=delta,
                side=side,
                reference_price=reference_price,
                estimated_execution_price=estimated_price,
                gross_amount=gross,
                costs=costs,
                profile_id=profile.profile_id,
                market_data_source_at=request.market_data_source_at,
                account_snapshot_row_id=account.row_id,
                account_snapshot_hash=account.content_hash,
                decision_equity=valuation.equity,
                decision_at=request.decision_at,
                decision_date=request.decision_at.astimezone(SHANGHAI).date(),
                valid_until=request.valid_until,
                reason_codes=reason_codes,
            )
            recommendations.append(recommendation)

        recommendations = self._scale_buys(recommendations, running_cash, request)
        buy_spend = _sum(
            *(
                item.gross_amount + item.costs.total
                for item in recommendations
                if item.side is DecisionSide.BUY
            )
        )
        remaining_cash = _round_money(_sum(running_cash, -buy_spend))
        ordered = tuple(
            sorted(
                recommendations,
                key=lambda item: (
                    0 if item.side is DecisionSide.SELL else 1 if item.side is DecisionSide.BUY else 2,
                    str(item.instrument),
                ),
            )
        )
        return ordered, remaining_cash

    @staticmethod
    def _scale_buys(
        recommendations: list[RebalanceRecommendation],
        available_cash: Decimal,
        request: CloseDecisionRequest,
    ) -> list[RebalanceRecommendation]:
        buy_total = _sum(
            *(
                item.gross_amount + item.costs.total
                for item in recommendations
                if item.side is DecisionSide.BUY
            )
        )
        if buy_total <= available_cash:
            return recommendations
        buy_indexes = tuple(
            index
            for index, item in enumerate(recommendations)
            if item.side is DecisionSide.BUY
        )

        def scaled_quantities(scale_units: int) -> dict[int, int]:
            result: dict[int, int] = {}
            for index in buy_indexes:
                item = recommendations[index]
                profile = request.rule_book.profile_for(
                    item.decision_date, request.instruments[item.instrument]
                )
                raw = item.quantity_delta * scale_units // WEIGHT_SCALE
                result[index] = raw // profile.buy_lot_size * profile.buy_lot_size
            return result

        def scaled_cost(scale_units: int) -> Decimal:
            total = Decimal("0.00")
            for index, quantity in scaled_quantities(scale_units).items():
                if quantity == 0:
                    continue
                item = recommendations[index]
                assert item.estimated_execution_price is not None
                profile = request.rule_book.profile_for(
                    item.decision_date, request.instruments[item.instrument]
                )
                gross, costs = _costs(
                    DecisionSide.BUY,
                    quantity,
                    item.estimated_execution_price,
                    profile,
                )
                total = _sum(total, gross, costs.total)
            return total

        lower = 0
        upper = WEIGHT_SCALE
        while lower < upper:
            middle = (lower + upper + 1) // 2
            if scaled_cost(middle) <= available_cash:
                lower = middle
            else:
                upper = middle - 1
        quantities = scaled_quantities(lower)
        scaled = list(recommendations)
        for index in buy_indexes:
            item = scaled[index]
            profile = request.rule_book.profile_for(
                item.decision_date, request.instruments[item.instrument]
            )
            quantity = quantities[index]
            if quantity == 0:
                side = DecisionSide.NONE
                price = None
                gross = Decimal("0.00")
                costs = ZERO_COSTS
            else:
                side = DecisionSide.BUY
                assert item.estimated_execution_price is not None
                price = item.estimated_execution_price
                gross, costs = _costs(side, quantity, price, profile)
            scaled[index] = replace(
                item,
                target_quantity=item.target_quantity - item.quantity_delta + quantity,
                quantity_delta=quantity,
                side=side,
                estimated_execution_price=price,
                gross_amount=gross,
                costs=costs,
                reason_codes=_codes(
                    item.reason_codes,
                    ("CASH_SCALED", "ROUNDED"),
                    (() if quantity else ("NO_TRADE",)),
                ),
            )
        return scaled
