from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import IntEnum, StrEnum
import re
from types import MappingProxyType
from typing import Protocol

from compass.domain.market import AssetType, Exchange, Instrument, InstrumentId
from compass.domain.weights import weight_to_units


class RiskStage(IntEnum):
    DATA_INSTRUMENT = 1
    STRATEGY = 2
    PORTFOLIO = 3
    ORDER = 4


class RiskSeverity(StrEnum):
    INFO = "info"
    ADJUST = "adjust"
    BLOCK = "block"


def _aware_datetime(value: object, *, label: str) -> datetime:
    if type(value) is not datetime:
        raise TypeError(f"{label} must be an exact datetime")
    assert isinstance(value, datetime)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value


def _non_negative_decimal(value: object, *, label: str) -> Decimal:
    if type(value) is not Decimal:
        raise TypeError(f"{label} must be an exact Decimal")
    assert isinstance(value, Decimal)
    if not value.is_finite() or value < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return value


def _optional_count(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an exact integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class RiskTarget:
    instrument: Instrument
    requested_weight: Decimal
    current_weight: Decimal = Decimal("0")
    strategy_id: str = ""

    def __post_init__(self) -> None:
        if type(self.instrument) is not Instrument:
            raise TypeError("instrument must be an exact Instrument")
        if type(self.instrument.instrument_id) is not InstrumentId:
            raise TypeError("instrument id must be an exact InstrumentId")
        instrument_id = self.instrument.instrument_id
        if type(instrument_id.exchange) is not Exchange:
            raise TypeError("instrument exchange must be an exact Exchange")
        if (
            type(instrument_id.code) is not str
            or len(instrument_id.code) != 6
            or re.fullmatch(r"[0-9]{6}", instrument_id.code) is None
        ):
            raise ValueError("instrument code must be a canonical six-digit code")
        if type(self.instrument.asset_type) is not AssetType:
            raise TypeError("instrument asset type must be an exact AssetType")
        if isinstance(self.instrument.lot_size, bool) or not isinstance(
            self.instrument.lot_size, int
        ):
            raise TypeError("instrument lot size must be an exact integer")
        if self.instrument.lot_size <= 0:
            raise ValueError("instrument lot size must be positive")
        if type(self.instrument.same_day_sell) is not bool:
            raise TypeError("instrument same-day sell flag must be an exact bool")
        weight_to_units(self.requested_weight, label="requested weight")
        weight_to_units(self.current_weight, label="current weight")
        if type(self.strategy_id) is not str or self.strategy_id != self.strategy_id.strip():
            raise ValueError("strategy_id must be a stable string")


@dataclass(frozen=True, slots=True)
class RiskContext:
    as_of: datetime
    data_as_of: datetime | None
    data_valid: bool
    tradable: bool
    other_invested_weight: Decimal = Decimal("0")
    other_stock_weight: Decimal = Decimal("0")
    strategy_other_weight: Decimal = Decimal("0")
    turnover_used_weight: Decimal = Decimal("0")
    observed_volume: int | None = None
    expected_order_quantity: int | None = None
    available_cash: Decimal | None = None
    account_equity: Decimal | None = None
    available_sell_quantity: int | None = None
    lot_size: int | None = None
    allow_odd_lot_sell: bool = False
    reference_price: Decimal | None = None
    minimum_trade_amount: Decimal | None = None
    price_constraint_ok: bool | None = None
    drawdown: Decimal | None = None
    unrealized_loss: Decimal | None = None
    unrealized_gain: Decimal | None = None
    consecutive_losses: int = 0
    metadata: Mapping[str, str] = MappingProxyType({})

    def __post_init__(self) -> None:
        as_of = _aware_datetime(self.as_of, label="as_of")
        if self.data_as_of is not None:
            data_as_of = _aware_datetime(self.data_as_of, label="data_as_of")
            if data_as_of > as_of:
                raise ValueError("data_as_of must not be after as_of")
        for name in ("data_valid", "tradable", "allow_odd_lot_sell"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        if self.price_constraint_ok is not None and type(self.price_constraint_ok) is not bool:
            raise TypeError("price_constraint_ok must be an exact bool when supplied")
        for name in (
            "other_invested_weight",
            "other_stock_weight",
            "strategy_other_weight",
            "turnover_used_weight",
        ):
            weight_to_units(getattr(self, name), label=name.replace("_", " "))
        for name in (
            "observed_volume",
            "expected_order_quantity",
            "available_sell_quantity",
            "lot_size",
        ):
            value = _optional_count(getattr(self, name), label=name)
            if name == "lot_size" and value == 0:
                raise ValueError("lot_size must be positive")
        for name in (
            "available_cash",
            "account_equity",
            "reference_price",
            "minimum_trade_amount",
            "drawdown",
            "unrealized_loss",
            "unrealized_gain",
        ):
            value = getattr(self, name)
            if value is not None:
                _non_negative_decimal(value, label=name)
        if self.reference_price is not None and self.reference_price == 0:
            raise ValueError("reference_price must be positive")
        if isinstance(self.consecutive_losses, bool) or not isinstance(
            self.consecutive_losses, int
        ):
            raise TypeError("consecutive_losses must be an exact integer")
        if self.consecutive_losses < 0:
            raise ValueError("consecutive_losses must be non-negative")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        checked_metadata: dict[str, str] = {}
        for key, metadata_value in self.metadata.items():
            if type(key) is not str or not key or key != key.strip():
                raise ValueError("metadata keys must be stable non-empty strings")
            if type(metadata_value) is not str:
                raise TypeError("metadata values must be strings")
            checked_metadata[key] = metadata_value
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(sorted(checked_metadata.items())))
        )


@dataclass(frozen=True, slots=True)
class RiskAdjustment:
    code: str
    stage: RiskStage
    severity: RiskSeverity
    before_weight: Decimal
    after_weight: Decimal
    reference_weight: Decimal
    message: str

    def __post_init__(self) -> None:
        if type(self.code) is not str or not re.fullmatch(r"[A-Z][A-Z0-9_]*", self.code):
            raise ValueError("risk adjustment code must be upper snake case")
        if type(self.stage) is not RiskStage:
            raise TypeError("risk adjustment stage must be an exact RiskStage")
        if type(self.severity) is not RiskSeverity:
            raise TypeError("risk severity must be an exact RiskSeverity")
        before = weight_to_units(self.before_weight, label="before weight")
        after = weight_to_units(self.after_weight, label="after weight")
        reference = weight_to_units(self.reference_weight, label="reference weight")
        if self.severity is RiskSeverity.INFO and after != before:
            raise ValueError("informational risk adjustments cannot change weight")
        if self.severity is RiskSeverity.ADJUST:
            if after == before:
                raise ValueError("risk adjustments must change weight")
            before_distance = abs(before - reference)
            after_distance = abs(after - reference)
            if after_distance > before_distance:
                raise ValueError("risk adjustments cannot increase trade magnitude")
            if (before - reference) * (after - reference) < 0:
                raise ValueError("risk adjustments cannot cross the current holding weight")
        if self.severity is RiskSeverity.BLOCK and after != 0:
            raise ValueError("blocking risk adjustments must set weight to zero")
        if (
            type(self.message) is not str
            or not self.message
            or self.message != self.message.strip()
        ):
            raise ValueError("risk adjustment message must be non-empty")


@dataclass(frozen=True, slots=True)
class RiskResult:
    requested_weight: Decimal
    final_weight: Decimal
    blocked: bool
    adjustments: tuple[RiskAdjustment, ...]

    def __post_init__(self) -> None:
        weight_to_units(self.requested_weight, label="requested weight")
        weight_to_units(self.final_weight, label="final weight")
        if type(self.blocked) is not bool:
            raise TypeError("blocked must be an exact bool")
        if type(self.adjustments) is not tuple or any(
            type(item) is not RiskAdjustment for item in self.adjustments
        ):
            raise TypeError("adjustments must be an exact tuple of RiskAdjustment values")
        expected = self.requested_weight
        found_block = False
        reference: Decimal | None = None
        for adjustment in self.adjustments:
            if adjustment.before_weight != expected:
                raise ValueError("risk adjustment trace is not continuous")
            if reference is None:
                reference = adjustment.reference_weight
            elif adjustment.reference_weight != reference:
                raise ValueError("risk adjustment trace changed its holding reference")
            expected = adjustment.after_weight
            found_block = found_block or adjustment.severity is RiskSeverity.BLOCK
        if self.final_weight != expected:
            raise ValueError("final weight does not match the adjustment trace")
        if self.blocked != found_block:
            raise ValueError("blocked state does not match adjustment severities")
        if self.blocked and self.final_weight != 0:
            raise ValueError("a blocked result must have zero final weight")


class RiskRule(Protocol):
    code: str
    stage: RiskStage
    priority: int
    enabled: bool

    def evaluate(
        self, context: RiskContext, target: RiskTarget, current_weight: Decimal
    ) -> RiskAdjustment | None: ...
