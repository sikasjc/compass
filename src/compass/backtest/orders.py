from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP, localcontext
from enum import StrEnum
import re
from types import MappingProxyType

from compass.domain.market import InstrumentId
from compass.domain.weights import weight_to_units


MONEY_QUANTUM = Decimal("0.01")
PRICE_QUANTUM = Decimal("0.0001")
MAX_DECIMAL_MAGNITUDE = Decimal("1E+24")


def exact_decimal(value: object, *, label: str, positive: bool = False) -> Decimal:
    if type(value) is not Decimal:
        raise TypeError(f"{label} must be an exact Decimal")
    assert isinstance(value, Decimal)
    if not value.is_finite() or value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be finite and {qualifier}")
    if value.copy_abs() > MAX_DECIMAL_MAGNITUDE:
        raise ValueError(f"{label} magnitude exceeds the supported bound")
    return value


def round_money(value: Decimal) -> Decimal:
    exact_decimal(value, label="money")
    with localcontext() as context:
        context.prec = 50
        return value.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)


def round_price(value: Decimal) -> Decimal:
    exact_decimal(value, label="price", positive=True)
    with localcontext() as context:
        context.prec = 50
        return value.quantize(PRICE_QUANTUM, rounding=ROUND_HALF_UP)


def exact_product(left: Decimal, right: Decimal | int) -> Decimal:
    with localcontext() as context:
        context.prec = 50
        return left * right


def exact_sum(*values: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = 50
        total = Decimal("0")
        for value in values:
            total += value
        return total


def exact_subtract(left: Decimal, right: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = 50
        return left - right


def _exact_date(value: object, *, label: str) -> date:
    if type(value) is not date:
        raise TypeError(f"{label} must be an exact date")
    assert isinstance(value, date)
    return value


def _stable_id(value: object, *, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", value) is None:
        raise ValueError(f"{label} must be a stable identifier")
    assert isinstance(value, str)
    return value


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(StrEnum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"


class CancellationReason(StrEnum):
    MISSING_BAR = "MISSING_BAR"
    SUSPENDED = "SUSPENDED"
    LIMIT_UP = "LIMIT_UP"
    LIMIT_DOWN = "LIMIT_DOWN"
    MARKET_STATUS_UNKNOWN = "MARKET_STATUS_UNKNOWN"
    T_PLUS_ONE = "T_PLUS_ONE"
    NO_POSITION = "NO_POSITION"
    INSUFFICIENT_POSITION = "INSUFFICIENT_POSITION"
    ODD_LOT_NOT_ALLOWED = "ODD_LOT_NOT_ALLOWED"
    LOT_SIZE = "LOT_SIZE"
    VOLUME_LIMIT = "VOLUME_LIMIT"
    INSUFFICIENT_CASH = "INSUFFICIENT_CASH"
    END_OF_DAY_UNFILLED = "END_OF_DAY_UNFILLED"
    NO_NEXT_SESSION = "NO_NEXT_SESSION"


def _freeze_weights(values: Mapping[str, Decimal]) -> Mapping[str, Decimal]:
    if not isinstance(values, Mapping):
        raise TypeError("sleeve_weights must be a mapping")
    checked: dict[str, Decimal] = {}
    for key, value in values.items():
        _stable_id(key, label="sleeve id")
        weight_to_units(value, label="sleeve weight")
        checked[key] = value
    return MappingProxyType(dict(sorted(checked.items())))


@dataclass(frozen=True, slots=True)
class Order:
    order_id: str
    instrument: InstrumentId
    side: OrderSide
    quantity: int
    created_on: date
    scheduled_for: date | None
    sleeve_weights: Mapping[str, Decimal]
    risk_codes: tuple[str, ...] = ()
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    cancellation_reason: CancellationReason | None = None

    def __post_init__(self) -> None:
        _stable_id(self.order_id, label="order_id")
        if type(self.instrument) is not InstrumentId:
            raise TypeError("instrument must be an exact InstrumentId")
        if type(self.side) is not OrderSide:
            raise TypeError("side must be an exact OrderSide")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int):
            raise TypeError("quantity must be an exact integer")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        _exact_date(self.created_on, label="created_on")
        if self.scheduled_for is not None:
            scheduled = _exact_date(self.scheduled_for, label="scheduled_for")
            if scheduled <= self.created_on:
                raise ValueError("scheduled_for must be after created_on")
        sleeve_weights = _freeze_weights(self.sleeve_weights)
        risk_codes = tuple(self.risk_codes)
        for code in risk_codes:
            if type(code) is not str or re.fullmatch(r"[A-Z][A-Z0-9_]*", code) is None:
                raise ValueError("risk codes must be upper snake case")
        if len(set(risk_codes)) != len(risk_codes):
            raise ValueError("risk codes must be unique")
        if type(self.status) is not OrderStatus:
            raise TypeError("status must be an exact OrderStatus")
        if isinstance(self.filled_quantity, bool) or not isinstance(self.filled_quantity, int):
            raise TypeError("filled_quantity must be an exact integer")
        if not 0 <= self.filled_quantity <= self.quantity:
            raise ValueError("filled_quantity must be between zero and quantity")
        if (
            self.cancellation_reason is not None
            and type(self.cancellation_reason) is not CancellationReason
        ):
            raise TypeError("cancellation_reason must be an exact CancellationReason")
        if self.status is OrderStatus.PENDING and (
            self.filled_quantity != 0 or self.cancellation_reason is not None
        ):
            raise ValueError("pending orders cannot have outcomes")
        if self.status is OrderStatus.FILLED and (
            self.filled_quantity != self.quantity or self.cancellation_reason is not None
        ):
            raise ValueError("filled orders require their full quantity and no cancellation")
        if self.status is OrderStatus.PARTIALLY_FILLED and (
            not 0 < self.filled_quantity < self.quantity or self.cancellation_reason is None
        ):
            raise ValueError("partially filled orders require a remainder reason")
        if self.status is OrderStatus.CANCELLED and (
            self.filled_quantity != 0 or self.cancellation_reason is None
        ):
            raise ValueError("cancelled orders require a cancellation reason")
        object.__setattr__(self, "sleeve_weights", sleeve_weights)
        object.__setattr__(self, "risk_codes", risk_codes)


@dataclass(frozen=True, slots=True)
class Fill:
    fill_id: str
    order_id: str
    instrument: InstrumentId
    side: OrderSide
    quantity: int
    trading_day: date
    price: Decimal
    gross_amount: Decimal
    commission: Decimal
    stamp_duty: Decimal
    transfer_fee: Decimal
    total_fee: Decimal
    profile_id: str

    def __post_init__(self) -> None:
        _stable_id(self.fill_id, label="fill_id")
        _stable_id(self.order_id, label="order_id")
        if type(self.instrument) is not InstrumentId:
            raise TypeError("instrument must be an exact InstrumentId")
        if type(self.side) is not OrderSide:
            raise TypeError("side must be an exact OrderSide")
        if (
            isinstance(self.quantity, bool)
            or not isinstance(self.quantity, int)
            or self.quantity <= 0
        ):
            raise ValueError("fill quantity must be a positive integer")
        _exact_date(self.trading_day, label="trading_day")
        exact_decimal(self.price, label="fill price", positive=True)
        if self.price != round_price(self.price):
            raise ValueError("fill price must use the four decimal price boundary")
        for name in (
            "gross_amount",
            "commission",
            "stamp_duty",
            "transfer_fee",
            "total_fee",
        ):
            value = exact_decimal(getattr(self, name), label=name)
            if value != round_money(value):
                raise ValueError(f"{name} must be rounded to cents")
        if self.total_fee != exact_sum(self.commission, self.stamp_duty, self.transfer_fee):
            raise ValueError("total_fee must equal its itemized components")
        expected_gross = round_money(exact_product(self.price, self.quantity))
        if self.gross_amount != expected_gross:
            raise ValueError("gross_amount must equal rounded price times quantity")
        if self.side is OrderSide.BUY and self.stamp_duty != Decimal("0.00"):
            raise ValueError("buy-side stamp duty must be zero")
        _stable_id(self.profile_id, label="profile_id")


@dataclass(frozen=True, slots=True)
class LedgerPosition:
    instrument: InstrumentId
    quantity: int
    available_quantity: int
    unsettled_quantity: int
    average_cost: Decimal
    mark_price: Decimal

    def __post_init__(self) -> None:
        if type(self.instrument) is not InstrumentId:
            raise TypeError("instrument must be an exact InstrumentId")
        for name in ("quantity", "available_quantity", "unsettled_quantity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.available_quantity + self.unsettled_quantity != self.quantity:
            raise ValueError("available and unsettled quantities must equal total quantity")
        exact_decimal(self.average_cost, label="average_cost")
        exact_decimal(self.mark_price, label="mark_price")

    @property
    def market_value(self) -> Decimal:
        return round_money(exact_product(self.mark_price, self.quantity))


@dataclass(frozen=True, slots=True)
class LedgerSnapshot:
    trading_day: date
    cash: Decimal
    withdrawable_cash: Decimal
    positions: tuple[LedgerPosition, ...]

    def __post_init__(self) -> None:
        _exact_date(self.trading_day, label="trading_day")
        cash = exact_decimal(self.cash, label="cash")
        withdrawable = exact_decimal(self.withdrawable_cash, label="withdrawable_cash")
        if cash != round_money(cash) or withdrawable != round_money(withdrawable):
            raise ValueError("ledger cash values must be rounded to cents")
        if withdrawable > cash:
            raise ValueError("withdrawable cash cannot exceed usable cash")
        positions = tuple(self.positions)
        if any(type(item) is not LedgerPosition for item in positions):
            raise TypeError("positions must contain exact LedgerPosition values")
        symbols = [str(item.instrument) for item in positions]
        if symbols != sorted(symbols) or len(set(symbols)) != len(symbols):
            raise ValueError("positions must be unique and sorted")
        object.__setattr__(self, "positions", positions)

    @property
    def equity(self) -> Decimal:
        return round_money(exact_sum(self.cash, *(item.market_value for item in self.positions)))


@dataclass(frozen=True, slots=True)
class SessionExecution:
    orders: tuple[Order, ...]
    fills: tuple[Fill, ...]
    snapshot: LedgerSnapshot
    used_profile_ids: tuple[str, ...]
    warnings: tuple[str, ...]
