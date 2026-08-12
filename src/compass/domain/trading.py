from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP, localcontext

from compass.domain.market import InstrumentId


MAX_ACCOUNT_DECIMAL = Decimal("1E+24")
MAX_POSITION_QUANTITY = 10**15


def _supported_decimal(value: Decimal, *, label: str) -> None:
    if value.copy_abs() > MAX_ACCOUNT_DECIMAL:
        raise ValueError(f"{label} exceeds the supported bound")


@dataclass(frozen=True, slots=True)
class Position:
    instrument: InstrumentId
    quantity: int
    available_quantity: int
    average_cost: Decimal
    mark_price: Decimal

    def __post_init__(self) -> None:
        if type(self.instrument) is not InstrumentId:
            raise TypeError("instrument must be an exact InstrumentId")
        try:
            if InstrumentId.parse(str(self.instrument)) != self.instrument:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("instrument must be canonical") from None
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int):
            raise TypeError("quantity must be an exact integer")
        if self.quantity < 0:
            raise ValueError("quantity must be non-negative")
        if self.quantity > MAX_POSITION_QUANTITY:
            raise ValueError("quantity exceeds the supported bound")
        if isinstance(self.available_quantity, bool) or not isinstance(
            self.available_quantity, int
        ):
            raise TypeError("available quantity must be an exact integer")
        if not 0 <= self.available_quantity <= self.quantity:
            raise ValueError("available quantity must be between zero and quantity")
        for label, value in (
            ("average cost", self.average_cost),
            ("mark price", self.mark_price),
        ):
            if type(value) is not Decimal:
                raise TypeError(f"{label} must be an exact Decimal")
            if not value.is_finite():
                raise ValueError(f"{label} must be finite")
            if value < 0:
                raise ValueError(f"{label} must be non-negative")
            _supported_decimal(value, label=label)
            with localcontext() as context:
                context.prec = 50
                if value * self.quantity > MAX_ACCOUNT_DECIMAL:
                    raise ValueError(f"{label} position value exceeds the supported bound")

    @property
    def market_value(self) -> Decimal:
        with localcontext() as context:
            context.prec = 50
            return (self.mark_price * self.quantity).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    as_of: date
    cash: Decimal
    positions: Sequence[Position]

    def __post_init__(self) -> None:
        if type(self.as_of) is not date or isinstance(self.as_of, datetime):
            raise TypeError("as_of must be an exact date")
        if type(self.cash) is not Decimal:
            raise TypeError("cash must be an exact Decimal")
        if not self.cash.is_finite():
            raise ValueError("cash must be finite")
        if self.cash < 0:
            raise ValueError("cash must be non-negative")
        _supported_decimal(self.cash, label="cash")
        with localcontext() as context:
            context.prec = 50
            rounded_cash = self.cash.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if self.cash != rounded_cash:
            raise ValueError("cash must use the exact cent boundary")
        if not isinstance(self.positions, Sequence) or isinstance(
            self.positions, (str, bytes, bytearray)
        ):
            raise TypeError("positions must be a sequence")
        positions = tuple(self.positions)
        if any(type(position) is not Position for position in positions):
            raise TypeError("positions must contain exact Position values")
        positions = tuple(sorted(positions, key=lambda item: str(item.instrument)))
        symbols = tuple(position.instrument for position in positions)
        if len(set(symbols)) != len(symbols):
            raise ValueError("positions must be unique by instrument")
        object.__setattr__(self, "positions", positions)
        if self.equity > MAX_ACCOUNT_DECIMAL:
            raise ValueError("account equity exceeds the supported bound")

    @property
    def equity(self) -> Decimal:
        with localcontext() as context:
            context.prec = 50
            return (
                self.cash + sum((p.market_value for p in self.positions), Decimal("0"))
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


@dataclass(frozen=True, slots=True)
class TargetIntent:
    strategy_id: str
    instrument: InstrumentId
    target_weight: Decimal
    score: float
    confidence: float
    reason_code: str
    valid_until: date

    def __post_init__(self) -> None:
        if not Decimal("0") <= self.target_weight <= Decimal("1"):
            raise ValueError("target weight must be between zero and one")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")


@dataclass(frozen=True, slots=True)
class CorporateAction:
    instrument: InstrumentId
    ex_date: date
    split_ratio: Decimal = Decimal("1")
    cash_dividend_per_share: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        if not self.split_ratio.is_finite():
            raise ValueError("split ratio must be finite")
        if self.split_ratio <= 0:
            raise ValueError("split ratio must be positive")
        if not self.cash_dividend_per_share.is_finite():
            raise ValueError("cash dividend per share must be finite")
        if self.cash_dividend_per_share < 0:
            raise ValueError("cash dividend per share must be non-negative")
