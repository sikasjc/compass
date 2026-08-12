from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

import pandas as pd  # type: ignore[import-untyped]


class Exchange(StrEnum):
    SSE = "SSE"
    SZSE = "SZSE"


class AssetType(StrEnum):
    STOCK = "stock"
    ETF = "etf"
    INDEX = "index"


@dataclass(frozen=True, slots=True)
class InstrumentId:
    exchange: Exchange
    code: str

    @classmethod
    def parse(cls, value: str) -> InstrumentId:
        exchange_text, separator, code = value.upper().partition(".")
        if separator != "." or len(code) != 6 or not code.isdigit():
            raise ValueError(f"invalid instrument id: {value}")
        return cls(Exchange(exchange_text), code)

    def __str__(self) -> str:
        return f"{self.exchange.value}.{self.code}"


@dataclass(frozen=True, slots=True)
class Instrument:
    instrument_id: InstrumentId
    asset_type: AssetType
    lot_size: int
    same_day_sell: bool

    def __post_init__(self) -> None:
        if self.lot_size <= 0:
            raise ValueError("lot size must be positive")


class BarFrame:
    """Validate daily bars indexed by timezone-naive trading dates.

    Intraday, fetch, and task timestamps use Asia/Shanghai elsewhere. Daily bar
    indices instead identify exchange trading dates, must not carry a timezone,
    and must be normalized to midnight.
    """

    REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume", "amount")
    OPTIONAL_COLUMNS = (
        "adjust_factor",
        "adjust_flag",
        "suspended",
        "limit_up",
        "limit_down",
        "previous_close",
        "exchange_reference_price",
        "price_limit_rate",
        "price_limit_rule_id",
        "risk_warning",
        "listing_regime_known",
    )

    @classmethod
    def validate(cls, frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(cls.REQUIRED_COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("bar index must be DatetimeIndex")
        if frame.index.tz is not None:
            raise ValueError("daily bar index must be timezone-naive trading dates")
        if not (frame.index == frame.index.normalize()).all():
            raise ValueError("daily bar index must be normalized midnight trading dates")
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError("bar dates must be unique and sorted")
        numeric_columns = ["open", "high", "low", "close", "volume", "amount"]
        try:
            values_are_finite = frame[numeric_columns].map(isfinite)
        except TypeError:
            raise ValueError("market data must be finite") from None
        if not values_are_finite.all().all():
            raise ValueError("market data must be finite")
        if (frame[["open", "high", "low", "close"]] <= 0).any().any():
            raise ValueError("prices must be positive")
        if (frame[["volume", "amount"]] < 0).any().any():
            raise ValueError("volume and amount must be non-negative")
        if "previous_close" in frame:
            previous_close = frame["previous_close"]
            valid_previous_close = previous_close.map(
                lambda value: (
                    pd.isna(value)
                    or (not isinstance(value, bool) and isfinite(value) and value > 0)
                )
            )
            if not valid_previous_close.all():
                raise ValueError("previous close must be missing or finite and positive")
        if "exchange_reference_price" in frame:
            reference_price = frame["exchange_reference_price"]
            valid_reference_price = reference_price.map(
                lambda value: (
                    pd.isna(value)
                    or (not isinstance(value, bool) and isfinite(value) and value > 0)
                )
            )
            if not valid_reference_price.all():
                raise ValueError("exchange reference price must be missing or finite and positive")
        if "price_limit_rate" in frame:
            price_limit_rate = frame["price_limit_rate"]
            valid_price_limit_rate = price_limit_rate.map(
                lambda value: (
                    pd.isna(value)
                    or (not isinstance(value, bool) and isfinite(value) and 0 < value <= 1)
                )
            )
            if not valid_price_limit_rate.all():
                raise ValueError("price limit rate must be missing or in (0, 1]")
        if "price_limit_rule_id" in frame:
            valid_rule_id = frame["price_limit_rule_id"].map(
                lambda value: (
                    pd.isna(value) or (type(value) is str and value.startswith("cn-price-limit-v"))
                )
            )
            if not valid_rule_id.all():
                raise ValueError("price limit rule id is invalid")
        if "risk_warning" in frame:
            valid_risk_warning = frame["risk_warning"].map(
                lambda value: pd.isna(value) or type(value) is bool
            )
            if not valid_risk_warning.all():
                raise ValueError("risk warning must be missing or an exact bool")
        if "listing_regime_known" in frame:
            valid_listing_regime = frame["listing_regime_known"].map(
                lambda value: pd.isna(value) or type(value) is bool
            )
            if not valid_listing_regime.all():
                raise ValueError("listing regime must be missing or an exact bool")
        if (frame["high"] < frame[["open", "low", "close"]].max(axis=1)).any():
            raise ValueError("high is below another price")
        if (frame["low"] > frame[["open", "high", "close"]].min(axis=1)).any():
            raise ValueError("low is above another price")
        columns = [
            column
            for column in (*cls.REQUIRED_COLUMNS, *cls.OPTIONAL_COLUMNS)
            if column in frame.columns
        ]
        return frame.loc[:, columns].copy()
