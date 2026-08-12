from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from compass.backtest.market_rules import MarketRuleProfile, PriceLimitMode
from compass.domain.market import AssetType, Exchange, Instrument


@dataclass(frozen=True, slots=True)
class EffectivePriceLimits:
    limit_up: Decimal | None
    limit_down: Decimal | None
    state_known: bool


def _cell_decimal(value: object, *, label: str) -> Decimal | None:
    if pd.isna(value):
        return None
    if type(value) is Decimal:
        result = value
    elif isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    elif isinstance(value, (int, float)):
        result = Decimal(str(value))
    else:
        raise TypeError(f"{label} must be numeric")
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _stock_rate(
    instrument: Instrument,
    day: date,
    row: pd.Series,
    profile: MarketRuleProfile,
) -> Decimal | None:
    known = row.get("listing_regime_known")
    if not isinstance(known, (bool, np.bool_)) or not bool(known):
        return None
    attested = _cell_decimal(row.get("price_limit_rate"), label="price_limit_rate")
    if attested is not None:
        return attested
    symbol = instrument.instrument_id
    if (
        symbol.exchange is Exchange.SSE
        and symbol.code.startswith(("688", "689"))
        and day >= date(2019, 7, 22)
    ) or (
        symbol.exchange is Exchange.SZSE
        and symbol.code.startswith(("300", "301"))
        and day >= date(2020, 8, 24)
    ):
        return Decimal("0.20")
    warning = row.get("risk_warning")
    if type(warning) is bool or isinstance(warning, np.bool_):
        return (
            profile.risk_warning_price_limit_rate
            if bool(warning)
            else profile.price_limit_rate
        )
    return None


def effective_price_limits(
    frame: pd.DataFrame,
    timestamp: pd.Timestamp,
    row: pd.Series,
    day: date,
    instrument: Instrument,
    profile: MarketRuleProfile,
    *,
    has_corporate_action: bool,
) -> EffectivePriceLimits:
    explicit_up = _cell_decimal(row.get("limit_up"), label="limit_up")
    explicit_down = _cell_decimal(row.get("limit_down"), label="limit_down")
    if explicit_up is not None or explicit_down is not None:
        return EffectivePriceLimits(
            explicit_up,
            explicit_down,
            explicit_up is not None and explicit_down is not None,
        )
    if profile.price_limit_mode is PriceLimitMode.NONE:
        return EffectivePriceLimits(None, None, True)
    reference_column = (
        "exchange_reference_price" if has_corporate_action else "previous_close"
    )
    reference = _cell_decimal(row.get(reference_column), label=reference_column)
    if reference is None:
        if has_corporate_action:
            return EffectivePriceLimits(None, None, False)
        prior = frame.loc[frame.index < timestamp, "close"]
        if prior.empty:
            return EffectivePriceLimits(None, None, False)
        reference = _cell_decimal(prior.iloc[-1], label="previous_close")
    assert reference is not None
    if instrument.asset_type is AssetType.ETF:
        rate = _cell_decimal(row.get("price_limit_rate"), label="price_limit_rate")
    else:
        rate = _stock_rate(instrument, day, row, profile)
    if rate is None or rate > Decimal("1"):
        return EffectivePriceLimits(None, None, False)
    quantum = Decimal("0.01")
    return EffectivePriceLimits(
        (reference * (Decimal("1") + rate)).quantize(
            quantum, rounding=ROUND_HALF_UP
        ),
        (reference * (Decimal("1") - rate)).quantize(
            quantum, rounding=ROUND_HALF_UP
        ),
        True,
    )
