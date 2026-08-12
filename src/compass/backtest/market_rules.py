from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum
import re

from compass.domain.market import AssetType, Exchange, Instrument, InstrumentId


class SettlementMode(StrEnum):
    T_PLUS_ZERO = "T+0"
    T_PLUS_ONE = "T+1"


class OddLotSellPolicy(StrEnum):
    FORBIDDEN = "forbidden"
    POSITION_REMAINDER_ONLY = "position_remainder_only"
    ALLOWED = "allowed"


class PriceLimitMode(StrEnum):
    PERCENTAGE = "percentage"
    NONE = "none"


def _exact_date(value: object, *, label: str) -> date:
    if type(value) is not date:
        raise TypeError(f"{label} must be an exact date")
    assert isinstance(value, date)
    return value


def _non_negative_decimal(
    value: object, *, label: str, maximum: Decimal | None = None
) -> Decimal:
    if type(value) is not Decimal:
        raise TypeError(f"{label} must be an exact Decimal")
    assert isinstance(value, Decimal)
    if not value.is_finite() or value < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must not exceed {maximum}")
    return value


@dataclass(frozen=True, slots=True)
class MarketRuleProfile:
    """Caller-configured market mechanics valid for one inclusive date interval."""

    profile_id: str
    exchange: Exchange
    asset_type: AssetType
    effective_from: date
    effective_to: date | None
    buy_lot_size: int
    odd_lot_sell_policy: OddLotSellPolicy
    settlement_mode: SettlementMode
    same_day_sell_eligible: bool
    price_limit_mode: PriceLimitMode
    price_limit_rate: Decimal | None
    risk_warning_price_limit_rate: Decimal | None
    commission_rate: Decimal
    minimum_commission: Decimal
    sell_stamp_duty_rate: Decimal
    transfer_fee_rate: Decimal
    slippage_bps: Decimal
    maximum_volume_participation: Decimal
    fee_profile_confirmed: bool

    def __post_init__(self) -> None:
        if type(self.profile_id) is not str or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.profile_id
        ):
            raise ValueError("profile_id must be a stable non-empty identifier")
        if type(self.exchange) is not Exchange:
            raise TypeError("exchange must be an exact Exchange")
        if type(self.asset_type) is not AssetType:
            raise TypeError("asset_type must be an exact AssetType")
        start = _exact_date(self.effective_from, label="effective_from")
        end = (
            None
            if self.effective_to is None
            else _exact_date(self.effective_to, label="effective_to")
        )
        if end is not None and end < start:
            raise ValueError("effective interval must end on or after its start")
        if isinstance(self.buy_lot_size, bool) or not isinstance(self.buy_lot_size, int):
            raise TypeError("buy_lot_size must be an exact integer")
        if self.buy_lot_size <= 0:
            raise ValueError("buy_lot_size must be positive")
        if type(self.odd_lot_sell_policy) is not OddLotSellPolicy:
            raise TypeError("odd_lot_sell_policy must be an exact OddLotSellPolicy")
        if type(self.settlement_mode) is not SettlementMode:
            raise TypeError("settlement_mode must be an exact SettlementMode")
        if type(self.same_day_sell_eligible) is not bool:
            raise TypeError("same_day_sell_eligible must be an exact bool")
        if self.same_day_sell_eligible and self.settlement_mode is not SettlementMode.T_PLUS_ZERO:
            raise ValueError("same-day sell eligibility requires T+0 settlement")
        if type(self.price_limit_mode) is not PriceLimitMode:
            raise TypeError("price_limit_mode must be an exact PriceLimitMode")
        if self.price_limit_mode is PriceLimitMode.NONE:
            if self.price_limit_rate is not None or self.risk_warning_price_limit_rate is not None:
                raise ValueError("price limit rates must be absent when price limits are disabled")
        else:
            if self.price_limit_rate is None:
                raise ValueError("price limit rate is required for percentage limits")
            regular = _non_negative_decimal(
                self.price_limit_rate, label="price_limit_rate", maximum=Decimal("1")
            )
            if regular == 0:
                raise ValueError("price_limit_rate must be positive")
            if self.risk_warning_price_limit_rate is not None:
                warning = _non_negative_decimal(
                    self.risk_warning_price_limit_rate,
                    label="risk_warning_price_limit_rate",
                    maximum=Decimal("1"),
                )
                if warning == 0:
                    raise ValueError("risk_warning_price_limit_rate must be positive")

        for name in ("commission_rate", "sell_stamp_duty_rate", "transfer_fee_rate"):
            _non_negative_decimal(getattr(self, name), label=name, maximum=Decimal("1"))
        _non_negative_decimal(self.minimum_commission, label="minimum_commission")
        slippage = _non_negative_decimal(self.slippage_bps, label="slippage_bps")
        if slippage >= Decimal("10000"):
            raise ValueError("slippage_bps must be less than 10000")
        participation = _non_negative_decimal(
            self.maximum_volume_participation,
            label="maximum_volume_participation",
            maximum=Decimal("1"),
        )
        if participation == 0:
            raise ValueError("maximum_volume_participation must be positive")
        if type(self.fee_profile_confirmed) is not bool:
            raise TypeError("fee_profile_confirmed must be an exact bool")


class MarketRuleBook:
    """Select non-overlapping market profiles by exact domain key and trading date."""

    def __init__(self, profiles: Iterable[MarketRuleProfile]) -> None:
        checked: list[MarketRuleProfile] = []
        profile_ids: set[str] = set()
        for profile in profiles:
            if type(profile) is not MarketRuleProfile:
                raise TypeError("profiles must contain exact MarketRuleProfile values")
            if profile.profile_id in profile_ids:
                raise ValueError(f"duplicate market rule profile id: {profile.profile_id}")
            profile_ids.add(profile.profile_id)
            checked.append(profile)
        checked.sort(
            key=lambda item: (
                item.exchange.value,
                item.asset_type.value,
                item.same_day_sell_eligible,
                item.effective_from,
                item.profile_id,
            )
        )
        previous_by_key: dict[tuple[Exchange, AssetType, bool], MarketRuleProfile] = {}
        for profile in checked:
            key = (
                profile.exchange,
                profile.asset_type,
                profile.same_day_sell_eligible,
            )
            previous = previous_by_key.get(key)
            if previous is not None and (
                previous.effective_to is None
                or profile.effective_from <= previous.effective_to
            ):
                raise ValueError(
                    "overlapping market rule intervals for "
                    f"{profile.exchange.value}/{profile.asset_type.value}/"
                    f"same_day_sell={profile.same_day_sell_eligible}"
                )
            previous_by_key[key] = profile
        self._profiles = tuple(checked)

    @property
    def profiles(self) -> tuple[MarketRuleProfile, ...]:
        return self._profiles

    def profile_for(self, trading_date: date, instrument: Instrument) -> MarketRuleProfile:
        day = _exact_date(trading_date, label="trading_date")
        if type(instrument) is not Instrument:
            raise TypeError("instrument must be an exact Instrument")
        if type(instrument.instrument_id) is not InstrumentId:
            raise TypeError("instrument id must be an exact InstrumentId")
        instrument_id = instrument.instrument_id
        if type(instrument_id.exchange) is not Exchange:
            raise TypeError("instrument exchange must be an exact Exchange")
        if (
            type(instrument_id.code) is not str
            or len(instrument_id.code) != 6
            or re.fullmatch(r"[0-9]{6}", instrument_id.code) is None
        ):
            raise ValueError("instrument code must be a canonical six-digit code")
        if type(instrument.asset_type) is not AssetType:
            raise TypeError("instrument asset type must be an exact AssetType")
        if type(instrument.same_day_sell) is not bool:
            raise TypeError("instrument same-day sell flag must be an exact bool")
        exchange = instrument_id.exchange
        for profile in self._profiles:
            if (
                profile.exchange is not exchange
                or profile.asset_type is not instrument.asset_type
                or profile.same_day_sell_eligible is not instrument.same_day_sell
            ):
                continue
            if profile.effective_from <= day and (
                profile.effective_to is None or day <= profile.effective_to
            ):
                return profile
        raise LookupError(
            "no market rule profile for "
            f"{instrument.instrument_id} ({instrument.asset_type.value}) on {day.isoformat()}"
        )
