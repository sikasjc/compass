from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal
from types import MappingProxyType
from typing import cast

from compass.backtest.market_rules import (
    MarketRuleBook,
    MarketRuleProfile,
    OddLotSellPolicy,
    PriceLimitMode,
    SettlementMode,
)
from compass.data.base import default_instrument_type
from compass.domain.market import AssetType, Instrument, InstrumentId
from compass.risk.base import RiskRule
from compass.risk.engine import RiskEngine
from compass.risk.rules import SingleEtfCapRule, SingleStockCapRule


def local_instruments(
    symbols: Sequence[InstrumentId],
) -> Mapping[InstrumentId, Instrument]:
    return MappingProxyType(
        {
            symbol: Instrument(
                instrument_id=symbol,
                asset_type=default_instrument_type(symbol),
                lot_size=(1 if default_instrument_type(symbol) is AssetType.INDEX else 100),
                same_day_sell=False,
            )
            for symbol in symbols
        }
    )


def _profile(
    instrument: Instrument,
    *,
    fee_confirmed: bool,
    commission_rate: Decimal = Decimal("0.0003"),
    minimum_commission: Decimal = Decimal("5.00"),
    slippage_bps: Decimal = Decimal("0"),
) -> MarketRuleProfile:
    asset_type = instrument.asset_type
    index = asset_type is AssetType.INDEX
    return MarketRuleProfile(
        profile_id=(
            f"standard-{instrument.instrument_id.exchange.value.lower()}-{asset_type.value}-v1"
        ),
        exchange=instrument.instrument_id.exchange,
        asset_type=asset_type,
        effective_from=date(1990, 1, 1),
        effective_to=None,
        buy_lot_size=instrument.lot_size,
        odd_lot_sell_policy=OddLotSellPolicy.POSITION_REMAINDER_ONLY,
        settlement_mode=SettlementMode.T_PLUS_ONE,
        same_day_sell_eligible=False,
        price_limit_mode=(PriceLimitMode.NONE if index else PriceLimitMode.PERCENTAGE),
        price_limit_rate=(None if index else Decimal("0.10")),
        risk_warning_price_limit_rate=(None if index else Decimal("0.05")),
        commission_rate=commission_rate,
        minimum_commission=minimum_commission,
        sell_stamp_duty_rate=(Decimal("0.0005") if asset_type is AssetType.STOCK else Decimal("0")),
        transfer_fee_rate=(
            Decimal("0.00001") if instrument.instrument_id.exchange.value == "SSE" else Decimal("0")
        ),
        slippage_bps=slippage_bps,
        maximum_volume_participation=Decimal("0.10"),
        fee_profile_confirmed=fee_confirmed,
    )


def local_rule_book(
    instruments: Mapping[InstrumentId, Instrument],
    *,
    fee_confirmed: bool,
    commission_rate: Decimal = Decimal("0.0003"),
    minimum_commission: Decimal = Decimal("5.00"),
    slippage_bps: Decimal = Decimal("0"),
) -> MarketRuleBook:
    profiles = {
        (
            item.instrument_id.exchange,
            item.asset_type,
            item.same_day_sell,
        ): _profile(
            item,
            fee_confirmed=fee_confirmed,
            commission_rate=commission_rate,
            minimum_commission=minimum_commission,
            slippage_bps=slippage_bps,
        )
        for item in instruments.values()
    }
    return MarketRuleBook(profiles.values())


def local_profile_ids(
    instruments: Mapping[InstrumentId, Instrument],
    *,
    fee_confirmed: bool,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            profile.profile_id
            for profile in local_rule_book(
                instruments,
                fee_confirmed=fee_confirmed,
            ).profiles
        )
    )


def local_risk_engine(*, active: bool) -> RiskEngine:
    rules: tuple[RiskRule, ...] = ()
    if active:
        rules = (
            cast(
                RiskRule,
                SingleEtfCapRule(maximum_weight=Decimal("0.30")),
            ),
            cast(
                RiskRule,
                SingleStockCapRule(maximum_weight=Decimal("0.10")),
            ),
        )
    return RiskEngine(rules)
