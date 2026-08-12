from dataclasses import FrozenInstanceError
from datetime import date, datetime
from decimal import Decimal

import pytest

from compass.backtest.market_rules import (
    MarketRuleBook,
    MarketRuleProfile,
    OddLotSellPolicy,
    PriceLimitMode,
    SettlementMode,
)
from compass.domain.market import AssetType, Exchange, Instrument, InstrumentId


def _profile(
    profile_id: str,
    effective_from: date,
    effective_to: date | None = None,
    **overrides: object,
) -> MarketRuleProfile:
    values: dict[str, object] = {
        "profile_id": profile_id,
        "exchange": Exchange.SSE,
        "asset_type": AssetType.STOCK,
        "effective_from": effective_from,
        "effective_to": effective_to,
        "buy_lot_size": 100,
        "odd_lot_sell_policy": OddLotSellPolicy.POSITION_REMAINDER_ONLY,
        "settlement_mode": SettlementMode.T_PLUS_ONE,
        "same_day_sell_eligible": False,
        "price_limit_mode": PriceLimitMode.PERCENTAGE,
        "price_limit_rate": Decimal("0.10"),
        "risk_warning_price_limit_rate": Decimal("0.05"),
        "commission_rate": Decimal("0.0002"),
        "minimum_commission": Decimal("5"),
        "sell_stamp_duty_rate": Decimal("0.0005"),
        "transfer_fee_rate": Decimal("0.00001"),
        "slippage_bps": Decimal("2"),
        "maximum_volume_participation": Decimal("0.10"),
        "fee_profile_confirmed": True,
    }
    values.update(overrides)
    return MarketRuleProfile(**values)  # type: ignore[arg-type]


def _stock() -> Instrument:
    return Instrument(InstrumentId.parse("SSE.600000"), AssetType.STOCK, 100, False)


def test_rule_book_selects_profile_at_inclusive_effective_boundaries() -> None:
    book = MarketRuleBook(
        (
            _profile("SSE-2026", date(2026, 7, 6)),
            _profile("SSE-2023", date(2023, 1, 1), date(2026, 7, 5)),
        )
    )

    assert book.profile_for(date(2026, 7, 5), _stock()).profile_id == "SSE-2023"
    assert book.profile_for(date(2026, 7, 6), _stock()).profile_id == "SSE-2026"
    assert tuple(profile.profile_id for profile in book.profiles) == ("SSE-2023", "SSE-2026")


def test_rule_book_defensively_copies_an_input_sequence() -> None:
    profiles = [_profile("SSE", date(2026, 1, 1))]
    book = MarketRuleBook(profiles)
    profiles.clear()

    assert book.profile_for(date(2026, 1, 1), _stock()).profile_id == "SSE"
    assert isinstance(book.profiles, tuple)


def test_rule_book_fails_closed_when_no_profile_matches() -> None:
    book = MarketRuleBook((_profile("SSE-2026", date(2026, 7, 6)),))

    with pytest.raises(LookupError, match="no market rule profile"):
        book.profile_for(date(2026, 7, 5), _stock())

    szse = Instrument(InstrumentId.parse("SZSE.000001"), AssetType.STOCK, 100, False)
    with pytest.raises(LookupError, match="no market rule profile"):
        book.profile_for(date(2026, 7, 6), szse)


def test_rule_book_rejects_overlapping_intervals_for_same_market_and_asset() -> None:
    with pytest.raises(ValueError, match="overlapping"):
        MarketRuleBook(
            (
                _profile("FIRST", date(2023, 1, 1), date(2026, 7, 6)),
                _profile("SECOND", date(2026, 7, 6)),
            )
        )


def test_rule_book_allows_concurrent_etf_profiles_with_distinct_turnaround_metadata() -> None:
    t_plus_one = _profile(
        "ETF-T1",
        date(2026, 1, 1),
        asset_type=AssetType.ETF,
        settlement_mode=SettlementMode.T_PLUS_ONE,
        same_day_sell_eligible=False,
    )
    t_plus_zero = _profile(
        "ETF-T0",
        date(2026, 1, 1),
        asset_type=AssetType.ETF,
        settlement_mode=SettlementMode.T_PLUS_ZERO,
        same_day_sell_eligible=True,
    )
    book = MarketRuleBook((t_plus_one, t_plus_zero))
    ordinary_etf = Instrument(InstrumentId.parse("SSE.510300"), AssetType.ETF, 100, False)
    turnaround_etf = Instrument(InstrumentId.parse("SSE.511010"), AssetType.ETF, 100, True)

    assert book.profile_for(date(2026, 7, 21), ordinary_etf).profile_id == "ETF-T1"
    assert book.profile_for(date(2026, 7, 21), turnaround_etf).profile_id == "ETF-T0"


def test_profile_rejects_invalid_interval_and_datetime_dates() -> None:
    with pytest.raises(ValueError, match="effective interval"):
        _profile("BAD", date(2026, 7, 7), date(2026, 7, 6))
    with pytest.raises(TypeError, match="effective_from"):
        _profile("BAD-DATE", datetime(2026, 7, 6), None)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["commission_rate", "minimum_commission", "slippage_bps"])
def test_profile_rejects_inexact_or_negative_decimal_fields(field: str) -> None:
    with pytest.raises(TypeError, match="exact Decimal"):
        _profile("FLOAT", date(2026, 1, 1), **{field: 0.1})
    with pytest.raises(ValueError, match="non-negative"):
        _profile("NEGATIVE", date(2026, 1, 1), **{field: Decimal("-0.1")})


@pytest.mark.parametrize(
    "field", ["commission_rate", "sell_stamp_duty_rate", "transfer_fee_rate"]
)
def test_profile_rejects_fee_rates_above_one(field: str) -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        _profile("BAD-RATE", date(2026, 1, 1), **{field: Decimal("1.01")})


def test_profile_validates_price_limit_configuration() -> None:
    with pytest.raises(ValueError, match="price limit rate is required"):
        _profile("MISSING", date(2026, 1, 1), price_limit_rate=None)
    with pytest.raises(ValueError, match="must be absent"):
        _profile(
            "NO-LIMIT-WITH-RATE",
            date(2026, 1, 1),
            price_limit_mode=PriceLimitMode.NONE,
            price_limit_rate=Decimal("0.10"),
            risk_warning_price_limit_rate=None,
        )

    profile = _profile(
        "NO-LIMIT",
        date(2026, 1, 1),
        price_limit_mode=PriceLimitMode.NONE,
        price_limit_rate=None,
        risk_warning_price_limit_rate=None,
    )
    assert profile.price_limit_rate is None


def test_profile_rejects_inconsistent_settlement_and_same_day_sale() -> None:
    with pytest.raises(ValueError, match="same-day sell"):
        _profile("BAD-T0", date(2026, 1, 1), same_day_sell_eligible=True)


@pytest.mark.parametrize("slippage", [Decimal("10000"), Decimal("10000.01")])
def test_profile_rejects_slippage_that_can_zero_a_sell_price(slippage: Decimal) -> None:
    with pytest.raises(ValueError, match="slippage_bps"):
        _profile("BAD-SLIPPAGE", date(2026, 1, 1), slippage_bps=slippage)


def test_rule_book_requires_exact_domain_types_and_profile_is_immutable() -> None:
    profile = _profile("SSE", date(2026, 1, 1))
    book = MarketRuleBook((profile,))

    with pytest.raises(TypeError, match="instrument"):
        book.profile_for(date(2026, 1, 1), "SSE.600000")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="trading_date"):
        book.profile_for(datetime(2026, 1, 1), _stock())  # type: ignore[arg-type]
    with pytest.raises(FrozenInstanceError):
        profile.buy_lot_size = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "instrument_id",
    [
        InstrumentId(exchange="SSE", code="600000"),  # type: ignore[arg-type]
        InstrumentId(exchange=Exchange.SSE, code="60X000"),
        InstrumentId(exchange=Exchange.SSE, code="１２３４５６"),
    ],
)
def test_rule_book_rejects_malformed_nested_instrument_id(
    instrument_id: InstrumentId,
) -> None:
    book = MarketRuleBook((_profile("SSE", date(2026, 1, 1)),))
    malformed = Instrument(instrument_id, AssetType.STOCK, 100, False)

    with pytest.raises((TypeError, ValueError), match="instrument"):
        book.profile_for(date(2026, 1, 1), malformed)
