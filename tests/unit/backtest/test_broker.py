from dataclasses import FrozenInstanceError, replace
from datetime import date
from decimal import Decimal, localcontext

import numpy as np
import pytest

from compass.backtest.broker import (
    Broker,
    DailyExecutionBar,
    InitialPosition,
)
from compass.backtest.market_rules import (
    MarketRuleBook,
    MarketRuleProfile,
    OddLotSellPolicy,
    PriceLimitMode,
    SettlementMode,
)
from compass.backtest.orders import (
    CancellationReason,
    Order,
    OrderSide,
    OrderStatus,
    round_money,
)
from compass.domain.market import AssetType, Exchange, Instrument, InstrumentId
from compass.domain.trading import CorporateAction


DAY = date(2026, 7, 21)


def _instrument(
    symbol: str = "SSE.600000",
    *,
    asset_type: AssetType = AssetType.STOCK,
    same_day_sell: bool = False,
) -> Instrument:
    return Instrument(InstrumentId.parse(symbol), asset_type, 100, same_day_sell)


def _profile(
    profile_id: str = "RULES",
    *,
    asset_type: AssetType = AssetType.STOCK,
    same_day_sell: bool = False,
    effective_from: date = date(2020, 1, 1),
    effective_to: date | None = None,
    odd_lot: OddLotSellPolicy = OddLotSellPolicy.POSITION_REMAINDER_ONLY,
    commission_rate: Decimal = Decimal("0"),
    minimum_commission: Decimal = Decimal("0"),
    stamp_duty: Decimal = Decimal("0"),
    transfer_fee: Decimal = Decimal("0"),
    slippage_bps: Decimal = Decimal("0"),
    volume_participation: Decimal = Decimal("1"),
    confirmed: bool = True,
) -> MarketRuleProfile:
    return MarketRuleProfile(
        profile_id=profile_id,
        exchange=Exchange.SSE,
        asset_type=asset_type,
        effective_from=effective_from,
        effective_to=effective_to,
        buy_lot_size=100,
        odd_lot_sell_policy=odd_lot,
        settlement_mode=(
            SettlementMode.T_PLUS_ZERO if same_day_sell else SettlementMode.T_PLUS_ONE
        ),
        same_day_sell_eligible=same_day_sell,
        price_limit_mode=PriceLimitMode.PERCENTAGE,
        price_limit_rate=Decimal("0.10"),
        risk_warning_price_limit_rate=Decimal("0.05"),
        commission_rate=commission_rate,
        minimum_commission=minimum_commission,
        sell_stamp_duty_rate=stamp_duty,
        transfer_fee_rate=transfer_fee,
        slippage_bps=slippage_bps,
        maximum_volume_participation=volume_participation,
        fee_profile_confirmed=confirmed,
    )


def _bar(
    price: str = "10",
    *,
    volume: int = 100_000,
    suspended: bool = False,
    limit_up: str | None = None,
    limit_down: str | None = None,
) -> DailyExecutionBar:
    value = Decimal(price)
    return DailyExecutionBar(
        open=value,
        close=value,
        volume=volume,
        suspended=suspended,
        limit_up=None if limit_up is None else Decimal(limit_up),
        limit_down=None if limit_down is None else Decimal(limit_down),
    )


def _order(
    instrument: Instrument,
    side: OrderSide,
    quantity: int,
    *,
    order_id: str = "order-1",
) -> Order:
    return Order(
        order_id=order_id,
        instrument=instrument.instrument_id,
        side=side,
        quantity=quantity,
        created_on=date(2026, 7, 20),
        scheduled_for=DAY,
        sleeve_weights={"main": Decimal("1")},
        risk_codes=("SINGLE_STOCK_CAP",),
    )


def _broker(
    instruments: tuple[Instrument, ...],
    profiles: tuple[MarketRuleProfile, ...],
    *,
    cash: Decimal = Decimal("100000"),
    positions: tuple[InitialPosition, ...] = (),
) -> Broker:
    return Broker(
        initial_cash=cash,
        instruments={item.instrument_id: item for item in instruments},
        rule_book=MarketRuleBook(profiles),
        initial_positions=positions,
    )


@pytest.mark.parametrize(
    ("bars", "side", "reason"),
    [
        ({}, OrderSide.BUY, CancellationReason.MISSING_BAR),
        ({"bar": _bar(suspended=True)}, OrderSide.BUY, CancellationReason.SUSPENDED),
        ({"bar": _bar("11", limit_up="11")}, OrderSide.BUY, CancellationReason.LIMIT_UP),
        ({"bar": _bar("9", limit_down="9")}, OrderSide.SELL, CancellationReason.LIMIT_DOWN),
    ],
)
def test_execution_failures_cancel_at_end_of_session_with_stable_reason(
    bars: dict[str, DailyExecutionBar], side: OrderSide, reason: CancellationReason
) -> None:
    instrument = _instrument()
    positions = (InitialPosition(instrument.instrument_id, 100, 100, Decimal("10"), Decimal("10")),)
    broker = _broker((instrument,), (_profile(),), positions=positions)
    market = {} if not bars else {instrument.instrument_id: bars["bar"]}

    result = broker.execute_session(DAY, market, (_order(instrument, side, 100),))

    assert result.orders[0].status is OrderStatus.CANCELLED
    assert result.orders[0].cancellation_reason is reason
    assert result.fills == ()


def test_buy_fee_components_and_side_aware_slippage_are_exact() -> None:
    instrument = _instrument()
    profile = _profile(
        commission_rate=Decimal("0.001"),
        minimum_commission=Decimal("5"),
        transfer_fee=Decimal("0.0001"),
        slippage_bps=Decimal("10"),
    )
    broker = _broker((instrument,), (profile,))

    result = broker.execute_session(
        DAY,
        {instrument.instrument_id: _bar("10")},
        (_order(instrument, OrderSide.BUY, 100),),
    )

    fill = result.fills[0]
    assert fill.price == Decimal("10.0100")
    assert fill.gross_amount == Decimal("1001.00")
    assert fill.commission == Decimal("5.00")
    assert fill.stamp_duty == Decimal("0.00")
    assert fill.transfer_fee == Decimal("0.10")
    assert fill.total_fee == Decimal("5.10")
    assert result.snapshot.cash == Decimal("98993.90")


def test_sell_charges_stamp_duty_and_proceeds_fund_same_day_buy() -> None:
    seller = _instrument("SSE.600000")
    buyer = _instrument("SSE.600001")
    profile = _profile(
        commission_rate=Decimal("0.001"),
        minimum_commission=Decimal("5"),
        stamp_duty=Decimal("0.001"),
        transfer_fee=Decimal("0.0001"),
    )
    broker = _broker(
        (seller, buyer),
        (profile,),
        cash=Decimal("0"),
        positions=(InitialPosition(seller.instrument_id, 100, 100, Decimal("10"), Decimal("10")),),
    )
    orders = (
        _order(buyer, OrderSide.BUY, 100, order_id="buy"),
        _order(seller, OrderSide.SELL, 100, order_id="sell"),
    )

    result = broker.execute_session(
        DAY,
        {seller.instrument_id: _bar("20"), buyer.instrument_id: _bar("10")},
        orders,
    )

    assert [fill.order_id for fill in result.fills] == ["sell", "buy"]
    sale = result.fills[0]
    assert sale.commission == Decimal("5.00")
    assert sale.stamp_duty == Decimal("2.00")
    assert sale.transfer_fee == Decimal("0.20")
    assert result.snapshot.cash == Decimal("987.70")
    assert result.snapshot.withdrawable_cash == Decimal("0.00")

    settled = broker.execute_session(date(2026, 7, 22), {}, ())
    assert settled.snapshot.withdrawable_cash == settled.snapshot.cash == Decimal("987.70")


def test_t_plus_one_buy_is_unavailable_until_next_session() -> None:
    instrument = _instrument()
    broker = _broker((instrument,), (_profile(),))
    buy = broker.execute_session(
        DAY,
        {instrument.instrument_id: _bar()},
        (_order(instrument, OrderSide.BUY, 100),),
    )
    assert buy.snapshot.positions[0].available_quantity == 0
    assert buy.snapshot.positions[0].unsettled_quantity == 100

    same_day_sell = broker.execute_session(
        DAY,
        {instrument.instrument_id: _bar()},
        (_order(instrument, OrderSide.SELL, 100, order_id="sell"),),
    )
    assert same_day_sell.orders[0].cancellation_reason is CancellationReason.T_PLUS_ONE

    next_day = broker.execute_session(date(2026, 7, 22), {}, ())
    assert next_day.snapshot.positions[0].available_quantity == 100
    assert next_day.snapshot.positions[0].unsettled_quantity == 0


def test_same_day_sell_eligible_etf_can_be_bought_then_sold() -> None:
    instrument = _instrument("SSE.511010", asset_type=AssetType.ETF, same_day_sell=True)
    broker = _broker(
        (instrument,),
        (_profile(asset_type=AssetType.ETF, same_day_sell=True),),
    )
    bought = broker.execute_session(
        DAY,
        {instrument.instrument_id: _bar()},
        (_order(instrument, OrderSide.BUY, 100),),
    )
    assert bought.snapshot.positions[0].available_quantity == 100

    sold = broker.execute_session(
        DAY,
        {instrument.instrument_id: _bar()},
        (_order(instrument, OrderSide.SELL, 100, order_id="sell"),),
    )
    assert sold.orders[0].status is OrderStatus.FILLED
    assert sold.snapshot.positions == ()


def test_position_remainder_policy_allows_only_whole_position_odd_lot_sale() -> None:
    instrument = _instrument()
    broker = _broker(
        (instrument,),
        (_profile(),),
        cash=Decimal("100"),
        positions=(
            InitialPosition(instrument.instrument_id, 150, 150, Decimal("8"), Decimal("10")),
        ),
    )

    partial = broker.execute_session(
        DAY,
        {instrument.instrument_id: _bar()},
        (_order(instrument, OrderSide.SELL, 50),),
    )
    assert partial.orders[0].cancellation_reason is CancellationReason.ODD_LOT_NOT_ALLOWED

    whole = broker.execute_session(
        DAY,
        {instrument.instrument_id: _bar()},
        (_order(instrument, OrderSide.SELL, 150, order_id="whole"),),
    )
    assert whole.orders[0].status is OrderStatus.FILLED
    assert whole.snapshot.positions == ()


def test_volume_cap_partially_fills_to_an_executable_lot() -> None:
    instrument = _instrument()
    broker = _broker(
        (instrument,),
        (_profile(volume_participation=Decimal("0.25")),),
    )

    result = broker.execute_session(
        DAY,
        {instrument.instrument_id: _bar(volume=1000)},
        (_order(instrument, OrderSide.BUY, 500),),
    )

    assert result.fills[0].quantity == 200
    assert result.orders[0].status is OrderStatus.PARTIALLY_FILLED
    assert result.orders[0].cancellation_reason is CancellationReason.VOLUME_LIMIT


def test_sell_volume_cap_is_floored_to_a_lot_before_odd_lot_policy() -> None:
    instrument = _instrument()
    broker = _broker(
        (instrument,),
        (_profile(volume_participation=Decimal("0.25")),),
        positions=(
            InitialPosition(instrument.instrument_id, 500, 500, Decimal("8"), Decimal("10")),
        ),
    )

    result = broker.execute_session(
        DAY,
        {instrument.instrument_id: _bar(volume=1000)},
        (_order(instrument, OrderSide.SELL, 500),),
    )

    assert result.fills[0].quantity == 200
    assert result.orders[0].status is OrderStatus.PARTIALLY_FILLED
    assert result.orders[0].cancellation_reason is CancellationReason.VOLUME_LIMIT


def test_proportional_buy_scaling_accounts_for_minimum_fees_and_is_deterministic() -> None:
    first = _instrument("SSE.600000")
    second = _instrument("SSE.600001")
    profile = _profile(minimum_commission=Decimal("5"))
    orders = (
        _order(first, OrderSide.BUY, 200, order_id="b"),
        _order(second, OrderSide.BUY, 200, order_id="a"),
    )

    def run() -> object:
        broker = _broker((first, second), (profile,), cash=Decimal("2010"))
        return broker.execute_session(
            DAY,
            {first.instrument_id: _bar(), second.instrument_id: _bar()},
            orders,
        )

    left = run()
    right = run()
    assert left == right
    assert [(fill.order_id, fill.quantity) for fill in left.fills] == [
        ("b", 100),
        ("a", 100),
    ]
    assert left.snapshot.cash == Decimal("0.00")


def test_execution_date_selects_dated_profile_and_preserves_unconfirmed_warning() -> None:
    instrument = _instrument()
    book = (
        _profile(
            "OLD",
            effective_from=date(2020, 1, 1),
            effective_to=date(2026, 7, 20),
            minimum_commission=Decimal("1"),
        ),
        _profile(
            "NEW",
            effective_from=DAY,
            minimum_commission=Decimal("5"),
            confirmed=False,
        ),
    )
    broker = _broker((instrument,), book)

    result = broker.execute_session(
        DAY,
        {instrument.instrument_id: _bar()},
        (_order(instrument, OrderSide.BUY, 100),),
    )

    assert result.fills[0].profile_id == "NEW"
    assert result.fills[0].commission == Decimal("5.00")
    assert result.used_profile_ids == ("NEW",)
    assert result.warnings == ("FEE_PROFILE_UNCONFIRMED:NEW",)


def test_split_and_cash_dividend_use_pre_action_shares_and_preserve_buckets() -> None:
    instrument = _instrument()
    broker = _broker(
        (instrument,),
        (_profile(),),
        cash=Decimal("100"),
        positions=(
            InitialPosition(instrument.instrument_id, 150, 100, Decimal("8"), Decimal("8")),
        ),
    )
    action = CorporateAction(
        instrument.instrument_id,
        DAY,
        split_ratio=Decimal("2"),
        cash_dividend_per_share=Decimal("0.10"),
    )

    result = broker.execute_session(DAY, {}, (), actions=(action,))

    position = result.snapshot.positions[0]
    # The action preserves both buckets first; the mandated next event then settles T+1.
    assert (position.quantity, position.available_quantity, position.unsettled_quantity) == (
        300,
        300,
        0,
    )
    assert position.average_cost == Decimal("4.0000")
    assert result.snapshot.cash == Decimal("115.00")


def test_fractional_split_and_invalid_public_inputs_are_rejected() -> None:
    instrument = _instrument()
    broker = _broker(
        (instrument,),
        (_profile(),),
        cash=Decimal("100"),
        positions=(
            InitialPosition(instrument.instrument_id, 101, 101, Decimal("10"), Decimal("10")),
        ),
    )
    action = CorporateAction(
        instrument.instrument_id,
        DAY,
        split_ratio=Decimal("1.5"),
        cash_dividend_per_share=Decimal("0.10"),
    )

    with pytest.raises(ValueError, match="fractional"):
        broker.execute_session(DAY, {}, (), actions=(action,))
    unchanged = broker.execute_session(DAY, {}, ())
    assert unchanged.snapshot.cash == Decimal("100.00")
    assert unchanged.snapshot.positions[0].quantity == 101
    with pytest.raises(TypeError, match="Decimal"):
        Broker(1000, {instrument.instrument_id: instrument}, MarketRuleBook((_profile(),)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        _order(instrument, OrderSide.BUY, 0)


def test_order_and_snapshot_are_immutable_and_equity_identity_is_exact() -> None:
    instrument = _instrument()
    broker = _broker((instrument,), (_profile(),))
    result = broker.execute_session(
        DAY,
        {instrument.instrument_id: _bar()},
        (_order(instrument, OrderSide.BUY, 100),),
    )

    assert result.snapshot.equity == Decimal("100000.00")
    assert result.snapshot.cash >= 0
    assert all(
        position.quantity >= position.available_quantity >= 0
        for position in result.snapshot.positions
    )
    with pytest.raises(FrozenInstanceError):
        result.orders[0].quantity = 1  # type: ignore[misc]


def test_cash_and_fee_accounting_is_independent_of_ambient_decimal_precision() -> None:
    instrument = _instrument()
    profile = _profile(
        commission_rate=Decimal("0.001"),
        minimum_commission=Decimal("5"),
        transfer_fee=Decimal("0.0001"),
        slippage_bps=Decimal("10"),
    )

    def execute(precision: int) -> object:
        with localcontext() as context:
            context.prec = precision
            broker = _broker(
                (instrument,),
                (profile,),
                cash=Decimal("123456789.01"),
            )
            return broker.execute_session(
                DAY,
                {instrument.instrument_id: _bar("10")},
                (_order(instrument, OrderSide.BUY, 100),),
            )

    assert execute(6) == execute(50)


def test_company_action_batch_rolls_back_when_a_later_action_is_fractional() -> None:
    first = _instrument("SSE.600000")
    second = _instrument("SSE.600001")
    broker = _broker(
        (first, second),
        (_profile(),),
        cash=Decimal("100"),
        positions=(
            InitialPosition(first.instrument_id, 100, 50, Decimal("9"), Decimal("9")),
            InitialPosition(second.instrument_id, 101, 101, Decimal("10"), Decimal("10")),
        ),
    )
    actions = (
        CorporateAction(
            first.instrument_id,
            DAY,
            split_ratio=Decimal("2"),
            cash_dividend_per_share=Decimal("0.10"),
        ),
        CorporateAction(second.instrument_id, DAY, split_ratio=Decimal("1.5")),
    )

    with pytest.raises(ValueError, match="fractional"):
        broker.execute_session(DAY, {}, (), actions=actions)

    unchanged = broker.execute_session(DAY, {}, ())
    assert unchanged.snapshot.cash == Decimal("100.00")
    assert [(item.quantity, item.average_cost) for item in unchanged.snapshot.positions] == [
        (100, Decimal("9")),
        (101, Decimal("10")),
    ]


def test_three_for_one_split_preserves_market_value_and_cost_basis() -> None:
    instrument = _instrument()
    broker = _broker(
        (instrument,),
        (_profile(),),
        positions=(
            InitialPosition(instrument.instrument_id, 100, 100, Decimal("10"), Decimal("10")),
        ),
    )

    result = broker.execute_session(
        DAY,
        {},
        (),
        actions=(CorporateAction(instrument.instrument_id, DAY, split_ratio=Decimal("3")),),
    )

    position = result.snapshot.positions[0]
    assert position.market_value == Decimal("1000.00")
    assert round_money(position.average_cost * position.quantity) == Decimal("1000.00")


def test_slippage_is_clamped_to_authoritative_price_limits() -> None:
    instrument = _instrument()
    buy_profile = _profile(slippage_bps=Decimal("100"))
    buy = _broker((instrument,), (buy_profile,)).execute_session(
        DAY,
        {instrument.instrument_id: _bar("10", limit_up="10.05")},
        (_order(instrument, OrderSide.BUY, 100),),
    )
    assert buy.fills[0].price == Decimal("10.0500")

    sell = _broker(
        (instrument,),
        (buy_profile,),
        positions=(
            InitialPosition(instrument.instrument_id, 100, 100, Decimal("10"), Decimal("10")),
        ),
    ).execute_session(
        DAY,
        {instrument.instrument_id: _bar("10", limit_down="9.95")},
        (_order(instrument, OrderSide.SELL, 100),),
    )
    assert sell.fills[0].price == Decimal("9.9500")


def test_no_limit_profile_ignores_stale_limit_columns() -> None:
    instrument = _instrument()
    profile = replace(
        _profile(slippage_bps=Decimal("100")),
        price_limit_mode=PriceLimitMode.NONE,
        price_limit_rate=None,
        risk_warning_price_limit_rate=None,
    )

    result = _broker((instrument,), (profile,)).execute_session(
        DAY,
        {instrument.instrument_id: _bar("10", limit_up="9")},
        (_order(instrument, OrderSide.BUY, 100),),
    )

    assert result.orders[0].status is OrderStatus.FILLED
    assert result.fills[0].price == Decimal("10.1000")


def test_daily_volume_budget_is_shared_across_orders_and_repeated_calls() -> None:
    instrument = _instrument()
    broker = _broker((instrument,), (_profile(),))
    bar = {instrument.instrument_id: _bar(volume=100)}
    first = broker.execute_session(
        DAY,
        bar,
        (
            _order(instrument, OrderSide.BUY, 100, order_id="a"),
            _order(instrument, OrderSide.BUY, 100, order_id="b"),
        ),
    )
    second = broker.execute_session(
        DAY,
        bar,
        (_order(instrument, OrderSide.BUY, 100, order_id="c"),),
    )

    assert sum(fill.quantity for fill in first.fills + second.fills) == 100
    assert second.orders[0].cancellation_reason is CancellationReason.VOLUME_LIMIT


def test_repeated_same_day_calls_require_compatible_bars() -> None:
    instrument = _instrument()
    broker = _broker((instrument,), (_profile(),))
    broker.execute_session(DAY, {instrument.instrument_id: _bar("10")}, ())

    with pytest.raises(ValueError, match="compatible"):
        broker.execute_session(DAY, {instrument.instrument_id: _bar("11")}, ())


@pytest.mark.parametrize("bad_key", ["SSE.600001", InstrumentId.parse("SSE.600001")])
def test_broker_rejects_unknown_or_non_domain_bar_keys_before_mutation(
    bad_key: object,
) -> None:
    instrument = _instrument()
    broker = _broker((instrument,), (_profile(),), cash=Decimal("100"))
    action = CorporateAction(
        instrument.instrument_id,
        DAY,
        cash_dividend_per_share=Decimal("1"),
    )

    with pytest.raises((TypeError, ValueError), match="bar"):
        broker.execute_session(DAY, {bad_key: _bar()}, (), actions=(action,))  # type: ignore[dict-item]

    unchanged = broker.execute_session(DAY, {}, ())
    assert unchanged.snapshot.cash == Decimal("100.00")


def test_numpy_boolean_suspension_is_normalized_but_strings_are_rejected() -> None:
    normalized = DailyExecutionBar(
        open=Decimal("10"), close=Decimal("10"), volume=100, suspended=np.bool_(False)
    )
    assert normalized.suspended is False

    with pytest.raises(TypeError, match="bool"):
        DailyExecutionBar(
            open=Decimal("10"),
            close=Decimal("10"),
            volume=100,
            suspended="False",  # type: ignore[arg-type]
        )


def test_cancellation_keeps_selected_profile_provenance_and_warning() -> None:
    instrument = _instrument()
    broker = _broker((instrument,), (_profile("UNCONFIRMED", confirmed=False),))

    result = broker.execute_session(
        DAY,
        {},
        (_order(instrument, OrderSide.BUY, 100),),
    )

    assert result.orders[0].cancellation_reason is CancellationReason.MISSING_BAR
    assert result.used_profile_ids == ("UNCONFIRMED",)
    assert result.warnings == ("FEE_PROFILE_UNCONFIRMED:UNCONFIRMED",)


def test_invalid_sells_have_distinct_position_and_settlement_reasons() -> None:
    instrument = _instrument()
    no_position = _broker((instrument,), (_profile(),)).execute_session(
        DAY,
        {instrument.instrument_id: _bar()},
        (_order(instrument, OrderSide.SELL, 100),),
    )
    assert no_position.orders[0].cancellation_reason is CancellationReason.NO_POSITION

    insufficient = _broker(
        (instrument,),
        (_profile(),),
        positions=(
            InitialPosition(instrument.instrument_id, 100, 100, Decimal("10"), Decimal("10")),
        ),
    ).execute_session(
        DAY,
        {instrument.instrument_id: _bar()},
        (_order(instrument, OrderSide.SELL, 200),),
    )
    assert insufficient.orders[0].cancellation_reason is CancellationReason.INSUFFICIENT_POSITION

    unavailable = _broker((instrument,), (_profile(),))
    unavailable.execute_session(
        date(2026, 7, 20),
        {instrument.instrument_id: _bar()},
        (
            Order(
                order_id="same-day-buy",
                instrument=instrument.instrument_id,
                side=OrderSide.BUY,
                quantity=100,
                created_on=date(2026, 7, 19),
                scheduled_for=date(2026, 7, 20),
                sleeve_weights={},
            ),
        ),
    )
    blocked = unavailable.execute_session(
        date(2026, 7, 20),
        {instrument.instrument_id: _bar()},
        (
            Order(
                order_id="same-day-sell",
                instrument=instrument.instrument_id,
                side=OrderSide.SELL,
                quantity=100,
                created_on=date(2026, 7, 19),
                scheduled_for=date(2026, 7, 20),
                sleeve_weights={},
            ),
        ),
    )
    assert blocked.orders[0].cancellation_reason is CancellationReason.T_PLUS_ONE


def test_extreme_decimal_magnitude_fails_with_stable_validation_error() -> None:
    with pytest.raises(ValueError, match="magnitude"):
        DailyExecutionBar(
            open=Decimal("1E+1000"),
            close=Decimal("1E+1000"),
            volume=100,
        )


@pytest.mark.parametrize(
    ("policy", "expected_status"),
    [
        (OddLotSellPolicy.ALLOWED, OrderStatus.FILLED),
        (OddLotSellPolicy.FORBIDDEN, OrderStatus.CANCELLED),
        (OddLotSellPolicy.POSITION_REMAINDER_ONLY, OrderStatus.CANCELLED),
    ],
)
def test_partial_odd_lot_sell_honors_each_policy(
    policy: OddLotSellPolicy, expected_status: OrderStatus
) -> None:
    instrument = _instrument()
    broker = _broker(
        (instrument,),
        (_profile(odd_lot=policy),),
        positions=(
            InitialPosition(instrument.instrument_id, 150, 150, Decimal("10"), Decimal("10")),
        ),
    )

    result = broker.execute_session(
        DAY,
        {instrument.instrument_id: _bar()},
        (_order(instrument, OrderSide.SELL, 50),),
    )

    assert result.orders[0].status is expected_status
    if policy is OddLotSellPolicy.ALLOWED:
        assert result.fills[0].quantity == 50
    else:
        assert result.orders[0].cancellation_reason is CancellationReason.ODD_LOT_NOT_ALLOWED


def test_allowed_policy_keeps_a_volume_limited_odd_lot_fill() -> None:
    instrument = _instrument()
    broker = _broker(
        (instrument,),
        (
            _profile(
                odd_lot=OddLotSellPolicy.ALLOWED,
                volume_participation=Decimal("0.50"),
            ),
        ),
        positions=(
            InitialPosition(instrument.instrument_id, 150, 150, Decimal("10"), Decimal("10")),
        ),
    )

    result = broker.execute_session(
        DAY,
        {instrument.instrument_id: _bar(volume=100)},
        (_order(instrument, OrderSide.SELL, 150),),
    )

    assert result.fills[0].quantity == 50
    assert result.orders[0].status is OrderStatus.PARTIALLY_FILLED
    assert result.orders[0].cancellation_reason is CancellationReason.VOLUME_LIMIT


def test_rounded_fill_price_stays_inside_sub_tick_price_limits() -> None:
    instrument = _instrument()
    profile = _profile(slippage_bps=Decimal("100"))
    buy = _broker((instrument,), (profile,)).execute_session(
        DAY,
        {instrument.instrument_id: _bar("10", limit_up="10.05005")},
        (_order(instrument, OrderSide.BUY, 100),),
    )
    assert buy.fills[0].price == Decimal("10.0500")
    assert buy.fills[0].price <= Decimal("10.05005")

    sell = _broker(
        (instrument,),
        (profile,),
        positions=(
            InitialPosition(instrument.instrument_id, 100, 100, Decimal("10"), Decimal("10")),
        ),
    ).execute_session(
        DAY,
        {instrument.instrument_id: _bar("10", limit_down="9.95004")},
        (_order(instrument, OrderSide.SELL, 100),),
    )
    assert sell.fills[0].price == Decimal("9.9501")
    assert sell.fills[0].price >= Decimal("9.95004")


@pytest.mark.parametrize("side", [OrderSide.BUY, OrderSide.SELL])
def test_tiny_execution_price_failure_is_atomic_and_allows_later_valid_fill(
    side: OrderSide,
) -> None:
    instrument = _instrument()
    positions = (InitialPosition(instrument.instrument_id, 100, 100, Decimal("10"), Decimal("10")),)
    broker = _broker(
        (instrument,),
        (_profile(),),
        cash=Decimal("1000"),
        positions=positions if side is OrderSide.SELL else (),
    )
    before = (broker.cash, broker.withdrawable_cash, broker.positions)

    with pytest.raises(ValueError, match="price.*positive"):
        broker.execute_session(
            DAY,
            {instrument.instrument_id: _bar("0.0000000001", volume=100)},
            (_order(instrument, side, 100, order_id="tiny"),),
        )

    assert (broker.cash, broker.withdrawable_cash, broker.positions) == before
    valid = broker.execute_session(
        DAY,
        {instrument.instrument_id: _bar("10", volume=100)},
        (_order(instrument, side, 100, order_id="valid"),),
    )
    assert valid.orders[0].status is OrderStatus.FILLED
    assert valid.fills[0].quantity == 100
