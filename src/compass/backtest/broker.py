from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from types import MappingProxyType

import numpy as np

from compass.backtest.market_rules import (
    MarketRuleBook,
    MarketRuleProfile,
    OddLotSellPolicy,
    PriceLimitMode,
    SettlementMode,
)
from compass.backtest.orders import (
    CancellationReason,
    Fill,
    LedgerPosition,
    LedgerSnapshot,
    Order,
    OrderSide,
    OrderStatus,
    PRICE_QUANTUM,
    SessionExecution,
    exact_decimal,
    exact_product,
    exact_subtract,
    exact_sum,
    round_money,
    round_price,
)
from compass.domain.market import Instrument, InstrumentId
from compass.domain.trading import CorporateAction


@dataclass(frozen=True, slots=True)
class DailyExecutionBar:
    open: Decimal
    close: Decimal
    volume: int
    suspended: bool = False
    limit_up: Decimal | None = None
    limit_down: Decimal | None = None
    price_limit_state_known: bool = True

    def __post_init__(self) -> None:
        exact_decimal(self.open, label="open", positive=True)
        exact_decimal(self.close, label="close", positive=True)
        if isinstance(self.volume, bool) or not isinstance(self.volume, int) or self.volume < 0:
            raise ValueError("volume must be a non-negative integer")
        if isinstance(self.suspended, np.bool_):
            object.__setattr__(self, "suspended", bool(self.suspended))
        elif type(self.suspended) is not bool:
            raise TypeError("suspended must be an exact bool or NumPy bool scalar")
        if type(self.price_limit_state_known) is not bool:
            raise TypeError("price_limit_state_known must be an exact bool")
        for name in ("limit_up", "limit_down"):
            value = getattr(self, name)
            if value is not None:
                exact_decimal(value, label=name, positive=True)
        if self.limit_up is not None and self.limit_down is not None:
            if self.limit_down > self.limit_up:
                raise ValueError("limit_down cannot exceed limit_up")


@dataclass(frozen=True, slots=True)
class InitialPosition:
    instrument: InstrumentId
    quantity: int
    available_quantity: int
    average_cost: Decimal
    mark_price: Decimal

    def __post_init__(self) -> None:
        if type(self.instrument) is not InstrumentId:
            raise TypeError("instrument must be an exact InstrumentId")
        if (
            isinstance(self.quantity, bool)
            or not isinstance(self.quantity, int)
            or self.quantity <= 0
        ):
            raise ValueError("initial quantity must be a positive integer")
        if (
            isinstance(self.available_quantity, bool)
            or not isinstance(self.available_quantity, int)
            or not 0 <= self.available_quantity <= self.quantity
        ):
            raise ValueError("available quantity must be between zero and quantity")
        exact_decimal(self.average_cost, label="average_cost")
        exact_decimal(self.mark_price, label="mark_price")


@dataclass(slots=True)
class _Holding:
    quantity: int
    available: int
    unsettled: int
    average_cost: Decimal
    mark_price: Decimal


@dataclass(frozen=True, slots=True)
class _BuyCandidate:
    order: Order
    instrument: Instrument
    profile: MarketRuleProfile
    bar: DailyExecutionBar
    price: Decimal
    maximum_quantity: int
    volume_limited: bool


def _date(value: object, *, label: str) -> date:
    if type(value) is not date:
        raise TypeError(f"{label} must be an exact date")
    assert isinstance(value, date)
    return value


def _ratio_floor(value: int, ratio: Decimal) -> int:
    numerator, denominator = ratio.as_integer_ratio()
    return value * numerator // denominator


def _divide(value: Decimal, divisor: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = 50
        return value / divisor


class Broker:
    """Deterministic daily broker with cents/half-up cash accounting.

    The bar's full-session volume is used only as a conservative participation cap;
    neither its close nor any future price is used to choose the open fill price.
    """

    def __init__(
        self,
        initial_cash: Decimal,
        instruments: Mapping[InstrumentId, Instrument],
        rule_book: MarketRuleBook,
        initial_positions: Sequence[InitialPosition] = (),
    ) -> None:
        cash = exact_decimal(initial_cash, label="initial_cash")
        if cash != round_money(cash):
            raise ValueError("initial_cash must be rounded to cents")
        if not isinstance(instruments, Mapping):
            raise TypeError("instruments must be a mapping")
        checked: dict[InstrumentId, Instrument] = {}
        for key, instrument in instruments.items():
            if type(key) is not InstrumentId or type(instrument) is not Instrument:
                raise TypeError("instruments must map InstrumentId to exact Instrument values")
            if key != instrument.instrument_id:
                raise ValueError("instrument key must match its identifier")
            if key in checked:
                raise ValueError("instruments must be unique")
            checked[key] = instrument
        if type(rule_book) is not MarketRuleBook:
            raise TypeError("rule_book must be an exact MarketRuleBook")
        holdings: dict[InstrumentId, _Holding] = {}
        for position in tuple(initial_positions):
            if type(position) is not InitialPosition:
                raise TypeError("initial_positions must contain exact InitialPosition values")
            if position.instrument not in checked:
                raise ValueError("initial position instrument is not configured")
            if position.instrument in holdings:
                raise ValueError("initial positions must be unique")
            holdings[position.instrument] = _Holding(
                position.quantity,
                position.available_quantity,
                position.quantity - position.available_quantity,
                position.average_cost,
                position.mark_price,
            )
        self._cash = cash
        self._withdrawable_cash = cash
        self._pending_withdrawable = Decimal("0.00")
        self._instruments = MappingProxyType(dict(checked))
        self._rule_book = rule_book
        self._holdings = holdings
        self._current_day: date | None = None
        self._volume_used: dict[InstrumentId, int] = {}
        self._session_bars: dict[InstrumentId, DailyExecutionBar] = {}

    @property
    def cash(self) -> Decimal:
        return self._cash

    @property
    def withdrawable_cash(self) -> Decimal:
        return self._withdrawable_cash

    def _snapshot(self, day: date) -> LedgerSnapshot:
        positions = tuple(
            LedgerPosition(
                instrument=symbol,
                quantity=holding.quantity,
                available_quantity=holding.available,
                unsettled_quantity=holding.unsettled,
                average_cost=holding.average_cost,
                mark_price=holding.mark_price,
            )
            for symbol, holding in sorted(self._holdings.items(), key=lambda item: str(item[0]))
            if holding.quantity
        )
        return LedgerSnapshot(day, self._cash, self._withdrawable_cash, positions)

    @property
    def positions(self) -> tuple[LedgerPosition, ...]:
        day = self._current_day or date.min
        return self._snapshot(day).positions

    def _start_day(self, day: date, actions: Sequence[CorporateAction]) -> None:
        if self._current_day is not None and day < self._current_day:
            raise ValueError("broker sessions must be non-decreasing")
        if self._current_day == day:
            if actions:
                raise ValueError("corporate actions can only be supplied on first session call")
            return
        checked_actions = tuple(actions)
        identities: set[tuple[InstrumentId, date]] = set()
        for action in checked_actions:
            if type(action) is not CorporateAction:
                raise TypeError("actions must contain exact CorporateAction values")
            identity = (action.instrument, action.ex_date)
            if identity in identities:
                raise ValueError("duplicate corporate action")
            identities.add(identity)
            if action.ex_date != day:
                raise ValueError("corporate action ex-date must match the session")
            if action.instrument not in self._instruments:
                raise ValueError("corporate action instrument is not configured")
        shadow_holdings = {
            symbol: _Holding(
                holding.quantity,
                holding.available,
                holding.unsettled,
                holding.average_cost,
                holding.mark_price,
            )
            for symbol, holding in self._holdings.items()
        }
        shadow_cash = self._cash
        shadow_withdrawable = self._withdrawable_cash
        for action in sorted(checked_actions, key=lambda item: str(item.instrument)):
            holding = shadow_holdings.get(action.instrument)
            if holding is None:
                continue
            pre_split_quantity = holding.quantity
            dividend = round_money(
                exact_product(action.cash_dividend_per_share, pre_split_quantity)
            )
            numerator, denominator = action.split_ratio.as_integer_ratio()
            quantities: list[int] = []
            for value in (holding.quantity, holding.available, holding.unsettled):
                product, remainder = divmod(value * numerator, denominator)
                if remainder:
                    raise ValueError("corporate action would create fractional shares")
                quantities.append(product)
            adjusted_average_cost = _divide(holding.average_cost, action.split_ratio)
            adjusted_mark_price = _divide(holding.mark_price, action.split_ratio)
            exact_decimal(adjusted_average_cost, label="split-adjusted average cost")
            exact_decimal(adjusted_mark_price, label="split-adjusted mark price")
            shadow_cash = round_money(exact_sum(shadow_cash, dividend))
            shadow_withdrawable = round_money(exact_sum(shadow_withdrawable, dividend))
            holding.quantity, holding.available, holding.unsettled = quantities
            holding.average_cost = adjusted_average_cost
            holding.mark_price = adjusted_mark_price
        for holding in shadow_holdings.values():
            holding.available += holding.unsettled
            holding.unsettled = 0
        shadow_withdrawable = round_money(
            exact_sum(shadow_withdrawable, self._pending_withdrawable)
        )
        self._holdings = shadow_holdings
        self._cash = shadow_cash
        self._withdrawable_cash = shadow_withdrawable
        self._pending_withdrawable = Decimal("0.00")
        self._current_day = day
        self._volume_used = {}
        self._session_bars = {}

    @staticmethod
    def _cancel(order: Order, reason: CancellationReason) -> Order:
        return replace(
            order,
            status=OrderStatus.CANCELLED,
            cancellation_reason=reason,
        )

    @staticmethod
    def _finish(order: Order, quantity: int, reason: CancellationReason | None) -> Order:
        if quantity == order.quantity:
            return replace(order, status=OrderStatus.FILLED, filled_quantity=quantity)
        if quantity > 0:
            assert reason is not None
            return replace(
                order,
                status=OrderStatus.PARTIALLY_FILLED,
                filled_quantity=quantity,
                cancellation_reason=reason,
            )
        assert reason is not None
        return Broker._cancel(order, reason)

    @staticmethod
    def _fill_price(side: OrderSide, bar: DailyExecutionBar, profile: MarketRuleProfile) -> Decimal:
        with localcontext() as context:
            context.prec = 50
            direction = Decimal("1") if side is OrderSide.BUY else Decimal("-1")
            multiplier = Decimal("1") + direction * profile.slippage_bps / Decimal("10000")
            slipped = bar.open * multiplier
            if profile.price_limit_mode is PriceLimitMode.PERCENTAGE:
                if side is OrderSide.BUY and bar.limit_up is not None:
                    slipped = min(slipped, bar.limit_up)
                if side is OrderSide.SELL and bar.limit_down is not None:
                    slipped = max(slipped, bar.limit_down)
            rounded = round_price(slipped)
            if (
                profile.price_limit_mode is PriceLimitMode.PERCENTAGE
                and side is OrderSide.BUY
                and bar.limit_up is not None
                and rounded > bar.limit_up
            ):
                rounded = bar.limit_up.quantize(PRICE_QUANTUM, rounding=ROUND_FLOOR)
            if (
                profile.price_limit_mode is PriceLimitMode.PERCENTAGE
                and side is OrderSide.SELL
                and bar.limit_down is not None
                and rounded < bar.limit_down
            ):
                rounded = bar.limit_down.quantize(PRICE_QUANTUM, rounding=ROUND_CEILING)
            return exact_decimal(rounded, label="execution price", positive=True)

    @staticmethod
    def _fees(
        side: OrderSide, quantity: int, price: Decimal, profile: MarketRuleProfile
    ) -> tuple[Decimal, Decimal, Decimal, Decimal, Decimal]:
        gross = round_money(exact_product(price, quantity))
        commission = round_money(
            max(profile.minimum_commission, exact_product(gross, profile.commission_rate))
        )
        stamp = (
            round_money(exact_product(gross, profile.sell_stamp_duty_rate))
            if side is OrderSide.SELL
            else Decimal("0.00")
        )
        transfer = round_money(exact_product(gross, profile.transfer_fee_rate))
        total = round_money(exact_sum(commission, stamp, transfer))
        return gross, commission, stamp, transfer, total

    @classmethod
    def _buy_cost(cls, candidate: _BuyCandidate, quantity: int) -> Decimal:
        if quantity == 0:
            return Decimal("0.00")
        gross, _, _, _, fee = cls._fees(OrderSide.BUY, quantity, candidate.price, candidate.profile)
        return round_money(exact_sum(gross, fee))

    def _common_precheck(
        self, order: Order, bars: Mapping[InstrumentId, DailyExecutionBar]
    ) -> tuple[
        Instrument,
        MarketRuleProfile,
        DailyExecutionBar | None,
        CancellationReason | None,
    ]:
        instrument = self._instruments.get(order.instrument)
        if instrument is None:
            raise ValueError(f"order instrument is not configured: {order.instrument}")
        assert self._current_day is not None
        if order.scheduled_for != self._current_day:
            raise ValueError("order is not scheduled for this session")
        profile = self._rule_book.profile_for(self._current_day, instrument)
        bar = bars.get(order.instrument)
        if bar is None:
            return instrument, profile, None, CancellationReason.MISSING_BAR
        if bar.suspended:
            return instrument, profile, bar, CancellationReason.SUSPENDED
        if (
            profile.price_limit_mode is PriceLimitMode.PERCENTAGE
            and not bar.price_limit_state_known
        ):
            return instrument, profile, bar, CancellationReason.MARKET_STATUS_UNKNOWN
        if (
            profile.price_limit_mode is PriceLimitMode.PERCENTAGE
            and order.side is OrderSide.BUY
            and bar.limit_up is not None
            and bar.open >= bar.limit_up
        ):
            return instrument, profile, bar, CancellationReason.LIMIT_UP
        if (
            profile.price_limit_mode is PriceLimitMode.PERCENTAGE
            and order.side is OrderSide.SELL
            and bar.limit_down is not None
            and bar.open <= bar.limit_down
        ):
            return instrument, profile, bar, CancellationReason.LIMIT_DOWN
        return instrument, profile, bar, None

    def _remaining_volume(
        self,
        instrument: InstrumentId,
        bar: DailyExecutionBar,
        profile: MarketRuleProfile,
    ) -> int:
        limit = _ratio_floor(bar.volume, profile.maximum_volume_participation)
        return max(0, limit - self._volume_used.get(instrument, 0))

    def _volume_quantity(
        self, order: Order, bar: DailyExecutionBar, profile: MarketRuleProfile
    ) -> int:
        return min(
            order.quantity,
            self._remaining_volume(order.instrument, bar, profile),
        )

    def _record_volume(self, instrument: InstrumentId, quantity: int) -> None:
        self._volume_used[instrument] = self._volume_used.get(instrument, 0) + quantity

    def _execute_sell(
        self,
        order: Order,
        bars: Mapping[InstrumentId, DailyExecutionBar],
    ) -> tuple[Order, Fill | None, str | None]:
        instrument, profile, bar, cancellation = self._common_precheck(order, bars)
        if cancellation is not None:
            return self._cancel(order, cancellation), None, profile.profile_id
        assert bar is not None
        holding = self._holdings.get(order.instrument)
        if holding is None:
            return self._cancel(order, CancellationReason.NO_POSITION), None, profile.profile_id
        if order.quantity > holding.quantity:
            return (
                self._cancel(order, CancellationReason.INSUFFICIENT_POSITION),
                None,
                profile.profile_id,
            )
        if order.quantity > holding.available:
            return self._cancel(order, CancellationReason.T_PLUS_ONE), None, profile.profile_id
        quantity = self._volume_quantity(order, bar, profile)
        volume_limited = quantity < order.quantity
        lot = profile.buy_lot_size
        if quantity % lot:
            whole_position_remainder = (
                profile.odd_lot_sell_policy is OddLotSellPolicy.POSITION_REMAINDER_ONLY
                and quantity == holding.quantity
                and quantity == order.quantity
            )
            if profile.odd_lot_sell_policy is OddLotSellPolicy.ALLOWED:
                pass
            elif not whole_position_remainder and volume_limited:
                quantity = quantity // lot * lot
            elif not whole_position_remainder:
                return (
                    self._cancel(order, CancellationReason.ODD_LOT_NOT_ALLOWED),
                    None,
                    profile.profile_id,
                )
        if quantity <= 0:
            return self._cancel(order, CancellationReason.VOLUME_LIMIT), None, profile.profile_id
        price = self._fill_price(OrderSide.SELL, bar, profile)
        gross, commission, stamp, transfer, total_fee = self._fees(
            OrderSide.SELL, quantity, price, profile
        )
        proceeds = round_money(exact_subtract(gross, total_fee))
        if proceeds < 0:
            return (
                self._cancel(order, CancellationReason.END_OF_DAY_UNFILLED),
                None,
                profile.profile_id,
            )
        new_cash = round_money(exact_sum(self._cash, proceeds))
        new_pending_withdrawable = round_money(exact_sum(self._pending_withdrawable, proceeds))
        day = self._current_day
        assert day is not None
        fill = Fill(
            fill_id=f"fill:{order.order_id}",
            order_id=order.order_id,
            instrument=order.instrument,
            side=OrderSide.SELL,
            quantity=quantity,
            trading_day=day,
            price=price,
            gross_amount=gross,
            commission=commission,
            stamp_duty=stamp,
            transfer_fee=transfer,
            total_fee=total_fee,
            profile_id=profile.profile_id,
        )
        reason = CancellationReason.VOLUME_LIMIT if volume_limited else None
        outcome = self._finish(order, quantity, reason)
        holding.quantity -= quantity
        holding.available -= quantity
        self._cash = new_cash
        self._pending_withdrawable = new_pending_withdrawable
        self._record_volume(order.instrument, quantity)
        if holding.quantity == 0:
            del self._holdings[order.instrument]
        return outcome, fill, profile.profile_id

    def _prepare_buy(
        self,
        order: Order,
        bars: Mapping[InstrumentId, DailyExecutionBar],
    ) -> _BuyCandidate | tuple[Order, str | None]:
        instrument, profile, bar, cancellation = self._common_precheck(order, bars)
        if cancellation is not None:
            return self._cancel(order, cancellation), profile.profile_id
        assert bar is not None
        if order.quantity % profile.buy_lot_size:
            return self._cancel(order, CancellationReason.LOT_SIZE), profile.profile_id
        volume_quantity = self._volume_quantity(order, bar, profile)
        maximum = volume_quantity // profile.buy_lot_size * profile.buy_lot_size
        if maximum <= 0:
            return self._cancel(order, CancellationReason.VOLUME_LIMIT), profile.profile_id
        return _BuyCandidate(
            order,
            instrument,
            profile,
            bar,
            self._fill_price(OrderSide.BUY, bar, profile),
            maximum,
            maximum < order.quantity,
        )

    def _allocate_buys(self, candidates: Sequence[_BuyCandidate]) -> dict[str, int]:
        ordered = sorted(
            candidates, key=lambda item: (str(item.order.instrument), item.order.order_id)
        )
        if (
            exact_sum(*(self._buy_cost(item, item.maximum_quantity) for item in ordered))
            <= self._cash
        ):
            return {item.order.order_id: item.maximum_quantity for item in ordered}
        scale = 1_000_000_000_000

        def quantities_at(factor: int) -> dict[str, int]:
            return {
                item.order.order_id: (
                    (item.maximum_quantity * factor // scale)
                    // item.profile.buy_lot_size
                    * item.profile.buy_lot_size
                )
                for item in ordered
            }

        def cost(values: Mapping[str, int]) -> Decimal:
            return exact_sum(
                *(self._buy_cost(item, values[item.order.order_id]) for item in ordered)
            )

        low, high = 0, scale
        while low < high:
            midpoint = (low + high + 1) // 2
            if cost(quantities_at(midpoint)) <= self._cash:
                low = midpoint
            else:
                high = midpoint - 1
        quantities = quantities_at(low)
        while True:
            ranked = sorted(
                ordered,
                key=lambda item: (
                    -((item.maximum_quantity * low) % scale),
                    str(item.order.instrument),
                    item.order.order_id,
                ),
            )
            changed = False
            for item in ranked:
                order_id = item.order.order_id
                proposed = quantities[order_id] + item.profile.buy_lot_size
                if proposed > item.maximum_quantity:
                    continue
                trial = dict(quantities)
                trial[order_id] = proposed
                if cost(trial) <= self._cash:
                    quantities = trial
                    changed = True
            if not changed:
                return quantities

    def _share_buy_volume_budget(
        self, candidates: Sequence[_BuyCandidate]
    ) -> tuple[_BuyCandidate, ...]:
        grouped: dict[InstrumentId, list[_BuyCandidate]] = {}
        for candidate in candidates:
            grouped.setdefault(candidate.order.instrument, []).append(candidate)
        adjusted: list[_BuyCandidate] = []
        for symbol in sorted(grouped, key=str):
            group = sorted(grouped[symbol], key=lambda item: item.order.order_id)
            first = group[0]
            remaining = self._remaining_volume(symbol, first.bar, first.profile)
            requested = sum(item.maximum_quantity for item in group)
            if requested <= remaining:
                adjusted.extend(group)
                continue
            quantities: dict[str, int] = {}
            remainders: dict[str, int] = {}
            allocated = 0
            for item in group:
                lot = item.profile.buy_lot_size
                numerator = item.maximum_quantity * remaining
                raw, remainder = divmod(numerator, requested)
                quantity = raw // lot * lot
                quantities[item.order.order_id] = quantity
                remainders[item.order.order_id] = remainder
                allocated += quantity
            ranked = sorted(
                group,
                key=lambda item: (-remainders[item.order.order_id], item.order.order_id),
            )
            while True:
                changed = False
                for item in ranked:
                    order_id = item.order.order_id
                    proposed = quantities[order_id] + item.profile.buy_lot_size
                    if (
                        proposed <= item.maximum_quantity
                        and allocated + item.profile.buy_lot_size <= remaining
                    ):
                        quantities[order_id] = proposed
                        allocated += item.profile.buy_lot_size
                        changed = True
                if not changed:
                    break
            adjusted.extend(
                replace(
                    item,
                    maximum_quantity=quantities[item.order.order_id],
                    volume_limited=quantities[item.order.order_id] < item.order.quantity,
                )
                for item in group
            )
        return tuple(adjusted)

    def _execute_buy(self, candidate: _BuyCandidate, quantity: int) -> tuple[Order, Fill | None]:
        order = candidate.order
        if quantity <= 0:
            zero_quantity_reason = (
                CancellationReason.VOLUME_LIMIT
                if candidate.volume_limited
                else CancellationReason.INSUFFICIENT_CASH
            )
            return self._cancel(order, zero_quantity_reason), None
        gross, commission, stamp, transfer, total_fee = self._fees(
            OrderSide.BUY, quantity, candidate.price, candidate.profile
        )
        cost = round_money(exact_sum(gross, total_fee))
        if cost > self._cash:
            raise RuntimeError("proportional buy allocation exceeded available cash")
        holding = self._holdings.get(order.instrument)
        old_quantity = 0 if holding is None else holding.quantity
        old_cost = (
            Decimal("0") if holding is None else exact_product(holding.average_cost, old_quantity)
        )
        new_quantity = old_quantity + quantity
        average_cost = round_price(_divide(exact_sum(old_cost, cost), Decimal(new_quantity)))
        available_delta = (
            quantity
            if candidate.profile.settlement_mode is SettlementMode.T_PLUS_ZERO
            and candidate.profile.same_day_sell_eligible
            else 0
        )
        spend_withdrawable = min(self._withdrawable_cash, cost)
        new_withdrawable_cash = round_money(
            exact_subtract(self._withdrawable_cash, spend_withdrawable)
        )
        spend_pending = round_money(exact_subtract(cost, spend_withdrawable))
        if spend_pending > self._pending_withdrawable:
            raise RuntimeError("cash buckets do not cover the allocated buy")
        new_pending_withdrawable = round_money(
            exact_subtract(self._pending_withdrawable, spend_pending)
        )
        new_cash = round_money(exact_subtract(self._cash, cost))
        day = self._current_day
        assert day is not None
        fill = Fill(
            fill_id=f"fill:{order.order_id}",
            order_id=order.order_id,
            instrument=order.instrument,
            side=OrderSide.BUY,
            quantity=quantity,
            trading_day=day,
            price=candidate.price,
            gross_amount=gross,
            commission=commission,
            stamp_duty=stamp,
            transfer_fee=transfer,
            total_fee=total_fee,
            profile_id=candidate.profile.profile_id,
        )
        reason: CancellationReason | None = None
        if quantity < order.quantity:
            reason = (
                CancellationReason.VOLUME_LIMIT
                if candidate.volume_limited and quantity == candidate.maximum_quantity
                else CancellationReason.INSUFFICIENT_CASH
            )
        outcome = self._finish(order, quantity, reason)
        if holding is None:
            self._holdings[order.instrument] = _Holding(
                new_quantity,
                available_delta,
                quantity - available_delta,
                average_cost,
                candidate.bar.close,
            )
        else:
            holding.quantity = new_quantity
            holding.available += available_delta
            holding.unsettled += quantity - available_delta
            holding.average_cost = average_cost
            holding.mark_price = candidate.bar.close
        self._withdrawable_cash = new_withdrawable_cash
        self._pending_withdrawable = new_pending_withdrawable
        self._cash = new_cash
        self._record_volume(order.instrument, quantity)
        return outcome, fill

    def execute_session(
        self,
        trading_day: date,
        bars: Mapping[InstrumentId, DailyExecutionBar],
        orders: Sequence[Order],
        *,
        actions: Sequence[CorporateAction] = (),
    ) -> SessionExecution:
        day = _date(trading_day, label="trading_day")
        if not isinstance(bars, Mapping):
            raise TypeError("bars must be a mapping")
        checked_bars: dict[InstrumentId, DailyExecutionBar] = {}
        for symbol, bar in bars.items():
            if type(symbol) is not InstrumentId:
                raise TypeError("bar keys must be exact InstrumentId values")
            if symbol not in self._instruments:
                raise ValueError(f"bar instrument is not configured: {symbol}")
            if type(bar) is not DailyExecutionBar:
                raise TypeError("bars must contain exact DailyExecutionBar values")
            existing = self._session_bars.get(symbol) if self._current_day == day else None
            if existing is not None and existing != bar:
                raise ValueError("repeated same-day calls require compatible bars")
            checked_bars[symbol] = bar
        checked_orders = tuple(orders)
        if any(type(order) is not Order for order in checked_orders):
            raise TypeError("orders must contain exact Order values")
        order_ids = [order.order_id for order in checked_orders]
        if len(set(order_ids)) != len(order_ids):
            raise ValueError("duplicate order id in session")
        if any(order.status is not OrderStatus.PENDING for order in checked_orders):
            raise ValueError("only pending orders can execute")
        for order in checked_orders:
            instrument = self._instruments.get(order.instrument)
            if instrument is None:
                raise ValueError(f"order instrument is not configured: {order.instrument}")
            if order.scheduled_for != day:
                raise ValueError("order is not scheduled for this session")
            profile = self._rule_book.profile_for(day, instrument)
            execution_bar = checked_bars.get(order.instrument)
            if execution_bar is not None and not execution_bar.suspended:
                self._fill_price(order.side, execution_bar, profile)
        self._start_day(day, actions)
        self._session_bars.update(checked_bars)

        outcomes: dict[str, Order] = {}
        fills: list[Fill] = []
        used_profiles: set[str] = set()
        warnings: set[str] = set()
        sells = sorted(
            (order for order in checked_orders if order.side is OrderSide.SELL),
            key=lambda item: (str(item.instrument), item.order_id),
        )
        for order in sells:
            outcome, fill, profile_id = self._execute_sell(order, checked_bars)
            outcomes[order.order_id] = outcome
            if profile_id is not None:
                used_profiles.add(profile_id)
                profile = self._rule_book.profile_for(day, self._instruments[order.instrument])
                if not profile.fee_profile_confirmed:
                    warnings.add(f"FEE_PROFILE_UNCONFIRMED:{profile_id}")
            if fill is not None:
                fills.append(fill)

        candidates: list[_BuyCandidate] = []
        buys = sorted(
            (order for order in checked_orders if order.side is OrderSide.BUY),
            key=lambda item: (str(item.instrument), item.order_id),
        )
        for order in buys:
            candidate = self._prepare_buy(order, checked_bars)
            if isinstance(candidate, tuple):
                outcome, profile_id = candidate
                outcomes[order.order_id] = outcome
                if profile_id is not None:
                    used_profiles.add(profile_id)
                    profile = self._rule_book.profile_for(day, self._instruments[order.instrument])
                    if not profile.fee_profile_confirmed:
                        warnings.add(f"FEE_PROFILE_UNCONFIRMED:{profile_id}")
            else:
                candidates.append(candidate)
                used_profiles.add(candidate.profile.profile_id)
                if not candidate.profile.fee_profile_confirmed:
                    warnings.add(f"FEE_PROFILE_UNCONFIRMED:{candidate.profile.profile_id}")
        candidates = list(self._share_buy_volume_budget(candidates))
        allocations = self._allocate_buys(candidates)
        for candidate in sorted(
            candidates, key=lambda item: (str(item.order.instrument), item.order.order_id)
        ):
            outcome, fill = self._execute_buy(candidate, allocations[candidate.order.order_id])
            outcomes[candidate.order.order_id] = outcome
            if fill is not None:
                fills.append(fill)

        for symbol, bar in checked_bars.items():
            if symbol in self._holdings:
                self._holdings[symbol].mark_price = bar.close
        final_orders = tuple(
            outcomes[order.order_id]
            for order in sorted(checked_orders, key=lambda item: item.order_id)
        )
        return SessionExecution(
            final_orders,
            tuple(fills),
            self._snapshot(day),
            tuple(sorted(used_profiles)),
            tuple(sorted(warnings)),
        )
