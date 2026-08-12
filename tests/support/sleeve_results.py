from __future__ import annotations

from datetime import date
from decimal import Decimal

from compass.backtest.engine import BacktestResult
from compass.backtest.orders import (
    Fill,
    LedgerPosition,
    LedgerSnapshot,
    Order,
    OrderSide,
    OrderStatus,
)
from compass.domain.market import InstrumentId


ETF = InstrumentId.parse("SSE.510300")
D1 = date(2026, 7, 20)
D2 = date(2026, 7, 21)
D3 = date(2026, 7, 22)
D4 = date(2026, 7, 23)
D5 = date(2026, 7, 24)


def _position(quantity: int, mark: str) -> LedgerPosition:
    return LedgerPosition(
        instrument=ETF,
        quantity=quantity,
        available_quantity=quantity,
        unsettled_quantity=0,
        average_cost=Decimal("10.0000"),
        mark_price=Decimal(mark),
    )


def _order(
    order_id: str,
    side: OrderSide,
    quantity: int,
    created_on: date,
    scheduled_for: date,
    sleeves: dict[str, Decimal],
) -> Order:
    return Order(
        order_id=order_id,
        instrument=ETF,
        side=side,
        quantity=quantity,
        created_on=created_on,
        scheduled_for=scheduled_for,
        sleeve_weights=sleeves,
        status=OrderStatus.FILLED,
        filled_quantity=quantity,
    )


def _fill(order: Order, price: str) -> Fill:
    gross = (Decimal(price) * order.quantity).quantize(Decimal("0.01"))
    return Fill(
        fill_id=f"fill:{order.order_id}",
        order_id=order.order_id,
        instrument=order.instrument,
        side=order.side,
        quantity=order.quantity,
        trading_day=order.scheduled_for,  # type: ignore[arg-type]
        price=Decimal(price),
        gross_amount=gross,
        commission=Decimal("0.00"),
        stamp_duty=Decimal("0.00"),
        transfer_fee=Decimal("0.00"),
        total_fee=Decimal("0.00"),
        profile_id="fees",
    )


def multi_reallocation_result() -> BacktestResult:
    first = _order(
        "run:initial",
        OrderSide.BUY,
        100,
        D1,
        D2,
        {"sleeve-a": Decimal("0.75"), "sleeve-b": Decimal("0.25")},
    )
    second = _order(
        "run:reallocation-1",
        OrderSide.SELL,
        10,
        D2,
        D3,
        {"sleeve-a": Decimal("0.25"), "sleeve-b": Decimal("0.75")},
    )
    third = _order(
        "run:reallocation-2",
        OrderSide.SELL,
        10,
        D3,
        D4,
        {"sleeve-a": Decimal("0.60"), "sleeve-b": Decimal("0.40")},
    )
    exit_order = _order(
        "run:exit",
        OrderSide.SELL,
        80,
        D4,
        D5,
        {"sleeve-a": Decimal("0.60"), "sleeve-b": Decimal("0.40")},
    )
    orders = (first, second, third, exit_order)
    return BacktestResult(
        run_id="run",
        orders=orders,
        fills=tuple(
            _fill(order, price)
            for order, price in zip(orders, ("10.0000", "11.0000", "12.0000", "13.0000"))
        ),
        ledger=(
            LedgerSnapshot(D1, Decimal("1000.00"), Decimal("1000.00"), ()),
            LedgerSnapshot(D2, Decimal("0.00"), Decimal("0.00"), (_position(100, "10.0000"),)),
            LedgerSnapshot(D3, Decimal("110.00"), Decimal("110.00"), (_position(90, "11.0000"),)),
            LedgerSnapshot(D4, Decimal("230.00"), Decimal("230.00"), (_position(80, "12.0000"),)),
            LedgerSnapshot(D5, Decimal("1270.00"), Decimal("1270.00"), ()),
        ),
        risk_traces=(),
        used_profile_ids=("fees",),
        warnings=("SAFE_CODE", "SAFE_CODE:safe-id"),
    )
