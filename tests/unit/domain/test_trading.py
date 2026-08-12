from datetime import date
from decimal import Decimal

import pytest

from compass.domain.market import InstrumentId
from compass.domain.trading import AccountSnapshot, CorporateAction, Position, TargetIntent


def instrument() -> InstrumentId:
    return InstrumentId.parse("SSE.510300")


def test_account_equity_uses_mark_price() -> None:
    snapshot = AccountSnapshot(
        as_of=date(2026, 7, 20),
        cash=Decimal("10000"),
        positions=(
            Position(
                instrument(),
                1000,
                1000,
                Decimal("4.00"),
                Decimal("4.20"),
            ),
        ),
    )
    assert snapshot.equity == Decimal("14200.00")


@pytest.mark.parametrize(
    ("quantity", "available_quantity", "average_cost", "mark_price"),
    [
        (-1, 0, Decimal("4.00"), Decimal("4.20")),
        (1000, -1, Decimal("4.00"), Decimal("4.20")),
        (1000, 1001, Decimal("4.00"), Decimal("4.20")),
        (1000, 1000, Decimal("-0.01"), Decimal("4.20")),
        (1000, 1000, Decimal("4.00"), Decimal("-0.01")),
    ],
)
def test_position_rejects_invalid_accounting_values(
    quantity: int,
    available_quantity: int,
    average_cost: Decimal,
    mark_price: Decimal,
) -> None:
    with pytest.raises(ValueError):
        Position(instrument(), quantity, available_quantity, average_cost, mark_price)


def test_account_snapshot_rejects_negative_cash() -> None:
    with pytest.raises(ValueError, match="cash"):
        AccountSnapshot(as_of=date(2026, 7, 20), cash=Decimal("-0.01"), positions=())


@pytest.mark.parametrize("field", ("average_cost", "mark_price"))
@pytest.mark.parametrize("value", (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")))
def test_position_rejects_non_finite_monetary_values(field: str, value: Decimal) -> None:
    position = {
        "quantity": 1000,
        "available_quantity": 1000,
        "average_cost": Decimal("4.00"),
        "mark_price": Decimal("4.20"),
    }
    position[field] = value

    with pytest.raises(ValueError, match="finite"):
        Position(instrument(), **position)


@pytest.mark.parametrize("cash", (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")))
def test_account_snapshot_rejects_non_finite_cash(cash: Decimal) -> None:
    with pytest.raises(ValueError, match="cash.*finite"):
        AccountSnapshot(as_of=date(2026, 7, 20), cash=cash, positions=())


@pytest.mark.parametrize("field", ("split_ratio", "cash_dividend_per_share"))
@pytest.mark.parametrize("value", (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")))
def test_corporate_action_rejects_non_finite_monetary_values(field: str, value: Decimal) -> None:
    corporate_action = {
        "ex_date": date(2026, 7, 20),
        "split_ratio": Decimal("1"),
        "cash_dividend_per_share": Decimal("0"),
    }
    corporate_action[field] = value

    with pytest.raises(ValueError, match="finite"):
        CorporateAction(instrument(), **corporate_action)


def test_target_intent_weight_is_bounded() -> None:
    intent = TargetIntent(
        strategy_id="rotation-main",
        instrument=instrument(),
        target_weight=Decimal("0.25"),
        score=1.2,
        confidence=0.8,
        reason_code="MOMENTUM_TOP_N",
        valid_until=date(2026, 7, 21),
    )
    assert intent.target_weight == Decimal("0.25")
