from __future__ import annotations

from decimal import Decimal, ROUND_FLOOR, getcontext, setcontext

import pytest

from compass.domain.weights import (
    WEIGHT_QUANTUM,
    WEIGHT_SCALE,
    units_to_weight,
    weight_to_units,
)


def test_weight_unit_contract_is_public_and_exact() -> None:
    assert WEIGHT_QUANTUM == Decimal("0.000000000001")
    assert WEIGHT_SCALE == 1_000_000_000_000
    assert weight_to_units(Decimal("0.24"), label="weight") == 240_000_000_000
    assert weight_to_units(Decimal("0.2400000000000"), label="weight") == 240_000_000_000
    assert units_to_weight(240_000_000_000) == Decimal("0.24")


@pytest.mark.parametrize(
    "value",
    [
        Decimal("0.0000000000001"),
        Decimal("1E-1000000"),
        Decimal("0.1234567890123"),
    ],
)
def test_weight_units_reject_non_representable_decimals(value: Decimal) -> None:
    with pytest.raises(ValueError, match="12 decimal places"):
        weight_to_units(value, label="test weight")


def test_weight_conversion_ignores_the_global_decimal_context() -> None:
    original = getcontext().copy()
    try:
        expected = units_to_weight(123_456_789_012)
        getcontext().prec = 2
        getcontext().rounding = ROUND_FLOOR
        actual = units_to_weight(123_456_789_012)
        units = weight_to_units(Decimal("0.123456789012"), label="weight")
    finally:
        setcontext(original)

    assert actual == expected == Decimal("0.123456789012")
    assert units == 123_456_789_012
