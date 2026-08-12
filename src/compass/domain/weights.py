from __future__ import annotations

from collections.abc import Callable, Mapping
from decimal import Decimal
from typing import TypeVar


WEIGHT_SCALE = 1_000_000_000_000
WEIGHT_QUANTUM = Decimal("0.000000000001")

K = TypeVar("K")


def weight_to_units(value: object, *, label: str) -> int:
    """Convert an exactly representable unit weight without Decimal arithmetic."""

    if type(value) is not Decimal:
        raise TypeError(f"{label} must be an exact Decimal")
    assert isinstance(value, Decimal)
    decimal_value = value
    if not decimal_value.is_finite() or not (
        Decimal("0") <= decimal_value <= Decimal("1")
    ):
        raise ValueError(f"{label} must be finite and between zero and one")
    if decimal_value.is_zero():
        return 0

    sign, digits, exponent = decimal_value.as_tuple()
    if not isinstance(exponent, int):  # finite values always have an integer exponent
        raise ValueError(f"{label} must be finite")
    if sign:
        raise ValueError(f"{label} must be finite and between zero and one")
    coefficient = int("".join(str(digit) for digit in digits))
    shift = exponent + 12
    if shift >= 0:
        units: int = coefficient * (10**shift)
    else:
        discarded_places = -shift
        if discarded_places > len(digits):
            raise ValueError(f"{label} must be exactly representable to 12 decimal places")
        divisor = 10**discarded_places
        units, remainder = divmod(coefficient, divisor)
        if remainder:
            raise ValueError(f"{label} must be exactly representable to 12 decimal places")
    if units > WEIGHT_SCALE:
        raise ValueError(f"{label} must be finite and between zero and one")
    return units


def units_to_weight(units: int) -> Decimal:
    if isinstance(units, bool) or not isinstance(units, int):
        raise TypeError("weight units must be an exact integer")
    if not 0 <= units <= WEIGHT_SCALE:
        raise ValueError("weight units must be between zero and the weight scale")
    digits = tuple(int(character) for character in str(units))
    return Decimal((0, digits, -12))


def round_ratio_half_even(numerator: int, denominator: int) -> int:
    """Round a non-negative rational to an integer using exact half-even rules."""

    if numerator < 0 or denominator <= 0:
        raise ValueError("rounding ratio must be non-negative with a positive denominator")
    quotient, remainder = divmod(numerator, denominator)
    doubled = remainder * 2
    if doubled > denominator or (doubled == denominator and quotient % 2 == 1):
        quotient += 1
    return quotient


def largest_remainder_units(
    values: Mapping[K, int],
    target_units: int,
    *,
    canonical_key: Callable[[K], str],
) -> tuple[dict[K, int], tuple[str, ...]]:
    """Scale integer weights exactly with Hamilton/largest-remainder allocation."""

    if target_units < 0:
        raise ValueError("target units must be non-negative")
    ordered = sorted(values, key=canonical_key)
    if any(isinstance(values[key], bool) or not isinstance(values[key], int) for key in ordered):
        raise TypeError("allocation values must be exact integers")
    if any(values[key] < 0 for key in ordered):
        raise ValueError("allocation values must be non-negative")
    total = sum(values[key] for key in ordered)
    if total == 0:
        if target_units:
            raise ValueError("positive target units require positive source units")
        return {key: 0 for key in ordered}, ()
    if target_units == total:
        return {key: values[key] for key in ordered}, ()

    result: dict[K, int] = {}
    remainders: dict[K, int] = {}
    allocated = 0
    for key in ordered:
        quotient, remainder = divmod(values[key] * target_units, total)
        result[key] = quotient
        remainders[key] = remainder
        allocated += quotient
    remaining = target_units - allocated
    ranked = sorted(ordered, key=lambda key: (-remainders[key], canonical_key(key)))
    recipients = tuple(canonical_key(key) for key in ranked[:remaining])
    for key in ranked[:remaining]:
        result[key] += 1
    return result, recipients
