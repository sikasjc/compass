from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from compass.analytics import (
    SleeveAccounting,
    SleeveAccountingPeriod,
    calculate_sleeve_accounting,
)
from tests.support.sleeve_results import D2, D3, D4, D5, multi_reallocation_result


def test_sleeve_accounting_is_available_from_the_public_analytics_package() -> None:
    result = calculate_sleeve_accounting(multi_reallocation_result())

    assert type(result) is SleeveAccounting
    assert all(type(period) is SleeveAccountingPeriod for period in result.periods)


def test_sleeve_accounting_uses_each_periods_then_effective_allocation_and_exit_fill() -> None:
    accounting = calculate_sleeve_accounting(multi_reallocation_result())
    by_day = {period.trading_day: period for period in accounting.periods}

    assert dict(by_day[D3].beginning_weights) == pytest.approx(
        {"sleeve-a": 0.75, "sleeve-b": 0.25}
    )
    assert dict(by_day[D4].beginning_weights) == pytest.approx(
        {"sleeve-a": 0.225, "sleeve-b": 0.675}
    )
    assert dict(by_day[D5].beginning_weights) == pytest.approx(
        {
            "sleeve-a": float(Decimal("576") / Decimal("1190")),
            "sleeve-b": float(Decimal("384") / Decimal("1190")),
        }
    )
    assert dict(by_day[D5].contributions) == pytest.approx(
        {
            "sleeve-a": float(Decimal("48") / Decimal("1190")),
            "sleeve-b": float(Decimal("32") / Decimal("1190")),
        }
    )
    assert by_day[D5].residual == pytest.approx(0.0)
    assert accounting.combined_residual == pytest.approx(0.0)


def test_sleeve_accounting_output_is_deeply_immutable() -> None:
    accounting = calculate_sleeve_accounting(multi_reallocation_result())

    with pytest.raises(TypeError):
        accounting.periods[-1].contributions["sleeve-a"] = 99.0  # type: ignore[index]


def test_new_entry_open_to_close_profit_is_explicitly_unattributed_residual() -> None:
    result = multi_reallocation_result()
    entry_position = replace(
        result.ledger[1].positions[0],
        mark_price=Decimal("11.0000"),
    )
    entry_ledger = replace(result.ledger[1], positions=(entry_position,))
    accounting = calculate_sleeve_accounting(
        replace(
            result,
            ledger=(result.ledger[0], entry_ledger, *result.ledger[2:]),
        )
    )
    entry_period = next(item for item in accounting.periods if item.trading_day == D2)

    assert sum(entry_period.contributions.values()) == pytest.approx(0.0)
    assert entry_period.residual == pytest.approx(0.10)


def test_sleeve_accounting_normalizes_absolute_per_instrument_sleeve_weights() -> None:
    result = multi_reallocation_result()
    first = replace(
        result.orders[0],
        sleeve_weights={
            "sleeve-a": Decimal("0.30"),
            "sleeve-b": Decimal("0.10"),
        },
    )
    accounting = calculate_sleeve_accounting(
        replace(result, orders=(first, *result.orders[1:]))
    )
    period = next(item for item in accounting.periods if item.trading_day == D3)

    assert dict(period.beginning_weights) == pytest.approx(
        {"sleeve-a": 0.75, "sleeve-b": 0.25}
    )
    assert sum(period.beginning_weights.values()) == pytest.approx(1.0)


def test_exit_fees_remain_in_residual_instead_of_sleeve_price_contribution() -> None:
    result = multi_reallocation_result()
    exit_fill = replace(
        result.fills[-1],
        commission=Decimal("5.00"),
        total_fee=Decimal("5.00"),
    )
    exit_ledger = replace(
        result.ledger[-1],
        cash=Decimal("1265.00"),
        withdrawable_cash=Decimal("1265.00"),
    )
    accounting = calculate_sleeve_accounting(
        replace(
            result,
            fills=(*result.fills[:-1], exit_fill),
            ledger=(*result.ledger[:-1], exit_ledger),
        )
    )
    period = accounting.periods[-1]

    assert sum(period.contributions.values()) == pytest.approx(
        float(Decimal("80") / Decimal("1190"))
    )
    assert period.residual == pytest.approx(float(Decimal("-5") / Decimal("1190")))


@pytest.mark.parametrize(
    "warnings",
    [
        ("unsafe warning",),
        ("UNSAFE\nWARNING",),
        ("SAFE_CODE:unsafe/path",),
        ("SAFE_CODE:identifier:extra",),
    ],
)
def test_backtest_result_rejects_unstructured_or_unsafe_warnings(
    warnings: tuple[str, ...],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(multi_reallocation_result(), warnings=warnings)


def test_backtest_result_accepts_structured_safe_warnings() -> None:
    result = replace(
        multi_reallocation_result(),
        warnings=("CODE", "CODE:safe-id"),
    )

    assert result.warnings == ("CODE", "CODE:safe-id")


@pytest.mark.parametrize(
    "warning",
    (
        "A" * 129,
        f"CODE:{'a' * 129}",
        f"{'A' * 128}:{'a' * 128}",
    ),
)
def test_backtest_result_rejects_warning_component_or_display_overflow(
    warning: str,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(multi_reallocation_result(), warnings=(warning,))


def test_backtest_result_accepts_warning_length_boundaries() -> None:
    result = replace(
        multi_reallocation_result(),
        warnings=(
            "A" * 128,
            f"{'B' * 128}:{'b' * 127}",
            f"CODE:{'c' * 128}",
        ),
    )

    assert tuple(map(len, result.warnings)) == (128, 256, 133)


def test_backtest_result_integrity_revalidates_exact_nested_values() -> None:
    result = multi_reallocation_result()
    object.__setattr__(result.orders[0], "quantity", 0)

    with pytest.raises((TypeError, ValueError)):
        result.verify_integrity()
