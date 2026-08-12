from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from compass.analytics.attribution import attribute_sleeves


def test_overlapping_sleeves_use_supplied_beginning_weights_without_shift() -> None:
    index = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    returns = pd.DataFrame(
        {"rotation": [0.10, -0.05, 0.02], "trend": [-0.02, 0.04, 0.01]},
        index=index,
    )
    weights = pd.DataFrame(
        {"rotation": [0.60, 0.25, 0.40], "trend": [0.30, 0.50, 0.50]},
        index=index,
    )
    returns_before = returns.copy(deep=True)
    weights_before = weights.copy(deep=True)

    contribution = attribute_sleeves(returns, weights)

    expected = pd.DataFrame(
        {
            "rotation": [0.060, -0.0125, 0.008],
            "trend": [-0.006, 0.020, 0.005],
            "total_contribution": [0.054, 0.0075, 0.013],
        },
        index=index,
    )
    pd.testing.assert_frame_equal(contribution, expected)
    pd.testing.assert_frame_equal(returns, returns_before)
    pd.testing.assert_frame_equal(weights, weights_before)


def test_cash_weight_contributes_zero_and_losing_sleeve_remains_negative() -> None:
    index = pd.to_datetime(["2026-01-02"])
    returns = pd.DataFrame({"winner": [0.10], "loser": [-0.20]}, index=index)
    weights = pd.DataFrame({"winner": [0.30], "loser": [0.20]}, index=index)

    result = attribute_sleeves(returns, weights)

    assert result.loc[index[0], "winner"] == pytest.approx(0.03)
    assert result.loc[index[0], "loser"] == pytest.approx(-0.04)
    assert result.loc[index[0], "total_contribution"] == pytest.approx(-0.01)


def test_empty_and_one_row_inputs_have_deterministic_shapes() -> None:
    empty_index = pd.DatetimeIndex([])
    empty_returns = pd.DataFrame(columns=["a", "b"], index=empty_index, dtype=float)
    empty_weights = pd.DataFrame(columns=["a", "b"], index=empty_index, dtype=float)

    empty = attribute_sleeves(empty_returns, empty_weights)
    one = attribute_sleeves(
        pd.DataFrame({"a": [0.10]}, index=pd.to_datetime(["2026-01-02"])),
        pd.DataFrame({"a": [1.0]}, index=pd.to_datetime(["2026-01-02"])),
    )

    assert list(empty.columns) == ["a", "b", "total_contribution"]
    assert empty.empty
    assert one.iloc[0].to_dict() == {"a": pytest.approx(0.10), "total_contribution": pytest.approx(0.10)}


@pytest.mark.parametrize(
    ("returns", "weights", "message"),
    [
        (
            pd.DataFrame({"a": [0.1]}, index=pd.to_datetime(["2026-01-02"])),
            pd.DataFrame({"b": [1.0]}, index=pd.to_datetime(["2026-01-02"])),
            "columns",
        ),
        (
            pd.DataFrame({"a": [0.1]}, index=pd.to_datetime(["2026-01-02"])),
            pd.DataFrame({"a": [1.0]}, index=pd.to_datetime(["2026-01-03"])),
            "indexes",
        ),
        (
            pd.DataFrame({"a": [0.1, 0.2]}, index=pd.to_datetime(["2026-01-03", "2026-01-02"])),
            pd.DataFrame({"a": [0.5, 0.5]}, index=pd.to_datetime(["2026-01-03", "2026-01-02"])),
            "increasing",
        ),
        (
            pd.DataFrame({"a": [0.1, 0.2]}, index=pd.to_datetime(["2026-01-02", "2026-01-02"])),
            pd.DataFrame({"a": [0.5, 0.5]}, index=pd.to_datetime(["2026-01-02", "2026-01-02"])),
            "unique",
        ),
        (
            pd.DataFrame({"a": [np.nan]}, index=pd.to_datetime(["2026-01-02"])),
            pd.DataFrame({"a": [1.0]}, index=pd.to_datetime(["2026-01-02"])),
            "finite",
        ),
        (
            pd.DataFrame({"a": [True]}, index=pd.to_datetime(["2026-01-02"])),
            pd.DataFrame({"a": [1.0]}, index=pd.to_datetime(["2026-01-02"])),
            "booleans",
        ),
        (
            pd.DataFrame({"a": [0.1]}, index=pd.to_datetime(["2026-01-02"])),
            pd.DataFrame({"a": [-0.1]}, index=pd.to_datetime(["2026-01-02"])),
            "non-negative",
        ),
        (
            pd.DataFrame({"a": [0.1], "b": [0.1]}, index=pd.to_datetime(["2026-01-02"])),
            pd.DataFrame({"a": [0.7], "b": [0.4]}, index=pd.to_datetime(["2026-01-02"])),
            "at most one",
        ),
    ],
)
def test_attribution_rejects_implicit_alignment_and_invalid_values(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        attribute_sleeves(returns, weights)


def test_weight_tolerance_is_explicit_and_small() -> None:
    index = pd.to_datetime(["2026-01-02"])
    returns = pd.DataFrame({"a": [0.1], "b": [0.2]}, index=index)

    accepted = attribute_sleeves(
        returns,
        pd.DataFrame({"a": [0.5], "b": [0.5000000000005]}, index=index),
    )

    assert accepted.loc[index[0], "total_contribution"] == pytest.approx(0.15)
    with pytest.raises(ValueError, match="at most one"):
        attribute_sleeves(
            returns,
            pd.DataFrame({"a": [0.5], "b": [0.500000000002]}, index=index),
        )


def test_total_contribution_is_reserved_and_cannot_be_a_sleeve() -> None:
    index = pd.to_datetime(["2026-01-02"])
    returns = pd.DataFrame(
        {"total_contribution": [0.50], "rotation": [0.10]},
        index=index,
    )
    weights = pd.DataFrame(
        {"total_contribution": [0.20], "rotation": [0.30]},
        index=index,
    )

    with pytest.raises(ValueError, match="total_contribution.*reserved"):
        attribute_sleeves(returns, weights)


def test_empty_attribution_still_requires_a_canonical_date_index() -> None:
    returns = pd.DataFrame(columns=["rotation"], dtype=float)
    weights = pd.DataFrame(columns=["rotation"], dtype=float)

    with pytest.raises(TypeError, match="indexes must contain dates"):
        attribute_sleeves(returns, weights)
