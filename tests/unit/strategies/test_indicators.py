import math
import sys

import pandas as pd
import pytest

from compass.strategies.indicators import (
    annualized_volatility,
    bollinger_bands,
    rsi,
    simple_moving_average,
)


def test_moving_average_does_not_look_forward() -> None:
    prices = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2026-01-01", periods=3))

    result = simple_moving_average(prices, window=2)

    assert pd.isna(result.iloc[0])
    assert result.iloc[1:].tolist() == [1.5, 2.5]


def test_indicators_preserve_index_without_mutating_prices() -> None:
    prices = pd.Series([10.0, 11.0, 12.0, 13.0], index=pd.date_range("2026-01-01", periods=4))
    original = prices.copy(deep=True)

    result = simple_moving_average(prices, window=2)

    pd.testing.assert_series_equal(prices, original)
    assert result.index.equals(prices.index)


@pytest.mark.parametrize("window", [True, False, 0, -1, 1.5, "2"])
def test_indicators_reject_invalid_window(window: object) -> None:
    prices = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="window"):
        simple_moving_average(prices, window=window)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_indicators_reject_non_finite_input(value: float) -> None:
    prices = pd.Series([1.0, value, 3.0])

    with pytest.raises(ValueError, match="finite"):
        simple_moving_average(prices, window=2)


def test_rsi_uses_wilder_conventions_for_directional_and_flat_prices() -> None:
    rising = pd.Series(range(1, 17), dtype=float)
    falling = pd.Series(range(16, 0, -1), dtype=float)
    flat = pd.Series([10.0] * 16)

    assert rsi(rising, window=14).iloc[-1] == 100.0
    assert rsi(falling, window=14).iloc[-1] == 0.0
    assert rsi(flat, window=14).iloc[-1] == 50.0


def test_rsi_seeds_with_simple_average_then_uses_wilder_smoothing() -> None:
    prices = pd.Series([100.0, 110.0, 107.0, 111.0])

    result = rsi(prices, window=2)

    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(76.923077, rel=1e-6)
    assert result.iloc[3] == pytest.approx(85.714286, rel=1e-6)


def test_bollinger_bands_use_population_standard_deviation() -> None:
    prices = pd.Series([1.0, 2.0, 3.0])

    result = bollinger_bands(prices, window=3, num_std=2.0, ddof=0)

    assert result.middle.iloc[-1] == 2.0
    assert result.upper.iloc[-1] == pytest.approx(2.0 + 2.0 * math.sqrt(2.0 / 3.0))
    assert result.lower.iloc[-1] == pytest.approx(2.0 - 2.0 * math.sqrt(2.0 / 3.0))


def test_annualized_volatility_uses_returns_and_stable_warmup() -> None:
    prices = pd.Series([100.0, 110.0, 121.0, 133.1])

    result = annualized_volatility(prices, window=2, annualization=252, ddof=0)

    assert result.iloc[:2].isna().all()
    assert result.iloc[-1] == pytest.approx(0.0)


def test_indicators_reject_unsorted_duplicate_and_complex_price_inputs() -> None:
    with pytest.raises(ValueError, match="index"):
        simple_moving_average(
            pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-01-02", "2026-01-01"])), window=2
        )
    with pytest.raises(ValueError, match="index"):
        simple_moving_average(
            pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-01-01", "2026-01-01"])), window=2
        )
    with pytest.raises(ValueError, match="numeric"):
        simple_moving_average(pd.Series([1 + 1j, 2 + 1j]), window=2)


def test_bollinger_rejects_a_multiplier_that_overflows_band_width() -> None:
    with pytest.raises(ValueError, match="num_std"):
        bollinger_bands(pd.Series([0.0, 100.0, 0.0]), window=3, num_std=sys.float_info.max)


def test_indicators_normalize_extreme_parameters_and_non_finite_returns_to_value_errors() -> None:
    prices = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="window"):
        simple_moving_average(prices, window=10**1000)
    with pytest.raises(ValueError, match="annualization"):
        annualized_volatility(prices, window=2, annualization=10**1000)
    with pytest.raises(ValueError, match="num_std"):
        bollinger_bands(prices, window=2, num_std=10**1000)
    with pytest.raises(ValueError, match="returns"):
        annualized_volatility(pd.Series([0.0, 1.0, 2.0]), window=2)
