from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


_MAX_NUMERIC_PARAMETER = np.iinfo(np.int64).max


@dataclass(frozen=True, slots=True)
class BollingerBands:
    """Three same-index rolling Bollinger-band series."""

    middle: pd.Series
    upper: pd.Series
    lower: pd.Series


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if value > _MAX_NUMERIC_PARAMETER:
        raise ValueError(f"{name} is too large")
    return value


def _valid_ddof(ddof: object, window: int) -> int:
    if isinstance(ddof, bool) or not isinstance(ddof, int) or not 0 <= ddof < window:
        raise ValueError("ddof must be an integer from zero through window - 1")
    return ddof


def _prices(prices: pd.Series) -> pd.Series:
    if not isinstance(prices, pd.Series):
        raise ValueError("prices must be a numeric pandas Series")
    if (
        not pd.api.types.is_numeric_dtype(prices.dtype)
        or pd.api.types.is_bool_dtype(prices.dtype)
        or pd.api.types.is_complex_dtype(prices.dtype)
    ):
        raise ValueError("prices must be a numeric pandas Series")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("prices index must be unique and monotonically increasing")
    values = prices.to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("prices must contain finite values")
    return pd.Series(values, index=prices.index.copy(), name=prices.name, dtype=float)


def simple_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """Trailing SMA with ``min_periods=window`` and no filling or sorting."""

    normalized = _prices(prices)
    validated_window = _positive_int(window, "window")
    return normalized.rolling(window=validated_window, min_periods=validated_window).mean()


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Wilder RSI seeded by a simple mean; flat windows are 50."""

    normalized = _prices(prices)
    validated_window = _positive_int(window, "window")
    values = normalized.to_numpy(dtype=float, copy=False)
    output = np.full(len(values), np.nan)
    if len(values) <= validated_window:
        return pd.Series(output, index=normalized.index, name=normalized.name, dtype=float)

    changes = np.diff(values)
    gains = np.maximum(changes, 0.0)
    losses = np.maximum(-changes, 0.0)
    average_gain = float(gains[:validated_window].mean())
    average_loss = float(losses[:validated_window].mean())
    output[validated_window] = _rsi_value(average_gain, average_loss)
    for index in range(validated_window + 1, len(values)):
        average_gain = (average_gain * (validated_window - 1) + gains[index - 1]) / validated_window
        average_loss = (average_loss * (validated_window - 1) + losses[index - 1]) / validated_window
        output[index] = _rsi_value(average_gain, average_loss)
    return pd.Series(output, index=normalized.index, name=normalized.name, dtype=float)


def _rsi_value(average_gain: float, average_loss: float) -> float:
    if average_gain == 0.0 and average_loss == 0.0:
        return 50.0
    if average_loss == 0.0:
        return 100.0
    if average_gain == 0.0:
        return 0.0
    return 100.0 - (100.0 / (1.0 + average_gain / average_loss))


def bollinger_bands(
    prices: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
    ddof: int = 0,
) -> BollingerBands:
    """Trailing bands using population standard deviation by default (``ddof=0``)."""

    normalized = _prices(prices)
    validated_window = _positive_int(window, "window")
    validated_ddof = _valid_ddof(ddof, validated_window)
    if isinstance(num_std, bool) or not isinstance(num_std, (int, float)):
        raise ValueError("num_std must be a finite positive number")
    try:
        multiplier = float(num_std)
    except OverflowError:
        raise ValueError("num_std must be a finite positive number") from None
    if not isfinite(multiplier) or multiplier <= 0:
        raise ValueError("num_std must be a finite positive number")
    rolling = normalized.rolling(window=validated_window, min_periods=validated_window)
    middle = rolling.mean()
    deviation = rolling.std(ddof=validated_ddof)
    try:
        with np.errstate(over="raise"):
            width = deviation * multiplier
            upper = middle + width
            lower = middle - width
    except FloatingPointError:
        raise ValueError("num_std produces non-finite Bollinger bands") from None
    for band in (width, upper, lower):
        values = band.to_numpy(dtype=float, copy=False)
        if not np.isfinite(values[~np.isnan(values)]).all():
            raise ValueError("num_std produces non-finite Bollinger bands")
    return BollingerBands(middle=middle, upper=upper, lower=lower)


def annualized_volatility(
    prices: pd.Series,
    window: int = 20,
    annualization: int = 252,
    ddof: int = 0,
) -> pd.Series:
    """Trailing annualized volatility of simple returns; no missing-data filling."""

    normalized = _prices(prices)
    validated_window = _positive_int(window, "window")
    validated_annualization = _positive_int(annualization, "annualization")
    validated_ddof = _valid_ddof(ddof, validated_window)
    returns = normalized.pct_change(fill_method=None)
    return_values = returns.iloc[1:].to_numpy(dtype=float, copy=False)
    if not np.isfinite(return_values).all():
        raise ValueError("returns must be finite")
    return returns.rolling(window=validated_window, min_periods=validated_window).std(
        ddof=validated_ddof
    ) * sqrt(validated_annualization)
