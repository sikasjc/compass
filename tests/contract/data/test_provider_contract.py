from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from compass.data.base import DailyBarRequest, ProviderCapabilityError
from compass.data.normalize import normalize_daily
from compass.data.providers.csv_provider import CsvProvider
from compass.domain.market import InstrumentId


def request() -> DailyBarRequest:
    return DailyBarRequest(
        InstrumentId.parse("SSE.510300"), date(2026, 7, 20), date(2026, 7, 21)
    )


def test_daily_bar_request_rejects_an_inverted_date_range() -> None:
    with pytest.raises(ValueError, match="start.*end"):
        DailyBarRequest(
            InstrumentId.parse("SSE.510300"), date(2026, 7, 22), date(2026, 7, 21)
        )


def test_csv_provider_returns_canonical_daily_bars() -> None:
    provider = CsvProvider(Path("tests/fixtures"))

    result = provider.fetch_daily(request())

    assert set(("open", "high", "low", "close", "volume", "amount")).issubset(result.columns)
    assert result.index.name == "date"
    assert result.index.is_monotonic_increasing
    assert result.index.tolist() == [pd.Timestamp("2026-07-20"), pd.Timestamp("2026-07-21")]


def test_csv_provider_preserves_canonical_optional_daily_fields() -> None:
    result = CsvProvider(Path("tests/fixtures")).fetch_daily(request())

    assert result["adjust_factor"].tolist() == [1.0, 1.01]
    assert result["suspended"].tolist() == [False, False]


@pytest.mark.parametrize(
    ("method", "argument", "capability"),
    [
        ("fetch_snapshot", [], "snapshot"),
        ("fetch_corporate_actions", request(), "corporate_actions"),
    ],
)
def test_csv_provider_explicitly_rejects_unsupported_capabilities(
    method: str, argument: object, capability: str
) -> None:
    provider = CsvProvider(Path("tests/fixtures"))

    with pytest.raises(ProviderCapabilityError) as error:
        getattr(provider, method)(argument)

    assert error.value.provider == "csv"
    assert error.value.capability == capability


def test_normalize_daily_maps_source_columns_and_normalizes_daily_dates() -> None:
    source = pd.DataFrame(
        {
            "trade_date": ["2026-07-21 15:00", "2026-07-20 09:30"],
            "opening": [4.1, 4.0],
            "highest": [4.3, 4.2],
            "lowest": [4.0, 3.9],
            "closing": [4.2, 4.1],
            "shares": [1200.0, 1000.0],
            "turnover": [5040.0, 4100.0],
            "factor": [1.01, 1.0],
        }
    )

    result = normalize_daily(
        source,
        {
            "trade_date": "date",
            "opening": "open",
            "highest": "high",
            "lowest": "low",
            "closing": "close",
            "shares": "volume",
            "turnover": "amount",
            "factor": "adjust_factor",
        },
    )

    assert result.index.tolist() == [pd.Timestamp("2026-07-20"), pd.Timestamp("2026-07-21")]
    assert result.index.name == "date"
    assert result["adjust_factor"].tolist() == [1.0, 1.01]


def test_normalize_daily_rejects_duplicate_mapped_canonical_columns() -> None:
    source = pd.DataFrame(columns=["first_date", "second_date"])

    with pytest.raises(ValueError, match="date"):
        normalize_daily(source, {"first_date": "date", "second_date": "date"})


def test_normalize_daily_rejects_a_mapped_column_that_collides_with_an_existing_column() -> None:
    source = pd.DataFrame(columns=["source_date", "date"])

    with pytest.raises(ValueError, match="date"):
        normalize_daily(source, {"source_date": "date"})


def test_csv_provider_rejects_duplicate_trading_dates(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "date": ["2026-07-20", "2026-07-20"],
            "open": [4.0, 4.0],
            "high": [4.2, 4.2],
            "low": [3.9, 3.9],
            "close": [4.1, 4.1],
            "volume": [1000.0, 1000.0],
            "amount": [4100.0, 4100.0],
        }
    ).to_csv(tmp_path / "daily_510300.csv", index=False)

    with pytest.raises(ValueError, match="unique"):
        CsvProvider(tmp_path).fetch_daily(request())


def test_csv_provider_rejects_unparseable_daily_dates(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "date": ["not-a-date"],
            "open": [4.0],
            "high": [4.2],
            "low": [3.9],
            "close": [4.1],
            "volume": [1000.0],
            "amount": [4100.0],
        }
    ).to_csv(tmp_path / "daily_510300.csv", index=False)

    with pytest.raises(ValueError, match="daily date"):
        CsvProvider(tmp_path).fetch_daily(request())
