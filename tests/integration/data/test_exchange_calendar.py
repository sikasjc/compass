from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from compass.data.base import DailyBarRequest
from compass.data.providers.akshare_provider import AkshareProvider
from compass.data.providers.baostock_provider import BaostockProvider
from compass.data.exchange_calendar import (
    CalendarUnavailableError,
    PersistedExchangeCalendar,
)
from compass.domain.market import InstrumentId


class CalendarProvider:
    name = "fixture-calendar"
    calendar_version = "fixture-v1"

    def fetch_exchange_sessions(self, start: date, end: date) -> tuple[date, ...]:
        assert (start, end) == (date(2026, 1, 1), date(2026, 1, 5))
        return (date(2026, 1, 2), date(2026, 1, 5))


class ExpandingCalendarProvider:
    name = "fixture-calendar"
    calendar_version = "fixture-v1"

    def fetch_exchange_sessions(self, start: date, end: date) -> tuple[date, ...]:
        if (start, end) == (date(2026, 1, 1), date(2026, 1, 5)):
            return (date(2026, 1, 2), date(2026, 1, 5))
        assert (start, end) == (date(2026, 2, 1), date(2026, 2, 3))
        return (date(2026, 2, 2), date(2026, 2, 3))


def test_exchange_calendar_is_provider_sourced_versioned_and_persistent(tmp_path: Path) -> None:
    """Replacing the cache with a weekday calendar would re-admit the holiday."""

    path = tmp_path / "exchange-calendar.json"
    calendar = PersistedExchangeCalendar(path, providers=(CalendarProvider(),))
    calendar.refresh(date(2026, 1, 1), date(2026, 1, 5))

    assert calendar.is_session(date(2026, 1, 1)) is False
    assert calendar.is_session(date(2026, 1, 2)) is True
    assert calendar.identity.provider == "fixture-calendar"
    assert calendar.identity.version == "fixture-v1"

    rebuilt = PersistedExchangeCalendar(path, providers=())
    request = DailyBarRequest(
        InstrumentId.parse("SSE.510300"),
        date(2026, 1, 1),
        date(2026, 1, 5),
    )
    assert tuple(item.date() for item in rebuilt.expected_sessions(request)) == (
        date(2026, 1, 2),
        date(2026, 1, 5),
    )


def test_exchange_calendar_fails_closed_outside_persisted_coverage(tmp_path: Path) -> None:
    calendar = PersistedExchangeCalendar(
        tmp_path / "exchange-calendar.json",
        providers=(CalendarProvider(),),
    )
    calendar.refresh(date(2026, 1, 1), date(2026, 1, 5))

    with pytest.raises(CalendarUnavailableError, match="CALENDAR_RANGE_UNAVAILABLE"):
        calendar.is_session(date(2026, 1, 6))


def test_exchange_calendar_preserves_content_addressed_historical_sessions(
    tmp_path: Path,
) -> None:
    calendar = PersistedExchangeCalendar(
        tmp_path / "exchange-calendar.json",
        providers=(ExpandingCalendarProvider(),),
    )
    first = calendar.refresh(date(2026, 1, 1), date(2026, 1, 5))
    calendar.refresh(date(2026, 2, 1), date(2026, 2, 3))

    assert calendar.sessions_for(first) == (
        date(2026, 1, 2),
        date(2026, 1, 5),
    )


def test_exchange_calendar_reuses_cache_when_requested_range_is_covered(
    tmp_path: Path,
) -> None:
    class Provider(CalendarProvider):
        calls = 0

        def fetch_exchange_sessions(self, start: date, end: date) -> tuple[date, ...]:
            self.calls += 1
            return super().fetch_exchange_sessions(start, end)

    provider = Provider()
    calendar = PersistedExchangeCalendar(
        tmp_path / "exchange-calendar.json",
        providers=(provider,),
    )
    identity = calendar.refresh(date(2026, 1, 1), date(2026, 1, 5))

    ensured = calendar.ensure_coverage(date(2026, 1, 2), date(2026, 1, 5))

    assert ensured == identity
    assert provider.calls == 1


def test_exchange_calendar_rejects_unbounded_ranges_before_calling_provider(
    tmp_path: Path,
) -> None:
    class Provider:
        name = "fixture-calendar"
        calendar_version = "fixture-v1"
        called = False

        def fetch_exchange_sessions(self, start: date, end: date) -> tuple[date, ...]:
            self.called = True
            return (start,)

    provider = Provider()
    calendar = PersistedExchangeCalendar(
        tmp_path / "exchange-calendar.json", providers=(provider,)
    )

    with pytest.raises(CalendarUnavailableError, match="CALENDAR_RANGE_TOO_LARGE"):
        calendar.refresh(date(1900, 1, 1), date(2100, 1, 1))

    assert provider.called is False


def test_exchange_calendar_rejects_oversized_cache_before_json_decode(
    tmp_path: Path,
) -> None:
    path = tmp_path / "exchange-calendar.json"
    path.write_bytes(b"x" * 2_000_000)
    calendar = PersistedExchangeCalendar(path, providers=())

    with pytest.raises(CalendarUnavailableError, match="CALENDAR_CACHE_TOO_LARGE"):
        _ = calendar.identity


def test_akshare_calendar_adapter_filters_the_authoritative_session_shape() -> None:
    class Client:
        def tool_trade_date_hist_sina(self):  # type: ignore[no-untyped-def]
            return __import__("pandas").DataFrame(
                {"trade_date": ["2025-12-31", "2026-01-02", "2026-01-05", "2026-01-06"]}
            )

    provider = AkshareProvider(client=Client())

    assert provider.fetch_exchange_sessions(date(2026, 1, 1), date(2026, 1, 5)) == (
        date(2026, 1, 2),
        date(2026, 1, 5),
    )


def test_baostock_calendar_adapter_keeps_only_explicit_trading_days() -> None:
    class Result:
        error_code = "0"
        fields = ["calendar_date", "is_trading_day"]

        def __init__(self) -> None:
            self._rows = iter(
                (["2026-01-01", "0"], ["2026-01-02", "1"], ["2026-01-05", "1"])
            )
            self._current: list[str] | None = None

        def next(self) -> bool:
            self._current = next(self._rows, None)
            return self._current is not None

        def get_row_data(self) -> list[str]:
            assert self._current is not None
            return self._current

    class Client:
        def query_trade_dates(self, **kwargs):  # type: ignore[no-untyped-def]
            assert kwargs == {"start_date": "2026-01-01", "end_date": "2026-01-05"}
            return Result()

    provider = BaostockProvider(client=Client())

    assert provider.fetch_exchange_sessions(date(2026, 1, 1), date(2026, 1, 5)) == (
        date(2026, 1, 2),
        date(2026, 1, 5),
    )
