from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from compass.services.automatic_market_sync import AutomaticMarketSync
from compass.ui.pages.settings import ProviderSetting, SettingsSnapshot


NOW = datetime(2026, 8, 12, 9, tzinfo=ZoneInfo("Asia/Shanghai"))


class MutableSettings:
    def __init__(self, snapshot: SettingsSnapshot) -> None:
        self.snapshot = snapshot

    def state(self) -> SettingsSnapshot:
        return self.snapshot


class RecordingData:
    def __init__(self) -> None:
        self.providers: list[str] = []
        self.active = False
        self.failures = 0
        self.snapshot = DataState((), ())

    def start_sync(self, provider: str) -> object:
        if self.active:
            raise ValueError("DATA_SYNC_ALREADY_ACTIVE")
        if self.failures:
            self.failures -= 1
            raise LookupError("TEMPORARY_PROVIDER_FAILURE")
        self.providers.append(provider)
        return object()

    def state(self) -> object:
        return self.snapshot


@dataclass(frozen=True)
class Preview:
    instrument: str
    last_day: str


@dataclass(frozen=True)
class DataState:
    watchlist_instruments: tuple[str, ...]
    previews: tuple[Preview, ...]


class ManualScheduler:
    def __init__(self) -> None:
        self.callback: Callable[[datetime], None] | None = None
        self.started = 0
        self.stopped = 0

    def register_five_minute(self, callback: Callable[[datetime], None]) -> int:
        self.callback = callback
        return 1

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def fire(self, boundary: datetime) -> None:
        assert self.callback is not None
        self.callback(boundary)


def _settings(
    *, startup: bool, interval: int | None, after_close: bool = False
) -> SettingsSnapshot:
    return SettingsSnapshot(
        providers=(ProviderSetting("tencent", "腾讯证券", True, 0, None),),
        log_level="INFO",
        automatic_sync_on_startup=startup,
        automatic_sync_interval_minutes=interval,
        automatic_sync_after_close=after_close,
    )


def test_startup_sync_submits_once_and_lifecycle_is_idempotent() -> None:
    settings = MutableSettings(_settings(startup=True, interval=None))
    data = RecordingData()
    scheduler = ManualScheduler()
    service = AutomaticMarketSync(
        settings=settings,
        data=data,
        scheduler=scheduler,
        clock=lambda: NOW,
    )

    service.start()
    service.start()
    service.stop()
    service.stop()

    assert data.providers == ["tencent"]
    assert scheduler.started == 1
    assert scheduler.stopped == 1


def test_interval_sync_anchors_configuration_and_skips_an_active_task() -> None:
    settings = MutableSettings(_settings(startup=False, interval=None))
    data = RecordingData()
    scheduler = ManualScheduler()
    service = AutomaticMarketSync(
        settings=settings,
        data=data,
        scheduler=scheduler,
        clock=lambda: NOW,
    )
    service.start()

    settings.snapshot = _settings(startup=False, interval=30)
    scheduler.fire(NOW + timedelta(minutes=5))
    scheduler.fire(NOW + timedelta(minutes=35))
    assert data.providers == ["tencent"]

    data.active = True
    scheduler.fire(NOW + timedelta(minutes=65))
    assert data.providers == ["tencent"]

    data.active = False
    scheduler.fire(NOW + timedelta(minutes=95))
    assert data.providers == ["tencent", "tencent"]


def test_close_sync_only_submits_when_latest_completed_session_is_missing() -> None:
    settings = MutableSettings(_settings(startup=False, interval=None, after_close=True))
    data = RecordingData()
    data.snapshot = DataState(("SSE.510300",), (Preview("SSE.510300", "2026-08-11"),))
    scheduler = ManualScheduler()
    service = AutomaticMarketSync(
        settings=settings,
        data=data,
        scheduler=scheduler,
        clock=lambda: NOW,
        latest_completed_session=lambda _: date(2026, 8, 11),
    )

    service.start()
    scheduler.fire(NOW + timedelta(minutes=5))
    assert data.providers == []

    data.snapshot = DataState(("SSE.510300",), (Preview("SSE.510300", "2026-08-10"),))
    second_scheduler = ManualScheduler()
    second = AutomaticMarketSync(
        settings=settings,
        data=data,
        scheduler=second_scheduler,
        clock=lambda: NOW,
        latest_completed_session=lambda _: date(2026, 8, 11),
    )
    second.start()
    assert data.providers == ["tencent"]
    data.snapshot = DataState(
        ("SSE.510300",),
        (Preview("SSE.510300", "2026-08-11"),),
    )
    second_scheduler.fire(NOW + timedelta(minutes=5))
    assert data.providers == ["tencent"]


def test_close_sync_retries_submission_failures_and_skips_an_empty_watchlist() -> None:
    settings = MutableSettings(_settings(startup=False, interval=None, after_close=True))
    data = RecordingData()
    data.snapshot = DataState(("SSE.510300",), ())
    data.failures = 1
    scheduler = ManualScheduler()
    service = AutomaticMarketSync(
        settings=settings,
        data=data,
        scheduler=scheduler,
        clock=lambda: NOW,
        latest_completed_session=lambda _: date(2026, 8, 11),
    )

    service.start()
    assert data.providers == []
    scheduler.fire(NOW + timedelta(minutes=5))
    assert data.providers == ["tencent"]

    empty_data = RecordingData()
    empty_scheduler = ManualScheduler()
    empty_service = AutomaticMarketSync(
        settings=settings,
        data=empty_data,
        scheduler=empty_scheduler,
        clock=lambda: NOW,
        latest_completed_session=lambda _: date(2026, 8, 11),
    )
    empty_service.start()
    empty_scheduler.fire(NOW + timedelta(minutes=5))
    assert empty_data.providers == []
