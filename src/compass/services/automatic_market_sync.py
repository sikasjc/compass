from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta
import logging
from threading import RLock
from typing import Protocol

from compass.services.scheduler import SHANGHAI
from compass.ui.pages.settings import SettingsSnapshot


Clock = Callable[[], datetime]
LOGGER = logging.getLogger("compass.automatic_sync")


class AutomaticSyncSettings(Protocol):
    def state(self) -> SettingsSnapshot: ...


class AutomaticSyncData(Protocol):
    def start_sync(self, provider: str) -> object: ...
    def state(self) -> object: ...


class AutomaticSyncScheduler(Protocol):
    def register_five_minute(self, callback: Callable[[datetime], None]) -> int: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...


class AutomaticMarketSync:
    """Submit incremental market syncs from persisted startup/interval settings."""

    def __init__(
        self,
        *,
        settings: AutomaticSyncSettings,
        data: AutomaticSyncData,
        scheduler: AutomaticSyncScheduler,
        clock: Clock,
        latest_completed_session: Callable[[datetime], date] | None = None,
    ) -> None:
        if not callable(clock):
            raise TypeError("automatic sync clock must be callable")
        self._settings = settings
        self._data = data
        self._scheduler = scheduler
        self._clock = clock
        self._latest_completed_session = latest_completed_session
        scheduler.register_five_minute(self._on_boundary)
        self._started = False
        self._last_interval: int | None = None
        self._last_triggered_at: datetime | None = None
        self._last_close_session: date | None = None
        self._lock = RLock()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            now = self._now()
            state = self._settings.state()
            self._last_interval = state.automatic_sync_interval_minutes
            self._last_triggered_at = now if self._last_interval is not None else None
            if state.automatic_sync_on_startup:
                self._attempt_submit(state)
                self._last_triggered_at = now
            if state.automatic_sync_after_close:
                self._attempt_close_sync(state, now)
            self._scheduler.start()

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            self._started = False
        self._scheduler.stop()

    def _on_boundary(self, boundary: datetime) -> None:
        with self._lock:
            if not self._started:
                return
            state = self._settings.state()
            if state.automatic_sync_after_close:
                self._attempt_close_sync(state, boundary)
            interval = state.automatic_sync_interval_minutes
            if interval is None:
                self._last_interval = None
                self._last_triggered_at = None
                return
            if interval != self._last_interval or self._last_triggered_at is None:
                self._last_interval = interval
                self._last_triggered_at = boundary
                return
            if boundary - self._last_triggered_at < timedelta(minutes=interval):
                return
            self._last_triggered_at = boundary
            self._attempt_submit(state)

    def _attempt_close_sync(self, state: SettingsSnapshot, moment: datetime) -> None:
        if self._latest_completed_session is None:
            return
        try:
            # Give end-of-day providers a short publication window after the close.
            completed = self._latest_completed_session(moment - timedelta(minutes=15))
            if completed == self._last_close_session:
                return
            data_state = self._data.state()
            previews = tuple(getattr(data_state, "previews"))
            watchlist = tuple(getattr(data_state, "watchlist_instruments"))
            if not watchlist:
                self._last_close_session = completed
                return
            covered = {
                item.instrument
                for item in previews
                if date.fromisoformat(item.last_day) >= completed
            }
            if set(watchlist).issubset(covered):
                self._last_close_session = completed
                return
            # Coverage, rather than task acceptance, is the completion condition.
            # A later boundary retries failed tasks and marks the day handled only
            # after every watched instrument covers the completed session.
            self._attempt_submit(state)
        except Exception as error:
            LOGGER.warning(
                "automatic close market sync check failed exception_type=%s",
                type(error).__name__,
            )

    def _attempt_submit(self, state: SettingsSnapshot) -> None:
        try:
            self._submit(state)
        except Exception as error:
            LOGGER.warning(
                "automatic market sync submission failed exception_type=%s",
                type(error).__name__,
            )

    def _submit(self, state: SettingsSnapshot) -> None:
        provider = next((item.provider for item in state.providers if item.available), None)
        if provider is None:
            raise LookupError("AUTOMATIC_SYNC_PROVIDER_UNAVAILABLE")
        try:
            self._data.start_sync(provider)
            LOGGER.info("automatic market sync submitted provider=%s", provider)
        except ValueError as error:
            if str(error) != "DATA_SYNC_ALREADY_ACTIVE":
                raise
            LOGGER.info("automatic market sync skipped because a sync is already active")

    def _now(self) -> datetime:
        value = self._clock()
        if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("automatic sync clock must return an aware datetime")
        return value.astimezone(SHANGHAI)
