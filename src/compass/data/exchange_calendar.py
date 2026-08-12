from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Protocol
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd  # type: ignore[import-untyped]

from compass.data.base import DailyBarRequest, ProviderCapabilityError


SHANGHAI = ZoneInfo("Asia/Shanghai")
_STABLE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_SCHEMA_VERSION = 1
_MAX_CACHE_BYTES = 1_048_576
_MAX_RANGE_DAYS = 36_600
_MAX_SESSIONS = 30_000


class CalendarUnavailableError(RuntimeError):
    """A trusted exchange-session range is absent or failed integrity checks."""


class ExchangeCalendarProvider(Protocol):
    name: str
    calendar_version: str

    def fetch_exchange_sessions(self, start: date, end: date) -> Sequence[date]: ...


@dataclass(frozen=True, slots=True)
class CalendarIdentity:
    calendar_id: str
    provider: str
    version: str
    covered_from: date
    covered_to: date


@dataclass(frozen=True, slots=True)
class _CalendarSnapshot:
    identity: CalendarIdentity
    sessions: tuple[date, ...]


def _exact_day(value: object, *, label: str) -> date:
    if type(value) is not date:
        raise TypeError(f"{label} must be an exact date")
    assert isinstance(value, date)
    return value


def _stable(value: object, *, label: str) -> str:
    if type(value) is not str or _STABLE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a stable identifier")
    assert isinstance(value, str)
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class PersistedExchangeCalendar:
    """Content-verified SSE/SZSE session cache with explicit provider identity."""

    def __init__(
        self,
        path: Path,
        *,
        providers: Sequence[ExchangeCalendarProvider],
    ) -> None:
        if not isinstance(path, Path):
            raise TypeError("calendar path must be a Path")
        self._path = path
        self._objects_dir = path.parent / f"{path.stem}.objects"
        self._providers = tuple(providers)

    @property
    def identity(self) -> CalendarIdentity:
        return self._load().identity

    def refresh(self, start: date, end: date) -> CalendarIdentity:
        checked_start = _exact_day(start, label="calendar start")
        checked_end = _exact_day(end, label="calendar end")
        if checked_start > checked_end:
            raise ValueError("calendar start must not follow end")
        if (checked_end - checked_start).days > _MAX_RANGE_DAYS:
            raise CalendarUnavailableError("CALENDAR_RANGE_TOO_LARGE")
        for provider in self._providers:
            try:
                name = _stable(provider.name, label="calendar provider")
                version = _stable(
                    provider.calendar_version,
                    label="calendar provider version",
                )
                sessions = self._bounded_sessions(
                    provider.fetch_exchange_sessions(checked_start, checked_end)
                )
                if (
                    not sessions
                    or tuple(sorted(set(sessions))) != sessions
                    or sessions[0] < checked_start
                    or sessions[-1] > checked_end
                ):
                    raise ValueError("calendar provider returned invalid sessions")
                payload = {
                    "covered_from": checked_start.isoformat(),
                    "covered_to": checked_end.isoformat(),
                    "exchange": "SSE_SZSE",
                    "provider": name,
                    "provider_version": version,
                    "schema_version": _SCHEMA_VERSION,
                    "sessions": [item.isoformat() for item in sessions],
                }
                calendar_id = sha256(_canonical(payload).encode("utf-8")).hexdigest()
                document = {**payload, "calendar_id": calendar_id}
                text = _canonical(document)
                self._write(self._objects_dir / f"{calendar_id}.json", text)
                self._write(self._path, text)
                return CalendarIdentity(
                    calendar_id,
                    name,
                    version,
                    checked_start,
                    checked_end,
                )
            except ProviderCapabilityError:
                continue
            except Exception:
                continue
        raise CalendarUnavailableError("CALENDAR_REFRESH_FAILED")

    def ensure_coverage(self, start: date, end: date) -> CalendarIdentity:
        """Reuse a verified cache and refresh only when the requested range is absent."""

        checked_start = _exact_day(start, label="calendar start")
        checked_end = _exact_day(end, label="calendar end")
        if checked_start > checked_end:
            raise ValueError("calendar start must not follow end")
        try:
            snapshot = self._load()
        except CalendarUnavailableError:
            return self.refresh(checked_start, checked_end)
        if (
            snapshot.identity.covered_from <= checked_start
            and snapshot.identity.covered_to >= checked_end
        ):
            return snapshot.identity
        return self.refresh(
            min(checked_start, snapshot.identity.covered_from),
            max(checked_end, snapshot.identity.covered_to),
        )

    def expected_sessions(self, request: DailyBarRequest) -> pd.DatetimeIndex:
        if type(request) is not DailyBarRequest:
            raise TypeError("calendar request must be an exact DailyBarRequest")
        try:
            snapshot = self._load()
            if request.start < snapshot.identity.covered_from or request.end > snapshot.identity.covered_to:
                raise CalendarUnavailableError("CALENDAR_RANGE_UNAVAILABLE")
        except CalendarUnavailableError:
            self.refresh(request.start, request.end)
            snapshot = self._load()
        sessions = tuple(
            item for item in snapshot.sessions if request.start <= item <= request.end
        )
        return pd.DatetimeIndex(sessions, name="date")

    def sessions_between(self, start: date, end: date) -> tuple[date, ...]:
        checked_start = _exact_day(start, label="calendar start")
        checked_end = _exact_day(end, label="calendar end")
        snapshot = self._load()
        if checked_start < snapshot.identity.covered_from or checked_end > snapshot.identity.covered_to:
            raise CalendarUnavailableError("CALENDAR_RANGE_UNAVAILABLE")
        return tuple(item for item in snapshot.sessions if checked_start <= item <= checked_end)

    def sessions_for(self, identity: CalendarIdentity) -> tuple[date, ...]:
        if type(identity) is not CalendarIdentity:
            raise TypeError("calendar identity must be an exact CalendarIdentity")
        if _HASH.fullmatch(identity.calendar_id) is None:
            raise CalendarUnavailableError("CALENDAR_IDENTITY_INVALID")
        snapshot = self._load(self._objects_dir / f"{identity.calendar_id}.json")
        if snapshot.identity != identity:
            raise CalendarUnavailableError("CALENDAR_IDENTITY_MISMATCH")
        return snapshot.sessions

    def is_session(self, day: date) -> bool:
        checked = _exact_day(day, label="calendar day")
        snapshot = self._load()
        if checked < snapshot.identity.covered_from or checked > snapshot.identity.covered_to:
            raise CalendarUnavailableError("CALENDAR_RANGE_UNAVAILABLE")
        return checked in set(snapshot.sessions)

    def latest_completed_session(self, moment: datetime) -> date:
        if type(moment) is not datetime or moment.tzinfo is None or moment.utcoffset() is None:
            raise TypeError("calendar completion moment must be timezone-aware")
        local = moment.astimezone(SHANGHAI)
        snapshot = self._load()
        cutoff = local.date()
        if local.timetz().replace(tzinfo=None) < time(15, 0):
            candidates = tuple(item for item in snapshot.sessions if item < cutoff)
        else:
            candidates = tuple(item for item in snapshot.sessions if item <= cutoff)
        if not candidates:
            raise CalendarUnavailableError("CALENDAR_COMPLETED_SESSION_UNAVAILABLE")
        if cutoff > snapshot.identity.covered_to:
            raise CalendarUnavailableError("CALENDAR_RANGE_UNAVAILABLE")
        return candidates[-1]

    @staticmethod
    def _write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _load(self, path: Path | None = None) -> _CalendarSnapshot:
        source = self._path if path is None else path
        try:
            if source.stat().st_size > _MAX_CACHE_BYTES:
                raise CalendarUnavailableError("CALENDAR_CACHE_TOO_LARGE")
        except CalendarUnavailableError:
            raise
        except OSError:
            raise CalendarUnavailableError("CALENDAR_CACHE_INVALID") from None
        try:
            text = source.read_text("utf-8")
            raw = json.loads(text)
            expected = {
                "calendar_id",
                "covered_from",
                "covered_to",
                "exchange",
                "provider",
                "provider_version",
                "schema_version",
                "sessions",
            }
            if type(raw) is not dict or set(raw) != expected or _canonical(raw) != text:
                raise ValueError
            if raw["exchange"] != "SSE_SZSE" or type(raw["schema_version"]) is not int or raw["schema_version"] != _SCHEMA_VERSION:
                raise ValueError
            calendar_id = raw.pop("calendar_id")
            if type(calendar_id) is not str or _HASH.fullmatch(calendar_id) is None:
                raise ValueError
            if sha256(_canonical(raw).encode("utf-8")).hexdigest() != calendar_id:
                raise ValueError
            covered_from = date.fromisoformat(raw["covered_from"])
            covered_to = date.fromisoformat(raw["covered_to"])
            sessions = tuple(date.fromisoformat(item) for item in raw["sessions"])
            if (
                type(raw["sessions"]) is not list
                or not sessions
                or tuple(sorted(set(sessions))) != sessions
                or sessions[0] < covered_from
                or sessions[-1] > covered_to
            ):
                raise ValueError
            return _CalendarSnapshot(
                CalendarIdentity(
                    calendar_id,
                    _stable(raw["provider"], label="calendar provider"),
                    _stable(raw["provider_version"], label="calendar provider version"),
                    covered_from,
                    covered_to,
                ),
                sessions,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            raise CalendarUnavailableError("CALENDAR_CACHE_INVALID") from None

    @staticmethod
    def _bounded_sessions(values: Sequence[date]) -> tuple[date, ...]:
        sessions: list[date] = []
        for index, item in enumerate(values):
            if index >= _MAX_SESSIONS:
                raise ValueError("calendar provider returned too many sessions")
            sessions.append(_exact_day(item, label="exchange session"))
        return tuple(sessions)
