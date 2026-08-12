from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
import socket
from threading import Lock, RLock
from time import monotonic
from typing import Any, TypeVar


T = TypeVar("T")
MINIMUM_MARKET_TIMEOUT_SECONDS = 3
MAXIMUM_MARKET_TIMEOUT_SECONDS = 60
DEFAULT_MARKET_TIMEOUT_SECONDS = 5
_REQUEST_TIMEOUT: ContextVar[int | None] = ContextVar(
    "compass_request_timeout",
    default=None,
)
_REQUESTS_PATCH_LOCK = Lock()
_REQUESTS_PATCHED = False


def validate_market_timeout_seconds(value: object) -> int:
    if (
        type(value) is not int
        or not MINIMUM_MARKET_TIMEOUT_SECONDS <= value <= MAXIMUM_MARKET_TIMEOUT_SECONDS
    ):
        raise ValueError("MARKET_REQUEST_TIMEOUT_INVALID")
    return value


def requests_call_with_timeout(seconds: int, call: Callable[[], T]) -> T:
    checked = validate_market_timeout_seconds(seconds)
    _ensure_requests_timeout_patch()
    token = _REQUEST_TIMEOUT.set(checked)
    try:
        return call()
    finally:
        _REQUEST_TIMEOUT.reset(token)


def _ensure_requests_timeout_patch() -> None:
    global _REQUESTS_PATCHED
    if _REQUESTS_PATCHED:
        return
    with _REQUESTS_PATCH_LOCK:
        if _REQUESTS_PATCHED:
            return
        import requests

        original = requests.sessions.Session.request

        def request_with_configured_timeout(
            session: Any,
            method: str,
            url: str,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            timeout = _REQUEST_TIMEOUT.get()
            if timeout is not None:
                kwargs["timeout"] = timeout
            return original(session, method, url, *args, **kwargs)

        setattr(requests.sessions.Session, "request", request_with_configured_timeout)
        _REQUESTS_PATCHED = True


class TimedSocketModule:
    """The subset of ``socket`` used by BaoStock, with a configurable timeout."""

    AF_INET = socket.AF_INET
    SOCK_STREAM = socket.SOCK_STREAM

    def __init__(self, seconds: int) -> None:
        self._seconds = validate_market_timeout_seconds(seconds)
        self._lock = RLock()

    def set_timeout(self, seconds: int) -> None:
        checked = validate_market_timeout_seconds(seconds)
        with self._lock:
            self._seconds = checked

    def socket(self, *args: object, **kwargs: object) -> DeadlineSocket:
        created = socket.socket(*args, **kwargs)  # type: ignore[arg-type]
        with self._lock:
            seconds = self._seconds
        return DeadlineSocket(created, seconds)


class DeadlineSocket:
    """Socket proxy whose connect/send/receive phases share one deadline."""

    def __init__(self, wrapped: socket.socket, seconds: int) -> None:
        self._wrapped = wrapped
        self._lock = RLock()
        self._seconds = validate_market_timeout_seconds(seconds)
        self._deadline = monotonic() + self._seconds
        self._wrapped.settimeout(self._seconds)

    def reset_deadline(self, seconds: int) -> None:
        checked = validate_market_timeout_seconds(seconds)
        with self._lock:
            self._seconds = checked
            self._deadline = monotonic() + checked
            self._wrapped.settimeout(checked)

    def connect(self, address: object) -> None:
        self._apply_remaining()
        self._wrapped.connect(address)  # type: ignore[arg-type]

    def send(self, data: bytes) -> int:
        self._apply_remaining()
        return self._wrapped.send(data)

    def recv(self, size: int) -> bytes:
        self._apply_remaining()
        return self._wrapped.recv(size)

    def close(self) -> None:
        self._wrapped.close()

    def settimeout(self, seconds: float | None) -> None:
        if seconds is None or int(seconds) != seconds:
            self._wrapped.settimeout(seconds)
            return
        self.reset_deadline(int(seconds))

    def gettimeout(self) -> float | None:
        return self._wrapped.gettimeout()

    def _apply_remaining(self) -> None:
        with self._lock:
            remaining = self._deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("MARKET_REQUEST_TIMEOUT")
        self._wrapped.settimeout(remaining)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)
