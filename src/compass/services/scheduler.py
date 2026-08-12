from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Condition, Event, RLock, Thread, current_thread
from types import TracebackType
from zoneinfo import ZoneInfo


SHANGHAI = ZoneInfo("Asia/Shanghai")
Clock = Callable[[], datetime]
Wait = Callable[[Event, float], bool]
FiveMinuteCallback = Callable[[datetime], None]
FailureReporter = Callable[["SchedulerFailure"], None]


def _default_clock() -> datetime:
    return datetime.now(SHANGHAI)


def _default_wait(stop: Event, delay: float) -> bool:
    return stop.wait(delay)


def _aware_shanghai(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("scheduler clock must return a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("scheduler clock must return a timezone-aware datetime")
    return value.astimezone(SHANGHAI)


@dataclass(frozen=True, slots=True)
class SchedulerFailure:
    code: str
    callback_name: str
    boundary: datetime | None
    exception_type: str

    def __post_init__(self) -> None:
        if type(self.code) is not str or not self.code:
            raise ValueError("scheduler failure code must be non-empty")
        if type(self.callback_name) is not str or not self.callback_name:
            raise ValueError("scheduler callback_name must be non-empty")
        if self.boundary is not None:
            object.__setattr__(self, "boundary", _aware_shanghai(self.boundary))
        if type(self.exception_type) is not str or not self.exception_type:
            raise ValueError("scheduler exception_type must be non-empty")


class LocalScheduler:
    """One local five-minute wall-clock loop with race-safe registration."""

    def __init__(
        self,
        *,
        clock: Clock = _default_clock,
        wait: Wait = _default_wait,
        on_failure: FailureReporter | None = None,
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        if not callable(wait):
            raise TypeError("wait must be callable")
        if on_failure is not None and not callable(on_failure):
            raise TypeError("on_failure must be callable or None")
        self._clock = clock
        self._wait = wait
        self._on_failure = on_failure
        self._callbacks: dict[int, FiveMinuteCallback] = {}
        self._active_callbacks: set[int] = set()
        self._next_handle = 1
        self._failures: list[SchedulerFailure] = []
        self._thread: Thread | None = None
        self._stop = Event()
        self._lock = RLock()
        self._callback_changed = Condition(self._lock)

    @property
    def running(self) -> bool:
        with self._lock:
            return (
                self._thread is not None
                and self._thread.is_alive()
                and not self._stop.is_set()
            )

    @property
    def failures(self) -> tuple[SchedulerFailure, ...]:
        with self._lock:
            return tuple(self._failures)

    def register_five_minute(self, callback: FiveMinuteCallback) -> int:
        if not callable(callback):
            raise TypeError("five-minute callback must be callable")
        with self._lock:
            if any(self._same_callback(existing, callback) for existing in self._callbacks.values()):
                raise ValueError("five-minute callback is already registered")
            handle = self._next_handle
            self._next_handle += 1
            self._callbacks[handle] = callback
            return handle

    def unregister_five_minute(self, handle: int) -> None:
        if type(handle) is not int or handle <= 0:
            raise ValueError("callback handle must be a positive exact integer")
        with self._lock:
            self._callbacks.pop(handle, None)
            if current_thread() is not self._thread:
                self._callback_changed.wait_for(
                    lambda: handle not in self._active_callbacks
                )

    @staticmethod
    def _same_callback(left: FiveMinuteCallback, right: FiveMinuteCallback) -> bool:
        if left is right:
            return True
        left_instance = getattr(left, "__self__", None)
        right_instance = getattr(right, "__self__", None)
        left_function = getattr(left, "__func__", None)
        right_function = getattr(right, "__func__", None)
        if left_instance is None or left_instance is not right_instance:
            return False
        if left_function is not None or right_function is not None:
            return left_function is not None and left_function is right_function
        left_name = getattr(left, "__name__", None)
        right_name = getattr(right, "__name__", None)
        return (
            type(left) is type(right)
            and type(left_name) is str
            and left_name == right_name
        )

    def start(self) -> None:
        while True:
            with self._lock:
                previous = self._thread
                if previous is None or not previous.is_alive():
                    self._stop = Event()
                    thread = Thread(
                        target=self._run,
                        name="compass-scheduler",
                        daemon=True,
                    )
                    self._thread = thread
                    thread.start()
                    return
                if not self._stop.is_set():
                    return
            if previous is current_thread():
                return
            previous.join()

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            self._stop.set()
        if thread is not None and thread is not current_thread():
            thread.join()
        with self._lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None

    def __enter__(self) -> LocalScheduler:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.stop()

    @staticmethod
    def _next_boundary(now: datetime) -> datetime:
        minute = (now.minute // 5 + 1) * 5
        base = now.replace(second=0, microsecond=0)
        if minute == 60:
            return base.replace(minute=0) + timedelta(hours=1)
        return base.replace(minute=minute)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                now = _aware_shanghai(self._clock())
                boundary = self._next_boundary(now)
            except Exception as error:
                self._record_failure(
                    "SCHEDULER_CLOCK_FAILED",
                    "scheduler_clock",
                    None,
                    error,
                )
                return
            try:
                stopped = self._wait(self._stop, (boundary - now).total_seconds())
                if type(stopped) is not bool:
                    raise TypeError("wait must return an exact bool")
            except Exception as error:
                self._record_failure(
                    "SCHEDULER_WAIT_FAILED",
                    "scheduler_wait",
                    boundary,
                    error,
                )
                return
            if stopped or self._stop.is_set():
                return
            with self._lock:
                callbacks = tuple(sorted(self._callbacks.items()))
            for handle, callback in callbacks:
                if self._stop.is_set():
                    return
                with self._lock:
                    if self._callbacks.get(handle) is not callback:
                        continue
                    self._active_callbacks.add(handle)
                try:
                    callback(boundary)
                except Exception as error:
                    self._record_failure(
                        "CALLBACK_FAILED",
                        self._callback_name(callback),
                        boundary,
                        error,
                    )
                finally:
                    with self._lock:
                        self._active_callbacks.discard(handle)
                        self._callback_changed.notify_all()

    @staticmethod
    def _callback_name(callback: FiveMinuteCallback) -> str:
        name = getattr(callback, "__name__", None)
        return name if type(name) is str and name else type(callback).__name__

    def _record_failure(
        self,
        code: str,
        callback_name: str,
        boundary: datetime | None,
        error: Exception,
    ) -> None:
        failure = SchedulerFailure(code, callback_name, boundary, type(error).__name__)
        with self._lock:
            self._failures.append(failure)
        if self._on_failure is not None:
            try:
                self._on_failure(failure)
            except Exception:
                pass
