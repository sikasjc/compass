from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import hashlib
import logging
from threading import Condition, RLock
from types import TracebackType
from typing import TypeAlias
from uuid import uuid4
from zoneinfo import ZoneInfo

from compass.services.safe_display import (
    safe_display_text,
    safe_exception_type,
    safe_identifier,
    stable_code,
)
from compass.services.diagnostic_log import safe_diagnostic_text


SHANGHAI = ZoneInfo("Asia/Shanghai")
_LOGGER = logging.getLogger("compass.tasks")
Operation: TypeAlias = Callable[[], object]
Clock: TypeAlias = Callable[[], datetime]
IdFactory: TypeAlias = Callable[[], str]


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLATION_REQUESTED = "cancellation_requested"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


_TERMINAL = {TaskStatus.CANCELLED, TaskStatus.SUCCEEDED, TaskStatus.FAILED}


class TaskManagerError(RuntimeError):
    """Base class for stable task-manager domain errors."""


class TaskConflictError(TaskManagerError):
    """A requested task conflicts with active work or an existing identity."""


class TaskUnknownError(TaskManagerError):
    """The requested task identity is not known to this manager."""


class TaskManagerClosedError(TaskManagerError):
    """New work was submitted after shutdown began."""


class TaskSubmissionError(TaskManagerError):
    """A task could not be admitted through an injected submission seam."""

    code = "TASK_SUBMISSION_FAILED"

    def __init__(self) -> None:
        super().__init__(self.code)


class TaskOperationError(RuntimeError):
    """An expected operation failure whose stable code is safe to display."""

    def __init__(self, code: str) -> None:
        self.code = stable_code(code, label="task operation error code")
        super().__init__(self.code)


@dataclass(frozen=True, slots=True)
class TaskFailure:
    code: str
    error_id: str
    exception_type: str

    def __post_init__(self) -> None:
        stable_code(self.code, label="task failure code")
        safe_identifier(self.error_id, label="task error id")
        safe_exception_type(self.exception_type)


@dataclass(frozen=True, slots=True)
class TaskSnapshot:
    task_id: str
    name: str
    heavy: bool
    status: TaskStatus
    submitted_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failure: TaskFailure | None = None

    def __post_init__(self) -> None:
        safe_identifier(self.task_id, label="task id")
        safe_display_text(self.name, label="task name")
        if type(self.heavy) is not bool:
            raise TypeError("task heavy must be an exact bool")
        if type(self.status) is not TaskStatus:
            raise TypeError("task status must be an exact TaskStatus")
        for label, value in (
            ("submitted_at", self.submitted_at),
            ("started_at", self.started_at),
            ("completed_at", self.completed_at),
        ):
            if value is None and label != "submitted_at":
                continue
            if type(value) is not datetime:
                raise TypeError(f"{label} must be an exact datetime")
            assert isinstance(value, datetime)
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{label} must be timezone-aware")
        if self.started_at is not None and self.started_at < self.submitted_at:
            raise ValueError("task start cannot precede submission")
        lower_bound = self.started_at or self.submitted_at
        if self.completed_at is not None and self.completed_at < lower_bound:
            raise ValueError("task completion cannot precede its prior transition")
        if self.failure is not None and type(self.failure) is not TaskFailure:
            raise TypeError("task failure must be an exact TaskFailure")
        self._validate_state()

    def _validate_state(self) -> None:
        if self.status is TaskStatus.QUEUED:
            valid = self.started_at is None and self.completed_at is None and self.failure is None
        elif self.status is TaskStatus.RUNNING:
            valid = self.started_at is not None and self.completed_at is None and self.failure is None
        elif self.status is TaskStatus.CANCELLATION_REQUESTED:
            valid = self.completed_at is None and self.failure is None
        elif self.status is TaskStatus.CANCELLED:
            valid = self.completed_at is not None and self.failure is None
        elif self.status is TaskStatus.SUCCEEDED:
            valid = (
                self.started_at is not None
                and self.completed_at is not None
                and self.failure is None
            )
        else:
            valid = self.completed_at is not None and self.failure is not None
        if not valid:
            raise ValueError("task snapshot lifecycle fields are inconsistent")


@dataclass(slots=True)
class _TaskRecord:
    task_id: str
    generation: int
    name: str
    heavy: bool
    status: TaskStatus
    submitted_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failure: TaskFailure | None = None
    future: Future[None] | None = None


def _now() -> datetime:
    return datetime.now(SHANGHAI)


def _identifier() -> str:
    return uuid4().hex


class TaskManager:
    """Thread-safe local executor with one active heavy-task reservation."""

    def __init__(
        self,
        *,
        executor: Executor | None = None,
        clock: Clock = _now,
        id_factory: IdFactory = _identifier,
        error_id_factory: IdFactory = _identifier,
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        if not callable(id_factory):
            raise TypeError("id_factory must be callable")
        if not callable(error_id_factory):
            raise TypeError("error_id_factory must be callable")
        self._executor = executor or ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="compass-task",
        )
        self._clock = clock
        self._id_factory = id_factory
        self._error_id_factory = error_id_factory
        self._records: dict[str, _TaskRecord] = {}
        self._error_ids: set[str] = set()
        self._next_generation = 1
        self._heavy_task_id: str | None = None
        self._closed = False
        self._lock = RLock()
        self._changed = Condition(self._lock)

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def submit(self, name: str, heavy: bool, operation: Operation) -> TaskSnapshot:
        if type(name) is not str or not name.strip():
            raise ValueError("task name must be non-empty text")
        if type(heavy) is not bool:
            raise TypeError("heavy must be an exact bool")
        if not callable(operation):
            raise TypeError("operation must be callable")
        clean_name = safe_display_text(name.strip(), label="task name")
        with self._lock:
            if self._closed:
                raise TaskManagerClosedError("TASK_MANAGER_CLOSED")
            if heavy and self._heavy_task_id is not None:
                raise TaskConflictError("HEAVY_TASK_ACTIVE")
            task_id = self._submission_id()
            if task_id in self._records:
                raise TaskConflictError("TASK_ID_CONFLICT")
            generation = self._next_generation
            self._next_generation += 1
            submitted_at = self._submission_timestamp()
            record = _TaskRecord(
                task_id=task_id,
                generation=generation,
                name=clean_name,
                heavy=heavy,
                status=TaskStatus.QUEUED,
                submitted_at=submitted_at,
            )
            self._records[task_id] = record
            if heavy:
                self._heavy_task_id = task_id
            submission_failed = False
            try:
                future = self._executor.submit(
                    self._execute,
                    task_id,
                    generation,
                    operation,
                )
            except Exception:
                self._records.pop(task_id, None)
                if self._heavy_task_id == task_id:
                    self._heavy_task_id = None
                submission_failed = True
            if submission_failed:
                raise TaskSubmissionError()
            assert future is not None
            record.future = future
            self._changed.notify_all()
            return self._snapshot(record)

    def status(self, task_id: str) -> TaskSnapshot:
        with self._lock:
            return self._snapshot(self._record(task_id))

    def snapshots(self) -> tuple[TaskSnapshot, ...]:
        with self._lock:
            return tuple(self._snapshot(record) for record in self._records.values())

    def discard_terminal(self, task_id: str) -> bool:
        with self._lock:
            record = self._record(task_id)
            if record.status not in _TERMINAL:
                raise TaskConflictError("TASK_NOT_TERMINAL")
            del self._records[task_id]
            self._changed.notify_all()
            return True

    def clear_terminal(self, *, name_prefix: str | None = None) -> int:
        if name_prefix is not None and (type(name_prefix) is not str or not name_prefix):
            raise ValueError("task name prefix must be non-empty text or None")
        with self._lock:
            task_ids = tuple(
                task_id
                for task_id, record in self._records.items()
                if record.status in _TERMINAL
                and (name_prefix is None or record.name.startswith(name_prefix))
            )
            for task_id in task_ids:
                del self._records[task_id]
            if task_ids:
                self._changed.notify_all()
            return len(task_ids)

    def cancel(self, task_id: str) -> TaskSnapshot:
        with self._lock:
            record = self._record(task_id)
            if record.status in _TERMINAL or record.status is TaskStatus.CANCELLATION_REQUESTED:
                return self._snapshot(record)
            future = record.future
            if record.status is TaskStatus.QUEUED and future is not None and future.cancel():
                self._finish_cancelled(record)
            else:
                record.status = TaskStatus.CANCELLATION_REQUESTED
                self._changed.notify_all()
            return self._snapshot(record)

    def wait(self, task_id: str, timeout: float | None = None) -> TaskSnapshot:
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout < 0:
                raise ValueError("timeout must be a non-negative number or None")
        with self._changed:
            record = self._record(task_id)
            completed = self._changed.wait_for(lambda: record.status in _TERMINAL, timeout)
            if not completed:
                raise TimeoutError("TASK_WAIT_TIMEOUT")
            return self._snapshot(record)

    def shutdown(self, *, wait: bool = True) -> None:
        if type(wait) is not bool:
            raise TypeError("wait must be an exact bool")
        with self._lock:
            first_shutdown = not self._closed
            self._closed = True
            if first_shutdown:
                for record in self._records.values():
                    if record.status in _TERMINAL:
                        continue
                    future = record.future
                    if (
                        record.status is TaskStatus.QUEUED
                        and future is not None
                        and future.cancel()
                    ):
                        self._finish_cancelled(record)
                    else:
                        record.status = TaskStatus.CANCELLATION_REQUESTED
                self._changed.notify_all()
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def __enter__(self) -> TaskManager:
        with self._lock:
            if self._closed:
                raise TaskManagerClosedError("TASK_MANAGER_CLOSED")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.shutdown()

    def _execute(self, task_id: str, generation: int, operation: Operation) -> None:
        try:
            with self._lock:
                record = self._records.get(task_id)
                if record is None or record.generation != generation:
                    return
                if record.status is TaskStatus.CANCELLATION_REQUESTED:
                    self._finish_cancelled(record)
                    return
                if record.status is not TaskStatus.QUEUED:
                    return
                started_at = self._timestamp()
                if started_at < record.submitted_at:
                    raise ValueError("task clock moved backwards at start")
                record.started_at = started_at
                record.status = TaskStatus.RUNNING
                self._changed.notify_all()
            failure: TaskFailure | None = None
            try:
                operation()
            except TaskOperationError as error:
                failure = self._failure(
                    record,
                    code=error.code,
                    exception_type=type(error).__name__,
                )
                _LOGGER.warning(
                    "task failed name=%s task_id=%s error_id=%s code=%s exception=%s detail=%s",
                    record.name,
                    record.task_id,
                    failure.error_id,
                    failure.code,
                    failure.exception_type,
                    safe_diagnostic_text(error),
                )
            except BaseException as error:
                failure = self._failure(
                    record,
                    code="TASK_OPERATION_FAILED",
                    exception_type=type(error).__name__,
                )
                _LOGGER.error(
                    "task failed name=%s task_id=%s error_id=%s code=%s exception=%s detail=%s",
                    record.name,
                    record.task_id,
                    failure.error_id,
                    failure.code,
                    failure.exception_type,
                    safe_diagnostic_text(error),
                )
            with self._lock:
                current = self._records.get(task_id)
                if current is not record or current.generation != generation:
                    return
                if current.status is TaskStatus.CANCELLATION_REQUESTED:
                    self._finish_cancelled(current)
                    return
                completed_at = self._timestamp()
                lower_bound = current.started_at or current.submitted_at
                if completed_at < lower_bound:
                    raise ValueError("task clock moved backwards at completion")
                current.completed_at = completed_at
                current.status = TaskStatus.FAILED if failure is not None else TaskStatus.SUCCEEDED
                current.failure = failure
                self._release_heavy(current)
                self._changed.notify_all()
        except BaseException as error:
            self._mark_internal_failure(task_id, generation, error)

    def _mark_internal_failure(
        self,
        task_id: str,
        generation: int,
        error: BaseException,
    ) -> None:
        with self._lock:
            record = self._records.get(task_id)
            if record is None or record.generation != generation or record.status in _TERMINAL:
                return
            if record.status is TaskStatus.CANCELLATION_REQUESTED:
                self._finish_cancelled(record)
                return
            record.status = TaskStatus.FAILED
            record.failure = self._failure(
                record,
                code="TASK_MANAGER_INTERNAL_FAILED",
                exception_type=type(error).__name__,
            )
            record.completed_at = self._safe_completion_timestamp(record)
            self._release_heavy(record)
            self._changed.notify_all()

    def _failure(self, record: _TaskRecord, *, code: str, exception_type: str) -> TaskFailure:
        with self._lock:
            try:
                candidate = self._new_id(self._error_id_factory, "error")
            except BaseException:
                candidate = ""
            digest = hashlib.sha256(
                f"{record.task_id}:{record.generation}".encode("utf-8")
            ).hexdigest()[:32]
            fallback = f"error-{digest}"
            error_id = candidate if candidate and candidate not in self._error_ids else fallback
            suffix = 1
            while error_id in self._error_ids:
                error_id = f"{fallback}-{suffix}"
                suffix += 1
            self._error_ids.add(error_id)
        return TaskFailure(code=code, error_id=error_id, exception_type=exception_type)

    def _finish_cancelled(self, record: _TaskRecord) -> None:
        record.status = TaskStatus.CANCELLED
        record.failure = None
        record.completed_at = self._safe_completion_timestamp(record)
        self._release_heavy(record)
        self._changed.notify_all()

    def _safe_completion_timestamp(self, record: _TaskRecord) -> datetime:
        lower_bound = record.started_at or record.submitted_at
        try:
            candidate = self._timestamp()
        except BaseException:
            return lower_bound
        return max(candidate, lower_bound)

    def _release_heavy(self, record: _TaskRecord) -> None:
        if record.heavy and self._heavy_task_id == record.task_id:
            self._heavy_task_id = None

    def _record(self, task_id: str) -> _TaskRecord:
        if type(task_id) is not str or task_id not in self._records:
            raise TaskUnknownError("UNKNOWN_TASK")
        return self._records[task_id]

    def _timestamp(self) -> datetime:
        value = self._clock()
        if type(value) is not datetime:
            raise TypeError("task clock must return an exact datetime")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("task clock must return a timezone-aware datetime")
        return value.astimezone(SHANGHAI)

    @staticmethod
    def _new_id(factory: IdFactory, label: str) -> str:
        value = factory()
        return safe_identifier(value, label=f"{label} id")

    def _submission_id(self) -> str:
        failed = False
        try:
            value = self._new_id(self._id_factory, "task")
        except Exception:
            failed = True
            value = ""
        if failed:
            raise TaskSubmissionError()
        return value

    def _submission_timestamp(self) -> datetime:
        failed = False
        try:
            value = self._timestamp()
        except Exception:
            failed = True
            value = datetime.now(SHANGHAI)
        if failed:
            raise TaskSubmissionError()
        return value

    @staticmethod
    def _snapshot(record: _TaskRecord) -> TaskSnapshot:
        return TaskSnapshot(
            task_id=record.task_id,
            name=record.name,
            heavy=record.heavy,
            status=record.status,
            submitted_at=record.submitted_at,
            started_at=record.started_at,
            completed_at=record.completed_at,
            failure=record.failure,
        )
