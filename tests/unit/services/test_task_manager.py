from __future__ import annotations

from concurrent.futures import Executor, Future, ThreadPoolExecutor
from datetime import datetime, timedelta
from threading import Barrier, Event, Lock, Thread
from zoneinfo import ZoneInfo

import pytest

from compass.services.task_manager import (
    TaskConflictError,
    TaskManager,
    TaskManagerClosedError,
    TaskManagerError,
    TaskOperationError,
    TaskFailure,
    TaskSnapshot,
    TaskStatus,
    TaskUnknownError,
)


SHANGHAI = ZoneInfo("Asia/Shanghai")


class FakeClock:
    def __init__(self) -> None:
        self._value = datetime(2026, 7, 23, 9, 0, tzinfo=SHANGHAI)
        self._lock = Lock()

    def __call__(self) -> datetime:
        with self._lock:
            value = self._value
            self._value += timedelta(seconds=1)
            return value


def _ids(*values: str):
    remaining = iter(values)
    return lambda: next(remaining)


def _public_exception_surface(error: BaseException) -> str:
    return " ".join(
        (
            str(error),
            repr(error),
            repr(error.__cause__),
            repr(error.__context__),
        )
    )


class FailingOnceExecutor(Executor):
    def __init__(self) -> None:
        self.failed = False

    def submit(self, fn, /, *args, **kwargs):
        if not self.failed:
            self.failed = True
            raise RuntimeError("token=executor-secret C:\\Users\\private")
        future: Future[object] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as error:
            future.set_exception(error)
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        del wait, cancel_futures


def test_task_transitions_are_immutable_ordered_and_deterministic() -> None:
    release = Event()
    entered = Event()
    manager = TaskManager(clock=FakeClock(), id_factory=_ids("task-a", "task-b"))

    def blocked() -> object:
        entered.set()
        release.wait()
        return {"mutable": ["result"]}

    first = manager.submit("同步行情", heavy=False, operation=blocked)
    assert entered.wait(1)
    second = manager.submit("刷新状态", heavy=False, operation=lambda: None)
    assert manager.wait(second.task_id, 1).status is TaskStatus.SUCCEEDED

    running = manager.status(first.task_id)
    assert running.status is TaskStatus.RUNNING
    assert running.task_id == "task-a"
    assert running.submitted_at < running.started_at
    assert manager.snapshots() == (running, manager.status(second.task_id))
    with pytest.raises((AttributeError, TypeError)):
        running.status = TaskStatus.FAILED  # type: ignore[misc]

    release.set()
    succeeded = manager.wait(first.task_id, 1)
    manager.shutdown()

    assert succeeded.status is TaskStatus.SUCCEEDED
    assert succeeded.completed_at is not None
    assert not hasattr(succeeded, "result")


def test_concurrent_submit_admits_only_one_active_heavy_task() -> None:
    manager = TaskManager(clock=FakeClock())
    barrier = Barrier(3)
    release = Event()
    admitted: list[str] = []
    conflicts: list[TaskConflictError] = []

    def operation() -> None:
        release.wait()

    def submit(name: str) -> None:
        barrier.wait()
        try:
            admitted.append(manager.submit(name, heavy=True, operation=operation).task_id)
        except TaskConflictError as error:
            conflicts.append(error)

    workers = [Thread(target=submit, args=(name,)) for name in ("sync", "backtest")]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join()

    assert len(admitted) == 1
    assert len(conflicts) == 1
    assert str(conflicts[0]) == "HEAVY_TASK_ACTIVE"
    manager.cancel(admitted[0])
    release.set()
    assert manager.wait(admitted[0], 1).status is TaskStatus.CANCELLED
    manager.shutdown()


def test_running_cancellation_retains_heavy_reservation_and_discards_completion() -> None:
    entered = Event()
    release = Event()
    manager = TaskManager(clock=FakeClock(), id_factory=_ids("heavy-1", "heavy-2"))

    def blocked() -> str:
        entered.set()
        release.wait()
        return "must be discarded"

    task = manager.submit("sync", heavy=True, operation=blocked)
    assert entered.wait(1)
    requested = manager.cancel(task.task_id)
    repeated = manager.cancel(task.task_id)

    assert requested.status is TaskStatus.CANCELLATION_REQUESTED
    assert repeated == requested
    with pytest.raises(TaskConflictError, match="HEAVY_TASK_ACTIVE"):
        manager.submit("backtest", heavy=True, operation=lambda: None)

    release.set()
    assert manager.wait(task.task_id, 1).status is TaskStatus.CANCELLED
    next_task = manager.submit("backtest", heavy=True, operation=lambda: None)
    assert manager.wait(next_task.task_id, 1).status is TaskStatus.SUCCEEDED
    manager.shutdown()


def test_queued_cancellation_is_immediate_and_never_runs_operation() -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    manager = TaskManager(executor=executor, clock=FakeClock())
    blocker_entered = Event()
    release = Event()
    queued_called = Event()

    def blocker() -> None:
        blocker_entered.set()
        release.wait()

    first = manager.submit("blocker", heavy=False, operation=blocker)
    assert blocker_entered.wait(1)
    queued = manager.submit("queued", heavy=True, operation=queued_called.set)

    assert manager.cancel(queued.task_id).status is TaskStatus.CANCELLED
    replacement = manager.submit("replacement", heavy=True, operation=lambda: None)
    release.set()
    assert manager.wait(first.task_id, 1).status is TaskStatus.SUCCEEDED
    assert manager.wait(replacement.task_id, 1).status is TaskStatus.SUCCEEDED
    assert queued_called.is_set() is False
    manager.shutdown()


def test_worker_failure_is_stable_secret_safe_and_does_not_escape(caplog) -> None:  # type: ignore[no-untyped-def]
    manager = TaskManager(
        clock=FakeClock(),
        id_factory=_ids("task-fail"),
        error_id_factory=_ids("error-local-1"),
    )
    secret = "token=top-secret C:\\Users\\private\\portfolio.env"

    def fail() -> None:
        raise RuntimeError(secret)

    failed = manager.wait(manager.submit("sync", heavy=False, operation=fail).task_id, 1)
    manager.shutdown()

    assert failed.status is TaskStatus.FAILED
    assert failed.failure is not None
    assert failed.failure.code == "TASK_OPERATION_FAILED"
    assert failed.failure.error_id == "error-local-1"
    assert failed.failure.exception_type == "RuntimeError"
    assert secret not in repr(failed)
    assert "top-secret" not in repr(failed)
    assert "Users" not in repr(failed)
    assert "error-local-1" in caplog.text
    assert "top-secret" not in caplog.text
    assert "token=[redacted]" in caplog.text


def test_expected_operation_failure_preserves_only_its_stable_actionable_code() -> None:
    manager = TaskManager(
        clock=FakeClock(),
        id_factory=_ids("task-expected-failure"),
        error_id_factory=_ids("error-expected-failure"),
    )

    failed = manager.wait(
        manager.submit(
            "sync",
            heavy=False,
            operation=lambda: (_ for _ in ()).throw(
                TaskOperationError("SYNC_WATCHLIST_MISSING")
            ),
        ).task_id,
        1,
    )
    manager.shutdown()

    assert failed.status is TaskStatus.FAILED
    assert failed.failure is not None
    assert failed.failure.code == "SYNC_WATCHLIST_MISSING"
    assert failed.failure.error_id == "error-expected-failure"
    assert failed.failure.exception_type == "TaskOperationError"
    with pytest.raises(ValueError):
        TaskOperationError("token=secret")


def test_unknown_ids_invalid_submissions_and_duplicate_ids_are_stable_errors() -> None:
    manager = TaskManager(clock=FakeClock(), id_factory=_ids("same", "same"))

    with pytest.raises(TaskUnknownError, match="UNKNOWN_TASK"):
        manager.status("missing")
    with pytest.raises(TaskUnknownError, match="UNKNOWN_TASK"):
        manager.cancel("missing")
    with pytest.raises(ValueError, match="task name"):
        manager.submit("", heavy=False, operation=lambda: None)
    with pytest.raises(TypeError, match="heavy"):
        manager.submit("task", heavy=1, operation=lambda: None)  # type: ignore[arg-type]

    task = manager.submit("first", heavy=False, operation=lambda: None)
    assert manager.wait(task.task_id, 1).status is TaskStatus.SUCCEEDED
    with pytest.raises(TaskConflictError, match="TASK_ID_CONFLICT"):
        manager.submit("second", heavy=False, operation=lambda: None)
    manager.shutdown()


def test_shutdown_and_context_manager_are_idempotent_and_reject_new_work() -> None:
    manager = TaskManager(clock=FakeClock())
    with manager as entered:
        assert entered is manager
        task = manager.submit("done", heavy=False, operation=lambda: None)
        assert manager.wait(task.task_id, 1).status is TaskStatus.SUCCEEDED

    manager.shutdown()
    assert manager.closed is True
    with pytest.raises(TaskManagerClosedError, match="TASK_MANAGER_CLOSED"):
        manager.submit("late", heavy=False, operation=lambda: None)


def test_wait_timeout_does_not_mutate_running_state() -> None:
    entered = Event()
    release = Event()
    manager = TaskManager(clock=FakeClock())

    def blocked() -> None:
        entered.set()
        release.wait()

    task = manager.submit("blocked", heavy=False, operation=blocked)
    assert entered.wait(1)
    with pytest.raises(TimeoutError, match="TASK_WAIT_TIMEOUT"):
        manager.wait(task.task_id, 0)
    assert manager.status(task.task_id).status is TaskStatus.RUNNING
    release.set()
    manager.shutdown()


def test_worker_clock_failure_cannot_leave_task_or_heavy_reservation_stuck() -> None:
    now = datetime(2026, 7, 23, 9, 0, tzinfo=SHANGHAI)
    calls = 0

    def broken_worker_clock() -> datetime:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("clock token=secret C:\\Users\\private")
        return now + timedelta(seconds=calls)

    manager = TaskManager(
        clock=broken_worker_clock,
        id_factory=_ids("heavy-broken", "heavy-next"),
        error_id_factory=_ids("clock-error"),
    )
    broken = manager.submit("broken", heavy=True, operation=lambda: None)

    failed = manager.wait(broken.task_id, 1)
    replacement = manager.submit("replacement", heavy=True, operation=lambda: None)

    assert failed.status is TaskStatus.FAILED
    assert failed.failure is not None
    assert failed.failure.code == "TASK_MANAGER_INTERNAL_FAILED"
    assert failed.failure.exception_type == "RuntimeError"
    assert "secret" not in repr(failed)
    assert manager.wait(replacement.task_id, 1).status is TaskStatus.SUCCEEDED
    manager.shutdown()


def test_error_id_factory_failure_uses_secret_safe_deterministic_fallback() -> None:
    def broken_error_id() -> str:
        raise RuntimeError("token=secret C:\\Users\\private")

    manager = TaskManager(
        clock=FakeClock(),
        id_factory=_ids("task-fallback"),
        error_id_factory=broken_error_id,
    )

    failed = manager.wait(
        manager.submit(
            "failure",
            heavy=False,
            operation=lambda: (_ for _ in ()).throw(ValueError("api_key=secret")),
        ).task_id,
        1,
    )

    assert failed.status is TaskStatus.FAILED
    assert failed.failure is not None
    assert failed.failure.error_id.startswith("error-")
    assert len(failed.failure.error_id) <= 128
    assert failed.failure.exception_type == "ValueError"
    manager.shutdown()


def test_second_shutdown_can_upgrade_nonblocking_shutdown_to_wait_for_exit() -> None:
    entered = Event()
    release = Event()
    wait_started = Event()
    wait_done = Event()
    manager = TaskManager(clock=FakeClock())

    def blocked() -> None:
        entered.set()
        release.wait()

    task = manager.submit("blocked", heavy=True, operation=blocked)
    assert entered.wait(1)
    manager.shutdown(wait=False)
    assert manager.status(task.task_id).status is TaskStatus.CANCELLATION_REQUESTED

    def wait_for_shutdown() -> None:
        wait_started.set()
        manager.shutdown(wait=True)
        wait_done.set()

    waiter = Thread(target=wait_for_shutdown)
    waiter.start()
    assert wait_started.wait(1)
    returned_before_operation_exit = wait_done.wait(0.1)
    release.set()
    assert wait_done.wait(1)
    waiter.join()

    assert returned_before_operation_exit is False
    assert manager.status(task.task_id).status is TaskStatus.CANCELLED


def test_submission_id_factory_failure_is_stable_secret_safe_and_leaves_no_reservation() -> None:
    calls = 0

    def flaky_id() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("token=id-secret C:\\Users\\private")
        return "task-after-id-failure"

    manager = TaskManager(clock=FakeClock(), id_factory=flaky_id)

    with pytest.raises(TaskManagerError) as captured:
        manager.submit("同步行情", heavy=True, operation=lambda: None)
    task = manager.submit("同步行情", heavy=True, operation=lambda: None)

    assert type(captured.value).__name__ == "TaskSubmissionError"
    assert getattr(captured.value, "code", None) == "TASK_SUBMISSION_FAILED"
    assert "secret" not in _public_exception_surface(captured.value)
    assert "Users" not in _public_exception_surface(captured.value)
    completed = manager.wait(task.task_id, 1)
    assert manager.snapshots() == (completed,)
    manager.shutdown()


def test_terminal_task_history_can_be_discarded_without_touching_active_tasks() -> None:
    release = Event()
    manager = TaskManager(
        clock=FakeClock(),
        id_factory=_ids("backtest-done", "backtest-active"),
    )
    completed = manager.submit("backtest:run-done", heavy=False, operation=lambda: None)
    assert manager.wait(completed.task_id, 1).status is TaskStatus.SUCCEEDED
    active = manager.submit(
        "backtest:run-active", heavy=False, operation=lambda: release.wait(1)
    )

    assert manager.clear_terminal(name_prefix="backtest:") == 1
    assert tuple(item.task_id for item in manager.snapshots()) == (active.task_id,)
    with pytest.raises(TaskManagerError, match="TASK_NOT_TERMINAL"):
        manager.discard_terminal(active.task_id)

    release.set()
    assert manager.wait(active.task_id, 1).status is TaskStatus.SUCCEEDED
    assert manager.discard_terminal(active.task_id) is True
    assert manager.snapshots() == ()
    manager.shutdown()


def test_submission_clock_failure_is_stable_secret_safe_and_leaves_no_reservation() -> None:
    calls = 0

    def flaky_clock() -> datetime:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("api_key=clock-secret C:\\Users\\private")
        return datetime(2026, 7, 23, 9, 0, calls, tzinfo=SHANGHAI)

    manager = TaskManager(clock=flaky_clock, id_factory=_ids("first-id", "second-id"))

    with pytest.raises(TaskManagerError) as captured:
        manager.submit("同步行情", heavy=True, operation=lambda: None)
    task = manager.submit("同步行情", heavy=True, operation=lambda: None)

    assert type(captured.value).__name__ == "TaskSubmissionError"
    assert "secret" not in _public_exception_surface(captured.value)
    assert manager.wait(task.task_id, 1).status is TaskStatus.SUCCEEDED
    manager.shutdown()


def test_executor_submission_failure_rolls_back_record_and_heavy_reservation() -> None:
    manager = TaskManager(
        executor=FailingOnceExecutor(),
        clock=FakeClock(),
        id_factory=_ids("executor-failed", "executor-next"),
    )

    with pytest.raises(TaskManagerError) as captured:
        manager.submit("同步行情", heavy=True, operation=lambda: None)
    assert manager.snapshots() == ()
    replacement = manager.submit("同步行情", heavy=True, operation=lambda: None)

    assert type(captured.value).__name__ == "TaskSubmissionError"
    assert "secret" not in _public_exception_surface(captured.value)
    assert manager.status(replacement.task_id).status is TaskStatus.SUCCEEDED
    manager.shutdown()


def test_duplicate_generated_error_ids_are_replaced_with_unique_safe_ids() -> None:
    manager = TaskManager(
        clock=FakeClock(),
        id_factory=_ids("failed-one", "failed-two"),
        error_id_factory=lambda: "duplicate-error",
    )

    first = manager.submit(
        "轻任务一", heavy=False, operation=lambda: (_ for _ in ()).throw(ValueError())
    )
    second = manager.submit(
        "轻任务二", heavy=False, operation=lambda: (_ for _ in ()).throw(ValueError())
    )
    first_failure = manager.wait(first.task_id, 1).failure
    second_failure = manager.wait(second.task_id, 1).failure

    assert first_failure is not None
    assert second_failure is not None
    assert first_failure.error_id != second_failure.error_id
    manager.shutdown()


def test_error_id_fallback_stays_safe_for_maximum_length_task_id() -> None:
    def broken_error_id() -> str:
        raise RuntimeError("token=secret")

    manager = TaskManager(
        clock=FakeClock(),
        id_factory=_ids("t" * 128),
        error_id_factory=broken_error_id,
    )
    task = manager.submit(
        "失败任务",
        heavy=False,
        operation=lambda: (_ for _ in ()).throw(ValueError()),
    )

    failed = manager.wait(task.task_id, 1)

    assert failed.status is TaskStatus.FAILED
    assert failed.failure is not None
    assert len(failed.failure.error_id) <= 128
    manager.shutdown()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TaskFailure("lowercase", "error-1", "RuntimeError"),
        lambda: TaskFailure("TASK_FAILED", "C:\\Users\\private", "RuntimeError"),
        lambda: TaskFailure("TASK_FAILED", "error-1", "Runtime Error token=secret"),
    ],
)
def test_task_failure_direct_construction_rejects_unsafe_fields(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TaskSnapshot(
            "task-1",
            "token=secret",
            False,
            TaskStatus.QUEUED,
            datetime(2026, 7, 23, 9, 0, tzinfo=SHANGHAI),
        ),
        lambda: TaskSnapshot(
            "task-1",
            "正常任务",
            False,
            TaskStatus.RUNNING,
            datetime(2026, 7, 23, 9, 0),
        ),
        lambda: TaskSnapshot(
            "task-1",
            "正常任务",
            False,
            TaskStatus.SUCCEEDED,
            datetime(2026, 7, 23, 9, 0, tzinfo=SHANGHAI),
        ),
        lambda: TaskSnapshot(
            "task-1",
            "正常任务",
            False,
            TaskStatus.FAILED,
            datetime(2026, 7, 23, 9, 0, tzinfo=SHANGHAI),
            datetime(2026, 7, 23, 9, 1, tzinfo=SHANGHAI),
            datetime(2026, 7, 23, 9, 2, tzinfo=SHANGHAI),
            None,
        ),
        lambda: TaskSnapshot(
            "task-1",
            "正常任务",
            False,
            TaskStatus.RUNNING,
            datetime(2026, 7, 23, 9, 2, tzinfo=SHANGHAI),
            datetime(2026, 7, 23, 9, 1, tzinfo=SHANGHAI),
        ),
    ],
)
def test_task_snapshot_direct_construction_rejects_unsafe_or_inconsistent_state(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_normal_chinese_task_name_remains_supported() -> None:
    manager = TaskManager(clock=FakeClock(), id_factory=_ids("chinese-name"))
    task = manager.submit("行情同步（主源）", heavy=False, operation=lambda: None)

    assert manager.wait(task.task_id, 1).name == "行情同步（主源）"
    manager.shutdown()
