from __future__ import annotations

from datetime import datetime, timedelta
from threading import Event, Lock, Thread
from zoneinfo import ZoneInfo

from compass.services.scheduler import LocalScheduler, SchedulerFailure


SHANGHAI = ZoneInfo("Asia/Shanghai")


class FakeClock:
    def __init__(self, now: datetime) -> None:
        self._now = now
        self._lock = Lock()

    def __call__(self) -> datetime:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += timedelta(seconds=seconds)


class AdvancingWaiter:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.delays: list[float] = []

    def __call__(self, stop: Event, delay: float) -> bool:
        self.delays.append(delay)
        if stop.is_set():
            return True
        self.clock.advance(delay)
        return False


def test_scheduler_aligns_to_five_minute_wall_clock_boundaries_without_drift() -> None:
    clock = FakeClock(datetime(2026, 7, 21, 10, 2, 30, tzinfo=SHANGHAI))
    waiter = AdvancingWaiter(clock)
    scheduler = LocalScheduler(clock=clock, wait=waiter)
    called: list[datetime] = []
    finished = Event()

    def callback(boundary: datetime) -> None:
        called.append(boundary)
        if len(called) == 1:
            clock.advance(120)
        else:
            scheduler.stop()
            finished.set()

    scheduler.register_five_minute(callback)
    scheduler.start()

    assert finished.wait(1)
    scheduler.stop()
    assert called == [
        datetime(2026, 7, 21, 10, 5, tzinfo=SHANGHAI),
        datetime(2026, 7, 21, 10, 10, tzinfo=SHANGHAI),
    ]
    assert waiter.delays[:2] == [150.0, 180.0]


def test_scheduler_start_and_stop_are_idempotent_and_only_one_loop_runs() -> None:
    entered = Event()
    waiter_calls = 0

    def blocking_wait(stop: Event, delay: float) -> bool:
        nonlocal waiter_calls
        del delay
        waiter_calls += 1
        entered.set()
        stop.wait()
        return True

    scheduler = LocalScheduler(
        clock=lambda: datetime(2026, 7, 21, 10, 2, tzinfo=SHANGHAI),
        wait=blocking_wait,
    )

    scheduler.start()
    scheduler.start()
    assert entered.wait(1)
    scheduler.stop()
    scheduler.stop()

    assert waiter_calls == 1
    assert scheduler.running is False


def test_scheduler_isolates_callback_exceptions_and_reports_them_explicitly() -> None:
    clock = FakeClock(datetime(2026, 7, 21, 10, 4, tzinfo=SHANGHAI))
    scheduler = LocalScheduler(clock=clock, wait=AdvancingWaiter(clock))
    reported: list[SchedulerFailure] = []
    healthy_calls: list[datetime] = []
    finished = Event()

    def failing(boundary: datetime) -> None:
        del boundary
        raise RuntimeError("boom")

    def healthy(boundary: datetime) -> None:
        healthy_calls.append(boundary)
        scheduler.stop()
        finished.set()

    scheduler = LocalScheduler(
        clock=clock,
        wait=AdvancingWaiter(clock),
        on_failure=reported.append,
    )
    scheduler.register_five_minute(failing)
    scheduler.register_five_minute(healthy)
    scheduler.start()

    assert finished.wait(1)
    scheduler.stop()
    assert healthy_calls == [datetime(2026, 7, 21, 10, 5, tzinfo=SHANGHAI)]
    assert len(reported) == 1
    assert reported[0].code == "CALLBACK_FAILED"
    assert reported[0].callback_name == "failing"
    assert scheduler.failures == tuple(reported)


def test_scheduler_registration_can_be_removed_before_dispatch() -> None:
    clock = FakeClock(datetime(2026, 7, 21, 10, 4, tzinfo=SHANGHAI))
    scheduler = LocalScheduler(clock=clock, wait=AdvancingWaiter(clock))
    removed_calls: list[datetime] = []
    kept_calls: list[datetime] = []
    finished = Event()

    removed_handle = scheduler.register_five_minute(removed_calls.append)
    scheduler.unregister_five_minute(removed_handle)

    def kept(boundary: datetime) -> None:
        kept_calls.append(boundary)
        scheduler.stop()
        finished.set()

    scheduler.register_five_minute(kept)
    scheduler.start()

    assert finished.wait(1)
    scheduler.stop()
    assert removed_calls == []
    assert kept_calls == [datetime(2026, 7, 21, 10, 5, tzinfo=SHANGHAI)]


def test_scheduler_rejects_duplicate_callback_until_it_is_unregistered() -> None:
    scheduler = LocalScheduler()

    def callback(boundary: datetime) -> None:
        del boundary

    handle = scheduler.register_five_minute(callback)
    try:
        scheduler.register_five_minute(callback)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate callback registration must fail")

    scheduler.unregister_five_minute(handle)
    scheduler.register_five_minute(callback)


def test_scheduler_rejects_duplicate_builtin_bound_method() -> None:
    scheduler = LocalScheduler()
    calls: list[datetime] = []

    scheduler.register_five_minute(calls.append)
    try:
        scheduler.register_five_minute(calls.append)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate built-in bound callback registration must fail")


def test_callback_unregistered_by_an_earlier_callback_is_skipped_in_same_boundary() -> None:
    clock = FakeClock(datetime(2026, 7, 21, 10, 4, tzinfo=SHANGHAI))
    scheduler = LocalScheduler(clock=clock, wait=AdvancingWaiter(clock))
    calls: list[str] = []
    finished = Event()
    second_handle = 0

    def first(boundary: datetime) -> None:
        del boundary
        calls.append("first")
        scheduler.unregister_five_minute(second_handle)

    def second(boundary: datetime) -> None:
        del boundary
        calls.append("second")

    def last(boundary: datetime) -> None:
        del boundary
        calls.append("last")
        scheduler.stop()
        finished.set()

    scheduler.register_five_minute(first)
    second_handle = scheduler.register_five_minute(second)
    scheduler.register_five_minute(last)
    scheduler.start()

    assert finished.wait(1)
    scheduler.stop()
    assert calls == ["first", "last"]


def test_scheduler_does_not_dispatch_after_stop_completes() -> None:
    entered = Event()
    callbacks: list[datetime] = []

    def blocked_until_stop(stop: Event, delay: float) -> bool:
        del delay
        entered.set()
        stop.wait()
        return True

    scheduler = LocalScheduler(
        clock=lambda: datetime(2026, 7, 21, 10, 4, tzinfo=SHANGHAI),
        wait=blocked_until_stop,
    )
    scheduler.register_five_minute(callbacks.append)
    scheduler.start()

    assert entered.wait(1)
    scheduler.stop()

    assert callbacks == []
    assert scheduler.running is False


def test_unregister_waits_until_an_already_claimed_callback_has_finished() -> None:
    clock = FakeClock(datetime(2026, 7, 21, 10, 4, tzinfo=SHANGHAI))
    scheduler = LocalScheduler(clock=clock, wait=AdvancingWaiter(clock))
    callback_entered = Event()
    release_callback = Event()
    unregister_started = Event()
    unregister_done = Event()

    def callback(boundary: datetime) -> None:
        del boundary
        callback_entered.set()
        release_callback.wait()
        scheduler.stop()

    handle = scheduler.register_five_minute(callback)
    scheduler.start()
    assert callback_entered.wait(1)

    def unregister() -> None:
        unregister_started.set()
        scheduler.unregister_five_minute(handle)
        unregister_done.set()

    worker = Thread(target=unregister)
    worker.start()
    assert unregister_started.wait(1)
    returned_while_callback_was_active = unregister_done.wait(0.1)
    release_callback.set()
    assert unregister_done.wait(1)
    worker.join()
    scheduler.stop()

    assert returned_while_callback_was_active is False


def test_start_waits_for_stopping_worker_then_returns_with_a_new_live_worker() -> None:
    clock = FakeClock(datetime(2026, 7, 21, 10, 4, tzinfo=SHANGHAI))
    first_boundary = True
    wait_lock = Lock()

    def one_boundary_then_block(stop: Event, delay: float) -> bool:
        nonlocal first_boundary
        with wait_lock:
            dispatch = first_boundary
            if dispatch:
                first_boundary = False
        if dispatch:
            clock.advance(delay)
            return False
        stop.wait()
        return True

    scheduler = LocalScheduler(clock=clock, wait=one_boundary_then_block)
    callback_entered = Event()
    release_callback = Event()
    stop_done = Event()
    restart_entered = Event()
    restart_done = Event()
    restart_observed_running: list[bool] = []

    def blocking_callback(boundary: datetime) -> None:
        del boundary
        callback_entered.set()
        release_callback.wait()

    handle = scheduler.register_five_minute(blocking_callback)
    scheduler.start()
    assert callback_entered.wait(1)

    def external_stop() -> None:
        scheduler.stop()
        stop_done.set()

    stopper = Thread(target=external_stop)
    stopper.start()
    for _ in range(100_000):
        if not scheduler.running:
            break
    assert scheduler.running is False

    def concurrent_restart() -> None:
        restart_entered.set()
        scheduler.start()
        restart_observed_running.append(scheduler.running)
        restart_done.set()

    restarter = Thread(target=concurrent_restart)
    restarter.start()
    assert restart_entered.wait(1)
    returned_before_old_worker_exit = restart_done.wait(0.1)

    release_callback.set()
    assert stop_done.wait(1)
    assert restart_done.wait(1)
    stopper.join()
    restarter.join()
    scheduler.unregister_five_minute(handle)
    try:
        assert returned_before_old_worker_exit is False
        assert restart_observed_running == [True]
    finally:
        scheduler.stop()


def test_scheduler_rejects_naive_clock_values_and_non_callable_callbacks() -> None:
    scheduler = LocalScheduler(clock=lambda: datetime(2026, 7, 21, 10, 4))

    try:
        scheduler.start()
        for _ in range(1000):
            if scheduler.failures:
                break
    finally:
        scheduler.stop()

    assert scheduler.failures[0].code == "SCHEDULER_CLOCK_FAILED"
    try:
        scheduler.register_five_minute(None)  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("non-callable registration must fail")
