from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal
from itertools import count
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic, sleep
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from sqlalchemy import func, select

from compass.config import Settings
from compass.backtest.engine import ExecutionTiming
from compass.data.base import DailyBarRequest
from compass.domain.market import InstrumentId
from compass.domain.trading import AccountSnapshot, Position
from compass.services.local_application import build_local_application
from compass.services.local_decision_gateway import SelectedDecisionStrategy
from compass.services.local_signal_center import AccountPositionInput
from compass.services.local_signal_center import SignalExecutionFillInput
from compass.services.task_manager import TaskOperationError, TaskStatus
from compass.storage.models import DatasetBundleRecord, DatasetManifestRecord
from compass.storage.signal_execution_repository import SignalExecutionStatus
from compass.strategies.base import StrategyFrequency
from compass.strategies.rule_dsl import DslVariable
from compass.ui.pages.data import DataSyncRange
from compass.ui.pages.settings import MarketProxyMode, MarketProxySetting
from compass.ui.pages.strategy_lab import (
    StrategyLabConfiguration,
    StrategyLabKind,
    StrategyLabInitialPosition,
    StrategyLegConfiguration,
    _signal_markers,
    _signal_rows,
)
from compass.ui.pages.strategies import StrategyFormModel
from compass.ui.pages.watchlists import WatchlistDraft


NOW = datetime(2026, 8, 9, 10, tzinfo=ZoneInfo("Asia/Shanghai"))
FIRST = InstrumentId.parse("SSE.510300")
SECOND = InstrumentId.parse("SZSE.159949")
INDEX = InstrumentId.parse("SSE.000300")


def _bars(day: date) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [4.0],
            "high": [4.2],
            "low": [3.9],
            "close": [4.1],
            "volume": [1000.0],
            "amount": [4100.0],
        },
        index=pd.DatetimeIndex([day.isoformat()], name="date"),
    )


class ConcurrentProvider:
    name = "akshare"

    def __init__(self) -> None:
        self.both_started = Event()
        self.release_second = Event()
        self._lock = Lock()
        self._active = 0
        self._started = 0
        self.max_active = 0

    def fetch_daily(self, request: DailyBarRequest) -> pd.DataFrame:
        with self._lock:
            self._active += 1
            self._started += 1
            self.max_active = max(self.max_active, self._active)
            if self._started == 2:
                self.both_started.set()
        try:
            assert self.both_started.wait(5)
            if request.instrument == SECOND:
                assert self.release_second.wait(5)
            frame = _bars(request.start)
            frame.attrs["instrument_name"] = str(request.instrument)
            return frame
        finally:
            with self._lock:
                self._active -= 1

    def fetch_snapshot(self, instruments: Sequence[InstrumentId]) -> pd.DataFrame:
        raise NotImplementedError

    def fetch_corporate_actions(self, request: DailyBarRequest) -> Sequence[Any]:
        return ()


class PartiallyFailingProvider:
    name = "akshare"

    def fetch_daily(self, request: DailyBarRequest) -> pd.DataFrame:
        frame = _bars(request.start)
        if request.instrument == SECOND:
            frame = frame.iloc[0:0]
        frame.attrs["instrument_name"] = str(request.instrument)
        return frame

    def fetch_snapshot(self, instruments: Sequence[InstrumentId]) -> pd.DataFrame:
        raise NotImplementedError

    def fetch_corporate_actions(self, request: DailyBarRequest) -> Sequence[Any]:
        return ()


class ImmediateProvider:
    name = "akshare"

    def fetch_daily(self, request: DailyBarRequest) -> pd.DataFrame:
        frame = _bars(request.start)
        frame.attrs["instrument_name"] = str(request.instrument)
        return frame

    def fetch_snapshot(self, instruments: Sequence[InstrumentId]) -> pd.DataFrame:
        raise NotImplementedError

    def fetch_corporate_actions(self, request: DailyBarRequest) -> Sequence[Any]:
        return ()


class RevaluingProvider(ImmediateProvider):
    def fetch_daily(self, request: DailyBarRequest) -> pd.DataFrame:
        frame = super().fetch_daily(request)
        if request.start >= date(2026, 8, 8):
            frame.loc[:, "close"] = 4.3
            frame.loc[:, "high"] = 4.4
        elif request.start >= date(2026, 8, 7):
            frame.loc[:, "close"] = 4.2
        return frame


class BacktestProvider:
    name = "akshare"

    def fetch_daily(self, request: DailyBarRequest) -> pd.DataFrame:
        index = pd.bdate_range(request.start, request.end, name="date")
        closes = [4.0 + position * 0.01 for position in range(len(index))]
        frame = pd.DataFrame(
            {
                "open": closes,
                "high": [item + 0.05 for item in closes],
                "low": [item - 0.05 for item in closes],
                "close": closes,
                "volume": [1_000_000.0] * len(index),
                "amount": [item * 1_000_000 for item in closes],
                "price_limit_rate": [0.10] * len(index),
            },
            index=index,
        )
        frame.attrs["instrument_name"] = "沪深300ETF"
        return frame

    def fetch_snapshot(self, instruments: Sequence[InstrumentId]) -> pd.DataFrame:
        raise NotImplementedError

    def fetch_corporate_actions(self, request: DailyBarRequest) -> Sequence[Any]:
        return ()


def _application(tmp_path: Path, provider: Any):  # type: ignore[no-untyped-def]
    sequence = count(1)
    settings = Settings.from_env(tmp_path)
    application = build_local_application(
        settings,
        providers=(provider,),
        clock=lambda: NOW,
        id_factory=lambda kind: f"{kind}-{next(sequence)}",
        expected_sessions=lambda request: pd.DatetimeIndex(
            [request.start.isoformat()], name="date"
        ),
        sync_window=lambda today: (date(2026, 8, 7), date(2026, 8, 7)),
    )
    application.watchlists.save_primary(WatchlistDraft("关注标的", (FIRST, SECOND)))
    return application


def test_partial_sync_keeps_every_successful_instrument(tmp_path: Path) -> None:
    application = _application(tmp_path, PartiallyFailingProvider())
    try:
        with pytest.raises(TaskOperationError):
            application.data_gateway.sync("akshare", date(2026, 8, 7), date(2026, 8, 7))

        previews = application.data_gateway.latest_market_previews()
        assert tuple(item.instrument for item in previews) == (FIRST,)
        page_state = application.models.data.state()
        assert page_state.watchlist_instruments == (FIRST, SECOND)
        assert page_state.missing_instruments == (SECOND,)
        history = application.data_gateway.sync_history(1, 10).entries
        assert len(history) == 1
        assert history[0].status is TaskStatus.FAILED
        assert history[0].instrument_count == 2
        assert history[0].completed_instrument_count == 1
        assert len(history[0].instrument_failures) == 1
        assert history[0].instrument_failures[0].instrument == SECOND
        assert history[0].instrument_failures[0].failure_code == "DATA_QUALITY_REJECTED"
        assert history[0].instrument_failures[0].quality_issue_codes == ("MISSING_SESSION",)
        assert history[0].instrument_failures[0].input_rows == 0
        assert history[0].instrument_failures[0].output_rows == 0
        with application.database.session_factory() as session:
            assert session.scalar(select(func.count()).select_from(DatasetBundleRecord)) == 1
            assert session.scalar(select(func.count()).select_from(DatasetManifestRecord)) == 1
    finally:
        application.shutdown()


def test_sync_calendar_uses_only_the_current_day_as_required_coverage(
    tmp_path: Path,
) -> None:
    application = _application(tmp_path, ImmediateProvider())
    requested: list[tuple[date, date]] = []
    application.data_gateway._refresh_calendar = (  # type: ignore[attr-defined]
        lambda start, end: requested.append((start, end))
    )
    application.data_gateway._latest_completed_session = (  # type: ignore[attr-defined]
        lambda moment: moment.date()
    )
    try:
        application.data_gateway.sync("akshare", date(2026, 8, 7), date(2026, 8, 7))

        assert requested == [(date(2026, 8, 7), NOW.date())]
    finally:
        application.shutdown()


def test_calendar_failure_has_a_specific_sync_error_before_market_requests(
    tmp_path: Path,
) -> None:
    application = _application(tmp_path, ImmediateProvider())

    def unavailable(start: date, end: date) -> None:
        del start, end
        raise RuntimeError("upstream calendar unavailable")

    application.data_gateway._refresh_calendar = unavailable  # type: ignore[attr-defined]
    try:
        with pytest.raises(TaskOperationError, match="SYNC_CALENDAR_UNAVAILABLE"):
            application.data_gateway.sync("akshare", date(2026, 8, 7), date(2026, 8, 7))

        assert application.data_gateway.latest_market_previews() == ()
        history = application.data_gateway.sync_history(1, 10).entries
        assert history[0].failure_code == "SYNC_CALENDAR_UNAVAILABLE"
        assert history[0].instrument_count is None
    finally:
        application.shutdown()


def test_selective_sync_preserves_other_current_instruments(tmp_path: Path) -> None:
    application = _application(tmp_path, ImmediateProvider())
    try:
        application.data_gateway.sync("akshare", date(2026, 8, 7), date(2026, 8, 7))
        before = {
            item.instrument: item.manifest_id
            for item in application.data_gateway.latest_market_previews()
        }

        application.data_gateway.sync(
            "akshare",
            date(2026, 8, 7),
            date(2026, 8, 7),
            instruments=(FIRST,),
        )

        after = {
            item.instrument: item.manifest_id
            for item in application.data_gateway.latest_market_previews()
        }
        assert set(after) == {FIRST, SECOND}
        assert after[SECOND] == before[SECOND]
        latest_history = application.data_gateway.sync_history(1, 10).entries[0]
        assert latest_history.instrument_count == 1
        assert latest_history.completed_instrument_count == 1
        with application.database.session_factory() as session:
            assert session.scalar(select(func.count()).select_from(DatasetBundleRecord)) == 1
            assert session.scalar(select(func.count()).select_from(DatasetManifestRecord)) == 2
    finally:
        application.shutdown()


def test_saved_strategy_can_be_copied_and_deleted(tmp_path: Path) -> None:
    application = _application(tmp_path, ImmediateProvider())
    try:
        pool = application.models.strategies.state().pools[0]
        parameters = application.models.strategies.parameters_from_json(
            "buy_and_hold", '{"target_weight":"1"}'
        )
        created = application.models.strategies.create(
            StrategyFormModel(
                name="长期持有",
                strategy_type="buy_and_hold",
                watchlist_id=pool.watchlist_id,
                frequency=StrategyFrequency.DAILY,
                parameters=parameters,
            )
        ).instance
        assert created is not None
        templates = application.strategy_lab_gateway.templates()
        assert tuple(item.instance_id for item in templates) == (created.instance_id,)
        assert templates[0].instruments == (FIRST, SECOND)
        assert templates[0].parameters["target_weight"] == "1"
        copied = application.models.strategies.copy(created.instance_id)

        assert application.models.strategies.delete(copied.instance_id) is True
        assert tuple(
            item.instance_id for item in application.models.strategies.state().instances
        ) == (created.instance_id,)
        assert application.models.strategies.delete(created.instance_id) is True
        assert application.models.strategies.state().instances == ()
    finally:
        application.shutdown()


def test_strategy_lab_runs_two_strategies_across_multiple_etfs(tmp_path: Path) -> None:
    sequence = count(1)
    settings = Settings.from_env(tmp_path)
    application = build_local_application(
        settings,
        providers=(BacktestProvider(),),
        clock=lambda: NOW,
        id_factory=lambda kind: f"{kind}-{next(sequence)}",
        expected_sessions=lambda request: pd.bdate_range(request.start, request.end, name="date"),
        sync_window=lambda today: (date(2026, 1, 2), date(2026, 8, 7)),
    )
    application.watchlists.save_primary(WatchlistDraft("关注标的", (INDEX, FIRST, SECOND)))
    try:
        strategy_state = application.models.strategies.state()
        assert tuple(item.strategy_type for item in strategy_state.definitions) == (
            "buy_and_hold",
            "cross_sectional_momentum",
            "dual_ma",
            "etf_rotation",
            "kronos_forecast",
            "mean_reversion",
            "rule_dsl",
        )
        assert strategy_state.pools[0].instruments == (FIRST, SECOND)
        application.data_gateway.sync("akshare", date(2026, 1, 2), date(2026, 8, 7))
        configuration = StrategyLabConfiguration(
            strategies=(
                StrategyLegConfiguration(
                    strategy_id="strategy-hold",
                    strategy=StrategyLabKind.BUY_AND_HOLD,
                    instruments=(FIRST, SECOND),
                    budget=Decimal("0.5"),
                ),
                StrategyLegConfiguration(
                    strategy_id="strategy-trend",
                    strategy=StrategyLabKind.DUAL_MA,
                    signal_instrument=INDEX,
                    instruments=(FIRST, SECOND),
                    budget=Decimal("0.5"),
                    short_window=20,
                    long_window=60,
                    confirmation_days=1,
                ),
            ),
            benchmark=INDEX,
            start=date(2026, 1, 2),
            end=date(2026, 8, 7),
            initial_cash=Decimal("1000000.00"),
            commission_rate=Decimal("0.0003"),
            minimum_commission=Decimal("5"),
            slippage_bps=Decimal("2"),
            execution_timing=ExecutionTiming.NEXT_CLOSE,
            initial_cash_weight=Decimal("0.5"),
            initial_positions=(
                StrategyLabInitialPosition(FIRST, Decimal("0.25")),
                StrategyLabInitialPosition(SECOND, Decimal("0.25")),
            ),
        )

        application.strategy_lab_gateway.run("backtest-test", configuration)

        report = application.strategy_lab_gateway.report("backtest-test")
        assert report is not None
        assert report.snapshot.market_rule_configuration["execution"] == "next_close"
        assert report.snapshot.instrument_pool["instruments"] == (
            str(INDEX),
            str(FIRST),
            str(SECOND),
        )
        assert report.snapshot.instrument_pool["signal_instruments"] == (str(INDEX),)
        assert report.snapshot.instrument_pool["trade_instruments"] == (
            str(FIRST),
            str(SECOND),
        )
        assert report.snapshot.instrument_pool["initial_cash_weight"] == Decimal("0.5")
        assert {item.instrument for item in report.result.ledger[0].positions} == {
            FIRST,
            SECOND,
        }
        assert report.snapshot.allocator_configuration["kind"] == ("deterministic_multi_strategy")
        assert tuple(item.sleeve_id for item in report.strategies) == (
            "strategy-hold",
            "strategy-trend",
        )
        assert report.result.fills, tuple(
            (order.status, order.cancellation_reason) for order in report.result.orders
        )
        assert {fill.instrument for fill in report.result.fills} == {FIRST, SECOND}
        assert len(report.result.fills) < 20
        signal_rows = _signal_rows(report)
        assert len(signal_rows) == len(report.result.orders)
        assert {row["status"] for row in signal_rows}
        buy_signals, sell_signals, _ = _signal_markers(report)
        assert buy_signals or sell_signals
        assert any(
            set(order.sleeve_weights) == {"strategy-hold", "strategy-trend"}
            for order in report.result.orders
        )
        assert report.metrics.total_return is not None
        assert application.strategy_lab_gateway.latest_report() == report
        history = application.strategy_lab_gateway.history()
        assert tuple(item.run_id for item in history) == ("backtest-test",)
        assert history[0].strategy_count == 2
        assert history[0].target_count == 2
        compared = application.strategy_lab_gateway.compare_report("backtest-test", SECOND)
        assert compared.run_id == report.run_id
        assert compared.benchmark_curve
        assert application.strategy_lab_gateway.report("backtest-test") == report
        assert application.strategy_lab_gateway.delete_report("backtest-test") is True
        assert application.strategy_lab_gateway.latest_report() is None
    finally:
        application.shutdown()


def test_strategy_lab_runs_custom_rule_dsl_with_exported_variables(
    tmp_path: Path,
) -> None:
    sequence = count(1)
    application = build_local_application(
        Settings.from_env(tmp_path),
        providers=(BacktestProvider(),),
        clock=lambda: NOW,
        id_factory=lambda kind: f"{kind}-{next(sequence)}",
        expected_sessions=lambda request: pd.bdate_range(request.start, request.end, name="date"),
        sync_window=lambda today: (date(2026, 1, 2), date(2026, 8, 7)),
    )
    application.watchlists.save_primary(WatchlistDraft("关注标的", (INDEX, FIRST)))
    try:
        application.data_gateway.sync("akshare", date(2026, 1, 2), date(2026, 8, 7))
        rule = StrategyLegConfiguration(
            strategy_id="custom-rule",
            strategy=StrategyLabKind.RULE_DSL,
            signal_instrument=INDEX,
            instruments=(FIRST,),
            budget=Decimal("1"),
            buy_expression="close > sma(close, window)",
            sell_expression="close < sma(close, window)",
            variables=(
                DslVariable(
                    name="window",
                    value=Decimal("2"),
                    minimum=Decimal("2"),
                    maximum=Decimal("20"),
                    step=Decimal("2"),
                ),
            ),
        )
        configuration = StrategyLabConfiguration(
            strategies=(rule,),
            benchmark=INDEX,
            start=date(2026, 1, 2),
            end=date(2026, 8, 7),
            initial_cash=Decimal("1000000.00"),
            commission_rate=Decimal("0.0003"),
            minimum_commission=Decimal("5"),
            slippage_bps=Decimal("2"),
            execution_timing=ExecutionTiming.NEXT_OPEN,
        )

        application.strategy_lab_gateway.run("backtest-dsl", configuration)

        report = application.strategy_lab_gateway.report("backtest-dsl")
        assert report is not None
        assert report.result.fills
        assert report.strategies[0].strategy_type == "rule_dsl"
        assert report.strategies[0].parameters["buy_expression"] == ("close > sma(close, window)")
        assert report.strategies[0].parameters["variables"] == (
            {
                "maximum": "20",
                "minimum": "2",
                "name": "window",
                "optimize": True,
                "step": "2",
                "value": "2",
            },
        )
    finally:
        application.shutdown()


def test_sync_uses_fixed_concurrency_and_persists_live_progress(
    tmp_path: Path,
) -> None:
    provider = ConcurrentProvider()
    application = _application(tmp_path, provider)
    failures: list[BaseException] = []

    def run() -> None:
        try:
            application.data_gateway.sync("akshare", date(2026, 8, 7), date(2026, 8, 7))
        except BaseException as error:
            failures.append(error)

    worker = Thread(target=run)
    worker.start()
    try:
        assert provider.both_started.wait(5)
        deadline = monotonic() + 5
        running = None
        while monotonic() < deadline:
            running = application.data_gateway.sync_history(1, 10).entries[0]
            if running.completed_instrument_count == 1:
                break
            sleep(0.01)
        assert running is not None
        assert running.status is TaskStatus.RUNNING
        assert running.instrument_count == 2
        assert running.completed_instrument_count == 1
        assert provider.max_active == 2
    finally:
        provider.release_second.set()
        worker.join(5)
        application.shutdown()

    assert not worker.is_alive()
    assert failures == []


def test_running_sync_cannot_be_cleared_and_can_be_stopped_safely(
    tmp_path: Path,
) -> None:
    provider = ConcurrentProvider()
    application = _application(tmp_path, provider)
    task = application.models.data.start_sync(
        "akshare",
        DataSyncRange(date(2026, 8, 7), date(2026, 8, 7)),
    )
    try:
        assert provider.both_started.wait(5)
        assert application.data_gateway.clear_sync_history() == 0
        running = application.data_gateway.sync_history(1, 10)
        assert running.total_items == 1
        assert running.entries[0].status is TaskStatus.RUNNING

        stopping = application.models.data.stop_sync()
        assert stopping.task_id == task.task_id
        assert stopping.status is TaskStatus.CANCELLATION_REQUESTED
        provider.release_second.set()

        finished = application.task_manager.wait(task.task_id, timeout=5)
        assert finished.status is TaskStatus.CANCELLED
        history = application.data_gateway.sync_history(1, 10)
        assert history.total_items == 1
        assert history.entries[0].status is TaskStatus.CANCELLED
        assert history.entries[0].completed_at is not None
        assert history.entries[0].failure_code is None
    finally:
        provider.release_second.set()
        application.shutdown()


def test_stale_selective_sync_target_is_rejected_before_task_submission(
    tmp_path: Path,
) -> None:
    application = _application(tmp_path, ImmediateProvider())
    application.watchlists.save_primary(WatchlistDraft("关注标的", (FIRST,)))
    try:
        with pytest.raises(ValueError, match="SYNC_TARGET_INVALID"):
            application.models.data.start_sync(
                "akshare",
                DataSyncRange(date(2026, 8, 7), date(2026, 8, 7)),
                (SECOND,),
            )

        assert application.models.data.state().active_sync is None
        assert application.data_gateway.sync_history(1, 10).total_items == 0
    finally:
        application.shutdown()


def test_failed_sync_is_released_and_does_not_block_the_next_task(tmp_path: Path) -> None:
    application = _application(tmp_path, PartiallyFailingProvider())
    try:
        first = application.models.data.start_sync(
            "akshare",
            DataSyncRange(date(2026, 8, 7), date(2026, 8, 7)),
        )
        failed = application.task_manager.wait(first.task_id, timeout=5)
        assert failed.status is TaskStatus.FAILED

        assert application.models.data.state().active_sync is None
        with pytest.raises(ValueError, match="DATA_SYNC_NOT_ACTIVE"):
            application.models.data.stop_sync()

        second = application.models.data.start_sync(
            "akshare",
            DataSyncRange(date(2026, 8, 7), date(2026, 8, 7)),
            (FIRST,),
        )
        assert second.task_id != first.task_id
        assert (
            application.task_manager.wait(second.task_id, timeout=5).status is TaskStatus.SUCCEEDED
        )
    finally:
        application.shutdown()


def test_market_proxy_setting_is_persisted_and_applied(tmp_path: Path) -> None:
    provider = ConcurrentProvider()
    application = _application(tmp_path, provider)
    try:
        setting = MarketProxySetting(MarketProxyMode.CUSTOM, "127.0.0.1", 7897)
        application.settings_gateway.set_market_proxy(setting)
        assert application.settings_gateway.state().market_proxy == setting

        application.settings_gateway.set_market_proxy(MarketProxySetting(MarketProxyMode.SYSTEM))
        assert application.settings_gateway.state().market_proxy.mode is MarketProxyMode.SYSTEM

        assert application.settings_gateway.state().market_request_timeout_seconds == 5
        application.settings_gateway.set_market_request_timeout(12)
        assert application.settings_gateway.state().market_request_timeout_seconds == 12
        application.settings_gateway.set_automatic_sync(True, 60, True)
        assert application.settings_gateway.state().automatic_sync_on_startup is True
        assert application.settings_gateway.state().automatic_sync_interval_minutes == 60
        assert application.settings_gateway.state().automatic_sync_after_close is True
    finally:
        application.shutdown()

    restarted = _application(tmp_path, ConcurrentProvider())
    try:
        assert restarted.settings_gateway.state().automatic_sync_on_startup is True
        assert restarted.settings_gateway.state().automatic_sync_interval_minutes == 60
        assert restarted.settings_gateway.state().automatic_sync_after_close is True
    finally:
        restarted.shutdown()


def test_history_pagination_and_market_data_cleanup_are_independent(
    tmp_path: Path,
) -> None:
    provider = ConcurrentProvider()
    provider.release_second.set()
    application = _application(tmp_path, provider)
    try:
        for _ in range(6):
            application.data_gateway.sync("akshare", date(2026, 8, 7), date(2026, 8, 7))

        first_page = application.data_gateway.sync_history(1, 5)
        second_page = application.data_gateway.sync_history(2, 5)
        assert first_page.total_items == 6
        assert first_page.total_pages == 2
        assert len(first_page.entries) == 5
        assert len(second_page.entries) == 1
        assert all(
            entry.instrument_count == entry.completed_instrument_count == 2
            for entry in first_page.entries + second_page.entries
        )
        with application.database.session_factory() as session:
            assert session.scalar(select(func.count()).select_from(DatasetBundleRecord)) == 1
            assert session.scalar(select(func.count()).select_from(DatasetManifestRecord)) == 2

        assert {item.instrument for item in application.data_gateway.latest_market_previews()} == {
            FIRST,
            SECOND,
        }
        assert application.data_gateway.clear_sync_history() == 6
        assert application.data_gateway.sync_history(1, 5).total_items == 0
        assert len(application.data_gateway.latest_market_previews()) == 2

        assert application.data_gateway.clear_market_data(FIRST) == 1
        assert tuple(
            item.instrument for item in application.data_gateway.latest_market_previews()
        ) == (SECOND,)
        assert application.data_gateway.clear_market_data() == 1
        assert application.data_gateway.latest_bundle() is None
        assert application.data_gateway.latest_market_previews() == ()
    finally:
        application.shutdown()


def test_startup_prunes_legacy_versions_but_keeps_current_bundle(
    tmp_path: Path,
) -> None:
    provider = ConcurrentProvider()
    provider.release_second.set()
    application = _application(tmp_path, provider)
    application.data_gateway.sync("akshare", date(2026, 8, 7), date(2026, 8, 7))
    current_ids = {
        preview.manifest_id for preview in application.data_gateway.latest_market_previews()
    }
    superseded = application.market_store.write_daily(
        str(FIRST),
        _bars(date(2026, 8, 7)),
        "akshare",
    )
    assert superseded.manifest_id not in current_ids
    assert application.market_store.manifest_path(superseded.manifest_id).exists()
    application.shutdown()

    restarted_provider = ConcurrentProvider()
    restarted_provider.release_second.set()
    restarted = _application(tmp_path, restarted_provider)
    try:
        assert {
            preview.manifest_id for preview in restarted.data_gateway.latest_market_previews()
        } == current_ids
        with restarted.database.session_factory() as session:
            assert session.scalar(select(func.count()).select_from(DatasetBundleRecord)) == 1
            assert session.scalar(select(func.count()).select_from(DatasetManifestRecord)) == 2
        assert not restarted.market_store.manifest_path(superseded.manifest_id).exists()
    finally:
        restarted.shutdown()


def test_sync_preserves_signal_snapshot_manifests_and_quarantines_missing_history(
    tmp_path: Path,
) -> None:
    application = _application(tmp_path, RevaluingProvider())
    try:
        account = application.accounts.save(
            AccountSnapshot(
                date(2026, 8, 6),
                Decimal("100000.00"),
                (
                    Position(
                        FIRST,
                        100,
                        100,
                        Decimal("4.00"),
                        Decimal("4.10"),
                    ),
                ),
            )
        )
        application.data_gateway.sync("akshare", date(2026, 8, 7), date(2026, 8, 7))
        pool = application.models.strategies.state().pools[0]
        parameters = application.models.strategies.parameters_from_json(
            "dual_ma",
            '{"short_window":2,"long_window":3,"confirmation_days":1,"target_weight":"1"}',
        )
        strategy = application.models.strategies.create(
            StrategyFormModel(
                name="双均线",
                strategy_type="dual_ma",
                watchlist_id=pool.watchlist_id,
                frequency=StrategyFrequency.DAILY,
                parameters=parameters,
            )
        ).instance
        application.decisions._calendar_coverage = None  # type: ignore[attr-defined]
        application.decisions._trading_calendar = (  # type: ignore[attr-defined]
            lambda day: day.weekday() < 5
        )
        decision = application.signal_center.generate(
            (SelectedDecisionStrategy(strategy.instance_id, Decimal("0.80")),),
            cash_reserve=Decimal("0.10"),
            minimum_trade_amount=Decimal("5000"),
        )
        assert decision.result.decision_equity != account.snapshot.equity
        referenced_ids = {reference.manifest_id for reference in decision.snapshot.market_manifests}

        current_bundle = application.data_gateway._bundles.latest()  # type: ignore[attr-defined]
        assert current_bundle is not None
        source_manifests = {
            InstrumentId.parse(manifest.instrument): manifest
            for manifest in (
                application.market_store.load_manifest(reference.manifest_id)
                for reference in current_bundle.market_manifests
            )
        }
        replacements = []
        for instrument in (FIRST, SECOND):
            frame = _bars(date(2026, 8, 8))
            frame.loc[:, "close"] = 4.3
            frame.loc[:, "high"] = 4.4
            source = source_manifests[instrument]
            replacements.append(
                (
                    instrument,
                    application.market_store.write_daily(
                        str(instrument),
                        frame,
                        "akshare",
                        quality_report_json=source.quality_report_json,
                        provenance_json=source.provenance_json,
                        instrument_name=source.instrument_name,
                    ),
                )
            )
        current = application.data_gateway._bundles.save(  # type: ignore[attr-defined]
            tuple(replacements),
            mode="strict",
            issue_codes=(),
            replace_current=True,
        )
        application.data_gateway._compact_market_versions(current)  # type: ignore[attr-defined]

        assert application.decisions.get(decision.decision_id) == decision
        assert all(
            application.market_store.manifest_path(manifest_id).is_file()
            for manifest_id in referenced_ids
        )

        current_ids = tuple(item.manifest_id for item in current.market_manifests)
        application.market_store.prune_superseded(current_ids)

        readable, invalid_count = application.decisions.readable_history()
        assert readable == ()
        assert invalid_count == 1
        signal_state = application.models.signals.state()
        assert signal_state.latest_decision is None
        assert signal_state.decision_history == ()
        assert signal_state.invalid_decision_count == 1
        assert application.signal_center.clear_invalid_decisions() == 1
        assert application.decisions.readable_history() == ((), 0)
    finally:
        application.shutdown()


def test_signal_center_isolates_accounts_settings_snapshots_and_decisions(
    tmp_path: Path,
) -> None:
    application = _application(tmp_path, RevaluingProvider())
    try:
        application.data_gateway.sync("akshare", date(2026, 8, 7), date(2026, 8, 7))
        pool = application.models.strategies.state().pools[0]
        strategy = application.models.strategies.create(
            StrategyFormModel(
                name="账户隔离策略",
                strategy_type="dual_ma",
                watchlist_id=pool.watchlist_id,
                frequency=StrategyFrequency.DAILY,
                parameters=application.models.strategies.parameters_from_json(
                    "dual_ma",
                    '{"short_window":2,"long_window":3,"confirmation_days":1,"target_weight":"1"}',
                ),
            )
        ).instance
        application.decisions._calendar_coverage = None  # type: ignore[attr-defined]
        application.decisions._trading_calendar = (  # type: ignore[attr-defined]
            lambda day: day.weekday() < 5
        )

        main_account = application.signal_center.save_account("100000.00", ())
        main_decision = application.signal_center.generate(
            (SelectedDecisionStrategy(strategy.instance_id, Decimal("0.60")),),
            cash_reserve=Decimal("0.20"),
            minimum_trade_amount=Decimal("2000"),
        )

        second_profile = application.signal_center.create_account("稳健账户")
        assert application.signal_center.latest_account() is None
        second_account = application.signal_center.save_account(
            "200000.00",
            (AccountPositionInput(str(FIRST), 100, 100, "4.00"),),
        )
        second_decision = application.signal_center.generate(
            (SelectedDecisionStrategy(strategy.instance_id, Decimal("0.50")),),
            cash_reserve=Decimal("0.30"),
            minimum_trade_amount=Decimal("3000"),
        )

        assert second_account.snapshot.cash == Decimal("200000.00")
        assert second_decision.result.account_id == second_profile.account_id
        assert application.signal_center.decision_history() == (second_decision,)
        assert application.signal_center.active_account_profile().cash_reserve == Decimal("0.30")

        application.signal_center.select_account("main")

        assert application.signal_center.latest_account() == main_account
        assert application.signal_center.decision_history() == (main_decision,)
        assert application.signal_center.decision(second_decision.decision_id) is None
        assert application.signal_center.active_account_profile().cash_reserve == Decimal("0.20")

        assert application.signal_center.clear_decisions() == 1
        assert application.signal_center.decision_history() == ()
        application.signal_center.select_account(second_profile.account_id)
        assert application.signal_center.decision_history() == (second_decision,)
    finally:
        application.shutdown()


def test_signal_decision_comparison_and_cleanup_use_the_frozen_account(
    tmp_path: Path,
) -> None:
    sequence = count(1)
    comparison_now = datetime(2026, 8, 12, 10, tzinfo=ZoneInfo("Asia/Shanghai"))
    application = build_local_application(
        Settings.from_env(tmp_path),
        providers=(BacktestProvider(),),
        clock=lambda: comparison_now,
        id_factory=lambda kind: f"{kind}-{next(sequence)}",
        expected_sessions=lambda request: pd.bdate_range(request.start, request.end, name="date"),
        sync_window=lambda today: (date(2026, 8, 3), date(2026, 8, 7)),
    )
    application.watchlists.save_primary(WatchlistDraft("关注标的", (FIRST, SECOND)))
    try:
        application.data_gateway.sync("akshare", date(2026, 8, 3), date(2026, 8, 7))
        pool = application.models.strategies.state().pools[0]
        strategy = application.models.strategies.create(
            StrategyFormModel(
                name="建议对照策略",
                strategy_type="dual_ma",
                watchlist_id=pool.watchlist_id,
                frequency=StrategyFrequency.DAILY,
                parameters=application.models.strategies.parameters_from_json(
                    "dual_ma",
                    '{"short_window":2,"long_window":3,"confirmation_days":1,"target_weight":"1"}',
                ),
            )
        ).instance
        application.decisions._calendar_coverage = None  # type: ignore[attr-defined]
        application.decisions._trading_calendar = (  # type: ignore[attr-defined]
            lambda day: day.weekday() < 5
        )
        application.signal_center.save_account(
            "100000.00",
            (AccountPositionInput(str(FIRST), 500, 500, "4.00"),),
        )
        decision = application.signal_center.generate(
            (SelectedDecisionStrategy(strategy.instance_id, Decimal("0.60")),),
            cash_reserve=Decimal("0.20"),
            minimum_trade_amount=Decimal("2000"),
        )
        application.signal_center.record_execution(
            decision.decision_id,
            SignalExecutionStatus.IGNORED,
            (),
            fees="0.00",
            recorded_at=comparison_now,
        )
        application.data_gateway.sync("akshare", date(2026, 8, 10), date(2026, 8, 11))

        comparison = application.signal_center.compare_decision(decision.decision_id)

        assert comparison.decision_id == decision.decision_id
        assert comparison.points[0].day == decision.result.decision_date
        assert comparison.points[-1].day == date(2026, 8, 11)
        assert comparison.points[0].ignored_equity == decision.result.decision_equity
        assert comparison.adopted_return == (
            comparison.points[-1].adopted_equity / decision.result.decision_equity - 1
        ).quantize(Decimal("0.0001"))

        assert application.signal_center.delete_decision(decision.decision_id) is True
        assert application.signal_center.decision(decision.decision_id) is None
        assert application.signal_center.execution(decision.decision_id) is None
    finally:
        application.shutdown()


def test_signal_center_shares_holdings_but_keeps_strategy_decisions_separate(
    tmp_path: Path,
) -> None:
    application = _application(tmp_path, RevaluingProvider())
    try:
        application.data_gateway.sync("akshare", date(2026, 8, 7), date(2026, 8, 7))
        pool = application.models.strategies.state().pools[0]
        strategy = application.models.strategies.create(
            StrategyFormModel(
                name="共享持仓策略",
                strategy_type="dual_ma",
                watchlist_id=pool.watchlist_id,
                frequency=StrategyFrequency.DAILY,
                parameters=application.models.strategies.parameters_from_json(
                    "dual_ma",
                    '{"short_window":2,"long_window":3,"confirmation_days":1,"target_weight":"1"}',
                ),
            )
        ).instance
        application.decisions._calendar_coverage = None  # type: ignore[attr-defined]
        application.decisions._trading_calendar = (  # type: ignore[attr-defined]
            lambda day: day.weekday() < 5
        )
        main_account = application.signal_center.save_account("100000.00", ())
        main_decision = application.signal_center.generate(
            (SelectedDecisionStrategy(strategy.instance_id, Decimal("0.60")),),
            cash_reserve=Decimal("0.20"),
            minimum_trade_amount=Decimal("2000"),
        )
        shared = application.signal_center.create_account("共享方案", "main")

        assert shared.holdings_account_id == "main"
        assert application.signal_center.latest_account() == main_account

        updated = application.signal_center.save_account(
            "90000.00",
            (AccountPositionInput(str(FIRST), 100, 100, "4.00"),),
        )
        shared_decision = application.signal_center.generate(
            (SelectedDecisionStrategy(strategy.instance_id, Decimal("0.50")),),
            cash_reserve=Decimal("0.30"),
            minimum_trade_amount=Decimal("3000"),
        )
        assert application.signal_center.decision_history() == (shared_decision,)
        application.signal_center.select_account("main")

        assert application.signal_center.latest_account() == updated
        assert len(application.signal_center.account_history()) == 2
        assert application.signal_center.decision_history() == (main_decision,)
        assert application.signal_center.decision(shared_decision.decision_id) is None
    finally:
        application.shutdown()


def test_signal_execution_updates_holdings_and_marks_prior_decision_stale(
    tmp_path: Path,
) -> None:
    sequence = count(1)
    application = build_local_application(
        Settings.from_env(tmp_path),
        providers=(BacktestProvider(),),
        clock=lambda: NOW,
        id_factory=lambda kind: f"{kind}-{next(sequence)}",
        expected_sessions=lambda request: pd.bdate_range(request.start, request.end, name="date"),
        sync_window=lambda today: (date(2026, 8, 3), date(2026, 8, 7)),
    )
    application.watchlists.save_primary(WatchlistDraft("关注标的", (FIRST, SECOND)))
    try:
        application.data_gateway.sync("akshare", date(2026, 8, 3), date(2026, 8, 7))
        pool = application.models.strategies.state().pools[0]
        strategy = application.models.strategies.create(
            StrategyFormModel(
                name="执行闭环策略",
                strategy_type="dual_ma",
                watchlist_id=pool.watchlist_id,
                frequency=StrategyFrequency.DAILY,
                parameters=application.models.strategies.parameters_from_json(
                    "dual_ma",
                    '{"short_window":2,"long_window":3,"confirmation_days":1,"target_weight":"1"}',
                ),
            )
        ).instance
        application.decisions._calendar_coverage = None  # type: ignore[attr-defined]
        application.decisions._trading_calendar = (  # type: ignore[attr-defined]
            lambda day: day.weekday() < 5
        )
        original = application.signal_center.save_account(
            "100000.00",
            (AccountPositionInput(str(FIRST), 500, 500, "4.00"),),
        )
        decision = application.signal_center.generate(
            (SelectedDecisionStrategy(strategy.instance_id, Decimal("0.60")),),
            cash_reserve=Decimal("0.20"),
            minimum_trade_amount=Decimal("2000"),
        )
        recommendation = next(
            item for item in decision.result.recommendations if item.quantity_delta != 0
        )

        execution = application.signal_center.record_execution(
            decision.decision_id,
            SignalExecutionStatus.PARTIAL,
            (
                SignalExecutionFillInput(
                    str(recommendation.instrument),
                    100 if recommendation.quantity_delta > 0 else -100,
                    recommendation.reference_price,
                ),
            ),
            fees="5.00",
            recorded_at=NOW + timedelta(minutes=1),
        )

        latest = application.signal_center.latest_account()
        assert latest is not None and latest.row_id != original.row_id
        assert execution.resulting_snapshot_row_id == latest.row_id
        assert application.signal_center.execution(decision.decision_id) == execution
        with pytest.raises(
            ValueError,
            match="SIGNAL_DECISION_ADOPTED_DELETE_FORBIDDEN",
        ):
            application.signal_center.delete_decision(decision.decision_id)
        assert application.signal_center.clear_decisions() == 0
        assert application.signal_center.decision(decision.decision_id) == decision
        assert application.signal_center.decision_freshness(decision).reasons == (
            "HOLDINGS_CHANGED",
        )
        with pytest.raises(ValueError, match="SIGNAL_EXECUTION_ALREADY_RECORDED"):
            application.signal_center.record_execution(
                decision.decision_id,
                SignalExecutionStatus.IGNORED,
                (),
                fees="0.00",
                recorded_at=NOW + timedelta(minutes=2),
            )

        new_decision = application.signal_center.generate(
            (SelectedDecisionStrategy(strategy.instance_id, Decimal("0.60")),),
            cash_reserve=Decimal("0.20"),
            minimum_trade_amount=Decimal("2000"),
        )
        new_recommendation = next(
            item for item in new_decision.result.recommendations if item.quantity_delta != 0
        )
        with pytest.raises(ValueError, match="SIGNAL_EXECUTION_INCOMPLETE"):
            application.signal_center.record_execution(
                new_decision.decision_id,
                SignalExecutionStatus.EXECUTED,
                (
                    SignalExecutionFillInput(
                        str(new_recommendation.instrument),
                        100 if new_recommendation.quantity_delta > 0 else -100,
                        new_recommendation.reference_price,
                    ),
                ),
                fees="0.00",
                recorded_at=NOW + timedelta(minutes=3),
            )
    finally:
        application.shutdown()
