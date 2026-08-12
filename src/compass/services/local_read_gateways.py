from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
import json
from typing import cast

import pandas as pd  # type: ignore[import-untyped]
from sqlalchemy import delete, func, select

from compass.backtest.engine import (
    BacktestEngine,
    BacktestRequest,
    DecisionTarget,
)
from compass.backtest.snapshot import RunSnapshot, StrategySnapshot
from compass.data.base import DailyBarRequest, MarketDataProvider, ProviderError
from compass.data.quality import DataQualityError, QualityMode
from compass.domain.market import InstrumentId
from compass.services.data_service import DataService, ExpectedSessions, SyncResult
from compass.services.dataset_provenance import (
    snapshot_data_quality,
    validate_dataset_provenance,
)
from compass.services.export_service import (
    DecisionExportRecord,
    is_reserved_decision_snapshot_run_id,
)
from compass.services.intraday_service import IntradayState
from compass.services.instrument_names import common_instrument_name
from compass.storage.account_repository import AccountRepository
from compass.storage.backtest_report_repository import BacktestReportRepository
from compass.storage.dataset_bundle_repository import (
    DatasetBundle,
    DatasetBundleRepository,
)
from compass.storage.decision_export_repository import DecisionExportRepository
from compass.storage.models import DatasetManifestRecord, TaskRun
from compass.storage.market_store import DatasetManifest
from compass.services.local_crud_gateways import (
    LocalSettingsGateway,
    LocalStrategyGateway,
    LocalWatchlistGateway,
)
from compass.services.local_market_configuration import (
    local_instruments,
    local_profile_ids,
    local_risk_engine,
    local_rule_book,
)
from compass.services.task_manager import TaskOperationError, TaskStatus
from compass.strategies.base import (
    Strategy,
    StrategyContext,
    StrategyDecisionStatus,
)
from compass.strategies.registry import StrategyRegistry
from compass.ui.components.charts import CurvePoint, MarketBarPoint
from compass.ui.pages.backtests import BacktestReport, BacktestSubmission
from compass.ui.pages.dashboard import (
    DashboardDataHealth,
    DashboardFailure,
    DashboardSnapshot,
)
from compass.ui.pages.data import (
    DataSyncHistoryPage,
    DataSyncHistoryEntry,
    DataSyncInstrumentFailure,
    DataSyncRange,
    DataSourceSnapshot,
    MarketDataPreview,
    QualitySummary,
)
from compass.ui.pages.strategies import strategy_parameters_json


@dataclass(frozen=True, slots=True)
class _SyncStatistics:
    instrument_count: int
    downloaded_rows: int
    reused_rows: int
    remaining_requested_sessions: int

    def __post_init__(self) -> None:
        if type(self.instrument_count) is not int or self.instrument_count <= 0:
            raise ValueError("sync instrument count must be a positive exact integer")
        for label, value in (
            ("downloaded_rows", self.downloaded_rows),
            ("reused_rows", self.reused_rows),
            ("remaining_requested_sessions", self.remaining_requested_sessions),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"sync {label} must be a non-negative exact integer")


@dataclass(frozen=True, slots=True)
class _SyncProgress:
    instrument_count: int
    completed_instrument_count: int

    def __post_init__(self) -> None:
        if type(self.instrument_count) is not int or self.instrument_count <= 0:
            raise ValueError("sync progress total must be a positive exact integer")
        if (
            type(self.completed_instrument_count) is not int
            or not 0 <= self.completed_instrument_count <= self.instrument_count
        ):
            raise ValueError("sync progress completed count is invalid")


class _SyncCancelled(RuntimeError):
    """Internal cooperative cancellation signal for a market sync."""


class _PartialSyncFailure(RuntimeError):
    def __init__(
        self,
        cause: Exception,
        failures: tuple[DataSyncInstrumentFailure, ...],
    ) -> None:
        self.cause = cause
        self.failures = failures
        super().__init__(str(cause))


class LocalDataGateway:
    """Synchronize enabled pools only after an explicit page action."""

    def __init__(
        self,
        sources: Sequence[DataSourceSnapshot],
        *,
        providers: Sequence[MarketDataProvider],
        service: DataService,
        bundles: DatasetBundleRepository,
        watchlists: LocalWatchlistGateway,
        settings: LocalSettingsGateway,
        sync_window: Callable[[date], tuple[date, date]],
        clock: Callable[[], datetime],
        refresh_calendar: Callable[[date, date], object] | None,
        latest_completed_session: Callable[[datetime], date] | None,
        quality_mode: QualityMode,
        protected_manifest_ids: Callable[[], Sequence[str]] | None = None,
    ) -> None:
        self._sources = tuple(sorted(sources, key=lambda item: item.provider))
        self._providers = {provider.name: provider for provider in providers}
        self._service = service
        self._bundles = bundles
        self._watchlists = watchlists
        self._settings = settings
        self._sync_window = sync_window
        self._clock = clock
        self._refresh_calendar = refresh_calendar
        self._latest_completed_session = latest_completed_session
        self._quality_mode = quality_mode
        self._protected_manifest_ids = protected_manifest_ids
        self._preview_cache: tuple[str, tuple[MarketDataPreview, ...]] | None = None
        self._recover_interrupted_syncs()
        self._compact_market_versions()

    def sources(self) -> tuple[DataSourceSnapshot, ...]:
        latest: dict[str, DatasetManifestRecord] = {}
        with self._service.store.database.session_factory() as session:
            rows = session.scalars(
                select(DatasetManifestRecord).order_by(
                    DatasetManifestRecord.created_at.desc(),
                    DatasetManifestRecord.manifest_id.desc(),
                )
            ).all()
            for row in rows:
                latest.setdefault(row.provider, row)
        cache_bytes = sum(
            path.stat().st_size
            for path in self._service.store.data_dir.glob("*.parquet")
            if path.is_file()
        )
        result: list[DataSourceSnapshot] = []
        for configured in self._sources:
            latest_row = latest.get(configured.provider)
            quality = None
            if latest_row is not None and latest_row.quality_report_json is not None:
                payload = json.loads(latest_row.quality_report_json)
                quality = QualitySummary(
                    accepted=payload["accepted"],
                    mode=payload["mode"],
                    input_rows=payload["input_rows"],
                    output_rows=payload["output_rows"],
                    issue_codes=tuple(sorted(item["code"] for item in payload["issues"])),
                )
            result.append(
                DataSourceSnapshot(
                    provider=configured.provider,
                    source_name=configured.source_name,
                    available=configured.available,
                    last_update=None if latest_row is None else latest_row.created_at,
                    latest_manifest_id=None if latest_row is None else latest_row.manifest_id,
                    latest_source=None if latest_row is None else latest_row.provider,
                    quality=quality,
                    cache_bytes=cache_bytes,
                )
            )
        return tuple(result)

    def sync(
        self,
        provider: str,
        start: date | None = None,
        end: date | None = None,
        *,
        instruments: Sequence[InstrumentId] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> None:
        if cancel_requested is not None and not callable(cancel_requested):
            raise TypeError("cancel_requested must be callable or None")
        cancellation_probe = cancel_requested or (lambda: False)
        now = self._checked_sync_clock()
        requested_start, requested_end = self._requested_sync_range(
            now.date(),
            start,
            end,
        )
        run_id = self._start_sync_history(
            provider,
            DataSyncRange(requested_start, requested_end),
            now,
        )
        try:
            statistics = self._sync_range(
                provider,
                requested_start,
                requested_end,
                run_id,
                now,
                cancellation_probe,
                instruments,
            )
        except _SyncCancelled:
            self._compact_market_versions()
            self._preview_cache = None
            self._finish_sync_history(run_id, TaskStatus.CANCELLED)
        except Exception as error:
            cause = error.cause if isinstance(error, _PartialSyncFailure) else error
            instrument_failures = error.failures if isinstance(error, _PartialSyncFailure) else ()
            failure_code = self._sync_failure_code(cause)
            self._finish_sync_history(
                run_id,
                TaskStatus.FAILED,
                failure_code=failure_code,
                failure_type=type(cause).__name__,
                instrument_failures=instrument_failures,
            )
            if isinstance(cause, TaskOperationError):
                raise cause
            raise TaskOperationError(failure_code) from None
        else:
            self._finish_sync_history(
                run_id,
                TaskStatus.SUCCEEDED,
                statistics=statistics,
            )

    def _sync_range(
        self,
        provider: str,
        start: date,
        end: date,
        run_id: int,
        now: datetime,
        cancel_requested: Callable[[], bool],
        requested_instruments: Sequence[InstrumentId] | None,
    ) -> _SyncStatistics:
        self._raise_if_sync_cancelled(cancel_requested)
        selected = self._providers.get(provider)
        if selected is None:
            raise TaskOperationError("DATA_SOURCE_UNAVAILABLE")
        primary_watchlist = self._watchlists.primary()
        watchlist_instruments = () if primary_watchlist is None else primary_watchlist.instruments
        if not watchlist_instruments:
            raise TaskOperationError("SYNC_WATCHLIST_MISSING")
        if requested_instruments is None:
            instruments = watchlist_instruments
        else:
            instruments = tuple(requested_instruments)
            if (
                not instruments
                or any(type(item) is not InstrumentId for item in instruments)
                or instruments != tuple(sorted(set(instruments), key=str))
                or not set(instruments).issubset(watchlist_instruments)
            ):
                raise TaskOperationError("SYNC_TARGET_INVALID")
        fixed_fallback = tuple(self._providers)
        ordered_names = (provider,) + tuple(item for item in fixed_fallback if item != provider)
        ordered_providers = tuple(self._providers[item] for item in ordered_names)
        trusted_manifests, earliest_trusted_day = self._best_trusted_manifests(
            instruments,
            start,
            end,
        )
        self._raise_if_sync_cancelled(cancel_requested)
        today = now.date()
        try:
            if self._refresh_calendar is not None:
                self._refresh_calendar(
                    min(start, earliest_trusted_day or start),
                    today,
                )
            if self._latest_completed_session is not None:
                end = min(end, self._latest_completed_session(now))
        except Exception:
            raise TaskOperationError("SYNC_CALENDAR_UNAVAILABLE") from None
        if start > end:
            raise TaskOperationError("SYNC_COMPLETED_SESSION_UNAVAILABLE")
        self._update_sync_history_range(run_id, DataSyncRange(start, end))
        manifests: list[tuple[InstrumentId, DatasetManifest]] = []
        issue_codes: set[str] = set()
        modes: set[str] = set()
        downloaded_rows = 0
        reused_rows = 0
        remaining_requested_sessions = 0
        completed_instruments = 0
        first_failure: Exception | None = None
        instrument_failures: list[DataSyncInstrumentFailure] = []
        self._update_sync_history_progress(run_id, len(instruments), 0)
        concurrency = 1 if provider == "baostock" else min(2, len(instruments))

        def sync_instrument(instrument: InstrumentId) -> tuple[InstrumentId, SyncResult]:
            self._raise_if_sync_cancelled(cancel_requested)
            return (
                instrument,
                self._service.sync_daily_incremental(
                    DailyBarRequest(instrument, start, end),
                    ordered_providers,
                    self._quality_mode,
                    trusted_manifest=trusted_manifests.get(instrument),
                ),
            )

        with ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix="compass-market-sync",
        ) as executor:
            futures = {
                executor.submit(sync_instrument, instrument): instrument
                for instrument in instruments
            }
            for future in as_completed(futures):
                if cancel_requested():
                    for pending in futures:
                        pending.cancel()
                    self._save_partial_sync_bundle(
                        watchlist_instruments,
                        manifests,
                        modes,
                        issue_codes,
                    )
                    raise _SyncCancelled()
                try:
                    instrument, result = future.result()
                except _SyncCancelled:
                    for pending in futures:
                        pending.cancel()
                    self._save_partial_sync_bundle(
                        watchlist_instruments,
                        manifests,
                        modes,
                        issue_codes,
                    )
                    raise
                except Exception as error:
                    if first_failure is None:
                        first_failure = error
                    instrument_failures.append(
                        self._instrument_sync_failure(futures[future], error)
                    )
                    continue
                manifests.append((instrument, result.manifest))
                issue_codes.update(item.code for item in result.quality_report.issues)
                issue_codes.update(result.degradation_codes)
                modes.add(result.quality_report.mode.value)
                downloaded_rows += result.downloaded_rows
                reused_rows += result.reused_rows
                remaining_requested_sessions += result.remaining_requested_sessions
                completed_instruments += 1
                self._update_sync_history_progress(
                    run_id,
                    len(instruments),
                    completed_instruments,
                )
                if cancel_requested():
                    for pending in futures:
                        pending.cancel()
                    self._save_partial_sync_bundle(
                        watchlist_instruments,
                        manifests,
                        modes,
                        issue_codes,
                    )
                    raise _SyncCancelled()
        self._save_partial_sync_bundle(
            watchlist_instruments,
            manifests,
            modes,
            issue_codes,
        )
        if first_failure is not None:
            raise _PartialSyncFailure(
                first_failure,
                tuple(sorted(instrument_failures, key=lambda item: str(item.instrument))),
            )
        return _SyncStatistics(
            len(instruments),
            downloaded_rows,
            reused_rows,
            remaining_requested_sessions,
        )

    def watchlist_instruments(self) -> tuple[InstrumentId, ...]:
        primary = self._watchlists.primary()
        return () if primary is None else primary.instruments

    def _save_partial_sync_bundle(
        self,
        selected_instruments: tuple[InstrumentId, ...],
        new_manifests: Sequence[tuple[InstrumentId, DatasetManifest]],
        modes: set[str],
        issue_codes: set[str],
    ) -> DatasetBundle | None:
        if not new_manifests:
            return None
        selected = set(selected_instruments)
        refreshed = {instrument for instrument, _ in new_manifests}
        merged: dict[InstrumentId, DatasetManifest] = {}
        current = self._bundles.latest()
        if current is not None:
            references = self._bundles.references_by_instrument(current)
            merged.update(
                {
                    instrument: self._bundles.load_manifest(reference.manifest_id)
                    for instrument, reference in references.items()
                    if instrument in selected
                }
            )
            if any(instrument not in refreshed for instrument in merged):
                modes.add(cast(str, current.data_quality["mode"]))
                issue_codes.update(cast(tuple[str, ...], current.data_quality["issue_codes"]))
        merged.update(new_manifests)
        bundle = self._bundles.save(
            tuple(sorted(merged.items(), key=lambda item: str(item[0]))),
            mode="degraded" if "degraded" in modes else "strict",
            issue_codes=tuple(sorted(issue_codes)),
            replace_current=True,
        )
        self._compact_market_versions(bundle)
        self._preview_cache = None
        return bundle

    @classmethod
    def _instrument_sync_failure(
        cls,
        instrument: InstrumentId,
        error: Exception,
    ) -> DataSyncInstrumentFailure:
        if isinstance(error, DataQualityError):
            report = error.report
            return DataSyncInstrumentFailure(
                instrument,
                cls._sync_failure_code(error),
                tuple(sorted(issue.code for issue in report.issues)),
                report.input_rows,
                report.output_rows,
            )
        return DataSyncInstrumentFailure(
            instrument,
            cls._sync_failure_code(error),
        )

    @staticmethod
    def _raise_if_sync_cancelled(cancel_requested: Callable[[], bool]) -> None:
        cancelled = cancel_requested()
        if type(cancelled) is not bool:
            raise TypeError("cancel_requested must return an exact bool")
        if cancelled:
            raise _SyncCancelled()

    def _best_trusted_manifests(
        self,
        instruments: tuple[InstrumentId, ...],
        start: date,
        end: date,
    ) -> tuple[dict[InstrumentId, DatasetManifest], date | None]:
        requested = {str(instrument): instrument for instrument in instruments}
        candidates: dict[
            InstrumentId,
            tuple[tuple[int, int, datetime], DatasetManifest, date],
        ] = {}
        with self._service.store.database.session_factory() as session:
            rows = session.scalars(
                select(DatasetManifestRecord)
                .where(DatasetManifestRecord.instrument.in_(tuple(requested)))
                .order_by(DatasetManifestRecord.created_at.desc())
            ).all()
        for row in rows:
            instrument = requested.get(row.instrument)
            if instrument is None:
                continue
            try:
                manifest = self._bundles.load_manifest(row.manifest_id)
                frame = self._bundles.read_manifest(row.manifest_id)
            except (OSError, TypeError, ValueError):
                continue
            if frame.empty:
                continue
            covered_rows = len(frame.loc[pd.Timestamp(start) : pd.Timestamp(end)])
            score = (covered_rows, len(frame), row.created_at)
            current = candidates.get(instrument)
            if current is None or score > current[0]:
                candidates[instrument] = (
                    score,
                    manifest,
                    frame.index[0].date(),
                )
        manifests = {instrument: manifest for instrument, (_, manifest, _) in candidates.items()}
        earliest = min(
            (first_day for _, _, first_day in candidates.values()),
            default=None,
        )
        return manifests, earliest

    def sync_history(self, page: int, page_size: int) -> DataSyncHistoryPage:
        if type(page) is not int or page <= 0:
            raise ValueError("SYNC_HISTORY_PAGE_INVALID")
        if type(page_size) is not int or not 1 <= page_size <= 50:
            raise ValueError("SYNC_HISTORY_PAGE_SIZE_INVALID")
        with self._service.store.database.session_factory() as session:
            condition = TaskRun.task_name.like("sync:%")
            total = session.scalar(select(func.count()).select_from(TaskRun).where(condition))
            assert type(total) is int
            total_pages = max(1, (total + page_size - 1) // page_size)
            if page > total_pages:
                page = total_pages
            rows = session.scalars(
                select(TaskRun)
                .where(condition)
                .order_by(TaskRun.id.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            ).all()
        return DataSyncHistoryPage(
            tuple(self._sync_history_entry(row) for row in rows),
            page,
            page_size,
            total,
        )

    def clear_sync_history(self) -> int:
        with self._service.store.database.session_factory.begin() as session:
            condition = (
                TaskRun.task_name.like("sync:%"),
                TaskRun.status.in_(
                    (
                        TaskStatus.CANCELLED.value,
                        TaskStatus.SUCCEEDED.value,
                        TaskStatus.FAILED.value,
                    )
                ),
            )
            deleted = session.scalar(select(func.count()).select_from(TaskRun).where(*condition))
            assert type(deleted) is int
            session.execute(
                delete(TaskRun).where(
                    *condition,
                )
            )
        return deleted

    def clear_market_data(self, instrument: InstrumentId | None = None) -> int:
        if instrument is not None and type(instrument) is not InstrumentId:
            raise TypeError("instrument must be an exact InstrumentId or None")
        with self._service.store.database.session_factory() as session:
            running = session.scalar(
                select(func.count())
                .select_from(TaskRun)
                .where(
                    TaskRun.task_name.like("sync:%"),
                    TaskRun.status == TaskStatus.RUNNING.value,
                )
            )
        if running:
            raise TaskOperationError("SYNC_ACTIVE")
        current = self._bundles.latest()
        remaining: tuple[tuple[InstrumentId, DatasetManifest], ...] = ()
        if instrument is not None and current is not None and instrument in current.instruments:
            references = self._bundles.references_by_instrument(current)
            remaining = tuple(
                (item, self._bundles.load_manifest(reference.manifest_id))
                for item, reference in references.items()
                if item != instrument
            )
        self._bundles.delete_referencing(instrument)
        deleted = self._service.store.delete_market_data(instrument)
        if remaining and current is not None:
            replacement = self._bundles.save(
                remaining,
                mode=cast(str, current.data_quality["mode"]),
                issue_codes=cast(tuple[str, ...], current.data_quality["issue_codes"]),
            )
            self._compact_market_versions(replacement)
        self._preview_cache = None
        return deleted

    def _compact_market_versions(self, current: DatasetBundle | None = None) -> None:
        selected = self._bundles.latest() if current is None else current
        if selected is None:
            self._service.store.prune_superseded(())
            return
        references = self._bundles.references_by_instrument(selected)
        self._bundles.delete_except(selected.bundle_id)
        preferred = {reference.manifest_id for reference in references.values()}
        if self._protected_manifest_ids is not None:
            for manifest_id in self._protected_manifest_ids():
                try:
                    self._bundles.load_manifest(manifest_id)
                except (LookupError, OSError, ValueError):
                    continue
                preferred.add(manifest_id)
        self._service.store.prune_superseded(tuple(sorted(preferred)))

    def _checked_sync_clock(self) -> datetime:
        value = self._clock()
        if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
            raise TypeError("sync clock must return a timezone-aware datetime")
        return value

    def _recover_interrupted_syncs(self) -> None:
        completed_at = self._checked_sync_clock()
        with self._service.store.database.session_factory.begin() as session:
            rows = session.scalars(
                select(TaskRun).where(
                    TaskRun.task_name.like("sync:%"),
                    TaskRun.status == TaskStatus.RUNNING.value,
                )
            ).all()
            for row in rows:
                provider, date_range, _, _, _, progress, _ = self._decode_sync_history_detail(
                    row.detail
                )
                row.status = TaskStatus.FAILED.value
                row.completed_at = max(completed_at, row.started_at)
                row.detail = self._sync_history_detail(
                    provider,
                    date_range,
                    "SYNC_INTERRUPTED",
                    "RuntimeError",
                    progress=progress,
                )

    def _requested_sync_range(
        self,
        today: date,
        start: date | None,
        end: date | None,
    ) -> tuple[date, date]:
        if start is None and end is None:
            return self._sync_window(today)
        if type(start) is not date or type(end) is not date:
            raise TypeError("custom sync range must contain exact dates")
        assert isinstance(start, date) and isinstance(end, date)
        if start > end or end > today:
            raise ValueError("custom sync range is invalid")
        return start, end

    def _start_sync_history(
        self,
        provider: str,
        date_range: DataSyncRange,
        started_at: datetime,
    ) -> int:
        detail = self._sync_history_detail(provider, date_range, None, None)
        with self._service.store.database.session_factory.begin() as session:
            row = TaskRun(
                task_name=f"sync:{provider}",
                status=TaskStatus.RUNNING.value,
                started_at=started_at,
                completed_at=None,
                detail=detail,
            )
            session.add(row)
            session.flush()
            return row.id

    def _update_sync_history_range(
        self,
        run_id: int,
        date_range: DataSyncRange,
    ) -> None:
        with self._service.store.database.session_factory.begin() as session:
            row = session.get(TaskRun, run_id)
            if row is None:
                raise LookupError("SYNC_HISTORY_UNKNOWN")
            provider, _, _, _, _, _, _ = self._decode_sync_history_detail(row.detail)
            row.detail = self._sync_history_detail(provider, date_range, None, None)

    def _update_sync_history_progress(
        self,
        run_id: int,
        instrument_count: int,
        completed_instrument_count: int,
    ) -> None:
        progress = _SyncProgress(instrument_count, completed_instrument_count)
        with self._service.store.database.session_factory.begin() as session:
            row = session.get(TaskRun, run_id)
            if row is None or row.status != TaskStatus.RUNNING.value:
                raise LookupError("SYNC_HISTORY_UNKNOWN")
            provider, date_range, _, _, _, _, _ = self._decode_sync_history_detail(row.detail)
            row.detail = self._sync_history_detail(
                provider,
                date_range,
                None,
                None,
                progress=progress,
            )

    def _finish_sync_history(
        self,
        run_id: int,
        status: TaskStatus,
        *,
        failure_code: str | None = None,
        failure_type: str | None = None,
        statistics: _SyncStatistics | None = None,
        instrument_failures: tuple[DataSyncInstrumentFailure, ...] = (),
    ) -> None:
        if status not in {
            TaskStatus.CANCELLED,
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
        }:
            raise ValueError("sync history terminal status is invalid")
        if (status is TaskStatus.SUCCEEDED) != (statistics is not None):
            raise ValueError("sync history statistics are inconsistent")
        if status is TaskStatus.CANCELLED and (
            failure_code is not None or failure_type is not None
        ):
            raise ValueError("cancelled sync history failure fields are inconsistent")
        with self._service.store.database.session_factory.begin() as session:
            row = session.get(TaskRun, run_id)
            if row is None:
                raise LookupError("SYNC_HISTORY_UNKNOWN")
            provider, date_range, _, _, _, progress, _ = self._decode_sync_history_detail(
                row.detail
            )
            row.status = status.value
            row.completed_at = max(self._checked_sync_clock(), row.started_at)
            row.detail = self._sync_history_detail(
                provider,
                date_range,
                failure_code,
                failure_type,
                statistics,
                (
                    _SyncProgress(statistics.instrument_count, statistics.instrument_count)
                    if statistics is not None
                    else progress
                ),
                instrument_failures,
            )

    @staticmethod
    def _sync_failure_code(error: Exception) -> str:
        if isinstance(error, TaskOperationError):
            return error.code
        if isinstance(error, ProviderError):
            return f"PROVIDER_{error.kind.value.upper()}"
        if isinstance(error, DataQualityError):
            return "DATA_QUALITY_REJECTED"
        return "TASK_OPERATION_FAILED"

    @staticmethod
    def _sync_history_detail(
        provider: str,
        date_range: DataSyncRange,
        failure_code: str | None,
        failure_type: str | None,
        statistics: _SyncStatistics | None = None,
        progress: _SyncProgress | None = None,
        instrument_failures: tuple[DataSyncInstrumentFailure, ...] = (),
    ) -> str:
        return json.dumps(
            {
                "downloaded_rows": (None if statistics is None else statistics.downloaded_rows),
                "completed_instrument_count": (
                    None if progress is None else progress.completed_instrument_count
                ),
                "end": date_range.end.isoformat(),
                "failure_code": failure_code,
                "failure_type": failure_type,
                "instrument_count": (None if progress is None else progress.instrument_count),
                "instrument_failures": [
                    {
                        "failure_code": item.failure_code,
                        "input_rows": item.input_rows,
                        "instrument": str(item.instrument),
                        "output_rows": item.output_rows,
                        "quality_issue_codes": list(item.quality_issue_codes),
                    }
                    for item in instrument_failures
                ],
                "provider": provider,
                "remaining_requested_sessions": (
                    None if statistics is None else statistics.remaining_requested_sessions
                ),
                "reused_rows": None if statistics is None else statistics.reused_rows,
                "schema_version": 4,
                "start": date_range.start.isoformat(),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _decode_sync_history_detail(
        value: str | None,
    ) -> tuple[
        str,
        DataSyncRange,
        str | None,
        str | None,
        _SyncStatistics | None,
        _SyncProgress | None,
        tuple[DataSyncInstrumentFailure, ...],
    ]:
        if type(value) is not str:
            raise ValueError("SYNC_HISTORY_INTEGRITY")
        decoded = json.loads(value)
        if type(decoded) is not dict:
            raise ValueError("SYNC_HISTORY_INTEGRITY")
        common_keys = {
            "end",
            "failure_code",
            "failure_type",
            "provider",
            "schema_version",
            "start",
        }
        schema_version = decoded.get("schema_version")
        if schema_version == 1 and set(decoded) == common_keys:
            statistics = None
            progress = None
            instrument_failures: tuple[DataSyncInstrumentFailure, ...] = ()
        elif schema_version == 2 and set(decoded) == common_keys | {
            "downloaded_rows",
            "instrument_count",
            "remaining_requested_sessions",
            "reused_rows",
        }:
            statistic_values = (
                decoded["instrument_count"],
                decoded["downloaded_rows"],
                decoded["reused_rows"],
                decoded["remaining_requested_sessions"],
            )
            if all(item is None for item in statistic_values):
                statistics = None
                progress = None
            elif (
                type(statistic_values[0]) is int
                and statistic_values[0] > 0
                and all(type(item) is int and item >= 0 for item in statistic_values[1:])
            ):
                statistics = _SyncStatistics(*cast(tuple[int, int, int, int], statistic_values))
                progress = _SyncProgress(
                    statistics.instrument_count,
                    statistics.instrument_count,
                )
            else:
                raise ValueError("SYNC_HISTORY_INTEGRITY")
            instrument_failures = ()
        elif schema_version in {3, 4} and set(decoded) == common_keys | {
            "completed_instrument_count",
            "downloaded_rows",
            "instrument_count",
            "remaining_requested_sessions",
            "reused_rows",
        } | ({"instrument_failures"} if schema_version == 4 else set()):
            progress_values = (
                decoded["instrument_count"],
                decoded["completed_instrument_count"],
            )
            if all(item is None for item in progress_values):
                progress = None
            elif all(type(item) is int for item in progress_values):
                progress = _SyncProgress(
                    cast(int, progress_values[0]),
                    cast(int, progress_values[1]),
                )
            else:
                raise ValueError("SYNC_HISTORY_INTEGRITY")
            result_values = (
                decoded["downloaded_rows"],
                decoded["reused_rows"],
                decoded["remaining_requested_sessions"],
            )
            if all(item is None for item in result_values):
                statistics = None
            elif progress is not None and all(
                type(item) is int and item >= 0 for item in result_values
            ):
                statistics = _SyncStatistics(
                    progress.instrument_count,
                    cast(int, result_values[0]),
                    cast(int, result_values[1]),
                    cast(int, result_values[2]),
                )
            else:
                raise ValueError("SYNC_HISTORY_INTEGRITY")
            if schema_version == 3:
                instrument_failures = ()
            else:
                raw_failures = decoded["instrument_failures"]
                if type(raw_failures) is not list:
                    raise ValueError("SYNC_HISTORY_INTEGRITY")
                parsed_failures: list[DataSyncInstrumentFailure] = []
                try:
                    for raw_failure in raw_failures:
                        if type(raw_failure) is not dict or set(raw_failure) != {
                            "failure_code",
                            "input_rows",
                            "instrument",
                            "output_rows",
                            "quality_issue_codes",
                        }:
                            raise ValueError
                        raw_issues = raw_failure["quality_issue_codes"]
                        if type(raw_issues) is not list:
                            raise ValueError
                        parsed_failures.append(
                            DataSyncInstrumentFailure(
                                InstrumentId.parse(raw_failure["instrument"]),
                                raw_failure["failure_code"],
                                tuple(raw_issues),
                                raw_failure["input_rows"],
                                raw_failure["output_rows"],
                            )
                        )
                    instrument_failures = tuple(parsed_failures)
                except (TypeError, ValueError):
                    raise ValueError("SYNC_HISTORY_INTEGRITY") from None
        else:
            raise ValueError("SYNC_HISTORY_INTEGRITY")
        if (
            type(decoded["provider"]) is not str
            or type(decoded["start"]) is not str
            or type(decoded["end"]) is not str
            or decoded["failure_code"] is not None
            and type(decoded["failure_code"]) is not str
            or decoded["failure_type"] is not None
            and type(decoded["failure_type"]) is not str
        ):
            raise ValueError("SYNC_HISTORY_INTEGRITY")
        date_range = DataSyncRange(
            date.fromisoformat(decoded["start"]),
            date.fromisoformat(decoded["end"]),
        )
        return (
            decoded["provider"],
            date_range,
            decoded["failure_code"],
            decoded["failure_type"],
            statistics,
            progress,
            instrument_failures,
        )

    @classmethod
    def _sync_history_entry(cls, row: TaskRun) -> DataSyncHistoryEntry:
        if not row.task_name.startswith("sync:"):
            raise ValueError("SYNC_HISTORY_INTEGRITY")
        (
            provider,
            date_range,
            failure_code,
            failure_type,
            statistics,
            progress,
            instrument_failures,
        ) = cls._decode_sync_history_detail(row.detail)
        if row.task_name != f"sync:{provider}":
            raise ValueError("SYNC_HISTORY_INTEGRITY")
        try:
            status = TaskStatus(row.status)
        except ValueError:
            raise ValueError("SYNC_HISTORY_INTEGRITY") from None
        return DataSyncHistoryEntry(
            run_id=row.id,
            provider=provider,
            date_range=date_range,
            status=status,
            started_at=row.started_at,
            completed_at=row.completed_at,
            failure_code=failure_code,
            failure_type=failure_type,
            instrument_count=(None if progress is None else progress.instrument_count),
            completed_instrument_count=(
                None if progress is None else progress.completed_instrument_count
            ),
            downloaded_rows=None if statistics is None else statistics.downloaded_rows,
            reused_rows=None if statistics is None else statistics.reused_rows,
            remaining_requested_sessions=(
                None if statistics is None else statistics.remaining_requested_sessions
            ),
            instrument_failures=instrument_failures,
        )

    def latest_bundle(self) -> DatasetBundle | None:
        return self._bundles.latest()

    def latest_bundle_manifest_id(self) -> str | None:
        bundle = self._bundles.latest()
        return None if bundle is None else bundle.primary_manifest_id

    def latest_market_previews(self) -> tuple[MarketDataPreview, ...]:
        bundle = self._bundles.latest()
        if bundle is None:
            return ()
        cached = self._preview_cache
        if cached is not None and cached[0] == bundle.bundle_id:
            return cached[1]
        previews: list[MarketDataPreview] = []
        references = self._bundles.references_by_instrument(bundle)
        for instrument, reference in references.items():
            manifest = self._bundles.load_manifest(reference.manifest_id)
            frame = self._bundles.read_manifest(reference.manifest_id)
            if frame.empty:
                continue
            visible = frame.tail(252)
            bars = tuple(
                MarketBarPoint(
                    day=index.date().isoformat(),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
                for index, row in visible.iterrows()
            )
            provenance = manifest.provenance
            missing_sessions = () if provenance is None else provenance.missing_sessions
            first_day = frame.index[0].date()
            last_day = frame.index[-1].date()
            previews.append(
                MarketDataPreview(
                    instrument=instrument,
                    provider=manifest.provider,
                    manifest_id=manifest.manifest_id,
                    total_rows=manifest.rows,
                    first_day=frame.index[0].date().isoformat(),
                    last_day=frame.index[-1].date().isoformat(),
                    bars=bars,
                    daily_complete=(None if provenance is None else provenance.daily_complete),
                    missing_session_count=(len(missing_sessions)),
                    instrument_name=(
                        manifest.instrument_name or common_instrument_name(instrument)
                    ),
                    leading_missing_session_count=sum(day < first_day for day in missing_sessions),
                    internal_missing_session_count=sum(
                        first_day <= day <= last_day for day in missing_sessions
                    ),
                    trailing_missing_session_count=sum(day > last_day for day in missing_sessions),
                )
            )
        result = tuple(previews)
        self._preview_cache = (bundle.bundle_id, result)
        return result

    def bundle_for_manifest(self, manifest_id: str) -> DatasetBundle:
        return self._bundles.for_manifest(manifest_id)


class _StrategyDecisionSource:
    def __init__(self, strategy: Strategy) -> None:
        self._strategy = strategy

    def targets(self, context: StrategyContext) -> DecisionTarget:
        decision = self._strategy.generate_targets(context)
        totals: dict[InstrumentId, Decimal] = {}
        sleeves: dict[InstrumentId, dict[str, Decimal]] = {}
        for intent in decision:
            totals[intent.instrument] = (
                totals.get(intent.instrument, Decimal("0")) + intent.target_weight
            )
            sleeves.setdefault(intent.instrument, {})[intent.strategy_id] = (
                sleeves.setdefault(intent.instrument, {}).get(
                    intent.strategy_id,
                    Decimal("0"),
                )
                + intent.target_weight
            )
        normalized: dict[InstrumentId, Mapping[str, Decimal]] = {}
        for instrument, by_strategy in sleeves.items():
            total = totals[instrument]
            normalized[instrument] = (
                {next(iter(by_strategy)): Decimal("1")}
                if total == 0 and len(by_strategy) == 1
                else {
                    strategy_id: weight / total
                    for strategy_id, weight in by_strategy.items()
                    if weight > 0
                }
            )
        return DecisionTarget(
            totals,
            normalized,
            preserve_unspecified=decision.status is StrategyDecisionStatus.SKIPPED,
        )


def _trusted_backtest_sessions(
    bars: Mapping[InstrumentId, pd.DataFrame],
    expected_sessions: ExpectedSessions,
) -> tuple[date, ...]:
    """Build the event clock from one trusted calendar, never bar intersection."""

    if not bars:
        raise LookupError("BACKTEST_SESSIONS_MISSING")
    observed = tuple(timestamp.date() for frame in bars.values() for timestamp in frame.index)
    if not observed:
        raise LookupError("BACKTEST_SESSIONS_MISSING")
    anchor = min(bars, key=str)
    calendar = expected_sessions(DailyBarRequest(anchor, min(observed), max(observed)))
    if not isinstance(calendar, pd.DatetimeIndex) or calendar.tz is not None:
        raise TypeError("backtest calendar must return timezone-naive sessions")
    sessions = tuple(timestamp.date() for timestamp in calendar)
    if not sessions or tuple(sorted(set(sessions))) != sessions:
        raise LookupError("BACKTEST_SESSIONS_MISSING")
    return sessions


def _benchmark_curve(
    frame: pd.DataFrame,
    sessions: tuple[date, ...],
) -> tuple[CurvePoint, ...]:
    closes = {
        timestamp.date(): float(value)
        for timestamp, value in frame["close"].items()
        if timestamp.date() in set(sessions)
    }
    first_close = next((closes[day] for day in sessions if day in closes), None)
    if first_close is None:
        raise LookupError("BACKTEST_BENCHMARK_DATA_MISSING")
    current = first_close
    points: list[CurvePoint] = []
    for day in sessions:
        current = closes.get(day, current)
        points.append(CurvePoint(day.isoformat(), current / first_close))
    return tuple(points)


class LocalBacktestGateway:
    def __init__(
        self,
        *,
        registry: StrategyRegistry,
        strategies: LocalStrategyGateway,
        bundles: DatasetBundleRepository,
        reports: BacktestReportRepository,
        decisions: DecisionExportRepository,
        settings: LocalSettingsGateway,
        app_git_commit: str | None,
        expected_sessions: ExpectedSessions,
    ) -> None:
        self._registry = registry
        self._strategies = strategies
        self._bundles = bundles
        self._reports = reports
        self._decisions = decisions
        self._settings = settings
        self._app_git_commit = app_git_commit
        self._expected_sessions = expected_sessions

    def list_reports(self) -> tuple[BacktestReport, ...]:
        return self._reports.list()

    def run(self, submission: BacktestSubmission) -> None:
        if type(submission) is not BacktestSubmission:
            raise TypeError("submission must be an exact BacktestSubmission")
        if is_reserved_decision_snapshot_run_id(submission.run_id):
            raise ValueError("BACKTEST_RUN_ID_RESERVED")
        app_git_commit = self._app_git_commit
        if app_git_commit is None:
            raise LookupError("BACKTEST_APP_VERSION_UNKNOWN")
        instance = next(
            (
                item
                for item in self._strategies.list()
                if item.instance_id == submission.strategy_instance_id
            ),
            None,
        )
        if instance is None:
            raise LookupError("BACKTEST_STRATEGY_MISSING")
        bundle = self._bundles.for_manifest(submission.manifest_id)
        pool = self._strategies.pool_instruments(instance.instance_id)
        manifest_by_instrument = self._bundles.references_by_instrument(bundle)
        if not set(pool).issubset(manifest_by_instrument):
            raise LookupError("BACKTEST_POOL_DATA_MISSING")
        bars = {
            instrument: self._bundles.read_manifest(manifest_by_instrument[instrument].manifest_id)
            for instrument in pool
        }
        dataset_manifests = tuple(
            self._bundles.load_manifest(manifest_by_instrument[instrument].manifest_id)
            for instrument in pool
        )
        provenance = validate_dataset_provenance(
            dataset_manifests,
            required_through=date.min,
            failure_prefix="BACKTEST",
            allow_quality_gaps=bundle.data_quality.get("mode") == "degraded",
        )
        sessions = provenance.sessions
        metadata = self._registry.describe(instance.strategy_type)
        parameters = metadata.parameters_type.model_validate_json(
            strategy_parameters_json(instance.parameters),
            strict=True,
        )
        strategy = self._registry.create(
            instance.strategy_type,
            parameters,
            instance.instance_id,
        )
        instruments = local_instruments(pool)
        state = self._settings.state()
        fee = state.fee_profile
        fee_confirmed = fee is not None and fee.confirmed
        risk_active = any(item.active for item in state.risk_templates)
        rule_book = local_rule_book(
            instruments,
            fee_confirmed=fee_confirmed,
        )
        result = BacktestEngine().run(
            BacktestRequest(
                run_id=submission.run_id,
                sessions=sessions,
                instruments=instruments,
                bars=bars,
                initial_cash=Decimal("100000.00"),
                initial_positions=(),
                corporate_actions=provenance.corporate_actions,
                decision_source=_StrategyDecisionSource(strategy),
                risk_engine=local_risk_engine(active=risk_active),
                rule_book=rule_book,
            )
        )
        snapshot = RunSnapshot(
            run_id=submission.run_id,
            schema_version=1,
            market_manifests=bundle.market_manifests,
            data_quality=snapshot_data_quality(bundle.data_quality, provenance),
            strategies=(
                StrategySnapshot(
                    sleeve_id=instance.instance_id,
                    strategy_type=instance.strategy_type,
                    strategy_version=instance.strategy_version,
                    parameters=parameters.model_dump(mode="json"),
                ),
            ),
            instrument_pool={
                "instruments": tuple(str(item) for item in pool),
                "snapshot_id": instance.pool_snapshot_id,
            },
            survivorship_bias={"mode": "static_pool", "warning": True},
            allocator_configuration={"kind": "single_strategy"},
            risk_configuration={
                "active": risk_active,
                "template_ids": tuple(
                    item.template_id for item in state.risk_templates if item.active
                ),
            },
            market_rule_configuration={
                "execution": "next_open",
                "profile_ids": local_profile_ids(
                    instruments,
                    fee_confirmed=fee_confirmed,
                ),
            },
            fee_profile_configuration={
                "confirmed": fee_confirmed,
                "profile_id": None if fee is None else fee.profile_id,
            },
            app_git_commit=app_git_commit,
            random_seed=0,
        )
        benchmark = _benchmark_curve(bars[pool[0]], sessions)
        report = BacktestReport.from_result(
            run_id=submission.run_id,
            configuration_id=submission.configuration_id,
            strategy_instance_ids=(instance.instance_id,),
            result=result,
            snapshot=snapshot,
            benchmark_curve=benchmark,
            export_available=True,
        )
        self._reports.save(report)

    def report(self, run_id: str) -> BacktestReport | None:
        return self._reports.get(run_id)

    def load_backtest(self, run_id: str) -> BacktestReport:
        report = self._reports.get(run_id)
        if report is None:
            raise LookupError("BACKTEST_REPORT_MISSING")
        return report

    def load_decision_export(self, decision_id: str) -> DecisionExportRecord:
        record = self._decisions.get(decision_id)
        if record is None:
            raise LookupError("DECISION_EXPORT_MISSING")
        return record


class LocalDashboardGateway:
    def __init__(
        self,
        accounts: AccountRepository,
        *,
        bundles: DatasetBundleRepository,
        reports: BacktestReportRepository,
        decisions: DecisionExportRepository,
    ) -> None:
        self._accounts = accounts
        self._bundles = bundles
        self._reports = reports
        self._decisions = decisions
        self._intraday: IntradayState | None = None

    def state(self) -> DashboardSnapshot:
        record = self._decisions.latest()
        decision = None if record is None else record.result
        account = self._accounts.latest()
        decision_failure = None
        if decision is not None:
            if (
                account is None
                or account.row_id != decision.account_snapshot_row_id
                or account.content_hash != decision.account_snapshot_hash
            ):
                assert record is not None
                decision = None
                decision_failure = DashboardFailure(
                    "DECISION_SUPERSEDED",
                    record.decision_id,
                )
        bundle = self._bundles.latest()
        data_health = (
            None
            if bundle is None
            else DashboardDataHealth(
                accepted=cast(bool, bundle.data_quality["accepted"]),
                mode=cast(str, bundle.data_quality["mode"]),
                issue_codes=cast(
                    tuple[str, ...],
                    bundle.data_quality["issue_codes"],
                ),
            )
        )
        latest_report = self._reports.latest()
        drawdown = (
            None
            if latest_report is None or latest_report.metrics.maximum_drawdown is None
            else Decimal(str(latest_report.metrics.maximum_drawdown))
        )
        if decision is not None:
            weights = {item.instrument: item.current_weight for item in decision.recommendations}
            targets = {item.instrument: item.final_weight for item in decision.recommendations}
        else:
            weights = {}
            if account is not None and account.snapshot.equity:
                weights = {
                    position.instrument: position.market_value / account.snapshot.equity
                    for position in account.snapshot.positions
                }
            targets = {}
        return DashboardSnapshot(
            data_health=data_health,
            account=account,
            current_drawdown=drawdown,
            risk_status=(
                None
                if decision is None
                else "BLOCKED"
                if any(item.blocked for item in decision.recommendations)
                else "NORMAL"
            ),
            actual_weights=weights,
            target_weights=targets,
            intraday=self._intraday,
            confirmed_decision=decision,
            decision_failure=decision_failure,
        )

    def set_intraday(self, state: IntradayState) -> None:
        self._intraday = state

    def clear_intraday(self) -> None:
        self._intraday = None
