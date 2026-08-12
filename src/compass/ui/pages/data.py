from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from threading import Event, RLock
from typing import Protocol, TypeVar, cast

from compass.domain.market import InstrumentId
from compass.services.instrument_names import common_instrument_name
from compass.services.safe_display import (
    frozen_errors,
    safe_display_text,
    safe_exception_type,
    safe_identifier,
    stable_code,
)
from compass.services.task_manager import (
    Operation,
    TaskFailure,
    TaskSnapshot,
    TaskStatus,
)
from compass.ui.components.charts import (
    MarketBarPoint,
    market_data_chart_options,
    thaw_chart_options,
)

from nicegui import ui
from nicegui.elements.timer import Timer


class DataPageError(RuntimeError):
    """A stable, secret-safe failure while reading page state."""

    def __init__(self, code: str) -> None:
        self.code = stable_code(code, label="data page error code")
        super().__init__(self.code)


T = TypeVar("T")
_MISSING = object()
_TERMINAL_SYNC_STATUSES = frozenset({TaskStatus.CANCELLED, TaskStatus.SUCCEEDED, TaskStatus.FAILED})
_SYNC_FAILURE_GUIDANCE = {
    "SYNC_WATCHLIST_MISSING": "请先在“标的池”创建并启用至少一个 ETF 或股票。",
    "SYNC_COMPLETED_SESSION_UNAVAILABLE": "当前没有可同步的已完成交易日，请在收盘后重试。",
    "SYNC_CALENDAR_UNAVAILABLE": "交易日历不可用；本次尚未请求行情，请检查网络或代理。",
    "SYNC_INTERRUPTED": "上次任务在应用退出前未完成，请重新发起同步。",
    "DATA_SOURCE_UNAVAILABLE": "该数据源当前不可用，请检查系统设置。",
    "PROVIDER_NETWORK": "行情源网络连接失败，请检查网络权限、代理或稍后重试。",
    "PROVIDER_RATE_LIMIT": "行情源触发访问频率限制，请稍后重试。",
    "PROVIDER_AUTHENTICATION": "行情源认证失败，请检查数据源配置。",
    "PROVIDER_MALFORMED_RESPONSE": "行情源返回结构异常，可能是上游接口变更。",
    "PROVIDER_CONFIGURATION": "行情源配置不完整，请检查系统设置。",
    "PROVIDER_CAPABILITY": "行情源不支持当前标的或所需字段。",
    "DATA_QUALITY_REJECTED": "返回数据未通过质量检查。",
}
_SYNC_RANGE_YEARS = {"2y": 2, "5y": 5, "10y": 10}
_QUALITY_ISSUE_GUIDANCE = {
    "INVALID_FRAME": "返回内容不是有效的行情表格",
    "DUPLICATE_COLUMN": "返回数据包含重复字段",
    "MISSING_COLUMN": "返回数据缺少必需字段",
    "INVALID_INDEX": "交易日期格式无效",
    "TIMEZONE_AWARE_INDEX": "交易日期携带了不应存在的时区",
    "NON_NORMALIZED_SESSION": "交易日期不是标准日线日期",
    "DUPLICATE_SESSION": "返回数据包含重复交易日",
    "OUT_OF_ORDER": "交易日期没有按顺序排列",
    "UNEXPECTED_SESSION": "返回数据包含非预期交易日",
    "MISSING_SESSION": "返回数据缺少预期交易日",
    "INVALID_NUMERIC_TYPE": "价格或成交字段不是有效数字",
    "NONFINITE_VALUE": "价格或成交字段包含空值或无穷值",
    "NONPOSITIVE_PRICE": "返回数据包含零或负价格",
    "NEGATIVE_ACTIVITY": "返回数据包含负成交量或成交额",
    "INVALID_OHLC": "开高低收价格关系异常",
    "INVALID_ADJUST_FACTOR": "复权因子无效",
    "INVALID_SUSPENDED": "停牌标记无效",
    "INVALID_LIMIT_PRICE": "涨跌停价格无效",
    "INVALID_LIMIT_RELATION": "涨跌停价格关系异常",
    "INVALID_ADJUST_FLAG": "复权标记无效",
    "LARGE_PRICE_JUMP": "相邻交易日价格跳变过大",
    "ADJUST_FACTOR_DISCONTINUITY": "复权因子出现不连续",
    "EMPTY_DATASET": "数据源没有返回任何可用行情",
}


def _years_before(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


@dataclass(frozen=True, slots=True)
class DataSyncRange:
    start: date
    end: date

    def __post_init__(self) -> None:
        if type(self.start) is not date or type(self.end) is not date:
            raise TypeError("data sync range must contain exact dates")
        if self.start > self.end:
            raise ValueError("data sync range start must not follow end")


@dataclass(frozen=True, slots=True)
class DataSyncRangeValidationResult:
    errors: Mapping[str, str]
    date_range: DataSyncRange | None

    def __post_init__(self) -> None:
        errors = frozen_errors(self.errors)
        if self.date_range is not None and type(self.date_range) is not DataSyncRange:
            raise TypeError("date_range must be an exact DataSyncRange or None")
        if (self.date_range is None) == (not bool(errors)):
            raise ValueError("validation must contain either errors or a date range")
        object.__setattr__(self, "errors", errors)


@dataclass(frozen=True, slots=True)
class DataSyncRangeForm:
    mode: object
    start_text: object = ""
    end_text: object = ""
    today: Callable[[], date] = date.today

    def __post_init__(self) -> None:
        if not callable(self.today):
            raise TypeError("today must be callable")

    def validate(self) -> DataSyncRangeValidationResult:
        current = self.today()
        if type(current) is not date:
            raise TypeError("today must return an exact date")
        if type(self.mode) is not str or self.mode not in {*_SYNC_RANGE_YEARS, "custom"}:
            return DataSyncRangeValidationResult({"mode": "请选择有效的行情区间"}, None)
        if self.mode in _SYNC_RANGE_YEARS:
            return DataSyncRangeValidationResult(
                {},
                DataSyncRange(_years_before(current, _SYNC_RANGE_YEARS[self.mode]), current),
            )
        start = self._date(self.start_text, label="开始日期")
        end = self._date(self.end_text, label="结束日期")
        errors: dict[str, str] = {}
        if isinstance(start, str):
            errors["start"] = start
        if isinstance(end, str):
            errors["end"] = end
        if errors:
            return DataSyncRangeValidationResult(errors, None)
        assert isinstance(start, date) and isinstance(end, date)
        if start > end:
            return DataSyncRangeValidationResult(
                {"range": "开始日期不能晚于结束日期"},
                None,
            )
        if end > current:
            return DataSyncRangeValidationResult(
                {"end": "结束日期不能晚于今天"},
                None,
            )
        return DataSyncRangeValidationResult({}, DataSyncRange(start, end))

    @staticmethod
    def _date(value: object, *, label: str) -> date | str:
        if type(value) is not str:
            return f"{label}必须使用日期格式"
        assert isinstance(value, str)
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            return f"{label}必须使用 YYYY-MM-DD"
        if parsed.isoformat() != value:
            return f"{label}必须使用 YYYY-MM-DD"
        return parsed


def _boundary_call(code: str, operation: Callable[[], T]) -> T:
    failed = False
    result: object = _MISSING
    try:
        result = operation()
    except Exception:
        failed = True
    if failed:
        raise DataPageError(code)
    return cast(T, result)


def _sync_task_snapshot(
    value: object,
    *,
    provider: str,
    task_id: str | None = None,
) -> TaskSnapshot:
    if type(value) is not TaskSnapshot:
        raise TypeError("task gateway must return an exact TaskSnapshot")
    assert isinstance(value, TaskSnapshot)
    if value.heavy is not True or value.name != f"sync:{provider}":
        raise ValueError("task gateway returned an inconsistent sync task")
    if task_id is not None and value.task_id != task_id:
        raise ValueError("task gateway returned an inconsistent task identity")
    return value


def _source_availability(value: Sequence[DataSourceSnapshot]) -> dict[str, bool]:
    state = DataPageState(tuple(value), None)
    return {source.provider: source.available for source in state.sources}


def sync_failure_text(failure: TaskFailure) -> str:
    if type(failure) is not TaskFailure:
        raise TypeError("sync failure must be an exact TaskFailure")
    guidance = _SYNC_FAILURE_GUIDANCE.get(failure.code)
    if guidance is None:
        return f"同步失败：{failure.code} / 错误 ID {failure.error_id}"
    return f"同步失败：{guidance}（{failure.code} / 错误 ID {failure.error_id}）"


@dataclass(frozen=True, slots=True)
class QualitySummary:
    accepted: bool
    mode: str
    input_rows: int
    output_rows: int
    issue_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool:
            raise TypeError("quality accepted must be an exact bool")
        if type(self.mode) is not str or self.mode not in {"strict", "degraded"}:
            raise ValueError("quality mode is invalid")
        if (
            type(self.input_rows) is not int
            or type(self.output_rows) is not int
            or self.input_rows < 0
            or not 0 <= self.output_rows <= self.input_rows
        ):
            raise ValueError("quality row counts are invalid")
        issues = tuple(self.issue_codes)
        for code in issues:
            stable_code(code, label="quality issue code")
        if len(set(issues)) != len(issues):
            raise ValueError("quality issue codes must be unique")
        if issues != tuple(sorted(issues)):
            raise ValueError("quality issue codes must be sorted")
        object.__setattr__(self, "issue_codes", issues)


@dataclass(frozen=True, slots=True)
class DataSourceSnapshot:
    provider: str
    source_name: str
    available: bool
    last_update: datetime | None
    latest_manifest_id: str | None
    latest_source: str | None
    quality: QualitySummary | None
    cache_bytes: int

    def __post_init__(self) -> None:
        safe_identifier(self.provider, label="provider")
        safe_display_text(self.source_name, label="source name")
        if type(self.available) is not bool:
            raise TypeError("available must be an exact bool")
        if self.last_update is not None:
            if type(self.last_update) is not datetime:
                raise TypeError("last_update must be an exact datetime")
            if self.last_update.tzinfo is None or self.last_update.utcoffset() is None:
                raise ValueError("last_update must be timezone-aware")
        if self.latest_manifest_id is not None:
            safe_identifier(self.latest_manifest_id, label="manifest id")
        if self.latest_source is not None:
            safe_identifier(self.latest_source, label="latest source")
        if self.quality is not None and type(self.quality) is not QualitySummary:
            raise TypeError("quality must be an exact QualitySummary or None")
        if type(self.cache_bytes) is not int or self.cache_bytes < 0:
            raise ValueError("cache bytes must be a non-negative exact integer")


def _preview_day(value: str, *, label: str) -> date:
    if type(value) is not str:
        raise TypeError(f"{label} must be exact text")
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{label} must be an ISO date") from None
    if parsed.isoformat() != value:
        raise ValueError(f"{label} must be a canonical ISO date")
    return parsed


@dataclass(frozen=True, slots=True)
class MarketDataPreview:
    instrument: InstrumentId
    provider: str
    manifest_id: str
    total_rows: int
    first_day: str
    last_day: str
    bars: tuple[MarketBarPoint, ...]
    daily_complete: bool | None
    missing_session_count: int
    instrument_name: str | None = None
    leading_missing_session_count: int = 0
    internal_missing_session_count: int = 0
    trailing_missing_session_count: int = 0

    def __post_init__(self) -> None:
        if type(self.instrument) is not InstrumentId:
            raise TypeError("preview instrument must be an exact InstrumentId")
        safe_identifier(self.provider, label="preview provider")
        safe_identifier(self.manifest_id, label="preview manifest id")
        if type(self.total_rows) is not int or self.total_rows <= 0:
            raise ValueError("preview total rows must be a positive exact integer")
        first_day = _preview_day(self.first_day, label="preview first day")
        last_day = _preview_day(self.last_day, label="preview last day")
        if first_day > last_day:
            raise ValueError("preview date range is invalid")
        bars = tuple(self.bars)
        if not bars or any(type(item) is not MarketBarPoint for item in bars):
            raise TypeError("preview bars must contain exact MarketBarPoint values")
        days = tuple(item.day for item in bars)
        if len(set(days)) != len(days) or days != tuple(sorted(days)):
            raise ValueError("preview bar dates must be unique and increasing")
        if self.total_rows < len(bars) or days[-1] != self.last_day:
            raise ValueError("preview rows are inconsistent with its dataset summary")
        if not self.first_day <= days[0] <= days[-1] <= self.last_day:
            raise ValueError("preview bars fall outside its dataset range")
        if self.daily_complete is not None and type(self.daily_complete) is not bool:
            raise TypeError("preview daily completeness must be an exact bool or None")
        if type(self.missing_session_count) is not int or self.missing_session_count < 0:
            raise ValueError("preview missing session count must be non-negative")
        missing_parts = (
            self.leading_missing_session_count,
            self.internal_missing_session_count,
            self.trailing_missing_session_count,
        )
        if any(type(item) is not int or item < 0 for item in missing_parts):
            raise ValueError("preview missing session parts must be non-negative")
        if sum(missing_parts) != self.missing_session_count:
            raise ValueError("preview missing session summary is inconsistent")
        if self.instrument_name is not None:
            safe_display_text(
                self.instrument_name,
                label="preview instrument name",
                maximum=128,
            )
        object.__setattr__(self, "bars", bars)

    @property
    def display_label(self) -> str:
        code = str(self.instrument)
        if self.instrument_name is None:
            return code
        return f"{self.instrument_name}（{code}）"


@dataclass(frozen=True, slots=True)
class DataSyncInstrumentFailure:
    instrument: InstrumentId
    failure_code: str
    quality_issue_codes: tuple[str, ...] = ()
    input_rows: int | None = None
    output_rows: int | None = None

    def __post_init__(self) -> None:
        if type(self.instrument) is not InstrumentId:
            raise TypeError("sync failure instrument must be an exact InstrumentId")
        stable_code(self.failure_code, label="instrument sync failure code")
        issues = tuple(self.quality_issue_codes)
        if any(type(item) is not str for item in issues) or issues != tuple(sorted(set(issues))):
            raise ValueError("instrument quality issue codes must be unique and sorted")
        for issue in issues:
            stable_code(issue, label="instrument quality issue code")
        rows = (self.input_rows, self.output_rows)
        if any(item is not None for item in rows):
            if (
                type(self.input_rows) is not int
                or self.input_rows < 0
                or type(self.output_rows) is not int
                or not 0 <= self.output_rows <= self.input_rows
            ):
                raise ValueError("instrument quality row counts are inconsistent")
        object.__setattr__(self, "quality_issue_codes", issues)


@dataclass(frozen=True, slots=True)
class DataSyncHistoryEntry:
    run_id: int
    provider: str
    date_range: DataSyncRange
    status: TaskStatus
    started_at: datetime
    completed_at: datetime | None
    failure_code: str | None = None
    failure_type: str | None = None
    instrument_count: int | None = None
    completed_instrument_count: int | None = None
    downloaded_rows: int | None = None
    reused_rows: int | None = None
    remaining_requested_sessions: int | None = None
    instrument_failures: tuple[DataSyncInstrumentFailure, ...] = ()

    def __post_init__(self) -> None:
        if type(self.run_id) is not int or self.run_id <= 0:
            raise ValueError("sync history id must be a positive exact integer")
        safe_identifier(self.provider, label="sync history provider")
        if type(self.date_range) is not DataSyncRange:
            raise TypeError("sync history range must be an exact DataSyncRange")
        if self.status not in {
            TaskStatus.RUNNING,
            TaskStatus.CANCELLED,
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
        }:
            raise ValueError("sync history status is invalid")
        for label, value in (
            ("started_at", self.started_at),
            ("completed_at", self.completed_at),
        ):
            if value is None and label == "completed_at":
                continue
            if type(value) is not datetime:
                raise TypeError(f"{label} must be an exact datetime")
            assert isinstance(value, datetime)
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{label} must be timezone-aware")
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("sync history completion cannot precede start")
        if self.status is TaskStatus.RUNNING:
            if (
                self.completed_at is not None
                or self.failure_code is not None
                or self.failure_type is not None
            ):
                raise ValueError("running sync history fields are inconsistent")
        elif self.status is TaskStatus.CANCELLED:
            if (
                self.completed_at is None
                or self.failure_code is not None
                or self.failure_type is not None
            ):
                raise ValueError("cancelled sync history fields are inconsistent")
        elif self.status is TaskStatus.SUCCEEDED:
            if (
                self.completed_at is None
                or self.failure_code is not None
                or self.failure_type is not None
            ):
                raise ValueError("successful sync history fields are inconsistent")
        elif self.completed_at is None or self.failure_code is None or self.failure_type is None:
            raise ValueError("failed sync history fields are inconsistent")
        if self.failure_code is not None:
            stable_code(self.failure_code, label="sync history failure code")
        if self.failure_type is not None:
            safe_exception_type(self.failure_type)
        instrument_failures = tuple(self.instrument_failures)
        if any(
            type(item) is not DataSyncInstrumentFailure for item in instrument_failures
        ) or tuple(item.instrument for item in instrument_failures) != tuple(
            sorted({item.instrument for item in instrument_failures}, key=str)
        ):
            raise ValueError("sync history instrument failures are inconsistent")
        if instrument_failures and self.status is not TaskStatus.FAILED:
            raise ValueError("only failed sync history may contain instrument failures")
        progress = (self.instrument_count, self.completed_instrument_count)
        if any(item is not None for item in progress):
            if (
                type(self.instrument_count) is not int
                or self.instrument_count <= 0
                or type(self.completed_instrument_count) is not int
                or not 0 <= self.completed_instrument_count <= self.instrument_count
            ):
                raise ValueError("sync history progress is inconsistent")
        statistics = (
            self.downloaded_rows,
            self.reused_rows,
            self.remaining_requested_sessions,
        )
        if any(item is not None for item in statistics):
            if (
                any(type(item) is not int or item < 0 for item in statistics)
                or self.status is not TaskStatus.SUCCEEDED
                or self.instrument_count is None
                or self.completed_instrument_count != self.instrument_count
            ):
                raise ValueError("sync history statistics are inconsistent")
        object.__setattr__(self, "instrument_failures", instrument_failures)


@dataclass(frozen=True, slots=True)
class DataSyncHistoryPage:
    entries: tuple[DataSyncHistoryEntry, ...]
    page: int
    page_size: int
    total_items: int

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if any(type(item) is not DataSyncHistoryEntry for item in entries):
            raise TypeError("sync history page entries are invalid")
        if type(self.page) is not int or self.page <= 0:
            raise ValueError("sync history page must be positive")
        if type(self.page_size) is not int or not 1 <= self.page_size <= 50:
            raise ValueError("sync history page size is invalid")
        if type(self.total_items) is not int or self.total_items < len(entries):
            raise ValueError("sync history total is invalid")
        if self.page > self.total_pages:
            raise ValueError("sync history page exceeds the available range")
        object.__setattr__(self, "entries", entries)

    @property
    def total_pages(self) -> int:
        return max(1, (self.total_items + self.page_size - 1) // self.page_size)


def sync_coverage_notice(
    entry: DataSyncHistoryEntry,
    previews: Sequence[MarketDataPreview],
    *,
    use_current_previews: bool,
) -> tuple[str, str]:
    """Explain whether uncovered sessions precede history or break current data."""

    remaining = entry.remaining_requested_sessions
    if entry.status is not TaskStatus.SUCCEEDED or remaining is None:
        raise ValueError("coverage notice requires successful sync statistics")
    if remaining == 0:
        return "所选区间已完整覆盖并写入。", "text-xs text-emerald-700"
    current = tuple(previews)
    if (
        not use_current_previews
        or entry.instrument_count != len(current)
        or any(type(item) is not MarketDataPreview for item in current)
    ):
        return (
            f"所选区间仍有 {remaining:,} 个标的交易日未覆盖；"
            "可能包含上市前或较早历史，请结合最新行情预览判断。",
            "text-xs text-amber-800",
        )
    leading_sessions = sum(item.leading_missing_session_count for item in current)
    current_gap_sessions = sum(
        item.internal_missing_session_count + item.trailing_missing_session_count
        for item in current
    )
    if leading_sessions + current_gap_sessions != remaining:
        return (
            f"所选区间仍有 {remaining:,} 个标的交易日未覆盖；"
            "可能包含上市前或较早历史，请结合最新行情预览判断。",
            "text-xs text-amber-800",
        )
    leading_instruments = sum(item.leading_missing_session_count > 0 for item in current)
    gap_instruments = sum(
        item.internal_missing_session_count + item.trailing_missing_session_count > 0
        for item in current
    )
    progress = f"{entry.completed_instrument_count}/{entry.instrument_count}"
    if current_gap_sessions == 0:
        return (
            f"同步成功：{progress} 个标的。{leading_instruments} 个标的仅有 "
            f"{leading_sessions:,} 个上市前或较早历史交易日未覆盖；"
            "现有行情区间连续，不属于内部缺口。",
            "text-xs text-amber-800",
        )
    if leading_sessions == 0:
        return (
            f"同步完成，但 {gap_instruments} 个标的在现有行情区间或最新端仍有 "
            f"{current_gap_sessions:,} 个交易日缺口，建议重试。",
            "text-xs text-red-700",
        )
    return (
        f"同步完成：{leading_instruments} 个标的有 {leading_sessions:,} 个上市前或"
        f"较早历史交易日未覆盖；另有 {gap_instruments} 个标的在现有行情区间或"
        f"最新端存在 {current_gap_sessions:,} 个交易日缺口，建议重试。",
        "text-xs text-red-700",
    )


@dataclass(frozen=True, slots=True)
class DataPageState:
    sources: tuple[DataSourceSnapshot, ...]
    active_sync: TaskSnapshot | None
    latest_bundle_manifest_id: str | None = None
    previews: tuple[MarketDataPreview, ...] = ()
    sync_history: tuple[DataSyncHistoryEntry, ...] = ()
    sync_history_page: int = 1
    sync_history_total_pages: int = 1
    sync_history_total_items: int = 0
    watchlist_instruments: tuple[InstrumentId, ...] = ()

    def __post_init__(self) -> None:
        sources = tuple(self.sources)
        if any(type(source) is not DataSourceSnapshot for source in sources):
            raise TypeError("sources must contain exact DataSourceSnapshot values")
        if self.active_sync is not None and type(self.active_sync) is not TaskSnapshot:
            raise TypeError("active_sync must be an exact TaskSnapshot or None")
        if self.latest_bundle_manifest_id is not None:
            safe_identifier(
                self.latest_bundle_manifest_id,
                label="latest dataset bundle manifest id",
            )
        previews = tuple(self.previews)
        if any(type(preview) is not MarketDataPreview for preview in previews):
            raise TypeError("previews must contain exact MarketDataPreview values")
        instruments = tuple(str(preview.instrument) for preview in previews)
        if len(set(instruments)) != len(instruments) or instruments != tuple(sorted(instruments)):
            raise ValueError("previews must have unique, sorted instruments")
        history = tuple(self.sync_history)
        if any(type(item) is not DataSyncHistoryEntry for item in history):
            raise TypeError("sync history must contain exact DataSyncHistoryEntry values")
        history_ids = tuple(item.run_id for item in history)
        if len(set(history_ids)) != len(history_ids):
            raise ValueError("sync history must have unique ids")
        if history_ids != tuple(sorted(history_ids, reverse=True)):
            raise ValueError("sync history must be newest first")
        if (
            type(self.sync_history_page) is not int
            or self.sync_history_page <= 0
            or type(self.sync_history_total_pages) is not int
            or self.sync_history_total_pages <= 0
            or self.sync_history_page > self.sync_history_total_pages
            or type(self.sync_history_total_items) is not int
            or self.sync_history_total_items < len(history)
        ):
            raise ValueError("sync history pagination is invalid")
        providers = tuple(source.provider for source in sources)
        if len(set(providers)) != len(providers):
            raise ValueError("data sources must have unique providers")
        if providers != tuple(sorted(providers)):
            raise ValueError("data sources must be sorted by provider")
        if self.active_sync is not None:
            if self.active_sync.heavy is not True:
                raise ValueError("active sync must be a heavy task")
            if self.active_sync.name not in {f"sync:{provider}" for provider in providers}:
                raise ValueError("active sync must correspond to a configured provider")
        watchlist_instruments = tuple(self.watchlist_instruments)
        if any(
            type(item) is not InstrumentId for item in watchlist_instruments
        ) or watchlist_instruments != tuple(sorted(set(watchlist_instruments), key=str)):
            raise ValueError("watchlist instruments must be unique and sorted")
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "previews", previews)
        object.__setattr__(self, "sync_history", history)
        object.__setattr__(self, "watchlist_instruments", watchlist_instruments)

    @property
    def missing_instruments(self) -> tuple[InstrumentId, ...]:
        ready = {preview.instrument for preview in self.previews}
        return tuple(
            instrument for instrument in self.watchlist_instruments if instrument not in ready
        )


class DataGateway(Protocol):
    def sources(self) -> Sequence[DataSourceSnapshot]: ...

    def sync(
        self,
        provider: str,
        start: date | None = None,
        end: date | None = None,
        *,
        instruments: Sequence[InstrumentId] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> None: ...

    def latest_bundle_manifest_id(self) -> str | None: ...

    def latest_market_previews(self) -> Sequence[MarketDataPreview]: ...

    def watchlist_instruments(self) -> Sequence[InstrumentId]: ...

    def sync_history(self, page: int, page_size: int) -> DataSyncHistoryPage: ...

    def clear_sync_history(self) -> int: ...

    def clear_market_data(self, instrument: InstrumentId | None = None) -> int: ...


class TaskGateway(Protocol):
    def submit(self, name: str, heavy: bool, operation: Operation) -> TaskSnapshot: ...

    def status(self, task_id: str) -> TaskSnapshot: ...

    def cancel(self, task_id: str) -> TaskSnapshot: ...


class DataPageModel:
    def __init__(self, gateway: DataGateway, tasks: TaskGateway) -> None:
        self._gateway = gateway
        self._tasks = tasks
        self._active_sync_id: str | None = None
        self._active_sync_provider: str | None = None
        self._active_sync_cancel: Event | None = None
        self._history_page = 1
        self._history_page_size = 10
        self._lock = RLock()

    def state(self) -> DataPageState:
        sources = _boundary_call(
            "DATA_STATUS_UNAVAILABLE",
            lambda: tuple(self._gateway.sources()),
        )
        active = _boundary_call(
            "DATA_TASK_STATUS_UNAVAILABLE",
            self._active_sync_snapshot,
        )
        bundle_manifest = _boundary_call(
            "DATA_STATUS_UNAVAILABLE",
            self._latest_bundle_manifest_id,
        )
        previews = _boundary_call(
            "DATA_PREVIEW_UNAVAILABLE",
            self._latest_market_previews,
        )
        history_page = _boundary_call(
            "DATA_SYNC_HISTORY_UNAVAILABLE",
            self._sync_history,
        )
        watchlist_instruments = _boundary_call(
            "DATA_WATCHLIST_UNAVAILABLE",
            self._watchlist_instruments,
        )
        return _boundary_call(
            "DATA_STATUS_UNAVAILABLE",
            lambda: DataPageState(
                sources,
                active,
                bundle_manifest,
                previews,
                history_page.entries,
                history_page.page,
                history_page.total_pages,
                history_page.total_items,
                watchlist_instruments,
            ),
        )

    def _latest_bundle_manifest_id(self) -> str | None:
        reader = getattr(self._gateway, "latest_bundle_manifest_id", None)
        if reader is None:
            return None
        if not callable(reader):
            raise TypeError("data gateway bundle anchor must be callable")
        value = reader()
        return (
            None
            if value is None
            else safe_identifier(value, label="latest dataset bundle manifest id")
        )

    def _latest_market_previews(self) -> tuple[MarketDataPreview, ...]:
        reader = getattr(self._gateway, "latest_market_previews", None)
        if reader is None:
            return ()
        if not callable(reader):
            raise TypeError("data gateway market preview reader must be callable")
        return tuple(reader())

    def _watchlist_instruments(self) -> tuple[InstrumentId, ...]:
        reader = getattr(self._gateway, "watchlist_instruments", None)
        if reader is None:
            return tuple(preview.instrument for preview in self._latest_market_previews())
        if not callable(reader):
            raise TypeError("data gateway watchlist reader must be callable")
        return tuple(reader())

    def _sync_history(self) -> DataSyncHistoryPage:
        reader = getattr(self._gateway, "sync_history", None)
        if reader is None:
            return DataSyncHistoryPage((), 1, self._history_page_size, 0)
        if not callable(reader):
            raise TypeError("data gateway sync history reader must be callable")
        with self._lock:
            page = self._history_page
        result = reader(page, self._history_page_size)
        if type(result) is not DataSyncHistoryPage:
            raise TypeError("data gateway sync history page is invalid")
        with self._lock:
            self._history_page = result.page
        return result

    def _active_sync_snapshot(self) -> TaskSnapshot | None:
        with self._lock:
            task_id = self._active_sync_id
            provider = self._active_sync_provider
        if task_id is None or provider is None:
            return None
        task = _sync_task_snapshot(
            self._tasks.status(task_id),
            provider=provider,
            task_id=task_id,
        )
        if task.status in _TERMINAL_SYNC_STATUSES:
            with self._lock:
                if (
                    self._active_sync_id == task_id
                    and self._active_sync_provider == provider
                ):
                    self._active_sync_id = None
                    self._active_sync_provider = None
                    self._active_sync_cancel = None
            return None
        return task

    def set_history_page(self, page: int) -> None:
        if type(page) is not int or page <= 0:
            raise ValueError("sync history page must be positive")
        with self._lock:
            self._history_page = page

    def clear_sync_history(self) -> int:
        deleted = _boundary_call(
            "DATA_SYNC_HISTORY_CLEAR_FAILED",
            self._gateway.clear_sync_history,
        )
        if type(deleted) is not int or deleted < 0:
            raise TypeError("data gateway history clear result is invalid")
        with self._lock:
            self._history_page = 1
        return deleted

    def clear_market_data(self, instrument: InstrumentId | None = None) -> int:
        if instrument is not None and type(instrument) is not InstrumentId:
            raise TypeError("instrument must be an exact InstrumentId or None")
        deleted = _boundary_call(
            "MARKET_DATA_CLEAR_FAILED",
            lambda: self._gateway.clear_market_data(instrument),
        )
        if type(deleted) is not int or deleted < 0:
            raise TypeError("data gateway market clear result is invalid")
        return deleted

    def start_sync(
        self,
        provider: str,
        date_range: DataSyncRange | None = None,
        instruments: Sequence[InstrumentId] | None = None,
    ) -> TaskSnapshot:
        stable_provider = safe_identifier(provider, label="provider")
        if date_range is not None and type(date_range) is not DataSyncRange:
            raise TypeError("date_range must be an exact DataSyncRange or None")
        targets = None if instruments is None else tuple(instruments)
        if targets is not None and (
            not targets
            or any(type(item) is not InstrumentId for item in targets)
            or targets != tuple(sorted(set(targets), key=str))
        ):
            raise ValueError("SYNC_TARGET_INVALID")
        active = _boundary_call(
            "DATA_TASK_STATUS_UNAVAILABLE",
            self._active_sync_snapshot,
        )
        if active is not None:
            raise ValueError("DATA_SYNC_ALREADY_ACTIVE")
        if targets is not None:
            watchlist = _boundary_call(
                "DATA_WATCHLIST_UNAVAILABLE",
                self._watchlist_instruments,
            )
            if not set(targets).issubset(watchlist):
                raise ValueError("SYNC_TARGET_INVALID")
        configured = _boundary_call(
            "DATA_STATUS_UNAVAILABLE",
            lambda: _source_availability(self._gateway.sources()),
        )
        if not configured.get(stable_provider, False):
            raise ValueError("DATA_SOURCE_UNAVAILABLE")
        cancel_event = Event()
        task = _boundary_call(
            "DATA_SYNC_SUBMISSION_FAILED",
            lambda: _sync_task_snapshot(
                self._tasks.submit(
                    f"sync:{stable_provider}",
                    True,
                    lambda: (
                        self._gateway.sync(
                            stable_provider,
                            instruments=targets,
                            cancel_requested=cancel_event.is_set,
                        )
                        if date_range is None
                        else self._gateway.sync(
                            stable_provider,
                            date_range.start,
                            date_range.end,
                            instruments=targets,
                            cancel_requested=cancel_event.is_set,
                        )
                    ),
                ),
                provider=stable_provider,
            ),
        )
        with self._lock:
            self._active_sync_id = task.task_id
            self._active_sync_provider = stable_provider
            self._active_sync_cancel = cancel_event
            self._history_page = 1
        return task

    def stop_sync(self) -> TaskSnapshot:
        with self._lock:
            task_id = self._active_sync_id
            provider = self._active_sync_provider
            cancel_event = self._active_sync_cancel
        if task_id is None or provider is None or cancel_event is None:
            raise ValueError("DATA_SYNC_NOT_ACTIVE")
        active = _boundary_call(
            "DATA_TASK_STATUS_UNAVAILABLE",
            self._active_sync_snapshot,
        )
        if active is None:
            raise ValueError("DATA_SYNC_NOT_ACTIVE")
        cancel_event.set()
        return _boundary_call(
            "DATA_SYNC_CANCELLATION_FAILED",
            lambda: _sync_task_snapshot(
                self._tasks.cancel(task_id),
                provider=provider,
                task_id=task_id,
            ),
        )

    def acknowledge_terminal_sync(self, task: TaskSnapshot) -> None:
        if type(task) is not TaskSnapshot or task.status not in _TERMINAL_SYNC_STATUSES:
            raise ValueError("only an exact terminal sync task can be acknowledged")
        with self._lock:
            if (
                self._active_sync_id != task.task_id
                or self._active_sync_provider is None
                or task.name != f"sync:{self._active_sync_provider}"
            ):
                raise ValueError("terminal sync task does not match the active sync")
            self._active_sync_id = None
            self._active_sync_provider = None
            self._active_sync_cancel = None


def render_data_page(model: DataPageModel | None) -> None:
    if model is None:
        ui.label("行情服务未配置；当前不会虚构可用来源、同步结果或缓存状态。").classes(
            "text-sm text-slate-600"
        )
        return
    selected_preview_instrument: str | None = None
    selected_source_provider: str | None = None
    selected_single_instrument: str | None = None
    poll_timer: Timer
    selected_range_mode = "2y"
    initial_range = DataSyncRangeForm(selected_range_mode).validate().date_range
    assert initial_range is not None
    selected_start_text = initial_range.start.isoformat()
    selected_end_text = initial_range.end.isoformat()

    @ui.refreshable
    def content() -> None:
        nonlocal selected_preview_instrument, selected_source_provider
        nonlocal selected_single_instrument
        nonlocal selected_range_mode
        nonlocal selected_start_text, selected_end_text
        try:
            state = model.state()
        except Exception:
            ui.label("行情状态读取失败，请查看本地日志。").classes("text-red-700")
            return
        source_names = {source.provider: source.source_name for source in state.sources}
        missing_instruments = state.missing_instruments

        with ui.row().classes("w-full gap-4 rounded border border-slate-200 bg-white px-4 py-3"):
            ui.label(f"关注标的 {len(state.watchlist_instruments)} 个").classes(
                "text-sm font-medium text-slate-700"
            )
            ui.label(f"数据就绪 {len(state.previews)} 个").classes(
                "text-sm font-medium text-emerald-700"
            )
            ui.label(f"缺失 {len(missing_instruments)} 个").classes(
                "text-sm font-medium "
                + ("text-amber-700" if missing_instruments else "text-slate-500")
            )

        def source_display_name(provider: str) -> str:
            return source_names.get(provider, "未知来源")

        def request_clear_market_data(instrument: InstrumentId | None) -> None:
            target = "全部标的" if instrument is None else str(instrument)
            with ui.dialog() as dialog, ui.card():
                ui.label(f"确认清理{target}的行情数据？").classes("font-semibold")
                ui.label("此操作会删除对应的当前本地行情数据和预览，无法从页面恢复。").classes(
                    "text-sm text-red-700"
                )

                def confirm() -> None:
                    nonlocal selected_preview_instrument
                    try:
                        deleted = model.clear_market_data(instrument)
                    except Exception:
                        ui.notify("行情数据清理失败；同步进行中时不能清理。", type="negative")
                    else:
                        if instrument is None or selected_preview_instrument == str(instrument):
                            selected_preview_instrument = None
                        ui.notify(f"已清理 {deleted} 份行情数据。", type="positive")
                    dialog.close()
                    content.refresh()

                with ui.row().classes("justify-end gap-2"):
                    ui.button("取消", on_click=dialog.close).props("flat")
                    ui.button("确认清理", on_click=confirm, color="negative")
            dialog.open()

        if state.active_sync is not None:
            with ui.row().classes(
                "w-full bg-slate-100 rounded px-3 py-2 items-center justify-between"
            ):
                ui.label(
                    "同步任务："
                    f"{source_display_name(state.active_sync.name.removeprefix('sync:'))} / "
                    f"{state.active_sync.status.value}"
                ).classes("text-sm")

                def stop_sync() -> None:
                    try:
                        model.stop_sync()
                    except Exception:
                        ui.notify("停止请求提交失败，请稍后重试。", type="negative")
                    else:
                        ui.notify(
                            "已请求停止；当前网络请求返回后任务会安全结束。",
                            type="warning",
                        )
                    content.refresh()

                stop_button = ui.button(
                    "停止任务",
                    on_click=stop_sync,
                    icon="stop_circle",
                ).props("outline color=negative aria-label=停止行情同步任务")
                if state.active_sync.status is TaskStatus.CANCELLATION_REQUESTED:
                    stop_button.disable()
            if state.active_sync.status is TaskStatus.CANCELLATION_REQUESTED:
                ui.label("正在停止：不会再开始新的标的请求，当前网络请求返回后结束。").classes(
                    "text-sm text-amber-800"
                )
            running_history = next(
                (
                    item
                    for item in state.sync_history
                    if item.status is TaskStatus.RUNNING
                    and item.provider == state.active_sync.name.removeprefix("sync:")
                ),
                None,
            )
            if (
                running_history is not None
                and running_history.instrument_count is not None
                and running_history.completed_instrument_count is not None
            ):
                completed = running_history.completed_instrument_count
                total = running_history.instrument_count
                ui.label(f"获取进度：{completed}/{total} 个标的已完成").classes(
                    "text-sm font-medium text-blue-700"
                )
                ui.linear_progress(value=completed / total).classes("w-full")
            if state.active_sync.failure is not None:
                ui.label(sync_failure_text(state.active_sync.failure)).classes(
                    "text-sm text-red-700"
                )
        if state.latest_bundle_manifest_id is not None:
            latest_day = max(
                (preview.last_day for preview in state.previews),
                default="暂无数据",
            )
            short_version = state.latest_bundle_manifest_id[:10]
            with ui.column().classes("w-full gap-1 bg-slate-100 rounded px-3 py-2"):
                ui.label(
                    f"当前行情数据：{len(state.previews)} 个标的，"
                    f"更新至 {latest_day} · 数据版本 {short_version}"
                ).classes("text-sm font-medium")
                ui.label(
                    "增量同步成功后只保留每个标的的最新行情；失败时保留当前可用数据。"
                ).classes("text-xs text-slate-600")
                current_gap_count = sum(
                    preview.internal_missing_session_count > 0
                    or preview.trailing_missing_session_count > 0
                    for preview in state.previews
                )
                history_limited_count = sum(
                    preview.leading_missing_session_count > 0
                    and preview.internal_missing_session_count == 0
                    and preview.trailing_missing_session_count == 0
                    for preview in state.previews
                )
                if state.previews and current_gap_count == 0:
                    ui.label("当前已写入数据区间内交易日连续，数据可用于检查和后续分析。").classes(
                        "text-xs text-emerald-700"
                    )
                if history_limited_count:
                    ui.label(
                        f"其中 {history_limited_count} 个标的只是较早历史覆盖不足，"
                        "不代表现有数据区间内部断档。"
                    ).classes("text-xs text-amber-800")
                if current_gap_count:
                    ui.label(
                        f"其中 {current_gap_count} 个标的在现有区间内或最新端仍有缺口，"
                        "建议重新同步后再使用。"
                    ).classes("text-xs text-red-700")
                with ui.expansion("查看完整版本校验码").classes("text-xs text-slate-500"):
                    ui.label(state.latest_bundle_manifest_id).classes(
                        "font-mono break-all select-all"
                    )
        if state.previews:
            with ui.card().classes("w-full border border-slate-200 shadow-none"):
                ui.label("最新行情数据预览").classes("text-lg font-semibold")
                ui.label(
                    "以下图表直接读取上方清单中的不可变日线数据，"
                    "可直接核对价格、成交量、日期范围和数据缺口。"
                ).classes("text-sm text-slate-600")
                choices = {
                    str(preview.instrument): preview.display_label for preview in state.previews
                }
                if selected_preview_instrument not in choices:
                    selected_preview_instrument = str(state.previews[0].instrument)

                def select_preview(event: object) -> None:
                    nonlocal selected_preview_instrument
                    selected = getattr(event, "value", None)
                    if type(selected) is str and selected in choices:
                        selected_preview_instrument = selected
                    preview_panel.refresh()

                selector = ui.select(
                    choices,
                    label="预览标的",
                    value=selected_preview_instrument,
                    on_change=select_preview,
                ).classes("w-full max-w-sm")

                @ui.refreshable
                def preview_panel() -> None:
                    selected = selector.value
                    preview = next(
                        (item for item in state.previews if str(item.instrument) == selected),
                        state.previews[0],
                    )
                    latest = preview.bars[-1]
                    with ui.row().classes("w-full gap-6 items-start"):
                        with ui.column().classes("gap-1"):
                            ui.label(f"标的：{preview.display_label}").classes("font-semibold")
                            ui.label(f"实际来源：{source_display_name(preview.provider)}").classes(
                                "text-sm text-slate-600"
                            )
                            ui.label(
                                f"完整区间：{preview.first_day} 至 {preview.last_day}"
                            ).classes("text-sm text-slate-600")
                            ui.label(
                                f"完整数据：{preview.total_rows} 行；"
                                f"图中加载最近 {len(preview.bars)} 行"
                            ).classes("text-sm text-slate-600")
                        with ui.column().classes("gap-1"):
                            ui.label(f"最新日线：{latest.day} / 收盘 {latest.close:.4f}").classes(
                                "font-semibold"
                            )
                            ui.label(
                                f"开 {latest.open:.4f} / 高 {latest.high:.4f} / 低 {latest.low:.4f}"
                            ).classes("text-sm text-slate-600")
                            ui.label(f"成交量：{latest.volume:,.0f}").classes(
                                "text-sm text-slate-600"
                            )
                            if preview.daily_complete is None:
                                completeness_text = "交易日完整性：历史数据未记录"
                                completeness_class = "text-sm text-amber-800"
                            elif (
                                preview.internal_missing_session_count == 0
                                and preview.trailing_missing_session_count == 0
                            ):
                                completeness_text = (
                                    f"数据已写入：{preview.first_day} 至 "
                                    f"{preview.last_day} 交易日连续"
                                )
                                completeness_class = "text-sm text-emerald-700"
                            else:
                                completeness_text = (
                                    f"缺少交易日：{preview.missing_session_count} 个"
                                )
                                completeness_class = "text-sm text-amber-800"
                            ui.label(completeness_text).classes(completeness_class)
                            if preview.leading_missing_session_count:
                                ui.label(
                                    f"较早历史未覆盖："
                                    f"{preview.leading_missing_session_count} 个交易日"
                                ).classes("text-xs text-amber-800")
                    ui.label(f"标的数据版本：{preview.manifest_id[:10]}（用于数据核对）").classes(
                        "text-xs text-slate-500"
                    )
                    ui.button(
                        "清理该标的行情",
                        on_click=lambda item=preview.instrument: request_clear_market_data(item),
                        icon="delete_outline",
                    ).props("outline color=negative aria-label=清理当前标的行情")
                    ui.echart(thaw_chart_options(market_data_chart_options(preview.bars))).classes(
                        "w-full h-[30rem]"
                    )
                    ui.label("最近 5 条日线").classes("font-semibold")
                    for bar in preview.bars[-5:]:
                        ui.label(
                            f"{bar.day}　开 {bar.open:.4f}　高 {bar.high:.4f}　"
                            f"低 {bar.low:.4f}　收 {bar.close:.4f}　"
                            f"量 {bar.volume:,.0f}"
                        ).classes("text-xs text-slate-600 font-mono")

                preview_panel()

        with ui.card().classes("w-full border border-slate-200 shadow-none"):
            ui.label("获取行情区间").classes("text-lg font-semibold")
            ui.label(
                "快捷区间按今天向前计算；本地已有数据默认可信，"
                "同步时只获取缺失交易日，并自动收敛到最近已完成交易日。"
            ).classes("text-sm text-slate-600")
            with ui.row().classes("w-full gap-3 items-end flex-wrap"):

                def change_range_mode(event: object) -> None:
                    nonlocal selected_range_mode, selected_start_text, selected_end_text
                    value = getattr(event, "value", None)
                    if type(value) is not str or value not in {*_SYNC_RANGE_YEARS, "custom"}:
                        return
                    selected_range_mode = value
                    if value in _SYNC_RANGE_YEARS:
                        resolved = DataSyncRangeForm(value).validate().date_range
                        assert resolved is not None
                        selected_start_text = resolved.start.isoformat()
                        selected_end_text = resolved.end.isoformat()
                        start_input.value = selected_start_text
                        end_input.value = selected_end_text
                        start_input.disable()
                        end_input.disable()
                    else:
                        start_input.enable()
                        end_input.enable()
                    range_errors.clear()

                range_mode = ui.toggle(
                    {
                        "2y": "近 2 年",
                        "5y": "近 5 年",
                        "10y": "近 10 年",
                        "custom": "自定义",
                    },
                    value=selected_range_mode,
                    on_change=change_range_mode,
                ).props("aria-label=行情区间方式")

                def change_start(event: object) -> None:
                    nonlocal selected_start_text
                    value = getattr(event, "value", None)
                    if type(value) is str:
                        selected_start_text = value
                    range_errors.clear()

                def change_end(event: object) -> None:
                    nonlocal selected_end_text
                    value = getattr(event, "value", None)
                    if type(value) is str:
                        selected_end_text = value
                    range_errors.clear()

                start_input = ui.input(
                    "开始日期",
                    value=selected_start_text,
                    on_change=change_start,
                ).props("type=date aria-label=行情开始日期")
                end_input = ui.input(
                    "结束日期",
                    value=selected_end_text,
                    on_change=change_end,
                ).props("type=date aria-label=行情结束日期")
                if selected_range_mode != "custom":
                    start_input.disable()
                    end_input.disable()
                del range_mode
            range_errors = ui.column().classes("w-full gap-1")

        if not state.sources:
            ui.label("尚未配置数据来源。").classes("text-sm text-slate-600")
        else:
            ui.label(
                "同步范围来自唯一的“关注标的”池。所选数据源优先，失败时按固定顺序"
                "尝试其余可用来源。"
            ).classes("text-sm text-slate-600")
            source_choices = {source.provider: source.source_name for source in state.sources}
            if selected_source_provider not in source_choices:
                selected_source_provider = (
                    "tencent" if "tencent" in source_choices else state.sources[0].provider
                )

            def select_source(event: object) -> None:
                nonlocal selected_source_provider
                value = getattr(event, "value", None)
                if type(value) is str and value in source_choices:
                    selected_source_provider = value
                    content.refresh()

            ui.select(
                source_choices,
                label="行情数据源",
                value=selected_source_provider,
                on_change=select_source,
            ).props("aria-label=行情数据源").classes("w-full max-w-sm")
            source = next(
                item for item in state.sources if item.provider == selected_source_provider
            )
            with ui.card().classes("w-full border border-slate-200 shadow-none"):
                with ui.row().classes("w-full justify-between items-start"):
                    with ui.column().classes("gap-1"):
                        with ui.row().classes("items-center gap-2"):
                            ui.label(source.source_name).classes("font-semibold")
                            if source.provider == "tencent":
                                ui.badge("推荐主源", color="positive")
                            elif source.provider == "akshare":
                                ui.badge("HTTP 备用源", color="blue-grey")
                            elif source.provider == "baostock":
                                ui.badge("备用源", color="grey")
                        if source.provider == "akshare":
                            ui.label("接口方式：AKShare 适配东方财富").classes(
                                "text-xs text-slate-500"
                            )
                        elif source.provider == "tencent":
                            ui.label("接口方式：腾讯证券历史行情").classes(
                                "text-xs text-slate-500"
                            )
                        ui.label("可用" if source.available else "不可用").classes(
                            "text-sm text-emerald-700"
                            if source.available
                            else "text-sm text-red-700"
                        )
                    with ui.column().classes("gap-1 text-right"):
                        ui.label(
                            "最后更新："
                            + (
                                source.last_update.isoformat(timespec="seconds")
                                if source.last_update
                                else "暂无"
                            )
                        ).classes("text-xs text-slate-600")
                        ui.label(
                            "最新数据源："
                            + (
                                "暂无"
                                if source.latest_source is None
                                else source_display_name(source.latest_source)
                            )
                        ).classes("text-xs text-slate-600")
                        ui.label(f"缓存：{source.cache_bytes} 字节").classes(
                            "text-xs text-slate-600"
                        )
                if source.quality is None:
                    ui.label("质量摘要：暂无").classes("text-xs text-slate-600")
                else:
                    ui.label(
                        "质量摘要："
                        f"{'通过' if source.quality.accepted else '未通过'} / "
                        f"{source.quality.mode} / "
                        f"{source.quality.output_rows}/{source.quality.input_rows} 行"
                    ).classes("text-sm")
                    if source.quality.issue_codes:
                        ui.label("质量问题：" + "、".join(source.quality.issue_codes)).classes(
                            "text-xs text-amber-800"
                        )

                def sync(
                    targets: tuple[InstrumentId, ...] | None = None,
                    *,
                    only_missing: bool = False,
                ) -> None:
                    validation = DataSyncRangeForm(
                        selected_range_mode,
                        selected_start_text,
                        selected_end_text,
                    ).validate()
                    range_errors.clear()
                    if validation.errors:
                        with range_errors:
                            for message in validation.errors.values():
                                ui.label(message).classes("text-sm text-red-700")
                        return
                    assert validation.date_range is not None
                    if only_missing:
                        try:
                            targets = model.state().missing_instruments
                        except Exception:
                            ui.notify("无法读取最新标的池，请刷新页面后重试。", type="negative")
                            return
                        if not targets:
                            ui.notify("当前没有缺失行情的标的。", type="info")
                            content.refresh()
                            return
                    try:
                        model.start_sync(
                            source.provider,
                            validation.date_range,
                            targets,
                        )
                    except Exception as error:
                        error_code = getattr(error, "code", None)
                        if error_code == "SYNC_TARGET_INVALID" or (
                            isinstance(error, ValueError)
                            and str(error) == "SYNC_TARGET_INVALID"
                        ):
                            ui.notify(
                                "标的池已经变化，已刷新页面，请重新选择后同步。",
                                type="warning",
                            )
                            content.refresh()
                        elif error_code == "DATA_SYNC_SUBMISSION_FAILED" or (
                            isinstance(error, ValueError)
                            and str(error) == "DATA_SYNC_ALREADY_ACTIVE"
                        ):
                            ui.notify("已有行情任务进行中，请稍后重试。", type="warning")
                            content.refresh()
                        else:
                            ui.notify(
                                "同步未启动，请检查任务状态或本地日志。",
                                type="negative",
                            )
                    else:
                        poll_timer.activate()
                        content.refresh()

                with ui.row().classes("w-full gap-2 items-center flex-wrap"):
                    all_button = ui.button(
                        "增量同步全部",
                        on_click=lambda: sync(),
                        icon="sync",
                    ).props(f"aria-label=同步全部{source.source_name}")
                    missing_button = ui.button(
                        "只同步缺失标的",
                        on_click=lambda: sync(only_missing=True),
                        icon="playlist_add_check",
                    ).props(f"outline aria-label=同步缺失标的{source.source_name}")
                    if not missing_instruments:
                        missing_button.disable()
                individual_options = {
                    str(instrument): (
                        f"{common_instrument_name(instrument) or '名称待同步'}（{instrument}）"
                    )
                    for instrument in state.watchlist_instruments
                }
                if selected_single_instrument not in individual_options:
                    selected_single_instrument = next(iter(individual_options), None)
                with ui.row().classes("w-full gap-2 items-end flex-wrap"):
                    individual_select = (
                        ui.select(
                            individual_options,
                            label="单个标的重试",
                            value=selected_single_instrument,
                        )
                        .props("aria-label=单个标的重试")
                        .classes("w-full max-w-sm")
                    )

                    def retry_one() -> None:
                        nonlocal selected_single_instrument
                        selected = individual_select.value
                        if type(selected) is not str:
                            ui.notify("请先选择一个标的。", type="warning")
                            return
                        selected_single_instrument = selected
                        sync((InstrumentId.parse(selected),))

                    retry_button = ui.button(
                        "重试该标的",
                        on_click=retry_one,
                        icon="refresh",
                    ).props("outline aria-label=重试单个标的")
                if not source.available or state.active_sync is not None:
                    all_button.disable()
                    missing_button.disable()
                    retry_button.disable()
                    individual_select.disable()

        with ui.card().classes("w-full border border-slate-200 shadow-none"):
            with ui.row().classes("w-full justify-between items-center"):
                ui.label("行情获取任务历史").classes("text-lg font-semibold")

                def request_clear_history() -> None:
                    with ui.dialog() as dialog, ui.card():
                        ui.label("清除行情任务历史？").classes("font-semibold")
                        ui.label("只清除已结束的记录，不会删除行情数据。")

                        def confirm() -> None:
                            try:
                                deleted = model.clear_sync_history()
                            except Exception:
                                ui.notify("任务历史清除失败。", type="negative")
                            else:
                                ui.notify(f"已清除 {deleted} 条任务历史。", type="positive")
                            dialog.close()
                            content.refresh()

                        with ui.row().classes("justify-end gap-2"):
                            ui.button("取消", on_click=dialog.close).props("flat")
                            ui.button("确认清除", on_click=confirm, color="negative")
                    dialog.open()

                ui.button("清除历史", on_click=request_clear_history, icon="delete_sweep").props(
                    "outline color=negative aria-label=清除行情任务历史"
                )
            ui.label(
                "记录每次获取的来源、请求区间、起止时间和安全错误码，便于重启后继续排查。"
            ).classes("text-sm text-slate-600")
            if not state.sync_history:
                ui.label("暂无历史记录；下一次获取行情后会显示在这里。").classes(
                    "text-sm text-slate-500"
                )
            for history_index, item in enumerate(state.sync_history):
                status_text = {
                    TaskStatus.RUNNING: "进行中",
                    TaskStatus.CANCELLED: "已停止",
                    TaskStatus.SUCCEEDED: "成功",
                    TaskStatus.FAILED: "失败",
                }[item.status]
                status_class = {
                    TaskStatus.RUNNING: "text-blue-700",
                    TaskStatus.CANCELLED: "text-slate-600",
                    TaskStatus.SUCCEEDED: "text-emerald-700",
                    TaskStatus.FAILED: "text-red-700",
                }[item.status]
                with ui.row().classes("w-full gap-4 items-start border-t border-slate-100 pt-2"):
                    ui.label(f"#{item.run_id}").classes("text-xs text-slate-500 font-mono")
                    with ui.column().classes("gap-1 grow"):
                        ui.label(
                            f"{source_display_name(item.provider)}　"
                            f"{item.date_range.start.isoformat()} 至 "
                            f"{item.date_range.end.isoformat()}"
                        ).classes("text-sm font-medium")
                        ui.label("开始：" + item.started_at.isoformat(timespec="seconds")).classes(
                            "text-xs text-slate-500"
                        )
                        if item.completed_at is not None:
                            ui.label(
                                "结束：" + item.completed_at.isoformat(timespec="seconds")
                            ).classes("text-xs text-slate-500")
                        if (
                            item.instrument_count is not None
                            and item.completed_instrument_count is not None
                        ):
                            ui.label(
                                f"获取进度：{item.completed_instrument_count}/"
                                f"{item.instrument_count} 个标的"
                            ).classes("text-xs text-blue-700")
                            if (
                                item.status is TaskStatus.FAILED
                                and item.completed_instrument_count > 0
                            ):
                                ui.label(
                                    "已完成标的的数据已保存；失败标的继续使用原有可信数据。"
                                ).classes("text-xs text-amber-800")
                        if item.failure_code is not None:
                            guidance = _SYNC_FAILURE_GUIDANCE.get(item.failure_code)
                            detail = (
                                item.failure_code
                                if guidance is None
                                else f"{item.failure_code}：{guidance}"
                            )
                            ui.label(detail).classes("text-xs text-red-700")
                        for failure in item.instrument_failures:
                            instrument_name = (
                                common_instrument_name(failure.instrument) or "名称待同步"
                            )
                            ui.label(
                                f"失败标的：{instrument_name}（{failure.instrument}） · "
                                f"{failure.failure_code}"
                            ).classes("text-xs font-medium text-red-700")
                            for issue_code in failure.quality_issue_codes:
                                issue_guidance = _QUALITY_ISSUE_GUIDANCE.get(
                                    issue_code, "未识别的质量规则"
                                )
                                ui.label(f"质量问题：{issue_code}（{issue_guidance}）").classes(
                                    "text-xs text-amber-800"
                                )
                            if failure.input_rows is not None:
                                ui.label(
                                    f"质量检查行数：输入 {failure.input_rows:,} 行，"
                                    f"可用 {failure.output_rows or 0:,} 行。"
                                ).classes("text-xs text-slate-500")
                        if item.status is TaskStatus.SUCCEEDED and item.downloaded_rows is not None:
                            if item.downloaded_rows > 0:
                                ui.label(
                                    f"已成功获取 {item.downloaded_rows:,} 行新数据；"
                                    f"复用本地可信数据 {item.reused_rows or 0:,} 行。"
                                ).classes("text-xs text-emerald-700")
                            else:
                                ui.label(
                                    f"无需重复下载；已复用本地可信数据 "
                                    f"{item.reused_rows or 0:,} 行。"
                                ).classes("text-xs text-emerald-700")
                            coverage_text, coverage_class = sync_coverage_notice(
                                item,
                                state.previews,
                                use_current_previews=(
                                    state.sync_history_page == 1 and history_index == 0
                                ),
                            )
                            ui.label(coverage_text).classes(coverage_class)
                    ui.label(status_text).classes(f"text-sm font-semibold {status_class}")
            with ui.row().classes("w-full justify-between items-center pt-2"):
                ui.label(
                    f"第 {state.sync_history_page}/{state.sync_history_total_pages} 页 · "
                    f"共 {state.sync_history_total_items} 条"
                ).classes("text-xs text-slate-500")

                def change_history_page(page: int) -> None:
                    model.set_history_page(page)
                    content.refresh()

                with ui.row().classes("gap-2"):
                    previous = ui.button(
                        "上一页",
                        on_click=lambda: change_history_page(state.sync_history_page - 1),
                    ).props("flat dense")
                    following = ui.button(
                        "下一页",
                        on_click=lambda: change_history_page(state.sync_history_page + 1),
                    ).props("flat dense")
                    if state.sync_history_page <= 1:
                        previous.disable()
                    if state.sync_history_page >= state.sync_history_total_pages:
                        following.disable()

        if state.previews:
            with ui.card().classes("w-full border border-red-200 shadow-none"):
                ui.label("数据清理").classes("text-lg font-semibold text-red-800")
                ui.label("可在上方预览中清理单个标的；这里可以清理全部标的的本地行情。").classes(
                    "text-sm text-slate-600"
                )
                ui.button(
                    "清理全部标的行情",
                    on_click=lambda: request_clear_market_data(None),
                    icon="delete_forever",
                ).props("color=negative aria-label=清理全部标的行情")

    def poll_sync() -> None:
        try:
            active_sync = model.state().active_sync
        except Exception:
            active_sync = None
        if active_sync is not None and active_sync.status in _TERMINAL_SYNC_STATUSES:
            try:
                model.acknowledge_terminal_sync(active_sync)
            except Exception:
                pass
            content.refresh()
            if active_sync.status is TaskStatus.SUCCEEDED:
                try:
                    missing_count = sum(
                        preview.missing_session_count > 0 for preview in model.state().previews
                    )
                except Exception:
                    missing_count = 0
                message = "同步完成，最新行情数据预览已刷新。"
                if missing_count:
                    message += f"仍有 {missing_count} 个标的存在历史缺口。"
                ui.notify(message, type="positive")
            elif active_sync.status is TaskStatus.FAILED:
                ui.notify(
                    "同步部分失败；成功获取的标的已保存，失败标的保留原有可信数据。",
                    type="negative",
                )
            elif active_sync.status is TaskStatus.CANCELLED:
                ui.notify(
                    "同步已停止；停止前成功获取的标的已保存。",
                    type="warning",
                )
            poll_timer.deactivate()
            return
        content.refresh()
        if active_sync is None:
            poll_timer.deactivate()

    poll_timer = ui.timer(1.0, poll_sync, active=False)
    content()
