from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
import json
from threading import RLock
from typing import Protocol

from nicegui import ui
from nicegui.elements.timer import Timer

from compass.backtest.engine import ExecutionTiming
from compass.backtest.orders import OrderStatus
from compass.domain.market import AssetType, InstrumentId
from compass.services.instrument_names import common_index_etf_pairs, common_instrument_name
from compass.services.safe_display import safe_display_text, safe_identifier, stable_code
from compass.services.task_manager import Operation, TaskSnapshot, TaskStatus
from compass.strategies.rule_dsl import DslVariable, RuleDslParameters
from compass.strategies.kronos_forecast import KronosForecastParameters, kronos_runtime_status
from compass.ui.components.charts import CurvePoint, equity_chart_options, thaw_chart_options
from compass.ui.pages.backtests import BacktestReport
from compass.ui.task_status import task_status_label


class StrategyLabKind(StrEnum):
    BUY_AND_HOLD = "buy_and_hold"
    DUAL_MA = "dual_ma"
    RULE_DSL = "rule_dsl"
    KRONOS_FORECAST = "kronos_forecast"


class StrategyLabRebalanceMode(StrEnum):
    SIGNAL_CHANGE = "signal_change"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    DAILY = "daily"


_TERMINAL_STATUSES = {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED}


@dataclass(frozen=True, slots=True)
class StrategyLabInstrument:
    instrument: InstrumentId
    name: str
    asset_type: AssetType
    first_day: date
    last_day: date
    rows: int

    def __post_init__(self) -> None:
        if type(self.instrument) is not InstrumentId:
            raise TypeError("instrument must be an exact InstrumentId")
        if type(self.name) is not str or not self.name.strip():
            raise ValueError("instrument name must be non-empty")
        if type(self.asset_type) is not AssetType:
            raise TypeError("asset_type must be an exact AssetType")
        if type(self.first_day) is not date or type(self.last_day) is not date:
            raise TypeError("instrument range must use exact dates")
        if self.first_day > self.last_day:
            raise ValueError("instrument range is reversed")
        if type(self.rows) is not int or self.rows <= 0:
            raise ValueError("instrument rows must be a positive exact integer")

    @property
    def tradable(self) -> bool:
        return self.asset_type in {AssetType.ETF, AssetType.STOCK}

    @property
    def label(self) -> str:
        return f"{self.name}（{self.instrument}）"


@dataclass(frozen=True, slots=True)
class StrategyLabTemplate:
    instance_id: str
    name: str
    strategy: StrategyLabKind
    strategy_version: str
    instruments: tuple[InstrumentId, ...]
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        safe_identifier(self.instance_id, label="strategy template id")
        safe_display_text(self.name, label="strategy template name")
        if type(self.strategy) is not StrategyLabKind:
            raise TypeError("strategy template kind must be exact")
        safe_identifier(self.strategy_version, label="strategy template version")
        instruments = tuple(sorted(set(self.instruments), key=str))
        if not instruments or any(type(item) is not InstrumentId for item in instruments):
            raise ValueError("strategy template instruments must contain InstrumentId values")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("strategy template parameters must be a mapping")
        object.__setattr__(self, "instruments", instruments)


@dataclass(frozen=True, slots=True)
class StrategyLegConfiguration:
    strategy_id: str
    strategy: StrategyLabKind
    instruments: tuple[InstrumentId, ...]
    budget: Decimal
    signal_instrument: InstrumentId | None = None
    short_window: int = 20
    long_window: int = 60
    confirmation_days: int = 1
    buy_expression: str = ""
    sell_expression: str = ""
    variables: tuple[DslVariable, ...] = ()
    kronos_parameters: KronosForecastParameters | None = None
    template_instance_id: str | None = None
    template_name: str | None = None

    def __post_init__(self) -> None:
        safe_identifier(self.strategy_id, label="strategy id")
        if type(self.strategy) is not StrategyLabKind:
            raise TypeError("strategy must be an exact StrategyLabKind")
        instruments = tuple(self.instruments)
        if not instruments or any(type(item) is not InstrumentId for item in instruments):
            raise ValueError("strategy instruments must contain InstrumentId values")
        instruments = tuple(sorted(set(instruments), key=str))
        object.__setattr__(self, "instruments", instruments)
        if type(self.budget) is not Decimal or not self.budget.is_finite():
            raise ValueError("budget must be a finite Decimal")
        if not Decimal("0") < self.budget <= Decimal("1"):
            raise ValueError("budget must be greater than zero and at most one")
        if self.signal_instrument is not None and type(self.signal_instrument) is not InstrumentId:
            raise TypeError("signal_instrument must be an exact InstrumentId or None")
        if (
            self.strategy in {StrategyLabKind.DUAL_MA, StrategyLabKind.RULE_DSL}
            and self.signal_instrument is None
        ):
            raise ValueError("signal strategy requires a signal_instrument")
        for name in ("short_window", "long_window", "confirmation_days"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be less than long_window")
        variables = tuple(self.variables)
        if any(type(item) is not DslVariable for item in variables):
            raise TypeError("DSL variables must contain exact DslVariable values")
        object.__setattr__(self, "variables", variables)
        if self.strategy is StrategyLabKind.RULE_DSL:
            RuleDslParameters(
                buy_expression=self.buy_expression,
                sell_expression=self.sell_expression,
                variables=variables,
                target_weight=Decimal("1"),
            )
        if self.strategy is StrategyLabKind.KRONOS_FORECAST:
            if self.kronos_parameters is None:
                raise ValueError("Kronos strategy requires Kronos parameters")
        elif self.kronos_parameters is not None:
            raise ValueError("only Kronos strategy may define Kronos parameters")
        if self.template_instance_id is not None:
            safe_identifier(self.template_instance_id, label="strategy template id")
        if self.template_name is not None:
            safe_display_text(self.template_name, label="strategy template name")
        if (self.template_instance_id is None) != (self.template_name is None):
            raise ValueError("strategy template id and name must be provided together")


@dataclass(frozen=True, slots=True)
class StrategyLabInitialPosition:
    instrument: InstrumentId
    target_weight: Decimal

    def __post_init__(self) -> None:
        if type(self.instrument) is not InstrumentId:
            raise TypeError("initial position instrument must be an exact InstrumentId")
        if type(self.target_weight) is not Decimal or not self.target_weight.is_finite():
            raise ValueError("initial position weight must be a finite Decimal")
        if not Decimal("0") < self.target_weight <= Decimal("1"):
            raise ValueError("initial position weight must be greater than zero and at most one")


@dataclass(frozen=True, slots=True)
class StrategyLabConfiguration:
    strategies: tuple[StrategyLegConfiguration, ...]
    benchmark: InstrumentId
    start: date
    end: date
    initial_cash: Decimal
    commission_rate: Decimal
    minimum_commission: Decimal
    slippage_bps: Decimal
    execution_timing: ExecutionTiming
    rebalance_mode: StrategyLabRebalanceMode = StrategyLabRebalanceMode.SIGNAL_CHANGE
    rebalance_drift: Decimal = Decimal("0.02")
    minimum_trade_amount: Decimal = Decimal("5000")
    initial_cash_weight: Decimal = Decimal("1")
    initial_positions: tuple[StrategyLabInitialPosition, ...] = ()

    def __post_init__(self) -> None:
        strategies = tuple(self.strategies)
        if not strategies or any(type(item) is not StrategyLegConfiguration for item in strategies):
            raise ValueError("strategies must contain StrategyLegConfiguration values")
        strategy_ids = tuple(item.strategy_id for item in strategies)
        if len(set(strategy_ids)) != len(strategy_ids):
            raise ValueError("strategy ids must be unique")
        if sum((item.budget for item in strategies), Decimal("0")) > Decimal("1"):
            raise ValueError("strategy budgets must sum to at most one")
        object.__setattr__(self, "strategies", strategies)
        if type(self.benchmark) is not InstrumentId:
            raise TypeError("benchmark must be an exact InstrumentId")
        if type(self.start) is not date or type(self.end) is not date or self.start >= self.end:
            raise ValueError("backtest range must contain at least two ordered dates")
        for name in ("initial_cash", "commission_rate", "minimum_commission", "slippage_bps"):
            value = getattr(self, name)
            if type(value) is not Decimal or not value.is_finite() or value < 0:
                raise ValueError(f"{name} must be a finite non-negative Decimal")
        if self.initial_cash < Decimal("10000") or self.initial_cash != self.initial_cash.quantize(
            Decimal("0.01")
        ):
            raise ValueError("initial_cash must be at least 10000 and rounded to cents")
        if self.commission_rate > Decimal("0.01"):
            raise ValueError("commission_rate must not exceed 1%")
        if self.slippage_bps >= Decimal("1000"):
            raise ValueError("slippage_bps must be less than 1000")
        if type(self.execution_timing) is not ExecutionTiming:
            raise TypeError("execution_timing must be an exact ExecutionTiming")
        if type(self.rebalance_mode) is not StrategyLabRebalanceMode:
            raise TypeError("rebalance_mode must be an exact StrategyLabRebalanceMode")
        if (
            type(self.rebalance_drift) is not Decimal
            or not self.rebalance_drift.is_finite()
            or not Decimal("0") <= self.rebalance_drift <= Decimal("0.25")
        ):
            raise ValueError("rebalance_drift must be between zero and 25%")
        if (
            type(self.minimum_trade_amount) is not Decimal
            or not self.minimum_trade_amount.is_finite()
            or self.minimum_trade_amount < 0
        ):
            raise ValueError("minimum_trade_amount must be finite and non-negative")
        if (
            type(self.initial_cash_weight) is not Decimal
            or not self.initial_cash_weight.is_finite()
            or not Decimal("0") <= self.initial_cash_weight <= Decimal("1")
        ):
            raise ValueError("initial_cash_weight must be between zero and one")
        positions = tuple(self.initial_positions)
        if any(type(item) is not StrategyLabInitialPosition for item in positions):
            raise TypeError("initial_positions must contain StrategyLabInitialPosition values")
        positions = tuple(sorted(positions, key=lambda item: str(item.instrument)))
        if len({item.instrument for item in positions}) != len(positions):
            raise ValueError("initial position instruments must be unique")
        total_weight = self.initial_cash_weight + sum(
            (item.target_weight for item in positions), Decimal("0")
        )
        if total_weight != Decimal("1"):
            raise ValueError("initial position and cash weights must sum to one")
        object.__setattr__(self, "initial_positions", positions)


@dataclass(frozen=True, slots=True)
class StrategyLabHistoryEntry:
    run_id: str
    status: TaskStatus
    submitted_at: datetime
    completed_at: datetime | None
    strategy_count: int
    target_count: int
    has_report: bool
    task_id: str | None = None
    failure_code: str | None = None

    def __post_init__(self) -> None:
        safe_identifier(self.run_id, label="backtest run id")
        if type(self.status) is not TaskStatus:
            raise TypeError("history status must be an exact TaskStatus")
        for label, value in (
            ("submitted_at", self.submitted_at),
            ("completed_at", self.completed_at),
        ):
            if value is None and label == "completed_at":
                continue
            if type(value) is not datetime:
                raise TypeError(f"{label} must be an exact datetime")
            assert isinstance(value, datetime)
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{label} must be timezone-aware")
        for name in ("strategy_count", "target_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if type(self.has_report) is not bool:
            raise TypeError("has_report must be an exact bool")
        if self.task_id is not None:
            safe_identifier(self.task_id, label="backtest task id")
        if self.failure_code is not None:
            stable_code(self.failure_code, label="backtest failure code")


@dataclass(frozen=True, slots=True)
class StrategyLabState:
    instruments: tuple[StrategyLabInstrument, ...]
    latest_report: BacktestReport | None
    active_task: TaskSnapshot | None = None
    active_report: BacktestReport | None = None
    failure_code: str | None = None
    history: tuple[StrategyLabHistoryEntry, ...] = ()
    history_page: int = 1
    history_total_pages: int = 1
    history_total_items: int = 0
    templates: tuple[StrategyLabTemplate, ...] = ()


class StrategyLabGateway(Protocol):
    def instruments(self) -> Sequence[StrategyLabInstrument]: ...
    def templates(self) -> Sequence[StrategyLabTemplate]: ...
    def latest_report(self) -> BacktestReport | None: ...
    def report(self, run_id: str) -> BacktestReport | None: ...
    def compare_report(self, run_id: str, benchmark: InstrumentId) -> BacktestReport: ...
    def history(self) -> Sequence[StrategyLabHistoryEntry]: ...
    def delete_report(self, run_id: str) -> bool: ...
    def clear_reports(self) -> int: ...
    def new_run_id(self) -> str: ...
    def run(self, run_id: str, configuration: StrategyLabConfiguration) -> None: ...


class TaskGateway(Protocol):
    def submit(self, name: str, heavy: bool, operation: Operation) -> TaskSnapshot: ...
    def status(self, task_id: str) -> TaskSnapshot: ...
    def snapshots(self) -> Sequence[TaskSnapshot]: ...
    def discard_terminal(self, task_id: str) -> bool: ...
    def clear_terminal(self, *, name_prefix: str | None = None) -> int: ...


class StrategyLabPageModel:
    def __init__(self, gateway: StrategyLabGateway, tasks: TaskGateway) -> None:
        self._gateway = gateway
        self._tasks = tasks
        self._active_run_id: str | None = None
        self._active_task: TaskSnapshot | None = None
        self._selected_run_id: str | None = None
        self._selected_report: BacktestReport | None = None
        self._history_page = 1
        self._history_page_size = 10
        self._history_query = ""
        self._lock = RLock()

    def start(self, configuration: StrategyLabConfiguration) -> TaskSnapshot:
        if type(configuration) is not StrategyLabConfiguration:
            raise TypeError("configuration must be an exact StrategyLabConfiguration")
        with self._lock:
            if self._active_task is not None:
                current = self._tasks.status(self._active_task.task_id)
                if current.status not in _TERMINAL_STATUSES:
                    raise RuntimeError("BACKTEST_ALREADY_RUNNING")
            run_id = safe_identifier(self._gateway.new_run_id(), label="backtest run id")
            name = f"backtest:{run_id}"
            task = self._tasks.submit(
                name,
                True,
                lambda: self._gateway.run(run_id, configuration),
            )
            self._active_run_id = run_id
            self._active_task = task
            self._selected_run_id = None
            self._selected_report = None
            return task

    def state(self) -> StrategyLabState:
        instruments = tuple(self._gateway.instruments())
        template_reader = getattr(self._gateway, "templates", None)
        templates = tuple(template_reader()) if callable(template_reader) else ()
        with self._lock:
            history_page = self._history_page
            selected_report = self._selected_report
        history = self._history()
        total_items = len(history)
        total_pages = max(1, (total_items + self._history_page_size - 1) // self._history_page_size)
        history_page = min(history_page, total_pages)
        with self._lock:
            self._history_page = history_page
        offset = (history_page - 1) * self._history_page_size
        page_history = history[offset : offset + self._history_page_size]
        with self._lock:
            run_id = self._active_run_id
            submitted = self._active_task
        if run_id is None or submitted is None:
            return StrategyLabState(
                instruments,
                None,
                active_report=selected_report,
                history=page_history,
                history_page=history_page,
                history_total_pages=total_pages,
                history_total_items=total_items,
                templates=templates,
            )
        active = self._tasks.status(submitted.task_id)
        report = (
            self._gateway.report(run_id)
            if active.status is TaskStatus.SUCCEEDED
            else selected_report
        )
        failure_code = None
        if active.status is TaskStatus.FAILED:
            assert active.failure is not None
            failure_code = active.failure.code
        return StrategyLabState(
            instruments,
            None,
            active,
            report,
            failure_code,
            page_history,
            history_page,
            total_pages,
            total_items,
            templates,
        )

    def _task_snapshots(self) -> tuple[TaskSnapshot, ...]:
        reader = getattr(self._tasks, "snapshots", None)
        if not callable(reader):
            with self._lock:
                return () if self._active_task is None else (self._active_task,)
        return tuple(reader())

    def _history(self) -> tuple[StrategyLabHistoryEntry, ...]:
        reader = getattr(self._gateway, "history", None)
        records = tuple(reader()) if callable(reader) else ()
        by_run_id = {item.run_id: item for item in records}
        for task in self._task_snapshots():
            prefix = "backtest:"
            if not task.name.startswith(prefix):
                continue
            run_id = task.name.removeprefix(prefix)
            existing = by_run_id.get(run_id)
            by_run_id[run_id] = StrategyLabHistoryEntry(
                run_id=run_id,
                status=task.status,
                submitted_at=task.submitted_at,
                completed_at=task.completed_at,
                strategy_count=0 if existing is None else existing.strategy_count,
                target_count=0 if existing is None else existing.target_count,
                has_report=existing is not None and existing.has_report,
                task_id=task.task_id,
                failure_code=None if task.failure is None else task.failure.code,
            )
        ordered = tuple(
            sorted(
                by_run_id.values(),
                key=lambda item: (item.submitted_at, item.run_id),
                reverse=True,
            )
        )
        with self._lock:
            query = self._history_query.casefold()
        return (
            ordered
            if not query
            else tuple(item for item in ordered if query in item.run_id.casefold())
        )

    def set_history_page(self, page: int) -> None:
        if type(page) is not int or page <= 0:
            raise ValueError("backtest history page must be positive")
        with self._lock:
            self._history_page = page

    def set_history_query(self, query: str) -> None:
        if type(query) is not str or len(query.strip()) > 128:
            raise ValueError("backtest history query is invalid")
        with self._lock:
            self._history_query = query.strip()
            self._history_page = 1

    def select_report(self, run_id: str) -> None:
        checked = safe_identifier(run_id, label="backtest run id")
        report = self._gateway.report(checked)
        if report is None:
            raise LookupError("BACKTEST_REPORT_MISSING")
        with self._lock:
            self._selected_run_id = checked
            self._selected_report = report

    def clear_report_selection(self) -> None:
        with self._lock:
            self._selected_run_id = None
            self._selected_report = None

    def compare_report(self, run_id: str, benchmark: InstrumentId) -> BacktestReport:
        checked_run_id = safe_identifier(run_id, label="backtest run id")
        if type(benchmark) is not InstrumentId:
            raise TypeError("benchmark must be an exact InstrumentId")
        reader = getattr(self._gateway, "compare_report", None)
        if callable(reader):
            compared = reader(checked_run_id, benchmark)
            if type(compared) is not BacktestReport:
                raise TypeError("comparison report must be an exact BacktestReport")
            return compared
        report = self._gateway.report(checked_run_id)
        if report is None:
            raise LookupError("BACKTEST_REPORT_MISSING")
        return report

    def delete_history(self, run_id: str) -> bool:
        checked = safe_identifier(run_id, label="backtest run id")
        matching = next(
            (item for item in self._task_snapshots() if item.name == f"backtest:{checked}"),
            None,
        )
        if matching is not None and matching.status not in _TERMINAL_STATUSES:
            raise RuntimeError("BACKTEST_TASK_ACTIVE")
        deleted = self._gateway.delete_report(checked)
        if matching is not None:
            discard = getattr(self._tasks, "discard_terminal", None)
            if callable(discard):
                discard(matching.task_id)
                deleted = True
        with self._lock:
            if self._selected_run_id == checked:
                self._selected_run_id = None
                self._selected_report = None
        return deleted

    def clear_history(self) -> int:
        before = {item.run_id for item in self._history()}
        self._gateway.clear_reports()
        clearer = getattr(self._tasks, "clear_terminal", None)
        if callable(clearer):
            clearer(name_prefix="backtest:")
        after = {item.run_id for item in self._history()}
        with self._lock:
            self._selected_run_id = None
            self._selected_report = None
            self._history_page = 1
        return len(before - after)

    def acknowledge(self, task: TaskSnapshot) -> None:
        if type(task) is not TaskSnapshot or task.status not in _TERMINAL_STATUSES:
            raise ValueError("only a terminal backtest task can be acknowledged")
        with self._lock:
            if self._active_task is None or self._active_task.task_id != task.task_id:
                raise ValueError("backtest task does not match the active task")
            self._active_run_id = None
            self._active_task = None


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _editable_json(value: object) -> object:
    if value is None or type(value) in {str, bool, int, float}:
        return value
    if type(value) is Decimal:
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _editable_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_editable_json(item) for item in value]
    raise TypeError("strategy template contains an unsupported value")


def _template_positive_int(parameters: Mapping[str, object], name: str, default: int) -> int:
    value = parameters.get(name, default)
    if type(value) is int and value > 0:
        return value
    if type(value) is str and value.isdecimal() and int(value) > 0:
        return int(value)
    return default


def _template_float(parameters: Mapping[str, object], name: str, default: float) -> float:
    value = parameters.get(name, default)
    if isinstance(value, (int, float, str)) and not isinstance(value, bool):
        try:
            checked = float(value)
        except ValueError:
            return default
        if checked == checked and checked not in {float("inf"), float("-inf")}:
            return checked
    return default


def _normalized_equity(report: BacktestReport) -> tuple[CurvePoint, ...]:
    first = report.equity_curve[0].value
    return tuple(CurvePoint(item.day, item.value / first) for item in report.equity_curve)


def _trade_markers(
    report: BacktestReport,
) -> tuple[tuple[CurvePoint, ...], tuple[CurvePoint, ...]]:
    normalized = {item.day: item.value for item in _normalized_equity(report)}
    buy_days = sorted(
        {fill.trading_day.isoformat() for fill in report.result.fills if fill.side.value == "buy"}
    )
    sell_days = sorted(
        {fill.trading_day.isoformat() for fill in report.result.fills if fill.side.value == "sell"}
    )
    return (
        tuple(CurvePoint(day, normalized[day]) for day in buy_days),
        tuple(CurvePoint(day, normalized[day]) for day in sell_days),
    )


def _signal_markers(
    report: BacktestReport,
) -> tuple[tuple[CurvePoint, ...], tuple[CurvePoint, ...], tuple[CurvePoint, ...]]:
    normalized = {item.day: item.value for item in _normalized_equity(report)}
    buy_days = sorted(
        {
            order.created_on.isoformat()
            for order in report.result.orders
            if order.side.value == "buy" and order.created_on.isoformat() in normalized
        }
    )
    sell_days = sorted(
        {
            order.created_on.isoformat()
            for order in report.result.orders
            if order.side.value == "sell" and order.created_on.isoformat() in normalized
        }
    )
    unfilled_days = sorted(
        {
            order.created_on.isoformat()
            for order in report.result.orders
            if order.status in {OrderStatus.CANCELLED, OrderStatus.PARTIALLY_FILLED}
            and order.created_on.isoformat() in normalized
        }
    )
    return (
        tuple(CurvePoint(day, normalized[day]) for day in buy_days),
        tuple(CurvePoint(day, normalized[day]) for day in sell_days),
        tuple(CurvePoint(day, normalized[day]) for day in unfilled_days),
    )


_ORDER_STATUS_LABELS = {
    OrderStatus.PENDING: "待成交",
    OrderStatus.FILLED: "已成交",
    OrderStatus.PARTIALLY_FILLED: "部分成交",
    OrderStatus.CANCELLED: "未成交",
}

_CANCELLATION_LABELS = {
    "MISSING_BAR": "缺少行情",
    "SUSPENDED": "停牌",
    "LIMIT_UP": "涨停无法买入",
    "LIMIT_DOWN": "跌停无法卖出",
    "MARKET_STATUS_UNKNOWN": "市场状态未知",
    "T_PLUS_ONE": "受 T+1 限制",
    "NO_POSITION": "没有可卖持仓",
    "INSUFFICIENT_POSITION": "可卖持仓不足",
    "ODD_LOT_NOT_ALLOWED": "不允许零股",
    "LOT_SIZE": "不足一手",
    "VOLUME_LIMIT": "超过成交量限制",
    "INSUFFICIENT_CASH": "现金不足",
    "END_OF_DAY_UNFILLED": "当日未成交",
    "NO_NEXT_SESSION": "回测结束前没有下一交易日",
}


def _strategy_rule_text(report: BacktestReport, sleeve_ids: Sequence[str], side: str) -> str:
    by_id = {item.sleeve_id: item for item in report.strategies}
    rules: list[str] = []
    for sleeve_id in sleeve_ids:
        snapshot = by_id.get(sleeve_id)
        if snapshot is None:
            continue
        if snapshot.strategy_type == StrategyLabKind.RULE_DSL.value:
            key = "buy_expression" if side == "buy" else "sell_expression"
            rules.append(f"{sleeve_id}: {snapshot.parameters.get(key, '—')}")
        elif snapshot.strategy_type == StrategyLabKind.DUAL_MA.value:
            short = snapshot.parameters.get("short_window", 20)
            long = snapshot.parameters.get("long_window", 60)
            operator = ">" if side == "buy" else "≤"
            rules.append(f"{sleeve_id}: MA{short} {operator} MA{long}")
        elif snapshot.strategy_type == StrategyLabKind.KRONOS_FORECAST.value:
            raw = snapshot.parameters.get("kronos_parameters", {})
            parameters = raw if isinstance(raw, Mapping) else {}
            horizon = parameters.get("horizon", 5)
            threshold = parameters.get("entry_return" if side == "buy" else "exit_return", "—")
            rules.append(f"{sleeve_id}: Kronos {horizon} 日预测阈值 {threshold}")
        else:
            rules.append(f"{sleeve_id}: 买入并持有")
    return "；".join(rules) or "组合目标仓位变化"


def _signal_rows(report: BacktestReport) -> list[dict[str, object]]:
    fill_by_order = {fill.order_id: fill for fill in report.result.fills}
    rows: list[dict[str, object]] = []
    for order in report.result.orders:
        sleeve_ids = tuple(order.sleeve_weights)
        if not sleeve_ids:
            matching_sleeves: list[str] = []
            for strategy in report.strategies:
                raw_targets = strategy.parameters.get("trade_instruments", ())
                targets = (
                    tuple(str(item) for item in raw_targets)
                    if isinstance(raw_targets, Sequence)
                    and not isinstance(raw_targets, (str, bytes, bytearray))
                    else ()
                )
                if str(order.instrument) in targets:
                    matching_sleeves.append(strategy.sleeve_id)
            sleeve_ids = tuple(matching_sleeves)
        fill = fill_by_order.get(order.order_id)
        reason_code = None if order.cancellation_reason is None else order.cancellation_reason.value
        rows.append(
            {
                "id": order.order_id,
                "signal_date": order.created_on.isoformat(),
                "trade_date": "—" if fill is None else fill.trading_day.isoformat(),
                "side": "买入" if order.side.value == "buy" else "卖出",
                "instrument": str(order.instrument),
                "strategies": "、".join(sleeve_ids) or "组合",
                "rule": _strategy_rule_text(report, sleeve_ids, order.side.value),
                "quantity": f"{order.filled_quantity}/{order.quantity}",
                "price": "—" if fill is None else str(fill.price),
                "status": _ORDER_STATUS_LABELS[order.status],
                "reason": (
                    "—"
                    if reason_code is None
                    else _CANCELLATION_LABELS.get(reason_code, reason_code)
                ),
            }
        )
    return rows[-100:]


def render_strategy_lab_page(model: StrategyLabPageModel | None) -> None:
    if model is None:
        ui.label("策略回测服务尚未配置。").classes("text-sm text-slate-600")
        return
    try:
        model.clear_report_selection()
        initial_state = model.state()
    except Exception:
        ui.label("策略回测状态读取失败，请查看本地日志。").classes("text-red-700")
        return
    if not initial_state.instruments:
        ui.label("当前没有可用于回测的行情数据，请先在行情数据页完成同步。").classes(
            "text-amber-700"
        )
        return

    by_id = {str(item.instrument): item for item in initial_state.instruments}
    available_etfs = tuple(
        item for item in initial_state.instruments if item.asset_type is AssetType.ETF
    )
    if not available_etfs:
        ui.label("当前没有已同步的 ETF 行情，请先在行情数据页完成同步。").classes("text-amber-700")
        return
    trade_options = {
        str(item.instrument): item.label
        for item in sorted(available_etfs, key=lambda value: str(value.instrument))
    }
    signal_options = {str(item.instrument): item.label for item in initial_state.instruments}
    benchmark_options = dict(signal_options)
    default_signal = next(
        (
            str(item.instrument)
            for item in initial_state.instruments
            if item.asset_type is AssetType.INDEX
        ),
        str(initial_state.instruments[0].instrument),
    )
    default_benchmark = default_signal
    common_start = max(item.first_day for item in initial_state.instruments)
    common_end = min(item.last_day for item in initial_state.instruments)
    if common_start >= common_end:
        common_start = min(item.first_day for item in initial_state.instruments)
        common_end = max(item.last_day for item in initial_state.instruments)

    strategy_labels = {
        StrategyLabKind.BUY_AND_HOLD.value: "买入持有",
        StrategyLabKind.DUAL_MA.value: "双均线趋势",
        StrategyLabKind.RULE_DSL.value: "自定义规则 DSL",
        StrategyLabKind.KRONOS_FORECAST.value: "Kronos K 线预测",
    }
    configured_strategies: list[StrategyLegConfiguration] = []
    next_strategy_number = 1
    selected_result_benchmarks: dict[str, str] = {}
    initial_position_percentages = {str(item.instrument): Decimal("0") for item in available_etfs}
    initial_cash_percentage = {"value": Decimal("100")}
    templates_by_id = {item.instance_id: item for item in initial_state.templates}
    active_template: dict[str, StrategyLabTemplate | None] = {"value": None}

    ui.label("多个策略共享一个账户，各自给出 ETF 目标仓位，再按资金占比合并执行。").classes(
        "text-sm text-slate-600"
    )
    with ui.card().classes("w-full border border-slate-200 shadow-none"):
        ui.label("账户与成交设置").classes("text-lg font-semibold")
        with ui.row().classes("w-full gap-4"):
            start_input = ui.input("开始日期", value=common_start.isoformat()).props("type=date")
            end_input = ui.input("结束日期", value=common_end.isoformat()).props("type=date")
            cash_input = ui.number("初始资金（元）", value=1_000_000, min=10_000, step=10_000)
            timing_select = ui.select(
                {
                    ExecutionTiming.NEXT_OPEN.value: "收盘产生信号，下一交易日开盘成交",
                    ExecutionTiming.NEXT_CLOSE.value: "收盘产生信号，下一交易日收盘成交",
                },
                value=ExecutionTiming.NEXT_OPEN.value,
                label="买卖时机",
            ).classes("min-w-80")
        with ui.row().classes("w-full gap-4"):
            commission_input = ui.number(
                "佣金率", value=0.0003, min=0, max=0.01, step=0.0001, format="%.4f"
            )
            minimum_commission_input = ui.number("最低佣金（元）", value=5, min=0, step=1)
            slippage_input = (
                ui.number("滑点（基点 / bps）", value=2, min=0, max=999, step=1)
                .props("suffix=基点 aria-label=滑点（基点）")
                .classes("min-w-48")
            )
        with ui.row().classes("w-full gap-4 items-end"):
            rebalance_select = ui.select(
                {
                    StrategyLabRebalanceMode.SIGNAL_CHANGE.value: "仅信号变化时（推荐）",
                    StrategyLabRebalanceMode.WEEKLY.value: "每周检查并再平衡",
                    StrategyLabRebalanceMode.MONTHLY.value: "每月检查并再平衡",
                    StrategyLabRebalanceMode.DAILY.value: "每日再平衡",
                },
                value=StrategyLabRebalanceMode.SIGNAL_CHANGE.value,
                label="再平衡方式",
            ).classes("min-w-72")
            drift_input = ui.number("仓位偏离阈值（%）", value=2, min=0, max=25, step=0.5)
            minimum_trade_input = ui.number("最小交易金额（元）", value=5000, min=0, step=1000)
        ui.label(
            "只有达到再平衡时点，且仓位偏离或预计交易金额达到阈值时才下单，可避免细小仓位变化造成频繁成交。"
        ).classes("text-xs text-slate-500")
        ui.label("滑点单位：1 基点（bps）= 0.01%。").classes("text-xs text-slate-500")
        initial_mode_select = ui.select(
            {
                "cash": "空仓（100% 现金）",
                "custom": "自定义初始持仓",
            },
            value="cash",
            label="初始持仓",
        ).classes("min-w-72")

        @ui.refreshable
        def initial_position_form() -> None:
            if str(initial_mode_select.value) != "custom":
                ui.label("初始账户为空仓，全部资金以现金开始回测。").classes(
                    "text-sm text-slate-500"
                )
                return
            ui.label("按初始总资产配置比例；现金与所有 ETF 仓位合计必须等于 100%。").classes(
                "text-sm text-slate-600"
            )
            summary = ui.label("").classes("text-sm")

            def update_summary() -> None:
                total = initial_cash_percentage["value"] + sum(
                    initial_position_percentages.values(), Decimal("0")
                )
                summary.set_text(
                    f"现金 {initial_cash_percentage['value']:g}% · "
                    f"ETF {total - initial_cash_percentage['value']:g}% · 合计 {total:g}%"
                )
                summary.classes(
                    replace=(
                        "text-sm text-emerald-700"
                        if total == Decimal("100")
                        else "text-sm text-red-700"
                    )
                )

            def update_cash(value: object) -> None:
                initial_cash_percentage["value"] = Decimal(str(value or 0))
                update_summary()
                reset_range()
                refresh_readiness()

            def update_position(symbol: str, value: object) -> None:
                initial_position_percentages[symbol] = Decimal(str(value or 0))
                update_summary()
                reset_range()
                refresh_readiness()

            with ui.row().classes("w-full gap-4 items-end flex-wrap"):
                ui.number(
                    "初始现金（%）",
                    value=float(initial_cash_percentage["value"]),
                    min=0,
                    max=100,
                    step=5,
                ).on_value_change(lambda event: update_cash(event.value))
                for item in available_etfs:
                    symbol = str(item.instrument)
                    ui.number(
                        f"{item.label} 初始仓位（%）",
                        value=float(initial_position_percentages[symbol]),
                        min=0,
                        max=100,
                        step=5,
                    ).classes("min-w-80").on_value_change(
                        lambda event, instrument=symbol: update_position(instrument, event.value)
                    )
            update_summary()

        def initial_mode_changed() -> None:
            initial_position_form.refresh()
            reset_range()
            refresh_readiness()

        initial_mode_select.on_value_change(lambda _: initial_mode_changed())
        initial_position_form()
        ui.label("初始账户由上方设置；所有策略在收盘后生成目标，成交时点由上方统一设置。").classes(
            "text-xs text-slate-500"
        )

    with ui.card().classes("w-full border border-slate-200 shadow-none"):
        ui.label("本次回测策略组合").classes("text-lg font-semibold")
        ui.label("这里的参数只用于本次回测；策略定义、版本和原理说明请在策略实验室管理。").classes(
            "text-sm text-slate-500"
        )
        template_select = ui.select(
            {
                item.instance_id: f"{item.name} · {strategy_labels[item.strategy.value]}"
                for item in initial_state.templates
            },
            value=None,
            label="已保存策略模板（可选）",
        ).classes("min-w-96")
        if not initial_state.templates:
            template_select.disable()
            ui.label("暂无可用于回测的已启用策略模板；也可以直接配置本次回测。 ").classes(
                "text-xs text-slate-500"
            )
        strategy_select = ui.select(
            strategy_labels,
            value=None,
            label="添加策略",
        ).classes("min-w-64")
        draft_controls: dict[str, object] = {}

        def initial_allocation() -> tuple[Decimal, tuple[StrategyLabInitialPosition, ...]]:
            if str(initial_mode_select.value) != "custom":
                return Decimal("1"), ()
            cash_weight = initial_cash_percentage["value"] / Decimal("100")
            positions = tuple(
                StrategyLabInitialPosition(
                    instrument=InstrumentId.parse(symbol),
                    target_weight=percentage / Decimal("100"),
                )
                for symbol, percentage in initial_position_percentages.items()
                if percentage > 0
            )
            total = cash_weight + sum((item.target_weight for item in positions), Decimal("0"))
            if total != Decimal("1"):
                raise ValueError("初始现金与 ETF 仓位合计必须等于 100%")
            return cash_weight, positions

        def selected_instrument_ids() -> set[str]:
            selected = {default_benchmark}
            for strategy in configured_strategies:
                selected.update(str(item) for item in strategy.instruments)
                if strategy.signal_instrument is not None:
                    selected.add(str(strategy.signal_instrument))
            if str(initial_mode_select.value) == "custom":
                selected.update(
                    symbol
                    for symbol, percentage in initial_position_percentages.items()
                    if percentage > 0
                )
            return selected

        def reset_range() -> None:
            selected_items = [by_id[item] for item in selected_instrument_ids()]
            start_input.value = max(item.first_day for item in selected_items).isoformat()
            end_input.value = min(item.last_day for item in selected_items).isoformat()

        def refresh_readiness() -> None:
            budget = sum((item.budget for item in configured_strategies), Decimal("0"))
            if not configured_strategies:
                readiness_label.set_text("请至少加入一个策略。")
                readiness_label.classes(replace="text-sm text-amber-700")
                run_button.disable()
            elif str(initial_mode_select.value) == "custom" and (
                initial_cash_percentage["value"]
                + sum(initial_position_percentages.values(), Decimal("0"))
                != Decimal("100")
            ):
                readiness_label.set_text("初始现金与 ETF 仓位合计必须等于 100%。")
                readiness_label.classes(replace="text-sm text-red-700")
                run_button.disable()
            elif budget > Decimal("1"):
                readiness_label.set_text(f"策略资金占比合计 {budget * 100:g}%，不能超过 100%。")
                readiness_label.classes(replace="text-sm text-red-700")
                run_button.disable()
            else:
                target_count = len(
                    {item for strategy in configured_strategies for item in strategy.instruments}
                )
                readiness_label.set_text(
                    f"已配置 {len(configured_strategies)} 个策略 · "
                    f"{target_count} 个 ETF · 资金占比 {budget * 100:g}%"
                )
                readiness_label.classes(replace="text-sm text-emerald-700")
                run_button.enable()

        def remove_strategy(strategy_id: str) -> None:
            configured_strategies[:] = [
                item for item in configured_strategies if item.strategy_id != strategy_id
            ]
            reset_range()
            strategy_cards.refresh()
            refresh_readiness()

        @ui.refreshable
        def strategy_cards() -> None:
            if not configured_strategies:
                ui.label("组合中还没有策略。").classes("text-sm text-slate-400")
                return
            for strategy in configured_strategies:
                with ui.card().classes("w-full border border-slate-100 bg-slate-50 shadow-none"):
                    with ui.row().classes("w-full items-center justify-between"):
                        ui.label(
                            f"{strategy.strategy_id} · "
                            f"{strategy_labels[strategy.strategy.value]} · "
                            f"资金 {strategy.budget * 100:g}%"
                        ).classes("font-medium")
                        ui.button(
                            icon="delete_outline",
                            on_click=lambda _, strategy_id=strategy.strategy_id: remove_strategy(
                                strategy_id
                            ),
                        ).props("flat round color=negative")
                    targets = "、".join(by_id[str(item)].label for item in strategy.instruments)
                    ui.label(f"目标 ETF：{targets}").classes("text-sm text-slate-600")
                    if strategy.template_name is not None:
                        ui.label(
                            f"来源模板：{strategy.template_name}（{strategy.template_instance_id}）"
                        ).classes("text-xs text-indigo-700")
                    if strategy.strategy is StrategyLabKind.DUAL_MA:
                        assert strategy.signal_instrument is not None
                        ui.label(
                            f"信号：{by_id[str(strategy.signal_instrument)].label} · "
                            f"均线 {strategy.short_window}/{strategy.long_window} · "
                            f"确认 {strategy.confirmation_days} 日"
                        ).classes("text-xs text-slate-500")
                    elif strategy.strategy is StrategyLabKind.RULE_DSL:
                        assert strategy.signal_instrument is not None
                        exported = "、".join(
                            item.name for item in strategy.variables if item.optimize
                        )
                        ui.label(
                            f"信号：{by_id[str(strategy.signal_instrument)].label} · "
                            f"买入 {strategy.buy_expression} · 卖出 {strategy.sell_expression}"
                        ).classes("text-xs text-slate-500")
                        ui.label(f"导出变量：{exported or '无'}").classes("text-xs text-indigo-700")
                    elif strategy.strategy is StrategyLabKind.KRONOS_FORECAST:
                        assert strategy.kronos_parameters is not None
                        kronos = strategy.kronos_parameters
                        ui.label(
                            f"{kronos.model_size}/{kronos.device} · 历史 {kronos.lookback} 日 · "
                            f"预测 {kronos.horizon} 日 · 每 {kronos.rebalance_interval} 日重算"
                        ).classes("text-xs text-indigo-700")

        def add_strategy() -> None:
            nonlocal next_strategy_number
            try:
                kind = StrategyLabKind(str(strategy_select.value))
                raw_instruments = getattr(draft_controls["instruments"], "value")
                if (
                    not isinstance(raw_instruments, Sequence)
                    or isinstance(raw_instruments, (str, bytes))
                    or not raw_instruments
                ):
                    raise ValueError("至少选择一个目标 ETF")
                signal_control = draft_controls.get("signal")
                strategy = StrategyLegConfiguration(
                    strategy_id=f"strategy-{next_strategy_number}",
                    strategy=kind,
                    instruments=tuple(InstrumentId.parse(str(item)) for item in raw_instruments),
                    budget=(
                        Decimal(str(getattr(draft_controls["budget"], "value"))) / Decimal("100")
                    ),
                    signal_instrument=(
                        None
                        if signal_control is None
                        else InstrumentId.parse(str(getattr(signal_control, "value")))
                    ),
                    short_window=int(getattr(draft_controls.get("short"), "value", 20)),
                    long_window=int(getattr(draft_controls.get("long"), "value", 60)),
                    confirmation_days=int(getattr(draft_controls.get("confirmation"), "value", 1)),
                    buy_expression=str(getattr(draft_controls.get("buy_expression"), "value", "")),
                    sell_expression=str(
                        getattr(draft_controls.get("sell_expression"), "value", "")
                    ),
                    variables=(
                        ()
                        if kind is not StrategyLabKind.RULE_DSL
                        else RuleDslParameters.model_validate_json(
                            json.dumps(
                                {
                                    "buy_expression": str(
                                        getattr(draft_controls["buy_expression"], "value")
                                    ),
                                    "sell_expression": str(
                                        getattr(draft_controls["sell_expression"], "value")
                                    ),
                                    "variables": json.loads(
                                        str(getattr(draft_controls["variables"], "value"))
                                    ),
                                    "target_weight": "1",
                                },
                                ensure_ascii=False,
                            ),
                            strict=True,
                        ).variables
                    ),
                    kronos_parameters=(
                        None
                        if kind is not StrategyLabKind.KRONOS_FORECAST
                        else KronosForecastParameters(
                            model_size=str(getattr(draft_controls["kronos_model"], "value")),
                            device=str(getattr(draft_controls["kronos_device"], "value")),
                            lookback=int(getattr(draft_controls["kronos_lookback"], "value")),
                            horizon=int(getattr(draft_controls["kronos_horizon"], "value")),
                            rebalance_interval=int(
                                getattr(draft_controls["kronos_rebalance"], "value")
                            ),
                            entry_return=Decimal(
                                str(getattr(draft_controls["kronos_entry"], "value"))
                            )
                            / Decimal("100"),
                            exit_return=Decimal(
                                str(getattr(draft_controls["kronos_exit"], "value"))
                            )
                            / Decimal("100"),
                            minimum_path_positive_ratio=Decimal(
                                str(getattr(draft_controls["kronos_positive"], "value"))
                            )
                            / Decimal("100"),
                            trend_window=int(getattr(draft_controls["kronos_trend"], "value")),
                            top_n=int(getattr(draft_controls["kronos_top_n"], "value")),
                            target_weight=Decimal("1"),
                            temperature=float(
                                getattr(draft_controls["kronos_temperature"], "value")
                            ),
                            top_p=float(getattr(draft_controls["kronos_top_p"], "value")),
                            sample_count=int(getattr(draft_controls["kronos_samples"], "value")),
                            seed=int(getattr(draft_controls["kronos_seed"], "value")),
                        )
                    ),
                    template_instance_id=(
                        None
                        if active_template["value"] is None
                        else active_template["value"].instance_id
                    ),
                    template_name=(
                        None if active_template["value"] is None else active_template["value"].name
                    ),
                )
                configured_strategies.append(strategy)
                next_strategy_number += 1
                reset_range()
                strategy_cards.refresh()
                refresh_readiness()
                ui.notify("策略已加入组合。", type="positive")
            except (InvalidOperation, KeyError, TypeError, ValueError) as error:
                ui.notify(f"策略未加入：{error}", type="negative")

        @ui.refreshable
        def strategy_form() -> None:
            draft_controls.clear()
            if strategy_select.value is None:
                ui.label("请选择一个策略开始配置。").classes("text-sm text-slate-400")
                return
            kind = StrategyLabKind(str(strategy_select.value))
            template = active_template["value"]
            if template is not None and template.strategy is not kind:
                template = None
            parameters = {} if template is None else template.parameters
            template_targets = (
                []
                if template is None
                else [str(item) for item in template.instruments if str(item) in trade_options]
            )
            template_signal = default_signal
            if template is not None:
                template_signal = next(
                    (
                        str(index)
                        for index, etf in common_index_etf_pairs()
                        if etf in template.instruments and str(index) in signal_options
                    ),
                    default_signal,
                )
            try:
                budget_default = float(
                    Decimal(str(parameters.get("target_weight", "0.5"))) * Decimal("100")
                )
            except (InvalidOperation, ValueError):
                budget_default = 50.0
            budget_default = min(100.0, max(1.0, budget_default))
            with ui.row().classes("w-full items-end gap-4"):
                if kind in {StrategyLabKind.DUAL_MA, StrategyLabKind.RULE_DSL}:
                    draft_controls["signal"] = ui.select(
                        signal_options,
                        value=template_signal,
                        label="信号标的",
                    ).classes("min-w-72")
                draft_controls["instruments"] = (
                    ui.select(
                        trade_options,
                        value=template_targets,
                        label="目标 ETF（可多选）",
                        multiple=True,
                    )
                    .props("use-chips")
                    .classes("min-w-96")
                )
                draft_controls["budget"] = ui.number(
                    "资金占比（%）", value=budget_default, min=1, max=100, step=5
                )
            if kind is StrategyLabKind.DUAL_MA:
                with ui.row().classes("w-full gap-4"):
                    draft_controls["short"] = ui.number(
                        "短均线（日）",
                        value=_template_positive_int(parameters, "short_window", 20),
                        min=1,
                        step=1,
                    )
                    draft_controls["long"] = ui.number(
                        "长均线（日）",
                        value=_template_positive_int(parameters, "long_window", 60),
                        min=2,
                        step=1,
                    )
                    draft_controls["confirmation"] = ui.number(
                        "连续确认（日）",
                        value=_template_positive_int(parameters, "confirmation_days", 1),
                        min=1,
                        step=1,
                    )
            if kind is StrategyLabKind.RULE_DSL:
                draft_controls["buy_expression"] = ui.textarea(
                    "买入 DSL",
                    value=str(
                        parameters.get(
                            "buy_expression",
                            "cross_above(sma(close, fast_window), sma(close, slow_window))",
                        )
                    ),
                ).classes("w-full font-mono")
                draft_controls["sell_expression"] = ui.textarea(
                    "卖出 DSL",
                    value=str(
                        parameters.get(
                            "sell_expression",
                            "cross_below(sma(close, fast_window), sma(close, slow_window))",
                        )
                    ),
                ).classes("w-full font-mono")
                draft_controls["variables"] = ui.textarea(
                    "导出变量 JSON",
                    value=json.dumps(
                        _editable_json(
                            parameters.get(
                                "variables",
                                [
                                    {
                                        "name": "fast_window",
                                        "value": "20",
                                        "minimum": "5",
                                        "maximum": "40",
                                        "step": "5",
                                        "optimize": True,
                                    },
                                    {
                                        "name": "slow_window",
                                        "value": "60",
                                        "minimum": "40",
                                        "maximum": "200",
                                        "step": "20",
                                        "optimize": True,
                                    },
                                ],
                            )
                        ),
                        ensure_ascii=False,
                        indent=2,
                    ),
                ).classes("w-full font-mono")
            if kind is StrategyLabKind.KRONOS_FORECAST:
                ui.label(
                    "模型在每个重算日批量预测目标 ETF，先按预测收益排名，再用趋势过滤和阈值生成仓位。"
                ).classes("text-sm text-slate-600")
                ui.label(
                    "严格样本外评估前，请先确认所用模型的预训练数据截止日期。"
                ).classes("text-xs text-amber-700")
                runtime_status = kronos_runtime_status()
                ui.label(runtime_status.display_text).classes(
                    "text-xs " + ("text-emerald-700" if runtime_status.cuda_available else "text-amber-700")
                )
                if runtime_status.action_text is not None:
                    ui.label(runtime_status.action_text).classes("text-xs text-slate-500 font-mono")
                with ui.row().classes("w-full gap-4"):
                    draft_controls["kronos_model"] = ui.select(
                        {"mini": "mini（推荐/轻量）", "small": "small", "base": "base"},
                        value=str(parameters.get("model_size", "mini")),
                        label="模型",
                    )
                    draft_controls["kronos_device"] = ui.select(
                        {"auto": "自动", "cuda": "NVIDIA GPU", "cpu": "CPU", "mps": "Apple GPU"},
                        value=str(parameters.get("device", "auto")),
                        label="推理设备",
                    )
                    draft_controls["kronos_lookback"] = ui.number(
                        "历史窗口（日）",
                        value=_template_positive_int(parameters, "lookback", 256),
                        min=64,
                        max=2048,
                    )
                    draft_controls["kronos_horizon"] = ui.number(
                        "预测周期（日）",
                        value=_template_positive_int(parameters, "horizon", 5),
                        min=1,
                        max=60,
                    )
                    draft_controls["kronos_rebalance"] = ui.number(
                        "重算间隔（日）",
                        value=_template_positive_int(parameters, "rebalance_interval", 5),
                        min=1,
                        max=60,
                    )
                with ui.row().classes("w-full gap-4"):
                    draft_controls["kronos_entry"] = ui.number(
                        "买入阈值（%）",
                        value=float(Decimal(str(parameters.get("entry_return", "0.02"))) * 100),
                    )
                    draft_controls["kronos_exit"] = ui.number(
                        "退出阈值（%）",
                        value=float(Decimal(str(parameters.get("exit_return", "-0.01"))) * 100),
                    )
                    draft_controls["kronos_positive"] = ui.number(
                        "正向路径比例（%）",
                        value=float(
                            Decimal(str(parameters.get("minimum_path_positive_ratio", "0.60")))
                            * 100
                        ),
                        min=0,
                        max=100,
                    )
                    draft_controls["kronos_trend"] = ui.number(
                        "趋势窗口（日）",
                        value=_template_positive_int(parameters, "trend_window", 60),
                        min=2,
                        max=512,
                    )
                    draft_controls["kronos_top_n"] = ui.number(
                        "最多持有",
                        value=_template_positive_int(parameters, "top_n", 2),
                        min=1,
                        max=20,
                    )
                with ui.row().classes("w-full gap-4"):
                    draft_controls["kronos_temperature"] = ui.number(
                        "采样温度",
                        value=_template_float(parameters, "temperature", 0.8),
                        min=0.01,
                        max=2,
                        step=0.1,
                    )
                    draft_controls["kronos_top_p"] = ui.number(
                        "Top P",
                        value=_template_float(parameters, "top_p", 0.9),
                        min=0.01,
                        max=1,
                        step=0.05,
                    )
                    draft_controls["kronos_samples"] = ui.number(
                        "采样路径",
                        value=_template_positive_int(parameters, "sample_count", 3),
                        min=1,
                        max=20,
                    )
                    draft_controls["kronos_seed"] = ui.number(
                        "随机种子",
                        value=max(0, int(_template_float(parameters, "seed", 42))),
                        min=0,
                    )
            if kind in {StrategyLabKind.RULE_DSL, StrategyLabKind.KRONOS_FORECAST}:
                ui.label(
                    "已保存模板会自动带入参数；本次回测仍可修改，实际参数会写入回测快照。"
                ).classes("text-xs text-slate-500")
            if template is not None:
                ui.label(
                    f"已载入模板：{template.name} · 版本 {template.strategy_version}；信号标的、目标 ETF 和资金占比仍可调整。"
                ).classes("text-xs text-indigo-700")
            ui.button("加入策略组合", on_click=add_strategy, icon="add").props("outline")

        def select_template() -> None:
            template = templates_by_id.get(str(template_select.value))
            active_template["value"] = template
            if template is not None:
                strategy_select.value = template.strategy.value
            strategy_form.refresh()

        def select_strategy_kind() -> None:
            template = active_template["value"]
            if template is not None and str(strategy_select.value) != template.strategy.value:
                active_template["value"] = None
                template_select.value = None
            strategy_form.refresh()

        template_select.on_value_change(lambda _: select_template())
        strategy_select.on_value_change(lambda _: select_strategy_kind())
        strategy_form()
        ui.separator()
        strategy_cards()
        readiness_label = ui.label("").classes("text-sm")

        def configuration() -> StrategyLabConfiguration:
            try:
                initial_cash_weight, initial_positions = initial_allocation()
                benchmark = next(
                    (
                        strategy.signal_instrument
                        for strategy in configured_strategies
                        if strategy.signal_instrument is not None
                    ),
                    InstrumentId.parse(default_benchmark),
                )
                return StrategyLabConfiguration(
                    strategies=tuple(configured_strategies),
                    benchmark=benchmark,
                    start=date.fromisoformat(str(start_input.value)),
                    end=date.fromisoformat(str(end_input.value)),
                    initial_cash=Decimal(str(cash_input.value)).quantize(Decimal("0.01")),
                    commission_rate=Decimal(str(commission_input.value)),
                    minimum_commission=Decimal(str(minimum_commission_input.value)),
                    slippage_bps=Decimal(str(slippage_input.value)),
                    execution_timing=ExecutionTiming(str(timing_select.value)),
                    rebalance_mode=StrategyLabRebalanceMode(str(rebalance_select.value)),
                    rebalance_drift=Decimal(str(drift_input.value)) / Decimal("100"),
                    minimum_trade_amount=Decimal(str(minimum_trade_input.value)).quantize(
                        Decimal("0.01")
                    ),
                    initial_cash_weight=initial_cash_weight,
                    initial_positions=initial_positions,
                )
            except (InvalidOperation, TypeError, ValueError) as error:
                raise ValueError("BACKTEST_CONFIGURATION_INVALID") from error

        def start_backtest() -> None:
            try:
                model.start(configuration())
            except Exception as error:
                ui.notify(
                    f"回测未启动：{getattr(error, 'code', str(error) or 'BACKTEST_SUBMISSION_FAILED')}",
                    type="negative",
                )
                return
            ui.notify("回测任务已启动。", type="positive")
            run_button.disable()
            status_panel.refresh()
            poll_timer.activate()

        with ui.row().classes("items-center gap-3"):
            run_button = ui.button("运行回测", on_click=start_backtest, icon="play_arrow").props(
                "color=primary"
            )
            ui.button(
                "前往策略实验室",
                on_click=lambda: ui.navigate.to("/strategies"),
                icon="science",
            ).props("flat")
            ui.button(
                "前往标的池添加 ETF",
                on_click=lambda: ui.navigate.to("/watchlists"),
                icon="playlist_add",
            ).props("flat")
        refresh_readiness()

    def refresh_history_and_result() -> None:
        history_panel.refresh()
        status_panel.refresh()

    def request_delete_history(run_id: str) -> None:
        with ui.dialog() as dialog, ui.card():
            ui.label("清理这条回测记录？").classes("font-semibold")
            ui.label(f"运行 ID：{run_id}").classes("text-xs text-slate-500 font-mono")
            ui.label("将清理任务记录和结果入口；运行中的任务不能清理。").classes(
                "text-sm text-slate-600"
            )

            def confirm() -> None:
                try:
                    deleted = model.delete_history(run_id)
                except Exception as error:
                    ui.notify(f"回测记录清理失败：{error}", type="negative")
                else:
                    ui.notify(
                        "回测记录已清理。" if deleted else "回测记录不存在。",
                        type="positive" if deleted else "warning",
                    )
                dialog.close()
                refresh_history_and_result()

            with ui.row().classes("justify-end gap-2"):
                ui.button("取消", on_click=dialog.close).props("flat")
                ui.button("确认清理", on_click=confirm, color="negative")
        dialog.open()

    def request_clear_history() -> None:
        with ui.dialog() as dialog, ui.card():
            ui.label("清理已结束的回测任务？").classes("font-semibold")
            ui.label("运行中或等待中的任务会被保留，也不会影响行情数据。").classes(
                "text-sm text-slate-600"
            )

            def confirm() -> None:
                try:
                    deleted = model.clear_history()
                except Exception:
                    ui.notify("回测任务历史清理失败。", type="negative")
                else:
                    ui.notify(f"已清理 {deleted} 条回测记录。", type="positive")
                dialog.close()
                refresh_history_and_result()

            with ui.row().classes("justify-end gap-2"):
                ui.button("取消", on_click=dialog.close).props("flat")
                ui.button("确认清理", on_click=confirm, color="negative")
        dialog.open()

    @ui.refreshable
    def history_panel() -> None:
        try:
            state = model.state()
        except Exception:
            ui.label("回测任务历史读取失败，请查看本地日志。").classes("text-red-700")
            return
        with ui.card().classes("w-full border border-slate-200 shadow-none"):
            with ui.row().classes("w-full items-center justify-between gap-3"):
                ui.label("回测任务历史").classes("text-lg font-semibold")
                ui.button(
                    "清理已结束记录",
                    on_click=request_clear_history,
                    icon="delete_sweep",
                ).props("outline color=negative")
            with ui.row().classes("w-full items-end gap-2"):
                query_input = ui.input("按运行 ID 查询").classes("w-full max-w-md")

                def query() -> None:
                    try:
                        model.set_history_query(str(query_input.value or ""))
                    except ValueError:
                        ui.notify("查询内容无效。", type="negative")
                        return
                    history_panel.refresh()

                def reset_query() -> None:
                    model.set_history_query("")
                    history_panel.refresh()

                def view_report(run_id: str) -> None:
                    model.select_report(run_id)
                    status_panel.refresh()

                def change_page(page: int) -> None:
                    model.set_history_page(page)
                    history_panel.refresh()

                ui.button("查询", on_click=query, icon="search").props("outline")
                ui.button("重置", on_click=reset_query).props("flat")
            ui.label(
                f"共 {state.history_total_items} 条；成功任务可以查看结果，运行中的任务不能清理。"
            ).classes("text-sm text-slate-600")
            if not state.history:
                ui.label("暂无匹配的回测任务。").classes("text-sm text-slate-500")
            status_labels = {
                TaskStatus.QUEUED: "等待中",
                TaskStatus.RUNNING: "运行中",
                TaskStatus.CANCELLATION_REQUESTED: "停止中",
                TaskStatus.CANCELLED: "已停止",
                TaskStatus.SUCCEEDED: "成功",
                TaskStatus.FAILED: "失败",
            }
            status_classes = {
                TaskStatus.QUEUED: "text-blue-700",
                TaskStatus.RUNNING: "text-blue-700",
                TaskStatus.CANCELLATION_REQUESTED: "text-amber-700",
                TaskStatus.CANCELLED: "text-slate-600",
                TaskStatus.SUCCEEDED: "text-emerald-700",
                TaskStatus.FAILED: "text-red-700",
            }
            for item in state.history:
                with ui.row().classes("w-full items-center gap-4 border-t border-slate-100 pt-3"):
                    with ui.column().classes("gap-1 grow"):
                        with ui.row().classes("items-center gap-3"):
                            ui.label(item.run_id).classes("text-xs font-mono text-slate-500")
                            ui.label(status_labels[item.status]).classes(
                                f"text-sm font-medium {status_classes[item.status]}"
                            )
                        details = [
                            f"提交 {item.submitted_at.astimezone().strftime('%Y-%m-%d %H:%M:%S')}"
                        ]
                        if item.strategy_count:
                            details.append(f"{item.strategy_count} 个策略")
                        if item.target_count:
                            details.append(f"{item.target_count} 个 ETF")
                        ui.label(" · ".join(details)).classes("text-sm text-slate-600")
                        if item.failure_code is not None:
                            ui.label(f"错误：{item.failure_code}").classes("text-xs text-red-700")
                    view_button = ui.button(
                        "查看结果",
                        on_click=lambda _, run_id=item.run_id: view_report(run_id),
                        icon="visibility",
                    ).props("flat")
                    if not item.has_report:
                        view_button.disable()
                    delete_button = ui.button(
                        "清理",
                        on_click=lambda _, run_id=item.run_id: request_delete_history(run_id),
                        icon="delete_outline",
                    ).props("flat color=negative")
                    if item.status not in _TERMINAL_STATUSES:
                        delete_button.disable()
            if state.history_total_pages > 1:
                with ui.row().classes("w-full justify-end items-center gap-2"):
                    previous = ui.button(
                        icon="chevron_left",
                        on_click=lambda: change_page(state.history_page - 1),
                    ).props("flat round")
                    if state.history_page <= 1:
                        previous.disable()
                    ui.label(f"第 {state.history_page} / {state.history_total_pages} 页").classes(
                        "text-sm text-slate-600"
                    )
                    following = ui.button(
                        icon="chevron_right",
                        on_click=lambda: change_page(state.history_page + 1),
                    ).props("flat round")
                    if state.history_page >= state.history_total_pages:
                        following.disable()

    @ui.refreshable
    def status_panel() -> None:
        try:
            state = model.state()
        except Exception:
            ui.label("回测状态读取失败，请查看本地日志。").classes("text-red-700")
            return
        task = state.active_task
        if task is not None:
            ui.label(f"当前任务：{task_status_label(task.status)}").classes("font-medium")
        if state.failure_code is not None:
            ui.label(f"回测失败：{state.failure_code}").classes("text-red-700")
        base_report = state.active_report or state.latest_report
        if base_report is None:
            ui.label(
                "回测结果默认不加载；请在任务历史中点击“查看结果”。加载后会缓存报告与基准曲线。"
            ).classes("text-sm text-slate-500")
            return
        result_start = base_report.result.ledger[0].trading_day
        result_end = base_report.result.ledger[-1].trading_day
        covered_benchmarks = {
            str(item.instrument): benchmark_options[str(item.instrument)]
            for item in initial_state.instruments
            if item.first_day <= result_start and item.last_day >= result_end
        }
        snapshot_benchmark = str(
            base_report.snapshot.instrument_pool.get("benchmark", default_benchmark)
        )
        selected_benchmark = selected_result_benchmarks.get(base_report.run_id, snapshot_benchmark)
        if selected_benchmark not in covered_benchmarks:
            selected_benchmark = next(iter(covered_benchmarks), snapshot_benchmark)
        comparison_error: str | None = None
        try:
            report = model.compare_report(
                base_report.run_id,
                InstrumentId.parse(selected_benchmark),
            )
        except Exception:
            report = base_report
            comparison_error = "所选基准无法覆盖该回测区间。"
        execution = str(report.snapshot.market_rule_configuration["execution"])
        execution_label = (
            "下一交易日开盘" if execution == ExecutionTiming.NEXT_OPEN.value else "下一交易日收盘"
        )
        raw_trade_instruments = report.snapshot.instrument_pool.get("trade_instruments", ())
        trade_instruments: tuple[str, ...] = (
            tuple(str(item) for item in raw_trade_instruments)
            if isinstance(raw_trade_instruments, (tuple, list))
            else ()
        )
        benchmark_text = selected_benchmark
        benchmark_item = by_id.get(benchmark_text)
        benchmark_label = (
            benchmark_item.label
            if benchmark_item is not None
            else f"{common_instrument_name(InstrumentId.parse(benchmark_text)) or benchmark_text}（{benchmark_text}）"
            if benchmark_text != "—"
            else "—"
        )
        buy_markers, sell_markers = _trade_markers(report)
        buy_signal_markers, sell_signal_markers, unfilled_markers = _signal_markers(report)
        with ui.card().classes("w-full border border-slate-200 shadow-none"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("组合回测结果").classes("text-lg font-semibold")
                ui.label(report.run_id).classes("text-xs text-slate-400")
            ui.label(
                f"{len(report.strategies)} 个策略 · {len(trade_instruments)} 个 ETF · "
                f"{execution_label}成交 · "
                f"{report.result.ledger[0].trading_day.isoformat()} 至 "
                f"{report.result.ledger[-1].trading_day.isoformat()}"
            ).classes("text-sm text-slate-600")
            ui.label(
                "△/▽ 是策略产生的买入/卖出信号，B/S 是实际成交，× 表示未完全成交；可拖动底部滑块或滚轮缩放。"
            ).classes("text-xs text-slate-500")
            for strategy in report.strategies:
                raw_targets = strategy.parameters.get("trade_instruments", ())
                targets = (
                    tuple(str(item) for item in raw_targets)
                    if isinstance(raw_targets, (tuple, list))
                    else ()
                )
                ui.label(
                    f"{strategy.sleeve_id} · {strategy_labels.get(strategy.strategy_type, strategy.strategy_type)} · "
                    f"资金 {Decimal(str(strategy.parameters['budget'])) * 100:g}% · "
                    f"目标 {'、'.join(str(item) for item in targets)}"
                ).classes("text-xs text-slate-500")
            metrics = (
                ("总收益", _percent(report.metrics.total_return)),
                ("年化收益", _percent(report.metrics.annualized_return)),
                ("最大回撤", _percent(report.metrics.maximum_drawdown)),
                ("夏普率", _number(report.metrics.sharpe_ratio)),
                ("基准收益", _percent(report.metrics.benchmark_total_return)),
                ("超额收益", _percent(report.metrics.excess_total_return)),
                (
                    "总费用",
                    "—"
                    if report.metrics.total_costs is None
                    else f"¥{report.metrics.total_costs:,.2f}",
                ),
            )
            with ui.row().classes("w-full gap-3"):
                for label, value in metrics:
                    with ui.card().classes(
                        "min-w-36 border border-slate-100 bg-slate-50 shadow-none"
                    ):
                        ui.label(label).classes("text-xs text-slate-500")
                        ui.label(value).classes("text-lg font-semibold")
            with ui.row().classes("w-full items-end gap-3"):
                result_benchmark_select = ui.select(
                    covered_benchmarks,
                    value=selected_benchmark,
                    label="图表比较基准",
                ).classes("min-w-80")

                def change_result_benchmark() -> None:
                    selected = str(result_benchmark_select.value)
                    selected_result_benchmarks[base_report.run_id] = selected
                    status_panel.refresh()

                result_benchmark_select.on_value_change(lambda _: change_result_benchmark())
                ui.label(f"当前：{benchmark_label}").classes("text-sm text-slate-500")
            if comparison_error is not None:
                ui.label(comparison_error).classes("text-sm text-red-700")
            ui.echart(
                thaw_chart_options(
                    equity_chart_options(
                        _normalized_equity(report),
                        benchmark=report.benchmark_curve,
                        drawdown=report.drawdown_curve,
                        buy_markers=buy_markers,
                        sell_markers=sell_markers,
                        buy_signal_markers=buy_signal_markers,
                        sell_signal_markers=sell_signal_markers,
                        unfilled_markers=unfilled_markers,
                    )
                )
            ).classes("w-full h-96")
            final_ledger = report.result.ledger[-1]
            ui.label(
                f"期末总资产 ¥{final_ledger.equity:,.2f} · "
                f"现金 ¥{final_ledger.cash:,.2f} · 成交 {len(report.result.fills)} 笔"
            ).classes("font-medium")
            signal_rows = _signal_rows(report)
            if signal_rows:
                ui.label("信号与成交对照").classes("text-base font-semibold mt-2")
                ui.label(
                    "信号日是策略决定买卖的日期；成交日通常是下一交易日。未成交原因会保留在订单记录中。"
                ).classes("text-xs text-slate-500")
                ui.table(
                    columns=[
                        {"name": "signal_date", "label": "信号日", "field": "signal_date"},
                        {"name": "trade_date", "label": "成交日", "field": "trade_date"},
                        {"name": "side", "label": "方向", "field": "side"},
                        {"name": "instrument", "label": "标的", "field": "instrument"},
                        {"name": "strategies", "label": "触发策略", "field": "strategies"},
                        {"name": "rule", "label": "规则", "field": "rule"},
                        {"name": "quantity", "label": "成交/委托", "field": "quantity"},
                        {"name": "price", "label": "成交价", "field": "price"},
                        {"name": "status", "label": "状态", "field": "status"},
                        {"name": "reason", "label": "未成交原因", "field": "reason"},
                    ],
                    rows=signal_rows,
                    row_key="id",
                    pagination=10,
                ).classes("w-full")
            if report.result.fills:
                ui.label("实际成交明细").classes("text-base font-semibold mt-2")
                rows = [
                    {
                        "id": fill.fill_id,
                        "date": fill.trading_day.isoformat(),
                        "side": "买入" if fill.side.value == "buy" else "卖出",
                        "instrument": str(fill.instrument),
                        "quantity": fill.quantity,
                        "price": str(fill.price),
                        "fee": str(fill.total_fee),
                    }
                    for fill in report.result.fills[-50:]
                ]
                ui.table(
                    columns=[
                        {"name": "date", "label": "成交日期", "field": "date"},
                        {"name": "side", "label": "方向", "field": "side"},
                        {
                            "name": "instrument",
                            "label": "标的",
                            "field": "instrument",
                        },
                        {"name": "quantity", "label": "数量", "field": "quantity"},
                        {"name": "price", "label": "价格", "field": "price"},
                        {"name": "fee", "label": "费用", "field": "fee"},
                    ],
                    rows=rows,
                    row_key="id",
                    pagination=10,
                ).classes("w-full")
            else:
                ui.label("该区间内没有产生可成交订单。").classes("text-sm text-amber-700")

    def poll() -> None:
        try:
            state = model.state()
        except Exception:
            poll_timer.deactivate()
            return
        status_panel.refresh()
        history_panel.refresh()
        if state.active_task is not None and state.active_task.status in _TERMINAL_STATUSES:
            try:
                model.acknowledge(state.active_task)
            except Exception:
                pass
            refresh_readiness()
            poll_timer.deactivate()

    poll_timer: Timer = ui.timer(1.0, poll, active=False)
    history_panel()
    status_panel()
