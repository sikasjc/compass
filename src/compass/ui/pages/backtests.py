from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from threading import RLock
from typing import Protocol, TypeVar, cast

from nicegui import ui
import pandas as pd  # type: ignore[import-untyped]

from compass.analytics.metrics import PerformanceMetrics, calculate_metrics
from compass.analytics.sleeve_accounting import (
    SleeveAccounting,
    calculate_sleeve_accounting,
)
from compass.backtest.engine import BacktestResult
from compass.backtest.snapshot import ManifestReference, RunSnapshot, StrategySnapshot
from compass.services.safe_display import safe_identifier, stable_code
from compass.services.task_manager import Operation, TaskSnapshot, TaskStatus
from compass.ui.components.charts import (
    AttributionPoint,
    CurvePoint,
    MonthlyReturnPoint,
    attribution_chart_options,
    equity_chart_options,
    monthly_return_chart_options,
    thaw_chart_options,
)
from compass.ui.task_status import task_status_label


T = TypeVar("T")
_MISSING = object()


class BacktestPageError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = stable_code(code, label="backtest page error code")
        super().__init__(self.code)


def _boundary_call(code: str, operation: Callable[[], T]) -> T:
    failed = False
    result: object = _MISSING
    try:
        result = operation()
    except Exception:
        failed = True
    if failed:
        raise BacktestPageError(code)
    return cast(T, result)


def _exact_tuple(value: Sequence[T], item_type: type[T], *, label: str) -> tuple[T, ...]:
    checked = tuple(value)
    if any(type(item) is not item_type for item in checked):
        raise TypeError(f"{label} must contain exact {item_type.__name__} values")
    return checked


@dataclass(frozen=True, slots=True)
class BacktestSubmission:
    run_id: str
    configuration_id: str
    strategy_instance_id: str
    manifest_id: str

    def __post_init__(self) -> None:
        for label, value in (
            ("backtest run id", self.run_id),
            ("backtest configuration id", self.configuration_id),
            ("strategy instance id", self.strategy_instance_id),
            ("market manifest id", self.manifest_id),
        ):
            safe_identifier(value, label=label)


@dataclass(frozen=True, slots=True)
class BacktestDataQualityView:
    accepted: bool
    mode: str
    issue_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool:
            raise TypeError("backtest data quality accepted must be an exact bool")
        if type(self.mode) is not str or self.mode not in {"strict", "degraded"}:
            raise ValueError("backtest data quality mode is invalid")
        issues = tuple(self.issue_codes)
        for code in issues:
            stable_code(code, label="backtest data quality issue code")
        if issues != tuple(sorted(set(issues))):
            raise ValueError("backtest data quality issues must be unique and sorted")
        object.__setattr__(self, "issue_codes", issues)


@dataclass(frozen=True, slots=True, init=False)
class BacktestReport:
    run_id: str
    configuration_id: str
    strategy_instance_ids: tuple[str, ...]
    result: BacktestResult
    snapshot: RunSnapshot
    metrics: PerformanceMetrics
    equity_curve: tuple[CurvePoint, ...]
    drawdown_curve: tuple[CurvePoint, ...]
    benchmark_curve: tuple[CurvePoint, ...]
    monthly_returns: tuple[MonthlyReturnPoint, ...]
    sleeve_accounting: SleeveAccounting
    attribution: tuple[AttributionPoint, ...]
    attribution_residuals: tuple[CurvePoint, ...]
    combined_trade_residual: Decimal
    export_available: bool

    @classmethod
    def from_result(
        cls,
        *,
        run_id: str,
        configuration_id: str,
        strategy_instance_ids: Sequence[str],
        result: BacktestResult,
        snapshot: RunSnapshot,
        benchmark_curve: Sequence[CurvePoint],
        export_available: bool,
    ) -> BacktestReport:
        if type(result) is not BacktestResult:
            raise TypeError("result must be an exact BacktestResult")
        result.verify_integrity()
        if type(snapshot) is not RunSnapshot:
            raise TypeError("snapshot must be an exact RunSnapshot")
        strategy_ids = tuple(strategy_instance_ids)
        benchmark = _exact_tuple(benchmark_curve, CurvePoint, label="benchmark curve")
        days = tuple(item.trading_day for item in result.ledger)
        day_labels = tuple(item.isoformat() for item in days)
        index = pd.to_datetime(days)
        if benchmark and tuple(item.day for item in benchmark) != day_labels:
            raise ValueError("benchmark dates must exactly match the result ledger")

        daily_costs = {day: Decimal("0") for day in days}
        daily_gross = {day: Decimal("0") for day in days}
        for fill in result.fills:
            daily_costs[fill.trading_day] += fill.total_fee
            daily_gross[fill.trading_day] += fill.gross_amount
        metric_frame = pd.DataFrame(
            {
                "equity": tuple(item.equity for item in result.ledger),
                "turnover": tuple(
                    float(daily_gross[item.trading_day] / item.equity) for item in result.ledger
                ),
                "costs": tuple(float(daily_costs[item.trading_day]) for item in result.ledger),
            },
            index=index,
        )
        benchmark_series = (
            None
            if not benchmark
            else pd.Series(tuple(item.value for item in benchmark), index=index)
        )
        metrics = calculate_metrics(metric_frame, benchmark_series)
        equity_values = tuple(float(item.equity) for item in result.ledger)
        equity = tuple(
            CurvePoint(day, value) for day, value in zip(day_labels, equity_values, strict=True)
        )
        peak = 0.0
        drawdown_points: list[CurvePoint] = []
        for day, value in zip(day_labels, equity_values, strict=True):
            peak = max(peak, value)
            drawdown_points.append(CurvePoint(day, value / peak - 1.0))
        monthly = tuple(
            MonthlyReturnPoint(month, value) for month, value in metrics.monthly_returns.items()
        )

        sleeve_accounting = calculate_sleeve_accounting(result)
        attribution = tuple(
            AttributionPoint(
                period.trading_day.isoformat(),
                sleeve,
                period.contributions[sleeve],
            )
            for period in sleeve_accounting.periods
            for sleeve in sleeve_accounting.sleeves
        )
        residuals = tuple(
            CurvePoint(period.trading_day.isoformat(), period.residual)
            for period in sleeve_accounting.periods
        )

        report = object.__new__(cls)
        values: dict[str, object] = {
            "run_id": run_id,
            "configuration_id": configuration_id,
            "strategy_instance_ids": strategy_ids,
            "result": result,
            "snapshot": snapshot,
            "metrics": metrics,
            "equity_curve": equity,
            "drawdown_curve": tuple(drawdown_points),
            "benchmark_curve": benchmark,
            "monthly_returns": monthly,
            "sleeve_accounting": sleeve_accounting,
            "attribution": attribution,
            "attribution_residuals": residuals,
            "combined_trade_residual": Decimal(str(sleeve_accounting.combined_residual)),
            "export_available": export_available,
        }
        for name, attribute_value in values.items():
            object.__setattr__(report, name, attribute_value)
        report.__post_init__()
        return report

    def __post_init__(self) -> None:
        safe_identifier(self.run_id, label="backtest report run id")
        safe_identifier(self.configuration_id, label="backtest configuration id")
        strategy_ids = tuple(self.strategy_instance_ids)
        for strategy_id in strategy_ids:
            safe_identifier(strategy_id, label="strategy instance id")
        if not strategy_ids or strategy_ids != tuple(sorted(set(strategy_ids))):
            raise ValueError("strategy instance ids must be non-empty, unique and sorted")
        if type(self.result) is not BacktestResult:
            raise TypeError("result must be an exact BacktestResult")
        self.result.verify_integrity()
        if type(self.snapshot) is not RunSnapshot:
            raise TypeError("snapshot must be an exact RunSnapshot")
        if type(self.metrics) is not PerformanceMetrics:
            raise TypeError("metrics must be exact PerformanceMetrics")
        if self.result.run_id != self.run_id or self.snapshot.run_id != self.run_id:
            raise ValueError("report, result and snapshot run identities must match")
        self.snapshot.verify_integrity()
        self._data_quality()
        snapshot_strategy_ids = tuple(item.sleeve_id for item in self.snapshot.strategies)
        if snapshot_strategy_ids != strategy_ids:
            raise ValueError("report strategy ids must match the reproducibility snapshot")
        equity = _exact_tuple(self.equity_curve, CurvePoint, label="equity curve")
        drawdown = _exact_tuple(self.drawdown_curve, CurvePoint, label="drawdown curve")
        benchmark = _exact_tuple(self.benchmark_curve, CurvePoint, label="benchmark curve")
        monthly = _exact_tuple(self.monthly_returns, MonthlyReturnPoint, label="monthly returns")
        if type(self.sleeve_accounting) is not SleeveAccounting:
            raise TypeError("sleeve accounting must be an exact canonical value")
        attribution = _exact_tuple(self.attribution, AttributionPoint, label="attribution")
        residuals = _exact_tuple(
            self.attribution_residuals, CurvePoint, label="attribution residuals"
        )
        if not equity:
            raise ValueError("equity curve must not be empty")
        equity_days = tuple(item.day for item in equity)
        ledger_days = tuple(item.trading_day.isoformat() for item in self.result.ledger)
        if equity_days != ledger_days:
            raise ValueError("equity curve dates must match the result ledger")
        if tuple(item.day for item in drawdown) != equity_days:
            raise ValueError("drawdown dates must match the equity curve")
        if benchmark and tuple(item.day for item in benchmark) != equity_days:
            raise ValueError("benchmark dates must match the equity curve")
        expected_monthly = tuple(
            MonthlyReturnPoint(month, value)
            for month, value in self.metrics.monthly_returns.items()
        )
        if monthly != expected_monthly:
            raise ValueError("monthly return points must match PerformanceMetrics")
        attribution_keys = tuple((item.day, item.sleeve) for item in attribution)
        if attribution_keys != tuple(sorted(set(attribution_keys))):
            raise ValueError("attribution points must be unique and sorted")
        if tuple(item.day for item in residuals) != equity_days:
            raise ValueError("attribution residual dates must match the equity curve")
        if self.combined_trade_residual != Decimal(str(sum(item.value for item in residuals))):
            raise ValueError("combined trade residual must reconcile attribution residuals")
        if type(self.combined_trade_residual) is not Decimal:
            raise TypeError("combined trade residual must be an exact Decimal")
        if not self.combined_trade_residual.is_finite():
            raise ValueError("combined trade residual must be finite")
        if type(self.export_available) is not bool:
            raise TypeError("export availability must be an exact bool")
        object.__setattr__(self, "strategy_instance_ids", strategy_ids)
        object.__setattr__(self, "equity_curve", equity)
        object.__setattr__(self, "drawdown_curve", drawdown)
        object.__setattr__(self, "benchmark_curve", benchmark)
        object.__setattr__(self, "monthly_returns", monthly)
        object.__setattr__(self, "attribution", attribution)
        object.__setattr__(self, "attribution_residuals", residuals)

    def verify_integrity(self) -> None:
        self.__post_init__()
        canonical = type(self).from_result(
            run_id=self.run_id,
            configuration_id=self.configuration_id,
            strategy_instance_ids=self.strategy_instance_ids,
            result=self.result,
            snapshot=self.snapshot,
            benchmark_curve=self.benchmark_curve,
            export_available=self.export_available,
        )
        for name in (
            "run_id",
            "configuration_id",
            "strategy_instance_ids",
            "result",
            "snapshot",
            "metrics",
            "equity_curve",
            "drawdown_curve",
            "benchmark_curve",
            "monthly_returns",
            "sleeve_accounting",
            "attribution",
            "attribution_residuals",
            "combined_trade_residual",
            "export_available",
        ):
            if getattr(self, name) != getattr(canonical, name):
                raise ValueError(f"backtest report canonical field mismatch: {name}")

    @property
    def warnings(self) -> tuple[str, ...]:
        self.result.verify_integrity()
        return self.result.warnings

    @property
    def fee_profile_ids(self) -> tuple[str, ...]:
        return self.result.used_profile_ids

    def _data_quality(self) -> BacktestDataQualityView:
        quality = self.snapshot.data_quality
        if set(quality) not in (
            {"accepted", "mode", "issue_codes"},
            {"accepted", "mode", "issue_codes", "dataset_provenance"},
        ):
            raise ValueError("snapshot data quality has unsupported fields")
        issue_codes = quality["issue_codes"]
        if not isinstance(issue_codes, Sequence) or isinstance(issue_codes, str):
            raise TypeError("snapshot data quality issue codes must be a sequence")
        return BacktestDataQualityView(
            accepted=cast(bool, quality["accepted"]),
            mode=cast(str, quality["mode"]),
            issue_codes=tuple(cast(Sequence[str], issue_codes)),
        )

    @property
    def data_quality(self) -> BacktestDataQualityView:
        return self._data_quality()

    @property
    def manifests(self) -> tuple[ManifestReference, ...]:
        return self.snapshot.market_manifests

    @property
    def strategies(self) -> tuple[StrategySnapshot, ...]:
        return self.snapshot.strategies

    @property
    def comparison_available(self) -> bool:
        return bool(self.benchmark_curve)


@dataclass(frozen=True, slots=True)
class BacktestPageState:
    reports: tuple[BacktestReport, ...]
    active_task: TaskSnapshot | None
    active_status: str
    active_report: BacktestReport | None
    failure_code: str | None = None
    error_id: str | None = None

    def __post_init__(self) -> None:
        reports = tuple(self.reports)
        if any(type(item) is not BacktestReport for item in reports):
            raise TypeError("reports must contain exact BacktestReport values")
        for report in reports:
            report.verify_integrity()
        ids = tuple(item.run_id for item in reports)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("backtest reports must be unique and sorted")
        if self.active_task is not None and type(self.active_task) is not TaskSnapshot:
            raise TypeError("active task must be an exact TaskSnapshot or None")
        safe_status = self.active_status
        if safe_status not in {"未运行", *(task_status_label(item) for item in TaskStatus)}:
            raise ValueError("active status is invalid")
        if self.active_task is None:
            if self.active_status not in {"未运行", "失败"} or self.active_report is not None:
                raise ValueError("task-free backtest state is inconsistent")
        else:
            if self.active_status != task_status_label(self.active_task.status):
                raise ValueError("active status must match task status")
            succeeded = self.active_task.status is TaskStatus.SUCCEEDED
            if succeeded != (self.active_report is not None):
                raise ValueError("only a succeeded task may expose a complete report")
        if self.active_report is not None and type(self.active_report) is not BacktestReport:
            raise TypeError("active report must be an exact BacktestReport or None")
        if self.active_report is not None:
            self.active_report.verify_integrity()
        failed = self.active_status == "失败"
        if failed:
            if self.failure_code is None or self.error_id is None:
                raise ValueError("failed backtest state requires safe failure identity")
            stable_code(self.failure_code, label="backtest failure code")
            safe_identifier(self.error_id, label="backtest error id")
        elif self.failure_code is not None or self.error_id is not None:
            raise ValueError("only failed state may expose failure identity")
        object.__setattr__(self, "reports", reports)


class BacktestGateway(Protocol):
    def list_reports(self) -> Sequence[BacktestReport]: ...
    def run(self, submission: BacktestSubmission) -> None: ...
    def report(self, run_id: str) -> BacktestReport | None: ...


class TaskGateway(Protocol):
    def submit(self, name: str, heavy: bool, operation: Operation) -> TaskSnapshot: ...
    def status(self, task_id: str) -> TaskSnapshot: ...
    def cancel(self, task_id: str) -> TaskSnapshot: ...


class BacktestExportGateway(Protocol):
    def export_backtest(self, run_id: str) -> tuple[Path, Path]: ...


class BacktestPageModel:
    def __init__(
        self,
        gateway: BacktestGateway,
        tasks: TaskGateway,
        *,
        exports: BacktestExportGateway | None = None,
    ) -> None:
        self._gateway = gateway
        self._tasks = tasks
        self._exports = exports
        self._active_submission: BacktestSubmission | None = None
        self._active_task: TaskSnapshot | None = None
        self._generation = 0
        self._lock = RLock()

    @property
    def export_available(self) -> bool:
        return self._exports is not None

    def with_exports(self, exports: BacktestExportGateway) -> BacktestPageModel:
        if not callable(getattr(exports, "export_backtest", None)):
            raise TypeError("exports must implement the backtest export boundary")
        composed = BacktestPageModel(
            self._gateway,
            self._tasks,
            exports=exports,
        )
        with self._lock:
            composed._active_submission = self._active_submission
            composed._active_task = self._active_task
            composed._generation = self._generation
        return composed

    def export_report(self, run_id: str) -> tuple[str, str]:
        checked = safe_identifier(run_id, label="backtest export run id")
        if self._exports is None:
            raise BacktestPageError("BACKTEST_EXPORT_UNAVAILABLE")
        exports = self._exports

        def export() -> tuple[str, str]:
            paths = exports.export_backtest(checked)
            if (
                type(paths) is not tuple
                or len(paths) != 2
                or any(not isinstance(path, Path) for path in paths)
            ):
                raise TypeError("backtest exporter must return two exact paths")
            expected_names = (
                f"backtest-{checked}.csv",
                f"backtest-{checked}.json",
            )
            if (
                tuple(path.name for path in paths) != expected_names
                or any(not path.is_absolute() or path.resolve() != path for path in paths)
                or paths[0].parent != paths[1].parent
                or any(not path.is_file() for path in paths)
            ):
                raise ValueError("backtest exporter returned invalid artifacts")
            return expected_names

        return _boundary_call("BACKTEST_EXPORT_FAILED", export)

    @staticmethod
    def _task(
        value: object,
        *,
        expected_name: str,
        task_id: str | None = None,
    ) -> TaskSnapshot:
        if type(value) is not TaskSnapshot:
            raise TypeError("task gateway must return an exact TaskSnapshot")
        assert isinstance(value, TaskSnapshot)
        if value.name != expected_name or value.heavy is not True:
            raise ValueError("backtest task identity is inconsistent")
        if task_id is not None and value.task_id != task_id:
            raise ValueError("backtest task id changed")
        return value

    def start(self, submission: BacktestSubmission) -> TaskSnapshot:
        if type(submission) is not BacktestSubmission:
            raise TypeError("submission must be an exact BacktestSubmission")
        expected_name = f"backtest:{submission.run_id}"
        with self._lock:
            self._generation += 1
            generation = self._generation

        def submit() -> TaskSnapshot:
            return self._task(
                self._tasks.submit(
                    expected_name,
                    True,
                    lambda: self._gateway.run(submission),
                ),
                expected_name=expected_name,
            )

        task = _boundary_call("BACKTEST_SUBMISSION_FAILED", submit)
        with self._lock:
            if self._generation == generation:
                self._active_submission = submission
                self._active_task = task
        return task

    def cancel_active(self) -> TaskSnapshot:
        with self._lock:
            submission = self._active_submission
            current = self._active_task
            generation = self._generation
        if submission is None or current is None:
            raise BacktestPageError("BACKTEST_CANCELLATION_UNAVAILABLE")
        expected_name = f"backtest:{submission.run_id}"
        cancelled = _boundary_call(
            "BACKTEST_CANCELLATION_FAILED",
            lambda: self._task(
                self._tasks.cancel(current.task_id),
                expected_name=expected_name,
                task_id=current.task_id,
            ),
        )
        if cancelled.status not in {
            TaskStatus.CANCELLATION_REQUESTED,
            TaskStatus.CANCELLED,
        }:
            raise BacktestPageError("BACKTEST_CANCELLATION_FAILED")
        with self._lock:
            if (
                self._generation == generation
                and self._active_submission == submission
                and self._active_task == current
            ):
                self._active_task = cancelled
        return cancelled

    def state(self) -> BacktestPageState:
        reports = _boundary_call(
            "BACKTEST_REPORTS_UNAVAILABLE",
            lambda: self._reports(self._gateway.list_reports()),
        )
        with self._lock:
            submission = self._active_submission
            submitted_task = self._active_task
        if submission is None or submitted_task is None:
            return BacktestPageState(reports, None, "未运行", None)
        expected_name = f"backtest:{submission.run_id}"
        active = _boundary_call(
            "BACKTEST_TASK_STATUS_UNAVAILABLE",
            lambda: self._task(
                self._tasks.status(submitted_task.task_id),
                expected_name=expected_name,
                task_id=submitted_task.task_id,
            ),
        )
        report: BacktestReport | None = None
        if active.status is TaskStatus.SUCCEEDED:
            report = _boundary_call(
                "BACKTEST_RESULT_UNAVAILABLE",
                lambda: self._matching_report(self._gateway.report(submission.run_id), submission),
            )
        failure_code: str | None = None
        error_id: str | None = None
        if active.status is TaskStatus.FAILED:
            assert active.failure is not None
            failure_code = active.failure.code
            error_id = active.failure.error_id
        public_task = None if active.status is TaskStatus.FAILED else active
        return _boundary_call(
            "BACKTEST_STATE_UNAVAILABLE",
            lambda: BacktestPageState(
                reports,
                public_task,
                task_status_label(active.status),
                report,
                failure_code,
                error_id,
            ),
        )

    @staticmethod
    def _reports(values: Sequence[BacktestReport]) -> tuple[BacktestReport, ...]:
        reports = tuple(values)
        if any(type(item) is not BacktestReport for item in reports):
            raise TypeError("gateway reports must be exact BacktestReport values")
        for report in reports:
            report.verify_integrity()
        return tuple(sorted(reports, key=lambda item: item.run_id))

    @staticmethod
    def _matching_report(value: object, submission: BacktestSubmission) -> BacktestReport:
        if type(value) is not BacktestReport:
            raise TypeError("succeeded task requires an exact BacktestReport")
        assert isinstance(value, BacktestReport)
        value.verify_integrity()
        if (
            value.run_id != submission.run_id
            or value.configuration_id != submission.configuration_id
            or submission.strategy_instance_id not in value.strategy_instance_ids
            or submission.manifest_id not in value.snapshot.market_manifest_ids
        ):
            raise ValueError("backtest result identity does not match its submission")
        return value


def render_backtests_page(model: BacktestPageModel | None) -> None:
    if model is None:
        ui.label("回测服务未配置；当前不会虚构任务、曲线或绩效结果。")
        return
    try:
        state = model.state()
    except Exception:
        ui.label("回测状态读取失败，请查看本地脱敏日志。").classes("text-red-700")
        return
    ui.label(f"任务状态：{state.active_status}").classes("font-semibold")
    ui.label("支持状态：" + " / ".join(task_status_label(status) for status in TaskStatus))
    run_input = ui.input("运行 ID")
    configuration_input = ui.input("配置 ID")
    strategy_input = ui.input("策略实例 ID")
    manifest_input = ui.input("行情清单 ID")
    action_feedback = ui.label("")

    def start_backtest() -> None:
        try:
            task = model.start(
                BacktestSubmission(
                    run_id=run_input.value,
                    configuration_id=configuration_input.value,
                    strategy_instance_id=strategy_input.value,
                    manifest_id=manifest_input.value,
                )
            )
            action_feedback.set_text(
                f"完整回测已提交：{task.task_id} / {task_status_label(task.status)}"
            )
        except Exception as error:
            action_feedback.set_text(
                f"完整回测提交失败：{getattr(error, 'code', 'BACKTEST_SUBMISSION_FAILED')}"
            )

    def refresh_status() -> None:
        try:
            refreshed = model.state()
            action_feedback.set_text(f"刷新状态：{refreshed.active_status}")
        except Exception as error:
            action_feedback.set_text(
                f"刷新状态失败：{getattr(error, 'code', 'BACKTEST_STATE_UNAVAILABLE')}"
            )

    def cancel_task() -> None:
        try:
            cancelled = model.cancel_active()
            action_feedback.set_text(
                f"请求取消：{cancelled.task_id} / {task_status_label(cancelled.status)}"
            )
        except Exception as error:
            action_feedback.set_text(
                f"取消失败：{getattr(error, 'code', 'BACKTEST_CANCELLATION_FAILED')}"
            )

    ui.button("启动完整回测", on_click=start_backtest)
    ui.button("刷新状态", on_click=refresh_status)
    ui.button("取消任务", on_click=cancel_task)
    if state.failure_code is not None:
        ui.label(f"失败：{state.failure_code} / 错误 ID {state.error_id}").classes("text-red-700")
    report = state.active_report
    if report is None:
        ui.label("当前没有可展示的完整成功结果。")
        return
    with ui.card().classes("w-full border border-slate-200 shadow-none"):
        ui.label(f"运行：{report.run_id}").classes("font-semibold")
        ui.label(f"配置：{report.configuration_id}")
        ui.label(f"快照：{report.snapshot.snapshot_id[:12]}")
        ui.label(f"应用提交：{report.snapshot.app_git_commit}")
        ui.label(f"随机种子：{report.snapshot.random_seed}")
        ui.label("费用档案：" + "、".join(report.fee_profile_ids))
        quality = report.data_quality
        ui.label(f"数据质量：{'通过' if quality.accepted else '未通过'} / {quality.mode}")
        ui.label(
            "数据质量问题：" + ("、".join(quality.issue_codes) if quality.issue_codes else "无")
        )
        for manifest in report.manifests:
            ui.label(f"行情清单：{manifest.manifest_id} / {manifest.content_hash}")
        for strategy in report.strategies:
            ui.label(
                f"策略袖套：{strategy.sleeve_id} / {strategy.strategy_type} / {strategy.strategy_version}"
            )
        metrics = (
            ("总收益", report.metrics.total_return),
            ("年化收益", report.metrics.annualized_return),
            ("年化波动", report.metrics.annualized_volatility),
            ("夏普比率", report.metrics.sharpe_ratio),
            ("最大回撤", report.metrics.maximum_drawdown),
            ("卡玛比率", report.metrics.calmar_ratio),
            ("胜率", report.metrics.win_rate),
            ("总换手", report.metrics.total_turnover),
            ("总费用", report.metrics.total_costs),
            ("基准总收益", report.metrics.benchmark_total_return),
            ("基准年化收益", report.metrics.benchmark_annualized_return),
            ("超额总收益", report.metrics.excess_total_return),
            ("超额年化收益", report.metrics.excess_annualized_return),
        )
        for label, value in metrics:
            ui.label(f"{label}：{'不可用' if value is None else value}")
        ui.label("比较：可用" if report.comparison_available else "比较：不可用")
        ui.echart(
            thaw_chart_options(
                equity_chart_options(
                    report.equity_curve,
                    benchmark=report.benchmark_curve,
                    drawdown=report.drawdown_curve,
                )
            )
        ).classes("w-full h-80")
        ui.echart(thaw_chart_options(monthly_return_chart_options(report.monthly_returns))).classes(
            "w-full h-64"
        )
        ui.echart(thaw_chart_options(attribution_chart_options(report.attribution))).classes(
            "w-full h-64"
        )
        ui.label("月度收益明细")
        for monthly_point in report.monthly_returns:
            ui.label(
                f"{monthly_point.month}："
                f"{'不可用' if monthly_point.value is None else monthly_point.value}"
            )
        ui.label("袖套归因明细")
        for attribution_point in report.attribution:
            ui.label(
                f"{attribution_point.day} / {attribution_point.sleeve} / {attribution_point.value}"
            )
        ui.label("归因残差明细")
        for residual_point in report.attribution_residuals:
            ui.label(f"{residual_point.day} / {residual_point.value}")
        ui.label("警告")
        for warning in report.warnings:
            ui.label(warning)
        ui.label(f"成交：{len(report.result.fills)} 笔")
        for fill in report.result.fills:
            ui.label(
                f"成交 {fill.trading_day.isoformat()} / {fill.instrument} / {fill.side.value} / "
                f"数量 {fill.quantity} / 价格 {fill.price} / 金额 {fill.gross_amount} / "
                f"佣金 {fill.commission} / 印花税 {fill.stamp_duty} / "
                f"过户费 {fill.transfer_fee} / 总费用 {fill.total_fee} / 档案 {fill.profile_id}"
            )
        ui.label(
            f"交易时点/新建仓、费用、取整等残差：{report.combined_trade_residual}"
        )
        if model.export_available:
            ui.label("导出：可用")

            def export_report() -> None:
                try:
                    csv_name, json_name = model.export_report(report.run_id)
                    action_feedback.set_text(f"导出完成：{csv_name}、{json_name}")
                except Exception as error:
                    action_feedback.set_text(
                        f"导出失败：{getattr(error, 'code', 'BACKTEST_EXPORT_FAILED')}"
                    )

            ui.button("导出 CSV/JSON", on_click=export_report)
        else:
            ui.label("导出：不可用")
