from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import PurePath
from threading import RLock
from types import MappingProxyType
from typing import Protocol, TypeVar, cast
from zoneinfo import ZoneInfo

from nicegui import ui

from compass.domain.market import InstrumentId
from compass.domain.trading import TargetIntent
from compass.portfolio.models import AllocationAdjustment
from compass.risk.base import RiskAdjustment
from compass.services.decision_service import (
    DecisionResult,
    DecisionSide,
    EstimatedCosts,
    RebalanceRecommendation,
    StrategyDecisionTrace,
)
from compass.services.intraday_service import IntradaySignal, IntradayState
from compass.services.safe_display import (
    safe_display_text,
    safe_identifier,
    stable_code,
)
from compass.storage.account_repository import StoredAccountSnapshot


SHANGHAI = ZoneInfo("Asia/Shanghai")
T = TypeVar("T")
_MISSING = object()
NO_DECISION_STATUS = "暂无确认决策"
CONFIRMED_STATUS = "已确认"
EXPIRED_STATUS = "已过期"
FAILED_STATUS = "失败"
_ACTION_STATUSES = {"idle", "running", "succeeded", "failed"}
_ACTION_OPERATIONS = {None, "generate", "export"}


@dataclass(frozen=True, slots=True)
class DashboardStrategyChoice:
    strategy_instance_id: str
    label: str

    def __post_init__(self) -> None:
        safe_identifier(self.strategy_instance_id, label="strategy instance id")
        safe_display_text(self.label, label="strategy choice label")


@dataclass(frozen=True, slots=True)
class DashboardManifestChoice:
    manifest_id: str
    bundle_id: str
    instruments: tuple[InstrumentId, ...]
    created_at: datetime

    def __post_init__(self) -> None:
        safe_identifier(self.manifest_id, label="manifest id")
        safe_identifier(self.bundle_id, label="dataset bundle id")
        instruments = tuple(self.instruments)
        if (
            not instruments
            or any(type(item) is not InstrumentId for item in instruments)
            or instruments != tuple(sorted(set(instruments), key=str))
        ):
            raise ValueError("manifest choice instruments must be non-empty, unique and sorted")
        _aware(self.created_at, label="manifest choice created_at")
        object.__setattr__(self, "instruments", instruments)


@dataclass(frozen=True, slots=True)
class DashboardActionReceipt:
    decision_id: str
    strategy_instance_id: str
    manifest_id: str
    generated_at: datetime

    def __post_init__(self) -> None:
        safe_identifier(self.decision_id, label="decision id")
        safe_identifier(self.strategy_instance_id, label="strategy instance id")
        safe_identifier(self.manifest_id, label="manifest id")
        _aware(self.generated_at, label="decision generated_at")


@dataclass(frozen=True, slots=True)
class DashboardExportReceipt:
    decision_id: str
    csv_filename: str
    json_filename: str
    exported_at: datetime

    def __post_init__(self) -> None:
        safe_identifier(self.decision_id, label="decision id")
        for suffix, filename in ((".csv", self.csv_filename), (".json", self.json_filename)):
            safe_identifier(filename, label="decision report filename")
            if PurePath(filename).name != filename or not filename.endswith(suffix):
                raise ValueError("decision report filename is invalid")
        if self.csv_filename.removesuffix(".csv") != self.json_filename.removesuffix(
            ".json"
        ):
            raise ValueError("decision report filenames must share one stem")
        _aware(self.exported_at, label="decision exported_at")


@dataclass(frozen=True, slots=True)
class DashboardActionState:
    status: str
    operation: str | None = None
    strategy_instance_id: str | None = None
    manifest_id: str | None = None
    decision_id: str | None = None
    report_filenames: tuple[str, ...] = ()
    completed_at: datetime | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        if self.status not in _ACTION_STATUSES:
            raise ValueError("dashboard action status is invalid")
        if self.operation not in _ACTION_OPERATIONS:
            raise ValueError("dashboard action operation is invalid")
        for label, value in (
            ("strategy instance id", self.strategy_instance_id),
            ("manifest id", self.manifest_id),
            ("decision id", self.decision_id),
        ):
            if value is not None:
                safe_identifier(value, label=label)
        filenames = tuple(self.report_filenames)
        for filename in filenames:
            safe_identifier(filename, label="decision report filename")
            if PurePath(filename).name != filename:
                raise ValueError("decision report filename must not contain a path")
        if self.completed_at is not None:
            _aware(self.completed_at, label="dashboard action completed_at")
        if self.error_code is not None:
            stable_code(self.error_code, label="dashboard action error code")
        if self.status == "idle":
            if any(
                value is not None
                for value in (
                    self.operation,
                    self.strategy_instance_id,
                    self.manifest_id,
                    self.decision_id,
                    self.completed_at,
                    self.error_code,
                )
            ) or filenames:
                raise ValueError("idle dashboard action cannot expose result values")
        elif self.operation is None:
            raise ValueError("non-idle dashboard action requires an operation")
        if self.status == "failed":
            if self.error_code is None or self.completed_at is not None or filenames:
                raise ValueError("failed dashboard action must expose only a safe error code")
        elif self.error_code is not None:
            raise ValueError("only failed dashboard action may expose an error code")
        if self.status == "succeeded":
            if self.decision_id is None or self.completed_at is None:
                raise ValueError("successful dashboard action requires decision identity and time")
            if self.operation == "generate" and (
                self.strategy_instance_id is None
                or self.manifest_id is None
                or filenames
            ):
                raise ValueError("generated decision status requires its selected inputs")
            if self.operation == "export" and len(filenames) != 2:
                raise ValueError("decision export status requires CSV and JSON filenames")
        if self.status == "running" and (
            self.completed_at is not None or self.error_code is not None or filenames
        ):
            raise ValueError("running dashboard action cannot expose a result")
        object.__setattr__(self, "report_filenames", filenames)


class DashboardPageError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = stable_code(code, label="dashboard page error code")
        super().__init__(self.code)


def _boundary_call(code: str, operation: Callable[[], T]) -> T:
    failed = False
    result: object = _MISSING
    try:
        result = operation()
    except Exception:
        failed = True
    if failed:
        raise DashboardPageError(code)
    return cast(T, result)


def _aware(value: object, *, label: str) -> datetime:
    if type(value) is not datetime:
        raise TypeError(f"{label} must be an exact datetime")
    assert isinstance(value, datetime)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value


def _weights(
    value: Mapping[InstrumentId, Decimal], *, label: str
) -> Mapping[InstrumentId, Decimal]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    checked: dict[InstrumentId, Decimal] = {}
    for instrument, weight in value.items():
        if type(instrument) is not InstrumentId:
            raise TypeError(f"{label} keys must be exact InstrumentId values")
        if InstrumentId.parse(str(instrument)) != instrument:
            raise ValueError(f"{label} instruments must be canonical")
        if type(weight) is not Decimal:
            raise TypeError(f"{label} values must be exact Decimal weights")
        if not weight.is_finite() or not Decimal("0") <= weight <= Decimal("1"):
            raise ValueError(f"{label} weights must be finite between zero and one")
        checked[instrument] = weight
    return MappingProxyType(dict(sorted(checked.items(), key=lambda item: str(item[0]))))


@dataclass(frozen=True, slots=True)
class DashboardDataHealth:
    accepted: bool
    mode: str
    issue_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool:
            raise TypeError("data health accepted must be an exact bool")
        if type(self.mode) is not str or self.mode not in {"strict", "degraded"}:
            raise ValueError("data health mode is invalid")
        issues = tuple(self.issue_codes)
        for code in issues:
            stable_code(code, label="data health issue code")
        if issues != tuple(sorted(set(issues))):
            raise ValueError("data health issue codes must be unique and sorted")
        object.__setattr__(self, "issue_codes", issues)


@dataclass(frozen=True, slots=True)
class DashboardFailure:
    code: str
    error_id: str

    def __post_init__(self) -> None:
        stable_code(self.code, label="dashboard failure code")
        safe_identifier(self.error_id, label="dashboard error id")


@dataclass(frozen=True, slots=True)
class DashboardSnapshot:
    data_health: DashboardDataHealth | None
    account: StoredAccountSnapshot | None
    current_drawdown: Decimal | None
    risk_status: str | None
    actual_weights: Mapping[InstrumentId, Decimal]
    target_weights: Mapping[InstrumentId, Decimal]
    intraday: IntradayState | None
    confirmed_decision: DecisionResult | None
    decision_failure: DashboardFailure | None

    def __post_init__(self) -> None:
        if self.data_health is not None and type(self.data_health) is not DashboardDataHealth:
            raise TypeError("data health must be exact DashboardDataHealth or None")
        if self.account is not None and type(self.account) is not StoredAccountSnapshot:
            raise TypeError("account must be an exact StoredAccountSnapshot or None")
        if self.current_drawdown is not None:
            if type(self.current_drawdown) is not Decimal:
                raise TypeError("current drawdown must be an exact Decimal or None")
            if not self.current_drawdown.is_finite() or not Decimal(
                "-1"
            ) <= self.current_drawdown <= Decimal("0"):
                raise ValueError("current drawdown must be finite between minus one and zero")
        if self.risk_status is not None:
            stable_code(self.risk_status, label="dashboard risk status")
        if self.intraday is not None and type(self.intraday) is not IntradayState:
            raise TypeError("intraday state must be exact IntradayState or None")
        if (
            self.confirmed_decision is not None
            and type(self.confirmed_decision) is not DecisionResult
        ):
            raise TypeError("confirmed decision must be exact DecisionResult or None")
        if (
            self.decision_failure is not None
            and type(self.decision_failure) is not DashboardFailure
        ):
            raise TypeError("decision failure must be exact DashboardFailure or None")
        if self.confirmed_decision is not None and self.decision_failure is not None:
            raise ValueError("confirmed decision and decision failure are mutually exclusive")
        actual = _weights(self.actual_weights, label="actual weights")
        target = _weights(self.target_weights, label="target weights")
        if self.confirmed_decision is None:
            if target:
                raise ValueError("target weights require a confirmed decision")
        else:
            decision = self.confirmed_decision
            expected = {item.instrument: item.final_weight for item in decision.recommendations}
            if dict(target) != expected:
                raise ValueError("target weights must exactly match confirmed recommendations")
            expected_actual = {
                item.instrument: item.current_weight for item in decision.recommendations
            }
            if dict(actual) != expected_actual:
                raise ValueError("actual weights must exactly match recommendation current weights")
            if self.account is None:
                raise ValueError("a confirmed decision requires the displayed account snapshot")
            if (
                self.account.account_id != decision.account_id
                or self.account.row_id != decision.account_snapshot_row_id
                or self.account.content_hash != decision.account_snapshot_hash
            ):
                raise ValueError("decision must reference the displayed account snapshot")
        object.__setattr__(self, "actual_weights", actual)
        object.__setattr__(self, "target_weights", target)


@dataclass(frozen=True, slots=True)
class DashboardRecommendationView:
    recommendation: RebalanceRecommendation
    actual_weight: Decimal
    target_weight: Decimal
    actionable: bool

    def __post_init__(self) -> None:
        if type(self.recommendation) is not RebalanceRecommendation:
            raise TypeError("recommendation must be an exact RebalanceRecommendation")
        for label, value in (
            ("actual weight", self.actual_weight),
            ("target weight", self.target_weight),
        ):
            if type(value) is not Decimal:
                raise TypeError(f"{label} must be an exact Decimal")
            if not value.is_finite() or not Decimal("0") <= value <= Decimal("1"):
                raise ValueError(f"{label} must be finite between zero and one")
        if self.target_weight != self.recommendation.final_weight:
            raise ValueError("target weight must match the exact recommendation")
        if self.actual_weight != self.recommendation.current_weight:
            raise ValueError("actual weight must match the exact recommendation")
        if type(self.actionable) is not bool:
            raise TypeError("recommendation actionability must be an exact bool")

    @property
    def instrument(self) -> InstrumentId:
        return self.recommendation.instrument

    @property
    def raw_intents(self) -> tuple[TargetIntent, ...]:
        return self.recommendation.raw_intents

    @property
    def strategy_decisions(self) -> tuple[StrategyDecisionTrace, ...]:
        return self.recommendation.strategy_decisions

    @property
    def allocated_weight(self) -> Decimal:
        return self.recommendation.allocated_weight

    @property
    def allocation_trace(self) -> tuple[AllocationAdjustment, ...]:
        return self.recommendation.allocation_trace

    @property
    def pre_risk_weight(self) -> Decimal:
        return self.recommendation.pre_risk_weight

    @property
    def risk_adjusted_weight(self) -> Decimal:
        return self.recommendation.final_weight

    @property
    def risk_adjustments(self) -> tuple[RiskAdjustment, ...]:
        return self.recommendation.risk_adjustments

    @property
    def current_quantity(self) -> int:
        return self.recommendation.current_quantity

    @property
    def final_quantity(self) -> int:
        return self.recommendation.target_quantity

    @property
    def side(self) -> DecisionSide:
        return self.recommendation.side

    @property
    def reference_price(self) -> Decimal:
        return self.recommendation.reference_price

    @property
    def estimated_execution_price(self) -> Decimal | None:
        return self.recommendation.estimated_execution_price

    @property
    def gross_amount(self) -> Decimal:
        return self.recommendation.gross_amount

    @property
    def costs(self) -> EstimatedCosts:
        return self.recommendation.costs

    @property
    def profile_id(self) -> str:
        return self.recommendation.profile_id

    @property
    def account_snapshot_row_id(self) -> int:
        return self.recommendation.account_snapshot_row_id

    @property
    def account_snapshot_hash(self) -> str:
        return self.recommendation.account_snapshot_hash

    @property
    def market_data_source_at(self) -> datetime:
        return self.recommendation.market_data_source_at

    @property
    def decision_at(self) -> datetime:
        return self.recommendation.decision_at

    @property
    def valid_until(self) -> date:
        return self.recommendation.valid_until

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return self.recommendation.reason_codes


@dataclass(frozen=True, slots=True)
class DashboardPageState:
    data_health: DashboardDataHealth | None
    account: StoredAccountSnapshot | None
    current_drawdown: Decimal | None
    risk_status: str | None
    actual_weights: Mapping[InstrumentId, Decimal]
    target_weights: Mapping[InstrumentId, Decimal]
    temporary_status: str | None
    temporary_signals: tuple[IntradaySignal, ...]
    intraday_source_at: datetime | None
    intraday_stale: bool | None
    decision_status: str
    decision_at: datetime | None
    decision_source_at: datetime | None
    valid_until: date | None
    confirmed_recommendations: tuple[DashboardRecommendationView, ...]
    decision_actionable: bool
    failure_code: str | None
    error_id: str | None

    def __post_init__(self) -> None:
        if self.data_health is not None and type(self.data_health) is not DashboardDataHealth:
            raise TypeError("data health must be exact DashboardDataHealth or None")
        if self.account is not None and type(self.account) is not StoredAccountSnapshot:
            raise TypeError("account must be an exact StoredAccountSnapshot or None")
        if self.current_drawdown is not None:
            if type(self.current_drawdown) is not Decimal:
                raise TypeError("current drawdown must be an exact Decimal or None")
            if not self.current_drawdown.is_finite() or not Decimal(
                "-1"
            ) <= self.current_drawdown <= Decimal("0"):
                raise ValueError("current drawdown must be finite between minus one and zero")
        if self.risk_status is not None:
            stable_code(self.risk_status, label="dashboard risk status")
        actual = _weights(self.actual_weights, label="actual weights")
        target = _weights(self.target_weights, label="target weights")
        if self.temporary_status not in {None, "临时"}:
            raise ValueError("temporary status is invalid")
        if type(self.temporary_signals) is not tuple or any(
            type(item) is not IntradaySignal for item in self.temporary_signals
        ):
            raise TypeError("temporary signals must be exact IntradaySignal values")
        if bool(self.temporary_signals) != (self.temporary_status == "临时"):
            raise ValueError("temporary status must match temporary signal availability")
        if any(item.confirmed is not False for item in self.temporary_signals):
            raise ValueError("temporary signals cannot be confirmed")
        signal_keys = tuple(
            (str(item.instrument), item.strategy_id, item.reason_code)
            for item in self.temporary_signals
        )
        if signal_keys != tuple(sorted(set(signal_keys))):
            raise ValueError("temporary signals must be unique and deterministically sorted")
        if self.intraday_source_at is None:
            if self.temporary_signals or self.intraday_stale is not None:
                raise ValueError("intraday details require a source timestamp")
        else:
            _aware(self.intraday_source_at, label="intraday source_at")
            if type(self.intraday_stale) is not bool:
                raise TypeError("intraday stale must be an exact bool when intraday exists")
            if any(item.source_at < self.intraday_source_at for item in self.temporary_signals):
                raise ValueError("temporary signal cannot precede the visible source timestamp")
        if self.decision_status not in {
            NO_DECISION_STATUS,
            CONFIRMED_STATUS,
            EXPIRED_STATUS,
            FAILED_STATUS,
        }:
            raise ValueError("decision status is invalid")
        recommendations = tuple(self.confirmed_recommendations)
        if any(type(item) is not DashboardRecommendationView for item in recommendations):
            raise TypeError("confirmed recommendations must contain exact view values")
        instruments = tuple(item.instrument for item in recommendations)
        if len(set(instruments)) != len(instruments):
            raise ValueError("confirmed recommendations must be unique by instrument")
        if type(self.decision_actionable) is not bool:
            raise TypeError("decision actionability must be an exact bool")
        if self.decision_status == FAILED_STATUS:
            if self.failure_code is None or self.error_id is None:
                raise ValueError("failed decision requires safe failure identity")
            stable_code(self.failure_code, label="dashboard failure code")
            safe_identifier(self.error_id, label="dashboard error id")
        elif self.failure_code is not None or self.error_id is not None:
            raise ValueError("only failed decision exposes failure identity")
        if self.decision_status in {NO_DECISION_STATUS, FAILED_STATUS}:
            if (
                self.decision_at is not None
                or self.decision_source_at is not None
                or self.valid_until is not None
                or recommendations
                or target
                or self.decision_actionable
            ):
                raise ValueError("decision-free state cannot expose confirmed decision values")
        else:
            decision_at = _aware(self.decision_at, label="decision_at")
            source_at = _aware(self.decision_source_at, label="decision source_at")
            if source_at > decision_at:
                raise ValueError("decision source cannot follow the decision")
            if type(self.valid_until) is not date:
                raise TypeError("valid_until must be an exact date")
            if not recommendations:
                raise ValueError("confirmed and expired decisions require recommendations")
            if self.account is None:
                raise ValueError("confirmed and expired decisions require an account snapshot")
            expected_target = {item.instrument: item.target_weight for item in recommendations}
            if dict(target) != expected_target:
                raise ValueError("target weights must exactly match visible recommendations")
            expected_actual = {item.instrument: item.actual_weight for item in recommendations}
            if dict(actual) != expected_actual:
                raise ValueError("actual weights must exactly match visible recommendations")
            if any(
                item.actual_weight != actual.get(item.instrument, Decimal("0"))
                or item.decision_at != decision_at
                or item.market_data_source_at != source_at
                or item.valid_until != self.valid_until
                or item.account_snapshot_row_id != self.account.row_id
                or item.account_snapshot_hash != self.account.content_hash
                for item in recommendations
            ):
                raise ValueError("recommendation provenance must match the dashboard state")
            expected_actionable = self.decision_status == CONFIRMED_STATUS
            if self.decision_actionable is not expected_actionable or any(
                item.actionable is not expected_actionable for item in recommendations
            ):
                raise ValueError("decision and recommendation actionability must match status")
        object.__setattr__(self, "actual_weights", actual)
        object.__setattr__(self, "target_weights", target)
        object.__setattr__(self, "temporary_signals", tuple(self.temporary_signals))
        object.__setattr__(self, "confirmed_recommendations", recommendations)


class DashboardGateway(Protocol):
    def state(self) -> DashboardSnapshot: ...


class DashboardActionGateway(Protocol):
    def list_strategies(self) -> tuple[DashboardStrategyChoice, ...]: ...

    def list_manifests(
        self,
        strategy_instance_id: str,
    ) -> tuple[DashboardManifestChoice, ...]: ...

    def generate_close_decision(
        self,
        strategy_instance_id: str,
        manifest_id: str,
    ) -> DashboardActionReceipt: ...

    def export_decision(self, decision_id: str | None) -> DashboardExportReceipt: ...


_ACTION_ERROR_CODES = {
    "ACCOUNT_SNAPSHOT_MISSING": "DASHBOARD_ACCOUNT_MISSING",
    "CALENDAR_CACHE_INVALID": "DASHBOARD_RULE_METADATA_UNAVAILABLE",
    "CALENDAR_RANGE_UNAVAILABLE": "DASHBOARD_RULE_METADATA_UNAVAILABLE",
    "DAILY_CLOSE_INCOMPLETE": "DASHBOARD_CLOSE_DATA_UNAVAILABLE",
    "DAILY_DATA_NOT_ACCEPTED": "DASHBOARD_CLOSE_DATA_UNAVAILABLE",
    "DATASET_BUNDLE_INTEGRITY": "DASHBOARD_MANIFEST_INELIGIBLE",
    "DATASET_BUNDLE_MISSING": "DASHBOARD_MANIFEST_INELIGIBLE",
    "DECISION_CALENDAR_MISMATCH": "DASHBOARD_CLOSE_DATA_UNAVAILABLE",
    "DECISION_DAILY_DATA_INCOMPLETE": "DASHBOARD_CLOSE_DATA_UNAVAILABLE",
    "DECISION_DAILY_DATA_STALE": "DASHBOARD_CLOSE_DATA_UNAVAILABLE",
    "DECISION_FEE_PROFILE_UNCONFIRMED": "DASHBOARD_FEE_UNCONFIRMED",
    "DECISION_MARKET_RULE_UNAVAILABLE": "DASHBOARD_RULE_METADATA_UNAVAILABLE",
    "DECISION_NEXT_SESSION_UNAVAILABLE": "DASHBOARD_RULE_METADATA_UNAVAILABLE",
    "DECISION_POOL_DATA_MISSING": "DASHBOARD_MANIFEST_INELIGIBLE",
    "DECISION_PROVENANCE_MISSING": "DASHBOARD_CLOSE_DATA_UNAVAILABLE",
    "DECISION_STRATEGY_UNAVAILABLE": "DASHBOARD_STRATEGY_MISSING",
    "DECISION_WATCHLIST_DISABLED": "DASHBOARD_STRATEGY_MISSING",
    "HELD_INSTRUMENT_DATA_MISSING": "DASHBOARD_CLOSE_DATA_UNAVAILABLE",
    "MARKET_DATA_FROM_FUTURE": "DASHBOARD_CLOSE_DATA_UNAVAILABLE",
    "MARKET_DATA_STALE": "DASHBOARD_CLOSE_DATA_UNAVAILABLE",
    "MARKET_RULE_PROFILE_MISSING": "DASHBOARD_RULE_METADATA_UNAVAILABLE",
    "MARKET_RULE_UNAVAILABLE": "DASHBOARD_RULE_METADATA_UNAVAILABLE",
    "STRATEGY_UNKNOWN": "DASHBOARD_STRATEGY_MISSING",
}


def _action_error(error: Exception, *, operation: str) -> str:
    raw = error.args[0] if len(error.args) == 1 else None
    candidate = raw.partition(":")[0] if type(raw) is str else None
    if candidate is not None:
        mapped = _ACTION_ERROR_CODES.get(candidate)
        if mapped is not None:
            return mapped
    return (
        "DASHBOARD_DECISION_EXPORT_FAILED"
        if operation == "export"
        else "DASHBOARD_DECISION_GENERATION_FAILED"
    )


class DashboardPageModel:
    def __init__(
        self,
        gateway: DashboardGateway,
        *,
        clock: Callable[[], datetime],
        actions: DashboardActionGateway | None = None,
    ) -> None:
        if not callable(clock):
            raise TypeError("dashboard clock must be callable")
        if actions is not None and any(
            not callable(getattr(actions, name, None))
            for name in (
                "list_strategies",
                "list_manifests",
                "generate_close_decision",
                "export_decision",
            )
        ):
            raise TypeError("dashboard actions must implement the action boundary")
        self._gateway = gateway
        self._clock = clock
        self._actions = actions
        self._action_state = DashboardActionState("idle")
        self._action_lock = RLock()

    @property
    def action_available(self) -> bool:
        return self._actions is not None

    def action_state(self) -> DashboardActionState:
        with self._action_lock:
            return self._action_state

    def strategy_choices(self) -> tuple[DashboardStrategyChoice, ...]:
        if self._actions is None:
            return ()
        choices = _boundary_call(
            "DASHBOARD_ACTION_OPTIONS_UNAVAILABLE",
            self._actions.list_strategies,
        )
        if type(choices) is not tuple or any(
            type(item) is not DashboardStrategyChoice for item in choices
        ):
            raise DashboardPageError("DASHBOARD_ACTION_OPTIONS_UNAVAILABLE")
        ids = tuple(item.strategy_instance_id for item in choices)
        if ids != tuple(sorted(set(ids))):
            raise DashboardPageError("DASHBOARD_ACTION_OPTIONS_UNAVAILABLE")
        return choices

    def manifest_choices(
        self,
        strategy_instance_id: str,
    ) -> tuple[DashboardManifestChoice, ...]:
        try:
            checked = safe_identifier(
                strategy_instance_id,
                label="strategy instance id",
            )
        except (TypeError, ValueError):
            raise DashboardPageError("DASHBOARD_STRATEGY_MISSING") from None
        actions = self._actions
        if actions is None:
            return ()
        choices = _boundary_call(
            "DASHBOARD_ACTION_OPTIONS_UNAVAILABLE",
            lambda: actions.list_manifests(checked),
        )
        if type(choices) is not tuple or any(
            type(item) is not DashboardManifestChoice for item in choices
        ):
            raise DashboardPageError("DASHBOARD_ACTION_OPTIONS_UNAVAILABLE")
        ids = tuple(item.manifest_id for item in choices)
        if ids != tuple(sorted(set(ids))):
            raise DashboardPageError("DASHBOARD_ACTION_OPTIONS_UNAVAILABLE")
        return choices

    def generate_close_decision(
        self,
        strategy_instance_id: str,
        manifest_id: str,
    ) -> DashboardActionState:
        try:
            strategy = safe_identifier(
                strategy_instance_id,
                label="strategy instance id",
            )
        except (TypeError, ValueError):
            return self._failed_action("generate", "DASHBOARD_STRATEGY_MISSING")
        try:
            manifest = safe_identifier(manifest_id, label="manifest id")
        except (TypeError, ValueError):
            return self._failed_action(
                "generate",
                "DASHBOARD_MANIFEST_INELIGIBLE",
                strategy_instance_id=strategy,
            )
        if self._actions is None:
            return self._failed_action(
                "generate",
                "DASHBOARD_DECISION_ACTIONS_UNAVAILABLE",
                strategy_instance_id=strategy,
                manifest_id=manifest,
            )
        with self._action_lock:
            self._action_state = DashboardActionState(
                "running",
                "generate",
                strategy,
                manifest,
            )
        try:
            receipt = self._actions.generate_close_decision(strategy, manifest)
            if (
                type(receipt) is not DashboardActionReceipt
                or receipt.strategy_instance_id != strategy
                or receipt.manifest_id != manifest
            ):
                raise ValueError("dashboard decision receipt mismatch")
        except Exception as error:
            return self._failed_action(
                "generate",
                _action_error(error, operation="generate"),
                strategy_instance_id=strategy,
                manifest_id=manifest,
            )
        result = DashboardActionState(
            "succeeded",
            "generate",
            strategy,
            manifest,
            receipt.decision_id,
            (),
            receipt.generated_at,
        )
        with self._action_lock:
            self._action_state = result
        return result

    def export_decision(self) -> DashboardActionState:
        if self._actions is None:
            return self._failed_action(
                "export",
                "DASHBOARD_DECISION_ACTIONS_UNAVAILABLE",
            )
        with self._action_lock:
            previous = self._action_state
            selected = previous.decision_id
            self._action_state = DashboardActionState(
                "running",
                "export",
                previous.strategy_instance_id,
                previous.manifest_id,
                selected,
            )
        try:
            receipt = self._actions.export_decision(selected)
            if type(receipt) is not DashboardExportReceipt or (
                selected is not None and receipt.decision_id != selected
            ):
                raise ValueError("dashboard export receipt mismatch")
        except Exception as error:
            return self._failed_action(
                "export",
                _action_error(error, operation="export"),
                strategy_instance_id=previous.strategy_instance_id,
                manifest_id=previous.manifest_id,
                decision_id=selected,
            )
        result = DashboardActionState(
            "succeeded",
            "export",
            previous.strategy_instance_id,
            previous.manifest_id,
            receipt.decision_id,
            (receipt.csv_filename, receipt.json_filename),
            receipt.exported_at,
        )
        with self._action_lock:
            self._action_state = result
        return result

    def _failed_action(
        self,
        operation: str,
        code: str,
        *,
        strategy_instance_id: str | None = None,
        manifest_id: str | None = None,
        decision_id: str | None = None,
    ) -> DashboardActionState:
        result = DashboardActionState(
            "failed",
            operation,
            strategy_instance_id,
            manifest_id,
            decision_id,
            (),
            None,
            code,
        )
        with self._action_lock:
            self._action_state = result
        return result

    def state(self) -> DashboardPageState:
        snapshot = _boundary_call("DASHBOARD_STATE_UNAVAILABLE", lambda: self._gateway.state())
        if type(snapshot) is not DashboardSnapshot:
            raise DashboardPageError("DASHBOARD_STATE_UNAVAILABLE")
        now = _boundary_call("DASHBOARD_CLOCK_UNAVAILABLE", self._clock)
        if type(now) is not datetime or now.tzinfo is None or now.utcoffset() is None:
            raise DashboardPageError("DASHBOARD_CLOCK_UNAVAILABLE")

        intraday = snapshot.intraday
        temporary = () if intraday is None else intraday.signals
        decision = snapshot.confirmed_decision
        failure = snapshot.decision_failure
        if failure is not None:
            status = FAILED_STATUS
        elif decision is None:
            status = NO_DECISION_STATUS
        elif decision.valid_until < now.astimezone(SHANGHAI).date():
            status = EXPIRED_STATUS
        else:
            status = CONFIRMED_STATUS
        actionable = status == CONFIRMED_STATUS
        recommendations: tuple[DashboardRecommendationView, ...] = ()
        if decision is not None:
            recommendations = tuple(
                self._recommendation(
                    item,
                    snapshot.actual_weights.get(item.instrument, Decimal("0")),
                    snapshot.target_weights[item.instrument],
                    actionable,
                )
                for item in decision.recommendations
            )
        return DashboardPageState(
            data_health=snapshot.data_health,
            account=snapshot.account,
            current_drawdown=snapshot.current_drawdown,
            risk_status=snapshot.risk_status,
            actual_weights=snapshot.actual_weights,
            target_weights=snapshot.target_weights,
            temporary_status="临时" if temporary else None,
            temporary_signals=temporary,
            intraday_source_at=None if intraday is None else intraday.source_at,
            intraday_stale=(
                None if intraday is None or intraday.source_at is None else intraday.stale
            ),
            decision_status=status,
            decision_at=None if decision is None else decision.decision_at,
            decision_source_at=None if decision is None else decision.market_data_source_at,
            valid_until=None if decision is None else decision.valid_until,
            confirmed_recommendations=recommendations,
            decision_actionable=actionable,
            failure_code=None if failure is None else failure.code,
            error_id=None if failure is None else failure.error_id,
        )

    @staticmethod
    def _recommendation(
        value: RebalanceRecommendation,
        actual_weight: Decimal,
        target_weight: Decimal,
        actionable: bool,
    ) -> DashboardRecommendationView:
        if type(value) is not RebalanceRecommendation:
            raise TypeError("decision recommendations must be exact values")
        return DashboardRecommendationView(
            recommendation=value,
            actual_weight=actual_weight,
            target_weight=target_weight,
            actionable=actionable,
        )


def _action_status_text(state: DashboardActionState) -> str:
    labels = {
        "idle": "空闲",
        "running": "运行中",
        "succeeded": "成功",
        "failed": "失败",
    }
    parts = [f"操作状态：{labels[state.status]}"]
    if state.operation == "generate":
        parts.append("生成收盘决策")
    elif state.operation == "export":
        parts.append("导出决策")
    if state.strategy_instance_id is not None:
        parts.append(f"策略 {state.strategy_instance_id}")
    if state.manifest_id is not None:
        parts.append(f"清单 {state.manifest_id}")
    if state.decision_id is not None:
        parts.append(f"决策 {state.decision_id}")
    parts.extend(state.report_filenames)
    if state.error_code is not None:
        parts.append(state.error_code)
    return " / ".join(parts)


def render_dashboard_page(model: DashboardPageModel | None) -> None:
    if model is None:
        ui.label("决策服务未配置；当前不会显示示例行情、账户或调仓建议。")
        return

    strategy_select = None
    manifest_select = None
    if model.action_available:
        try:
            strategies = model.strategy_choices()
            default_strategy = strategies[0].strategy_instance_id if strategies else None
            manifests = (
                model.manifest_choices(default_strategy)
                if default_strategy is not None
                else ()
            )
        except Exception:
            strategies = ()
            manifests = ()
            default_strategy = None
            ui.label(
                "收盘决策操作：不可用 / DASHBOARD_ACTION_OPTIONS_UNAVAILABLE"
            ).classes("text-red-700")

        def update_manifests(event: object) -> None:
            assert manifest_select is not None
            selected = getattr(event, "value", None)
            try:
                choices = model.manifest_choices(selected)
            except Exception:
                choices = ()
            options = {
                item.manifest_id: (
                    f"{item.manifest_id} / {', '.join(str(symbol) for symbol in item.instruments)}"
                )
                for item in choices
            }
            manifest_select.set_options(
                options,
                value=choices[0].manifest_id if choices else None,
            )
            generate_button.set_enabled(bool(choices))

        strategy_select = ui.select(
            {
                item.strategy_instance_id: item.label
                for item in strategies
            },
            label="策略实例",
            value=default_strategy,
            on_change=update_manifests,
        )
        manifest_select = ui.select(
            {
                item.manifest_id: (
                    f"{item.manifest_id} / {', '.join(str(symbol) for symbol in item.instruments)}"
                )
                for item in manifests
            },
            label="收盘数据清单",
            value=manifests[0].manifest_id if manifests else None,
        )

    @ui.refreshable
    def refreshed_state() -> None:
        try:
            state = model.state()
        except Exception:
            ui.label("今日决策读取失败，请查看本地脱敏日志。").classes("text-red-700")
            return
        if model.action_available:
            ui.label(_action_status_text(model.action_state()))
        _render_dashboard_state(state)

    if model.action_available:
        assert strategy_select is not None and manifest_select is not None

        def generate() -> None:
            model.generate_close_decision(
                strategy_select.value,
                manifest_select.value,
            )
            refreshed_state.refresh()

        def export() -> None:
            model.export_decision()
            refreshed_state.refresh()

        with ui.row():
            generate_button = ui.button("生成收盘决策", on_click=generate)
            ui.button("导出决策", on_click=export)
        generate_button.set_enabled(
            strategy_select.value is not None and manifest_select.value is not None
        )
    refreshed_state()


def _render_dashboard_state(state: DashboardPageState) -> None:
    if state.data_health is None:
        ui.label("数据质量：暂无")
    else:
        ui.label(
            f"数据质量：{'通过' if state.data_health.accepted else '未通过'} / {state.data_health.mode}"
        )
        ui.label(
            "数据质量问题："
            + ("、".join(state.data_health.issue_codes) if state.data_health.issue_codes else "无")
        )
    ui.label(f"风险状态：{state.risk_status or '暂无'}")
    ui.label(f"当前回撤：{'暂无' if state.current_drawdown is None else state.current_drawdown}")
    if state.account is None:
        ui.label("账户：暂无")
    else:
        account = state.account.snapshot
        cash_ratio = None if account.equity == 0 else account.cash / account.equity
        ui.label(f"账户 ID：{state.account.account_id}")
        ui.label(f"账户日期：{account.as_of.isoformat()}")
        ui.label(f"账户权益：{account.equity}")
        ui.label(f"账户现金：{account.cash}")
        ui.label(f"现金占比：{'不可用' if cash_ratio is None else cash_ratio}")
    ui.label("实际权重")
    if not state.actual_weights:
        ui.label("实际权重：无持仓")
    for instrument, weight in state.actual_weights.items():
        ui.label(f"实际权重 {instrument}：{weight}")
    ui.label("目标权重")
    if not state.target_weights:
        ui.label("目标权重：无")
    for instrument, weight in state.target_weights.items():
        ui.label(f"目标权重 {instrument}：{weight}")
    if state.intraday_source_at is None:
        ui.label("日内临时状态：暂无")
    else:
        ui.label(
            f"日内临时状态：{'过期' if state.intraday_stale else '最新'} / "
            f"来源 {state.intraday_source_at.isoformat()}"
        )
    for signal in state.temporary_signals:
        ui.label(
            f"临时信号 / {signal.strategy_id} / {signal.instrument} / "
            f"目标 {signal.target_weight} / 分数 {signal.score} / "
            f"置信度 {signal.confidence} / {signal.reason_code}"
        )
    actionability = "可执行" if state.decision_actionable else "不可执行"
    ui.label(f"决策状态：{state.decision_status} / {actionability}")
    if state.decision_at is not None:
        ui.label(f"决策时间：{state.decision_at.isoformat()}")
        assert state.decision_source_at is not None and state.valid_until is not None
        ui.label(f"决策行情来源：{state.decision_source_at.isoformat()}")
        ui.label(f"有效期：{state.valid_until.isoformat()}")
    if state.failure_code is not None:
        ui.label(f"失败：{state.failure_code} / 错误 ID {state.error_id}").classes("text-red-700")
    for item in state.confirmed_recommendations:
        with ui.card().classes("w-full border border-slate-200 shadow-none"):
            ui.label(str(item.instrument)).classes("font-semibold")
            ui.label("可执行建议" if item.actionable else "历史建议 / 不可执行")
            ui.label("策略决策")
            if not item.strategy_decisions:
                ui.label("策略决策：无")
            for trace in item.strategy_decisions:
                ui.label(
                    f"策略决策 / {trace.strategy_id} / {trace.status.value} / {trace.reason_code}"
                )
            ui.label("原始意图")
            if not item.raw_intents:
                ui.label("原始意图：无")
            for intent in item.raw_intents:
                ui.label(
                    f"原始意图 / {intent.strategy_id} / {intent.instrument} / "
                    f"目标 {intent.target_weight} / 分数 {intent.score} / "
                    f"置信度 {intent.confidence} / {intent.reason_code} / "
                    f"有效期 {intent.valid_until.isoformat()}"
                )
            ui.label(
                f"分配阶段：分配权重 {item.allocated_weight} / 风控前权重 {item.pre_risk_weight}"
            )
            if not item.allocation_trace:
                ui.label("分配轨迹：无调整")
            for allocation_adjustment in item.allocation_trace:
                ui.label(
                    f"分配轨迹 / {allocation_adjustment.stage.value} / {allocation_adjustment.group} / "
                    f"{allocation_adjustment.before_units}->{allocation_adjustment.after_units} / "
                    f"上限 {allocation_adjustment.limit_units} / {allocation_adjustment.reason_code}"
                )
            ui.label(f"风控阶段：最终权重 {item.risk_adjusted_weight}")
            if not item.risk_adjustments:
                ui.label("风控轨迹：无调整")
            for risk_adjustment in item.risk_adjustments:
                ui.label(
                    f"风控轨迹 / {risk_adjustment.stage.value} / {risk_adjustment.severity.value} / "
                    f"{risk_adjustment.before_weight}->{risk_adjustment.after_weight} / "
                    f"参照 {risk_adjustment.reference_weight} / {risk_adjustment.code} / {risk_adjustment.message}"
                )
            ui.label(
                f"执行阶段：当前数量 {item.current_quantity} / 目标数量 {item.final_quantity} / "
                f"方向 {item.side.value}"
            )
            ui.label(
                f"参考价格 {item.reference_price} / 执行价格 "
                f"{'无' if item.estimated_execution_price is None else item.estimated_execution_price} / "
                f"成交金额 {item.gross_amount}"
            )
            ui.label(
                f"费用：佣金 {item.costs.commission} / 印花税 {item.costs.stamp_duty} / "
                f"过户费 {item.costs.transfer_fee} / 合计 {item.costs.total} / 档案 {item.profile_id}"
            )
            ui.label(
                f"账户快照：行 {item.account_snapshot_row_id} / 哈希 {item.account_snapshot_hash}"
            )
            ui.label(f"行情来源：{item.market_data_source_at.isoformat()}")
            ui.label(f"决策时间：{item.decision_at.isoformat()}")
            ui.label(f"有效期：{item.valid_until.isoformat()}")
            ui.label("原因代码：" + "、".join(item.reason_codes))
