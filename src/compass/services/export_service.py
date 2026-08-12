from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Protocol
from uuid import uuid4

from compass.backtest.orders import Fill, LedgerPosition, LedgerSnapshot, Order
from compass.backtest.snapshot import ManifestReference, RunSnapshot, StrategySnapshot
from compass.security import is_credential_key
from compass.services.decision_service import (
    DecisionResult,
    RebalanceRecommendation,
    StrategyDecisionTrace,
)
from compass.services.safe_display import safe_display_text, safe_identifier, stable_code
from compass.ui.pages.backtests import BacktestReport


ADVISORY_DISCLAIMER = "仅供研究与信息参考，不构成投资建议，不连接券商，也不会提交订单。"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CSV_FORMULA_PREFIXES = frozenset(("=", "+", "-", "@"))
_LEGACY_DECISION_SNAPSHOT_PREFIX = "decision:"
_CURRENT_DECISION_SNAPSHOT_PREFIX = "decision-v2:"


class ExportServiceError(RuntimeError):
    """A stable, context-free failure at the public export boundary."""

    def __init__(self, code: str) -> None:
        self.code = stable_code(code, label="export error code")
        super().__init__(self.code)


class ExportReadGateway(Protocol):
    """The only data source an exporter may consult."""

    def load_backtest(self, run_id: str) -> BacktestReport: ...

    def load_decision_export(self, decision_id: str) -> DecisionExportRecord: ...


def decision_snapshot_run_id(decision_id: str) -> str:
    """Return the fixed-width, collision-isolated run ID for a new decision snapshot."""

    checked = safe_identifier(decision_id, label="decision id")
    return f"{_CURRENT_DECISION_SNAPSHOT_PREFIX}{sha256(checked.encode('utf-8')).hexdigest()}"


def is_reserved_decision_snapshot_run_id(run_id: object) -> bool:
    return type(run_id) is str and run_id.startswith(
        (_LEGACY_DECISION_SNAPSHOT_PREFIX, _CURRENT_DECISION_SNAPSHOT_PREFIX)
    )


def is_decision_snapshot_run_id(run_id: object, decision_id: str) -> bool:
    checked = safe_identifier(decision_id, label="decision id")
    return type(run_id) is str and run_id in {
        f"{_LEGACY_DECISION_SNAPSHOT_PREFIX}{checked}",
        decision_snapshot_run_id(checked),
    }


def _decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("non-finite decimal")
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered in {"", "-0"} else rendered


def _public_value(value: object, *, label: str) -> object:
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is str:
        return safe_display_text(value, label=label, maximum=4096)
    if type(value) is float:
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("non-finite float")
        return value
    if type(value) is Decimal:
        assert isinstance(value, Decimal)
        return _decimal(value)
    if type(value) in (date, datetime):
        assert isinstance(value, (date, datetime))
        if type(value) is datetime and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("naive datetime")
        return value.isoformat()
    if isinstance(value, Enum):
        return _public_value(value.value, label=label)
    if isinstance(value, Mapping):
        checked: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("public mapping keys must be exact strings")
            if is_credential_key(key):
                raise ValueError("credential field is not exportable")
            checked_key = safe_display_text(
                key,
                label="public mapping key",
                maximum=4096,
            )
            checked[checked_key] = _public_value(item, label=f"{label}.{checked_key}")
        return dict(sorted(checked.items()))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _public_value(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"unsupported public export value: {type(value).__name__}")


def _freeze_public_mapping(value: object, *, label: str) -> Mapping[str, object]:
    checked = _public_value(value, label=label)
    if not isinstance(checked, Mapping):
        raise TypeError(f"{label} must be a mapping")

    def freeze(item: object) -> object:
        if isinstance(item, Mapping):
            return MappingProxyType({key: freeze(child) for key, child in item.items()})
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        return item

    frozen = freeze(checked)
    assert isinstance(frozen, Mapping)
    return frozen


@dataclass(frozen=True, slots=True)
class DecisionManifestProvenance:
    manifest_id: str
    provider: str
    content_hash: str
    relative_data_path: str

    def __post_init__(self) -> None:
        safe_identifier(self.manifest_id, label="decision manifest id")
        safe_display_text(self.provider, label="decision manifest provider")
        if (
            type(self.content_hash) is not str
            or _SHA256.fullmatch(self.content_hash) is None
        ):
            raise ValueError("decision manifest hash must be a lowercase SHA-256 digest")
        if self.relative_data_path != f"objects/{self.content_hash}.parquet":
            raise ValueError("decision manifest path must match its content hash")

    def verify_integrity(self) -> None:
        self.__post_init__()


@dataclass(frozen=True, slots=True)
class DecisionStrategyProvenance:
    strategy_instance_id: str
    strategy_type: str
    strategy_version: str
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        safe_identifier(self.strategy_instance_id, label="decision strategy instance id")
        safe_identifier(self.strategy_type, label="decision strategy type")
        safe_identifier(self.strategy_version, label="decision strategy version")
        object.__setattr__(
            self,
            "parameters",
            _freeze_public_mapping(
                self.parameters,
                label=f"decision strategy {self.strategy_instance_id} parameters",
            ),
        )

    def verify_integrity(self) -> None:
        self.__post_init__()


@dataclass(frozen=True, slots=True)
class DecisionExportRecord:
    decision_id: str
    result: DecisionResult
    market_manifests: tuple[DecisionManifestProvenance, ...]
    strategies: tuple[DecisionStrategyProvenance, ...]
    snapshot: RunSnapshot

    def __post_init__(self) -> None:
        safe_identifier(self.decision_id, label="decision export id")
        if type(self.result) is not DecisionResult:
            raise TypeError("decision export result must be an exact DecisionResult")
        self.result.__post_init__()
        if type(self.snapshot) is not RunSnapshot:
            raise TypeError("decision export snapshot must be an exact RunSnapshot")
        self.snapshot.verify_integrity()
        if not is_decision_snapshot_run_id(self.snapshot.run_id, self.decision_id):
            raise ValueError("decision export snapshot run id is invalid")
        if type(self.market_manifests) is not tuple or not self.market_manifests or any(
            type(item) is not DecisionManifestProvenance
            for item in self.market_manifests
        ):
            raise TypeError("decision export manifests must be a non-empty exact tuple")
        for manifest in self.market_manifests:
            manifest.verify_integrity()
        manifest_ids = tuple(item.manifest_id for item in self.market_manifests)
        if manifest_ids != tuple(sorted(set(manifest_ids))):
            raise ValueError("decision export manifests must be unique and sorted")
        if type(self.strategies) is not tuple or not self.strategies or any(
            type(item) is not DecisionStrategyProvenance for item in self.strategies
        ):
            raise TypeError("decision export strategies must be a non-empty exact tuple")
        for strategy in self.strategies:
            strategy.verify_integrity()
        strategy_ids = tuple(item.strategy_instance_id for item in self.strategies)
        decision_strategy_ids = tuple(
            item.strategy_id for item in self.result.strategy_decisions
        )
        if (
            strategy_ids != tuple(sorted(set(strategy_ids)))
            or strategy_ids != decision_strategy_ids
        ):
            raise ValueError("decision strategy provenance must exactly match the result")
        expected_manifests = tuple(
            ManifestReference(item.manifest_id, item.content_hash)
            for item in self.market_manifests
        )
        expected_strategies = tuple(
            StrategySnapshot(
                sleeve_id=item.strategy_instance_id,
                strategy_type=item.strategy_type,
                strategy_version=item.strategy_version,
                parameters=item.parameters,
            )
            for item in self.strategies
        )
        if (
            self.snapshot.market_manifests != expected_manifests
            or self.snapshot.strategies != expected_strategies
        ):
            raise ValueError("decision export snapshot provenance does not match record")

    def verify_integrity(self) -> None:
        self.__post_init__()


def _order(order: Order) -> Mapping[str, object]:
    return {
        "order_id": order.order_id,
        "instrument": str(order.instrument),
        "side": order.side.value,
        "quantity": order.quantity,
        "created_on": order.created_on,
        "scheduled_for": order.scheduled_for,
        "sleeve_weights": order.sleeve_weights,
        "risk_codes": order.risk_codes,
        "status": order.status.value,
        "filled_quantity": order.filled_quantity,
        "cancellation_reason": (
            None if order.cancellation_reason is None else order.cancellation_reason.value
        ),
    }


def _fill(fill: Fill) -> Mapping[str, object]:
    return {
        "fill_id": fill.fill_id,
        "order_id": fill.order_id,
        "instrument": str(fill.instrument),
        "side": fill.side.value,
        "quantity": fill.quantity,
        "trading_day": fill.trading_day,
        "price": fill.price,
        "gross_amount": fill.gross_amount,
        "commission": fill.commission,
        "stamp_duty": fill.stamp_duty,
        "transfer_fee": fill.transfer_fee,
        "total_fee": fill.total_fee,
        "profile_id": fill.profile_id,
    }


def _position(position: LedgerPosition) -> Mapping[str, object]:
    return {
        "instrument": str(position.instrument),
        "quantity": position.quantity,
        "available_quantity": position.available_quantity,
        "unsettled_quantity": position.unsettled_quantity,
        "average_cost": position.average_cost,
        "mark_price": position.mark_price,
        "market_value": position.market_value,
    }


def _ledger(snapshot: LedgerSnapshot) -> Mapping[str, object]:
    return {
        "trading_day": snapshot.trading_day,
        "cash": snapshot.cash,
        "withdrawable_cash": snapshot.withdrawable_cash,
        "equity": snapshot.equity,
        "positions": tuple(_position(item) for item in snapshot.positions),
    }


def _snapshot(snapshot: RunSnapshot) -> Mapping[str, object]:
    return {
        "snapshot_id": snapshot.snapshot_id,
        "run_id": snapshot.run_id,
        "schema_version": snapshot.schema_version,
        "market_manifests": tuple(
            {"manifest_id": item.manifest_id, "content_hash": item.content_hash}
            for item in snapshot.market_manifests
        ),
        "data_quality": snapshot.data_quality,
        "strategies": tuple(
            {
                "sleeve_id": item.sleeve_id,
                "strategy_type": item.strategy_type,
                "strategy_version": item.strategy_version,
                "parameters": item.parameters,
            }
            for item in snapshot.strategies
        ),
        "instrument_pool": snapshot.instrument_pool,
        "survivorship_bias": snapshot.survivorship_bias,
        "allocator_configuration": snapshot.allocator_configuration,
        "risk_configuration": snapshot.risk_configuration,
        "market_rule_configuration": snapshot.market_rule_configuration,
        "fee_profile_configuration": snapshot.fee_profile_configuration,
        "app_git_commit": snapshot.app_git_commit,
        "random_seed": snapshot.random_seed,
    }


def _metrics(report: BacktestReport) -> Mapping[str, object]:
    metrics = report.metrics
    return {
        "total_return": metrics.total_return,
        "annualized_return": metrics.annualized_return,
        "annualized_volatility": metrics.annualized_volatility,
        "sharpe_ratio": metrics.sharpe_ratio,
        "maximum_drawdown": metrics.maximum_drawdown,
        "calmar_ratio": metrics.calmar_ratio,
        "win_rate": metrics.win_rate,
        "total_turnover": metrics.total_turnover,
        "total_costs": metrics.total_costs,
        "monthly_returns": metrics.monthly_returns,
        "benchmark_total_return": metrics.benchmark_total_return,
        "benchmark_annualized_return": metrics.benchmark_annualized_return,
        "excess_total_return": metrics.excess_total_return,
        "excess_annualized_return": metrics.excess_annualized_return,
    }


def _backtest_payload(report: BacktestReport) -> Mapping[str, object]:
    report.verify_integrity()
    return {
        "schema_version": 1,
        "export_type": "backtest",
        "run_id": report.run_id,
        "configuration_id": report.configuration_id,
        "strategy_instance_ids": report.strategy_instance_ids,
        "reproduction": _snapshot(report.snapshot),
        "results": {
            "metrics": _metrics(report),
            "equity_curve": tuple(
                {"day": item.day, "value": item.value} for item in report.equity_curve
            ),
            "drawdown_curve": tuple(
                {"day": item.day, "value": item.value} for item in report.drawdown_curve
            ),
            "benchmark_curve": tuple(
                {"day": item.day, "value": item.value} for item in report.benchmark_curve
            ),
            "monthly_returns": tuple(
                {"month": item.month, "value": item.value} for item in report.monthly_returns
            ),
            "orders": tuple(_order(item) for item in report.result.orders),
            "fills": tuple(_fill(item) for item in report.result.fills),
            "ledger": tuple(_ledger(item) for item in report.result.ledger),
            "risk_traces": tuple(
                {
                    "decision_date": item.decision_date,
                    "instrument": str(item.instrument),
                    "requested_weight": item.result.requested_weight,
                    "final_weight": item.result.final_weight,
                    "blocked": item.result.blocked,
                    "adjustments": tuple(
                        {
                            "stage": adjustment.stage.value,
                            "severity": adjustment.severity.value,
                            "code": adjustment.code,
                            "message": adjustment.message,
                            "before_weight": adjustment.before_weight,
                            "after_weight": adjustment.after_weight,
                            "reference_weight": adjustment.reference_weight,
                        }
                        for adjustment in item.adjustments
                    ),
                }
                for item in report.result.risk_traces
            ),
            "used_profile_ids": report.result.used_profile_ids,
            "warnings": report.result.warnings,
            "sleeve_attribution": tuple(
                {"day": item.day, "sleeve": item.sleeve, "value": item.value}
                for item in report.attribution
            ),
            "attribution_residuals": tuple(
                {"day": item.day, "value": item.value}
                for item in report.attribution_residuals
            ),
            "combined_trade_residual": report.combined_trade_residual,
        },
        "disclaimer": ADVISORY_DISCLAIMER,
    }


def _strategy_trace(trace: StrategyDecisionTrace) -> Mapping[str, object]:
    return {
        "strategy_id": trace.strategy_id,
        "status": trace.status.value,
        "reason_code": trace.reason_code,
        "details": trace.details,
    }


def _recommendation(item: RebalanceRecommendation) -> Mapping[str, object]:
    return {
        "instrument": str(item.instrument),
        "raw_stage": {
            "strategy_decisions": tuple(_strategy_trace(trace) for trace in item.strategy_decisions),
            "intents": tuple(
                {
                    "strategy_id": intent.strategy_id,
                    "instrument": str(intent.instrument),
                    "target_weight": intent.target_weight,
                    "score": intent.score,
                    "confidence": intent.confidence,
                    "reason_code": intent.reason_code,
                    "valid_until": intent.valid_until,
                }
                for intent in item.raw_intents
            ),
        },
        "allocation_stage": {
            "allocated_weight": item.allocated_weight,
            "pre_risk_weight": item.pre_risk_weight,
            "adjustments": tuple(
                {
                    "stage": adjustment.stage.value,
                    "group": adjustment.group,
                    "input_keys": adjustment.input_keys,
                    "before_units": adjustment.before_units,
                    "after_units": adjustment.after_units,
                    "limit_units": adjustment.limit_units,
                    "reason_code": adjustment.reason_code,
                }
                for adjustment in item.allocation_trace
            ),
        },
        "risk_stage": {
            "current_weight": item.current_weight,
            "final_weight": item.final_weight,
            "blocked": item.blocked,
            "adjustments": tuple(
                {
                    "stage": adjustment.stage.value,
                    "severity": adjustment.severity.value,
                    "code": adjustment.code,
                    "message": adjustment.message,
                    "before_weight": adjustment.before_weight,
                    "after_weight": adjustment.after_weight,
                    "reference_weight": adjustment.reference_weight,
                }
                for adjustment in item.risk_adjustments
            ),
        },
        "final_stage": {
            "current_quantity": item.current_quantity,
            "target_quantity": item.target_quantity,
            "quantity_delta": item.quantity_delta,
            "side": item.side.value,
            "reference_price": item.reference_price,
            "estimated_execution_price": item.estimated_execution_price,
            "gross_amount": item.gross_amount,
            "costs": {
                "commission": item.costs.commission,
                "stamp_duty": item.costs.stamp_duty,
                "transfer_fee": item.costs.transfer_fee,
                "total": item.costs.total,
            },
            "profile_id": item.profile_id,
            "reason_codes": item.reason_codes,
        },
    }


def _decision_payload(record: DecisionExportRecord) -> Mapping[str, object]:
    result = record.result
    return {
        "schema_version": 1,
        "export_type": "decision",
        "decision_id": record.decision_id,
        "account_id": result.account_id,
        "provenance": {
            "account_snapshot_row_id": result.account_snapshot_row_id,
            "account_snapshot_hash": result.account_snapshot_hash,
            "decision_equity": result.decision_equity,
            "decision_at": result.decision_at,
            "decision_date": result.decision_date,
            "valid_until": result.valid_until,
            "market_data_source_at": result.market_data_source_at,
            "reproduction": _snapshot(record.snapshot),
            "market_manifests": tuple(
                {
                    "manifest_id": item.manifest_id,
                    "provider": item.provider,
                    "content_hash": item.content_hash,
                    "relative_data_path": item.relative_data_path,
                }
                for item in record.market_manifests
            ),
            "strategies": tuple(
                {
                    "strategy_instance_id": item.strategy_instance_id,
                    "strategy_type": item.strategy_type,
                    "strategy_version": item.strategy_version,
                    "parameters": item.parameters,
                }
                for item in record.strategies
            ),
        },
        "strategy_decisions": tuple(_strategy_trace(item) for item in result.strategy_decisions),
        "recommendations": tuple(_recommendation(item) for item in result.recommendations),
        "remaining_cash": result.remaining_cash,
        "warnings": result.warnings,
        "disclaimer": ADVISORY_DISCLAIMER,
    }


def _flatten(value: object, *, path: str = "") -> list[tuple[str, str]]:
    if isinstance(value, Mapping):
        rows: list[tuple[str, str]] = []
        for key, item in value.items():
            child = key if not path else f"{path}.{key}"
            rows.extend(_flatten(item, path=child))
        return rows
    if isinstance(value, list):
        rows = []
        for index, item in enumerate(value):
            rows.extend(_flatten(item, path=f"{path}[{index}]"))
        if not value:
            rows.append((path, "[]"))
        return rows
    return [(path, json.dumps(value, ensure_ascii=False, separators=(",", ":")))]


def _csv_bytes(payload: Mapping[str, object]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("section", "key", "value"))
    for path, value in _flatten(payload):
        section, separator, key = path.partition(".")
        key_cell = key if separator else section
        if section[:1] in _CSV_FORMULA_PREFIXES:
            section = f"'{section}"
        if key_cell[:1] in _CSV_FORMULA_PREFIXES:
            key_cell = f"'{key_cell}"
        if len(value) > 1 and value[0] == '"' and value[1] in _CSV_FORMULA_PREFIXES:
            value = f"'{value}"
        writer.writerow((section, key_cell, value))
    return stream.getvalue().encode("utf-8-sig")


@dataclass(frozen=True, slots=True)
class _Artifact:
    suffix: str
    payload: bytes


class _Replace(Protocol):
    def __call__(self, source: Path, destination: Path) -> None: ...


def _replace_file(source: Path, destination: Path) -> None:
    os.replace(source, destination)


class ExportService:
    """Write deterministic public artifacts from injected immutable read models."""

    def __init__(
        self,
        reports_root: Path,
        gateway: ExportReadGateway,
        *,
        replace: _Replace = _replace_file,
    ) -> None:
        if not isinstance(reports_root, Path):
            raise TypeError("reports_root must be a Path")
        if not callable(getattr(gateway, "load_backtest", None)) or not callable(
            getattr(gateway, "load_decision_export", None)
        ):
            raise TypeError("gateway must implement the export read boundary")
        self._root = reports_root.resolve()
        self._gateway = gateway
        self._replace = replace

    def export_backtest(self, run_id: str) -> tuple[Path, Path]:
        checked = self._identifier(run_id)
        try:
            report = self._gateway.load_backtest(checked)
        except Exception:
            raise ExportServiceError("EXPORT_BACKTEST_READ_FAILED") from None
        if type(report) is not BacktestReport:
            raise ExportServiceError("EXPORT_BACKTEST_READ_FAILED")
        try:
            payload = _public_value(_backtest_payload(report), label="backtest export")
        except Exception:
            raise ExportServiceError("EXPORT_PAYLOAD_UNSAFE") from None
        assert isinstance(payload, Mapping)
        return self._write_pair(f"backtest-{checked}", payload)

    def export_decision(self, decision_id: str) -> tuple[Path, Path]:
        checked = self._identifier(decision_id)
        try:
            record = self._gateway.load_decision_export(checked)
        except Exception:
            raise ExportServiceError("EXPORT_DECISION_READ_FAILED") from None
        if type(record) is not DecisionExportRecord:
            raise ExportServiceError("EXPORT_DECISION_READ_FAILED")
        try:
            record.verify_integrity()
            if record.decision_id != checked:
                raise ValueError("decision export identity changed")
            payload = _public_value(
                _decision_payload(record),
                label="decision export",
            )
        except Exception:
            raise ExportServiceError("EXPORT_PAYLOAD_UNSAFE") from None
        assert isinstance(payload, Mapping)
        return self._write_pair(f"decision-{checked}", payload)

    @staticmethod
    def _identifier(value: str) -> str:
        try:
            return safe_identifier(value, label="export id")
        except (TypeError, ValueError):
            raise ExportServiceError("EXPORT_ID_INVALID") from None

    def _write_pair(self, stem: str, payload: Mapping[str, object]) -> tuple[Path, Path]:
        artifacts = (
            _Artifact(".csv", _csv_bytes(payload)),
            _Artifact(
                ".json",
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                ).encode("utf-8"),
            ),
        )
        temporary: list[Path] = []
        destinations: list[Path] = []
        rollbacks: list[tuple[Path, Path]] = []
        installed: list[Path] = []
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            for artifact in artifacts:
                destination = (self._root / f"{stem}{artifact.suffix}").resolve()
                destination.relative_to(self._root)
                staged = self._root / f".{stem}.{uuid4().hex}{artifact.suffix}.tmp"
                staged.write_bytes(artifact.payload)
                temporary.append(staged)
                destinations.append(destination)
            for destination in destinations:
                if destination.exists():
                    rollback = self._root / f".{destination.name}.{uuid4().hex}.rollback"
                    self._replace(destination, rollback)
                    rollbacks.append((destination, rollback))
            for staged, destination in zip(temporary, destinations, strict=True):
                self._replace(staged, destination)
                installed.append(destination)
        except Exception:
            rollback_failed = False
            for destination in installed:
                try:
                    destination.unlink(missing_ok=True)
                except OSError:
                    pass
            for destination, rollback in reversed(rollbacks):
                try:
                    if rollback.exists():
                        self._replace(rollback, destination)
                except Exception:
                    rollback_failed = True
            for staged in temporary:
                try:
                    staged.unlink(missing_ok=True)
                except OSError:
                    pass
            if rollback_failed:
                raise ExportServiceError("EXPORT_ROLLBACK_FAILED") from None
            raise ExportServiceError("EXPORT_WRITE_FAILED") from None
        for _, rollback in rollbacks:
            try:
                rollback.unlink(missing_ok=True)
            except OSError:
                pass
        return destinations[0], destinations[1]
