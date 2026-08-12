from __future__ import annotations

import csv
import io
import json
import os
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from compass.backtest.snapshot import ManifestReference, RunSnapshot, StrategySnapshot
from compass.domain.market import InstrumentId
from compass.domain.trading import TargetIntent
from compass.services.decision_service import (
    DecisionResult,
    DecisionSide,
    RebalanceRecommendation,
    StrategyDecisionTrace,
    ZERO_COSTS,
)
from compass.services.export_service import (
    DecisionExportRecord,
    DecisionManifestProvenance,
    DecisionStrategyProvenance,
    ExportService,
    ExportServiceError,
)
from compass.strategies.base import StrategyDecisionStatus
from compass.ui.pages.backtests import BacktestReport
from tests.support.sleeve_results import multi_reallocation_result


SHANGHAI = ZoneInfo("Asia/Shanghai")
SENTINEL = "example-secret-sentinel"
DISCLAIMER = "仅供研究与信息参考，不构成投资建议，不连接券商，也不会提交订单。"


def _backtest_report(
    *,
    export_available: bool = False,
    allocator_key: str = "kind",
    allocator_value: str = "deterministic",
    random_seed: int = 17,
) -> BacktestReport:
    result = multi_reallocation_result()
    snapshot = RunSnapshot(
        run_id=result.run_id,
        schema_version=1,
        market_manifests=(
            ManifestReference("manifest-etf-bars-1", "a" * 64),
            ManifestReference("manifest-benchmark-1", "b" * 64),
        ),
        data_quality={
            "accepted": True,
            "mode": "strict",
            "issue_codes": ("MISSING_ADJUST_FACTOR",),
        },
        strategies=(
            StrategySnapshot(
                "sleeve-a",
                "etf_rotation",
                "1.2.0",
                {"lookbacks": (20, 60, 120), "top_n": 3},
            ),
            StrategySnapshot("sleeve-b", "dual_ma", "1.1.0", {"fast": 20, "slow": 60}),
        ),
        instrument_pool={
            "snapshot_id": "pool-20260720",
            "instruments": ("SSE.510300", "SZSE.159915"),
        },
        survivorship_bias={"present": True, "warning": "static-current-constituents"},
        allocator_configuration={
            allocator_key: allocator_value,
            "cash_floor": Decimal("0.10"),
        },
        risk_configuration={"profile_id": "balanced-v2", "single_etf_cap": Decimal("0.60")},
        market_rule_configuration={"profile_ids": ("SSE-2026",), "execution": "next_open"},
        fee_profile_configuration={"profile_id": "fees", "confirmed": True},
        app_git_commit="1" * 40,
        random_seed=random_seed,
    )
    return BacktestReport.from_result(
        run_id=result.run_id,
        configuration_id="configuration-1",
        strategy_instance_ids=("sleeve-a", "sleeve-b"),
        result=result,
        snapshot=snapshot,
        benchmark_curve=(),
        export_available=export_available,
    )


def _decision() -> DecisionResult:
    instrument = InstrumentId.parse("SSE.510300")
    decision_at = datetime(2026, 7, 24, 15, 5, tzinfo=SHANGHAI)
    source_at = datetime(2026, 7, 24, 15, 0, tzinfo=SHANGHAI)
    valid_until = date(2026, 7, 27)
    trace = StrategyDecisionTrace(
        strategy_id="rotation-main",
        status=StrategyDecisionStatus.GENERATED,
        reason_code="TARGETS_GENERATED",
        details={"candidate_count": 1},
    )
    intent = TargetIntent(
        strategy_id="rotation-main",
        instrument=instrument,
        target_weight=Decimal("0"),
        score=1.0,
        confidence=0.8,
        reason_code="MOMENTUM_TOP_N",
        valid_until=valid_until,
    )
    recommendation = RebalanceRecommendation(
        instrument=instrument,
        raw_intents=(intent,),
        strategy_decisions=(trace,),
        allocated_weight=Decimal("0"),
        allocation_trace=(),
        pre_risk_weight=Decimal("0"),
        current_weight=Decimal("0"),
        final_weight=Decimal("0"),
        risk_adjustments=(),
        blocked=False,
        current_quantity=0,
        target_quantity=0,
        quantity_delta=0,
        side=DecisionSide.NONE,
        reference_price=Decimal("4.2000"),
        estimated_execution_price=None,
        gross_amount=Decimal("0.00"),
        costs=ZERO_COSTS,
        profile_id="fees",
        market_data_source_at=source_at,
        account_snapshot_row_id=7,
        account_snapshot_hash="c" * 64,
        decision_equity=Decimal("100000.00"),
        decision_at=decision_at,
        decision_date=date(2026, 7, 24),
        valid_until=valid_until,
        reason_codes=("NO_TRADE",),
    )
    return DecisionResult(
        account_id="main",
        account_snapshot_row_id=7,
        account_snapshot_hash="c" * 64,
        decision_equity=Decimal("100000.00"),
        decision_at=decision_at,
        decision_date=date(2026, 7, 24),
        valid_until=valid_until,
        market_data_source_at=source_at,
        strategy_decisions=(trace,),
        recommendations=(recommendation,),
        remaining_cash=Decimal("100000.00"),
        warnings=("NO_TRADE",),
    )


def _decision_export_record() -> DecisionExportRecord:
    content_hash = "d" * 64
    parameters = {"lookbacks": (20, 60, 120), "top_n": 3}
    return DecisionExportRecord(
        decision_id="decision-20260724",
        result=_decision(),
        market_manifests=(
            DecisionManifestProvenance(
                manifest_id="manifest-decision-bars-1",
                provider="fixture-provider",
                content_hash=content_hash,
                relative_data_path=f"objects/{content_hash}.parquet",
            ),
        ),
        strategies=(
            DecisionStrategyProvenance(
                strategy_instance_id="rotation-main",
                strategy_type="etf_rotation",
                strategy_version="1.2.0",
                parameters=parameters,
            ),
        ),
        snapshot=RunSnapshot(
            run_id="decision:decision-20260724",
            schema_version=1,
            market_manifests=(
                ManifestReference("manifest-decision-bars-1", content_hash),
            ),
            data_quality={"accepted": True, "issue_codes": (), "mode": "strict"},
            strategies=(
                StrategySnapshot(
                    "rotation-main",
                    "etf_rotation",
                    "1.2.0",
                    parameters,
                ),
            ),
            instrument_pool={
                "instruments": ("SSE.510300",),
                "snapshot_id": "pool-20260724",
            },
            survivorship_bias={"mode": "static_pool", "warning": True},
            allocator_configuration={
                "cash_reserve": Decimal("0.10"),
                "kind": "deterministic_allocator",
            },
            risk_configuration={"active": False, "rules": (), "templates": ()},
            market_rule_configuration={
                "execution": "close_decision",
                "profiles": (),
            },
            fee_profile_configuration={
                "confirmed": True,
                "profile_id": "fees",
                "schedules": (),
            },
            app_git_commit="1" * 40,
            random_seed=0,
        ),
    )


class _Gateway:
    def __init__(self, report: BacktestReport, decision: DecisionExportRecord) -> None:
        self.report = report
        self.decision = decision
        self.unreachable_secret = SENTINEL

    def load_backtest(self, run_id: str) -> BacktestReport:
        if run_id != self.report.run_id:
            raise LookupError("missing")
        return self.report

    def load_decision_export(self, decision_id: str) -> DecisionExportRecord:
        if decision_id != "decision-20260724":
            raise LookupError("missing")
        return self.decision


def test_backtest_export_is_reproducible_bom_csv_and_utf8_json(tmp_path: Path) -> None:
    report = _backtest_report()
    service = ExportService(tmp_path / "reports", _Gateway(report, _decision_export_record()))

    paths = service.export_backtest(report.run_id)

    assert paths == (
        (tmp_path / "reports" / "backtest-run.csv").resolve(),
        (tmp_path / "reports" / "backtest-run.json").resolve(),
    )
    csv_bytes = paths[0].read_bytes()
    json_bytes = paths[1].read_bytes()
    assert csv_bytes.startswith(b"\xef\xbb\xbf")
    assert not json_bytes.startswith(b"\xef\xbb\xbf")
    assert csv_bytes.decode("utf-8-sig").splitlines()[0] == "section,key,value"
    payload = json.loads(json_bytes.decode("utf-8"))
    assert payload["schema_version"] == 1
    assert payload["export_type"] == "backtest"
    assert payload["run_id"] == report.run_id
    assert payload["configuration_id"] == report.configuration_id
    assert payload["reproduction"]["snapshot_id"] == report.snapshot.snapshot_id
    assert payload["reproduction"]["market_manifests"] == [
        {"manifest_id": "manifest-etf-bars-1", "content_hash": "a" * 64},
        {"manifest_id": "manifest-benchmark-1", "content_hash": "b" * 64},
    ]
    assert payload["reproduction"]["strategies"][0]["parameters"]["lookbacks"] == [20, 60, 120]
    assert payload["reproduction"]["instrument_pool"]["snapshot_id"] == "pool-20260720"
    assert payload["reproduction"]["allocator_configuration"]["kind"] == "deterministic"
    assert payload["reproduction"]["risk_configuration"]["profile_id"] == "balanced-v2"
    assert payload["reproduction"]["market_rule_configuration"]["execution"] == "next_open"
    assert payload["reproduction"]["fee_profile_configuration"]["confirmed"] is True
    assert payload["reproduction"]["data_quality"]["mode"] == "strict"
    assert payload["reproduction"]["app_git_commit"] == "1" * 40
    assert payload["reproduction"]["random_seed"] == 17
    assert payload["results"]["equity_curve"]
    assert payload["results"]["fills"]
    assert payload["results"]["sleeve_attribution"]
    assert payload["disclaimer"] == DISCLAIMER
    combined = csv_bytes.decode("utf-8-sig") + json_bytes.decode("utf-8")
    assert "manifest-etf-bars-1" in combined
    assert "COMPASS_TEST_SECRET" not in combined
    assert SENTINEL not in combined
    assert str(tmp_path) not in combined


def test_backtest_report_can_advertise_an_injected_export_gateway(tmp_path: Path) -> None:
    report = _backtest_report(export_available=True)
    service = ExportService(tmp_path / "reports", _Gateway(report, _decision_export_record()))

    paths = service.export_backtest(report.run_id)

    assert report.export_available is True
    assert all(path.is_file() for path in paths)


@pytest.mark.parametrize(
    "unsafe_key",
    (
        "C:/synthetic-private/task17",
        "https://synthetic@example.invalid/field",
    ),
)
def test_export_rejects_unsafe_public_mapping_keys(
    tmp_path: Path,
    unsafe_key: str,
) -> None:
    report = _backtest_report(allocator_key=unsafe_key)
    service = ExportService(tmp_path / "reports", _Gateway(report, _decision_export_record()))

    with pytest.raises(ExportServiceError) as raised:
        service.export_backtest(report.run_id)

    assert str(raised.value) == "EXPORT_PAYLOAD_UNSAFE"
    assert unsafe_key not in str(raised.value)


def test_decision_export_preserves_four_stages_and_has_no_submission_surface(
    tmp_path: Path,
) -> None:
    service = ExportService(
        tmp_path / "reports",
        _Gateway(_backtest_report(), _decision_export_record()),
    )

    paths = service.export_decision("decision-20260724")

    assert tuple(path.name for path in paths) == (
        "decision-decision-20260724.csv",
        "decision-decision-20260724.json",
    )
    payload = json.loads(paths[1].read_text("utf-8"))
    recommendation = payload["recommendations"][0]
    assert recommendation["raw_stage"]["intents"][0]["reason_code"] == "MOMENTUM_TOP_N"
    assert recommendation["allocation_stage"]["allocated_weight"] == "0"
    assert recommendation["risk_stage"]["final_weight"] == "0"
    assert recommendation["final_stage"]["side"] == "none"
    assert payload["provenance"]["account_snapshot_row_id"] == 7
    assert payload["provenance"]["account_snapshot_hash"] == "c" * 64
    assert payload["provenance"]["market_data_source_at"] == "2026-07-24T15:00:00+08:00"
    assert payload["provenance"]["market_manifests"] == [
        {
            "manifest_id": "manifest-decision-bars-1",
            "provider": "fixture-provider",
            "content_hash": "d" * 64,
            "relative_data_path": f"objects/{'d' * 64}.parquet",
        }
    ]
    assert payload["provenance"]["strategies"] == [
        {
            "strategy_instance_id": "rotation-main",
            "strategy_type": "etf_rotation",
            "strategy_version": "1.2.0",
            "parameters": {"lookbacks": [20, 60, 120], "top_n": 3},
        }
    ]
    assert payload["disclaimer"] == DISCLAIMER
    lowered = paths[1].read_text("utf-8").casefold()
    assert "submit_order" not in lowered
    assert "broker_endpoint" not in lowered
    assert SENTINEL not in lowered


@pytest.mark.parametrize("unsafe_part", ("provider", "parameter-key"))
def test_decision_export_rejects_forged_unsafe_provenance(
    tmp_path: Path,
    unsafe_part: str,
) -> None:
    record = _decision_export_record()
    manifest = record.market_manifests[0]
    strategy = record.strategies[0]
    if unsafe_part == "provider":
        forged_manifest = object.__new__(DecisionManifestProvenance)
        object.__setattr__(forged_manifest, "manifest_id", manifest.manifest_id)
        object.__setattr__(forged_manifest, "provider", "C:/synthetic-private/task17")
        object.__setattr__(forged_manifest, "content_hash", manifest.content_hash)
        object.__setattr__(
            forged_manifest,
            "relative_data_path",
            manifest.relative_data_path,
        )
        manifests = (forged_manifest,)
        strategies = record.strategies
    else:
        forged_strategy = object.__new__(DecisionStrategyProvenance)
        object.__setattr__(
            forged_strategy,
            "strategy_instance_id",
            strategy.strategy_instance_id,
        )
        object.__setattr__(forged_strategy, "strategy_type", strategy.strategy_type)
        object.__setattr__(forged_strategy, "strategy_version", strategy.strategy_version)
        object.__setattr__(
            forged_strategy,
            "parameters",
            {"https://synthetic@example.invalid/field": "public-value"},
        )
        manifests = record.market_manifests
        strategies = (forged_strategy,)
    forged = object.__new__(DecisionExportRecord)
    object.__setattr__(forged, "decision_id", record.decision_id)
    object.__setattr__(forged, "result", record.result)
    object.__setattr__(forged, "market_manifests", manifests)
    object.__setattr__(forged, "strategies", strategies)
    service = ExportService(
        tmp_path / "reports",
        _Gateway(_backtest_report(), forged),
    )

    with pytest.raises(ExportServiceError) as raised:
        service.export_decision(record.decision_id)

    assert str(raised.value) == "EXPORT_PAYLOAD_UNSAFE"
    assert str(tmp_path) not in str(raised.value)


def test_export_names_are_stable_confined_and_leave_no_temporary_files(tmp_path: Path) -> None:
    report = _backtest_report()
    reports = tmp_path / "reports"
    service = ExportService(reports, _Gateway(report, _decision_export_record()))

    first = service.export_backtest(report.run_id)
    first_bytes = tuple(path.read_bytes() for path in first)
    second = service.export_backtest(report.run_id)

    assert second == first
    assert tuple(path.read_bytes() for path in second) == first_bytes
    assert all(path.parent == reports.resolve() for path in second)
    assert sorted(path.name for path in reports.iterdir()) == [
        "backtest-run.csv",
        "backtest-run.json",
    ]


def test_failed_replacement_preserves_the_complete_existing_export_pair(
    tmp_path: Path,
) -> None:
    reports = tmp_path / "reports"
    first = ExportService(
        reports,
        _Gateway(_backtest_report(random_seed=17), _decision_export_record()),
    ).export_backtest("run")
    original = tuple(path.read_bytes() for path in first)
    replace_calls = 0

    def fail_second_install(source: Path, destination: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 4:
            raise OSError(f"{SENTINEL}:{tmp_path}")
        os.replace(source, destination)

    service = ExportService(
        reports,
        _Gateway(_backtest_report(random_seed=18), _decision_export_record()),
        replace=fail_second_install,
    )

    with pytest.raises(ExportServiceError) as raised:
        service.export_backtest("run")

    assert str(raised.value) == "EXPORT_WRITE_FAILED"
    assert tuple(path.read_bytes() for path in first) == original
    assert sorted(path.name for path in reports.iterdir()) == [
        "backtest-run.csv",
        "backtest-run.json",
    ]


def test_failed_rollback_preserves_unrecovered_old_artifact_with_distinct_safe_error(
    tmp_path: Path,
) -> None:
    reports = tmp_path / "reports"
    first = ExportService(
        reports,
        _Gateway(_backtest_report(random_seed=17), _decision_export_record()),
    ).export_backtest("run")
    original = tuple(path.read_bytes() for path in first)
    replace_calls = 0

    def fail_install_and_first_rollback(source: Path, destination: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls in {4, 5}:
            raise OSError(f"{SENTINEL}:{tmp_path}")
        os.replace(source, destination)

    service = ExportService(
        reports,
        _Gateway(_backtest_report(random_seed=18), _decision_export_record()),
        replace=fail_install_and_first_rollback,
    )

    with pytest.raises(ExportServiceError) as raised:
        service.export_backtest("run")

    json_rollbacks = tuple(reports.glob(".backtest-run.json.*.rollback"))
    assert str(raised.value) == "EXPORT_ROLLBACK_FAILED"
    assert SENTINEL not in str(raised.value)
    assert str(tmp_path) not in str(raised.value)
    assert first[0].read_bytes() == original[0]
    assert not first[1].exists()
    assert len(json_rollbacks) == 1
    assert json_rollbacks[0].read_bytes() == original[1]
    assert not tuple(reports.glob(".backtest-run.csv.*.rollback"))


@pytest.mark.parametrize("formula", ("=1+1", "+1+1", "-1+1", "@SUM(1,1)"))
def test_csv_formula_like_strings_are_explicitly_encoded_as_text(
    tmp_path: Path,
    formula: str,
) -> None:
    report = _backtest_report(
        allocator_key="formula_text",
        allocator_value=formula,
    )
    service = ExportService(
        tmp_path / "reports",
        _Gateway(report, _decision_export_record()),
    )

    csv_path, json_path = service.export_backtest(report.run_id)

    rows = list(csv.reader(io.StringIO(csv_path.read_text("utf-8-sig"))))
    formula_row = next(
        row
        for row in rows
        if row[0] == "reproduction"
        and row[1] == "allocator_configuration.formula_text"
    )
    assert formula_row[2].startswith("'")
    payload = json.loads(json_path.read_text("utf-8"))
    assert payload["reproduction"]["allocator_configuration"]["formula_text"] == formula


@pytest.mark.parametrize("run_id", ("../escape", "C:/private", "unsafe/id"))
def test_export_rejects_unsafe_ids_with_stable_public_error(tmp_path: Path, run_id: str) -> None:
    service = ExportService(
        tmp_path / "reports",
        _Gateway(_backtest_report(), _decision_export_record()),
    )

    with pytest.raises(ExportServiceError) as raised:
        service.export_backtest(run_id)

    assert str(raised.value) == "EXPORT_ID_INVALID"
    assert str(tmp_path) not in str(raised.value)


def test_export_translates_gateway_and_write_failures_without_private_context(
    tmp_path: Path,
) -> None:
    private_context = f"{SENTINEL} https://user:password@example.invalid {tmp_path}"

    class _FailingGateway(_Gateway):
        def load_backtest(self, run_id: str) -> BacktestReport:
            raise RuntimeError(private_context)

    service = ExportService(
        tmp_path / "reports",
        _FailingGateway(_backtest_report(), _decision_export_record()),
    )

    with pytest.raises(ExportServiceError) as raised:
        service.export_backtest("run")

    assert str(raised.value) == "EXPORT_BACKTEST_READ_FAILED"
    assert private_context not in str(raised.value)
