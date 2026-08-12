from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date
from decimal import Decimal, localcontext
from enum import Enum, StrEnum
import hashlib
from types import MappingProxyType

import pandas as pd  # type: ignore[import-untyped]
import pytest

from compass.backtest.engine import BacktestEngine, BacktestRequest, DecisionTarget
from compass.backtest.market_rules import (
    MarketRuleBook,
    MarketRuleProfile,
    OddLotSellPolicy,
    PriceLimitMode,
    SettlementMode,
)
from compass.backtest.snapshot import (
    ManifestReference,
    OriginalDataUnavailableError,
    ReplayManifest,
    RunSnapshot,
    SnapshotIntegrityError,
    SnapshotObjectIntegrityError,
    SnapshotObjectMissingError,
    StrategySnapshot,
    reproduce,
)
from compass.domain.market import AssetType, Exchange, Instrument, InstrumentId
from compass.risk.engine import RiskEngine
from compass.strategies.base import StrategyContext


SYMBOL = InstrumentId.parse("SSE.510300")
INSTRUMENT = Instrument(SYMBOL, AssetType.ETF, 100, False)


class _PoolMode(StrEnum):
    STATIC = "static"


class _Source:
    def targets(self, context: StrategyContext) -> DecisionTarget:
        return DecisionTarget(
            weights={SYMBOL: Decimal("1")},
            sleeve_weights={SYMBOL: {"rotation": Decimal("1")}},
        )


def _profile() -> MarketRuleProfile:
    return MarketRuleProfile(
        profile_id="ETF-ZERO-COST",
        exchange=Exchange.SSE,
        asset_type=AssetType.ETF,
        effective_from=date(2020, 1, 1),
        effective_to=None,
        buy_lot_size=100,
        odd_lot_sell_policy=OddLotSellPolicy.POSITION_REMAINDER_ONLY,
        settlement_mode=SettlementMode.T_PLUS_ONE,
        same_day_sell_eligible=False,
        price_limit_mode=PriceLimitMode.PERCENTAGE,
        price_limit_rate=Decimal("0.10"),
        risk_warning_price_limit_rate=None,
        commission_rate=Decimal("0"),
        minimum_commission=Decimal("0"),
        sell_stamp_duty_rate=Decimal("0"),
        transfer_fee_rate=Decimal("0"),
        slippage_bps=Decimal("0"),
        maximum_volume_participation=Decimal("1"),
        fee_profile_confirmed=True,
    )


def _bars(second_close: str = "4.20") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [Decimal("4.00"), Decimal("4.10")],
            "high": [Decimal("4.10"), Decimal("4.30")],
            "low": [Decimal("3.90"), Decimal("4.00")],
            "close": [Decimal("4.00"), Decimal(second_close)],
            "volume": [100_000, 100_000],
            "amount": [Decimal("400000"), Decimal("420000")],
            "suspended": [False, False],
            "limit_up": [Decimal("4.40"), Decimal("4.40")],
            "limit_down": [Decimal("3.60"), Decimal("3.60")],
        },
        index=pd.to_datetime(["2026-07-20", "2026-07-21"]),
    )


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _snapshot(**overrides: object) -> RunSnapshot:
    values: dict[str, object] = {
        "run_id": "snapshot-run",
        "schema_version": 1,
        "market_manifests": (ManifestReference("manifest-original", _hash("original")),),
        "data_quality": {"mode": "strict", "accepted": True},
        "strategies": (
            StrategySnapshot(
                sleeve_id="rotation",
                strategy_type="etf_rotation",
                strategy_version="1.2.0",
                parameters={"lookbacks": [20, 60], "minimum_score": Decimal("0.01")},
            ),
        ),
        "instrument_pool": {"instruments": [str(SYMBOL)], "mode": _PoolMode.STATIC},
        "survivorship_bias": {"present": True, "reason": "current_static_pool"},
        "allocator_configuration": {"type": "deterministic", "cash_floor": Decimal("0.10")},
        "risk_configuration": {"rules": []},
        "market_rule_configuration": {"profile_ids": ["ETF-ZERO-COST"]},
        "fee_profile_configuration": {"profile": "zero", "confirmed": True},
        "app_git_commit": "ed23ae6c4d13cc5b8ecb4ce164b599037aaa44ad",
        "random_seed": 7,
    }
    values.update(overrides)
    return RunSnapshot(**values)  # type: ignore[arg-type]


class _Repository:
    def __init__(self) -> None:
        self.snapshots: dict[str, RunSnapshot] = {}
        self.manifests: dict[str, tuple[ManifestReference, pd.DataFrame]] = {}
        self.latest_manifest_id: str | None = None
        self.exact_reads: list[str] = []

    def save_snapshot(self, snapshot: RunSnapshot) -> None:
        self.snapshots[snapshot.snapshot_id] = snapshot

    def load_snapshot(self, snapshot_id: str) -> RunSnapshot:
        return self.snapshots[snapshot_id]

    def load_manifest_ref(self, manifest_id: str) -> ManifestReference:
        self.exact_reads.append(manifest_id)
        return self.manifests[manifest_id][0]

    def read_manifest(self, manifest_id: str) -> pd.DataFrame:
        return self.manifests[manifest_id][1].copy(deep=True)

    def set_latest_manifest(self, manifest_id: str) -> None:
        self.latest_manifest_id = manifest_id


class _RequestFactory:
    def __init__(self) -> None:
        self.calls = 0
        self.seen_manifest_ids: tuple[str, ...] = ()

    def build_request(
        self,
        snapshot: RunSnapshot,
        manifests: tuple[ReplayManifest, ...],
    ) -> BacktestRequest:
        self.calls += 1
        self.seen_manifest_ids = tuple(item.reference.manifest_id for item in manifests)
        assert len(manifests) == 1
        return BacktestRequest(
            run_id=snapshot.run_id,
            sessions=(date(2026, 7, 20), date(2026, 7, 21)),
            instruments={SYMBOL: INSTRUMENT},
            bars={SYMBOL: manifests[0].bars},
            initial_cash=Decimal("1000"),
            initial_positions=(),
            corporate_actions=(),
            decision_source=_Source(),
            risk_engine=RiskEngine(()),
            rule_book=MarketRuleBook((_profile(),)),
        )


class _FailingRepository(_Repository):
    def __init__(
        self,
        source: _Repository,
        *,
        failure_stage: str,
        failure: SnapshotObjectMissingError | SnapshotObjectIntegrityError,
    ) -> None:
        super().__init__()
        self.snapshots.update(source.snapshots)
        self.manifests.update(source.manifests)
        self.latest_manifest_id = source.latest_manifest_id
        self.failure_stage = failure_stage
        self.failure = failure

    def load_snapshot(self, snapshot_id: str) -> RunSnapshot:
        if self.failure_stage == "snapshot":
            raise self.failure
        return super().load_snapshot(snapshot_id)

    def load_manifest_ref(self, manifest_id: str) -> ManifestReference:
        if self.failure_stage == "manifest_ref":
            raise self.failure
        return super().load_manifest_ref(manifest_id)

    def read_manifest(self, manifest_id: str) -> pd.DataFrame:
        if self.failure_stage == "manifest_data":
            raise self.failure
        return super().read_manifest(manifest_id)


def _prepared_repository() -> tuple[_Repository, RunSnapshot]:
    repository = _Repository()
    snapshot = _snapshot()
    reference = snapshot.market_manifests[0]
    repository.save_snapshot(snapshot)
    repository.manifests[reference.manifest_id] = (reference, _bars())
    repository.manifests["manifest-latest"] = (
        ManifestReference("manifest-latest", _hash("latest")),
        _bars("8.00"),
    )
    return repository, snapshot


def test_reproduction_uses_original_manifest_not_latest_data_and_runs_fresh() -> None:
    repository, snapshot = _prepared_repository()
    original_factory = _RequestFactory()
    original = BacktestEngine().run(
        original_factory.build_request(
            snapshot,
            (ReplayManifest(snapshot.market_manifests[0], _bars()),),
        )
    )
    repository.set_latest_manifest("manifest-latest")
    factory = _RequestFactory()

    replay = reproduce(snapshot.snapshot_id, repository=repository, request_factory=factory)

    assert replay.result == original
    assert replay.snapshot is snapshot
    assert replay.snapshot.market_manifest_ids == ("manifest-original",)
    assert repository.exact_reads == ["manifest-original"]
    assert factory.seen_manifest_ids == ("manifest-original",)
    assert factory.calls == 1


def test_missing_original_manifest_fails_without_using_latest() -> None:
    repository, snapshot = _prepared_repository()
    del repository.manifests["manifest-original"]
    repository.set_latest_manifest("manifest-latest")

    with pytest.raises(OriginalDataUnavailableError, match="ORIGINAL_DATA_UNAVAILABLE"):
        reproduce(snapshot.snapshot_id, repository=repository, request_factory=_RequestFactory())


def test_changed_manifest_hash_fails_integrity_check() -> None:
    repository, snapshot = _prepared_repository()
    repository.manifests["manifest-original"] = (
        ManifestReference("manifest-original", _hash("changed")),
        _bars(),
    )

    with pytest.raises(SnapshotIntegrityError, match="MANIFEST_HASH_MISMATCH"):
        reproduce(snapshot.snapshot_id, repository=repository, request_factory=_RequestFactory())


def test_tampered_snapshot_payload_or_requested_id_fails_before_replay() -> None:
    repository, snapshot = _prepared_repository()
    object.__setattr__(snapshot, "run_id", "tampered-run")

    with pytest.raises(SnapshotIntegrityError, match="SNAPSHOT_HASH_MISMATCH"):
        reproduce(snapshot.snapshot_id, repository=repository, request_factory=_RequestFactory())
    with pytest.raises(OriginalDataUnavailableError, match="SNAPSHOT_NOT_FOUND"):
        reproduce("0" * 64, repository=repository, request_factory=_RequestFactory())


def test_canonical_hash_ignores_mapping_insertion_order_and_uses_explicit_scalars() -> None:
    left = _snapshot(
        data_quality={"accepted": True, "mode": "strict"},
        allocator_configuration={"cash_floor": Decimal("0.10"), "type": "deterministic"},
    )
    right = _snapshot(
        data_quality={"mode": "strict", "accepted": True},
        allocator_configuration={"type": "deterministic", "cash_floor": Decimal("0.10")},
    )

    assert left.snapshot_id == right.snapshot_id
    assert left.canonical_payload_json == right.canonical_payload_json
    assert '"$decimal":"0.1"' in left.canonical_payload_json
    assert '"$enum"' in left.canonical_payload_json


def test_snapshot_deep_freezes_inputs_and_caller_mutation_cannot_change_hash() -> None:
    parameters: dict[str, object] = {"lookbacks": [20, 60]}
    pool: dict[str, object] = {"instruments": [str(SYMBOL)]}
    snapshot = _snapshot(
        strategies=(StrategySnapshot("rotation", "etf_rotation", "1", parameters),),
        instrument_pool=pool,
    )
    original_id = snapshot.snapshot_id
    original_json = snapshot.canonical_payload_json

    parameters["lookbacks"] = [5]
    pool["instruments"] = []

    assert snapshot.snapshot_id == original_id
    assert snapshot.canonical_payload_json == original_json
    assert snapshot.strategies[0].parameters["lookbacks"] == (20, 60)
    assert snapshot.instrument_pool["instruments"] == (str(SYMBOL),)
    assert isinstance(snapshot.instrument_pool, MappingProxyType)
    with pytest.raises(TypeError):
        snapshot.instrument_pool["new"] = True  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        snapshot.random_seed = 9  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("run_id", "another-run"),
        ("schema_version", 2),
        ("market_manifests", (ManifestReference("manifest-two", _hash("two")),)),
        ("data_quality", {"mode": "degraded"}),
        ("strategies", (StrategySnapshot("other", "dual_ma", "1", {}),)),
        ("instrument_pool", {"instruments": ["SZSE.159915"]}),
        ("survivorship_bias", {"present": False}),
        ("allocator_configuration", {"type": "equal"}),
        ("risk_configuration", {"rules": ["cap"]}),
        ("market_rule_configuration", {"profile_ids": ["OTHER"]}),
        ("fee_profile_configuration", {"profile": "fees"}),
        ("app_git_commit", "f" * 40),
        ("random_seed", 8),
    ],
)
def test_changing_any_captured_field_changes_snapshot_id(field: str, changed: object) -> None:
    original = _snapshot()
    modified = _snapshot(**{field: changed})

    assert modified.snapshot_id != original.snapshot_id


def test_snapshot_rejects_unsupported_values_and_invalid_supplied_id() -> None:
    with pytest.raises(TypeError, match="unsupported canonical value"):
        _snapshot(data_quality={"bad": {"set"}})
    with pytest.raises(SnapshotIntegrityError, match="SNAPSHOT_HASH_MISMATCH"):
        _snapshot(snapshot_id="0" * 64)


def test_repository_protocol_is_dependency_explicit_not_module_global() -> None:
    repository, snapshot = _prepared_repository()
    factory = _RequestFactory()

    first = reproduce(snapshot.snapshot_id, repository=repository, request_factory=factory)
    second = reproduce(snapshot.snapshot_id, repository=repository, request_factory=factory)

    assert first.result == second.result
    assert factory.calls == 2


def test_decimal_canonicalization_is_collision_free_and_context_independent() -> None:
    left_value = Decimal("1.12345678901234567890123456789")
    right_value = Decimal("1.12345678901234567890123456788")
    left = _snapshot(data_quality={"value": left_value})
    right = _snapshot(data_quality={"value": right_value})

    with localcontext() as context:
        context.prec = 10
        low_precision = _snapshot(data_quality={"value": left_value})
    with localcontext() as context:
        context.prec = 80
        high_precision = _snapshot(data_quality={"value": left_value})

    assert left.snapshot_id != right.snapshot_id
    assert left.canonical_payload_json != right.canonical_payload_json
    assert low_precision.snapshot_id == high_precision.snapshot_id == left.snapshot_id
    assert low_precision.canonical_payload_json == high_precision.canonical_payload_json


def test_decimal_canonicalization_equates_trailing_and_signed_zero_without_exponent_blowup() -> None:
    tenth = _snapshot(data_quality={"value": Decimal("0.10")})
    normalized_tenth = _snapshot(data_quality={"value": Decimal("0.1")})
    positive_zero = _snapshot(data_quality={"value": Decimal("0")})
    negative_zero = _snapshot(data_quality={"value": Decimal("-0.000")})
    extreme = _snapshot(data_quality={"value": Decimal("1e100000")})

    assert tenth.snapshot_id == normalized_tenth.snapshot_id
    assert positive_zero.snapshot_id == negative_zero.snapshot_id
    assert len(extreme.canonical_payload_json) < 2_000


def test_mutable_enum_value_is_captured_without_retaining_caller_reference() -> None:
    mutable_enum = Enum("MutableConfiguration", {"ACTIVE": {"enabled": True}})
    member = mutable_enum.ACTIVE
    snapshot = _snapshot(instrument_pool={"instruments": [str(SYMBOL)], "mode": member})
    original_json = snapshot.canonical_payload_json
    original_id = snapshot.snapshot_id
    captured = snapshot.instrument_pool["mode"]

    member.value["enabled"] = False

    assert captured is not member
    assert snapshot.canonical_payload_json == original_json
    assert snapshot.computed_snapshot_id == original_id


@pytest.mark.parametrize(
    "field",
    [
        "data_quality",
        "instrument_pool",
        "survivorship_bias",
        "allocator_configuration",
        "risk_configuration",
        "market_rule_configuration",
        "fee_profile_configuration",
    ],
)
def test_snapshot_rejects_semantically_empty_required_sections(field: str) -> None:
    with pytest.raises(ValueError, match=rf"{field} must not be empty"):
        _snapshot(**{field: {}})


@pytest.mark.parametrize("stage", ["snapshot", "manifest_ref", "manifest_data"])
@pytest.mark.parametrize(
    ("failure_type", "public_error", "message"),
    [
        (
            SnapshotObjectMissingError,
            OriginalDataUnavailableError,
            "(SNAPSHOT_NOT_FOUND|ORIGINAL_DATA_UNAVAILABLE)",
        ),
        (
            SnapshotObjectIntegrityError,
            SnapshotIntegrityError,
            "(SNAPSHOT_REPOSITORY_INTEGRITY|MANIFEST_REPOSITORY_INTEGRITY|MANIFEST_INTEGRITY_ERROR)",
        ),
    ],
)
def test_repository_boundary_translates_standardized_failures_at_every_load_stage(
    stage: str,
    failure_type: type[SnapshotObjectMissingError] | type[SnapshotObjectIntegrityError],
    public_error: type[OriginalDataUnavailableError] | type[SnapshotIntegrityError],
    message: str,
) -> None:
    source, snapshot = _prepared_repository()
    repository = _FailingRepository(
        source,
        failure_stage=stage,
        failure=failure_type("backend details"),
    )
    factory = _RequestFactory()

    with pytest.raises(public_error, match=message):
        reproduce(snapshot.snapshot_id, repository=repository, request_factory=factory)

    assert factory.calls == 0
