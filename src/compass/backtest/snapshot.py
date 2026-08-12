from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
import hashlib
import json
from math import isfinite
import re
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import pandas as pd  # type: ignore[import-untyped]

from compass.backtest.engine import BacktestEngine, BacktestRequest, BacktestResult


_STABLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")
_CONTENT_HASH = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{7,64}\Z")


class SnapshotIntegrityError(ValueError):
    """A snapshot or one of its exact dependencies was changed."""


class OriginalDataUnavailableError(LookupError):
    """An exact historical object required for reproduction is unavailable."""


class SnapshotRepositoryError(Exception):
    """Base class for storage-adapter failures at the snapshot boundary."""


class SnapshotObjectMissingError(SnapshotRepositoryError, LookupError):
    """An exact snapshot repository object does not exist."""


class SnapshotObjectIntegrityError(SnapshotRepositoryError, ValueError):
    """A snapshot repository object failed its backend integrity check."""


def _stable_id(value: object, *, label: str) -> str:
    if type(value) is not str or _STABLE_ID.fullmatch(value) is None:
        raise ValueError(f"{label} must be a stable identifier")
    assert isinstance(value, str)
    return value


def _canonical_decimal(value: Decimal) -> str:
    """Encode one finite Decimal without consulting the ambient Decimal context."""

    parts = value.as_tuple()
    digits = list(parts.digits)
    if not digits or all(digit == 0 for digit in digits):
        return "0"
    exponent = parts.exponent
    assert isinstance(exponent, int)
    while digits[-1] == 0:
        digits.pop()
        exponent += 1
    coefficient = "".join(str(digit) for digit in digits)
    sign = "-" if parts.sign else ""
    if exponent >= 0 and len(coefficient) + exponent <= 128:
        return f"{sign}{coefficient}{'0' * exponent}"
    if exponent < 0:
        decimal_point = len(coefficient) + exponent
        if decimal_point > 0:
            return f"{sign}{coefficient[:decimal_point]}.{coefficient[decimal_point:]}"
        leading_zeros = -decimal_point
        if 2 + leading_zeros + len(coefficient) <= 128:
            return f"{sign}0.{'0' * leading_zeros}{coefficient}"
    adjusted_exponent = exponent + len(coefficient) - 1
    mantissa = coefficient[0]
    if len(coefficient) > 1:
        mantissa = f"{mantissa}.{coefficient[1:]}"
    return f"{sign}{mantissa}e{adjusted_exponent:+d}"


@dataclass(frozen=True, slots=True)
class _FrozenEnum:
    enum_type: str
    value: object


def _freeze_value(value: object, *, path: str) -> object:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f"{path} float must be finite")
        return value
    if type(value) is Decimal:
        assert isinstance(value, Decimal)
        if not value.is_finite():
            raise ValueError(f"{path} Decimal must be finite")
        return value
    if type(value) is _FrozenEnum:
        assert isinstance(value, _FrozenEnum)
        if type(value.enum_type) is not str or not value.enum_type:
            raise ValueError(f"{path} enum type must be non-empty text")
        return _FrozenEnum(
            enum_type=value.enum_type,
            value=_freeze_value(value.value, path=f"{path}.enum_value"),
        )
    if isinstance(value, Enum):
        enum_type = type(value)
        return _FrozenEnum(
            enum_type=f"{enum_type.__module__}.{enum_type.__qualname__}",
            value=_freeze_value(value.value, path=f"{path}.enum_value"),
        )
    if type(value) in (date, datetime):
        return value
    if isinstance(value, Mapping):
        checked: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} mapping keys must be exact strings")
            checked[key] = _freeze_value(item, path=f"{path}.{key}")
        return MappingProxyType(dict(sorted(checked.items())))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            _freeze_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)
        )
    raise TypeError(f"unsupported canonical value at {path}: {type(value).__name__}")


def _freeze_mapping(
    value: object,
    *,
    label: str,
    require_non_empty: bool = False,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    frozen = _freeze_value(value, path=label)
    assert isinstance(frozen, Mapping)
    if require_non_empty and not frozen:
        raise ValueError(f"{label} must not be empty")
    return frozen


def _canonical_value(value: object, *, path: str) -> object:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f"{path} float must be finite")
        return {"$float": format(value, ".17g")}
    if type(value) is Decimal:
        assert isinstance(value, Decimal)
        if not value.is_finite():
            raise ValueError(f"{path} Decimal must be finite")
        return {"$decimal": _canonical_decimal(value)}
    if isinstance(value, _FrozenEnum):
        return {
            "$enum": value.enum_type,
            "value": _canonical_value(value.value, path=f"{path}.enum_value"),
        }
    if isinstance(value, Enum):
        return _canonical_value(_freeze_value(value, path=path), path=path)
    if type(value) is datetime:
        assert isinstance(value, datetime)
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{path} datetime must include a timezone")
        return {"$datetime": value.isoformat()}
    if type(value) is date:
        assert isinstance(value, date)
        return {"$date": value.isoformat()}
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} mapping keys must be exact strings")
            result[key] = _canonical_value(item, path=f"{path}.{key}")
        return result
    if type(value) is tuple:
        assert isinstance(value, tuple)
        return [
            _canonical_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)
        ]
    raise TypeError(f"unsupported canonical value at {path}: {type(value).__name__}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        _canonical_value(value, path="snapshot"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _decode_canonical_value(value: object, *, path: str) -> object:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is list:
        assert isinstance(value, list)
        return tuple(
            _decode_canonical_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if type(value) is not dict:
        raise ValueError(f"{path} has an unsupported canonical value")
    assert isinstance(value, dict)
    if set(value) == {"$float"}:
        raw = value["$float"]
        if type(raw) is not str:
            raise ValueError(f"{path} float tag is invalid")
        parsed_float = float(raw)
        if not isfinite(parsed_float) or format(parsed_float, ".17g") != raw:
            raise ValueError(f"{path} float tag is not canonical")
        return parsed_float
    if set(value) == {"$decimal"}:
        raw = value["$decimal"]
        if type(raw) is not str:
            raise ValueError(f"{path} decimal tag is invalid")
        parsed_decimal = Decimal(raw)
        if not parsed_decimal.is_finite() or _canonical_decimal(parsed_decimal) != raw:
            raise ValueError(f"{path} decimal tag is not canonical")
        return parsed_decimal
    if set(value) == {"$date"}:
        raw = value["$date"]
        if type(raw) is not str:
            raise ValueError(f"{path} date tag is invalid")
        parsed_date = date.fromisoformat(raw)
        if parsed_date.isoformat() != raw:
            raise ValueError(f"{path} date tag is not canonical")
        return parsed_date
    if set(value) == {"$datetime"}:
        raw = value["$datetime"]
        if type(raw) is not str:
            raise ValueError(f"{path} datetime tag is invalid")
        parsed_datetime = datetime.fromisoformat(raw)
        if (
            parsed_datetime.tzinfo is None
            or parsed_datetime.utcoffset() is None
            or parsed_datetime.isoformat() != raw
        ):
            raise ValueError(f"{path} datetime tag is not canonical")
        return parsed_datetime
    if set(value) == {"$enum", "value"}:
        enum_type = value["$enum"]
        if type(enum_type) is not str or not enum_type:
            raise ValueError(f"{path} enum tag is invalid")
        return _FrozenEnum(
            enum_type=enum_type,
            value=_decode_canonical_value(value["value"], path=f"{path}.enum_value"),
        )
    checked: dict[str, object] = {}
    for key, item in value.items():
        if type(key) is not str:
            raise ValueError(f"{path} mapping key is invalid")
        checked[key] = _decode_canonical_value(item, path=f"{path}.{key}")
    return checked


@dataclass(frozen=True, slots=True)
class ManifestReference:
    manifest_id: str
    content_hash: str

    def __post_init__(self) -> None:
        _stable_id(self.manifest_id, label="manifest_id")
        if type(self.content_hash) is not str or _CONTENT_HASH.fullmatch(self.content_hash) is None:
            raise ValueError("content_hash must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class StrategySnapshot:
    sleeve_id: str
    strategy_type: str
    strategy_version: str
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        _stable_id(self.sleeve_id, label="sleeve_id")
        _stable_id(self.strategy_type, label="strategy_type")
        _stable_id(self.strategy_version, label="strategy_version")
        object.__setattr__(
            self,
            "parameters",
            _freeze_mapping(self.parameters, label=f"strategy[{self.sleeve_id}].parameters"),
        )


@dataclass(frozen=True, slots=True, init=False)
class RunSnapshot:
    snapshot_id: str
    run_id: str
    schema_version: int
    market_manifests: tuple[ManifestReference, ...]
    data_quality: Mapping[str, object]
    strategies: tuple[StrategySnapshot, ...]
    instrument_pool: Mapping[str, object]
    survivorship_bias: Mapping[str, object]
    allocator_configuration: Mapping[str, object]
    risk_configuration: Mapping[str, object]
    market_rule_configuration: Mapping[str, object]
    fee_profile_configuration: Mapping[str, object]
    app_git_commit: str
    random_seed: int

    def __init__(
        self,
        *,
        run_id: str,
        schema_version: int,
        market_manifests: Sequence[ManifestReference],
        data_quality: Mapping[str, object],
        strategies: Sequence[StrategySnapshot],
        instrument_pool: Mapping[str, object],
        survivorship_bias: Mapping[str, object],
        allocator_configuration: Mapping[str, object],
        risk_configuration: Mapping[str, object],
        market_rule_configuration: Mapping[str, object],
        fee_profile_configuration: Mapping[str, object],
        app_git_commit: str,
        random_seed: int,
        snapshot_id: str | None = None,
    ) -> None:
        _stable_id(run_id, label="run_id")
        if type(schema_version) is not int:
            raise TypeError("schema_version must be an exact integer")
        if schema_version <= 0:
            raise ValueError("schema_version must be positive")
        manifests = tuple(market_manifests)
        if not manifests or any(type(item) is not ManifestReference for item in manifests):
            raise TypeError("market_manifests must contain exact ManifestReference values")
        manifest_ids = tuple(item.manifest_id for item in manifests)
        if len(set(manifest_ids)) != len(manifest_ids):
            raise ValueError("market manifest ids must be unique")
        checked_strategies = tuple(strategies)
        if not checked_strategies or any(
            type(item) is not StrategySnapshot for item in checked_strategies
        ):
            raise TypeError("strategies must contain exact StrategySnapshot values")
        sleeve_ids = tuple(item.sleeve_id for item in checked_strategies)
        if len(set(sleeve_ids)) != len(sleeve_ids):
            raise ValueError("strategy sleeve ids must be unique")
        if type(app_git_commit) is not str or _GIT_COMMIT.fullmatch(app_git_commit) is None:
            raise ValueError("app_git_commit must be a lowercase hexadecimal Git commit")
        if type(random_seed) is not int:
            raise TypeError("random_seed must be an exact integer")
        if random_seed < 0:
            raise ValueError("random_seed must be non-negative")

        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "market_manifests", manifests)
        object.__setattr__(
            self,
            "data_quality",
            _freeze_mapping(data_quality, label="data_quality", require_non_empty=True),
        )
        object.__setattr__(self, "strategies", checked_strategies)
        object.__setattr__(
            self,
            "instrument_pool",
            _freeze_mapping(instrument_pool, label="instrument_pool", require_non_empty=True),
        )
        object.__setattr__(
            self,
            "survivorship_bias",
            _freeze_mapping(
                survivorship_bias, label="survivorship_bias", require_non_empty=True
            ),
        )
        object.__setattr__(
            self,
            "allocator_configuration",
            _freeze_mapping(
                allocator_configuration,
                label="allocator_configuration",
                require_non_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "risk_configuration",
            _freeze_mapping(
                risk_configuration, label="risk_configuration", require_non_empty=True
            ),
        )
        object.__setattr__(
            self,
            "market_rule_configuration",
            _freeze_mapping(
                market_rule_configuration,
                label="market_rule_configuration",
                require_non_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "fee_profile_configuration",
            _freeze_mapping(
                fee_profile_configuration,
                label="fee_profile_configuration",
                require_non_empty=True,
            ),
        )
        object.__setattr__(self, "app_git_commit", app_git_commit)
        object.__setattr__(self, "random_seed", random_seed)
        computed = self.computed_snapshot_id
        if snapshot_id is not None and snapshot_id != computed:
            raise SnapshotIntegrityError("SNAPSHOT_HASH_MISMATCH: supplied id does not match payload")
        object.__setattr__(self, "snapshot_id", computed)

    @property
    def market_manifest_ids(self) -> tuple[str, ...]:
        return tuple(item.manifest_id for item in self.market_manifests)

    def _payload(self) -> Mapping[str, object]:
        return {
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "market_manifests": tuple(
                {"manifest_id": item.manifest_id, "content_hash": item.content_hash}
                for item in self.market_manifests
            ),
            "data_quality": self.data_quality,
            "strategies": tuple(
                {
                    "sleeve_id": item.sleeve_id,
                    "strategy_type": item.strategy_type,
                    "strategy_version": item.strategy_version,
                    "parameters": item.parameters,
                }
                for item in self.strategies
            ),
            "instrument_pool": self.instrument_pool,
            "survivorship_bias": self.survivorship_bias,
            "allocator_configuration": self.allocator_configuration,
            "risk_configuration": self.risk_configuration,
            "market_rule_configuration": self.market_rule_configuration,
            "fee_profile_configuration": self.fee_profile_configuration,
            "app_git_commit": self.app_git_commit,
            "random_seed": self.random_seed,
        }

    @property
    def canonical_payload_json(self) -> str:
        return _canonical_json(self._payload())

    @property
    def computed_snapshot_id(self) -> str:
        return hashlib.sha256(self.canonical_payload_json.encode("utf-8")).hexdigest()

    @property
    def content_hash(self) -> str:
        return self.computed_snapshot_id

    def verify_integrity(self) -> None:
        if self.snapshot_id != self.computed_snapshot_id:
            raise SnapshotIntegrityError("SNAPSHOT_HASH_MISMATCH: snapshot payload was changed")

    @classmethod
    def from_canonical_payload_json(
        cls,
        payload_json: str,
        *,
        snapshot_id: str,
    ) -> RunSnapshot:
        """Rehydrate the exact versioned payload persisted by ``canonical_payload_json``."""

        if type(payload_json) is not str:
            raise TypeError("snapshot payload must be exact text")
        if type(snapshot_id) is not str or _CONTENT_HASH.fullmatch(snapshot_id) is None:
            raise ValueError("snapshot_id must be a lowercase SHA-256 digest")
        try:
            raw = json.loads(payload_json)
            decoded = _decode_canonical_value(raw, path="snapshot")
        except (ValueError, TypeError, json.JSONDecodeError):
            raise SnapshotIntegrityError("SNAPSHOT_PAYLOAD_INVALID") from None
        if not isinstance(decoded, Mapping):
            raise SnapshotIntegrityError("SNAPSHOT_PAYLOAD_INVALID")
        expected = {
            "run_id",
            "schema_version",
            "market_manifests",
            "data_quality",
            "strategies",
            "instrument_pool",
            "survivorship_bias",
            "allocator_configuration",
            "risk_configuration",
            "market_rule_configuration",
            "fee_profile_configuration",
            "app_git_commit",
            "random_seed",
        }
        if set(decoded) != expected:
            raise SnapshotIntegrityError("SNAPSHOT_PAYLOAD_INVALID")
        raw_manifests = decoded["market_manifests"]
        raw_strategies = decoded["strategies"]
        if type(raw_manifests) is not tuple or type(raw_strategies) is not tuple:
            raise SnapshotIntegrityError("SNAPSHOT_PAYLOAD_INVALID")
        try:
            manifests = tuple(
                ManifestReference(
                    manifest_id=item["manifest_id"],
                    content_hash=item["content_hash"],
                )
                for item in raw_manifests
                if isinstance(item, Mapping)
                and set(item) == {"manifest_id", "content_hash"}
            )
            strategies = tuple(
                StrategySnapshot(
                    sleeve_id=item["sleeve_id"],
                    strategy_type=item["strategy_type"],
                    strategy_version=item["strategy_version"],
                    parameters=item["parameters"],
                )
                for item in raw_strategies
                if isinstance(item, Mapping)
                and set(item)
                == {"sleeve_id", "strategy_type", "strategy_version", "parameters"}
                and isinstance(item["parameters"], Mapping)
            )
            if len(manifests) != len(raw_manifests) or len(strategies) != len(raw_strategies):
                raise ValueError("invalid snapshot sequence entry")
            mapping_fields = {
                name: decoded[name]
                for name in (
                    "data_quality",
                    "instrument_pool",
                    "survivorship_bias",
                    "allocator_configuration",
                    "risk_configuration",
                    "market_rule_configuration",
                    "fee_profile_configuration",
                )
            }
            if any(not isinstance(value, Mapping) for value in mapping_fields.values()):
                raise TypeError("snapshot sections must be mappings")
            snapshot = cls(
                run_id=decoded["run_id"],
                schema_version=decoded["schema_version"],
                market_manifests=manifests,
                data_quality=mapping_fields["data_quality"],
                strategies=strategies,
                instrument_pool=mapping_fields["instrument_pool"],
                survivorship_bias=mapping_fields["survivorship_bias"],
                allocator_configuration=mapping_fields["allocator_configuration"],
                risk_configuration=mapping_fields["risk_configuration"],
                market_rule_configuration=mapping_fields["market_rule_configuration"],
                fee_profile_configuration=mapping_fields["fee_profile_configuration"],
                app_git_commit=decoded["app_git_commit"],
                random_seed=decoded["random_seed"],
                snapshot_id=snapshot_id,
            )
        except (KeyError, TypeError, ValueError):
            raise SnapshotIntegrityError("SNAPSHOT_PAYLOAD_INVALID") from None
        if snapshot.canonical_payload_json != payload_json:
            raise SnapshotIntegrityError("SNAPSHOT_PAYLOAD_NOT_CANONICAL")
        return snapshot


@dataclass(frozen=True, slots=True, init=False)
class ReplayManifest:
    reference: ManifestReference
    _bars: pd.DataFrame

    def __init__(self, reference: ManifestReference, bars: pd.DataFrame) -> None:
        if type(reference) is not ManifestReference:
            raise TypeError("reference must be an exact ManifestReference")
        if not isinstance(bars, pd.DataFrame):
            raise TypeError("bars must be a DataFrame")
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "_bars", bars.copy(deep=True))

    @property
    def bars(self) -> pd.DataFrame:
        return self._bars.copy(deep=True)


@runtime_checkable
class SnapshotRepository(Protocol):
    """Exact-ID storage boundary with backend-independent failure semantics.

    Adapters must translate missing objects to ``SnapshotObjectMissingError``
    and corrupt/unreadable objects to ``SnapshotObjectIntegrityError``. The
    ``read_manifest`` postcondition includes verifying the stored manifest bytes
    against that manifest's recorded content hash before returning its bars.
    """

    def save_snapshot(self, snapshot: RunSnapshot) -> None:
        """Persist a content-addressed immutable snapshot."""

    def load_snapshot(self, snapshot_id: str) -> RunSnapshot:
        """Load exactly ``snapshot_id`` or raise a missing-object error."""

    def load_manifest_ref(self, manifest_id: str) -> ManifestReference:
        """Load exact manifest metadata without consulting a latest pointer."""

    def read_manifest(self, manifest_id: str) -> pd.DataFrame:
        """Read exact, byte-hash-verified ``manifest_id`` or raise a repository error."""


@runtime_checkable
class ReplayRequestFactory(Protocol):
    def build_request(
        self,
        snapshot: RunSnapshot,
        manifests: tuple[ReplayManifest, ...],
    ) -> BacktestRequest:
        """Reconstruct a fresh request from the verified historical dependencies."""


@dataclass(frozen=True, slots=True)
class ReproductionResult:
    snapshot: RunSnapshot
    result: BacktestResult

    def __post_init__(self) -> None:
        if type(self.snapshot) is not RunSnapshot:
            raise TypeError("snapshot must be an exact RunSnapshot")
        if type(self.result) is not BacktestResult:
            raise TypeError("result must be an exact BacktestResult")
        if self.result.run_id != self.snapshot.run_id:
            raise ValueError("reproduced result run_id must match the original snapshot")


def reproduce(
    snapshot_id: str,
    *,
    repository: SnapshotRepository,
    request_factory: ReplayRequestFactory,
) -> ReproductionResult:
    """Replay one exact snapshot with no latest-data or cached-result fallback."""

    if type(snapshot_id) is not str or _CONTENT_HASH.fullmatch(snapshot_id) is None:
        raise ValueError("snapshot_id must be a lowercase SHA-256 digest")
    if not isinstance(repository, SnapshotRepository):
        raise TypeError("repository must implement SnapshotRepository")
    if not isinstance(request_factory, ReplayRequestFactory):
        raise TypeError("request_factory must implement ReplayRequestFactory")
    try:
        snapshot = repository.load_snapshot(snapshot_id)
    except (SnapshotObjectMissingError, KeyError, FileNotFoundError) as error:
        raise OriginalDataUnavailableError(
            f"SNAPSHOT_NOT_FOUND: original snapshot {snapshot_id} is unavailable"
        ) from error
    except (SnapshotObjectIntegrityError, ValueError) as error:
        raise SnapshotIntegrityError(
            f"SNAPSHOT_REPOSITORY_INTEGRITY: snapshot {snapshot_id} failed validation"
        ) from error
    except SnapshotRepositoryError as error:
        raise SnapshotIntegrityError(
            f"SNAPSHOT_REPOSITORY_INTEGRITY: snapshot {snapshot_id} repository failure"
        ) from error
    if type(snapshot) is not RunSnapshot or snapshot.snapshot_id != snapshot_id:
        raise SnapshotIntegrityError("SNAPSHOT_ID_MISMATCH: repository returned another snapshot")
    snapshot.verify_integrity()

    replay_manifests: list[ReplayManifest] = []
    for expected in snapshot.market_manifests:
        try:
            actual = repository.load_manifest_ref(expected.manifest_id)
        except (SnapshotObjectMissingError, KeyError, FileNotFoundError) as error:
            raise OriginalDataUnavailableError(
                f"ORIGINAL_DATA_UNAVAILABLE: manifest {expected.manifest_id} is unavailable"
            ) from error
        except (SnapshotObjectIntegrityError, ValueError) as error:
            raise SnapshotIntegrityError(
                f"MANIFEST_REPOSITORY_INTEGRITY: manifest {expected.manifest_id} metadata failed validation"
            ) from error
        except SnapshotRepositoryError as error:
            raise SnapshotIntegrityError(
                f"MANIFEST_REPOSITORY_INTEGRITY: manifest {expected.manifest_id} repository failure"
            ) from error
        if type(actual) is not ManifestReference or actual.manifest_id != expected.manifest_id:
            raise SnapshotIntegrityError(
                f"MANIFEST_ID_MISMATCH: expected exact manifest {expected.manifest_id}"
            )
        if actual.content_hash != expected.content_hash:
            raise SnapshotIntegrityError(
                f"MANIFEST_HASH_MISMATCH: manifest {expected.manifest_id} changed"
            )
        try:
            bars = repository.read_manifest(expected.manifest_id)
        except (SnapshotObjectMissingError, KeyError, FileNotFoundError) as error:
            raise OriginalDataUnavailableError(
                f"ORIGINAL_DATA_UNAVAILABLE: manifest {expected.manifest_id} data is unavailable"
            ) from error
        except (SnapshotObjectIntegrityError, ValueError) as error:
            raise SnapshotIntegrityError(
                f"MANIFEST_INTEGRITY_ERROR: manifest {expected.manifest_id} failed validation"
            ) from error
        except SnapshotRepositoryError as error:
            raise SnapshotIntegrityError(
                f"MANIFEST_INTEGRITY_ERROR: manifest {expected.manifest_id} repository failure"
            ) from error
        replay_manifests.append(ReplayManifest(expected, bars))

    request = request_factory.build_request(snapshot, tuple(replay_manifests))
    if type(request) is not BacktestRequest:
        raise TypeError("request_factory must return an exact BacktestRequest")
    result = BacktestEngine().run(request)
    return ReproductionResult(snapshot=snapshot, result=result)
