from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import cast

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from compass.backtest.snapshot import (
    SnapshotObjectIntegrityError,
    SnapshotObjectMissingError,
)
from compass.services.export_service import (
    DecisionExportRecord,
    DecisionManifestProvenance,
    DecisionStrategyProvenance,
    decision_snapshot_run_id,
)
from compass.services.decision_service import DecisionResult
from compass.services.safe_display import safe_identifier
from compass.storage.account_repository import AccountIntegrityError, AccountRepository
from compass.storage.canonical_json import (
    canonical_json,
    content_hash,
    decode_canonical_json,
)
from compass.storage.database import Database
from compass.storage.decision_result_codec import (
    decode_decision_result,
    encode_decision_result,
    json_value,
)
from compass.storage.market_store import MarketStore
from compass.storage.models import (
    BacktestReportRecord,
    DecisionExportRecordRow,
    LegacyDecisionExportRecord,
    RunSnapshotRecord,
)
from compass.storage.run_snapshot_repository import RunSnapshotRepository
from compass.storage.write_order import next_write_order


Clock = Callable[[], datetime]
_SCHEMA_VERSION = 2


class DecisionExportRepository:
    def __init__(
        self,
        database: Database,
        market_store: MarketStore,
        snapshots: RunSnapshotRepository,
        *,
        clock: Clock,
    ) -> None:
        self._database = database
        self._market_store = market_store
        self._snapshots = snapshots
        self._clock = clock

    def save(
        self,
        record: DecisionExportRecord,
        *,
        before_persist: Callable[[Session], None] | None = None,
    ) -> None:
        if type(record) is not DecisionExportRecord:
            raise TypeError("record must be an exact DecisionExportRecord")
        if before_persist is not None and not callable(before_persist):
            raise TypeError("before_persist must be callable")
        record.verify_integrity()
        if record.snapshot.run_id != decision_snapshot_run_id(record.decision_id):
            raise ValueError("DECISION_EXPORT_INTEGRITY")
        self._verify_account_link(record.result)
        self._verify_manifests(record)
        payload_json = canonical_json(self._payload(record))
        created_at = self._clock()
        if (
            type(created_at) is not datetime
            or created_at.tzinfo is None
            or created_at.utcoffset() is None
        ):
            raise ValueError("decision repository clock must be timezone-aware")
        try:
            with self._database.session_factory.begin() as session:
                if before_persist is not None:
                    before_persist(session)
                if self._settle_existing_identity(session, record):
                    return
                try:
                    self._snapshots.save_snapshot_in_session(session, record.snapshot)
                except (SnapshotObjectIntegrityError, SnapshotObjectMissingError, ValueError):
                    raise ValueError("DECISION_EXPORT_INTEGRITY") from None
                session.add(
                    DecisionExportRecordRow(
                        decision_id=record.decision_id,
                        snapshot_id=record.snapshot.snapshot_id,
                        schema_version=_SCHEMA_VERSION,
                        payload_json=payload_json,
                        content_hash=content_hash(payload_json),
                        created_at=created_at,
                        write_order=next_write_order(session),
                    )
                )
        except IntegrityError:
            raise ValueError("DECISION_EXPORT_IMMUTABLE") from None

    def _settle_existing_identity(
        self,
        session: Session,
        record: DecisionExportRecord,
    ) -> bool:
        """Reject legacy IDs and settle an idempotent active save before snapshot writes."""

        if session.get(LegacyDecisionExportRecord, record.decision_id) is not None:
            raise ValueError("DECISION_EXPORT_ID_CONFLICT")
        existing = session.get(DecisionExportRecordRow, record.decision_id)
        if existing is None:
            return False
        if self._decode(existing) != record:
            raise ValueError("DECISION_EXPORT_ID_CONFLICT")
        return True

    def get(self, decision_id: str) -> DecisionExportRecord | None:
        checked = safe_identifier(decision_id, label="decision id")
        with self._database.session_factory() as session:
            row = session.get(DecisionExportRecordRow, checked)
            return None if row is None else self._decode(row)

    def latest(self) -> DecisionExportRecord | None:
        with self._database.session_factory() as session:
            row = session.scalars(
                select(DecisionExportRecordRow)
                .order_by(
                    DecisionExportRecordRow.write_order.desc(),
                )
                .limit(1)
            ).first()
            return None if row is None else self._decode(row)

    def history(self) -> tuple[DecisionExportRecord, ...]:
        with self._database.session_factory() as session:
            rows = session.scalars(
                select(DecisionExportRecordRow).order_by(
                    DecisionExportRecordRow.write_order.desc(),
                )
            ).all()
            return tuple(self._decode(row) for row in rows)

    def readable_history(
        self,
        protected_decision_ids: frozenset[str] = frozenset(),
    ) -> tuple[tuple[DecisionExportRecord, ...], int]:
        """Return intact records while quarantining broken external dependencies."""

        protected = self._protected_ids(protected_decision_ids)
        records: list[DecisionExportRecord] = []
        invalid_count = 0
        with self._database.session_factory() as session:
            rows = session.scalars(
                select(DecisionExportRecordRow).order_by(
                    DecisionExportRecordRow.write_order.desc(),
                )
            ).all()
            for row in rows:
                try:
                    records.append(self._decode(row))
                except ValueError as error:
                    if str(error) != "DECISION_EXPORT_INTEGRITY":
                        raise
                    if row.decision_id not in protected:
                        invalid_count += 1
        return tuple(records), invalid_count

    def delete(self, decision_id: str) -> bool:
        checked = safe_identifier(decision_id, label="decision id")
        with self._database.session_factory.begin() as session:
            row = session.get(DecisionExportRecordRow, checked)
            legacy = session.get(LegacyDecisionExportRecord, checked)
            if row is None and legacy is None:
                return False
            snapshot_id = None if row is None else row.snapshot_id
            if row is not None:
                session.delete(row)
            if legacy is not None:
                session.delete(legacy)
            if snapshot_id is not None:
                self._delete_unreferenced_snapshot(session, snapshot_id, checked)
            return True

    def clear_invalid(
        self,
        protected_decision_ids: frozenset[str] = frozenset(),
    ) -> tuple[str, ...]:
        protected = self._protected_ids(protected_decision_ids)
        invalid_ids: list[str] = []
        with self._database.session_factory.begin() as session:
            rows = session.scalars(select(DecisionExportRecordRow)).all()
            for row in rows:
                try:
                    self._decode(row)
                except ValueError as error:
                    if str(error) != "DECISION_EXPORT_INTEGRITY":
                        raise
                    if row.decision_id in protected:
                        continue
                    invalid_ids.append(row.decision_id)
                    snapshot_id = row.snapshot_id
                    session.delete(row)
                    if snapshot_id is not None:
                        self._delete_unreferenced_snapshot(
                            session,
                            snapshot_id,
                            row.decision_id,
                        )
        return tuple(sorted(invalid_ids))

    @staticmethod
    def _protected_ids(values: frozenset[str]) -> frozenset[str]:
        if type(values) is not frozenset:
            raise TypeError("protected decision ids must be a frozenset")
        return frozenset(
            safe_identifier(item, label="protected decision id") for item in values
        )

    @staticmethod
    def _delete_unreferenced_snapshot(
        session: Session,
        snapshot_id: str,
        deleting_decision_id: str,
    ) -> None:
        other_decision = session.scalar(
            select(DecisionExportRecordRow.decision_id).where(
                DecisionExportRecordRow.snapshot_id == snapshot_id,
                DecisionExportRecordRow.decision_id != deleting_decision_id,
            )
        )
        report = session.scalar(
            select(BacktestReportRecord.run_id).where(
                BacktestReportRecord.snapshot_id == snapshot_id
            )
        )
        if other_decision is None and report is None:
            session.execute(
                delete(RunSnapshotRecord).where(
                    RunSnapshotRecord.snapshot_id == snapshot_id
                )
            )

    def referenced_account_snapshot_ids(self) -> frozenset[int]:
        """Read only the immutable account link, without requiring live market files."""

        referenced: set[int] = set()
        with self._database.session_factory() as session:
            current = session.scalars(select(DecisionExportRecordRow)).all()
            legacy = session.scalars(select(LegacyDecisionExportRecord)).all()
            for current_row in current:
                try:
                    payload = decode_canonical_json(
                        current_row.payload_json,
                        current_row.content_hash,
                    )
                    result = decode_decision_result(payload["result"])
                except (KeyError, TypeError, ValueError):
                    raise ValueError("DECISION_EXPORT_INTEGRITY") from None
                referenced.add(result.account_snapshot_row_id)
            for legacy_row in legacy:
                try:
                    payload = decode_canonical_json(
                        legacy_row.payload_json,
                        legacy_row.content_hash,
                    )
                    result = decode_decision_result(payload["result"])
                except (KeyError, TypeError, ValueError):
                    raise ValueError("DECISION_EXPORT_INTEGRITY") from None
                referenced.add(result.account_snapshot_row_id)
        return frozenset(referenced)

    def legacy_manifests(
        self,
        row: LegacyDecisionExportRecord,
    ) -> tuple[DecisionManifestProvenance, ...]:
        """Validate a quarantined v1 payload without treating it as reproducible."""

        result, manifests = self._decode_legacy(row)
        self._verify_account_link(result)
        self._verify_manifest_provenance(manifests)
        return manifests

    @staticmethod
    def _decode_legacy(
        row: LegacyDecisionExportRecord,
    ) -> tuple[DecisionResult, tuple[DecisionManifestProvenance, ...]]:
        try:
            payload = decode_canonical_json(row.payload_json, row.content_hash)
            if (
                type(row.schema_version) is not int
                or row.schema_version != 1
                or type(payload.get("schema_version")) is not int
                or payload["schema_version"] != 1
                or set(payload)
                != {
                    "decision_id",
                    "manifests",
                    "result",
                    "schema_version",
                    "strategies",
                }
                or payload["decision_id"] != row.decision_id
                or type(payload["manifests"]) is not list
                or type(payload["strategies"]) is not list
            ):
                raise ValueError
            safe_identifier(row.decision_id, label="legacy decision id")
            manifests = []
            for raw in cast(list[object], payload["manifests"]):
                if not isinstance(raw, Mapping):
                    raise ValueError
                item = cast(Mapping[str, object], raw)
                if set(item) != {
                    "content_hash",
                    "manifest_id",
                    "provider",
                    "relative_data_path",
                }:
                    raise ValueError
                manifests.append(
                    DecisionManifestProvenance(
                        manifest_id=cast(str, item["manifest_id"]),
                        provider=cast(str, item["provider"]),
                        content_hash=cast(str, item["content_hash"]),
                        relative_data_path=cast(str, item["relative_data_path"]),
                    )
                )
            strategies = []
            for raw in cast(list[object], payload["strategies"]):
                if not isinstance(raw, Mapping):
                    raise ValueError
                item = cast(Mapping[str, object], raw)
                parameters = item["parameters"]
                if not isinstance(parameters, Mapping) or set(item) != {
                    "parameters",
                    "strategy_instance_id",
                    "strategy_type",
                    "strategy_version",
                }:
                    raise ValueError
                strategies.append(
                    DecisionStrategyProvenance(
                        strategy_instance_id=cast(str, item["strategy_instance_id"]),
                        strategy_type=cast(str, item["strategy_type"]),
                        strategy_version=cast(str, item["strategy_version"]),
                        parameters=cast(Mapping[str, object], parameters),
                    )
                )
            manifest_ids = tuple(item.manifest_id for item in manifests)
            if not manifests or manifest_ids != tuple(sorted(set(manifest_ids))):
                raise ValueError
            strategy_ids = tuple(item.strategy_instance_id for item in strategies)
            result = decode_decision_result(payload["result"])
            if (
                not strategies
                or strategy_ids != tuple(sorted(set(strategy_ids)))
                or strategy_ids
                != tuple(item.strategy_id for item in result.strategy_decisions)
            ):
                raise ValueError
            return result, tuple(manifests)
        except (KeyError, TypeError, ValueError):
            raise ValueError("LEGACY_DECISION_EXPORT_INTEGRITY") from None

    @staticmethod
    def _payload(record: DecisionExportRecord) -> dict[str, object]:
        return {
            "decision_id": record.decision_id,
            "manifests": [
                {
                    "content_hash": item.content_hash,
                    "manifest_id": item.manifest_id,
                    "provider": item.provider,
                    "relative_data_path": item.relative_data_path,
                }
                for item in record.market_manifests
            ],
            "result": encode_decision_result(record.result),
            "schema_version": _SCHEMA_VERSION,
            "snapshot_id": record.snapshot.snapshot_id,
            "strategies": [
                {
                    "parameters": json_value(item.parameters),
                    "strategy_instance_id": item.strategy_instance_id,
                    "strategy_type": item.strategy_type,
                    "strategy_version": item.strategy_version,
                }
                for item in record.strategies
            ],
        }

    def _decode(self, row: DecisionExportRecordRow) -> DecisionExportRecord:
        try:
            payload = decode_canonical_json(row.payload_json, row.content_hash)
            if (
                type(row.schema_version) is not int
                or row.schema_version != _SCHEMA_VERSION
                or type(payload.get("schema_version")) is not int
                or payload["schema_version"] != _SCHEMA_VERSION
                or set(payload)
                != {
                    "decision_id",
                    "manifests",
                    "result",
                    "schema_version",
                    "snapshot_id",
                    "strategies",
                }
                or payload["decision_id"] != row.decision_id
                or type(row.snapshot_id) is not str
                or type(payload["snapshot_id"]) is not str
                or payload["snapshot_id"] != row.snapshot_id
                or type(payload["manifests"]) is not list
                or type(payload["strategies"]) is not list
            ):
                raise ValueError
            manifests = []
            for raw in cast(list[object], payload["manifests"]):
                item = cast(Mapping[str, object], raw)
                if not isinstance(raw, Mapping) or set(item) != {
                    "content_hash",
                    "manifest_id",
                    "provider",
                    "relative_data_path",
                }:
                    raise ValueError
                manifests.append(
                    DecisionManifestProvenance(
                        manifest_id=cast(str, item["manifest_id"]),
                        provider=cast(str, item["provider"]),
                        content_hash=cast(str, item["content_hash"]),
                        relative_data_path=cast(str, item["relative_data_path"]),
                    )
                )
            strategies = []
            for raw in cast(list[object], payload["strategies"]):
                if not isinstance(raw, Mapping):
                    raise ValueError
                item = cast(Mapping[str, object], raw)
                parameters = item["parameters"]
                if not isinstance(parameters, Mapping) or set(item) != {
                    "parameters",
                    "strategy_instance_id",
                    "strategy_type",
                    "strategy_version",
                }:
                    raise ValueError
                strategies.append(
                    DecisionStrategyProvenance(
                        strategy_instance_id=cast(
                            str,
                            item["strategy_instance_id"],
                        ),
                        strategy_type=cast(str, item["strategy_type"]),
                        strategy_version=cast(str, item["strategy_version"]),
                        parameters=cast(Mapping[str, object], parameters),
                    )
                )
            record = DecisionExportRecord(
                decision_id=row.decision_id,
                result=decode_decision_result(payload["result"]),
                market_manifests=tuple(manifests),
                strategies=tuple(strategies),
                snapshot=self._snapshots.load_snapshot(row.snapshot_id),
            )
            record.verify_integrity()
            self._verify_account_link(record.result)
            self._verify_manifests(record)
            return record
        except (
            KeyError,
            SnapshotObjectIntegrityError,
            SnapshotObjectMissingError,
            TypeError,
            ValueError,
        ):
            raise ValueError("DECISION_EXPORT_INTEGRITY") from None

    def _verify_manifests(self, record: DecisionExportRecord) -> None:
        self._verify_manifest_provenance(record.market_manifests)

    def _verify_manifest_provenance(
        self,
        manifests: Sequence[DecisionManifestProvenance],
    ) -> None:
        for expected in manifests:
            manifest = self._market_store.load_manifest(expected.manifest_id)
            if (
                manifest.provider != expected.provider
                or manifest.content_hash != expected.content_hash
                or manifest.relative_data_path != expected.relative_data_path
            ):
                raise ValueError("DECISION_EXPORT_INTEGRITY")

    def _verify_account_link(self, result: DecisionResult) -> None:
        try:
            stored = AccountRepository(
                self._database,
                result.account_id,
                self._clock,
            ).get(result.account_snapshot_row_id)
            # The decision revalues persisted positions at its own market date,
            # so its equity can legitimately differ from the account snapshot.
            if (
                stored is None
                or stored.content_hash != result.account_snapshot_hash
            ):
                raise ValueError
        except (AccountIntegrityError, TypeError, ValueError):
            raise ValueError("DECISION_EXPORT_INTEGRITY") from None
