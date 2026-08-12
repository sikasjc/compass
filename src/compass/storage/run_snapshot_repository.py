from __future__ import annotations

from collections.abc import Callable
from contextlib import closing
from datetime import datetime
import sqlite3
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd  # type: ignore[import-untyped]
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from compass.backtest.snapshot import (
    ManifestReference,
    RunSnapshot,
    SnapshotObjectIntegrityError,
    SnapshotObjectMissingError,
)
from compass.storage.database import Database
from compass.storage.market_store import ManifestIntegrityError, MarketStore
from compass.storage.models import RunSnapshotRecord


SHANGHAI = ZoneInfo("Asia/Shanghai")


def _now() -> datetime:
    return datetime.now(SHANGHAI)


class RunSnapshotRepository:
    """Concrete immutable SQLite/MarketStore adapter for the snapshot protocol."""

    def __init__(
        self,
        database: Database,
        market_store: MarketStore,
        *,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        if type(database) is not Database:
            raise TypeError("database must be an exact Database")
        if type(market_store) is not MarketStore:
            raise TypeError("market_store must be an exact MarketStore")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._database = database
        self._market_store = market_store
        self._clock = clock

    def save_snapshot(self, snapshot: RunSnapshot) -> None:
        if type(snapshot) is not RunSnapshot:
            raise TypeError("snapshot must be an exact RunSnapshot")
        snapshot.verify_integrity()
        self.assert_run_id_available(snapshot)
        try:
            with self._database.session_factory.begin() as session:
                self.save_snapshot_in_session(session, snapshot)
        except IntegrityError:
            raise SnapshotObjectIntegrityError("run snapshot is immutable") from None

    def save_snapshot_in_session(self, session: Session, snapshot: RunSnapshot) -> None:
        """Persist one immutable snapshot in an owning transaction."""

        if type(snapshot) is not RunSnapshot:
            raise TypeError("snapshot must be an exact RunSnapshot")
        snapshot.verify_integrity()
        for expected in snapshot.market_manifests:
            actual = self.load_manifest_ref(expected.manifest_id)
            if actual != expected:
                raise SnapshotObjectIntegrityError("manifest reference changed")
        created_at = self._clock()
        if (
            type(created_at) is not datetime
            or created_at.tzinfo is None
            or created_at.utcoffset() is None
        ):
            raise ValueError("snapshot clock must return a timezone-aware datetime")
        row = RunSnapshotRecord(
            snapshot_id=snapshot.snapshot_id,
            run_id=snapshot.run_id,
            schema_version=snapshot.schema_version,
            payload_json=snapshot.canonical_payload_json,
            content_hash=snapshot.content_hash,
            created_at=created_at,
        )
        self._assert_run_id_available(session, snapshot)
        existing = session.get(RunSnapshotRecord, snapshot.snapshot_id)
        if existing is not None:
            checked = self._decode_row(existing)
            if checked.canonical_payload_json != snapshot.canonical_payload_json:
                raise SnapshotObjectIntegrityError("snapshot identity collision")
            return
        session.add(row)

    def assert_run_id_available(self, snapshot: RunSnapshot) -> None:
        """Reject a different immutable snapshot that already owns ``snapshot.run_id``."""

        if type(snapshot) is not RunSnapshot:
            raise TypeError("snapshot must be an exact RunSnapshot")
        snapshot.verify_integrity()
        with self._database.session_factory() as session:
            self._assert_run_id_available(session, snapshot)

    def load_snapshot(self, snapshot_id: str) -> RunSnapshot:
        with self._database.session_factory() as session:
            row = session.get(RunSnapshotRecord, snapshot_id)
            if row is None:
                raise SnapshotObjectMissingError("snapshot is unavailable")
            return self._decode_row(row)

    def list_snapshots(self) -> tuple[RunSnapshot, ...]:
        with self._database.session_factory() as session:
            rows = session.scalars(
                select(RunSnapshotRecord).order_by(
                    RunSnapshotRecord.run_id,
                    RunSnapshotRecord.snapshot_id,
                )
            ).all()
            return tuple(self._decode_row(row) for row in rows)

    def referenced_manifest_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    reference.manifest_id
                    for snapshot in self.list_snapshots()
                    for reference in snapshot.market_manifests
                }
            )
        )

    def load_manifest_ref(self, manifest_id: str) -> ManifestReference:
        try:
            if not self._market_store.manifest_path(manifest_id).is_file():
                raise SnapshotObjectMissingError("manifest is unavailable")
            manifest = self._market_store.load_manifest(manifest_id)
            return ManifestReference(manifest.manifest_id, manifest.content_hash)
        except SnapshotObjectMissingError:
            raise
        except (ManifestIntegrityError, OSError, TypeError, ValueError):
            raise SnapshotObjectIntegrityError("manifest failed validation") from None

    def read_manifest(self, manifest_id: str) -> pd.DataFrame:
        try:
            if not self._market_store.manifest_path(manifest_id).is_file():
                raise SnapshotObjectMissingError("manifest is unavailable")
            return self._market_store.read_manifest(manifest_id)
        except SnapshotObjectMissingError:
            raise
        except (ManifestIntegrityError, OSError, TypeError, ValueError):
            raise SnapshotObjectIntegrityError("manifest failed validation") from None

    @staticmethod
    def snapshots_from_database(database_path: Path) -> tuple[RunSnapshot, ...]:
        """Read persisted snapshots from a validated, read-only backup candidate."""

        try:
            uri = f"{database_path.resolve().as_uri()}?mode=ro"
            with closing(sqlite3.connect(uri, uri=True)) as connection:
                rows = connection.execute(
                    """
                    SELECT snapshot_id, run_id, schema_version, payload_json, content_hash
                    FROM run_snapshots
                    ORDER BY run_id, snapshot_id
                    """
                ).fetchall()
        except (OSError, sqlite3.Error):
            raise SnapshotObjectIntegrityError("backup snapshot metadata is unavailable") from None
        snapshots: list[RunSnapshot] = []
        for snapshot_id, run_id, schema_version, payload_json, content_hash in rows:
            try:
                snapshot = RunSnapshot.from_canonical_payload_json(
                    payload_json,
                    snapshot_id=snapshot_id,
                )
                if (
                    snapshot.run_id != run_id
                    or snapshot.schema_version != schema_version
                    or snapshot.content_hash != content_hash
                ):
                    raise ValueError("snapshot summary mismatch")
            except (TypeError, ValueError):
                raise SnapshotObjectIntegrityError(
                    "backup snapshot metadata failed validation"
                ) from None
            snapshots.append(snapshot)
        return tuple(snapshots)

    @staticmethod
    def _decode_row(row: RunSnapshotRecord) -> RunSnapshot:
        try:
            snapshot = RunSnapshot.from_canonical_payload_json(
                row.payload_json,
                snapshot_id=row.snapshot_id,
            )
            if (
                snapshot.run_id != row.run_id
                or snapshot.schema_version != row.schema_version
                or snapshot.content_hash != row.content_hash
            ):
                raise ValueError("snapshot summary mismatch")
            return snapshot
        except (TypeError, ValueError):
            raise SnapshotObjectIntegrityError("snapshot failed validation") from None

    @classmethod
    def _assert_run_id_available(cls, session: Session, snapshot: RunSnapshot) -> None:
        rows = session.scalars(
            select(RunSnapshotRecord).where(RunSnapshotRecord.run_id == snapshot.run_id)
        ).all()
        for row in rows:
            existing = cls._decode_row(row)
            if existing.snapshot_id != snapshot.snapshot_id:
                raise SnapshotObjectIntegrityError("snapshot run id collision")
