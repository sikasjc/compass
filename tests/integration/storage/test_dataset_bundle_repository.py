from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from compass.domain.market import InstrumentId
from compass.storage.database import Database
from compass.storage.dataset_bundle_repository import DatasetBundleRepository
from compass.storage.market_store import DatasetManifest, MarketStore
from compass.storage.models import DatasetBundleRecord
from compass.storage.canonical_json import canonical_json, content_hash, decode_canonical_json


SHANGHAI = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 7, 29, 15, 0, tzinfo=SHANGHAI)


def _bars(offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [10.0 + offset, 10.1 + offset],
            "high": [10.2 + offset, 10.3 + offset],
            "low": [9.9 + offset, 10.0 + offset],
            "close": [10.1 + offset, 10.2 + offset],
            "volume": [1000.0, 1001.0],
            "amount": [10100.0, 10210.0],
        },
        index=pd.DatetimeIndex(["2026-07-28", "2026-07-29"], name="date"),
    )


def _create_actual_local_artifact_v1_database(
    database_path: Path,
    *,
    payload_json: str,
) -> None:
    """Create the complete 25f6494 physical profile, not a one-table hybrid."""

    database = Database.sqlite_at(database_path)
    database.create_schema()
    database.engine.dispose()
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            DROP TABLE decision_exports;
            DROP TABLE backtest_reports;
            DROP TABLE dataset_bundles;
            DROP TABLE legacy_decision_exports;
            DROP TABLE local_insertion_sequences;
            CREATE TABLE dataset_bundles (
                bundle_id VARCHAR(128) NOT NULL,
                primary_manifest_id VARCHAR(32) NOT NULL,
                schema_version INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                content_hash VARCHAR(64) NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (bundle_id)
            );
            CREATE UNIQUE INDEX ix_dataset_bundles_primary_manifest_id
                ON dataset_bundles (primary_manifest_id);
            CREATE TABLE backtest_reports (
                run_id VARCHAR(128) NOT NULL,
                snapshot_id VARCHAR(64) NOT NULL,
                schema_version INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                content_hash VARCHAR(64) NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (run_id)
            );
            CREATE UNIQUE INDEX ix_backtest_reports_snapshot_id
                ON backtest_reports (snapshot_id);
            CREATE TABLE decision_exports (
                decision_id VARCHAR(128) NOT NULL,
                schema_version INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                content_hash VARCHAR(64) NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (decision_id)
            );
            """
        )
        connection.execute(
            """
            INSERT INTO dataset_bundles (
                bundle_id, primary_manifest_id, schema_version, payload_json,
                content_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "z-old",
                "b" * 32,
                1,
                payload_json,
                content_hash(payload_json),
                NOW.isoformat(),
            ),
        )
        connection.commit()


def test_latest_bundle_uses_persisted_insert_order_after_rebuild(tmp_path: Path) -> None:
    database_path = tmp_path / "state" / "compass.db"
    database = Database.sqlite_at(database_path)
    database.create_schema()
    store = MarketStore(tmp_path / "state" / "market", database)
    ids = iter(("z-old", "a-new"))
    repository = DatasetBundleRepository(
        database,
        store,
        clock=lambda: NOW,
        id_factory=lambda _: next(ids),
    )
    instrument = InstrumentId.parse("SSE.510300")
    first_manifest = store.write_daily(str(instrument), _bars(), provider="fixture")
    first = repository.save(((instrument, first_manifest),), mode="strict", issue_codes=())
    second_manifest = store.write_daily(str(instrument), _bars(), provider="fixture")
    second = repository.save(((instrument, second_manifest),), mode="strict", issue_codes=())

    assert first.bundle_id == "z-old"
    assert second.bundle_id == "a-new"
    assert repository.latest() == second

    database.engine.dispose()
    rebuilt_database = Database.sqlite_at(database_path)
    rebuilt_database.create_schema()
    rebuilt = DatasetBundleRepository(
        rebuilt_database,
        MarketStore(tmp_path / "state" / "market", rebuilt_database),
        clock=lambda: NOW,
        id_factory=lambda _: "unused",
    )
    assert rebuilt.latest() == second


def test_dataset_bundle_payload_binds_its_database_primary_key(tmp_path: Path) -> None:
    database = Database.sqlite_at(tmp_path / "state" / "compass.db")
    database.create_schema()
    store = MarketStore(tmp_path / "state" / "market", database)
    repository = DatasetBundleRepository(
        database,
        store,
        clock=lambda: NOW,
        id_factory=lambda _: "bundle-original",
    )
    instrument = InstrumentId.parse("SSE.510300")
    manifest = store.write_daily(str(instrument), _bars(), provider="fixture")
    bundle = repository.save(((instrument, manifest),), mode="strict", issue_codes=())

    with database.session_factory.begin() as session:
        row = session.get(DatasetBundleRecord, bundle.bundle_id)
        assert row is not None
        row.bundle_id = "bundle-tampered"

    with pytest.raises(ValueError, match="DATASET_BUNDLE_INTEGRITY"):
        repository.latest()


def test_dataset_bundle_rejects_an_unrecognized_nested_manifest_field(tmp_path: Path) -> None:
    """A validated bundle must not silently ignore future/hostile manifest fields."""

    database = Database.sqlite_at(tmp_path / "state" / "compass.db")
    database.create_schema()
    store = MarketStore(tmp_path / "state" / "market", database)
    repository = DatasetBundleRepository(
        database,
        store,
        clock=lambda: NOW,
        id_factory=lambda _: "bundle-nested-shape",
    )
    instrument = InstrumentId.parse("SSE.510300")
    manifest = store.write_daily(str(instrument), _bars(), provider="fixture")
    bundle = repository.save(((instrument, manifest),), mode="strict", issue_codes=())

    with database.session_factory.begin() as session:
        row = session.get(DatasetBundleRecord, bundle.bundle_id)
        assert row is not None
        payload = decode_canonical_json(row.payload_json, row.content_hash)
        manifests = payload["manifests"]
        assert type(manifests) is list and type(manifests[0]) is dict
        manifests[0]["unrecognized"] = "must-not-be-ignored"
        row.payload_json = canonical_json(payload)
        row.content_hash = content_hash(row.payload_json)

    with pytest.raises(ValueError, match="DATASET_BUNDLE_INTEGRITY"):
        repository.latest()


def test_bundle_save_reloads_exact_manifests_and_objects_before_any_insert(tmp_path: Path) -> None:
    database = Database.sqlite_at(tmp_path / "state" / "compass.db")
    database.create_schema()
    store = MarketStore(tmp_path / "state" / "market", database)
    identifier = iter(("bundle-1", "bundle-2", "bundle-3", "bundle-valid"))
    repository = DatasetBundleRepository(
        database,
        store,
        clock=lambda: NOW,
        id_factory=lambda _: next(identifier),
    )
    first = InstrumentId.parse("SSE.510300")
    second = InstrumentId.parse("SZSE.159915")
    first_manifest = store.write_daily(str(first), _bars(), provider="fixture")
    second_manifest = store.write_daily(str(second), _bars(), provider="fixture")

    with pytest.raises(ValueError, match="DATASET_BUNDLE_INTEGRITY"):
        repository.save(
            ((first, second_manifest), (second, first_manifest)),
            mode="strict",
            issue_codes=(),
        )
    assert repository.latest() is None

    with pytest.raises(ValueError, match="DATASET_BUNDLE_INTEGRITY"):
        repository.save(
            ((first, replace(first_manifest, manifest_id="f" * 32)),),
            mode="strict",
            issue_codes=(),
        )
    assert repository.latest() is None

    with pytest.raises(ValueError, match="DATASET_BUNDLE_INTEGRITY"):
        repository.save(
            ((first, replace(first_manifest, content_hash="e" * 64)),),
            mode="strict",
            issue_codes=(),
        )
    assert repository.latest() is None

    (store.data_dir / f"{first_manifest.content_hash}.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="DATASET_BUNDLE_INTEGRITY"):
        repository.save(((first, first_manifest),), mode="strict", issue_codes=())
    assert repository.latest() is None

    valid_first = store.write_daily(str(first), _bars(1.0), "fixture")
    valid_second = store.write_daily(str(second), _bars(2.0), "fixture")
    bundle = repository.save(
        ((first, valid_first), (second, valid_second)),
        mode="strict",
        issue_codes=(),
    )
    assert bundle.instruments == (first, second)


def test_bundle_save_rejects_metadata_swap_between_validation_and_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = Database.sqlite_at(tmp_path / "state" / "compass.db")
    database.create_schema()
    store = MarketStore(tmp_path / "state" / "market", database)
    repository = DatasetBundleRepository(
        database,
        store,
        clock=lambda: NOW,
        id_factory=lambda _: "bundle-race",
    )
    instrument = InstrumentId.parse("SSE.510300")
    first = store.write_daily(str(instrument), _bars(), provider="fixture")
    replacement_source = store.write_daily(
        str(instrument),
        _bars(1.0),
        provider="fixture",
    )
    replacement = DatasetManifest(
        manifest_id=first.manifest_id,
        instrument=replacement_source.instrument,
        provider=replacement_source.provider,
        content_hash=replacement_source.content_hash,
        relative_data_path=replacement_source.relative_data_path,
        rows=replacement_source.rows,
        created_at=replacement_source.created_at,
        quality_report_json=replacement_source.quality_report_json,
    )
    calls = 0

    def swapped_manifest(manifest_id: str) -> DatasetManifest:
        nonlocal calls
        assert manifest_id == first.manifest_id
        calls += 1
        return first if calls == 1 else replacement

    monkeypatch.setattr(store, "_load_manifest", swapped_manifest)

    with pytest.raises(ValueError, match="DATASET_BUNDLE_INTEGRITY"):
        repository.save(
            ((instrument, replacement),),
            mode="strict",
            issue_codes=(),
        )

    assert calls == 1
    assert repository.latest() is None


def test_schema_migration_backfills_old_bundle_write_order_and_identity(tmp_path: Path) -> None:
    database_path = tmp_path / "state" / "compass.db"
    database_path.parent.mkdir()
    legacy_payload = canonical_json(
        {
            "created_at": NOW.isoformat(),
            "data_quality": {"accepted": True, "issue_codes": [], "mode": "strict"},
            "instruments": ["SSE.510300"],
            "manifests": [{"content_hash": "a" * 64, "manifest_id": "b" * 32}],
            "primary_manifest_id": "b" * 32,
            "schema_version": 1,
        }
    )
    _create_actual_local_artifact_v1_database(
        database_path,
        payload_json=legacy_payload,
    )

    database = Database.sqlite_at(database_path)
    database.create_schema()
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT schema_version, payload_json, content_hash, write_order FROM dataset_bundles"
        ).fetchone()
    assert row is not None
    schema_version, payload_json, stored_hash, write_order = row
    assert schema_version == 2
    assert type(write_order) is int and write_order > 0
    decoded = decode_canonical_json(payload_json, stored_hash)
    assert decoded["bundle_id"] == "z-old"
    assert decoded["schema_version"] == 2


@pytest.mark.parametrize("payload_schema", (True, 1.0))
def test_schema_migration_rejects_non_integer_v1_bundle_schema(
    tmp_path: Path,
    payload_schema: object,
) -> None:
    database_path = tmp_path / "state" / "compass.db"
    database_path.parent.mkdir()
    legacy_payload = canonical_json(
        {
            "created_at": NOW.isoformat(),
            "data_quality": {"accepted": True, "issue_codes": [], "mode": "strict"},
            "instruments": ["SSE.510300"],
            "manifests": [{"content_hash": "a" * 64, "manifest_id": "b" * 32}],
            "primary_manifest_id": "b" * 32,
            "schema_version": payload_schema,
        }
    )
    _create_actual_local_artifact_v1_database(
        database_path,
        payload_json=legacy_payload,
    )
    before = database_path.read_bytes()

    with pytest.raises(ValueError, match="dataset bundle migration payload is invalid"):
        Database.sqlite_at(database_path).create_schema()

    assert database_path.read_bytes() == before
