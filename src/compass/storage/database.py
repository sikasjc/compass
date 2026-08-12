from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import sqlite3
from collections.abc import Callable
from typing import cast

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from compass.config import Settings
from compass.storage.canonical_json import canonical_json, content_hash, decode_canonical_json
from compass.storage.models import Base


_LOCAL_ARTIFACT_V1_REBUILT_TABLES = frozenset(
    {
        "backtest_reports",
        "dataset_bundles",
        "decision_exports",
    }
)
_LOCAL_ARTIFACT_V1_MISSING_TABLES = frozenset(
    {
        "legacy_decision_exports",
        "local_insertion_sequences",
    }
)
_PRECANONICAL_CURRENT_REBUILT_TABLES = _LOCAL_ARTIFACT_V1_REBUILT_TABLES
_WRITE_ORDER_TABLES = (
    "dataset_bundles",
    "backtest_reports",
    "decision_exports",
)
_LOCAL_ARTIFACT_V1_DDL = (
    """
    CREATE TABLE dataset_bundles (
        bundle_id VARCHAR(128) NOT NULL,
        primary_manifest_id VARCHAR(32) NOT NULL,
        schema_version INTEGER NOT NULL,
        payload_json TEXT NOT NULL,
        content_hash VARCHAR(64) NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (bundle_id)
    )
    """,
    """
    CREATE TABLE backtest_reports (
        run_id VARCHAR(128) NOT NULL,
        snapshot_id VARCHAR(64) NOT NULL,
        schema_version INTEGER NOT NULL,
        payload_json TEXT NOT NULL,
        content_hash VARCHAR(64) NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (run_id)
    )
    """,
    """
    CREATE TABLE decision_exports (
        decision_id VARCHAR(128) NOT NULL,
        schema_version INTEGER NOT NULL,
        payload_json TEXT NOT NULL,
        content_hash VARCHAR(64) NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (decision_id)
    )
    """,
)
_PRECANONICAL_CURRENT_DDL = (
    """
    CREATE TABLE dataset_bundles (
        bundle_id VARCHAR(128) NOT NULL,
        primary_manifest_id VARCHAR(32) NOT NULL,
        schema_version INTEGER NOT NULL,
        payload_json TEXT NOT NULL,
        content_hash VARCHAR(64) NOT NULL,
        created_at TEXT NOT NULL,
        write_order INTEGER,
        PRIMARY KEY (bundle_id)
    )
    """,
    """
    CREATE TABLE backtest_reports (
        run_id VARCHAR(128) NOT NULL,
        snapshot_id VARCHAR(64) NOT NULL,
        schema_version INTEGER NOT NULL,
        payload_json TEXT NOT NULL,
        content_hash VARCHAR(64) NOT NULL,
        created_at TEXT NOT NULL,
        write_order INTEGER,
        PRIMARY KEY (run_id)
    )
    """,
    """
    CREATE TABLE decision_exports (
        decision_id VARCHAR(128) NOT NULL,
        schema_version INTEGER NOT NULL,
        payload_json TEXT NOT NULL,
        content_hash VARCHAR(64) NOT NULL,
        created_at TEXT NOT NULL,
        write_order INTEGER,
        snapshot_id TEXT,
        PRIMARY KEY (decision_id)
    )
    """,
)

# These definitions are intentionally copied from the released a1e0125 and
# 25f6494 SQLite layouts.  Legacy admission must not drift if current SQLAlchemy
# models change in the future.
_A1_TABLE_DDL = (
    """
    CREATE TABLE account_snapshots (
        id INTEGER NOT NULL, account_id VARCHAR(128) NOT NULL,
        captured_at TEXT NOT NULL, as_of VARCHAR(10) NOT NULL,
        payload_json TEXT NOT NULL, content_hash VARCHAR(64) NOT NULL,
        cash FLOAT NOT NULL, market_value FLOAT NOT NULL, equity FLOAT NOT NULL,
        PRIMARY KEY (id)
    )
    """,
    """
    CREATE TABLE backtest_runs (
        id INTEGER NOT NULL, strategy_instance_id INTEGER, status VARCHAR(32) NOT NULL,
        started_at TEXT NOT NULL, completed_at TEXT, result_json TEXT, PRIMARY KEY (id)
    )
    """,
    """
    CREATE TABLE dataset_manifests (
        manifest_id VARCHAR(32) NOT NULL, instrument VARCHAR(16) NOT NULL,
        provider VARCHAR(64) NOT NULL, content_hash VARCHAR(64) NOT NULL,
        relative_data_path VARCHAR(512) NOT NULL, rows INTEGER NOT NULL,
        created_at TEXT NOT NULL, quality_report_json TEXT, PRIMARY KEY (manifest_id)
    )
    """,
    """
    CREATE TABLE decision_runs (
        id INTEGER NOT NULL, strategy_instance_id INTEGER, status VARCHAR(32) NOT NULL,
        decided_at TEXT NOT NULL, decision_json TEXT NOT NULL, PRIMARY KEY (id)
    )
    """,
    """
    CREATE TABLE run_snapshots (
        snapshot_id VARCHAR(64) NOT NULL, run_id VARCHAR(128) NOT NULL,
        schema_version INTEGER NOT NULL, payload_json TEXT NOT NULL,
        content_hash VARCHAR(64) NOT NULL, created_at TEXT NOT NULL,
        PRIMARY KEY (snapshot_id)
    )
    """,
    """
    CREATE TABLE strategy_instances (
        id INTEGER NOT NULL, name VARCHAR(128) NOT NULL, status VARCHAR(32) NOT NULL,
        configuration_json TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY (id)
    )
    """,
    """
    CREATE TABLE task_runs (
        id INTEGER NOT NULL, task_name VARCHAR(128) NOT NULL, status VARCHAR(32) NOT NULL,
        started_at TEXT NOT NULL, completed_at TEXT, detail TEXT, PRIMARY KEY (id)
    )
    """,
    """
    CREATE TABLE watchlists (
        id INTEGER NOT NULL, name VARCHAR(128) NOT NULL, instruments_json TEXT NOT NULL,
        created_at TEXT NOT NULL, PRIMARY KEY (id)
    )
    """,
)
_A1_INDEX_DDL = (
    "CREATE INDEX ix_account_snapshots_content_hash ON account_snapshots (content_hash)",
    "CREATE INDEX ix_dataset_manifests_content_hash ON dataset_manifests (content_hash)",
    "CREATE UNIQUE INDEX ix_run_snapshots_run_id ON run_snapshots (run_id)",
)
_LOCAL_ARTIFACT_V1_OWN_TABLE_DDL = (
    """
    CREATE TABLE local_watchlists (
        watchlist_id VARCHAR(128) NOT NULL, schema_version INTEGER NOT NULL,
        payload_json TEXT NOT NULL, content_hash VARCHAR(64) NOT NULL,
        updated_at TEXT NOT NULL, PRIMARY KEY (watchlist_id)
    )
    """,
    """
    CREATE TABLE local_strategy_versions (
        instance_id VARCHAR(128) NOT NULL, lineage_id VARCHAR(128) NOT NULL,
        version INTEGER NOT NULL, enabled INTEGER NOT NULL, schema_version INTEGER NOT NULL,
        payload_json TEXT NOT NULL, content_hash VARCHAR(64) NOT NULL,
        created_at TEXT NOT NULL, PRIMARY KEY (instance_id)
    )
    """,
    """
    CREATE TABLE local_settings (
        settings_id VARCHAR(64) NOT NULL, schema_version INTEGER NOT NULL,
        payload_json TEXT NOT NULL, content_hash VARCHAR(64) NOT NULL,
        updated_at TEXT NOT NULL, PRIMARY KEY (settings_id)
    )
    """,
)
_LOCAL_ARTIFACT_V1_INDEX_DDL = (
    "CREATE INDEX ix_local_strategy_versions_lineage_id "
    "ON local_strategy_versions (lineage_id)",
    "CREATE UNIQUE INDEX ix_dataset_bundles_primary_manifest_id "
    "ON dataset_bundles (primary_manifest_id)",
    "CREATE UNIQUE INDEX ix_backtest_reports_snapshot_id "
    "ON backtest_reports (snapshot_id)",
)


@dataclass(frozen=True)
class _SchemaSignature:
    objects: tuple[tuple[str, str, str, str | None], ...]
    table_xinfo: tuple[tuple[str, tuple[tuple[object, ...], ...]], ...]
    indexes: tuple[
        tuple[
            str,
            tuple[
                tuple[str, int, str, int, tuple[tuple[object, ...], ...]],
                ...,
            ],
        ],
        ...,
    ]
    foreign_keys: tuple[tuple[str, tuple[tuple[object, ...], ...]], ...]


@dataclass(frozen=True)
class _SchemaLayout:
    signature: _SchemaSignature
    table_sql: tuple[tuple[str, str], ...]
    index_sql: tuple[tuple[str, str, str], ...]


def _quote_identifier(value: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError("SQLite identifier is invalid")
    return '"' + value.replace('"', '""') + '"'


def _normalized_sql(value: str | None) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError("SQLite schema SQL is invalid")
    return " ".join(value.split())


def _user_table_names(connection: sqlite3.Connection) -> tuple[str, ...]:
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()
    names: list[str] = []
    for (name,) in rows:
        if type(name) is not str:
            raise ValueError("SQLite table name is invalid")
        names.append(name)
    return tuple(names)


def _validate_internal_schema_objects(connection: sqlite3.Connection) -> None:
    """Allow only SQLite internals that are intrinsic to the known layouts."""

    rows = connection.execute(
        """
        SELECT type, name, tbl_name, sql
        FROM sqlite_master
        WHERE name LIKE 'sqlite_%'
        ORDER BY type, name
        """
    ).fetchall()
    for kind, name, table_name, statement in rows:
        if type(kind) is not str or type(name) is not str or type(table_name) is not str:
            raise ValueError("SQLite internal schema object is invalid")
        normalized = _normalized_sql(statement)
        if (
            kind == "table"
            and name == "sqlite_sequence"
            and table_name == "sqlite_sequence"
            and normalized == "CREATE TABLE sqlite_sequence(name,seq)"
        ):
            continue
        if (
            kind == "index"
            and name.startswith("sqlite_autoindex_")
            and normalized is None
        ):
            # The table-level PRAGMA signature below constrains these implicit
            # PK/UNIQUE indexes to the admitted physical layouts.
            continue
        raise ValueError("SQLite internal schema object is not admitted")


def _schema_layout_from_connection(connection: sqlite3.Connection) -> _SchemaLayout:
    _validate_internal_schema_objects(connection)
    object_rows = connection.execute(
        """
        SELECT type, name, tbl_name, sql
        FROM sqlite_master
        WHERE name NOT LIKE 'sqlite_%'
        ORDER BY type, name
        """
    ).fetchall()
    objects: list[tuple[str, str, str, str | None]] = []
    table_sql: list[tuple[str, str]] = []
    index_sql: list[tuple[str, str, str]] = []
    for kind, name, table_name, statement in object_rows:
        if type(kind) is not str or type(name) is not str or type(table_name) is not str:
            raise ValueError("SQLite schema object is invalid")
        normalized = _normalized_sql(statement)
        objects.append((kind, name, table_name, normalized))
        if kind == "table":
            if normalized is None:
                raise ValueError("SQLite table definition is invalid")
            table_sql.append((name, normalized))
        elif kind == "index" and normalized is not None:
            index_sql.append((name, table_name, normalized))

    table_xinfo: list[tuple[str, tuple[tuple[object, ...], ...]]] = []
    indexes: list[
        tuple[
            str,
            tuple[
                tuple[str, int, str, int, tuple[tuple[object, ...], ...]],
                ...,
            ],
        ]
    ] = []
    foreign_keys: list[tuple[str, tuple[tuple[object, ...], ...]]] = []
    for table_name in _user_table_names(connection):
        quoted = _quote_identifier(table_name)
        xinfo = tuple(
            tuple(row)
            for row in connection.execute(f"PRAGMA table_xinfo({quoted})").fetchall()
        )
        table_xinfo.append((table_name, xinfo))

        table_indexes: list[tuple[str, int, str, int, tuple[tuple[object, ...], ...]]] = []
        for row in connection.execute(f"PRAGMA index_list({quoted})").fetchall():
            if len(row) != 5:
                raise ValueError("SQLite index metadata is invalid")
            _, index_name, unique, origin, partial = row
            if (
                type(index_name) is not str
                or type(unique) is not int
                or type(origin) is not str
                or type(partial) is not int
            ):
                raise ValueError("SQLite index metadata is invalid")
            index_xinfo = tuple(
                tuple(item)
                for item in connection.execute(
                    f"PRAGMA index_xinfo({_quote_identifier(index_name)})"
                ).fetchall()
            )
            table_indexes.append((index_name, unique, origin, partial, index_xinfo))
        indexes.append((table_name, tuple(sorted(table_indexes))))

        foreign_keys.append(
            (
                table_name,
                tuple(
                    tuple(row)
                    for row in connection.execute(
                        f"PRAGMA foreign_key_list({quoted})"
                    ).fetchall()
                ),
            )
        )

    signature = _SchemaSignature(
        objects=tuple(objects),
        table_xinfo=tuple(table_xinfo),
        indexes=tuple(indexes),
        foreign_keys=tuple(foreign_keys),
    )
    return _SchemaLayout(
        signature=signature,
        table_sql=tuple(sorted(table_sql)),
        index_sql=tuple(sorted(index_sql)),
    )


def _create_write_order_indexes(connection: sqlite3.Connection) -> None:
    for table_name in _WRITE_ORDER_TABLES:
        connection.execute(
            "CREATE UNIQUE INDEX "
            f"{_quote_identifier(f'ix_{table_name}_write_order')} "
            f"ON {_quote_identifier(table_name)}(write_order)"
        )
    connection.execute(
        "CREATE UNIQUE INDEX ix_decision_exports_snapshot_id "
        "ON decision_exports(snapshot_id)"
    )


@lru_cache(maxsize=1)
def _canonical_schema_layout() -> _SchemaLayout:
    engine = create_engine("sqlite://")
    try:
        Base.metadata.create_all(engine)
        source_connection = engine.raw_connection()
        try:
            connection = cast(sqlite3.Connection, source_connection.driver_connection)
            _create_write_order_indexes(connection)
            return _schema_layout_from_connection(connection)
        finally:
            source_connection.close()
    finally:
        engine.dispose()


@lru_cache(maxsize=1)
def _a1_signature() -> _SchemaSignature:
    connection = sqlite3.connect(":memory:")
    try:
        for statement in _A1_TABLE_DDL:
            connection.execute(statement)
        for statement in _A1_INDEX_DDL:
            connection.execute(statement)
        return _schema_layout_from_connection(connection).signature
    finally:
        connection.close()


@lru_cache(maxsize=1)
def _local_artifact_v1_signature() -> _SchemaSignature:
    connection = sqlite3.connect(":memory:")
    try:
        for statement in _A1_TABLE_DDL:
            connection.execute(statement)
        for statement in _A1_INDEX_DDL:
            connection.execute(statement)
        for statement in _LOCAL_ARTIFACT_V1_OWN_TABLE_DDL:
            connection.execute(statement)
        for statement in _LOCAL_ARTIFACT_V1_DDL:
            connection.execute(statement)
        for statement in _LOCAL_ARTIFACT_V1_INDEX_DDL:
            connection.execute(statement)
        return _schema_layout_from_connection(connection).signature
    finally:
        connection.close()


@lru_cache(maxsize=1)
def _precanonical_current_signature() -> _SchemaSignature:
    canonical = _canonical_schema_layout()
    connection = sqlite3.connect(":memory:")
    try:
        current_table_names = {name for name, _ in canonical.table_sql}
        preserved = current_table_names - _PRECANONICAL_CURRENT_REBUILT_TABLES
        _create_canonical_tables(connection, canonical, preserved)
        for statement in _PRECANONICAL_CURRENT_DDL:
            connection.execute(statement)
        for index_name, table_name, statement in canonical.index_sql:
            if (
                table_name in _PRECANONICAL_CURRENT_REBUILT_TABLES
                and not index_name.endswith("_write_order")
                and index_name != "ix_decision_exports_snapshot_id"
            ):
                connection.execute(statement)
        for table_name in _WRITE_ORDER_TABLES:
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                f"ix_{table_name}_write_order ON {table_name}(write_order)"
            )
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS "
            "ix_decision_exports_snapshot_id ON decision_exports(snapshot_id)"
        )
        return _schema_layout_from_connection(connection).signature
    finally:
        connection.close()


def _is_empty_schema(signature: _SchemaSignature) -> bool:
    return not (
        signature.objects
        or signature.table_xinfo
        or signature.indexes
        or signature.foreign_keys
    )


def _schema_variant(
    signature: _SchemaSignature,
    *,
    allow_empty: bool,
) -> str | None:
    canonical = _canonical_schema_layout()
    if allow_empty and _is_empty_schema(signature):
        return "empty"
    if signature == canonical.signature:
        return "current"
    if signature == _a1_signature():
        return "a1"
    if signature == _local_artifact_v1_signature():
        return "local-artifact-v1"
    if signature == _precanonical_current_signature():
        return "precanonical-current"
    return None


def _create_canonical_tables(
    connection: sqlite3.Connection,
    layout: _SchemaLayout,
    table_names: set[str],
) -> None:
    table_sql = dict(layout.table_sql)
    for table_name in sorted(table_names):
        statement = table_sql.get(table_name)
        if statement is None:
            raise ValueError("canonical SQLite table is unavailable")
        connection.execute(statement)
    _create_canonical_indexes(connection, layout, table_names)


def _create_canonical_indexes(
    connection: sqlite3.Connection,
    layout: _SchemaLayout,
    table_names: set[str],
) -> None:
    for _, table_name, statement in layout.index_sql:
        if table_name in table_names:
            connection.execute(statement)


def _drop_named_indexes(connection: sqlite3.Connection, table_name: str) -> None:
    quoted = _quote_identifier(table_name)
    rows = connection.execute(f"PRAGMA index_list({quoted})").fetchall()
    for row in rows:
        if len(row) != 5:
            raise ValueError("SQLite index metadata is invalid")
        _, index_name, _, origin, _ = row
        if type(index_name) is not str or type(origin) is not str:
            raise ValueError("SQLite index metadata is invalid")
        if origin == "c":
            connection.execute(f"DROP INDEX {_quote_identifier(index_name)}")


def _next_write_order(connection: sqlite3.Connection) -> int:
    cursor = connection.execute("INSERT INTO local_insertion_sequences DEFAULT VALUES")
    value = cursor.lastrowid
    if type(value) is not int or value <= 0:
        raise ValueError("local artifact write order is invalid")
    return value


def _migrated_dataset_bundle_payload(
    bundle_id: object,
    schema_version: object,
    payload_json: object,
    stored_hash: object,
) -> tuple[int, str, str]:
    expected = {
        "created_at",
        "data_quality",
        "instruments",
        "manifests",
        "primary_manifest_id",
        "schema_version",
    }
    if (
        type(bundle_id) is not str
        or type(schema_version) is not int
        or schema_version != 1
        or type(payload_json) is not str
        or type(stored_hash) is not str
    ):
        raise ValueError("dataset bundle migration metadata is invalid")
    decoded = decode_canonical_json(payload_json, stored_hash)
    if (
        set(decoded) != expected
        or type(decoded["schema_version"]) is not int
        or decoded["schema_version"] != 1
    ):
        raise ValueError("dataset bundle migration payload is invalid")
    decoded["bundle_id"] = bundle_id
    decoded["schema_version"] = 2
    migrated = canonical_json(decoded)
    return 2, migrated, content_hash(migrated)


def _rebuild_dataset_bundles(
    connection: sqlite3.Connection,
    source_table: str,
) -> None:
    quoted_source = _quote_identifier(source_table)
    rows = connection.execute(
        "SELECT bundle_id, primary_manifest_id, schema_version, payload_json, content_hash, "
        f"created_at FROM {quoted_source} ORDER BY rowid"
    ).fetchall()
    for bundle_id, primary_manifest_id, schema_version, payload_json, stored_hash, created_at in rows:
        migrated_schema, migrated_payload, migrated_hash = _migrated_dataset_bundle_payload(
            bundle_id,
            schema_version,
            payload_json,
            stored_hash,
        )
        connection.execute(
            """
            INSERT INTO dataset_bundles (
                bundle_id, primary_manifest_id, schema_version, payload_json, content_hash,
                created_at, write_order
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                bundle_id,
                primary_manifest_id,
                migrated_schema,
                migrated_payload,
                migrated_hash,
                created_at,
                _next_write_order(connection),
            ),
        )


def _rebuild_backtest_reports(
    connection: sqlite3.Connection,
    source_table: str,
) -> None:
    quoted_source = _quote_identifier(source_table)
    rows = connection.execute(
        "SELECT run_id, snapshot_id, schema_version, payload_json, content_hash, created_at "
        f"FROM {quoted_source} ORDER BY rowid"
    ).fetchall()
    for run_id, snapshot_id, schema_version, payload_json, stored_hash, created_at in rows:
        connection.execute(
            """
            INSERT INTO backtest_reports (
                run_id, snapshot_id, schema_version, payload_json, content_hash, created_at,
                write_order
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                snapshot_id,
                schema_version,
                payload_json,
                stored_hash,
                created_at,
                _next_write_order(connection),
            ),
        )


def _rebuild_decision_exports(
    connection: sqlite3.Connection,
    source_table: str,
) -> None:
    quoted_source = _quote_identifier(source_table)
    rows = connection.execute(
        "SELECT decision_id, schema_version, payload_json, content_hash, created_at "
        f"FROM {quoted_source} ORDER BY rowid"
    ).fetchall()
    for decision_id, schema_version, payload_json, stored_hash, created_at in rows:
        if type(schema_version) is not int or schema_version != 1:
            raise ValueError("legacy decision export schema version is invalid")
        existing = connection.execute(
            "SELECT 1 FROM legacy_decision_exports WHERE decision_id = ?",
            (decision_id,),
        ).fetchone()
        if existing is not None:
            raise ValueError("legacy decision export identity conflicts")
        connection.execute(
            """
            INSERT INTO legacy_decision_exports (
                decision_id, schema_version, payload_json, content_hash, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (decision_id, schema_version, payload_json, stored_hash, created_at),
        )


def _copy_precanonical_dataset_bundles(
    connection: sqlite3.Connection,
    source_table: str,
) -> None:
    connection.execute(
        """
        INSERT INTO dataset_bundles (
            bundle_id, primary_manifest_id, schema_version, payload_json, content_hash,
            created_at, write_order
        )
        SELECT bundle_id, primary_manifest_id, schema_version, payload_json, content_hash,
               created_at, write_order
        FROM """
        + _quote_identifier(source_table)
    )


def _copy_precanonical_backtest_reports(
    connection: sqlite3.Connection,
    source_table: str,
) -> None:
    connection.execute(
        """
        INSERT INTO backtest_reports (
            run_id, snapshot_id, schema_version, payload_json, content_hash, created_at,
            write_order
        )
        SELECT run_id, snapshot_id, schema_version, payload_json, content_hash, created_at,
               write_order
        FROM """
        + _quote_identifier(source_table)
    )


def _copy_precanonical_decision_exports(
    connection: sqlite3.Connection,
    source_table: str,
) -> None:
    connection.execute(
        """
        INSERT INTO decision_exports (
            decision_id, snapshot_id, schema_version, payload_json, content_hash, created_at,
            write_order
        )
        SELECT decision_id, snapshot_id, schema_version, payload_json, content_hash, created_at,
               write_order
        FROM """
        + _quote_identifier(source_table)
    )


def _rebuild_v1_table(
    connection: sqlite3.Connection,
    layout: _SchemaLayout,
    table_name: str,
    copy_rows: Callable[[sqlite3.Connection, str], None],
) -> None:
    source_table = f"__migration_source_{table_name}"
    connection.execute(
        f"ALTER TABLE {_quote_identifier(table_name)} "
        f"RENAME TO {_quote_identifier(source_table)}"
    )
    _drop_named_indexes(connection, source_table)
    _create_canonical_tables(connection, layout, {table_name})
    copy_rows(connection, source_table)
    connection.execute(f"DROP TABLE {_quote_identifier(source_table)}")


def _migrate_local_artifact_v1(
    connection: sqlite3.Connection,
    layout: _SchemaLayout,
) -> None:
    _create_canonical_tables(
        connection,
        layout,
        set(_LOCAL_ARTIFACT_V1_MISSING_TABLES),
    )
    _rebuild_v1_table(connection, layout, "dataset_bundles", _rebuild_dataset_bundles)
    _rebuild_v1_table(connection, layout, "backtest_reports", _rebuild_backtest_reports)
    _rebuild_v1_table(connection, layout, "decision_exports", _rebuild_decision_exports)


def _rebuild_precanonical_current(
    connection: sqlite3.Connection,
    layout: _SchemaLayout,
) -> None:
    _rebuild_v1_table(
        connection,
        layout,
        "dataset_bundles",
        _copy_precanonical_dataset_bundles,
    )
    _rebuild_v1_table(
        connection,
        layout,
        "backtest_reports",
        _copy_precanonical_backtest_reports,
    )
    _rebuild_v1_table(
        connection,
        layout,
        "decision_exports",
        _copy_precanonical_decision_exports,
    )


def _quarantine_current_legacy_decision_exports(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """
        SELECT decision_id, schema_version, payload_json, content_hash, created_at
        FROM decision_exports
        WHERE schema_version = 1
        """
    ).fetchall()
    for decision_id, schema_version, payload_json, stored_hash, created_at in rows:
        if type(schema_version) is not int:
            raise ValueError("legacy decision export schema version is invalid")
        existing = connection.execute(
            "SELECT 1 FROM legacy_decision_exports WHERE decision_id = ?",
            (decision_id,),
        ).fetchone()
        if existing is not None:
            raise ValueError("legacy decision export identity conflicts")
        connection.execute(
            """
            INSERT INTO legacy_decision_exports (
                decision_id, schema_version, payload_json, content_hash, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (decision_id, schema_version, payload_json, stored_hash, created_at),
        )
        connection.execute(
            "DELETE FROM decision_exports WHERE decision_id = ?",
            (decision_id,),
        )
    collision = connection.execute(
        """
        SELECT 1
        FROM decision_exports AS active
        INNER JOIN legacy_decision_exports AS legacy
            ON legacy.decision_id = active.decision_id
        LIMIT 1
        """
    ).fetchone()
    if collision is not None:
        raise ValueError("active and legacy decision identities overlap")


def validate_local_insertion_sequence(connection: sqlite3.Connection) -> None:
    """Require an unmodified allocation history and bounded artifact write orders."""

    allocation_rows = connection.execute(
        "SELECT id FROM local_insertion_sequences ORDER BY id"
    ).fetchall()
    allocation_count = len(allocation_rows)
    for expected_id, (value,) in enumerate(allocation_rows, start=1):
        if type(value) is not int or value != expected_id:
            raise ValueError("local artifact write-order sequence is invalid")

    values: list[int] = []
    for table_name in _WRITE_ORDER_TABLES:
        rows = connection.execute(
            f"SELECT write_order FROM {_quote_identifier(table_name)}"
        ).fetchall()
        for (value,) in rows:
            if type(value) is not int or value <= 0:
                raise ValueError("local artifact write order is invalid")
            values.append(value)
    if len(set(values)) != len(values):
        raise ValueError("local artifact write order is not globally unique")
    if any(value > allocation_count for value in values):
        raise ValueError("local artifact write-order sequence is invalid")

    sequence_rows = connection.execute(
        "SELECT name, seq FROM sqlite_sequence ORDER BY name"
    ).fetchall()
    if allocation_count == 0:
        if sequence_rows:
            raise ValueError("local artifact write-order sequence is invalid")
        return
    if (
        len(sequence_rows) != 1
        or type(sequence_rows[0][0]) is not str
        or sequence_rows[0][0] != "local_insertion_sequences"
        or type(sequence_rows[0][1]) is not int
        or sequence_rows[0][1] != allocation_count
    ):
        raise ValueError("local artifact write-order sequence is invalid")


def _seed_a1_local_settings(connection: sqlite3.Connection) -> None:
    """Backfill the new singleton together with the a1 DDL migration."""

    payload_json = canonical_json(
        {
            "active_risk_template": None,
            "fee_confirmed": False,
            "log_level": "INFO",
            "provider_priority": [],
            "schema_version": 1,
        }
    )
    connection.execute(
        """
        INSERT INTO local_settings (
            settings_id, schema_version, payload_json, content_hash, updated_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            "application",
            1,
            payload_json,
            content_hash(payload_json),
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def _migrate_sqlite_schema(path: Path, *, allow_empty: bool) -> None:
    """Migrate only preflighted legacy schemas inside one SQLite transaction."""

    with closing(sqlite3.connect(path)) as connection:
        try:
            connection.execute("BEGIN IMMEDIATE")
            layout = _canonical_schema_layout()
            variant = _schema_variant(
                _schema_layout_from_connection(connection).signature,
                allow_empty=allow_empty,
            )
            if variant is None:
                raise ValueError("SQLite schema is not a supported canonical or legacy layout")
            if variant == "empty":
                _create_canonical_tables(
                    connection,
                    layout,
                    {name for name, _ in layout.table_sql},
                )
            elif variant == "a1":
                current = set(_user_table_names(connection))
                _create_canonical_tables(
                    connection,
                    layout,
                    {name for name, _ in layout.table_sql} - current,
                )
                _seed_a1_local_settings(connection)
            elif variant == "local-artifact-v1":
                _migrate_local_artifact_v1(connection, layout)
            elif variant == "precanonical-current":
                _rebuild_precanonical_current(connection, layout)
            _quarantine_current_legacy_decision_exports(connection)
            validate_local_insertion_sequence(connection)
            if _schema_layout_from_connection(connection).signature != layout.signature:
                raise ValueError("SQLite migration did not produce the canonical schema")
            connection.commit()
        except Exception:
            connection.rollback()
            raise


class Database:
    def __init__(self, settings: Settings) -> None:
        settings.ensure_directories()
        self.engine: Engine = create_engine(settings.database_url)
        self.session_factory: sessionmaker[Session] = sessionmaker(self.engine)

    @classmethod
    def sqlite_at(cls, path: Path) -> Database:
        path.parent.mkdir(parents=True, exist_ok=True)
        database = cls.__new__(cls)
        database.engine = create_engine(f"sqlite:///{path.as_posix()}")
        database.session_factory = sessionmaker(database.engine)
        return database

    def create_schema(self, *, allow_empty: bool = True) -> None:
        if self.engine.url.get_backend_name() == "sqlite":
            _migrate_sqlite_schema(self.sqlite_path, allow_empty=allow_empty)
            return
        Base.metadata.create_all(self.engine)

    @property
    def sqlite_path(self) -> Path:
        if self.engine.url.get_backend_name() != "sqlite":
            raise ValueError("database must use SQLite")
        value = self.engine.url.database
        if value is None or value == ":memory:":
            raise ValueError("database must use a file-backed SQLite database")
        return Path(value).resolve()

    def online_backup(self, destination: Path) -> None:
        """Create a transactionally consistent copy through SQLite's backup API."""

        if not isinstance(destination, Path):
            raise TypeError("backup destination must be a Path")
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_connection = self.engine.raw_connection()
        try:
            source = cast(sqlite3.Connection, source_connection.driver_connection)
            with closing(sqlite3.connect(destination)) as target:
                source.backup(target)
        finally:
            source_connection.close()
