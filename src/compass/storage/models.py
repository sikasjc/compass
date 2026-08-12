from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator


class OffsetDateTime(TypeDecorator[datetime]):
    """Store timezone-aware datetimes as their offset-preserving ISO-8601 text."""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> str | None:
        del dialect
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("datetime values must be timezone-aware")
        return value.isoformat()

    def process_result_value(self, value: Any, dialect: Dialect) -> datetime | None:
        del dialect
        if value is None:
            return None
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("stored datetime values must include an offset")
        return parsed


class Base(DeclarativeBase):
    pass


class AccountSnapshot(Base):
    __tablename__ = "account_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[str] = mapped_column(String(128))
    captured_at: Mapped[datetime] = mapped_column(OffsetDateTime())
    as_of: Mapped[str] = mapped_column(String(10))
    payload_json: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    cash: Mapped[float] = mapped_column(Float)
    market_value: Mapped[float] = mapped_column(Float)
    equity: Mapped[float] = mapped_column(Float)


class StrategyInstance(Base):
    __tablename__ = "strategy_instances"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32))
    configuration_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(OffsetDateTime())


class Watchlist(Base):
    __tablename__ = "watchlists"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    instruments_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(OffsetDateTime())


class DatasetManifestRecord(Base):
    __tablename__ = "dataset_manifests"

    manifest_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    instrument: Mapped[str] = mapped_column(String(16))
    provider: Mapped[str] = mapped_column(String(64))
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    relative_data_path: Mapped[str] = mapped_column(String(512))
    rows: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(OffsetDateTime())
    quality_report_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    strategy_instance_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32))
    started_at: Mapped[datetime] = mapped_column(OffsetDateTime())
    completed_at: Mapped[datetime | None] = mapped_column(OffsetDateTime(), nullable=True)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class RunSnapshotRecord(Base):
    __tablename__ = "run_snapshots"

    snapshot_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    schema_version: Mapped[int] = mapped_column(Integer)
    payload_json: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(OffsetDateTime())


class DecisionRun(Base):
    __tablename__ = "decision_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    strategy_instance_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32))
    decided_at: Mapped[datetime] = mapped_column(OffsetDateTime())
    decision_json: Mapped[str] = mapped_column(Text)


class TaskRun(Base):
    __tablename__ = "task_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_name: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32))
    started_at: Mapped[datetime] = mapped_column(OffsetDateTime())
    completed_at: Mapped[datetime | None] = mapped_column(OffsetDateTime(), nullable=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)


class LocalWatchlistRecord(Base):
    __tablename__ = "local_watchlists"

    watchlist_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[int] = mapped_column(Integer)
    payload_json: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    updated_at: Mapped[datetime] = mapped_column(OffsetDateTime())


class LocalStrategyVersionRecord(Base):
    __tablename__ = "local_strategy_versions"

    instance_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    lineage_id: Mapped[str] = mapped_column(String(128), index=True)
    version: Mapped[int] = mapped_column(Integer)
    enabled: Mapped[int] = mapped_column(Integer)
    schema_version: Mapped[int] = mapped_column(Integer)
    payload_json: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(OffsetDateTime())


class LocalSettingsRecord(Base):
    __tablename__ = "local_settings"

    settings_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    schema_version: Mapped[int] = mapped_column(Integer)
    payload_json: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    updated_at: Mapped[datetime] = mapped_column(OffsetDateTime())


class DatasetBundleRecord(Base):
    __tablename__ = "dataset_bundles"

    bundle_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    primary_manifest_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    schema_version: Mapped[int] = mapped_column(Integer)
    payload_json: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(OffsetDateTime())
    write_order: Mapped[int] = mapped_column(Integer, nullable=False)


class BacktestReportRecord(Base):
    __tablename__ = "backtest_reports"

    run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    schema_version: Mapped[int] = mapped_column(Integer)
    payload_json: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(OffsetDateTime())
    write_order: Mapped[int] = mapped_column(Integer, nullable=False)


class DecisionExportRecordRow(Base):
    __tablename__ = "decision_exports"

    decision_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    # Nullable keeps the additive SQLite migration compatible with pre-snapshot
    # rows.  Current schema-version records require a non-null, exact link.
    snapshot_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    schema_version: Mapped[int] = mapped_column(Integer)
    payload_json: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(OffsetDateTime())
    write_order: Mapped[int] = mapped_column(Integer, nullable=False)


class LegacyDecisionExportRecord(Base):
    """Auditable v1 decision rows that cannot satisfy current provenance guarantees."""

    __tablename__ = "legacy_decision_exports"

    decision_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[int] = mapped_column(Integer)
    payload_json: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(OffsetDateTime())


class LocalInsertionSequence(Base):
    __tablename__ = "local_insertion_sequences"
    __table_args__ = {"sqlite_autoincrement": True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
