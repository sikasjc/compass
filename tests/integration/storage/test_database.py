from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import inspect, select

from compass.config import Settings
from compass.storage.database import Database
from compass.storage.models import TaskRun


def test_create_schema_creates_all_persistence_tables(tmp_path: Path) -> None:
    settings = Settings.from_env(tmp_path)
    settings.ensure_directories()
    database = Database(settings)

    database.create_schema()

    assert set(inspect(database.engine).get_table_names()) == {
        "account_snapshots",
        "backtest_reports",
        "strategy_instances",
        "watchlists",
        "dataset_manifests",
        "backtest_runs",
        "dataset_bundles",
        "decision_exports",
        "decision_runs",
        "legacy_decision_exports",
        "local_settings",
        "local_insertion_sequences",
        "local_strategy_versions",
        "local_watchlists",
        "run_snapshots",
        "task_runs",
    }


def test_sqlite_round_trips_aware_datetimes_with_their_offset(tmp_path: Path) -> None:
    settings = Settings.from_env(tmp_path)
    database = Database(settings)
    timestamp = datetime(2026, 7, 20, 15, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    database.create_schema()

    with database.session_factory.begin() as session:
        session.add(
            TaskRun(task_name="daily", status="done", started_at=timestamp, completed_at=timestamp)
        )

    with database.session_factory() as session:
        task_run = session.scalar(select(TaskRun))

    assert task_run is not None
    assert task_run.started_at == timestamp
    assert task_run.completed_at == timestamp
