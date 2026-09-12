from contextlib import closing
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
import sqlite3
import subprocess
import sys
from zoneinfo import ZoneInfo

from compass.domain.trading import AccountSnapshot
from compass.storage.account_repository import AccountRepository
from compass.storage.database import Database
from compass.storage.signal_execution_repository import (
    SignalExecutionRecord, SignalExecutionRepository, SignalExecutionStatus,
)


NOW = datetime(2026, 9, 11, tzinfo=ZoneInfo("Asia/Shanghai"))


def test_previous_schema_and_json_execution_migrate_once(tmp_path: Path) -> None:
    database = Database.sqlite_at(tmp_path / "compass.db")
    database.create_schema()
    accounts = AccountRepository(database, "main", lambda: NOW)
    original = accounts.save(AccountSnapshot(date(2026, 9, 11), Decimal("100"), ()))
    database.engine.dispose()
    with closing(sqlite3.connect(database.sqlite_path)) as connection:
        connection.execute("DROP TABLE signal_execution_registry")
        connection.commit()
    legacy_path = tmp_path / "executions.json"
    legacy = SignalExecutionRepository(legacy_path)
    record = SignalExecutionRecord("decision-1", "main", SignalExecutionStatus.IGNORED, (),
                                   Decimal("0"), NOW)
    legacy.save(record)
    before = legacy_path.read_bytes()
    database.create_schema(allow_empty=False)
    upgraded = SignalExecutionRepository(legacy_path, database)
    assert upgraded.get(record.decision_id) == record
    assert accounts.latest() == original
    assert legacy_path.read_bytes() == before
    upgraded.delete((record.decision_id,))
    assert SignalExecutionRepository(legacy_path, database).get(record.decision_id) is None
    database.engine.dispose()


def test_process_exit_rolls_back_uncommitted_account_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "compass.db"
    database = Database.sqlite_at(path)
    database.create_schema()
    repository = AccountRepository(database, "main", lambda: NOW)
    original = repository.save(AccountSnapshot(NOW.date(), Decimal("100"), ()))
    script = """
import os, sys
from pathlib import Path
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo
from compass.storage.database import Database
from compass.storage.account_repository import AccountRepository
from compass.domain.trading import AccountSnapshot
now = datetime(2026, 9, 11, tzinfo=ZoneInfo('Asia/Shanghai'))
database = Database.sqlite_at(Path(sys.argv[1]))
repository = AccountRepository(database, 'main', lambda: now)
with repository.transaction() as transaction:
    repository.save(AccountSnapshot(now.date(), Decimal('90'), ()),
                    expected_row_id=int(sys.argv[2]), transaction=transaction)
    os._exit(0)
"""
    subprocess.run([sys.executable, "-c", script, str(path), str(original.row_id)],
                   check=True, timeout=20)
    assert repository.latest() == original
    database.engine.dispose()
