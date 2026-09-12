from contextlib import closing
from pathlib import Path
from zipfile import ZipFile
import json
import os
import sqlite3

import pytest

from compass.services.backup_service import BackupService


def runtime(root: Path) -> BackupService:
    for directory in ("data/market", "reports", "logs"):
        (root / directory).mkdir(parents=True)
    with closing(sqlite3.connect(root / "data/compass.db")) as database:
        database.execute("CREATE TABLE example(value TEXT)")
        database.execute("INSERT INTO example VALUES ('original')")
        database.commit()
    (root / "data/market/prices.parquet").write_bytes(b"market-snapshot")
    (root / "data/signal_accounts.json").write_text('{"name":"account"}', "utf-8")
    (root / "reports/report.json").write_text("original-report", "utf-8")
    (root / "logs/compass.log").write_text("original-log", "utf-8")
    return BackupService(root)


def test_backup_restore_preserves_complete_data_and_previous_version(tmp_path: Path) -> None:
    service = runtime(tmp_path)
    archive = service.create()
    assert service.inspect(archive)["files"] == 5
    (tmp_path / "reports/report.json").write_text("new-report", "utf-8")
    service.stage(archive)
    assert (tmp_path / "reports/report.json").read_text("utf-8") == "new-report"
    assert service.apply_pending() is True
    assert service.apply_pending() is False
    assert (tmp_path / "reports/report.json").read_text("utf-8") == "original-report"
    previous = next(service.recovery.glob("previous-*"))
    assert (previous / "reports/report.json").read_text("utf-8") == "new-report"
    assert (tmp_path / "data/market/prices.parquet").read_bytes() == b"market-snapshot"
    with closing(sqlite3.connect(tmp_path / "data/compass.db")) as database:
        assert database.execute("SELECT value FROM example").fetchone() == ("original",)


def test_interrupted_restore_rolls_back_then_finishes_on_next_start(
    tmp_path: Path, monkeypatch
) -> None:
    service = runtime(tmp_path)
    archive = service.create()
    (tmp_path / "reports/report.json").write_text("new-report", "utf-8")
    service.stage(archive)
    replace = os.replace

    def interrupt(source, target):
        if Path(source) == tmp_path / "reports":
            raise KeyboardInterrupt("simulate process interruption")
        return replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", interrupt)
        with pytest.raises(KeyboardInterrupt):
            service.apply_pending()
    assert service.apply_pending() is True
    assert (tmp_path / "reports/report.json").read_text("utf-8") == "original-report"
    previous = next(service.recovery.glob("previous-*"))
    assert (previous / "reports/report.json").read_text("utf-8") == "new-report"


@pytest.mark.parametrize("name", ["../escape", "data/../escape", "data/CON", "data/a:stream"])
def test_restore_rejects_unsafe_archive_paths(tmp_path: Path, name: str) -> None:
    archive = tmp_path / "unsafe.zip"
    with ZipFile(archive, "w") as output:
        output.writestr(name, b"payload")
    with pytest.raises(ValueError, match="路径"):
        BackupService(tmp_path).stage(archive)
    assert not (tmp_path / ".recovery/pending.json").exists()


def test_staged_corruption_does_not_replace_current_data(tmp_path: Path) -> None:
    service = runtime(tmp_path)
    service.stage(service.create())
    token = json.loads((service.recovery / "pending.json").read_text("utf-8"))["token"]
    (service.recovery / f"staged-{token}/reports/report.json").write_text("damaged", "utf-8")
    with pytest.raises(ValueError, match="发生变化"):
        service.apply_pending()
    assert (tmp_path / "reports/report.json").read_text("utf-8") == "original-report"
    service.cancel_pending()
    assert service.apply_pending() is False


def test_oversize_archive_is_not_offered_and_leaves_no_partial_backup(tmp_path, monkeypatch):
    import compass.services.backup_service as module
    service = runtime(tmp_path)
    monkeypatch.setattr(module, "MAX_ARCHIVE_BYTES", 100)
    with pytest.raises(ValueError, match="512 MiB"):
        service.create()
    assert list((tmp_path / ".backups").glob("*.zip")) == []


def test_backup_compression_does_not_hold_sqlite_writer(tmp_path, monkeypatch):
    import compass.services.backup_service as module

    service = runtime(tmp_path)
    original = module.shutil.copyfileobj
    wrote = False

    def copy_after_edit(source, target, length=0):
        nonlocal wrote
        if not wrote:
            with closing(sqlite3.connect(tmp_path / "data/compass.db", timeout=0.1)) as database:
                database.execute("UPDATE example SET value='new'")
                database.commit()
            wrote = True
        return original(source, target, length)

    monkeypatch.setattr(module.shutil, "copyfileobj", copy_after_edit)
    archive = service.create()
    service.stage(archive)
    staged = next(service.recovery.glob("staged-*"))
    with closing(sqlite3.connect(staged / "data/compass.db")) as database:
        assert database.execute("SELECT value FROM example").fetchone() == ("original",)
    service.cancel_pending()
    assert not staged.exists()
    with closing(sqlite3.connect(tmp_path / "data/compass.db")) as database:
        assert database.execute("SELECT value FROM example").fetchone() == ("new",)


def test_failed_stage_cleans_files_and_retention_keeps_latest_five(tmp_path, monkeypatch):
    service = runtime(tmp_path)
    archives = [service.create() for _ in range(6)]
    assert service.prune_backups() == 1
    assert len(list((tmp_path / ".backups").glob("*.zip"))) == 5
    def fail(*args):
        raise OSError("failed marker write")
    monkeypatch.setattr(os, "link", fail)
    with pytest.raises(OSError):
        service.stage(archives[-1])
    assert list(service.recovery.glob("staged-*")) == []
    assert not (service.recovery / "pending.json").exists()
