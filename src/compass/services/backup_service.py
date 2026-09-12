from __future__ import annotations

from datetime import datetime, timezone
from contextlib import closing
from hashlib import sha256
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from uuid import uuid4
from zipfile import ZipFile, ZIP_DEFLATED
import json
import os
import shutil
import sqlite3


_DIRECTORIES = ("data", "reports", "logs")
MAX_ARCHIVE_BYTES = 512 * 1024**2
_MAX_BYTES = 2 * 1024**3


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class BackupService:
    """Verified complete backups; restore is staged and applied before opening storage."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.recovery = self.root / ".recovery"

    def _files(self) -> dict[str, Path]:
        files = {}
        for directory in _DIRECTORIES:
            for path in (self.root / directory).rglob("*"):
                if path.is_symlink() or not path.resolve().is_relative_to(self.root):
                    raise ValueError("运行目录包含外部链接，无法制作完整备份。")
                if path.is_file() and not path.name.endswith(("-wal", "-shm", ".tmp")):
                    files[path.relative_to(self.root).as_posix()] = path
        return files

    def create(self) -> Path:
        destination = self.root / ".backups" / f"compass-{uuid4().hex}.zip"
        destination.parent.mkdir(parents=True, exist_ok=True)
        database = self.root / "data" / "compass.db"
        if not database.is_file():
            raise ValueError("数据库尚未创建。")
        files = self._files()
        if len(files) >= 100_000 or sum(path.stat().st_size for path in files.values()) > _MAX_BYTES:
            raise ValueError("数据超过完整备份上限（解压后 2 GiB / 100000 个文件）。")
        required = sum(path.stat().st_size for path in files.values()) + MAX_ARCHIVE_BYTES
        if shutil.disk_usage(destination.parent).free < required:
            raise ValueError("磁盘空间不足，无法暂存并生成备份。")
        try:
            with TemporaryDirectory(dir=destination.parent) as temporary:
                snapshot = Path(temporary)
                # SQLite's online backup produces a coherent database snapshot.
                # Release its writer reservation before copying potentially
                # large market/report/log files, whose writers are independent
                # of the database lock and are checked by hashes below.
                with closing(sqlite3.connect(database, timeout=10)) as reservation:
                    reservation.execute("BEGIN IMMEDIATE")
                    copied_db = snapshot / "data" / "compass.db"
                    copied_db.parent.mkdir(parents=True)
                    with (
                        closing(sqlite3.connect(database)) as source,
                        closing(sqlite3.connect(copied_db)) as target,
                    ):
                        source.backup(target)
                    reservation.rollback()
                files = self._files()
                if sum(path.stat().st_size for path in files.values()) > _MAX_BYTES:
                    raise ValueError("备份数据已超过 2 GiB，请清理后重试。")
                fingerprints = {"data/compass.db": _file_hash(copied_db)}
                for name, path in files.items():
                    if name == "data/compass.db":
                        continue
                    copy = snapshot / name
                    copy.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("rb") as source_file, copy.open("wb") as target_file:
                        shutil.copyfileobj(source_file, target_file, 1024 * 1024)
                    fingerprints[name] = _file_hash(copy)
                current = self._files()
                if current.keys() != files.keys() or any(
                    _file_hash(path) != fingerprints[name]
                    for name, path in current.items() if name != "data/compass.db"
                ):
                    raise ValueError("备份期间数据发生变化，请等待后台任务结束后重试。")
                with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
                    for name in files:
                        archive.write(snapshot / name, name)
                    archive.writestr("manifest.json", json.dumps({
                        "schema_version": 1,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "files": fingerprints,
                    }, ensure_ascii=False))
                self.inspect(destination)
            return destination
        except BaseException:
            destination.unlink(missing_ok=True)
            raise

    def prune_backups(self, keep: int = 5) -> int:
        if keep < 1:
            raise ValueError("至少保留一份备份。")
        folder = (self.root / ".backups").resolve()
        backups = sorted(folder.glob("compass-*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        removed = 0
        for path in backups[keep:]:
            if path.is_symlink() or path.resolve().parent != folder:
                continue
            path.unlink()
            removed += 1
        return removed

    def inspect(self, archive_path: Path) -> dict[str, object]:
        if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("备份压缩包超过 512 MiB，请清理数据后重新备份。")
        with ZipFile(archive_path) as archive:
            infos = archive.infolist()
            if len(infos) > 100_000 or sum(item.file_size for item in infos) > _MAX_BYTES:
                raise ValueError("备份内容超过支持的大小或文件数量。")
            names = [item.filename for item in infos]
            if len({name.casefold() for name in names}) != len(names):
                raise ValueError("备份包含重名文件。")
            for item in infos:
                name = PurePosixPath(item.filename)
                if item.filename == "manifest.json":
                    continue
                reserved = {"CON", "PRN", "AUX", "NUL"} | {
                    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
                }
                if (
                    item.is_dir()
                    or name.is_absolute()
                    or not name.parts
                    or name.as_posix() != item.filename
                    or name.parts[0] not in _DIRECTORIES
                    or any(
                        part in {".", ".."}
                        or ":" in part
                        or "\\" in part
                        or part.endswith((" ", "."))
                        or part.split(".")[0].upper() in reserved
                        for part in name.parts
                    )
                    or item.external_attr >> 16 & 0o170000 == 0o120000
                ):
                    raise ValueError("备份包含不允许的路径或链接。")
            try:
                manifest = json.loads(archive.read("manifest.json"))
                files = manifest["files"]
                if (
                    manifest["schema_version"] != 1
                    or not isinstance(files, dict)
                    or set(files) != set(names) - {"manifest.json"}
                    or "data/compass.db" not in files
                ):
                    raise ValueError
                for name, expected in files.items():
                    with archive.open(name) as stream:
                        digest = sha256()
                        while chunk := stream.read(1024 * 1024):
                            digest.update(chunk)
                    if digest.hexdigest() != expected:
                        raise ValueError
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("备份校验失败，文件不完整或格式不受支持。") from error
            return {
                "created_at": str(manifest.get("created_at", "未知")),
                "files": len(files),
                "bytes": sum(item.file_size for item in infos),
            }

    def stage(self, archive_path: Path) -> dict[str, object]:
        summary = self.inspect(archive_path)
        self.recovery.mkdir(parents=True, exist_ok=True)
        if (self.recovery / "pending.json").exists():
            raise ValueError("已有待恢复备份，请先重启完成恢复或取消待恢复任务。")
        token = uuid4().hex
        staging = self.recovery / f"staged-{token}"
        staging.mkdir()
        try:
            with ZipFile(archive_path) as archive:
                (staging / "manifest.json").write_bytes(archive.read("manifest.json"))
                for item in archive.infolist():
                    if item.filename == "manifest.json":
                        continue
                    target = staging / item.filename
                    if not target.resolve().is_relative_to(staging.resolve()):
                        raise ValueError("恢复路径超出暂存目录。")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(item) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
            with closing(sqlite3.connect(staging / "data" / "compass.db")) as database:
                if database.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                    raise ValueError("备份数据库完整性检查失败。")
            pending = self.recovery / f"pending-{token}.tmp"
            pending.write_text(
                json.dumps(
                    {
                        "token": token,
                        "manifest_hash": sha256((staging / "manifest.json").read_bytes()).hexdigest(),
                    }
                ),
                "utf-8",
            )
            try:
                os.link(pending, self.recovery / "pending.json")
            finally:
                pending.unlink(missing_ok=True)
        except BaseException:
            self._remove_staging(staging)
            raise
        return summary

    def _remove_staging(self, path: Path) -> None:
        recovery = self.recovery.resolve()
        resolved = path.resolve()
        suffix = path.name.removeprefix("staged-")
        if (path.is_symlink() or resolved.parent != recovery or not path.name.startswith("staged-")
                or len(suffix) != 32 or any(c not in "0123456789abcdef" for c in suffix)):
            raise ValueError("恢复暂存路径无效，未执行清理。")
        if resolved.exists():
            shutil.rmtree(resolved)

    def cancel_pending(self) -> None:
        if (self.recovery / "applying.json").exists():
            raise ValueError("恢复已经开始，请先重启完成恢复，不能直接取消。")
        marker = self.recovery / "pending.json"
        if not marker.exists():
            return
        token = json.loads(marker.read_text("utf-8"))["token"]
        if type(token) is not str or len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
            raise ValueError("待恢复任务无效，未执行清理。")
        marker.unlink()
        self._remove_staging(self.recovery / f"staged-{token}")

    def apply_pending(self) -> bool:
        marker = self.recovery / "pending.json"
        journal = self.recovery / "applying.json"
        if not marker.exists():
            # A removed pending marker is the commit point of a successful restore.
            journal.unlink(missing_ok=True)
            return False
        pending = json.loads(marker.read_text("utf-8"))
        token = pending["token"]
        if (
            type(token) is not str
            or len(token) != 32
            or any(c not in "0123456789abcdef" for c in token)
        ):
            raise ValueError("待恢复任务无效。")
        staging = self.recovery / f"staged-{token}"
        previous = self.recovery / f"previous-{token}"
        previous.mkdir(exist_ok=True)
        # Every final move target is resolved and checked before any directory move.
        for directory in _DIRECTORIES:
            for path in (self.root / directory, staging / directory, previous / directory):
                if path.is_symlink() or not path.resolve().is_relative_to(self.root):
                    raise ValueError("恢复目标不在运行目录内。")

        def rollback(original: list[str], incoming: list[str]) -> None:
            for directory in _DIRECTORIES:
                current, old, new = self.root / directory, previous / directory, staging / directory
                if directory in original and old.exists():
                    if current.exists():
                        if new.exists():
                            raise ValueError("恢复状态冲突，原数据仍保留在 .recovery。")
                        os.replace(current, new)
                    os.replace(old, current)
                elif directory not in original and directory in incoming and not new.exists():
                    if current.exists():
                        os.replace(current, new)

        if journal.exists():
            interrupted = json.loads(journal.read_text("utf-8"))
            if interrupted["token"] != token:
                raise ValueError("恢复任务状态不一致，请保留 .recovery 并检查日志。")
            rollback(interrupted["original"], interrupted["incoming"])
            journal.unlink()
        manifest_bytes = (staging / "manifest.json").read_bytes()
        if sha256(manifest_bytes).hexdigest() != pending["manifest_hash"]:
            raise ValueError("暂存备份清单发生变化，恢复已停止。")
        files = json.loads(manifest_bytes)["files"]
        actual = {
            path.relative_to(staging).as_posix(): path
            for directory in _DIRECTORIES
            for path in (staging / directory).rglob("*")
            if path.is_file()
        }
        if set(actual) != set(files) or any(
            path.is_symlink()
            or not path.resolve().is_relative_to(staging.resolve())
            or _file_hash(path) != files[name]
            for name, path in actual.items()
        ):
            raise ValueError("暂存备份不完整或发生变化，恢复已停止。")
        original = [name for name in _DIRECTORIES if (self.root / name).exists()]
        incoming = [name for name in _DIRECTORIES if (staging / name).exists()]
        journal.write_text(
            json.dumps({"token": token, "original": original, "incoming": incoming}), "utf-8"
        )
        try:
            for directory in _DIRECTORIES:
                current = self.root / directory
                if current.exists():
                    os.replace(current, previous / directory)
                restored = staging / directory
                if restored.exists():
                    os.replace(restored, current)
            marker.unlink()
            journal.unlink()
        except Exception:
            if marker.exists():
                rollback(original, incoming)
                journal.unlink(missing_ok=True)
            raise
        return True
