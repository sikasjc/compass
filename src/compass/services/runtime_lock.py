from __future__ import annotations

from pathlib import Path
from typing import BinaryIO
import json
import os
import sys


class RuntimeLock:
    """Hold an OS lock for the complete application lifetime, including restore."""

    def __init__(self, root: Path, port: int) -> None:
        self.root = root.resolve()
        self.port = port
        self._stream: BinaryIO | None = None

    def __enter__(self) -> RuntimeLock:
        self.root.mkdir(parents=True, exist_ok=True)
        stream = (self.root / ".compass.lock").open("a+b")
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            stream.close()
            try:
                active = json.loads((self.root / ".compass-instance.json").read_text("utf-8"))
                port = int(active["port"])
                address = f"http://127.0.0.1:{port}/" if 1 <= port <= 65535 else ""
            except (OSError, ValueError, KeyError, TypeError):
                address = ""
            raise RuntimeError(
                f"此数据目录已有 Compass 实例运行：{self.root}。请使用已有页面 {address}，"
                "或先关闭原实例再启动。"
            ) from error
        self._stream = stream
        try:
            (self.root / ".compass-instance.json").write_text(
                json.dumps({"pid": os.getpid(), "port": self.port}), "utf-8",
            )
        except OSError:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_: object) -> None:
        if self._stream is not None:
            # Closing the descriptor releases the lock even after process failure.
            self._stream.close()
            self._stream = None
