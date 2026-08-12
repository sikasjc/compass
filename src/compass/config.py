from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import sys


_DATA_DIR_ENVIRONMENT = "COMPASS_DATA_DIR"


def _default_runtime_root(platform: str | None = None) -> Path:
    configured = os.getenv(_DATA_DIR_ENVIRONMENT, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    active_platform = sys.platform if platform is None else platform
    if active_platform == "win32":
        local_app_data = os.getenv("LOCALAPPDATA", "").strip()
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
        return (base / "Compass").resolve()
    if active_platform == "darwin":
        return (Path.home() / "Library" / "Application Support" / "Compass").resolve()
    xdg_data_home = os.getenv("XDG_DATA_HOME", "").strip()
    if xdg_data_home:
        return (Path(xdg_data_home).expanduser() / "compass").resolve()
    return (Path.home() / ".local" / "share" / "compass").resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    root: Path
    timezone: str
    database_url: str
    market_data_dir: Path
    reports_dir: Path
    logs_dir: Path

    @classmethod
    def from_env(cls, root: Path | None = None) -> "Settings":
        runtime_root = root.resolve() if root is not None else _default_runtime_root()
        data_dir = runtime_root / "data"
        return cls(
            root=runtime_root,
            timezone="Asia/Shanghai",
            database_url=f"sqlite:///{(data_dir / 'compass.db').as_posix()}",
            market_data_dir=data_dir / "market",
            reports_dir=runtime_root / "reports",
            logs_dir=runtime_root / "logs",
        )

    def ensure_directories(self) -> None:
        for path in (self.market_data_dir, self.reports_dir, self.logs_dir):
            path.mkdir(parents=True, exist_ok=True)
