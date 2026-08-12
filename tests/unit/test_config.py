import os
from pathlib import Path

from compass.config import Settings, _default_runtime_root


def test_settings_create_only_local_runtime_directories(tmp_path: Path) -> None:
    settings = Settings.from_env(tmp_path)
    settings.ensure_directories()

    assert settings.timezone == "Asia/Shanghai"
    database_path = (tmp_path / "data" / "compass.db").as_posix()
    assert settings.database_url == f"sqlite:///{database_path}"
    assert settings.market_data_dir == tmp_path / "data" / "market"
    assert settings.reports_dir == tmp_path / "reports"
    assert settings.market_data_dir.is_dir()
    assert settings.reports_dir.is_dir()


def test_default_settings_use_local_app_data_outside_repository(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    local_app_data = tmp_path / "LocalAppData"
    monkeypatch.delenv("COMPASS_DATA_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", os.fspath(local_app_data))

    monkeypatch.setattr("compass.config.sys.platform", "win32")
    settings = Settings.from_env()

    assert settings.root == local_app_data / "Compass"
    assert settings.market_data_dir == settings.root / "data" / "market"
    assert settings.logs_dir == settings.root / "logs"


def test_runtime_data_directory_can_be_overridden(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    custom = tmp_path / "shared-runtime"
    monkeypatch.setenv("COMPASS_DATA_DIR", os.fspath(custom))

    settings = Settings.from_env()

    assert settings.root == custom


def test_macos_uses_application_support(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("COMPASS_DATA_DIR", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    assert _default_runtime_root("darwin") == (
        tmp_path / "Library" / "Application Support" / "Compass"
    )


def test_linux_uses_xdg_data_home_when_available(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    xdg = tmp_path / "xdg-data"
    monkeypatch.delenv("COMPASS_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", os.fspath(xdg))

    assert _default_runtime_root("linux") == xdg / "compass"


def test_linux_falls_back_to_home_local_share(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("COMPASS_DATA_DIR", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    assert _default_runtime_root("linux") == tmp_path / ".local" / "share" / "compass"
