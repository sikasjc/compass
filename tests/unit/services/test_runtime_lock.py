from pathlib import Path
import subprocess
import sys

from compass.services.runtime_lock import RuntimeLock
from compass.ui.navigation import account_url, account_storage_key
from compass.config import runtime_label


def test_runtime_lock_rejects_other_process_and_releases(tmp_path: Path) -> None:
    script = """
from pathlib import Path
import sys
from compass.services.runtime_lock import RuntimeLock
try:
    with RuntimeLock(Path(sys.argv[1]), 8102):
        pass
except RuntimeError as error:
    print(error)
    sys.exit(3)
"""
    with RuntimeLock(tmp_path, 8101):
        result = subprocess.run([sys.executable, "-c", script, str(tmp_path)],
                                capture_output=True, text=True, timeout=20)
        assert result.returncode == 3
        assert "127.0.0.1:8101" in result.stdout
    with RuntimeLock(tmp_path, 8102):
        pass


def test_account_navigation_preserves_other_query_parameters(tmp_path: Path) -> None:
    assert account_url("/backtests?configuration_key=a#result", "b") == (
        "/backtests?configuration_key=a&account_id=b#result"
    )
    assert account_url("/account?account_id=a", "b") == "/account?account_id=b"
    assert account_storage_key(tmp_path) != account_storage_key(tmp_path / "other")


def test_custom_and_test_environments_are_not_labelled_production(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("COMPASS_ENV", raising=False)
    assert runtime_label(tmp_path) != "正式数据"
    monkeypatch.setenv("COMPASS_ENV", "test")
    assert runtime_label(tmp_path) == "测试数据"
