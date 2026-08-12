from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


SENTINEL = "example-secret-sentinel"
pytestmark = pytest.mark.skipif(os.name != "nt", reason="PowerShell launcher is Windows-only")
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "start.ps1"
IDENTITY_COMMAND = (
    "run python -c import platform,sys; print(platform.python_implementation()); "
    "print(f'{sys.version_info.major}.{sys.version_info.minor}')"
)


def _write_fakes(tmp_path: Path) -> tuple[Path, Path, Path]:
    python = tmp_path / "fake-python.ps1"
    python.write_text(
        """
$implementation = if ($env:FQ_FAKE_PYTHON_IMPL) { $env:FQ_FAKE_PYTHON_IMPL } else { "CPython" }
$version = if ($env:FQ_FAKE_PYTHON_VERSION) { $env:FQ_FAKE_PYTHON_VERSION } else { "3.12" }
Write-Output $implementation
Write-Output $version
exit 0
""".strip(),
        "utf-8",
    )
    uv = tmp_path / "fake-uv.ps1"
    uv.write_text(
        """
$line = "$($PWD.Path)|$($args -join ' ')`n"
[System.IO.File]::AppendAllText($env:FQ_LAUNCHER_LOG, $line)
if ($args[0] -eq "--version") {
    Write-Output "uv 1.0.0"
    exit 0
}
if ($args[0] -eq "sync") {
    if ($env:FQ_FAKE_SYNC_STDERR -eq "1") {
        & powershell -NoProfile -Command "[Console]::Error.WriteLine('Resolved 1 package')"
    }
    exit [int]$env:FQ_FAKE_SYNC_EXIT
}
if (
    $args[0] -eq "run" -and
    $args[1] -eq "python" -and
    $args[2] -eq "-c"
) {
    $implementation = if ($env:FQ_FAKE_PYTHON_IMPL) { $env:FQ_FAKE_PYTHON_IMPL } else { "CPython" }
    $version = if ($env:FQ_FAKE_PYTHON_VERSION) { $env:FQ_FAKE_PYTHON_VERSION } else { "3.12" }
    Write-Output $implementation
    Write-Output $version
    exit 0
}
if ($args[0] -eq "run") {
    exit [int]$env:FQ_FAKE_RUN_EXIT
}
exit 99
""".strip(),
        "utf-8",
    )
    return python, uv, tmp_path / "launcher.log"


def _run(
    tmp_path: Path,
    *,
    implementation: str = "CPython",
    version: str = "3.12",
    sync_exit: int = 0,
    sync_stderr: bool = False,
    run_exit: int = 0,
    environment_file: Path | None = None,
    port: int | None = None,
    provide_python_command: bool = True,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    python, uv, log = _write_fakes(tmp_path)
    environment = os.environ.copy()
    environment.update(
        {
            "FQ_FAKE_PYTHON_IMPL": implementation,
            "FQ_FAKE_PYTHON_VERSION": version,
            "FQ_FAKE_SYNC_EXIT": str(sync_exit),
            "FQ_FAKE_SYNC_STDERR": "1" if sync_stderr else "0",
            "FQ_FAKE_RUN_EXIT": str(run_exit),
            "FQ_LAUNCHER_LOG": str(log),
            "COMPASS_TEST_SECRET": SENTINEL,
        }
    )
    arguments = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(SCRIPT),
        "-UvCommand",
        str(uv),
    ]
    if provide_python_command:
        arguments.extend(("-PythonCommand", str(python)))
    if environment_file is not None:
        arguments.extend(("-EnvironmentFile", str(environment_file)))
    if port is not None:
        arguments.extend(("-Port", str(port)))
    completed = subprocess.run(
        arguments,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    lines = [] if not log.exists() else log.read_text("utf-8").splitlines()
    return completed, lines


def test_launcher_runs_from_any_cwd_and_propagates_application_exit(
    tmp_path: Path,
) -> None:
    completed, lines = _run(tmp_path, run_exit=23)

    assert completed.returncode == 23
    assert lines == [
        f"{ROOT}|--version",
        f"{ROOT}|sync",
        f"{ROOT}|{IDENTITY_COMMAND}",
        f"{ROOT}|run python -m compass.ui.app --port 8080",
    ]
    assert SENTINEL not in completed.stdout
    assert SENTINEL not in completed.stderr
    assert str(tmp_path) not in completed.stdout
    assert str(tmp_path) not in completed.stderr


def test_launcher_requires_cpython_312_and_never_runs_uv_when_wrong(
    tmp_path: Path,
) -> None:
    completed, lines = _run(tmp_path, implementation="PyPy", version="3.11")

    assert completed.returncode != 0
    assert "CPython 3.12" in completed.stderr
    assert "需要" in completed.stderr
    assert lines == []
    assert SENTINEL not in completed.stderr


def test_launcher_stops_after_uv_sync_failure_and_propagates_exit(tmp_path: Path) -> None:
    completed, lines = _run(tmp_path, sync_exit=19)

    assert completed.returncode == 19
    assert lines == [f"{ROOT}|--version", f"{ROOT}|sync"]
    assert "依赖同步失败" in completed.stderr
    assert SENTINEL not in completed.stderr


def test_launcher_accepts_successful_uv_sync_progress_on_stderr(tmp_path: Path) -> None:
    completed, lines = _run(tmp_path, sync_stderr=True)

    assert completed.returncode == 0
    assert lines == [
        f"{ROOT}|--version",
        f"{ROOT}|sync",
        f"{ROOT}|{IDENTITY_COMMAND}",
        f"{ROOT}|run python -m compass.ui.app --port 8080",
    ]
    assert SENTINEL not in completed.stdout
    assert SENTINEL not in completed.stderr


def test_launcher_passes_untracked_dotenv_to_uv_without_echoing_it(tmp_path: Path) -> None:
    environment_file = tmp_path / ".env"
    environment_file.write_text(f"COMPASS_TEST_SECRET={SENTINEL}", "utf-8")

    completed, lines = _run(tmp_path, environment_file=environment_file)

    assert completed.returncode == 0
    assert lines == [
        f"{ROOT}|--version",
        f"{ROOT}|sync",
        f"{ROOT}|{IDENTITY_COMMAND}",
        f"{ROOT}|run --env-file {environment_file} python -m compass.ui.app --port 8080",
    ]
    assert SENTINEL not in completed.stdout
    assert SENTINEL not in completed.stderr
    assert str(environment_file) not in completed.stdout
    assert str(environment_file) not in completed.stderr


def test_launcher_passes_an_explicit_valid_local_port(tmp_path: Path) -> None:
    completed, lines = _run(tmp_path, port=8081)

    assert completed.returncode == 0
    assert lines[-1] == f"{ROOT}|run python -m compass.ui.app --port 8081"
    assert SENTINEL not in completed.stdout
    assert SENTINEL not in completed.stderr


def test_launcher_rejects_an_out_of_range_port_before_uv(tmp_path: Path) -> None:
    completed, lines = _run(tmp_path, port=0)

    assert completed.returncode != 0
    assert "端口" in completed.stderr
    assert lines == []


def test_launcher_uses_uv_managed_cpython_when_python_is_not_on_path(
    tmp_path: Path,
) -> None:
    completed, lines = _run(tmp_path, provide_python_command=False)

    assert completed.returncode == 0
    assert lines == [
        f"{ROOT}|--version",
        f"{ROOT}|sync",
        f"{ROOT}|{IDENTITY_COMMAND}",
        f"{ROOT}|run python -m compass.ui.app --port 8080",
    ]
    assert "CPython 3.12" not in completed.stderr
    assert SENTINEL not in completed.stdout
    assert SENTINEL not in completed.stderr
