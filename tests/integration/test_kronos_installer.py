from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


pytestmark = pytest.mark.skipif(os.name != "nt", reason="PowerShell installer is Windows-only")
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "install-kronos.ps1"


def _run_installer(tmp_path: Path, *arguments: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    log = tmp_path / "installer.log"
    uv = tmp_path / "fake-uv.ps1"
    uv.write_text(
        """
[System.IO.File]::AppendAllText(
    $env:COMPASS_INSTALLER_LOG,
    "$($args -join ' ')|$env:HTTPS_PROXY`n"
)
if ($args[0] -eq "--version") { Write-Output "uv 1.0.0"; exit 0 }
if ($args[0] -eq "sync") { exit 0 }
if ($args[0] -eq "run") { Write-Output "CUDA available: True"; exit 0 }
exit 99
""".strip(),
        "utf-8",
    )
    nvidia_smi = tmp_path / "fake-nvidia-smi.ps1"
    nvidia_smi.write_text('Write-Output "RTX 4070 Ti, 596.36"; exit 0', "utf-8")
    environment = os.environ.copy()
    environment["COMPASS_INSTALLER_LOG"] = str(log)
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-UvCommand",
            str(uv),
            "-NvidiaSmiCommand",
            str(nvidia_smi),
            *arguments,
        ],
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


def test_cuda_installer_passes_proxy_and_runs_self_checks(tmp_path: Path) -> None:
    completed, lines = _run_installer(
        tmp_path,
        "-Mode",
        "CUDA",
        "-Proxy",
        "http://127.0.0.1:7897",
        "-IncludeDev",
    )

    assert completed.returncode == 0
    assert lines[0].startswith("--version|")
    assert "sync --extra kronos-cuda --extra dev|http://127.0.0.1:7897" in lines
    assert sum(line.startswith("run python -c") for line in lines) == 2
    assert "安装并验证完成" in completed.stdout


def test_cpu_installer_does_not_require_nvidia(tmp_path: Path) -> None:
    completed, lines = _run_installer(tmp_path, "-Mode", "CPU")

    assert completed.returncode == 0
    assert any(line.startswith("sync --extra kronos|") for line in lines)


def test_installer_rejects_invalid_proxy_before_sync(tmp_path: Path) -> None:
    completed, lines = _run_installer(tmp_path, "-Proxy", "127.0.0.1:7897")

    assert completed.returncode == 22
    assert not any(line.startswith("sync ") for line in lines)
    assert "代理地址" in completed.stderr
