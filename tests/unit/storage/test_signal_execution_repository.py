from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from compass.storage.signal_execution_repository import (
    SignalExecutionFill,
    SignalExecutionRecord,
    SignalExecutionRepository,
    SignalExecutionStatus,
)


NOW = datetime(2026, 8, 12, 9, 30, tzinfo=ZoneInfo("Asia/Shanghai"))


def test_signal_execution_repository_persists_exact_partial_execution(tmp_path) -> None:
    path = tmp_path / "executions.json"
    repository = SignalExecutionRepository(path)
    record = SignalExecutionRecord(
        "decision-1:main",
        "main",
        SignalExecutionStatus.PARTIAL,
        (SignalExecutionFill("SSE.510300", 100, Decimal("4.1234")),),
        Decimal("5.00"),
        NOW,
        9,
    )

    repository.save(record)

    assert SignalExecutionRepository(path).get(record.decision_id) == record
    with pytest.raises(ValueError, match="SIGNAL_EXECUTION_ALREADY_RECORDED"):
        repository.save(record)


def test_signal_execution_repository_records_ignored_without_snapshot(tmp_path) -> None:
    repository = SignalExecutionRepository(tmp_path / "executions.json")
    record = SignalExecutionRecord(
        "decision-ignored:main",
        "main",
        SignalExecutionStatus.IGNORED,
        (),
        Decimal("0"),
        NOW,
    )

    assert repository.save(record) == record
