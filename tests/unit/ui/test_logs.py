from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from compass.services.diagnostic_log import DiagnosticLogEntry
from compass.ui.pages.logs import LogsPageModel
from compass.ui.pages.settings import SettingsSnapshot


NOW = datetime(2026, 8, 9, 10, tzinfo=ZoneInfo("Asia/Shanghai"))


class Gateway:
    def __init__(self) -> None:
        self.level = "INFO"
        self.read_arguments: tuple[int, str | None, str] | None = None
        self.entry = DiagnosticLogEntry(
            NOW,
            "ERROR",
            "compass.market_data",
            "request failed",
        )

    def state(self) -> SettingsSnapshot:
        return SettingsSnapshot((), self.level)

    def set_log_level(self, level: str) -> None:
        self.level = level

    def read_logs(
        self, limit: int, level: str | None, query: str
    ) -> tuple[DiagnosticLogEntry, ...]:
        self.read_arguments = (limit, level, query)
        return (self.entry,)


def test_logs_page_model_owns_level_and_filtered_log_access() -> None:
    gateway = Gateway()
    model = LogsPageModel(gateway)

    model.set_log_level("DEBUG")
    entries = model.logs(limit=50, level="ERROR", query=" request ")

    assert model.log_level() == "DEBUG"
    assert entries == (gateway.entry,)
    assert gateway.read_arguments == (50, "ERROR", "request")


def test_logs_page_model_rejects_invalid_filters() -> None:
    model = LogsPageModel(Gateway())

    with pytest.raises(ValueError, match="LOG_LEVEL_INVALID"):
        model.logs(level="TRACE")
    with pytest.raises(ValueError, match="LOG_QUERY_INVALID"):
        model.logs(query="x" * 101)
