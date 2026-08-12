from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from compass.data.base import DailyBarRequest, ProviderCapabilityError
from compass.data.normalize import normalize_daily
from compass.domain.market import BarFrame, InstrumentId
from compass.domain.trading import CorporateAction


class CsvProvider:
    """Read canonical daily bar fixtures from ``daily_<instrument-code>.csv`` files."""

    name = "csv"

    def __init__(self, root: Path) -> None:
        self.root = root

    def fetch_daily(self, request: DailyBarRequest) -> pd.DataFrame:
        path = self.root / f"daily_{request.instrument.code}.csv"
        mapping = {column: column for column in (*BarFrame.REQUIRED_COLUMNS, *BarFrame.OPTIONAL_COLUMNS)}
        frame = normalize_daily(pd.read_csv(path), mapping)
        return BarFrame.validate(frame.loc[pd.Timestamp(request.start) : pd.Timestamp(request.end)])

    def fetch_snapshot(self, instruments: Sequence[InstrumentId]) -> pd.DataFrame:
        raise ProviderCapabilityError(self.name, "snapshot")

    def fetch_corporate_actions(self, request: DailyBarRequest) -> Sequence[CorporateAction]:
        raise ProviderCapabilityError(self.name, "corporate_actions")
