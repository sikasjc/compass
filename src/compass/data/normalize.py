from __future__ import annotations

from collections.abc import Mapping

import pandas as pd  # type: ignore[import-untyped]

from compass.domain.market import BarFrame


def normalize_daily(frame: pd.DataFrame, mapping: Mapping[str, str]) -> pd.DataFrame:
    """Map source fields to canonical names and return a validated daily BarFrame.

    ``mapping`` maps source column names to canonical column names.  Daily dates
    identify exchange trading sessions, so timestamps are made timezone-naive and
    normalized to their local calendar midnight before validation.
    """

    mapped_targets = list(mapping.values())
    duplicate_mapped_targets = {
        target for target in mapped_targets if mapped_targets.count(target) > 1
    }
    if duplicate_mapped_targets:
        raise ValueError(f"duplicate canonical column: {sorted(duplicate_mapped_targets)[0]}")

    normalized_columns = [mapping.get(column, column) for column in frame.columns]
    duplicate_columns = {
        column for column in normalized_columns if normalized_columns.count(column) > 1
    }
    if duplicate_columns:
        raise ValueError(f"duplicate canonical column: {sorted(duplicate_columns)[0]}")

    normalized = frame.rename(columns=mapping).copy()
    if "date" not in normalized.columns:
        raise ValueError("daily data must include a date column")

    try:
        dates = pd.DatetimeIndex(pd.to_datetime(normalized.pop("date"), errors="raise"))
    except (TypeError, ValueError) as error:
        raise ValueError("daily date values must be parseable") from error
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    normalized.index = dates.normalize()
    normalized.index.name = "date"
    return BarFrame.validate(normalized.sort_index())
