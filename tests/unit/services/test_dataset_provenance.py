from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from compass.data.exchange_calendar import CalendarIdentity
from compass.services.dataset_provenance import validate_dataset_provenance
from compass.storage.market_store import (
    CorporateActionProviderAttempt,
    DailyProviderAttempt,
    DatasetManifest,
    DatasetProvenance,
)


NOW = datetime(2026, 8, 11, 16, tzinfo=ZoneInfo("Asia/Shanghai"))


def _manifest(
    suffix: str,
    sessions: tuple[date, ...],
    *,
    calendar_id: str,
    covered_to: date,
    calendar_provider: str = "akshare",
) -> DatasetManifest:
    provenance = DatasetProvenance(
        daily_attempts=(DailyProviderAttempt("tencent", "selected"),),
        selected_provider="tencent",
        fetched_at=NOW,
        source_at=None,
        calendar=CalendarIdentity(
            calendar_id,
            calendar_provider,
            "akshare-tool-trade-date-hist-sina-v1",
            date(2026, 8, 7),
            covered_to,
        ),
        completed_through=sessions[-1],
        daily_complete=True,
        corporate_actions_status="unavailable",
        corporate_actions_provider=None,
        corporate_actions=(),
        calendar_sessions=sessions,
        missing_sessions=(),
        corporate_action_attempts=(
            CorporateActionProviderAttempt("tencent", "unsupported", "capability"),
        ),
    )
    return DatasetManifest(
        manifest_id=suffix * 64,
        instrument=f"SSE.{suffix * 6}",
        provider="tencent",
        content_hash=suffix * 64,
        relative_data_path=f"objects/{suffix * 64}.parquet",
        rows=len(sessions),
        created_at=NOW.isoformat(),
        provenance_json=provenance.to_json(),
    )


def test_calendar_extensions_are_compatible_on_their_common_session_range() -> None:
    earlier = _manifest(
        "a",
        (date(2026, 8, 7), date(2026, 8, 10), date(2026, 8, 11)),
        calendar_id="1" * 64,
        covered_to=date(2026, 8, 23),
    )
    extended = _manifest(
        "b",
        (date(2026, 8, 10), date(2026, 8, 11)),
        calendar_id="2" * 64,
        covered_to=date(2026, 8, 25),
    )

    result = validate_dataset_provenance(
        (earlier, extended),
        required_through=date(2026, 8, 11),
        failure_prefix="DECISION",
    )

    assert result.sessions == (date(2026, 8, 10), date(2026, 8, 11))


def test_calendar_validation_still_rejects_a_real_session_disagreement() -> None:
    expected = _manifest(
        "a",
        (date(2026, 8, 7), date(2026, 8, 10), date(2026, 8, 11)),
        calendar_id="1" * 64,
        covered_to=date(2026, 8, 23),
    )
    conflicting = _manifest(
        "b",
        (date(2026, 8, 7), date(2026, 8, 10), date(2026, 8, 12)),
        calendar_id="2" * 64,
        covered_to=date(2026, 8, 25),
    )

    with pytest.raises(LookupError, match="DECISION_CALENDAR_MISMATCH"):
        validate_dataset_provenance(
            (expected, conflicting),
            required_through=date(2026, 8, 10),
            failure_prefix="DECISION",
        )

