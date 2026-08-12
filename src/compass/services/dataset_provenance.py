from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
import json

from compass.domain.trading import CorporateAction
from compass.storage.market_store import DatasetManifest


@dataclass(frozen=True, slots=True)
class TrustedDatasetProvenance:
    fetched_at: datetime
    completed_through: date
    corporate_actions: tuple[CorporateAction, ...]
    sessions: tuple[date, ...]
    snapshot_payloads: tuple[Mapping[str, object], ...]


def validate_dataset_provenance(
    manifests: Sequence[DatasetManifest],
    *,
    required_through: date,
    failure_prefix: str,
    allow_quality_gaps: bool = False,
) -> TrustedDatasetProvenance:
    checked = tuple(sorted(manifests, key=lambda item: item.manifest_id))
    if not checked:
        raise LookupError(f"{failure_prefix}_PROVENANCE_MISSING")
    provenances = tuple(item.provenance for item in checked)
    if any(item is None for item in provenances):
        raise LookupError(f"{failure_prefix}_PROVENANCE_MISSING")
    resolved = tuple(item for item in provenances if item is not None)
    if any(not item.daily_complete for item in resolved):
        raise LookupError(f"{failure_prefix}_DAILY_DATA_INCOMPLETE")
    if not allow_quality_gaps and any(item.missing_sessions for item in resolved):
        raise LookupError(f"{failure_prefix}_DAILY_DATA_INCOMPLETE")
    if any(item.completed_through < required_through for item in resolved):
        raise LookupError(f"{failure_prefix}_DAILY_DATA_STALE")
    calendar_versions = {
        (item.calendar.provider, item.calendar.version) for item in resolved
    }
    if len(calendar_versions) != 1:
        raise LookupError(f"{failure_prefix}_CALENDAR_MISMATCH")
    common_start = max(item.calendar_sessions[0] for item in resolved)
    common_end = min(item.calendar_sessions[-1] for item in resolved)
    if common_start > common_end:
        raise LookupError(f"{failure_prefix}_CALENDAR_MISMATCH")
    common_calendars = {
        tuple(
            day
            for day in item.calendar_sessions
            if common_start <= day <= common_end
        )
        for item in resolved
    }
    if len(common_calendars) != 1:
        raise LookupError(f"{failure_prefix}_CALENDAR_MISMATCH")
    sessions = next(iter(common_calendars))
    if not sessions or sessions[-1] < required_through:
        raise LookupError(f"{failure_prefix}_DAILY_DATA_STALE")
    fetched_at = max(item.fetched_at for item in resolved)
    completed_through = min(item.completed_through for item in resolved)
    actions = tuple(
        sorted(
            {
                action
                for item in resolved
                for action in item.corporate_actions
            },
            key=lambda item: (
                str(item.instrument),
                item.ex_date,
                item.split_ratio,
                item.cash_dividend_per_share,
            ),
        )
    )
    payloads = tuple(
        {
            "instrument": manifest.instrument,
            "manifest_id": manifest.manifest_id,
            "provenance": json.loads(manifest.provenance_json or "null"),
        }
        for manifest in checked
    )
    return TrustedDatasetProvenance(
        fetched_at,
        completed_through,
        actions,
        sessions,
        payloads,
    )


def snapshot_data_quality(
    bundle_quality: Mapping[str, object],
    provenance: TrustedDatasetProvenance,
) -> Mapping[str, object]:
    return {
        **bundle_quality,
        "dataset_provenance": provenance.snapshot_payloads,
    }
