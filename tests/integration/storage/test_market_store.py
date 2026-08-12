from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from sqlalchemy import select

from compass.data.exchange_calendar import CalendarIdentity
from compass.storage.market_store import (
    CorporateActionProviderAttempt,
    DailyProviderAttempt,
    DatasetProvenance,
    ManifestIntegrityError,
    MarketStore,
)
from compass.storage.database import Database
from compass.storage.models import DatasetManifestRecord


def daily_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [4.0],
            "high": [4.2],
            "low": [3.9],
            "close": [4.1],
            "volume": [1000.0],
            "amount": [4100.0],
        },
        index=pd.DatetimeIndex(["2026-07-20"], name="date"),
    )


def records_for(store: MarketStore) -> list[DatasetManifestRecord]:
    with store.database.session_factory() as session:
        return list(session.scalars(select(DatasetManifestRecord)))


def quality_report_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "mode": "strict",
        "blocking": False,
        "accepted": True,
        "input_rows": 1,
        "output_rows": 1,
        "removed_sessions": [],
        "removed_rows": [],
        "issues": [],
    }
    payload.update(overrides)
    return payload


def quality_report_json(**overrides: object) -> str:
    return json.dumps(
        quality_report_payload(**overrides),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def forged_invalid_frame_report_json() -> str:
    return quality_report_json(
        issues=[
            {
                "code": "INVALID_FRAME",
                "message": "forged severity",
                "sessions": [],
                "blocking": False,
                "recoverable": True,
            }
        ]
    )


def provenance_json(*, fetched_at: datetime) -> str:
    return DatasetProvenance(
        daily_attempts=(DailyProviderAttempt("fixture", "selected"),),
        selected_provider="fixture",
        fetched_at=fetched_at,
        source_at=None,
        calendar=CalendarIdentity(
            "a" * 64,
            "fixture-calendar",
            "fixture-v1",
            date(2026, 7, 20),
            date(2026, 7, 20),
        ),
        completed_through=date(2026, 7, 20),
        daily_complete=True,
        corporate_actions_status="unavailable",
        corporate_actions_provider=None,
        corporate_actions=(),
        calendar_sessions=(date(2026, 7, 20),),
        missing_sessions=(),
        corporate_action_attempts=(
            CorporateActionProviderAttempt("fixture", "unsupported", "capability"),
        ),
    ).to_json()


def test_manifest_replays_exact_content_from_its_content_addressed_object(tmp_path: Path) -> None:
    bars = daily_bars()
    store = MarketStore(tmp_path)

    manifest = store.write_daily("SSE.510300", bars, provider="fixture")

    replay = store.read_manifest(manifest.manifest_id)
    raw_manifest = json.loads(store.manifest_path(manifest.manifest_id).read_text("utf-8"))
    object_path = tmp_path / raw_manifest["relative_data_path"]
    expected_hash = hashlib.sha256(object_path.read_bytes()).hexdigest()

    pd.testing.assert_frame_equal(replay, bars, check_freq=False)
    assert manifest.content_hash == expected_hash
    assert raw_manifest["content_hash"] == expected_hash
    assert raw_manifest["relative_data_path"] == f"objects/{expected_hash}.parquet"
    assert object_path.is_file()
    assert store.manifest_path(manifest.manifest_id).is_file()


def test_manifest_metadata_is_persisted_with_shanghai_timestamp(tmp_path: Path) -> None:
    store = MarketStore(tmp_path)

    manifest = store.write_daily("SSE.510300", daily_bars(), provider="fixture")

    [record] = records_for(store)
    created_at = pd.Timestamp(manifest.created_at)
    assert record.manifest_id == manifest.manifest_id
    assert record.instrument == "SSE.510300"
    assert record.provider == "fixture"
    assert record.content_hash == manifest.content_hash
    assert record.relative_data_path == manifest.relative_data_path
    assert record.rows == 1
    assert record.created_at.isoformat() == manifest.created_at
    assert created_at.utcoffset() == timedelta(hours=8)


def test_manifest_persists_optional_instrument_name(tmp_path: Path) -> None:
    store = MarketStore(tmp_path)

    named = store.write_daily(
        "SSE.510300",
        daily_bars(),
        provider="fixture",
        instrument_name="沪深300ETF",
    )
    unnamed = store.write_daily("SZSE.159915", daily_bars(), provider="fixture")

    assert store.load_manifest(named.manifest_id).instrument_name == "沪深300ETF"
    assert store.load_manifest(unnamed.manifest_id).instrument_name is None
    raw = json.loads(store.manifest_path(named.manifest_id).read_text("utf-8"))
    assert raw["instrument_name"] == "沪深300ETF"


def test_optional_quality_report_round_trips_through_manifest_and_database(tmp_path: Path) -> None:
    store = MarketStore(tmp_path)
    report_json = quality_report_json()

    manifest = store.write_daily(
        "SSE.510300",
        daily_bars(),
        provider="fixture",
        quality_report_json=report_json,
    )

    raw = json.loads(store.manifest_path(manifest.manifest_id).read_text("utf-8"))
    [record] = records_for(store)
    assert manifest.quality_report_json == report_json
    assert raw["quality_report_json"] == report_json
    assert record.quality_report_json == report_json
    pd.testing.assert_frame_equal(
        store.read_manifest(manifest.manifest_id), daily_bars(), check_freq=False
    )


@pytest.mark.parametrize("quality_report_json", ["not-json", "[]", "null"])
def test_invalid_quality_report_is_rejected_before_any_manifest_is_written(
    tmp_path: Path, quality_report_json: str
) -> None:
    store = MarketStore(tmp_path)

    with pytest.raises(ValueError, match="quality report"):
        store.write_daily(
            "SSE.510300",
            daily_bars(),
            provider="fixture",
            quality_report_json=quality_report_json,
        )

    assert list(store.data_dir.iterdir()) == []
    assert list(store.manifest_dir.iterdir()) == []


@pytest.mark.parametrize(
    "report_json",
    [
        '{"accepted":NaN}',
        json.dumps({"accepted": True}),
        quality_report_json(mode="permissive"),
        quality_report_json(mode=[]),
        quality_report_json(input_rows="1"),
        quality_report_json(blocking=True),
        quality_report_json(accepted=False),
        quality_report_json(output_rows=2),
        quality_report_json(input_rows=2, output_rows=1, removed_rows=[]),
    ],
)
def test_quality_report_schema_and_derived_values_are_strictly_validated_before_write(
    tmp_path: Path, report_json: str
) -> None:
    store = MarketStore(tmp_path)

    with pytest.raises(ValueError, match="quality report"):
        store.write_daily(
            "SSE.510300",
            daily_bars(),
            provider="fixture",
            quality_report_json=report_json,
        )

    assert list(store.data_dir.iterdir()) == []
    assert list(store.manifest_dir.iterdir()) == []


def test_quality_report_output_rows_must_equal_manifest_rows(tmp_path: Path) -> None:
    store = MarketStore(tmp_path)

    with pytest.raises(ValueError, match="quality report"):
        store.write_daily(
            "SSE.510300",
            daily_bars(),
            provider="fixture",
            quality_report_json=quality_report_json(input_rows=2, output_rows=2),
        )


def test_write_rejects_issue_flags_that_disagree_with_immutable_policy(tmp_path: Path) -> None:
    store = MarketStore(tmp_path)

    with pytest.raises(ValueError, match="quality report"):
        store.write_daily(
            "SSE.510300",
            daily_bars(),
            provider="fixture",
            quality_report_json=forged_invalid_frame_report_json(),
        )


@pytest.mark.parametrize(
    "tampered_report",
    [
        '{"accepted":Infinity}',
        json.dumps(quality_report_payload(mode="unknown")),
        json.dumps(quality_report_payload(), indent=2),
    ],
)
def test_read_revalidates_and_requires_canonical_quality_report(
    tmp_path: Path, tampered_report: str
) -> None:
    store = MarketStore(tmp_path)
    manifest = store.write_daily(
        "SSE.510300",
        daily_bars(),
        provider="fixture",
        quality_report_json=quality_report_json(),
    )
    manifest_path = store.manifest_path(manifest.manifest_id)
    raw = json.loads(manifest_path.read_text("utf-8"))
    raw["quality_report_json"] = tampered_report
    manifest_path.write_text(json.dumps(raw), "utf-8")

    with pytest.raises(ManifestIntegrityError, match="quality report"):
        store.read_manifest(manifest.manifest_id)


def test_read_rejects_forged_issue_flags_even_when_json_is_canonical(tmp_path: Path) -> None:
    store = MarketStore(tmp_path)
    manifest = store.write_daily(
        "SSE.510300",
        daily_bars(),
        provider="fixture",
        quality_report_json=quality_report_json(),
    )
    manifest_path = store.manifest_path(manifest.manifest_id)
    raw = json.loads(manifest_path.read_text("utf-8"))
    raw["quality_report_json"] = forged_invalid_frame_report_json()
    manifest_path.write_text(json.dumps(raw), "utf-8")

    with pytest.raises(ManifestIntegrityError, match="quality report"):
        store.read_manifest(manifest.manifest_id)


def test_manifest_identity_binds_canonical_provenance_metadata(tmp_path: Path) -> None:
    store = MarketStore(tmp_path)
    manifest = store.write_daily(
        "SSE.510300",
        daily_bars(),
        provider="fixture",
        provenance_json=provenance_json(
            fetched_at=datetime(2026, 7, 20, 15, 1, tzinfo=ZoneInfo("Asia/Shanghai"))
        ),
    )
    manifest_path = store.manifest_path(manifest.manifest_id)
    raw = json.loads(manifest_path.read_text("utf-8"))
    raw["provenance_json"] = provenance_json(
        fetched_at=datetime(2026, 7, 20, 15, 2, tzinfo=ZoneInfo("Asia/Shanghai"))
    )
    manifest_path.write_text(json.dumps(raw), "utf-8")

    with pytest.raises(ManifestIntegrityError, match="identity"):
        store.load_manifest(manifest.manifest_id)


def test_old_write_call_remains_compatible_without_a_quality_report(tmp_path: Path) -> None:
    store = MarketStore(tmp_path)

    manifest = store.write_daily("SSE.510300", daily_bars(), provider="fixture")

    assert manifest.quality_report_json is None
    assert records_for(store)[0].quality_report_json is None


def test_manifest_written_before_quality_field_existed_still_replays(tmp_path: Path) -> None:
    store = MarketStore(tmp_path)
    manifest = store.write_daily("SSE.510300", daily_bars(), provider="fixture")
    manifest_path = store.manifest_path(manifest.manifest_id)
    raw = json.loads(manifest_path.read_text("utf-8"))
    raw.pop("quality_report_json")
    manifest_path.write_text(json.dumps(raw), "utf-8")

    pd.testing.assert_frame_equal(
        store.read_manifest(manifest.manifest_id), daily_bars(), check_freq=False
    )


def test_injected_database_receives_manifest_metadata(tmp_path: Path) -> None:
    database = Database.sqlite_at(tmp_path / "shared-metadata.sqlite3")
    store = MarketStore(tmp_path / "market", database=database)

    manifest = store.write_daily("SSE.510300", daily_bars(), provider="fixture")

    [record] = records_for(store)
    assert record.manifest_id == manifest.manifest_id
    assert record.content_hash == manifest.content_hash


@pytest.mark.parametrize("manifest_id", ["../escape", "a/b", "A" * 32, "a" * 31])
def test_manifest_path_rejects_non_generated_identifiers(tmp_path: Path, manifest_id: str) -> None:
    store = MarketStore(tmp_path)

    with pytest.raises(ValueError, match="manifest id"):
        store.manifest_path(manifest_id)


def test_manifest_path_redirection_is_rejected(tmp_path: Path) -> None:
    store = MarketStore(tmp_path)
    manifest = store.write_daily("SSE.510300", daily_bars(), provider="fixture")
    manifest_path = store.manifest_path(manifest.manifest_id)
    raw_manifest = json.loads(manifest_path.read_text("utf-8"))
    replacement = daily_bars()
    replacement.loc[:, "close"] = 4.15
    replacement_path = tmp_path / "objects" / "replacement.parquet"
    replacement.to_parquet(replacement_path)
    raw_manifest["relative_data_path"] = "objects/replacement.parquet"
    manifest_path.write_text(json.dumps(raw_manifest), "utf-8")

    with pytest.raises(ManifestIntegrityError, match="data path"):
        store.read_manifest(manifest.manifest_id)


def test_manifest_rejects_object_hash_tampering(tmp_path: Path) -> None:
    store = MarketStore(tmp_path)
    manifest = store.write_daily("SSE.510300", daily_bars(), provider="fixture")
    object_path = tmp_path / manifest.relative_data_path
    object_path.write_bytes(b"tampered")

    with pytest.raises(ManifestIntegrityError, match="content hash"):
        store.read_manifest(manifest.manifest_id)


def test_manifest_rejects_row_count_tampering(tmp_path: Path) -> None:
    store = MarketStore(tmp_path)
    manifest = store.write_daily("SSE.510300", daily_bars(), provider="fixture")
    manifest_path = store.manifest_path(manifest.manifest_id)
    raw_manifest = json.loads(manifest_path.read_text("utf-8"))
    raw_manifest["rows"] = 2
    manifest_path.write_text(json.dumps(raw_manifest), "utf-8")

    with pytest.raises(ManifestIntegrityError, match="row count"):
        store.read_manifest(manifest.manifest_id)


def test_bool_and_string_optional_values_have_distinct_stored_objects(tmp_path: Path) -> None:
    boolean_bars = daily_bars().assign(suspended=[True])
    string_bars = daily_bars().assign(suspended=["True"])
    store = MarketStore(tmp_path)

    boolean_manifest = store.write_daily("SSE.510300", boolean_bars, provider="fixture")
    string_manifest = store.write_daily("SSE.510300", string_bars, provider="fixture")

    assert boolean_manifest.content_hash != string_manifest.content_hash
    pd.testing.assert_frame_equal(
        store.read_manifest(boolean_manifest.manifest_id), boolean_bars, check_freq=False
    )
    pd.testing.assert_frame_equal(
        store.read_manifest(string_manifest.manifest_id), string_bars, check_freq=False
    )


def test_manifest_write_failure_removes_new_object_manifest_and_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MarketStore(tmp_path)
    original_write_text = Path.write_text

    def fail_manifest_write(path: Path, *args: object, **kwargs: object) -> int:
        if path.parent == store.manifest_dir:
            raise OSError("manifest write failed")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_manifest_write)

    with pytest.raises(OSError, match="manifest write failed"):
        store.write_daily("SSE.510300", daily_bars(), provider="fixture")

    assert list(store.data_dir.iterdir()) == []
    assert list(store.manifest_dir.iterdir()) == []
    assert records_for(store) == []


def test_parquet_write_failure_leaves_no_published_files_or_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MarketStore(tmp_path)

    def fail_parquet(*args: object, **kwargs: object) -> None:
        raise OSError("parquet write failed")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_parquet)

    with pytest.raises(OSError, match="parquet write failed"):
        store.write_daily("SSE.510300", daily_bars(), provider="fixture")

    assert list(store.data_dir.iterdir()) == []
    assert list(store.manifest_dir.iterdir()) == []
    assert records_for(store) == []


def test_database_failure_rolls_back_new_manifest_but_keeps_existing_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MarketStore(tmp_path)
    existing = store.write_daily("SSE.510300", daily_bars(), provider="fixture")

    def fail_persist(*args: object, **kwargs: object) -> None:
        raise OSError("database write failed")

    monkeypatch.setattr(store, "_persist_manifest", fail_persist)

    with pytest.raises(OSError, match="database write failed"):
        store.write_daily("SSE.510300", daily_bars(), provider="different-provider")

    assert (tmp_path / existing.relative_data_path).is_file()
    assert list(store.manifest_dir.iterdir()) == [store.manifest_path(existing.manifest_id)]
    assert [record.manifest_id for record in records_for(store)] == [existing.manifest_id]
    assert not [path for path in store.data_dir.iterdir() if path.name.startswith(".")]
