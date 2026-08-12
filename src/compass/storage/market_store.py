from __future__ import annotations

from collections.abc import Sequence
import hashlib
from io import BytesIO
import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from threading import RLock
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd  # type: ignore[import-untyped]
from sqlalchemy import delete, select

from compass.domain.market import BarFrame
from compass.data.exchange_calendar import CalendarIdentity
from compass.domain.market import InstrumentId
from compass.domain.trading import CorporateAction
from compass.domain.quality_report import canonicalize_quality_report_json
from compass.storage.database import Database
from compass.storage.models import DatasetManifestRecord

_MANIFEST_ID = re.compile(r"(?:[0-9a-f]{32}|[0-9a-f]{64})\Z")
_CONTENT_HASH = re.compile(r"[0-9a-f]{64}\Z")
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_PROVENANCE_SCHEMA_VERSION = 2
_ATTEMPT_OUTCOMES = {"failed", "quality_rejected", "selected"}
_ACTION_ATTEMPT_OUTCOMES = {"failed", "selected", "unsupported"}


class ManifestIntegrityError(ValueError):
    """Raised when a manifest cannot safely reproduce its recorded dataset."""


@dataclass(frozen=True, slots=True)
class DailyProviderAttempt:
    provider: str
    outcome: str
    failure_category: str | None = None


@dataclass(frozen=True, slots=True)
class CorporateActionProviderAttempt:
    provider: str
    outcome: str
    failure_category: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetProvenance:
    daily_attempts: tuple[DailyProviderAttempt, ...]
    selected_provider: str
    fetched_at: datetime
    source_at: datetime | None
    calendar: CalendarIdentity
    completed_through: date
    daily_complete: bool
    corporate_actions_status: str
    corporate_actions_provider: str | None
    corporate_actions: tuple[CorporateAction, ...]
    calendar_sessions: tuple[date, ...]
    missing_sessions: tuple[date, ...]
    corporate_action_attempts: tuple[CorporateActionProviderAttempt, ...]

    def to_json(self) -> str:
        payload = {
            "calendar": {
                "calendar_id": self.calendar.calendar_id,
                "covered_from": self.calendar.covered_from.isoformat(),
                "covered_to": self.calendar.covered_to.isoformat(),
                "provider": self.calendar.provider,
                "sessions": [item.isoformat() for item in self.calendar_sessions],
                "version": self.calendar.version,
            },
            "completed_through": self.completed_through.isoformat(),
            "corporate_actions": {
                "attempts": [
                    {
                        "failure_category": attempt.failure_category,
                        "outcome": attempt.outcome,
                        "provider": attempt.provider,
                    }
                    for attempt in self.corporate_action_attempts
                ],
                "items": [
                    {
                        "cash_dividend_per_share": str(action.cash_dividend_per_share),
                        "ex_date": action.ex_date.isoformat(),
                        "instrument": str(action.instrument),
                        "split_ratio": str(action.split_ratio),
                    }
                    for action in self.corporate_actions
                ],
                "provider": self.corporate_actions_provider,
                "status": self.corporate_actions_status,
            },
            "daily_attempts": [
                {
                    "failure_category": attempt.failure_category,
                    "outcome": attempt.outcome,
                    "provider": attempt.provider,
                }
                for attempt in self.daily_attempts
            ],
            "daily_complete": self.daily_complete,
            "fetched_at": self.fetched_at.isoformat(),
            "missing_sessions": [item.isoformat() for item in self.missing_sessions],
            "schema_version": _PROVENANCE_SCHEMA_VERSION,
            "selected_provider": self.selected_provider,
            "source_at": None if self.source_at is None else self.source_at.isoformat(),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str, *, require_canonical: bool = True) -> DatasetProvenance:
        try:
            raw = json.loads(value)
            if type(raw) is not dict or set(raw) != {
                "calendar",
                "completed_through",
                "corporate_actions",
                "daily_attempts",
                "daily_complete",
                "fetched_at",
                "missing_sessions",
                "schema_version",
                "selected_provider",
                "source_at",
            }:
                raise ValueError
            if raw["schema_version"] != _PROVENANCE_SCHEMA_VERSION:
                raise ValueError
            attempts_raw = raw["daily_attempts"]
            if type(attempts_raw) is not list or not attempts_raw:
                raise ValueError
            attempts: list[DailyProviderAttempt] = []
            for item in attempts_raw:
                if type(item) is not dict or set(item) != {
                    "failure_category",
                    "outcome",
                    "provider",
                }:
                    raise ValueError
                if (
                    type(item["provider"]) is not str
                    or not item["provider"]
                    or item["outcome"] not in _ATTEMPT_OUTCOMES
                    or (
                        item["failure_category"] is not None
                        and (
                            type(item["failure_category"]) is not str
                            or not item["failure_category"]
                        )
                    )
                ):
                    raise ValueError
                attempts.append(
                    DailyProviderAttempt(
                        item["provider"], item["outcome"], item["failure_category"]
                    )
                )
            selected_provider = raw["selected_provider"]
            if (
                type(selected_provider) is not str
                or not selected_provider
                or attempts[-1] != DailyProviderAttempt(selected_provider, "selected", None)
            ):
                raise ValueError
            fetched_at = cls._moment(raw["fetched_at"])
            source_at = None if raw["source_at"] is None else cls._moment(raw["source_at"])
            if source_at is not None and source_at > fetched_at:
                raise ValueError
            calendar_raw = raw["calendar"]
            if type(calendar_raw) is not dict or set(calendar_raw) != {
                "calendar_id",
                "covered_from",
                "covered_to",
                "provider",
                "sessions",
                "version",
            }:
                raise ValueError
            calendar = CalendarIdentity(
                cls._text(calendar_raw["calendar_id"]),
                cls._text(calendar_raw["provider"]),
                cls._text(calendar_raw["version"]),
                date.fromisoformat(calendar_raw["covered_from"]),
                date.fromisoformat(calendar_raw["covered_to"]),
            )
            completed_through = date.fromisoformat(raw["completed_through"])
            if not calendar.covered_from <= completed_through <= calendar.covered_to:
                raise ValueError
            sessions_raw = calendar_raw["sessions"]
            if type(sessions_raw) is not list:
                raise ValueError
            calendar_sessions = tuple(date.fromisoformat(item) for item in sessions_raw)
            if (
                not calendar_sessions
                or tuple(sorted(set(calendar_sessions))) != calendar_sessions
                or calendar_sessions[0] < calendar.covered_from
                or calendar_sessions[-1] > calendar.covered_to
                or completed_through != calendar_sessions[-1]
            ):
                raise ValueError
            if type(raw["daily_complete"]) is not bool:
                raise ValueError
            missing_raw = raw["missing_sessions"]
            if type(missing_raw) is not list:
                raise ValueError
            missing_sessions = tuple(date.fromisoformat(item) for item in missing_raw)
            if (
                tuple(sorted(set(missing_sessions))) != missing_sessions
                or not set(missing_sessions).issubset(calendar_sessions)
                or raw["daily_complete"] != (completed_through not in missing_sessions)
            ):
                raise ValueError
            action_raw = raw["corporate_actions"]
            if type(action_raw) is not dict or set(action_raw) != {
                "attempts",
                "items",
                "provider",
                "status",
            }:
                raise ValueError
            status = action_raw["status"]
            provider = action_raw["provider"]
            if status not in {"available", "unavailable"}:
                raise ValueError
            if provider is not None:
                provider = cls._text(provider)
            if (status == "available") != (provider is not None):
                raise ValueError
            items = action_raw["items"]
            if type(items) is not list or (status == "unavailable" and items):
                raise ValueError
            actions = tuple(cls._action(item) for item in items)
            action_attempts_raw = action_raw["attempts"]
            if type(action_attempts_raw) is not list or not action_attempts_raw:
                raise ValueError
            action_attempts: list[CorporateActionProviderAttempt] = []
            for item in action_attempts_raw:
                if type(item) is not dict or set(item) != {
                    "failure_category",
                    "outcome",
                    "provider",
                }:
                    raise ValueError
                if (
                    type(item["provider"]) is not str
                    or not item["provider"]
                    or item["outcome"] not in _ACTION_ATTEMPT_OUTCOMES
                    or (
                        item["failure_category"] is not None
                        and (
                            type(item["failure_category"]) is not str
                            or not item["failure_category"]
                        )
                    )
                ):
                    raise ValueError
                action_attempts.append(
                    CorporateActionProviderAttempt(
                        item["provider"],
                        item["outcome"],
                        item["failure_category"],
                    )
                )
            if status == "available":
                if action_attempts[-1] != CorporateActionProviderAttempt(
                    provider, "selected", None
                ):
                    raise ValueError
            elif any(item.outcome == "selected" for item in action_attempts):
                raise ValueError
            provenance = cls(
                tuple(attempts),
                selected_provider,
                fetched_at,
                source_at,
                calendar,
                completed_through,
                raw["daily_complete"],
                status,
                provider,
                actions,
                calendar_sessions,
                missing_sessions,
                tuple(action_attempts),
            )
            if require_canonical and provenance.to_json() != value:
                raise ValueError
            return provenance
        except (
            AttributeError,
            InvalidOperation,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            raise ValueError("dataset provenance is invalid") from None

    @staticmethod
    def _text(value: object) -> str:
        if type(value) is not str or not value:
            raise ValueError
        return value

    @staticmethod
    def _moment(value: object) -> datetime:
        if type(value) is not str:
            raise ValueError
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError
        return parsed

    @classmethod
    def _action(cls, value: object) -> CorporateAction:
        if type(value) is not dict or set(value) != {
            "cash_dividend_per_share",
            "ex_date",
            "instrument",
            "split_ratio",
        }:
            raise ValueError
        return CorporateAction(
            InstrumentId.parse(cls._text(value["instrument"])),
            date.fromisoformat(value["ex_date"]),
            split_ratio=Decimal(cls._text(value["split_ratio"])),
            cash_dividend_per_share=Decimal(cls._text(value["cash_dividend_per_share"])),
        )


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    manifest_id: str
    instrument: str
    provider: str
    content_hash: str
    relative_data_path: str
    rows: int
    created_at: str
    quality_report_json: str | None = None
    provenance_json: str | None = None
    instrument_name: str | None = None

    @property
    def provenance(self) -> DatasetProvenance | None:
        if self.provenance_json is None:
            return None
        return DatasetProvenance.from_json(self.provenance_json)


class MarketStore:
    def __init__(self, root: Path, database: Database | None = None) -> None:
        self.root = root
        self.data_dir = root / "objects"
        self.manifest_dir = root / "manifests"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.database = database or Database.sqlite_at(root / "market_metadata.sqlite3")
        self.database.create_schema()
        self._write_lock = RLock()

    def manifest_path(self, manifest_id: str) -> Path:
        if _MANIFEST_ID.fullmatch(manifest_id) is None:
            raise ValueError("invalid manifest id")
        return self.manifest_dir / f"{manifest_id}.json"

    def write_daily(
        self,
        instrument: str,
        bars: pd.DataFrame,
        provider: str,
        *,
        quality_report_json: str | None = None,
        provenance_json: str | None = None,
        instrument_name: str | None = None,
    ) -> DatasetManifest:
        with self._write_lock:
            return self._write_daily_unlocked(
                instrument,
                bars,
                provider,
                quality_report_json=quality_report_json,
                provenance_json=provenance_json,
                instrument_name=instrument_name,
            )

    def _write_daily_unlocked(
        self,
        instrument: str,
        bars: pd.DataFrame,
        provider: str,
        *,
        quality_report_json: str | None = None,
        provenance_json: str | None = None,
        instrument_name: str | None = None,
    ) -> DatasetManifest:
        clean = BarFrame.validate(bars)
        clean.attrs.clear()
        quality_report_json = self._canonical_quality_report(
            quality_report_json, manifest_rows=len(clean)
        )
        if provenance_json is not None:
            provenance_json = DatasetProvenance.from_json(
                provenance_json, require_canonical=True
            ).to_json()
        instrument_name = self._instrument_name(instrument_name)
        object_path, content_hash, object_created = self._write_content_object(clean)
        created_at = datetime.now(_SHANGHAI).isoformat()
        unsigned_manifest = DatasetManifest(
            manifest_id="",
            instrument=instrument,
            provider=provider,
            content_hash=content_hash,
            relative_data_path=f"objects/{content_hash}.parquet",
            rows=len(clean),
            created_at=created_at,
            quality_report_json=quality_report_json,
            provenance_json=provenance_json,
            instrument_name=instrument_name,
        )
        manifest = DatasetManifest(
            manifest_id=self._manifest_identity(unsigned_manifest),
            instrument=unsigned_manifest.instrument,
            provider=unsigned_manifest.provider,
            content_hash=unsigned_manifest.content_hash,
            relative_data_path=unsigned_manifest.relative_data_path,
            rows=unsigned_manifest.rows,
            created_at=unsigned_manifest.created_at,
            quality_report_json=unsigned_manifest.quality_report_json,
            provenance_json=unsigned_manifest.provenance_json,
            instrument_name=unsigned_manifest.instrument_name,
        )
        manifest_written = False
        try:
            self._write_manifest_atomically(manifest)
            manifest_written = True
            self._persist_manifest(manifest)
        except Exception:
            if manifest_written:
                self.manifest_path(manifest.manifest_id).unlink(missing_ok=True)
            if object_created:
                object_path.unlink(missing_ok=True)
            raise
        return manifest

    @staticmethod
    def _canonical_quality_report(value: str | None, *, manifest_rows: int) -> str | None:
        if value is None:
            return None
        try:
            return canonicalize_quality_report_json(
                value,
                manifest_rows=manifest_rows,
                require_accepted=True,
            )
        except ValueError:
            raise ValueError("quality report is invalid") from None

    def read_manifest(self, manifest_id: str) -> pd.DataFrame:
        manifest = self._load_manifest(manifest_id)
        return self._read_loaded_manifest(manifest)

    def load_manifest(self, manifest_id: str) -> DatasetManifest:
        """Return metadata and validate the same immutable manifest/object snapshot."""

        manifest = self._load_manifest(manifest_id)
        self._read_loaded_manifest(manifest)
        return manifest

    def _read_loaded_manifest(self, manifest: DatasetManifest) -> pd.DataFrame:
        if len(manifest.manifest_id) == 32 and manifest.provenance_json is not None:
            raise ManifestIntegrityError("legacy manifest provenance is not identity-bound")
        if manifest.quality_report_json is not None:
            try:
                canonicalize_quality_report_json(
                    manifest.quality_report_json,
                    manifest_rows=manifest.rows,
                    require_accepted=True,
                    require_canonical=True,
                )
            except ValueError:
                raise ManifestIntegrityError("manifest quality report is invalid") from None
        if manifest.provenance_json is not None:
            try:
                DatasetProvenance.from_json(manifest.provenance_json, require_canonical=True)
            except ValueError:
                raise ManifestIntegrityError("manifest provenance is invalid") from None
        expected_path = f"objects/{manifest.content_hash}.parquet"
        if manifest.relative_data_path != expected_path:
            raise ManifestIntegrityError("manifest data path does not match its content hash")
        if _CONTENT_HASH.fullmatch(manifest.content_hash) is None:
            raise ManifestIntegrityError("manifest content hash is invalid")

        object_path = (self.root / manifest.relative_data_path).resolve()
        try:
            object_path.relative_to(self.data_dir.resolve())
        except ValueError as error:
            raise ManifestIntegrityError("manifest data path escapes the object store") from error
        try:
            object_payload = object_path.read_bytes()
        except OSError as error:
            raise ManifestIntegrityError("manifest object cannot be read") from error
        actual_hash = hashlib.sha256(object_payload).hexdigest()
        if actual_hash != manifest.content_hash:
            raise ManifestIntegrityError("manifest content hash does not match the object")

        try:
            replay = pd.read_parquet(BytesIO(object_payload))
        except Exception as error:
            raise ManifestIntegrityError("manifest object is not valid Parquet") from error
        if len(replay) != manifest.rows:
            raise ManifestIntegrityError("manifest row count does not match the object")
        try:
            validated = BarFrame.validate(replay)
        except ValueError as error:
            raise ManifestIntegrityError("manifest object is not a valid daily BarFrame") from error
        if (
            len(manifest.manifest_id) == 64
            and self._manifest_identity(manifest) != manifest.manifest_id
        ):
            raise ManifestIntegrityError("manifest identity does not bind its metadata")
        return validated

    @staticmethod
    def _manifest_identity(manifest: DatasetManifest) -> str:
        payload = asdict(manifest)
        payload.pop("manifest_id")
        # Manifests created before names were captured did not contain this key.
        if payload["instrument_name"] is None:
            payload.pop("instrument_name")
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _instrument_name(value: object) -> str | None:
        if value is None:
            return None
        if type(value) is not str:
            raise TypeError("instrument name must be exact text or None")
        assert isinstance(value, str)
        if not value or value != value.strip() or len(value) > 128:
            raise ValueError("instrument name must be non-empty trimmed text")
        if any(unicodedata.category(character).startswith("C") for character in value):
            raise ValueError("instrument name contains control characters")
        return value

    def _load_manifest(self, manifest_id: str) -> DatasetManifest:
        try:
            raw = json.loads(self.manifest_path(manifest_id).read_text("utf-8"))
            manifest = DatasetManifest(**raw)
        except (OSError, TypeError, json.JSONDecodeError) as error:
            raise ManifestIntegrityError("manifest cannot be decoded") from error
        if manifest.manifest_id != manifest_id:
            raise ManifestIntegrityError("manifest id does not match its filename")
        try:
            self._instrument_name(manifest.instrument_name)
        except (TypeError, ValueError):
            raise ManifestIntegrityError("manifest instrument name is invalid") from None
        return manifest

    def delete_market_data(self, instrument: InstrumentId | None = None) -> int:
        if instrument is not None and type(instrument) is not InstrumentId:
            raise TypeError("instrument must be an exact InstrumentId or None")
        with self._write_lock:
            return self._delete_market_data_unlocked(instrument)

    def _delete_market_data_unlocked(self, instrument: InstrumentId | None) -> int:
        with self.database.session_factory.begin() as session:
            statement = select(DatasetManifestRecord)
            if instrument is not None:
                statement = statement.where(DatasetManifestRecord.instrument == str(instrument))
            rows = session.scalars(statement).all()
            if not rows:
                return 0
            manifest_ids = tuple(row.manifest_id for row in rows)
            content_hashes = tuple(sorted({row.content_hash for row in rows}))
            session.execute(
                delete(DatasetManifestRecord).where(
                    DatasetManifestRecord.manifest_id.in_(manifest_ids)
                )
            )
            remaining_hashes = set(
                session.scalars(
                    select(DatasetManifestRecord.content_hash).where(
                        DatasetManifestRecord.content_hash.in_(content_hashes)
                    )
                ).all()
            )
        for manifest_id in manifest_ids:
            self.manifest_path(manifest_id).unlink(missing_ok=True)
        for content_hash in content_hashes:
            if content_hash not in remaining_hashes:
                (self.data_dir / f"{content_hash}.parquet").unlink(missing_ok=True)
        return len(manifest_ids)

    def prune_superseded(self, preferred_manifest_ids: Sequence[str]) -> int:
        preferred = tuple(preferred_manifest_ids)
        if any(
            type(manifest_id) is not str or _MANIFEST_ID.fullmatch(manifest_id) is None
            for manifest_id in preferred
        ) or len(set(preferred)) != len(preferred):
            raise ValueError("preferred manifest ids are invalid")
        with self._write_lock:
            return self._prune_superseded_unlocked(preferred)

    def _prune_superseded_unlocked(self, preferred: tuple[str, ...]) -> int:
        with self.database.session_factory.begin() as session:
            rows = session.scalars(
                select(DatasetManifestRecord).order_by(
                    DatasetManifestRecord.instrument,
                    DatasetManifestRecord.created_at.desc(),
                    DatasetManifestRecord.manifest_id.desc(),
                )
            ).all()
            by_id = {row.manifest_id: row for row in rows}
            if any(manifest_id not in by_id for manifest_id in preferred):
                raise LookupError("preferred manifest is missing")
            preferred_instruments = tuple(
                by_id[manifest_id].instrument for manifest_id in preferred
            )
            keep = set(preferred)
            seen = set(preferred_instruments)
            for row in rows:
                if row.instrument not in seen:
                    keep.add(row.manifest_id)
                    seen.add(row.instrument)
            selected = tuple(row for row in rows if row.manifest_id not in keep)
            if not selected:
                return 0
            manifest_ids = tuple(row.manifest_id for row in selected)
            content_hashes = tuple(sorted({row.content_hash for row in selected}))
            session.execute(
                delete(DatasetManifestRecord).where(
                    DatasetManifestRecord.manifest_id.in_(manifest_ids)
                )
            )
            remaining_hashes = set(
                session.scalars(
                    select(DatasetManifestRecord.content_hash).where(
                        DatasetManifestRecord.content_hash.in_(content_hashes)
                    )
                ).all()
            )
        for manifest_id in manifest_ids:
            self.manifest_path(manifest_id).unlink(missing_ok=True)
        for content_hash in content_hashes:
            if content_hash not in remaining_hashes:
                (self.data_dir / f"{content_hash}.parquet").unlink(missing_ok=True)
        return len(manifest_ids)

    def _write_content_object(self, bars: pd.DataFrame) -> tuple[Path, str, bool]:
        temporary = self.data_dir / f".{uuid4().hex}.parquet"
        try:
            bars.to_parquet(temporary)
            content_hash = hashlib.sha256(temporary.read_bytes()).hexdigest()
            destination = self.data_dir / f"{content_hash}.parquet"
            if destination.exists():
                return destination, content_hash, False
            os.replace(temporary, destination)
            return destination, content_hash, True
        finally:
            temporary.unlink(missing_ok=True)

    def _write_manifest_atomically(self, manifest: DatasetManifest) -> None:
        destination = self.manifest_path(manifest.manifest_id)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(asdict(manifest), ensure_ascii=False), "utf-8")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _persist_manifest(self, manifest: DatasetManifest) -> None:
        with self.database.session_factory.begin() as session:
            session.add(
                DatasetManifestRecord(
                    manifest_id=manifest.manifest_id,
                    instrument=manifest.instrument,
                    provider=manifest.provider,
                    content_hash=manifest.content_hash,
                    relative_data_path=manifest.relative_data_path,
                    rows=manifest.rows,
                    created_at=datetime.fromisoformat(manifest.created_at),
                    quality_report_json=manifest.quality_report_json,
                )
            )
