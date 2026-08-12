from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import cast

import pandas as pd  # type: ignore[import-untyped]
from sqlalchemy import delete, select

from compass.backtest.snapshot import ManifestReference
from compass.domain.market import InstrumentId
from compass.services.safe_display import safe_identifier, stable_code
from compass.storage.canonical_json import (
    canonical_json,
    content_hash,
    decode_canonical_json,
)
from compass.storage.database import Database
from compass.storage.market_store import DatasetManifest, ManifestIntegrityError, MarketStore
from compass.storage.models import DatasetBundleRecord
from compass.storage.write_order import next_write_order


Clock = Callable[[], datetime]
IdFactory = Callable[[str], str]
_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    bundle_id: str
    primary_manifest_id: str
    instruments: tuple[InstrumentId, ...]
    market_manifests: tuple[ManifestReference, ...]
    data_quality: Mapping[str, object]
    created_at: datetime

    def __post_init__(self) -> None:
        safe_identifier(self.bundle_id, label="dataset bundle id")
        safe_identifier(self.primary_manifest_id, label="primary manifest id")
        instruments = tuple(self.instruments)
        manifests = tuple(self.market_manifests)
        if (
            not instruments
            or any(type(item) is not InstrumentId for item in instruments)
            or instruments != tuple(sorted(set(instruments), key=str))
        ):
            raise ValueError("bundle instruments must be non-empty, unique and sorted")
        if (
            len(manifests) != len(instruments)
            or any(type(item) is not ManifestReference for item in manifests)
            or len({item.manifest_id for item in manifests}) != len(manifests)
            or manifests != tuple(sorted(manifests, key=lambda item: item.manifest_id))
            or manifests[0].manifest_id != self.primary_manifest_id
        ):
            raise ValueError("bundle manifest references are inconsistent")
        if set(self.data_quality) != {"accepted", "issue_codes", "mode"}:
            raise ValueError("bundle data quality shape is invalid")
        if self.data_quality["accepted"] is not True:
            raise ValueError("saved bundle data quality must be accepted")
        if self.data_quality["mode"] not in {"strict", "degraded"}:
            raise ValueError("bundle data quality mode is invalid")
        issue_codes = self.data_quality["issue_codes"]
        if (
            type(issue_codes) is not tuple
            or any(type(item) is not str for item in issue_codes)
            or issue_codes != tuple(sorted(set(issue_codes)))
        ):
            raise ValueError("bundle issue codes must be a sorted immutable tuple")
        for code in issue_codes:
            stable_code(code, label="bundle issue code")
        if (
            type(self.created_at) is not datetime
            or self.created_at.tzinfo is None
            or self.created_at.utcoffset() is None
        ):
            raise ValueError("bundle created_at must be timezone-aware")
        object.__setattr__(self, "instruments", instruments)
        object.__setattr__(self, "market_manifests", manifests)
        object.__setattr__(
            self,
            "data_quality",
            MappingProxyType(
                {
                    "accepted": True,
                    "issue_codes": issue_codes,
                    "mode": self.data_quality["mode"],
                }
            ),
        )


class DatasetBundleRepository:
    def __init__(
        self,
        database: Database,
        market_store: MarketStore,
        *,
        clock: Clock,
        id_factory: IdFactory,
    ) -> None:
        self._database = database
        self._market_store = market_store
        self._clock = clock
        self._id_factory = id_factory

    def save(
        self,
        manifests: Sequence[tuple[InstrumentId, DatasetManifest]],
        *,
        mode: str,
        issue_codes: Sequence[str],
        replace_current: bool = False,
    ) -> DatasetBundle:
        if type(replace_current) is not bool:
            raise TypeError("replace_current must be an exact bool")
        checked_manifests: list[tuple[InstrumentId, DatasetManifest]] = []
        for instrument, supplied in manifests:
            if type(instrument) is not InstrumentId or type(supplied) is not DatasetManifest:
                raise ValueError("DATASET_BUNDLE_INTEGRITY")
            try:
                actual = self._market_store.load_manifest(supplied.manifest_id)
                if actual != supplied or InstrumentId.parse(actual.instrument) != instrument:
                    raise ValueError
            except (ManifestIntegrityError, OSError, TypeError, ValueError):
                raise ValueError("DATASET_BUNDLE_INTEGRITY") from None
            checked_manifests.append((instrument, actual))
        ordered_instruments = tuple(sorted(checked_manifests, key=lambda item: str(item[0])))
        if not ordered_instruments:
            raise ValueError("dataset bundle requires manifests")
        references = tuple(
            sorted(
                (
                    ManifestReference(item.manifest_id, item.content_hash)
                    for _, item in ordered_instruments
                ),
                key=lambda item: item.manifest_id,
            )
        )
        created_at = self._clock()
        if (
            type(created_at) is not datetime
            or created_at.tzinfo is None
            or created_at.utcoffset() is None
        ):
            raise ValueError("bundle clock must return a timezone-aware datetime")
        bundle = DatasetBundle(
            bundle_id=safe_identifier(
                self._id_factory("bundle"),
                label="dataset bundle id",
            ),
            primary_manifest_id=references[0].manifest_id,
            instruments=tuple(item[0] for item in ordered_instruments),
            market_manifests=references,
            data_quality={
                "accepted": True,
                "issue_codes": tuple(sorted(set(issue_codes))),
                "mode": mode,
            },
            created_at=created_at,
        )
        payload_json = canonical_json(self._payload(bundle))
        with self._database.session_factory.begin() as session:
            if replace_current:
                session.execute(delete(DatasetBundleRecord))
            session.add(
                DatasetBundleRecord(
                    bundle_id=bundle.bundle_id,
                    primary_manifest_id=bundle.primary_manifest_id,
                    schema_version=_SCHEMA_VERSION,
                    payload_json=payload_json,
                    content_hash=content_hash(payload_json),
                    created_at=bundle.created_at,
                    write_order=next_write_order(session),
                )
            )
        return bundle

    def latest(self) -> DatasetBundle | None:
        with self._database.session_factory() as session:
            row = session.scalars(
                select(DatasetBundleRecord)
                .order_by(
                    DatasetBundleRecord.write_order.desc(),
                )
                .limit(1)
            ).first()
            return None if row is None else self._bundle(row)

    def list(self) -> tuple[DatasetBundle, ...]:
        with self._database.session_factory() as session:
            rows = session.scalars(
                select(DatasetBundleRecord).order_by(
                    DatasetBundleRecord.write_order.desc(),
                )
            ).all()
            return tuple(self._bundle(row) for row in rows)

    def delete_referencing(self, instrument: InstrumentId | None = None) -> int:
        if instrument is not None and type(instrument) is not InstrumentId:
            raise TypeError("instrument must be an exact InstrumentId or None")
        bundles = self.list()
        selected = tuple(
            bundle.bundle_id
            for bundle in bundles
            if instrument is None or instrument in bundle.instruments
        )
        if not selected:
            return 0
        with self._database.session_factory.begin() as session:
            session.execute(
                delete(DatasetBundleRecord).where(DatasetBundleRecord.bundle_id.in_(selected))
            )
        return len(selected)

    def delete_except(self, bundle_id: str) -> int:
        checked = safe_identifier(bundle_id, label="dataset bundle id")
        with self._database.session_factory.begin() as session:
            selected = tuple(
                session.scalars(
                    select(DatasetBundleRecord.bundle_id).where(
                        DatasetBundleRecord.bundle_id != checked
                    )
                ).all()
            )
            if selected:
                session.execute(
                    delete(DatasetBundleRecord).where(DatasetBundleRecord.bundle_id.in_(selected))
                )
        return len(selected)

    def for_manifest(self, manifest_id: str) -> DatasetBundle:
        checked = safe_identifier(manifest_id, label="manifest id")
        with self._database.session_factory() as session:
            row = session.scalars(
                select(DatasetBundleRecord).where(
                    DatasetBundleRecord.primary_manifest_id == checked
                )
            ).one_or_none()
            if row is None:
                raise LookupError("DATASET_BUNDLE_MISSING")
            return self._bundle(row)

    def read_manifest(self, manifest_id: str) -> pd.DataFrame:
        return self._market_store.read_manifest(manifest_id)

    def load_manifest(self, manifest_id: str) -> DatasetManifest:
        return self._market_store.load_manifest(manifest_id)

    def references_by_instrument(
        self,
        bundle: DatasetBundle,
    ) -> Mapping[InstrumentId, ManifestReference]:
        if type(bundle) is not DatasetBundle:
            raise TypeError("bundle must be an exact DatasetBundle")
        references: dict[InstrumentId, ManifestReference] = {}
        for reference in bundle.market_manifests:
            manifest = self._market_store.load_manifest(reference.manifest_id)
            instrument = InstrumentId.parse(manifest.instrument)
            if (
                instrument not in bundle.instruments
                or instrument in references
                or manifest.content_hash != reference.content_hash
            ):
                raise ValueError("DATASET_BUNDLE_INTEGRITY")
            references[instrument] = reference
        if set(references) != set(bundle.instruments):
            raise ValueError("DATASET_BUNDLE_INTEGRITY")
        return MappingProxyType(dict(sorted(references.items(), key=lambda item: str(item[0]))))

    @staticmethod
    def _payload(bundle: DatasetBundle) -> dict[str, object]:
        return {
            "bundle_id": bundle.bundle_id,
            "created_at": bundle.created_at.isoformat(),
            "data_quality": {
                "accepted": bundle.data_quality["accepted"],
                "issue_codes": list(cast(tuple[str, ...], bundle.data_quality["issue_codes"])),
                "mode": bundle.data_quality["mode"],
            },
            "instruments": [str(item) for item in bundle.instruments],
            "manifests": [
                {
                    "content_hash": item.content_hash,
                    "manifest_id": item.manifest_id,
                }
                for item in bundle.market_manifests
            ],
            "primary_manifest_id": bundle.primary_manifest_id,
            "schema_version": _SCHEMA_VERSION,
        }

    def _bundle(self, row: DatasetBundleRecord) -> DatasetBundle:
        decoded = decode_canonical_json(row.payload_json, row.content_hash)
        if (
            type(row.schema_version) is not int
            or row.schema_version != _SCHEMA_VERSION
            or type(decoded.get("schema_version")) is not int
            or decoded["schema_version"] != _SCHEMA_VERSION
            or set(decoded)
            != {
                "bundle_id",
                "created_at",
                "data_quality",
                "instruments",
                "manifests",
                "primary_manifest_id",
                "schema_version",
            }
            or type(decoded["instruments"]) is not list
            or type(decoded["manifests"]) is not list
            or type(decoded["bundle_id"]) is not str
            or type(decoded["created_at"]) is not str
            or type(decoded["primary_manifest_id"]) is not str
            or type(decoded["data_quality"]) is not dict
        ):
            raise ValueError("DATASET_BUNDLE_INTEGRITY")
        try:
            instruments = tuple(
                InstrumentId.parse(cast(str, item))
                for item in cast(list[object], decoded["instruments"])
            )
            raw_manifests = cast(list[object], decoded["manifests"])
            if any(
                type(item) is not dict
                or set(item) != {"content_hash", "manifest_id"}
                or type(item["manifest_id"]) is not str
                or type(item["content_hash"]) is not str
                for item in raw_manifests
            ):
                raise TypeError
            references = tuple(
                ManifestReference(
                    cast(str, cast(dict[str, object], item)["manifest_id"]),
                    cast(str, cast(dict[str, object], item)["content_hash"]),
                )
                for item in raw_manifests
            )
            raw_quality = cast(dict[str, object], decoded["data_quality"])
            issue_values = raw_quality["issue_codes"]
            if (
                set(raw_quality) != {"accepted", "issue_codes", "mode"}
                or type(raw_quality["accepted"]) is not bool
                or type(raw_quality["mode"]) is not str
                or type(issue_values) is not list
                or any(type(item) is not str for item in issue_values)
            ):
                raise TypeError
            bundle = DatasetBundle(
                bundle_id=row.bundle_id,
                primary_manifest_id=row.primary_manifest_id,
                instruments=instruments,
                market_manifests=references,
                data_quality={
                    "accepted": raw_quality["accepted"],
                    "issue_codes": tuple(cast(list[str], issue_values)),
                    "mode": raw_quality["mode"],
                },
                created_at=datetime.fromisoformat(decoded["created_at"]),
            )
            if (
                bundle.bundle_id != decoded["bundle_id"]
                or bundle.primary_manifest_id != decoded["primary_manifest_id"]
                or bundle.created_at != row.created_at
            ):
                raise ValueError
            found_instruments = set()
            for reference in bundle.market_manifests:
                manifest = self._market_store.load_manifest(reference.manifest_id)
                if manifest.content_hash != reference.content_hash:
                    raise ValueError
                found_instruments.add(InstrumentId.parse(manifest.instrument))
            if found_instruments != set(bundle.instruments):
                raise ValueError
            return bundle
        except (KeyError, TypeError, ValueError):
            raise ValueError("DATASET_BUNDLE_INTEGRITY") from None
