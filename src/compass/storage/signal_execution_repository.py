from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, DecimalException
from enum import StrEnum
import json
import os
from pathlib import Path
from threading import RLock
from uuid import uuid4

from compass.services.safe_display import safe_identifier
from compass.storage.canonical_json import canonical_json, content_hash, decode_canonical_json


_SCHEMA_VERSION = 1


class SignalExecutionStatus(StrEnum):
    EXECUTED = "executed"
    PARTIAL = "partial"
    IGNORED = "ignored"


def _decimal(value: object, *, label: str) -> Decimal:
    if type(value) is not str:
        raise ValueError(f"{label} must be a canonical decimal string")
    try:
        parsed = Decimal(value)
    except DecimalException:
        raise ValueError(f"{label} must be a canonical decimal string") from None
    if not parsed.is_finite() or str(parsed.normalize()) != value:
        raise ValueError(f"{label} must be a canonical decimal string")
    return parsed


def _decimal_text(value: Decimal) -> str:
    if type(value) is not Decimal or not value.is_finite():
        raise TypeError("execution decimals must be finite and exact")
    return str(value.normalize())


@dataclass(frozen=True, slots=True)
class SignalExecutionFill:
    instrument: str
    quantity_delta: int
    execution_price: Decimal

    def __post_init__(self) -> None:
        safe_identifier(self.instrument, label="execution instrument")
        if isinstance(self.quantity_delta, bool) or not isinstance(self.quantity_delta, int):
            raise TypeError("execution quantity must be an integer")
        if self.quantity_delta == 0:
            raise ValueError("execution quantity must be non-zero")
        if (
            type(self.execution_price) is not Decimal
            or not self.execution_price.is_finite()
            or self.execution_price <= 0
        ):
            raise ValueError("execution price must be positive")


@dataclass(frozen=True, slots=True)
class SignalExecutionRecord:
    decision_id: str
    profile_id: str
    status: SignalExecutionStatus
    fills: tuple[SignalExecutionFill, ...]
    fees: Decimal
    recorded_at: datetime
    resulting_snapshot_row_id: int | None = None

    def __post_init__(self) -> None:
        safe_identifier(self.decision_id, label="execution decision id")
        safe_identifier(self.profile_id, label="execution profile id")
        if type(self.status) is not SignalExecutionStatus:
            raise TypeError("execution status must be exact")
        fills = tuple(self.fills)
        if any(type(item) is not SignalExecutionFill for item in fills):
            raise TypeError("execution fills must be exact")
        instruments = tuple(item.instrument for item in fills)
        if instruments != tuple(sorted(set(instruments))):
            raise ValueError("execution fills must be unique and sorted")
        if self.status is SignalExecutionStatus.IGNORED and fills:
            raise ValueError("ignored execution cannot contain fills")
        if self.status is not SignalExecutionStatus.IGNORED and not fills:
            raise ValueError("completed execution requires fills")
        if type(self.fees) is not Decimal or not self.fees.is_finite() or self.fees < 0:
            raise ValueError("execution fees must be non-negative")
        if type(self.recorded_at) is not datetime or self.recorded_at.tzinfo is None:
            raise ValueError("execution time must be timezone-aware")
        if self.status is SignalExecutionStatus.IGNORED:
            if self.resulting_snapshot_row_id is not None:
                raise ValueError("ignored execution cannot reference a snapshot")
        elif (
            isinstance(self.resulting_snapshot_row_id, bool)
            or not isinstance(self.resulting_snapshot_row_id, int)
            or self.resulting_snapshot_row_id <= 0
        ):
            raise ValueError("completed execution must reference a snapshot")
        object.__setattr__(self, "fills", fills)


class SignalExecutionRepository:
    def __init__(self, path: Path) -> None:
        if not isinstance(path, Path):
            raise TypeError("execution registry path must be a Path")
        self._path = path
        self._lock = RLock()
        if not path.exists():
            self._write(())

    def get(self, decision_id: str) -> SignalExecutionRecord | None:
        checked = safe_identifier(decision_id, label="execution decision id")
        return next((item for item in self.history() if item.decision_id == checked), None)

    def history(self) -> tuple[SignalExecutionRecord, ...]:
        with self._lock:
            return self._read()

    def save(self, record: SignalExecutionRecord) -> SignalExecutionRecord:
        if type(record) is not SignalExecutionRecord:
            raise TypeError("execution record must be exact")
        with self._lock:
            records = self._read()
            if any(item.decision_id == record.decision_id for item in records):
                raise ValueError("SIGNAL_EXECUTION_ALREADY_RECORDED")
            self._write(tuple(sorted((*records, record), key=lambda item: item.decision_id)))
            return record

    def delete(self, decision_ids: Sequence[str]) -> int:
        checked = frozenset(
            safe_identifier(item, label="execution decision id") for item in decision_ids
        )
        if not checked:
            return 0
        with self._lock:
            records = self._read()
            kept = tuple(item for item in records if item.decision_id not in checked)
            deleted = len(records) - len(kept)
            if deleted:
                self._write(kept)
            return deleted

    def _read(self) -> tuple[SignalExecutionRecord, ...]:
        try:
            text = self._path.read_text("utf-8")
            wrapper = json.loads(text)
            if type(wrapper) is not dict or set(wrapper) != {"content_hash", "payload_json"}:
                raise ValueError
            if canonical_json(wrapper) != text:
                raise ValueError
            payload = decode_canonical_json(wrapper["payload_json"], wrapper["content_hash"])
            if set(payload) != {"records", "schema_version"} or payload["schema_version"] != 1:
                raise ValueError
            raw_records = payload["records"]
            if type(raw_records) is not list:
                raise ValueError
            records = []
            for raw in raw_records:
                if type(raw) is not dict or set(raw) != {
                    "decision_id",
                    "fees",
                    "fills",
                    "profile_id",
                    "recorded_at",
                    "resulting_snapshot_row_id",
                    "status",
                }:
                    raise ValueError
                raw_fills = raw["fills"]
                if type(raw_fills) is not list:
                    raise ValueError
                fills = tuple(
                    SignalExecutionFill(
                        item["instrument"],
                        item["quantity_delta"],
                        _decimal(item["execution_price"], label="execution price"),
                    )
                    for item in raw_fills
                    if type(item) is dict
                    and set(item) == {"execution_price", "instrument", "quantity_delta"}
                )
                if len(fills) != len(raw_fills):
                    raise ValueError
                records.append(
                    SignalExecutionRecord(
                        raw["decision_id"],
                        raw["profile_id"],
                        SignalExecutionStatus(raw["status"]),
                        fills,
                        _decimal(raw["fees"], label="execution fees"),
                        datetime.fromisoformat(raw["recorded_at"]),
                        raw["resulting_snapshot_row_id"],
                    )
                )
            ordered = tuple(records)
            if tuple(item.decision_id for item in ordered) != tuple(
                sorted({item.decision_id for item in ordered})
            ):
                raise ValueError
            return ordered
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            raise ValueError("SIGNAL_EXECUTION_REGISTRY_INTEGRITY") from None

    def _write(self, records: tuple[SignalExecutionRecord, ...]) -> None:
        payload_json = canonical_json(
            {
                "records": [
                    {
                        "decision_id": record.decision_id,
                        "fees": _decimal_text(record.fees),
                        "fills": [
                            {
                                "execution_price": _decimal_text(fill.execution_price),
                                "instrument": fill.instrument,
                                "quantity_delta": fill.quantity_delta,
                            }
                            for fill in record.fills
                        ],
                        "profile_id": record.profile_id,
                        "recorded_at": record.recorded_at.isoformat(),
                        "resulting_snapshot_row_id": record.resulting_snapshot_row_id,
                        "status": record.status.value,
                    }
                    for record in records
                ],
                "schema_version": _SCHEMA_VERSION,
            }
        )
        document = canonical_json(
            {"content_hash": content_hash(payload_json), "payload_json": payload_json}
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(f".{self._path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(document, "utf-8")
            os.replace(temporary, self._path)
        finally:
            temporary.unlink(missing_ok=True)
