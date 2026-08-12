from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, DecimalException
import hashlib
import json
from math import isfinite
import re
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import delete, select

from compass.domain.market import InstrumentId
from compass.domain.trading import AccountSnapshot, Position
from compass.storage.database import Database
from compass.storage.models import AccountSnapshot as AccountSnapshotRow


SHANGHAI = ZoneInfo("Asia/Shanghai")
_ACCOUNT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")


class AccountIntegrityError(ValueError):
    """A stored manual-account snapshot failed its exact integrity checks."""


@dataclass(frozen=True, slots=True)
class StoredAccountSnapshot:
    row_id: int
    account_id: str
    captured_at: datetime
    content_hash: str
    snapshot: AccountSnapshot

    def __post_init__(self) -> None:
        if isinstance(self.row_id, bool) or not isinstance(self.row_id, int) or self.row_id <= 0:
            raise ValueError("row_id must be a positive integer")
        _account_id(self.account_id)
        _aware_datetime(self.captured_at, label="captured_at")
        if type(self.content_hash) is not str or _HASH.fullmatch(self.content_hash) is None:
            raise ValueError("content_hash must be a lowercase SHA-256 digest")
        if type(self.snapshot) is not AccountSnapshot:
            raise TypeError("snapshot must be an exact AccountSnapshot")


def _account_id(value: object) -> str:
    if type(value) is not str or _ACCOUNT_ID.fullmatch(value) is None:
        raise ValueError("account_id must be a stable non-empty identifier")
    assert isinstance(value, str)
    return value


def _aware_datetime(value: object, *, label: str) -> datetime:
    if type(value) is not datetime:
        raise TypeError(f"{label} must be an exact datetime")
    assert isinstance(value, datetime)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value


def _canonical_decimal(value: Decimal) -> str:
    sign, digits, exponent = value.as_tuple()
    if not any(digits):
        return "0"
    checked = list(digits)
    assert isinstance(exponent, int)
    while checked[-1] == 0:
        checked.pop()
        exponent += 1
    coefficient = "".join(str(digit) for digit in checked)
    prefix = "-" if sign else ""
    if exponent >= 0 and len(coefficient) + exponent <= 128:
        return f"{prefix}{coefficient}{'0' * exponent}"
    if exponent < 0:
        point = len(coefficient) + exponent
        if point > 0:
            return f"{prefix}{coefficient[:point]}.{coefficient[point:]}"
        leading_zeroes = -point
        if 2 + leading_zeroes + len(coefficient) <= 128:
            return f"{prefix}0.{'0' * leading_zeroes}{coefficient}"
    adjusted_exponent = exponent + len(coefficient) - 1
    mantissa = coefficient[0]
    if len(coefficient) > 1:
        mantissa = f"{mantissa}.{coefficient[1:]}"
    return f"{prefix}{mantissa}e{adjusted_exponent:+d}"


def _payload(snapshot: AccountSnapshot) -> dict[str, object]:
    return {
        "as_of": snapshot.as_of.isoformat(),
        "cash": _canonical_decimal(snapshot.cash),
        "positions": [
            {
                "instrument": str(position.instrument),
                "quantity": position.quantity,
                "available_quantity": position.available_quantity,
                "average_cost": _canonical_decimal(position.average_cost),
                "mark_price": _canonical_decimal(position.mark_price),
            }
            for position in snapshot.positions
        ],
        "schema_version": 1,
    }


def _payload_json(snapshot: AccountSnapshot) -> str:
    return json.dumps(
        _payload(snapshot), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _content_hash(payload_json: str) -> str:
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def _parse_exact_decimal(value: object, *, label: str) -> Decimal:
    if type(value) is not str:
        raise ValueError(f"{label} must be a canonical decimal string")
    assert isinstance(value, str)
    try:
        decimal_value = Decimal(value)
    except DecimalException:
        raise ValueError(f"{label} must be a canonical decimal string") from None
    if not decimal_value.is_finite() or _canonical_decimal(decimal_value) != value:
        raise ValueError(f"{label} must be a canonical decimal string")
    return decimal_value


def _decode(payload_json: str) -> AccountSnapshot:
    decoded: Any = json.loads(payload_json)
    if type(decoded) is not dict or set(decoded) != {
        "as_of",
        "cash",
        "positions",
        "schema_version",
    }:
        raise ValueError("account snapshot payload has an invalid shape")
    if type(decoded["schema_version"]) is not int or decoded["schema_version"] != 1:
        raise ValueError("account snapshot schema version is unsupported")
    if type(decoded["as_of"]) is not str:
        raise ValueError("account snapshot as_of must be an ISO date")
    as_of = date.fromisoformat(decoded["as_of"])
    raw_positions = decoded["positions"]
    if type(raw_positions) is not list:
        raise ValueError("account snapshot positions must be a list")
    positions: list[Position] = []
    for raw in raw_positions:
        if type(raw) is not dict or set(raw) != {
            "instrument",
            "quantity",
            "available_quantity",
            "average_cost",
            "mark_price",
        }:
            raise ValueError("account position payload has an invalid shape")
        if type(raw["instrument"]) is not str:
            raise ValueError("account position instrument must be canonical")
        positions.append(
            Position(
                instrument=InstrumentId.parse(raw["instrument"]),
                quantity=raw["quantity"],
                available_quantity=raw["available_quantity"],
                average_cost=_parse_exact_decimal(raw["average_cost"], label="average_cost"),
                mark_price=_parse_exact_decimal(raw["mark_price"], label="mark_price"),
            )
        )
    return AccountSnapshot(
        as_of=as_of,
        cash=_parse_exact_decimal(decoded["cash"], label="cash"),
        positions=positions,
    )


class AccountRepository:
    """Account-scoped append-only manual snapshot history."""

    def __init__(
        self,
        database: Database,
        account_id: str,
        clock: Callable[[], datetime],
    ) -> None:
        if type(database) is not Database:
            raise TypeError("database must be an exact Database")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._database = database
        self._account_id = _account_id(account_id)
        self._clock = clock

    @property
    def account_id(self) -> str:
        return self._account_id

    def save(self, snapshot: AccountSnapshot) -> StoredAccountSnapshot:
        if type(snapshot) is not AccountSnapshot:
            raise TypeError("snapshot must be an exact AccountSnapshot")
        captured_at = _aware_datetime(self._clock(), label="captured_at")
        if getattr(captured_at.tzinfo, "key", None) != SHANGHAI.key:
            raise ValueError("captured_at clock must use Asia/Shanghai")
        payload_json = _payload_json(snapshot)
        snapshot_hash = _content_hash(payload_json)
        market_value = snapshot.equity - snapshot.cash
        float_summaries = tuple(float(value) for value in (snapshot.cash, market_value, snapshot.equity))
        if not all(isfinite(value) for value in float_summaries):
            raise ValueError("account summary exceeds the supported bound")
        row = AccountSnapshotRow(
            account_id=self._account_id,
            captured_at=captured_at,
            as_of=snapshot.as_of.isoformat(),
            payload_json=payload_json,
            content_hash=snapshot_hash,
            cash=float_summaries[0],
            market_value=float_summaries[1],
            equity=float_summaries[2],
        )
        with self._database.session_factory() as session:
            latest = session.scalars(
                select(AccountSnapshotRow)
                .where(AccountSnapshotRow.account_id == self._account_id)
                .order_by(AccountSnapshotRow.id.desc())
                .limit(1)
            ).first()
            if latest is not None and latest.content_hash == snapshot_hash:
                existing = self._record(latest)
                if existing.snapshot != snapshot:
                    raise AccountIntegrityError(
                        f"ACCOUNT_SNAPSHOT_INTEGRITY:{self._account_id}:{latest.id}"
                    )
                return existing
            session.add(row)
            session.commit()
            session.refresh(row)
            return self._record(row)

    def compact_duplicates(self, protected_row_ids: frozenset[int]) -> int:
        if type(protected_row_ids) is not frozenset or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in protected_row_ids
        ):
            raise TypeError("protected snapshot ids must be positive integers")
        with self._database.session_factory.begin() as session:
            rows = session.scalars(
                select(AccountSnapshotRow)
                .where(AccountSnapshotRow.account_id == self._account_id)
                .order_by(AccountSnapshotRow.id.asc())
            ).all()
            grouped: dict[str, list[AccountSnapshotRow]] = {}
            for row in rows:
                self._record(row)
                grouped.setdefault(row.content_hash, []).append(row)
            delete_ids: list[int] = []
            for duplicates in grouped.values():
                protected = [row for row in duplicates if row.id in protected_row_ids]
                keep_ids = {row.id for row in protected}
                if not keep_ids:
                    keep_ids.add(duplicates[-1].id)
                delete_ids.extend(row.id for row in duplicates if row.id not in keep_ids)
            if delete_ids:
                session.execute(
                    delete(AccountSnapshotRow).where(AccountSnapshotRow.id.in_(delete_ids))
                )
            return len(delete_ids)

    def latest(self) -> StoredAccountSnapshot | None:
        with self._database.session_factory() as session:
            row = session.scalars(
                select(AccountSnapshotRow)
                .where(AccountSnapshotRow.account_id == self._account_id)
                .order_by(AccountSnapshotRow.id.desc())
                .limit(1)
            ).first()
            return None if row is None else self._record(row)

    def get(self, row_id: int) -> StoredAccountSnapshot | None:
        if isinstance(row_id, bool) or not isinstance(row_id, int) or row_id <= 0:
            raise ValueError("account snapshot id must be a positive integer")
        with self._database.session_factory() as session:
            row = session.get(AccountSnapshotRow, row_id)
            return None if row is None else self._record(row)

    def history(self) -> tuple[StoredAccountSnapshot, ...]:
        with self._database.session_factory() as session:
            rows = session.scalars(
                select(AccountSnapshotRow)
                .where(AccountSnapshotRow.account_id == self._account_id)
                .order_by(AccountSnapshotRow.id.asc())
            ).all()
            return tuple(self._record(row) for row in rows)

    def holding_since(self, as_of: date) -> dict[InstrumentId, date]:
        """Return the first date of each uninterrupted aggregated manual holding.

        Manual snapshots do not model lots.  Quantity changes therefore retain
        the earliest date in the current uninterrupted positive-position run;
        an omitted position closes that run, and a later reappearance starts a
        new one.  Same-day edits deterministically use insertion order.
        """

        if type(as_of) is not date:
            raise TypeError("holding age as_of must be an exact date")
        records = tuple(
            sorted(
                (item for item in self.history() if item.snapshot.as_of <= as_of),
                key=lambda item: (item.snapshot.as_of, item.row_id),
            )
        )
        starts: dict[InstrumentId, date] = {}
        previous: set[InstrumentId] = set()
        for record in records:
            current = {
                position.instrument
                for position in record.snapshot.positions
                if position.quantity > 0
            }
            for instrument in current:
                if instrument not in previous:
                    starts[instrument] = record.snapshot.as_of
            for instrument in previous - current:
                starts.pop(instrument, None)
            previous = current
        return dict(sorted(starts.items(), key=lambda item: str(item[0])))

    def _record(self, row: AccountSnapshotRow) -> StoredAccountSnapshot:
        try:
            if row.account_id != self._account_id:
                raise ValueError("account scope mismatch")
            if type(row.payload_json) is not str:
                raise ValueError("payload must be text")
            if _content_hash(row.payload_json) != row.content_hash:
                raise ValueError("payload hash mismatch")
            snapshot = _decode(row.payload_json)
            if _payload_json(snapshot) != row.payload_json:
                raise ValueError("payload is not canonical")
            if row.as_of != snapshot.as_of.isoformat():
                raise ValueError("as_of summary mismatch")
            summaries = (
                (row.cash, snapshot.cash),
                (row.market_value, snapshot.equity - snapshot.cash),
                (row.equity, snapshot.equity),
            )
            if any(not isfinite(stored) or stored != float(expected) for stored, expected in summaries):
                raise ValueError("accounting summary mismatch")
            return StoredAccountSnapshot(
                row_id=row.id,
                account_id=row.account_id,
                captured_at=row.captured_at,
                content_hash=row.content_hash,
                snapshot=snapshot,
            )
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise AccountIntegrityError(
                f"ACCOUNT_SNAPSHOT_INTEGRITY:{self._account_id}:{row.id}"
            ) from error
