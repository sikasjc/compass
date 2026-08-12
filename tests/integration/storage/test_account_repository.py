from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select, update

from compass.domain.market import InstrumentId
from compass.domain.trading import AccountSnapshot, Position
from compass.storage.account_repository import (
    AccountIntegrityError,
    AccountRepository,
    StoredAccountSnapshot,
    _content_hash,
    _decode,
    _payload_json,
)
from compass.storage.database import Database
from compass.storage.models import AccountSnapshot as AccountSnapshotRow


SHANGHAI = ZoneInfo("Asia/Shanghai")


def _clock(*values: datetime):
    remaining = iter(values)
    return lambda: next(remaining)


def _position(
    symbol: str = "SSE.510300",
    *,
    quantity: int = 100,
    available: int = 100,
    average_cost: str = "3.987654321",
    mark_price: str = "4.123456789",
) -> Position:
    return Position(
        instrument=InstrumentId.parse(symbol),
        quantity=quantity,
        available_quantity=available,
        average_cost=Decimal(average_cost),
        mark_price=Decimal(mark_price),
    )


def _snapshot(cash: str, *positions: Position) -> AccountSnapshot:
    return AccountSnapshot(
        as_of=date(2026, 7, 22),
        cash=Decimal(cash),
        positions=positions,
    )


def test_account_history_is_append_only_deterministic_and_account_scoped(tmp_path: Path) -> None:
    database = Database.sqlite_at(tmp_path / "accounts.db")
    database.create_schema()
    first_at = datetime(2026, 7, 22, 15, 1, tzinfo=SHANGHAI)
    second_at = datetime(2026, 7, 22, 15, 2, tzinfo=SHANGHAI)
    other_at = datetime(2026, 7, 22, 15, 3, tzinfo=SHANGHAI)
    main = AccountRepository(database, "main", _clock(first_at, second_at))
    other = AccountRepository(database, "other", _clock(other_at))

    first = main.save(_snapshot("1000.00", _position(quantity=100)))
    correction = main.save(_snapshot("900.00", _position(quantity=120)))
    other.save(_snapshot("88.00"))

    assert isinstance(first, StoredAccountSnapshot)
    assert first.row_id < correction.row_id
    assert first.account_id == correction.account_id == "main"
    assert first.captured_at == first_at
    assert main.history() == (first, correction)
    assert main.latest() == correction
    assert other.latest() is not None
    assert other.latest().snapshot.cash == Decimal("88.00")
    assert tuple(record.account_id for record in other.history()) == ("other",)


def test_account_repository_reuses_an_identical_latest_snapshot(tmp_path: Path) -> None:
    database = Database.sqlite_at(tmp_path / "accounts.db")
    database.create_schema()
    repository = AccountRepository(
        database,
        "main",
        _clock(
            datetime(2026, 7, 22, 15, 1, tzinfo=SHANGHAI),
            datetime(2026, 7, 22, 15, 2, tzinfo=SHANGHAI),
        ),
    )
    snapshot = _snapshot("1000.00", _position(quantity=100))

    first = repository.save(snapshot)
    repeated = repository.save(snapshot)

    assert repeated == first
    assert repository.history() == (first,)


def test_account_repository_compacts_duplicates_except_protected_rows(tmp_path: Path) -> None:
    database = Database.sqlite_at(tmp_path / "accounts.db")
    database.create_schema()
    repository = AccountRepository(
        database,
        "main",
        _clock(datetime(2026, 7, 22, 15, 1, tzinfo=SHANGHAI)),
    )
    snapshot = _snapshot("1000.00", _position(quantity=100))
    first = repository.save(snapshot)
    payload_json = _payload_json(snapshot)
    with database.session_factory.begin() as session:
        session.add(
            AccountSnapshotRow(
                account_id="main",
                captured_at=datetime(2026, 7, 22, 15, 2, tzinfo=SHANGHAI),
                as_of=snapshot.as_of.isoformat(),
                payload_json=payload_json,
                content_hash=_content_hash(payload_json),
                cash=1000.0,
                market_value=float(snapshot.equity - snapshot.cash),
                equity=float(snapshot.equity),
            )
        )
    duplicate = repository.latest()
    assert duplicate is not None and duplicate.row_id != first.row_id

    assert repository.compact_duplicates(frozenset({first.row_id})) == 1
    assert repository.history() == (first,)


def test_account_repository_restart_preserves_exact_decimals_and_defensive_copies(
    tmp_path: Path,
) -> None:
    path = tmp_path / "accounts.db"
    database = Database.sqlite_at(path)
    database.create_schema()
    caller_positions = [_position("SZSE.159915"), _position("SSE.510300")]
    snapshot = AccountSnapshot(date(2026, 7, 22), Decimal("1234.50"), caller_positions)
    repository = AccountRepository(
        database,
        "main",
        _clock(datetime(2026, 7, 22, 15, 5, tzinfo=SHANGHAI)),
    )

    saved = repository.save(snapshot)
    caller_positions.clear()
    restarted = Database.sqlite_at(path)
    restarted.create_schema()
    restored = AccountRepository(
        restarted,
        "main",
        _clock(datetime(2026, 7, 22, 15, 6, tzinfo=SHANGHAI)),
    ).latest()

    assert restored == saved
    assert restored is not None
    assert restored.snapshot.positions == (
        _position("SSE.510300"),
        _position("SZSE.159915"),
    )
    assert restored.snapshot.positions[0].average_cost == Decimal("3.987654321")
    assert restored.snapshot.positions[0].mark_price == Decimal("4.123456789")
    assert len(snapshot.positions) == 2
    assert not hasattr(repository, "update")
    assert not hasattr(repository, "delete")


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("payload_json", "not-json"),
        ("content_hash", "0" * 64),
        ("cash", 999.0),
        ("market_value", 999.0),
        ("equity", 999.0),
    ],
)
def test_account_repository_detects_payload_hash_and_summary_corruption(
    tmp_path: Path, column: str, value: object
) -> None:
    database = Database.sqlite_at(tmp_path / f"corrupt-{column}.db")
    database.create_schema()
    repository = AccountRepository(
        database,
        "main",
        _clock(datetime(2026, 7, 22, 15, 5, tzinfo=SHANGHAI)),
    )
    saved = repository.save(_snapshot("1000.00", _position()))
    with database.session_factory() as session:
        session.execute(
            update(AccountSnapshotRow)
            .where(AccountSnapshotRow.id == saved.row_id)
            .values({column: value})
        )
        session.commit()

    with pytest.raises(AccountIntegrityError, match="ACCOUNT_SNAPSHOT_INTEGRITY"):
        repository.latest()


def test_account_repository_missing_and_clock_validation(tmp_path: Path) -> None:
    database = Database.sqlite_at(tmp_path / "accounts.db")
    database.create_schema()
    assert AccountRepository(
        database,
        "missing",
        _clock(datetime(2026, 7, 22, 15, 0, tzinfo=SHANGHAI)),
    ).latest() is None
    repository = AccountRepository(
        database,
        "main",
        _clock(datetime(2026, 7, 22, 15, 0)),
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        repository.save(_snapshot("0.00"))


@pytest.mark.parametrize("schema_version", (True, 1.0))
def test_account_payload_decoder_rejects_noninteger_schema_version(
    schema_version: object,
) -> None:
    payload_json = json.dumps(
        {
            "as_of": "2026-07-22",
            "cash": "0",
            "positions": [],
            "schema_version": schema_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    with pytest.raises(ValueError, match="schema version"):
        _decode(payload_json)


@pytest.mark.parametrize(
    "snapshot",
    [
        lambda: AccountSnapshot(datetime(2026, 7, 22), Decimal("0.00"), ()),
        lambda: AccountSnapshot(date(2026, 7, 22), Decimal("0.001"), ()),
        lambda: AccountSnapshot(date(2026, 7, 22), Decimal("0.00"), [_position(), _position()]),
    ],
)
def test_account_snapshot_rejects_non_exact_or_impossible_accounting(snapshot) -> None:
    with pytest.raises((TypeError, ValueError)):
        snapshot()


@pytest.mark.parametrize(
    ("field", "value"),
    [("quantity", True), ("available_quantity", False), ("average_cost", 1.2), ("mark_price", 1)],
)
def test_position_rejects_non_exact_accounting_types(field: str, value: object) -> None:
    values: dict[str, object] = {
        "instrument": InstrumentId.parse("SSE.510300"),
        "quantity": 100,
        "available_quantity": 100,
        "average_cost": Decimal("4"),
        "mark_price": Decimal("4"),
    }
    values[field] = value
    with pytest.raises((TypeError, ValueError)):
        Position(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AccountSnapshot(date(2026, 7, 22), Decimal("1E+25"), ()),
        lambda: Position(
            InstrumentId.parse("SSE.510300"),
            100,
            100,
            Decimal("1E+25"),
            Decimal("1"),
        ),
        lambda: Position(
            InstrumentId.parse("SSE.510300"),
            10**16,
            10**16,
            Decimal("1"),
            Decimal("1"),
        ),
    ],
)
def test_account_values_outside_supported_exact_boundary_are_rejected(factory) -> None:
    with pytest.raises(ValueError, match="supported bound"):
        factory()


def test_tiny_exact_decimal_uses_compact_canonical_payload(tmp_path: Path) -> None:
    database = Database.sqlite_at(tmp_path / "compact.db")
    database.create_schema()
    repository = AccountRepository(
        database,
        "main",
        _clock(datetime(2026, 7, 22, 15, 5, tzinfo=SHANGHAI)),
    )
    tiny = Decimal("1E-100000")
    repository.save(
        _snapshot(
            "1000.00",
            _position(average_cost=str(tiny), mark_price="1"),
        )
    )
    with database.session_factory() as session:
        payload = session.scalar(select(AccountSnapshotRow.payload_json))

    assert payload is not None
    assert len(payload) < 500
    assert repository.latest().snapshot.positions[0].average_cost == tiny


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("cash",), "not-a-decimal"),
        (("as_of",), "not-an-iso-date"),
        (("positions", 0, "instrument"), "BAD"),
        (("positions", 0, "quantity"), True),
        (("positions", 0, "quantity"), 10**16),
        (("positions", 0, "average_cost"), "1.00"),
    ],
)
def test_hash_valid_malformed_payload_always_raises_repository_integrity_error(
    tmp_path: Path, path: tuple[object, ...], bad_value: object
) -> None:
    database = Database.sqlite_at(tmp_path / "malformed.db")
    database.create_schema()
    repository = AccountRepository(
        database,
        "main",
        _clock(datetime(2026, 7, 22, 15, 5, tzinfo=SHANGHAI)),
    )
    saved = repository.save(_snapshot("1000.00", _position()))
    with database.session_factory() as session:
        row = session.get(AccountSnapshotRow, saved.row_id)
        assert row is not None
        payload = json.loads(row.payload_json)
        target = payload
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = bad_value
        row.payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        row.content_hash = hashlib.sha256(row.payload_json.encode("utf-8")).hexdigest()
        session.commit()

    with pytest.raises(AccountIntegrityError, match="ACCOUNT_SNAPSHOT_INTEGRITY"):
        repository.latest()
    with pytest.raises(AccountIntegrityError, match="ACCOUNT_SNAPSHOT_INTEGRITY"):
        repository.history()
