from __future__ import annotations

from sqlalchemy.orm import Session

from compass.storage.models import LocalInsertionSequence


def next_write_order(session: Session) -> int:
    """Allocate one transactionally persisted SQLite insertion identity."""

    sequence = LocalInsertionSequence()
    session.add(sequence)
    session.flush()
    return sequence.id
