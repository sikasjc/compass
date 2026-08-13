from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import json
import os
from pathlib import Path
from threading import RLock
from typing import cast
from uuid import uuid4

from compass.storage.canonical_json import canonical_json, content_hash, decode_canonical_json
from compass.strategies.rule_document import (
    RuleStrategyDraft,
    document_from_payload,
)


_SCHEMA_VERSION = 1


class StrategyDraftRepository:
    def __init__(self, path: Path) -> None:
        if not isinstance(path, Path):
            raise TypeError("strategy draft repository path must be a Path")
        self._path = path
        self._lock = RLock()
        if not path.exists():
            self._write(())

    def list(self) -> tuple[RuleStrategyDraft, ...]:
        with self._lock:
            return self._read()

    @staticmethod
    def new_draft_id() -> str:
        return f"draft-{uuid4().hex}"

    def get(self, draft_id: str) -> RuleStrategyDraft:
        with self._lock:
            value = next((item for item in self._read() if item.draft_id == draft_id), None)
            if value is None:
                raise LookupError("STRATEGY_DRAFT_UNKNOWN")
            return value

    def save(self, draft: RuleStrategyDraft) -> RuleStrategyDraft:
        if type(draft) is not RuleStrategyDraft:
            raise TypeError("strategy draft must be exact")
        with self._lock:
            values = [item for item in self._read() if item.draft_id != draft.draft_id]
            values.append(draft)
            self._write(tuple(values))
        return draft

    def delete(self, draft_id: str) -> bool:
        with self._lock:
            existing = self._read()
            updated = tuple(item for item in existing if item.draft_id != draft_id)
            if len(updated) == len(existing):
                return False
            self._write(updated)
            return True

    def _read(self) -> tuple[RuleStrategyDraft, ...]:
        try:
            outer = cast(dict[str, object], json.loads(self._path.read_text("utf-8")))
            if set(outer) != {"content_hash", "payload"}:
                raise ValueError
            payload_text = outer["payload"]
            expected_hash = outer["content_hash"]
            if type(payload_text) is not str or type(expected_hash) is not str:
                raise ValueError
            payload = decode_canonical_json(payload_text, expected_hash)
            if set(payload) != {"drafts", "schema_version"} or payload["schema_version"] != 1:
                raise ValueError
            raw_drafts = payload["drafts"]
            if type(raw_drafts) is not list:
                raise ValueError
            drafts = tuple(self._draft(item) for item in raw_drafts)
            if tuple(item.draft_id for item in drafts) != tuple(
                sorted({item.draft_id for item in drafts})
            ):
                raise ValueError
            return drafts
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            raise ValueError("STRATEGY_DRAFT_INTEGRITY") from None

    @staticmethod
    def _draft(value: object) -> RuleStrategyDraft:
        if not isinstance(value, Mapping) or set(value) != {
            "document",
            "draft_id",
            "pool_snapshot_id",
            "source_instance_id",
            "updated_at",
            "watchlist_id",
        }:
            raise ValueError
        document = value["document"]
        if not isinstance(document, Mapping):
            raise ValueError
        return RuleStrategyDraft(
            draft_id=cast(str, value["draft_id"]),
            watchlist_id=cast(str, value["watchlist_id"]),
            pool_snapshot_id=cast(str, value["pool_snapshot_id"]),
            document=document_from_payload(document),
            updated_at=datetime.fromisoformat(cast(str, value["updated_at"])),
            source_instance_id=cast(str | None, value["source_instance_id"]),
        )

    def _write(self, drafts: tuple[RuleStrategyDraft, ...]) -> None:
        ordered = tuple(sorted(drafts, key=lambda item: item.draft_id))
        payload = canonical_json(
            {
                "drafts": [
                    {
                        "document": draft.document.canonical_payload(),
                        "draft_id": draft.draft_id,
                        "pool_snapshot_id": draft.pool_snapshot_id,
                        "source_instance_id": draft.source_instance_id,
                        "updated_at": draft.updated_at.isoformat(),
                        "watchlist_id": draft.watchlist_id,
                    }
                    for draft in ordered
                ],
                "schema_version": _SCHEMA_VERSION,
            }
        )
        outer = canonical_json({"content_hash": content_hash(payload), "payload": payload})
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(f".{self._path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(outer, "utf-8")
            os.replace(temporary, self._path)
        finally:
            temporary.unlink(missing_ok=True)
