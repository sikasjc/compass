from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from typing import Any


class CanonicalJsonIntegrityError(ValueError):
    """Stored local JSON was malformed, non-canonical, or changed after writing."""


def canonical_json(value: Mapping[str, object]) -> str:
    if not isinstance(value, Mapping):
        raise TypeError("canonical JSON root must be a mapping")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_hash(payload_json: str) -> str:
    if type(payload_json) is not str:
        raise TypeError("payload JSON must be exact text")
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def decode_canonical_json(
    payload_json: str,
    expected_hash: str,
) -> dict[str, object]:
    if type(payload_json) is not str or type(expected_hash) is not str:
        raise CanonicalJsonIntegrityError("LOCAL_JSON_INTEGRITY")
    if content_hash(payload_json) != expected_hash:
        raise CanonicalJsonIntegrityError("LOCAL_JSON_INTEGRITY")
    try:
        decoded: Any = json.loads(payload_json)
    except json.JSONDecodeError:
        raise CanonicalJsonIntegrityError("LOCAL_JSON_INTEGRITY") from None
    if type(decoded) is not dict:
        raise CanonicalJsonIntegrityError("LOCAL_JSON_INTEGRITY")
    if canonical_json(decoded) != payload_json:
        raise CanonicalJsonIntegrityError("LOCAL_JSON_INTEGRITY")
    return decoded
