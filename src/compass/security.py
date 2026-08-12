from __future__ import annotations

import re


_CAMEL_BOUNDARY = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
)
_KEY_SEPARATOR = re.compile(r"[^a-zA-Z0-9]+")
_SENSITIVE_SEGMENTS = {
    "credential",
    "credentials",
    "passwd",
    "password",
    "sig",
    "signature",
    "auth",
    "authorization",
    "bearer",
    "sessionid",
    "token",
    "secret",
    "key",
}
_SENSITIVE_COMPACT_KEYS = {
    "credential",
    "credentials",
    "passwd",
    "password",
    "session",
    "sessionid",
    "sig",
    "signature",
    "auth",
    "authorization",
    "bearer",
    "accesstoken",
    "token",
    "secret",
    "apikey",
    "key",
}


def normalize_credential_key(value: str) -> tuple[str, ...]:
    """Return case-folded key segments across camel and connector spellings."""

    camel_split = _CAMEL_BOUNDARY.sub("_", value)
    return tuple(
        segment
        for segment in _KEY_SEPARATOR.split(camel_split.casefold())
        if segment
    )


def is_credential_key(value: str) -> bool:
    """Return whether a normalized field name denotes credential material."""

    segments = normalize_credential_key(value)
    if not segments:
        return False
    compact = "".join(segments)
    if compact in _SENSITIVE_COMPACT_KEYS:
        return True
    segment_set = set(segments)
    if segment_set.intersection(_SENSITIVE_SEGMENTS):
        return True
    return {"session", "id"}.issubset(segment_set)
