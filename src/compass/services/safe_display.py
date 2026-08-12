from __future__ import annotations

from collections.abc import Mapping
import re
from types import MappingProxyType
import unicodedata
from urllib.parse import parse_qsl, urlsplit

from compass.security import is_credential_key


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_STABLE_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,127}\Z")
_EXCEPTION_TYPE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,127}\Z")
_ERROR_FIELD = re.compile(r"[a-z][a-z0-9_.]{0,127}\Z")
_ASSIGNMENT = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?P<key>[A-Za-z][A-Za-z0-9]*(?:[-_ ]+[A-Za-z0-9]+)*)"
    r"\s*[:=]\s*\S+"
)
_BEARER = re.compile(r"(?i)\bbearer\s+\S+")
_ABSOLUTE_PATH = re.compile(
    r"(?i)(?<![A-Za-z0-9+.-])"
    r"(?:[A-Z]:[\\/]|\\\\|/(?:home|Users|var|tmp|etc|root|opt)(?:/|$))"
)
_URL = re.compile(r"https?://[^\s]+", re.IGNORECASE)


def _contains_credential_assignment(value: str) -> bool:
    for match in _ASSIGNMENT.finditer(value):
        key = match.group("key")
        if is_credential_key(key) or is_credential_key(key.rsplit(" ", 1)[-1]):
            return True
    return False


def safe_identifier(value: object, *, label: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a safe identifier")
    assert isinstance(value, str)
    if _contains_credential_assignment(value):
        raise ValueError(f"{label} contains credential text")
    return value


def stable_code(value: object, *, label: str) -> str:
    if type(value) is not str or _STABLE_CODE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a stable code")
    assert isinstance(value, str)
    return value


def safe_exception_type(value: object) -> str:
    if type(value) is not str or _EXCEPTION_TYPE.fullmatch(value) is None:
        raise ValueError("exception type must be a safe type name")
    assert isinstance(value, str)
    return value


def safe_display_text(value: object, *, label: str, maximum: int = 256) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be exact text")
    assert isinstance(value, str)
    if not value or value != value.strip() or len(value) > maximum:
        raise ValueError(f"{label} must be non-empty trimmed text")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError(f"{label} contains control characters")
    for candidate in _URL.findall(value):
        parsed = urlsplit(candidate)
        if parsed.username is not None or parsed.password is not None:
            raise ValueError(f"{label} contains URL credentials")
        if any(is_credential_key(key) for key, _ in parse_qsl(parsed.query)):
            raise ValueError(f"{label} contains URL credentials")
    if _contains_credential_assignment(value) or _BEARER.search(value):
        raise ValueError(f"{label} contains credential text")
    if _ABSOLUTE_PATH.search(value):
        raise ValueError(f"{label} contains an absolute path")
    return value


def frozen_errors(value: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("errors must be a mapping")
    copied = dict(value)
    for field, message in copied.items():
        if type(field) is not str or _ERROR_FIELD.fullmatch(field) is None:
            raise ValueError("error fields must be stable identifiers")
        safe_display_text(message, label="error message")
    return MappingProxyType(copied)
