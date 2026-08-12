from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from types import MappingProxyType
from typing import NoReturn


_REPORT_FIELDS = {
    "mode",
    "blocking",
    "accepted",
    "input_rows",
    "output_rows",
    "removed_sessions",
    "removed_rows",
    "issues",
}
_ISSUE_FIELDS = {"code", "message", "sessions", "blocking", "recoverable"}
_REMOVED_ROW_FIELDS = {"position", "timestamp", "raw_index_text", "reason_codes"}


@dataclass(frozen=True, slots=True)
class IssuePolicy:
    blocking: bool
    recoverable: bool
    allowed_removed_reason: bool


ISSUE_POLICY = MappingProxyType(
    {
        "INVALID_FRAME": IssuePolicy(True, False, False),
        "DUPLICATE_COLUMN": IssuePolicy(True, False, True),
        "MISSING_COLUMN": IssuePolicy(True, False, True),
        "INVALID_INDEX": IssuePolicy(True, False, True),
        "TIMEZONE_AWARE_INDEX": IssuePolicy(True, False, True),
        "NON_NORMALIZED_SESSION": IssuePolicy(True, True, True),
        "DUPLICATE_SESSION": IssuePolicy(True, True, True),
        "OUT_OF_ORDER": IssuePolicy(True, True, False),
        "UNEXPECTED_SESSION": IssuePolicy(True, True, True),
        "MISSING_SESSION": IssuePolicy(True, True, False),
        "INVALID_NUMERIC_TYPE": IssuePolicy(True, True, True),
        "NONFINITE_VALUE": IssuePolicy(True, True, True),
        "NONPOSITIVE_PRICE": IssuePolicy(True, True, True),
        "NEGATIVE_ACTIVITY": IssuePolicy(True, True, True),
        "INVALID_OHLC": IssuePolicy(True, True, True),
        "INVALID_ADJUST_FACTOR": IssuePolicy(True, True, True),
        "INVALID_SUSPENDED": IssuePolicy(True, True, True),
        "INVALID_LIMIT_PRICE": IssuePolicy(True, True, True),
        "INVALID_LIMIT_RELATION": IssuePolicy(True, True, True),
        "INVALID_ADJUST_FLAG": IssuePolicy(True, True, True),
        "LARGE_PRICE_JUMP": IssuePolicy(False, True, False),
        "ADJUST_FACTOR_DISCONTINUITY": IssuePolicy(False, True, False),
        "EMPTY_DATASET": IssuePolicy(True, True, False),
    }
)
QUALITY_ISSUE_CODES = tuple(ISSUE_POLICY)
_ISSUE_ORDER = {code: position for position, code in enumerate(QUALITY_ISSUE_CODES)}


def _invalid() -> NoReturn:
    raise ValueError("quality report schema is invalid") from None


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _string_list(value: object, *, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        _invalid()
    result = value
    if not allow_empty and not result:
        _invalid()
    if result != sorted(set(result)):
        _invalid()
    return result


def _reject_constant(value: str) -> NoReturn:
    del value
    _invalid()


def canonicalize_quality_report_json(
    value: str,
    *,
    manifest_rows: int | None = None,
    require_accepted: bool = False,
    require_canonical: bool = False,
) -> str:
    """Validate a persisted quality report and return its canonical JSON form."""

    try:
        payload = json.loads(value, parse_constant=_reject_constant)
    except (TypeError, json.JSONDecodeError, ValueError):
        _invalid()
    if not isinstance(payload, dict) or set(payload) != _REPORT_FIELDS:
        _invalid()

    mode = payload["mode"]
    blocking = payload["blocking"]
    accepted = payload["accepted"]
    input_rows = payload["input_rows"]
    output_rows = payload["output_rows"]
    if not isinstance(mode, str) or mode not in {"strict", "degraded"}:
        _invalid()
    if type(blocking) is not bool or type(accepted) is not bool:
        _invalid()
    if not _is_int(input_rows) or not _is_int(output_rows):
        _invalid()
    if input_rows < 0 or output_rows < 0 or output_rows > input_rows:
        _invalid()
    if manifest_rows is not None and output_rows != manifest_rows:
        _invalid()

    raw_issues = payload["issues"]
    if not isinstance(raw_issues, list):
        _invalid()
    issue_codes: set[str] = set()
    ordered_issue_codes: list[str] = []
    derived_blocking = False
    all_recoverable = True
    for issue in raw_issues:
        if not isinstance(issue, dict) or set(issue) != _ISSUE_FIELDS:
            _invalid()
        code = issue["code"]
        message = issue["message"]
        issue_blocking = issue["blocking"]
        recoverable = issue["recoverable"]
        if not isinstance(code, str) or code not in _ISSUE_ORDER or code in issue_codes:
            _invalid()
        if not isinstance(message, str) or not message:
            _invalid()
        if type(issue_blocking) is not bool or type(recoverable) is not bool:
            _invalid()
        policy = ISSUE_POLICY[code]
        if issue_blocking is not policy.blocking or recoverable is not policy.recoverable:
            _invalid()
        _string_list(issue["sessions"])
        issue_codes.add(code)
        ordered_issue_codes.append(code)
        derived_blocking = derived_blocking or issue_blocking
        all_recoverable = all_recoverable and recoverable
    if ordered_issue_codes != sorted(ordered_issue_codes, key=_ISSUE_ORDER.__getitem__):
        _invalid()

    raw_removed_rows = payload["removed_rows"]
    if not isinstance(raw_removed_rows, list):
        _invalid()
    positions: list[int] = []
    derived_removed_sessions: set[str] = set()
    for row in raw_removed_rows:
        if not isinstance(row, dict) or set(row) != _REMOVED_ROW_FIELDS:
            _invalid()
        position = row["position"]
        timestamp = row["timestamp"]
        raw_index_text = row["raw_index_text"]
        if not _is_int(position) or position < 0 or position >= input_rows:
            _invalid()
        if not isinstance(raw_index_text, str) or not raw_index_text:
            _invalid()
        parsed_timestamp: datetime | None = None
        if timestamp is not None:
            if not isinstance(timestamp, str) or "T" not in timestamp:
                _invalid()
            try:
                parsed_timestamp = datetime.fromisoformat(timestamp)
            except (OverflowError, ValueError):
                _invalid()
        elif raw_index_text != "NaT":
            _invalid()
        raw_reasons = row["reason_codes"]
        if not isinstance(raw_reasons, list) or not raw_reasons:
            _invalid()
        if not all(isinstance(reason, str) and reason in _ISSUE_ORDER for reason in raw_reasons):
            _invalid()
        reasons = raw_reasons
        if reasons != sorted(set(reasons), key=_ISSUE_ORDER.__getitem__):
            _invalid()
        if not set(reasons).issubset(issue_codes):
            _invalid()
        if not all(ISSUE_POLICY[reason].allowed_removed_reason for reason in reasons):
            _invalid()
        positions.append(position)
        if parsed_timestamp is not None:
            derived_removed_sessions.add(parsed_timestamp.date().isoformat())
    if positions != sorted(set(positions)):
        _invalid()
    if len(raw_removed_rows) != input_rows - output_rows:
        _invalid()

    removed_sessions = _string_list(payload["removed_sessions"])
    if removed_sessions != sorted(derived_removed_sessions):
        _invalid()
    if blocking is not derived_blocking:
        _invalid()
    derived_accepted = (
        output_rows > 0 and all_recoverable and (mode == "degraded" or not derived_blocking)
    )
    if accepted is not derived_accepted or (require_accepted and not accepted):
        _invalid()
    if "EMPTY_DATASET" in issue_codes:
        if input_rows != 0 or output_rows != 0 or issue_codes != {"EMPTY_DATASET"}:
            _invalid()

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if require_canonical and value != canonical:
        _invalid()
    return canonical
