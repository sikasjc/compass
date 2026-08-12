from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
import json
from math import isfinite
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from compass.domain.market import BarFrame
from compass.domain.quality_report import ISSUE_POLICY, QUALITY_ISSUE_CODES


class QualityMode(StrEnum):
    STRICT = "strict"
    DEGRADED = "degraded"

    @classmethod
    def parse(cls, value: QualityMode | str) -> QualityMode:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("quality mode must be QualityMode or text")
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"unknown quality mode: {value!r}") from None


@dataclass(frozen=True, slots=True)
class QualityIssue:
    code: str
    message: str
    sessions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.code not in ISSUE_POLICY:
            raise ValueError("unknown quality issue code")

    @property
    def blocking(self) -> bool:
        return ISSUE_POLICY[self.code].blocking

    @property
    def recoverable(self) -> bool:
        return ISSUE_POLICY[self.code].recoverable

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "sessions": list(self.sessions),
            "blocking": self.blocking,
            "recoverable": self.recoverable,
        }


@dataclass(frozen=True, slots=True)
class RemovedRow:
    position: int
    timestamp: str | None
    raw_index_text: str
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "position": self.position,
            "timestamp": self.timestamp,
            "raw_index_text": self.raw_index_text,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class QualityReport:
    mode: QualityMode
    issues: tuple[QualityIssue, ...]
    input_rows: int
    output_rows: int
    removed_sessions: tuple[str, ...]
    removed_rows: tuple[RemovedRow, ...]
    _frame: pd.DataFrame = field(repr=False, compare=False)

    @property
    def blocking(self) -> bool:
        return any(issue.blocking for issue in self.issues)

    @property
    def accepted(self) -> bool:
        if self.output_rows == 0:
            return False
        if any(not issue.recoverable for issue in self.issues):
            return False
        return self.mode is QualityMode.DEGRADED or not self.blocking

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame.copy()

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "blocking": self.blocking,
            "accepted": self.accepted,
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "removed_sessions": list(self.removed_sessions),
            "removed_rows": [row.to_dict() for row in self.removed_rows],
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


class DataQualityError(RuntimeError):
    def __init__(
        self,
        provider: str,
        report: QualityReport,
        attempts: tuple[str, ...] = (),
    ) -> None:
        self.provider = provider
        self.report = report
        self.attempts = attempts
        codes = ", ".join(issue.code for issue in report.issues) or "EMPTY_DATASET"
        super().__init__(f"{provider}: daily data failed quality gate ({codes})")


_ISSUE_ORDER = {code: position for position, code in enumerate(QUALITY_ISSUE_CODES)}


def _session_text(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return str(value)


def _sessions(index: pd.Index[Any]) -> tuple[str, ...]:
    return tuple(sorted({_session_text(value) for value in index}))


def _finite_number(value: object) -> bool:
    if not _canonical_number(value):
        return False
    try:
        return bool(isfinite(cast(Any, value)))
    except (OverflowError, TypeError, ValueError):
        return False


def _canonical_number(value: object) -> bool:
    if isinstance(
        value,
        (bool, complex, Decimal, np.bool_, np.complexfloating, np.longdouble),
    ):
        return False
    if type(value) is int:
        return -(2**63) <= value <= 2**63 - 1
    if type(value) is float:
        return True
    if isinstance(value, np.integer):
        return value.dtype.itemsize <= 8 and -(2**63) <= int(value) <= 2**63 - 1
    if isinstance(value, np.floating):
        return value.dtype.itemsize <= 8
    return False


def _positive_number(value: object) -> bool:
    if not _finite_number(value):
        return False
    try:
        return bool(value > 0)  # type: ignore[operator]
    except (OverflowError, TypeError, ValueError):
        return False


def _audit_index(value: object) -> tuple[str | None, str]:
    if isinstance(value, pd.Timestamp) and not pd.isna(value):
        return str(value.isoformat()), str(value)
    return None, str(value)


def _removed_sessions(rows: tuple[RemovedRow, ...]) -> tuple[str, ...]:
    return tuple(sorted({row.timestamp[:10] for row in rows if row.timestamp is not None}))


class DailyQualityGate:
    """Evaluate canonical daily bars without manufacturing or imputing values."""

    def __init__(self, *, expected_sessions: pd.DatetimeIndex) -> None:
        if not isinstance(expected_sessions, pd.DatetimeIndex):
            raise TypeError("expected sessions must be a DatetimeIndex")
        if expected_sessions.tz is not None:
            raise ValueError("expected sessions must be timezone-naive trading dates")
        if not (expected_sessions == expected_sessions.normalize()).all():
            raise ValueError("expected sessions must be normalized trading dates")
        if expected_sessions.has_duplicates:
            raise ValueError("expected sessions must be unique")
        if not expected_sessions.is_monotonic_increasing:
            raise ValueError("expected sessions must be sorted")
        self.expected_sessions = expected_sessions.copy()

    def evaluate(
        self, frame: pd.DataFrame, mode: QualityMode | str = QualityMode.STRICT
    ) -> QualityReport:
        quality_mode = QualityMode.parse(mode)
        if not isinstance(frame, pd.DataFrame):
            issue = QualityIssue(
                "INVALID_FRAME",
                "daily data must be a DataFrame",
            )
            return self._report(quality_mode, (issue,), 0, pd.DataFrame(), (), ())

        input_rows = len(frame)
        if frame.columns.has_duplicates:
            issue = QualityIssue(
                "DUPLICATE_COLUMN",
                "daily data columns must be unique",
            )
            removed_rows = self._all_rows(frame.index, "DUPLICATE_COLUMN")
            return self._report(
                quality_mode,
                (issue,),
                input_rows,
                pd.DataFrame(),
                _removed_sessions(removed_rows),
                removed_rows,
            )
        missing_columns = sorted(set(BarFrame.REQUIRED_COLUMNS).difference(frame.columns))
        if missing_columns:
            issue = QualityIssue(
                "MISSING_COLUMN",
                f"daily data is missing required columns: {', '.join(missing_columns)}",
            )
            removed_rows = self._all_rows(frame.index, "MISSING_COLUMN")
            return self._report(
                quality_mode,
                (issue,),
                input_rows,
                pd.DataFrame(),
                _removed_sessions(removed_rows),
                removed_rows,
            )
        if not isinstance(frame.index, pd.DatetimeIndex):
            issue = QualityIssue(
                "INVALID_INDEX",
                "daily data index must be a DatetimeIndex",
            )
            removed_rows = self._all_rows(frame.index, "INVALID_INDEX")
            return self._report(
                quality_mode,
                (issue,),
                input_rows,
                pd.DataFrame(),
                _removed_sessions(removed_rows),
                removed_rows,
            )
        if frame.index.tz is not None:
            issue = QualityIssue(
                "TIMEZONE_AWARE_INDEX",
                "daily trading sessions must be timezone-naive",
            )
            removed_rows = self._all_rows(frame.index, "TIMEZONE_AWARE_INDEX")
            return self._report(
                quality_mode,
                (issue,),
                input_rows,
                pd.DataFrame(),
                _removed_sessions(removed_rows),
                removed_rows,
            )

        working = frame.copy()
        issues: list[QualityIssue] = []
        invalid_positions = pd.Series(False, index=range(input_rows), dtype=bool)
        reason_codes: list[set[str]] = [set() for _ in range(input_rows)]

        def mark(mask: object, code: str) -> None:
            values = np.asarray(mask, dtype=bool)
            for position in np.flatnonzero(values):
                invalid_positions.iloc[int(position)] = True
                reason_codes[int(position)].add(code)

        normalized_mask = working.index == working.index.normalize()
        if not normalized_mask.all():
            bad = working.index[~normalized_mask]
            issues.append(
                QualityIssue(
                    "NON_NORMALIZED_SESSION",
                    "daily rows must use normalized trading dates",
                    _sessions(bad),
                )
            )
            mark(~normalized_mask, "NON_NORMALIZED_SESSION")

        duplicate_mask = working.index.duplicated(keep=False)
        if duplicate_mask.any():
            issues.append(
                QualityIssue(
                    "DUPLICATE_SESSION",
                    "duplicate trading sessions are ambiguous",
                    _sessions(working.index[duplicate_mask]),
                )
            )
            mark(duplicate_mask, "DUPLICATE_SESSION")

        if not working.index.is_monotonic_increasing:
            issues.append(
                QualityIssue(
                    "OUT_OF_ORDER",
                    "daily rows were not ordered by trading session",
                )
            )

        unexpected_mask = ~working.index.isin(self.expected_sessions)
        if unexpected_mask.any():
            issues.append(
                QualityIssue(
                    "UNEXPECTED_SESSION",
                    "daily data contains rows outside expected trading sessions",
                    _sessions(working.index[unexpected_mask]),
                )
            )
            mark(unexpected_mask, "UNEXPECTED_SESSION")

        numeric_columns = list(BarFrame.REQUIRED_COLUMNS)
        finite_masks: dict[str, pd.Series[bool]] = {}
        for column in numeric_columns:
            finite_masks[column] = working[column].map(_finite_number)
        numeric_valid = pd.concat(finite_masks, axis=1).all(axis=1)
        if (~numeric_valid).any():
            invalid_type_mask = pd.Series(False, index=working.index, dtype=bool)
            nonfinite_mask = pd.Series(False, index=working.index, dtype=bool)
            for column in numeric_columns:
                for position, value in enumerate(working[column].tolist()):
                    if finite_masks[column].iloc[position]:
                        continue
                    if _canonical_number(value):
                        nonfinite_mask.iloc[position] = True
                    else:
                        invalid_type_mask.iloc[position] = True
            if invalid_type_mask.any():
                issues.append(
                    QualityIssue(
                        "INVALID_NUMERIC_TYPE",
                        "market values must be numeric",
                        _sessions(working.index[invalid_type_mask]),
                    )
                )
                mark(invalid_type_mask.to_numpy(), "INVALID_NUMERIC_TYPE")
            if nonfinite_mask.any():
                issues.append(
                    QualityIssue(
                        "NONFINITE_VALUE",
                        "market values must be finite",
                        _sessions(working.index[nonfinite_mask]),
                    )
                )
                mark(nonfinite_mask.to_numpy(), "NONFINITE_VALUE")

        valid_numeric_positions = numeric_valid.to_numpy()
        if valid_numeric_positions.any():
            numeric = working.loc[numeric_valid, numeric_columns]
            nonpositive_price = (numeric[["open", "high", "low", "close"]] <= 0).any(axis=1)
            if nonpositive_price.any():
                issues.append(
                    QualityIssue(
                        "NONPOSITIVE_PRICE",
                        "prices must be positive",
                        _sessions(numeric.index[nonpositive_price]),
                    )
                )
                mask = pd.Series(False, index=range(input_rows), dtype=bool)
                mask.iloc[numeric_valid.to_numpy().nonzero()[0][nonpositive_price]] = True
                mark(mask.to_numpy(), "NONPOSITIVE_PRICE")

            negative_activity = (numeric[["volume", "amount"]] < 0).any(axis=1)
            if negative_activity.any():
                issues.append(
                    QualityIssue(
                        "NEGATIVE_ACTIVITY",
                        "volume and amount must be non-negative",
                        _sessions(numeric.index[negative_activity]),
                    )
                )
                mask = pd.Series(False, index=range(input_rows), dtype=bool)
                mask.iloc[numeric_valid.to_numpy().nonzero()[0][negative_activity]] = True
                mark(mask.to_numpy(), "NEGATIVE_ACTIVITY")

            high_invalid = numeric["high"] < numeric[["open", "low", "close"]].max(axis=1)
            low_invalid = numeric["low"] > numeric[["open", "high", "close"]].min(axis=1)
            invalid_ohlc = high_invalid | low_invalid
            if invalid_ohlc.any():
                issues.append(
                    QualityIssue(
                        "INVALID_OHLC",
                        "OHLC price relationships are invalid",
                        _sessions(numeric.index[invalid_ohlc]),
                    )
                )
                mask = pd.Series(False, index=range(input_rows), dtype=bool)
                mask.iloc[numeric_valid.to_numpy().nonzero()[0][invalid_ohlc]] = True
                mark(mask.to_numpy(), "INVALID_OHLC")

        if "adjust_factor" in working.columns:
            valid_factor = working["adjust_factor"].map(_positive_number)
            invalid_factor = ~valid_factor
            if invalid_factor.any():
                issues.append(
                    QualityIssue(
                        "INVALID_ADJUST_FACTOR",
                        "adjustment factors must be finite and positive",
                        _sessions(working.index[invalid_factor]),
                    )
                )
                mark(invalid_factor.to_numpy(), "INVALID_ADJUST_FACTOR")

        if "suspended" in working.columns:
            valid_suspended = working["suspended"].map(
                lambda value: isinstance(value, (bool, np.bool_))
            )
            if (~valid_suspended).any():
                issues.append(
                    QualityIssue(
                        "INVALID_SUSPENDED",
                        "suspended values must be boolean",
                        _sessions(working.index[~valid_suspended]),
                    )
                )
                mark((~valid_suspended).to_numpy(), "INVALID_SUSPENDED")

        valid_limits: dict[str, pd.Series[bool]] = {}
        invalid_limit = pd.Series(False, index=working.index, dtype=bool)
        for column in ("limit_up", "limit_down"):
            if column not in working.columns:
                continue
            valid_limits[column] = working[column].map(_positive_number)
            invalid_limit |= ~valid_limits[column]
        if invalid_limit.any():
            issues.append(
                QualityIssue(
                    "INVALID_LIMIT_PRICE",
                    "price limits must be finite and positive",
                    _sessions(working.index[invalid_limit]),
                )
            )
            mark(invalid_limit.to_numpy(), "INVALID_LIMIT_PRICE")

        invalid_relation = pd.Series(False, index=working.index, dtype=bool)
        for position in range(input_rows):
            if not numeric_valid.iloc[position]:
                continue
            upper_valid = "limit_up" not in valid_limits or valid_limits["limit_up"].iloc[position]
            lower_valid = (
                "limit_down" not in valid_limits or valid_limits["limit_down"].iloc[position]
            )
            if not upper_valid or not lower_valid:
                continue
            upper = working["limit_up"].iloc[position] if "limit_up" in working else None
            lower = working["limit_down"].iloc[position] if "limit_down" in working else None
            try:
                invalid = (
                    (upper is not None and working["high"].iloc[position] > upper)
                    or (lower is not None and working["low"].iloc[position] < lower)
                    or (upper is not None and lower is not None and lower > upper)
                )
            except (OverflowError, TypeError, ValueError):
                invalid = True
            invalid_relation.iloc[position] = bool(invalid)
        if invalid_relation.any():
            issues.append(
                QualityIssue(
                    "INVALID_LIMIT_RELATION",
                    "price limits conflict with each other or the observed range",
                    _sessions(working.index[invalid_relation]),
                )
            )
            mark(invalid_relation.to_numpy(), "INVALID_LIMIT_RELATION")

        if "adjust_flag" in working.columns:
            valid_adjust_flag = working["adjust_flag"].map(
                lambda value: isinstance(value, str) and value in {"1", "2", "3"}
            )
            if (~valid_adjust_flag).any():
                issues.append(
                    QualityIssue(
                        "INVALID_ADJUST_FLAG",
                        "adjust flags must be canonical BaoStock strings 1, 2, or 3",
                        _sessions(working.index[~valid_adjust_flag]),
                    )
                )
                mark((~valid_adjust_flag).to_numpy(), "INVALID_ADJUST_FLAG")

        cleaned = working.iloc[(~invalid_positions).to_numpy()].sort_index()
        missing = self.expected_sessions.difference(cleaned.index)
        if len(missing):
            issues.append(
                QualityIssue(
                    "MISSING_SESSION",
                    "daily data is missing expected trading sessions",
                    _sessions(missing),
                )
            )

        if len(cleaned) > 1:
            jumps = cleaned["close"].pct_change().abs() > 0.5
            if jumps.any():
                issues.append(
                    QualityIssue(
                        "LARGE_PRICE_JUMP",
                        "close price changes by more than 50% between adjacent sessions",
                        _sessions(cleaned.index[jumps]),
                    )
                )
            if "adjust_factor" in cleaned.columns:
                factor_jumps = cleaned["adjust_factor"].pct_change().abs() > 0.5
                if factor_jumps.any():
                    issues.append(
                        QualityIssue(
                            "ADJUST_FACTOR_DISCONTINUITY",
                            "adjustment factor changes by more than 50% between adjacent sessions",
                            _sessions(cleaned.index[factor_jumps]),
                        )
                    )

        if cleaned.empty and not issues:
            issues.append(QualityIssue("EMPTY_DATASET", "daily data contains no rows"))
        elif not cleaned.empty:
            cleaned = BarFrame.validate(cleaned)

        ordered_issues = tuple(sorted(issues, key=lambda issue: _ISSUE_ORDER[issue.code]))
        removed_rows = tuple(
            RemovedRow(
                position=position,
                timestamp=_audit_index(working.index[position])[0],
                raw_index_text=_audit_index(working.index[position])[1],
                reason_codes=tuple(
                    sorted(reason_codes[position], key=lambda code: _ISSUE_ORDER[code])
                ),
            )
            for position in range(input_rows)
            if invalid_positions.iloc[position]
        )
        return self._report(
            quality_mode,
            ordered_issues,
            input_rows,
            cleaned,
            _removed_sessions(removed_rows),
            removed_rows,
        )

    @staticmethod
    def _all_rows(index: pd.Index[Any], reason_code: str) -> tuple[RemovedRow, ...]:
        return tuple(
            RemovedRow(
                position=position,
                timestamp=_audit_index(value)[0],
                raw_index_text=_audit_index(value)[1],
                reason_codes=(reason_code,),
            )
            for position, value in enumerate(index)
        )

    @staticmethod
    def _report(
        mode: QualityMode,
        issues: tuple[QualityIssue, ...],
        input_rows: int,
        frame: pd.DataFrame,
        removed_sessions: tuple[str, ...],
        removed_rows: tuple[RemovedRow, ...],
    ) -> QualityReport:
        return QualityReport(
            mode=mode,
            issues=issues,
            input_rows=input_rows,
            output_rows=len(frame),
            removed_sessions=removed_sessions,
            removed_rows=removed_rows,
            _frame=frame.copy(),
        )
