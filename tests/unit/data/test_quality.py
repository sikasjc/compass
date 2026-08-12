from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from decimal import Decimal
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from compass.data.quality import DailyQualityGate, QualityMode, RemovedRow
from compass.domain.quality_report import ISSUE_POLICY


def bars_for(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [4.0] * len(dates),
            "high": [4.2] * len(dates),
            "low": [3.9] * len(dates),
            "close": [4.1] * len(dates),
            "volume": [1000.0] * len(dates),
            "amount": [4100.0] * len(dates),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


def test_strict_mode_reports_a_missing_expected_session_first() -> None:
    gate = DailyQualityGate(expected_sessions=pd.DatetimeIndex(["2026-07-20", "2026-07-21"]))

    result = gate.evaluate(bars_for(["2026-07-20"]), QualityMode.STRICT)

    assert result.blocking
    assert result.accepted is False
    assert result.issues[0].code == "MISSING_SESSION"
    assert result.issues[0].sessions == ("2026-07-21",)


def test_expected_sessions_use_naive_normalized_trading_dates() -> None:
    aware = pd.DatetimeIndex(["2026-07-20 00:00+08:00"])
    intraday = pd.DatetimeIndex(["2026-07-20 09:30"])

    with pytest.raises(ValueError, match="timezone-naive"):
        DailyQualityGate(expected_sessions=aware)
    with pytest.raises(ValueError, match="normalized"):
        DailyQualityGate(expected_sessions=intraday)


def test_degraded_mode_only_removes_rows_with_concrete_invalid_values() -> None:
    frame = bars_for(["2026-07-20", "2026-07-21", "2026-07-22"])
    frame.loc[pd.Timestamp("2026-07-20"), "volume"] = -1
    frame.loc[pd.Timestamp("2026-07-21"), "high"] = 3.8
    gate = DailyQualityGate(expected_sessions=frame.index)

    result = gate.evaluate(frame, QualityMode.DEGRADED)

    assert result.accepted
    assert result.removed_sessions == ("2026-07-20", "2026-07-21")
    assert result.frame.index.tolist() == [pd.Timestamp("2026-07-22")]
    assert [issue.code for issue in result.issues] == [
        "MISSING_SESSION",
        "NEGATIVE_ACTIVITY",
        "INVALID_OHLC",
    ]


def test_degraded_mode_records_but_does_not_fabricate_a_missing_session() -> None:
    gate = DailyQualityGate(expected_sessions=pd.DatetimeIndex(["2026-07-20", "2026-07-21"]))

    result = gate.evaluate(bars_for(["2026-07-20"]), QualityMode.DEGRADED)

    assert result.accepted
    assert result.frame.index.tolist() == [pd.Timestamp("2026-07-20")]
    assert result.issues[0].code == "MISSING_SESSION"
    assert result.removed_sessions == ()


def test_duplicate_sessions_are_all_removed_instead_of_choosing_a_value() -> None:
    frame = pd.concat([bars_for(["2026-07-20"]), bars_for(["2026-07-20", "2026-07-21"])])
    gate = DailyQualityGate(expected_sessions=pd.DatetimeIndex(["2026-07-20", "2026-07-21"]))

    result = gate.evaluate(frame, QualityMode.DEGRADED)

    assert result.frame.index.tolist() == [pd.Timestamp("2026-07-21")]
    duplicate = next(issue for issue in result.issues if issue.code == "DUPLICATE_SESSION")
    assert duplicate.sessions == ("2026-07-20",)


def test_removed_rows_preserve_position_full_timestamp_and_all_stable_reasons() -> None:
    frame = bars_for(["2026-07-20 09:30", "2026-07-20 09:30", "2026-07-21"])
    gate = DailyQualityGate(expected_sessions=pd.DatetimeIndex(["2026-07-21"]))

    result = gate.evaluate(frame, QualityMode.DEGRADED)

    assert result.removed_sessions == ("2026-07-20",)
    assert result.removed_rows == (
        RemovedRow(
            position=0,
            timestamp="2026-07-20T09:30:00",
            raw_index_text="2026-07-20 09:30:00",
            reason_codes=(
                "NON_NORMALIZED_SESSION",
                "DUPLICATE_SESSION",
                "UNEXPECTED_SESSION",
            ),
        ),
        RemovedRow(
            position=1,
            timestamp="2026-07-20T09:30:00",
            raw_index_text="2026-07-20 09:30:00",
            reason_codes=(
                "NON_NORMALIZED_SESSION",
                "DUPLICATE_SESSION",
                "UNEXPECTED_SESSION",
            ),
        ),
    )
    assert len(result.removed_rows) == result.input_rows - result.output_rows
    assert result.to_dict()["removed_rows"][0]["position"] == 0
    with pytest.raises(FrozenInstanceError):
        result.removed_rows[0].position = 10  # type: ignore[misc]


def test_quality_report_is_deterministic_complete_and_json_serializable() -> None:
    frame = bars_for(["2026-07-21", "2026-07-20"])
    frame.loc[pd.Timestamp("2026-07-21"), "close"] = float("nan")
    gate = DailyQualityGate(
        expected_sessions=pd.DatetimeIndex(["2026-07-20", "2026-07-21", "2026-07-22"])
    )

    first = gate.evaluate(frame, QualityMode.DEGRADED)
    second = gate.evaluate(frame, QualityMode.DEGRADED)
    payload = first.to_dict()

    assert payload == second.to_dict()
    assert [issue["code"] for issue in payload["issues"]] == [
        "OUT_OF_ORDER",
        "MISSING_SESSION",
        "NONFINITE_VALUE",
    ]
    assert payload["input_rows"] == 2
    assert payload["output_rows"] == 1
    assert payload["mode"] == "degraded"
    json.dumps(payload)
    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json()) == payload


def test_strict_mode_blocks_out_of_order_rows_while_degraded_canonicalizes_order() -> None:
    frame = bars_for(["2026-07-21", "2026-07-20"])
    expected = pd.DatetimeIndex(["2026-07-20", "2026-07-21"])
    gate = DailyQualityGate(expected_sessions=expected)

    strict = gate.evaluate(frame, QualityMode.STRICT)
    degraded = gate.evaluate(frame, QualityMode.DEGRADED)

    assert strict.accepted is False
    assert degraded.accepted
    assert degraded.frame.index.equals(expected)


def test_invalid_optional_suspension_value_is_removed_not_coerced() -> None:
    frame = bars_for(["2026-07-20", "2026-07-21"])
    frame["suspended"] = ["False", False]
    gate = DailyQualityGate(expected_sessions=frame.index)

    result = gate.evaluate(frame, QualityMode.DEGRADED)

    assert result.frame.index.tolist() == [pd.Timestamp("2026-07-21")]
    assert any(issue.code == "INVALID_SUSPENDED" for issue in result.issues)


def test_duplicate_columns_produce_a_structured_unrecoverable_report() -> None:
    frame = bars_for(["2026-07-20"])
    frame = pd.concat([frame, frame[["open"]]], axis=1)
    gate = DailyQualityGate(expected_sessions=frame.index)

    result = gate.evaluate(frame, QualityMode.DEGRADED)

    assert result.accepted is False
    assert result.issues[0].code == "DUPLICATE_COLUMN"
    assert len(result.removed_rows) == result.input_rows - result.output_rows


def test_issue_policy_is_the_single_immutable_source_of_severity() -> None:
    assert isinstance(ISSUE_POLICY, MappingProxyType)
    assert ISSUE_POLICY["INVALID_FRAME"].blocking is True
    assert ISSUE_POLICY["INVALID_FRAME"].recoverable is False
    with pytest.raises(TypeError):
        ISSUE_POLICY["INVALID_FRAME"] = ISSUE_POLICY["EMPTY_DATASET"]  # type: ignore[index]


class OverflowingNumber:
    def __float__(self) -> float:
        raise OverflowError("synthetic overflow")


@pytest.mark.parametrize(
    "invalid_value",
    [
        "4.0",
        float("nan"),
        object(),
        OverflowingNumber(),
        10**400,
        1 + 2j,
        np.complex128(1 + 2j),
        Decimal("4.0"),
        np.longdouble("4.0"),
    ],
)
def test_malformed_required_numbers_always_return_a_report(
    invalid_value: object,
) -> None:
    frame = bars_for(["2026-07-20"])
    frame["open"] = pd.Series([invalid_value], index=frame.index, dtype=object)
    gate = DailyQualityGate(expected_sessions=frame.index)

    result = gate.evaluate(frame, QualityMode.DEGRADED)

    assert result.accepted is False
    assert result.removed_rows[0].reason_codes in {
        ("INVALID_NUMERIC_TYPE",),
        ("NONFINITE_VALUE",),
    }


@pytest.mark.parametrize(
    ("column", "value", "issue_code"),
    [
        ("adjust_factor", float("nan"), "INVALID_ADJUST_FACTOR"),
        ("adjust_factor", "1.0", "INVALID_ADJUST_FACTOR"),
        ("suspended", "False", "INVALID_SUSPENDED"),
        ("limit_up", float("nan"), "INVALID_LIMIT_PRICE"),
        ("limit_down", 0, "INVALID_LIMIT_PRICE"),
        ("adjust_flag", 3, "INVALID_ADJUST_FLAG"),
        ("adjust_flag", "4", "INVALID_ADJUST_FLAG"),
    ],
)
def test_present_optional_values_must_be_canonical_and_nonnullable(
    column: str, value: object, issue_code: str
) -> None:
    frame = bars_for(["2026-07-20"])
    frame[column] = pd.Series([value], index=frame.index, dtype=object)
    gate = DailyQualityGate(expected_sessions=frame.index)

    result = gate.evaluate(frame, QualityMode.DEGRADED)

    assert result.accepted is False
    assert any(issue.code == issue_code for issue in result.issues)
    assert issue_code in result.removed_rows[0].reason_codes


@pytest.mark.parametrize(
    ("limit_up", "limit_down"),
    [(3.8, 3.9), (4.3, 4.05)],
)
def test_price_limit_relationships_are_checked(limit_up: float, limit_down: float) -> None:
    frame = bars_for(["2026-07-20"])
    frame["limit_up"] = [limit_up]
    frame["limit_down"] = [limit_down]
    gate = DailyQualityGate(expected_sessions=frame.index)

    result = gate.evaluate(frame, QualityMode.DEGRADED)

    assert result.accepted is False
    assert any(issue.code == "INVALID_LIMIT_RELATION" for issue in result.issues)


def test_missing_columns_or_an_empty_clean_result_cannot_be_accepted_degraded() -> None:
    missing = bars_for(["2026-07-20"]).drop(columns="amount")
    invalid = bars_for(["2026-07-20"])
    invalid.loc[:, "open"] = -1
    gate = DailyQualityGate(expected_sessions=pd.DatetimeIndex(["2026-07-20"]))

    missing_result = gate.evaluate(missing, QualityMode.DEGRADED)
    invalid_result = gate.evaluate(invalid, QualityMode.DEGRADED)

    assert missing_result.accepted is False
    assert missing_result.issues[0].code == "MISSING_COLUMN"
    assert invalid_result.accepted is False
    assert invalid_result.frame.empty


def test_quality_mode_rejects_unknown_or_non_text_values() -> None:
    assert QualityMode.parse("strict") is QualityMode.STRICT
    with pytest.raises(ValueError, match="quality mode"):
        QualityMode.parse("permissive")
    with pytest.raises(TypeError, match="quality mode"):
        QualityMode.parse(1)
