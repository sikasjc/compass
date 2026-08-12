from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date, datetime
from decimal import Decimal
import json
from pathlib import Path
import traceback
from typing import Any

import numpy as np
import pandas as pd
import pytest

from compass.data.base import (
    DailyBarRequest,
    ProviderConfigurationError,
    ProviderError,
    ProviderErrorKind,
)
from compass.data.exchange_calendar import CalendarIdentity
from compass.data.quality import DataQualityError, QualityMode
from compass.domain.market import InstrumentId
from compass.domain.trading import CorporateAction
from compass.services.data_service import DataService
from compass.storage.market_store import MarketStore


def request() -> DailyBarRequest:
    return DailyBarRequest(InstrumentId.parse("SSE.510300"), date(2026, 7, 20), date(2026, 7, 21))


def bars(dates: Sequence[str] = ("2026-07-20", "2026-07-21")) -> pd.DataFrame:
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


class FakeProvider:
    def __init__(
        self,
        name: str,
        outcomes: Sequence[object],
        *,
        expected_request: DailyBarRequest | None = None,
    ) -> None:
        self.name = name
        self._outcomes = list(outcomes)
        self.calls = 0
        self.expected_request = expected_request or request()

    def fetch_daily(self, requested: DailyBarRequest) -> pd.DataFrame:
        assert requested == self.expected_request
        outcome = self._outcomes[min(self.calls, len(self._outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, pd.DataFrame):
            return outcome.copy()
        return outcome  # type: ignore[return-value]

    def fetch_snapshot(self, instruments: Sequence[InstrumentId]) -> pd.DataFrame:
        raise NotImplementedError

    def fetch_corporate_actions(self, requested: DailyBarRequest) -> Sequence[Any]:
        raise NotImplementedError


def service(
    tmp_path: Path, *, sleeper: Callable[[float], None] = lambda delay: None
) -> DataService:
    return DataService(
        MarketStore(tmp_path),
        expected_sessions=lambda requested: pd.DatetimeIndex(["2026-07-20", "2026-07-21"]),
        sleeper=sleeper,
    )


def provider_error(name: str, kind: ProviderErrorKind) -> ProviderError:
    return ProviderError(kind, name, "synthetic offline failure")


def test_fallback_never_silently_overwrites_provider_identity(tmp_path: Path) -> None:
    primary = FakeProvider("primary", [bars(["2026-07-20"])])
    fallback = FakeProvider("fallback", [bars()])

    result = service(tmp_path).sync_daily(request(), (primary, fallback), "strict")

    assert result.provider == "fallback"
    assert result.attempts == ("primary", "fallback")
    assert result.manifest.provider == "fallback"
    assert result.quality_report.mode is QualityMode.STRICT


def test_incremental_sync_reuses_trusted_rows_and_fetches_only_missing_sessions(
    tmp_path: Path,
) -> None:
    data_service = DataService(
        MarketStore(tmp_path),
        expected_sessions=lambda requested: pd.date_range(
            requested.start,
            requested.end,
            freq="D",
        ),
        sleeper=lambda delay: None,
    )
    initial_frame = bars(["2026-07-20"])
    initial_frame["adjust_flag"] = "3"
    initial_provider = FakeProvider("primary", [initial_frame])
    initial = data_service.sync_daily(
        request(),
        (initial_provider,),
        QualityMode.DEGRADED,
    )
    missing_request = DailyBarRequest(
        InstrumentId.parse("SSE.510300"),
        date(2026, 7, 21),
        date(2026, 7, 21),
    )
    missing_provider = FakeProvider(
        "primary",
        [bars(["2026-07-21"])],
        expected_request=missing_request,
    )

    result = data_service.sync_daily_incremental(
        request(),
        (missing_provider,),
        QualityMode.STRICT,
        trusted_manifest=initial.manifest,
    )

    assert missing_provider.calls == 1
    assert result.manifest.rows == 2
    assert result.manifest.manifest_id != initial.manifest.manifest_id
    assert tuple(result.quality_report.frame.index.date) == (
        date(2026, 7, 20),
        date(2026, 7, 21),
    )
    assert "adjust_flag" not in result.quality_report.frame
    assert result.quality_report.issues == ()


def test_incremental_sync_skips_provider_when_requested_data_is_already_trusted(
    tmp_path: Path,
) -> None:
    data_service = service(tmp_path)
    initial = data_service.sync_daily(
        request(),
        (FakeProvider("primary", [bars()]),),
        QualityMode.STRICT,
    )
    unused = FakeProvider("primary", [AssertionError("provider must not be called")])

    result = data_service.sync_daily_incremental(
        request(),
        (unused,),
        QualityMode.STRICT,
        trusted_manifest=initial.manifest,
    )

    assert unused.calls == 0
    assert result.manifest.manifest_id != initial.manifest.manifest_id
    assert result.manifest.content_hash == initial.manifest.content_hash
    assert result.attempts == ()


def test_degraded_incremental_sync_keeps_trusted_rows_when_a_gap_stays_unavailable(
    tmp_path: Path,
) -> None:
    data_service = DataService(
        MarketStore(tmp_path),
        expected_sessions=lambda requested: pd.date_range(
            requested.start,
            requested.end,
            freq="D",
        ),
        sleeper=lambda delay: None,
    )
    initial = data_service.sync_daily(
        request(),
        (FakeProvider("primary", [bars(["2026-07-20"])]),),
        QualityMode.DEGRADED,
    )
    missing_request = DailyBarRequest(
        InstrumentId.parse("SSE.510300"),
        date(2026, 7, 21),
        date(2026, 7, 21),
    )
    unavailable = FakeProvider(
        "primary",
        [bars(())],
        expected_request=missing_request,
    )

    result = data_service.sync_daily_incremental(
        request(),
        (unavailable,),
        QualityMode.DEGRADED,
        trusted_manifest=initial.manifest,
    )

    assert unavailable.calls == 1
    assert result.manifest.rows == 1
    assert result.quality_report.accepted is True
    assert tuple(issue.code for issue in result.quality_report.issues) == ("MISSING_SESSION",)
    assert result.attempts == ("primary",)


def test_rule_attestation_gate_falls_back_before_persisting_default_path(
    tmp_path: Path,
) -> None:
    unattested = FakeProvider("unattested", [bars()])
    attested_frame = bars()
    attested_frame["price_limit_rate"] = Decimal("0.10")
    attested_frame["price_limit_rule_id"] = "cn-price-limit-v1:etf-name:0.10"
    attested_frame["listing_regime_known"] = True
    fallback = FakeProvider("attested", [attested_frame])

    result = DataService(
        MarketStore(tmp_path),
        expected_sessions=lambda requested: pd.DatetimeIndex(["2026-07-20", "2026-07-21"]),
        require_rule_attestation=True,
        sleeper=lambda delay: None,
    ).sync_daily(request(), (unattested, fallback), QualityMode.STRICT)

    assert result.provider == "attested"
    assert result.manifest.provenance is not None
    assert tuple(
        (item.provider, item.outcome, item.failure_category)
        for item in result.manifest.provenance.daily_attempts
    ) == (
        ("unattested", "quality_rejected", "market_rules"),
        ("attested", "selected", None),
    )


@pytest.mark.parametrize("kind", [ProviderErrorKind.NETWORK, ProviderErrorKind.RATE_LIMIT])
def test_retryable_errors_use_three_delays_and_four_total_attempts(
    tmp_path: Path, kind: ProviderErrorKind
) -> None:
    delays: list[float] = []
    primary = FakeProvider(
        "primary",
        [provider_error("primary", kind)] * 3 + [bars()],
    )

    result = service(tmp_path, sleeper=delays.append).sync_daily(
        request(), (primary,), QualityMode.STRICT
    )

    assert result.provider == "primary"
    assert result.attempts == ("primary", "primary", "primary", "primary")
    assert delays == [0.25, 0.5, 1.0]


def test_exhausted_retryable_provider_falls_back_only_after_four_attempts(tmp_path: Path) -> None:
    delays: list[float] = []
    primary = FakeProvider("primary", [provider_error("primary", ProviderErrorKind.NETWORK)])
    fallback = FakeProvider("fallback", [bars()])

    result = service(tmp_path, sleeper=delays.append).sync_daily(
        request(), (primary, fallback), QualityMode.STRICT
    )

    assert result.attempts == ("primary", "primary", "primary", "primary", "fallback")
    assert primary.calls == 4
    assert fallback.calls == 1
    assert delays == [0.25, 0.5, 1.0]


@pytest.mark.parametrize(
    "kind",
    [
        ProviderErrorKind.AUTHENTICATION,
        ProviderErrorKind.MALFORMED_RESPONSE,
        ProviderErrorKind.CONFIGURATION,
        ProviderErrorKind.CAPABILITY,
    ],
)
def test_non_retryable_errors_stop_without_fallback(
    tmp_path: Path, kind: ProviderErrorKind
) -> None:
    primary = FakeProvider("primary", [provider_error("primary", kind)])
    fallback = FakeProvider("fallback", [bars()])

    with pytest.raises(ProviderError) as caught:
        service(tmp_path).sync_daily(request(), (primary, fallback), QualityMode.STRICT)

    assert caught.value.kind is kind
    assert primary.calls == 1
    assert fallback.calls == 0
    assert list((tmp_path / "manifests").iterdir()) == []


def exception_with_secret_cause(kind: ProviderErrorKind) -> ProviderError:
    error = ProviderError(kind, "primary", "typed provider failure")
    error.__cause__ = ValueError("token=sentinel-service-secret")
    return error


@pytest.mark.parametrize(
    "outcome",
    [
        exception_with_secret_cause(ProviderErrorKind.AUTHENTICATION),
        ValueError("https://upstream.invalid/?token=sentinel-service-secret"),
    ],
)
def test_immediate_fetch_failures_are_typed_and_suppress_secret_context(
    tmp_path: Path, outcome: Exception
) -> None:
    primary = FakeProvider("primary", [outcome])
    fallback = FakeProvider("fallback", [bars()])

    with pytest.raises(ProviderError) as caught:
        service(tmp_path).sync_daily(request(), (primary, fallback), QualityMode.STRICT)

    formatted = "".join(traceback.format_exception(caught.value))
    assert caught.value.kind in {
        ProviderErrorKind.AUTHENTICATION,
        ProviderErrorKind.MALFORMED_RESPONSE,
    }
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    assert "sentinel-service-secret" not in formatted
    assert fallback.calls == 0


def test_exhausted_retryable_failure_suppresses_original_secret_cause(tmp_path: Path) -> None:
    failure = exception_with_secret_cause(ProviderErrorKind.NETWORK)
    primary = FakeProvider("primary", [failure])

    with pytest.raises(ProviderError) as caught:
        service(tmp_path).sync_daily(request(), (primary,), QualityMode.STRICT)

    formatted = "".join(traceback.format_exception(caught.value))
    assert caught.value.kind is ProviderErrorKind.NETWORK
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    assert "sentinel-service-secret" not in formatted


@pytest.mark.parametrize("invalid_frame", [None, object(), [1, 2, 3]])
def test_non_dataframe_success_is_malformed_and_never_falls_back(
    tmp_path: Path, invalid_frame: object
) -> None:
    primary = FakeProvider("primary", [invalid_frame])
    fallback = FakeProvider("fallback", [bars()])

    with pytest.raises(ProviderError) as caught:
        service(tmp_path).sync_daily(request(), (primary, fallback), QualityMode.STRICT)

    assert caught.value.kind is ProviderErrorKind.MALFORMED_RESPONSE
    assert primary.calls == 1
    assert fallback.calls == 0


def test_strict_quality_failure_uses_fallback_and_never_writes_rejected_data(
    tmp_path: Path,
) -> None:
    first = bars()
    first.loc[pd.Timestamp("2026-07-20"), "volume"] = -1
    primary = FakeProvider("primary", [first])
    fallback = FakeProvider("fallback", [bars()])

    result = service(tmp_path).sync_daily(request(), (primary, fallback), QualityMode.STRICT)

    assert result.provider == "fallback"
    assert result.manifest.rows == 2
    assert len(list((tmp_path / "manifests").glob("*.json"))) == 1


def test_all_strict_quality_failures_raise_with_last_complete_report(tmp_path: Path) -> None:
    first = FakeProvider("first", [bars(["2026-07-20"])])
    second = FakeProvider("second", [bars(["2026-07-21"])])

    with pytest.raises(DataQualityError) as caught:
        service(tmp_path).sync_daily(request(), (first, second), QualityMode.STRICT)

    assert caught.value.provider == "second"
    assert caught.value.attempts == ("first", "second")
    assert caught.value.report.issues[0].code == "MISSING_SESSION"
    assert list((tmp_path / "manifests").iterdir()) == []


def test_degraded_mode_removes_only_invalid_rows_and_returns_full_report(tmp_path: Path) -> None:
    source = bars()
    source.loc[pd.Timestamp("2026-07-20"), "volume"] = -1
    provider = FakeProvider("primary", [source])

    result = service(tmp_path).sync_daily(request(), (provider,), "degraded")

    stored = MarketStore(tmp_path).read_manifest(result.manifest.manifest_id)
    assert stored.index.tolist() == [pd.Timestamp("2026-07-21")]
    assert result.quality_report.removed_sessions == ("2026-07-20",)
    assert [issue.code for issue in result.quality_report.issues] == [
        "MISSING_SESSION",
        "NEGATIVE_ACTIVITY",
    ]
    assert result.manifest.provenance is not None
    assert result.manifest.provenance.daily_complete is True
    assert result.manifest.provenance.missing_sessions == (date(2026, 7, 20),)
    assert json.loads(result.manifest.quality_report_json) == result.quality_report.to_dict()


def test_degraded_manifest_preserves_domain_ordered_multi_reason_removed_rows(
    tmp_path: Path,
) -> None:
    source = bars(("2026-07-20 09:30", "2026-07-20 09:30", "2026-07-21"))
    provider = FakeProvider("primary", [source])

    result = service(tmp_path).sync_daily(request(), (provider,), QualityMode.DEGRADED)

    assert result.manifest.rows == 1
    assert result.quality_report.removed_rows[0].reason_codes == (
        "NON_NORMALIZED_SESSION",
        "DUPLICATE_SESSION",
        "UNEXPECTED_SESSION",
    )


@pytest.mark.parametrize("split_invalid_limits", [False, True])
def test_degraded_sync_merges_invalid_limit_issue_and_persists_row_audit(
    tmp_path: Path, split_invalid_limits: bool
) -> None:
    limit_request = DailyBarRequest(
        InstrumentId.parse("SSE.510300"), date(2026, 7, 20), date(2026, 7, 22)
    )
    source = bars(("2026-07-20", "2026-07-21", "2026-07-22"))
    if split_invalid_limits:
        source["limit_up"] = [float("nan"), 4.3, 4.3]
        source["limit_down"] = [3.8, 0.0, 3.8]
        expected_sessions = ("2026-07-20", "2026-07-21")
        expected_replay = [pd.Timestamp("2026-07-22")]
    else:
        source["limit_up"] = [float("nan"), 4.3, 4.3]
        source["limit_down"] = [0.0, 3.8, 3.8]
        expected_sessions = ("2026-07-20",)
        expected_replay = [pd.Timestamp("2026-07-21"), pd.Timestamp("2026-07-22")]
    provider = FakeProvider("primary", [source], expected_request=limit_request)
    data_service = DataService(
        MarketStore(tmp_path),
        expected_sessions=lambda requested: pd.DatetimeIndex(
            ["2026-07-20", "2026-07-21", "2026-07-22"]
        ),
        sleeper=lambda delay: None,
    )

    result = data_service.sync_daily(limit_request, (provider,), QualityMode.DEGRADED)

    limit_issues = [
        issue for issue in result.quality_report.issues if issue.code == "INVALID_LIMIT_PRICE"
    ]
    assert len(limit_issues) == 1
    assert limit_issues[0].sessions == expected_sessions
    assert all(
        row.reason_codes == ("INVALID_LIMIT_PRICE",) for row in result.quality_report.removed_rows
    )
    replay = MarketStore(tmp_path).read_manifest(result.manifest.manifest_id)
    assert replay.index.tolist() == expected_replay


def test_degraded_sync_persists_nat_as_explicit_non_timestamp_audit(tmp_path: Path) -> None:
    source = bars(("2026-07-20", "2026-07-21"))
    source.index = pd.DatetimeIndex([pd.NaT, "2026-07-21"], name="date")
    provider = FakeProvider("primary", [source])

    result = service(tmp_path).sync_daily(request(), (provider,), QualityMode.DEGRADED)

    [removed] = result.quality_report.removed_rows
    assert removed.position == 0
    assert removed.timestamp is None
    assert removed.raw_index_text == "NaT"
    assert result.quality_report.removed_sessions == ()
    replay = MarketStore(tmp_path).read_manifest(result.manifest.manifest_id)
    assert replay.index.tolist() == [pd.Timestamp("2026-07-21")]


def test_degraded_sync_drops_longdouble_before_parquet_write(tmp_path: Path) -> None:
    source = bars()
    source["open"] = pd.Series([np.longdouble("4.0"), 4.0], index=source.index, dtype=object)
    provider = FakeProvider("primary", [source])

    result = service(tmp_path).sync_daily(request(), (provider,), QualityMode.DEGRADED)

    assert result.manifest.rows == 1
    assert result.quality_report.removed_rows[0].reason_codes == ("INVALID_NUMERIC_TYPE",)
    replay = MarketStore(tmp_path).read_manifest(result.manifest.manifest_id)
    assert replay.index.tolist() == [pd.Timestamp("2026-07-21")]


def test_success_persists_the_canonical_complete_quality_report(tmp_path: Path) -> None:
    provider = FakeProvider("primary", [bars()])

    result = service(tmp_path).sync_daily(request(), (provider,), QualityMode.STRICT)

    persisted = json.loads(result.manifest.quality_report_json)
    assert persisted == result.quality_report.to_dict()
    raw_manifest = json.loads(
        MarketStore(tmp_path).manifest_path(result.manifest.manifest_id).read_text("utf-8")
    )
    assert json.loads(raw_manifest["quality_report_json"]) == persisted


def test_empty_or_duplicate_provider_lists_fail_before_fetch_or_write(tmp_path: Path) -> None:
    duplicate_one = FakeProvider("duplicate", [bars()])
    duplicate_two = FakeProvider("duplicate", [bars()])
    data_service = service(tmp_path)

    with pytest.raises(ValueError, match="at least one"):
        data_service.sync_daily(request(), (), QualityMode.STRICT)
    with pytest.raises(ValueError, match="duplicate"):
        data_service.sync_daily(request(), (duplicate_one, duplicate_two), QualityMode.STRICT)

    assert duplicate_one.calls == 0
    assert duplicate_two.calls == 0
    assert list((tmp_path / "manifests").iterdir()) == []


def test_data_service_requires_an_explicit_exchange_session_calendar(tmp_path: Path) -> None:
    with pytest.raises(ProviderConfigurationError, match="expected sessions"):
        DataService(MarketStore(tmp_path))


@pytest.mark.parametrize(
    ("holiday_start", "holiday_end", "sessions"),
    [
        (date(2026, 2, 13), date(2026, 2, 24), ("2026-02-13", "2026-02-24")),
        (date(2026, 9, 30), date(2026, 10, 9), ("2026-09-30", "2026-10-09")),
    ],
)
def test_sync_trusts_injected_exchange_sessions_not_weekday_inference(
    tmp_path: Path,
    holiday_start: date,
    holiday_end: date,
    sessions: tuple[str, str],
) -> None:
    holiday_request = DailyBarRequest(InstrumentId.parse("SSE.510300"), holiday_start, holiday_end)
    provider = FakeProvider(
        "primary",
        [bars(sessions)],
        expected_request=holiday_request,
    )
    data_service = DataService(
        MarketStore(tmp_path),
        expected_sessions=lambda requested: pd.DatetimeIndex(sessions),
        sleeper=lambda delay: None,
    )

    result = data_service.sync_daily(holiday_request, (provider,), QualityMode.STRICT)

    assert result.manifest.rows == 2
    assert result.quality_report.issues == ()


def test_store_failure_is_not_reported_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_service = service(tmp_path)
    provider = FakeProvider("primary", [bars()])

    def fail_write(*args: object, **kwargs: object) -> object:
        raise OSError("synthetic store failure")

    monkeypatch.setattr(data_service.store, "write_daily", fail_write)

    with pytest.raises(OSError, match="store failure"):
        data_service.sync_daily(request(), (provider,), QualityMode.STRICT)
    assert list((tmp_path / "manifests").iterdir()) == []


def test_sleeper_failure_stops_without_fallback_or_write(tmp_path: Path) -> None:
    primary = FakeProvider("primary", [provider_error("primary", ProviderErrorKind.NETWORK)])
    fallback = FakeProvider("fallback", [bars()])

    def fail_sleep(delay: float) -> None:
        raise RuntimeError(f"sleep failed at {delay}")

    with pytest.raises(RuntimeError, match="sleep failed"):
        service(tmp_path, sleeper=fail_sleep).sync_daily(
            request(), (primary, fallback), QualityMode.STRICT
        )

    assert primary.calls == 1
    assert fallback.calls == 0
    assert list((tmp_path / "manifests").iterdir()) == []


def test_sleeper_failure_propagates_without_raw_provider_secret_context(tmp_path: Path) -> None:
    primary = FakeProvider("primary", [ConnectionError("token=sentinel-sleeper-secret")])

    def fail_sleep(delay: float) -> None:
        raise RuntimeError(f"sleep failed at {delay}")

    with pytest.raises(RuntimeError) as caught:
        service(tmp_path, sleeper=fail_sleep).sync_daily(request(), (primary,), QualityMode.STRICT)

    formatted = "".join(traceback.format_exception(caught.value))
    assert "sentinel-sleeper-secret" not in formatted


def test_final_provider_failure_is_the_error_reported_after_an_earlier_quality_failure(
    tmp_path: Path,
) -> None:
    quality_failure = FakeProvider("quality", [bars(["2026-07-20"])])
    network_failure = FakeProvider(
        "network", [provider_error("network", ProviderErrorKind.NETWORK)]
    )

    with pytest.raises(ProviderError) as caught:
        service(tmp_path).sync_daily(
            request(), (quality_failure, network_failure), QualityMode.STRICT
        )

    assert caught.value.kind is ProviderErrorKind.NETWORK


def test_manifest_persists_exact_fallback_calendar_and_completion_provenance(
    tmp_path: Path,
) -> None:
    source_at = datetime.fromisoformat("2026-07-21T15:01:00+08:00")
    fetched_at = datetime.fromisoformat("2026-07-21T15:02:00+08:00")
    incomplete = bars(("2026-07-20",))
    complete = bars()
    complete.attrs["source_at"] = source_at
    primary = FakeProvider("primary", [incomplete])
    fallback = FakeProvider("fallback", [complete])
    calendar = CalendarIdentity(
        "a" * 64,
        "calendar-source",
        "calendar-v3",
        date(2026, 7, 1),
        date(2026, 7, 31),
    )
    store = MarketStore(tmp_path)
    data_service = DataService(
        store,
        expected_sessions=lambda requested: pd.DatetimeIndex(["2026-07-20", "2026-07-21"]),
        calendar_identity=lambda requested: calendar,
        completed_session=lambda requested: date(2026, 7, 21),
        clock=lambda: fetched_at,
        sleeper=lambda delay: None,
    )

    result = data_service.sync_daily(request(), (primary, fallback), QualityMode.STRICT)
    provenance = result.manifest.provenance

    assert provenance is not None
    assert tuple(
        (item.provider, item.outcome, item.failure_category) for item in provenance.daily_attempts
    ) == (
        ("primary", "quality_rejected", "data_quality"),
        ("fallback", "selected", None),
    )
    assert provenance.selected_provider == "fallback"
    assert provenance.source_at == source_at
    assert provenance.fetched_at == fetched_at
    assert provenance.calendar == calendar
    assert provenance.calendar_sessions == (
        date(2026, 7, 20),
        date(2026, 7, 21),
    )
    assert provenance.completed_through == date(2026, 7, 21)
    assert provenance.daily_complete is True
    reloaded = store.load_manifest(result.manifest.manifest_id)
    assert reloaded.provenance_json == result.manifest.provenance_json
    assert reloaded.provenance == provenance


def test_sync_persists_provider_company_actions_in_manifest_provenance(
    tmp_path: Path,
) -> None:
    action = CorporateAction(
        request().instrument,
        date(2026, 7, 21),
        split_ratio=Decimal("1.2"),
        cash_dividend_per_share=Decimal("0.05"),
    )

    class ActionProvider(FakeProvider):
        def fetch_corporate_actions(self, requested: DailyBarRequest) -> Sequence[CorporateAction]:
            assert requested == request()
            return (action,)

    result = DataService(
        MarketStore(tmp_path),
        expected_sessions=lambda requested: pd.DatetimeIndex(["2026-07-20", "2026-07-21"]),
        require_corporate_actions=True,
        sleeper=lambda delay: None,
    ).sync_daily(request(), (ActionProvider("actions", [bars()]),), QualityMode.STRICT)

    assert result.corporate_actions == (action,)
    assert result.degradation_codes == ()
    assert result.manifest.provenance is not None
    assert result.manifest.provenance.corporate_actions == (action,)
    assert result.manifest.provenance.corporate_actions_status == "available"
    assert result.manifest.provenance.corporate_actions_provider == "actions"


def test_required_company_actions_try_all_providers_before_strict_failure(
    tmp_path: Path,
) -> None:
    action = CorporateAction(
        request().instrument,
        date(2026, 7, 21),
        cash_dividend_per_share=Decimal("0.05"),
    )

    class FailingActions(FakeProvider):
        def fetch_corporate_actions(self, requested: DailyBarRequest) -> Sequence[CorporateAction]:
            raise ConnectionError("offline action endpoint")

    class FallbackActions(FakeProvider):
        def fetch_corporate_actions(self, requested: DailyBarRequest) -> Sequence[CorporateAction]:
            return (action,)

    result = DataService(
        MarketStore(tmp_path),
        expected_sessions=lambda requested: pd.DatetimeIndex(["2026-07-20", "2026-07-21"]),
        require_corporate_actions=True,
        sleeper=lambda delay: None,
    ).sync_daily(
        request(),
        (FailingActions("daily", [bars()]), FallbackActions("fallback", [bars()])),
        QualityMode.STRICT,
    )

    assert result.corporate_actions == (action,)
    assert result.manifest.provenance is not None
    assert tuple(
        (item.provider, item.outcome, item.failure_category)
        for item in result.manifest.provenance.corporate_action_attempts
    ) == (
        ("daily", "failed", "network"),
        ("fallback", "selected", None),
    )


def test_required_company_actions_fail_closed_in_strict_and_degrade_explicitly(
    tmp_path: Path,
) -> None:
    strict = DataService(
        MarketStore(tmp_path / "strict"),
        expected_sessions=lambda requested: pd.DatetimeIndex(["2026-07-20", "2026-07-21"]),
        require_corporate_actions=True,
        sleeper=lambda delay: None,
    )

    with pytest.raises(ProviderError) as caught:
        strict.sync_daily(request(), (FakeProvider("daily", [bars()]),), QualityMode.STRICT)

    assert caught.value.kind is ProviderErrorKind.CAPABILITY
    assert list((tmp_path / "strict" / "manifests").iterdir()) == []

    degraded = DataService(
        MarketStore(tmp_path / "degraded"),
        expected_sessions=lambda requested: pd.DatetimeIndex(["2026-07-20", "2026-07-21"]),
        require_corporate_actions=True,
        sleeper=lambda delay: None,
    ).sync_daily(
        request(),
        (FakeProvider("daily", [bars()]),),
        QualityMode.DEGRADED,
    )

    assert degraded.corporate_actions == ()
    assert degraded.degradation_codes == ("CORPORATE_ACTIONS_UNAVAILABLE",)
    assert degraded.manifest.provenance is not None
    assert degraded.manifest.provenance.corporate_actions_status == "unavailable"
