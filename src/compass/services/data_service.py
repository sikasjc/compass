from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256
from time import sleep
from typing import TypeAlias
from zoneinfo import ZoneInfo

import pandas as pd  # type: ignore[import-untyped]

from compass.data.base import (
    DailyBarRequest,
    MarketDataProvider,
    ProviderCapabilityError,
    ProviderConfigurationError,
    ProviderError,
    ProviderErrorKind,
    translate_provider_error,
    default_instrument_type,
)
from compass.data.exchange_calendar import CalendarIdentity
from compass.data.quality import (
    DailyQualityGate,
    DataQualityError,
    QualityMode,
    QualityReport,
)
from compass.storage.market_store import DatasetManifest, MarketStore
from compass.services.safe_display import safe_display_text
from compass.domain.trading import CorporateAction
from compass.domain.market import AssetType
from compass.storage.market_store import (
    CorporateActionProviderAttempt,
    DailyProviderAttempt,
    DatasetProvenance,
)


ExpectedSessions: TypeAlias = Callable[[DailyBarRequest], pd.DatetimeIndex]
Sleeper: TypeAlias = Callable[[float], None]
CalendarIdentitySource: TypeAlias = Callable[[DailyBarRequest], CalendarIdentity]
CompletedSession: TypeAlias = Callable[[DailyBarRequest], date]
Clock: TypeAlias = Callable[[], datetime]
_RETRY_DELAYS = (0.25, 0.5, 1.0)
_RETRYABLE = {ProviderErrorKind.NETWORK, ProviderErrorKind.RATE_LIMIT}
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_INJECTED_CALENDAR_ID = sha256(b"injected-expected-sessions").hexdigest()


def _now() -> datetime:
    return datetime.now(tz=_SHANGHAI)


def _injected_calendar(request: DailyBarRequest) -> CalendarIdentity:
    return CalendarIdentity(
        _INJECTED_CALENDAR_ID,
        "injected",
        "expected-sessions-v1",
        request.start,
        request.end,
    )


@dataclass(frozen=True, slots=True)
class SyncResult:
    provider: str
    attempts: tuple[str, ...]
    manifest: DatasetManifest
    quality_report: QualityReport
    corporate_actions: tuple[CorporateAction, ...] = ()
    degradation_codes: tuple[str, ...] = ()
    downloaded_rows: int = 0
    reused_rows: int = 0
    remaining_requested_sessions: int = 0

    def __post_init__(self) -> None:
        for label, value in (
            ("downloaded_rows", self.downloaded_rows),
            ("reused_rows", self.reused_rows),
            ("remaining_requested_sessions", self.remaining_requested_sessions),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{label} must be a non-negative exact integer")


class DataService:
    def __init__(
        self,
        store: MarketStore,
        *,
        expected_sessions: ExpectedSessions | None = None,
        sleeper: Sleeper = sleep,
        calendar_identity: CalendarIdentitySource | None = None,
        completed_session: CompletedSession = lambda request: request.end,
        clock: Clock = _now,
        require_corporate_actions: bool = False,
        require_rule_attestation: bool = False,
    ) -> None:
        if expected_sessions is None:
            raise ProviderConfigurationError(
                "data_service", "an explicit expected sessions calendar is required"
            ) from None
        self.store = store
        self._expected_sessions = expected_sessions
        self._sleeper = sleeper
        self._calendar_identity = calendar_identity or _injected_calendar
        self._completed_session = completed_session
        self._clock = clock
        self._require_corporate_actions = require_corporate_actions
        self._require_rule_attestation = require_rule_attestation

    def sync_daily(
        self,
        request: DailyBarRequest,
        providers: Sequence[MarketDataProvider],
        mode: QualityMode | str,
    ) -> SyncResult:
        quality_mode = QualityMode.parse(mode)
        provider_list = tuple(providers)
        self._validate_providers(provider_list)
        expected = self._expected_sessions(request)
        gate = DailyQualityGate(expected_sessions=expected)
        attempts: list[str] = []
        provenance_attempts: list[DailyProviderAttempt] = []
        last_failure: ProviderError | DataQualityError | None = None

        for provider in provider_list:
            fetched: object = None
            fetch_succeeded = False
            for retry_number in range(len(_RETRY_DELAYS) + 1):
                attempts.append(provider.name)
                try:
                    fetched = provider.fetch_daily(request)
                except Exception as raw_error:
                    translated = translate_provider_error(provider.name, raw_error)
                    error = ProviderError(
                        translated.kind,
                        provider.name,
                        translated.message,
                    )
                    provenance_attempts.append(
                        DailyProviderAttempt(
                            provider.name,
                            "failed",
                            error.kind.value,
                        )
                    )
                else:
                    fetch_succeeded = True
                    break
                if error.kind not in _RETRYABLE:
                    raise error from None
                if retry_number == len(_RETRY_DELAYS):
                    last_failure = error
                    break
                self._sleeper(_RETRY_DELAYS[retry_number])

            if not fetch_succeeded:
                continue
            if not isinstance(fetched, pd.DataFrame):
                raise ProviderError(
                    ProviderErrorKind.MALFORMED_RESPONSE,
                    provider.name,
                    "provider did not return a daily DataFrame",
                ) from None

            fetched_at = self._checked_moment(self._clock(), label="data fetch clock")
            source_at = self._source_at(fetched.attrs.get("source_at"), fetched_at)
            raw_instrument_name = fetched.attrs.get("instrument_name")
            try:
                instrument_name = (
                    None
                    if raw_instrument_name is None
                    else safe_display_text(
                        raw_instrument_name,
                        label="provider instrument name",
                        maximum=128,
                    )
                )
            except (TypeError, ValueError):
                # Optional display metadata must never block valid backtest prices.
                instrument_name = None
            report = gate.evaluate(fetched, quality_mode)
            if not report.accepted:
                provenance_attempts.append(
                    DailyProviderAttempt(
                        provider.name,
                        "quality_rejected",
                        "data_quality",
                    )
                )
                last_failure = DataQualityError(
                    provider.name,
                    report,
                    tuple(attempts),
                )
                continue

            if self._require_rule_attestation and not self._has_rule_attestation(
                report.frame, request
            ):
                provenance_attempts.append(
                    DailyProviderAttempt(
                        provider.name,
                        "quality_rejected",
                        "market_rules",
                    )
                )
                last_failure = ProviderCapabilityError(
                    provider.name, "price_limit_rule_attestation"
                )
                continue

            provenance_attempts.append(DailyProviderAttempt(provider.name, "selected", None))
            corporate_actions, action_provider, action_attempts = self._corporate_actions(
                request,
                provider_list,
                selected=provider,
                mode=quality_mode,
            )
            degradation_codes = (
                ("CORPORATE_ACTIONS_UNAVAILABLE",)
                if self._require_corporate_actions and action_provider is None
                else ()
            )
            calendar = self._calendar_identity(request)
            completed_through = self._completed_session(request)
            self._validate_calendar(calendar, request, completed_through)
            expected_days = {timestamp.date() for timestamp in expected}
            calendar_sessions = tuple(timestamp.date() for timestamp in expected)
            stored_days = {timestamp.date() for timestamp in report.frame.index}
            missing_sessions = tuple(sorted(expected_days - stored_days))
            daily_complete = completed_through in stored_days
            provenance = DatasetProvenance(
                tuple(provenance_attempts),
                provider.name,
                fetched_at,
                source_at,
                calendar,
                completed_through,
                daily_complete,
                "available" if action_provider is not None else "unavailable",
                action_provider,
                corporate_actions,
                calendar_sessions,
                missing_sessions,
                action_attempts,
            )
            manifest = self.store.write_daily(
                str(request.instrument),
                report.frame,
                provider=provider.name,
                quality_report_json=report.to_json(),
                provenance_json=provenance.to_json(),
                instrument_name=instrument_name,
            )
            return SyncResult(
                provider=provider.name,
                attempts=tuple(attempts),
                manifest=manifest,
                quality_report=report,
                corporate_actions=corporate_actions,
                degradation_codes=degradation_codes,
                downloaded_rows=report.output_rows,
                remaining_requested_sessions=len(missing_sessions),
            )

        if isinstance(last_failure, DataQualityError):
            raise DataQualityError(
                last_failure.provider,
                last_failure.report,
                tuple(attempts),
            )
        if last_failure is not None:
            raise last_failure from None
        raise RuntimeError("daily synchronization ended without a result")

    def sync_daily_incremental(
        self,
        request: DailyBarRequest,
        providers: Sequence[MarketDataProvider],
        mode: QualityMode | str,
        *,
        trusted_manifest: DatasetManifest | None = None,
    ) -> SyncResult:
        """Reuse a trusted immutable dataset and fetch only missing requested sessions."""
        if trusted_manifest is None:
            return self.sync_daily(request, providers, mode)
        if type(trusted_manifest) is not DatasetManifest:
            raise TypeError("trusted manifest must be an exact DatasetManifest or None")
        loaded = self.store.load_manifest(trusted_manifest.manifest_id)
        if loaded != trusted_manifest or loaded.instrument != str(request.instrument):
            raise ValueError("trusted manifest does not match the daily request")

        trusted = self.store.read_manifest(loaded.manifest_id)
        if trusted.empty:
            return self.sync_daily(request, providers, mode)
        requested_sessions = self._expected_sessions(request)
        requested_days = tuple(timestamp.date() for timestamp in requested_sessions)
        trusted_days = {timestamp.date() for timestamp in trusted.index}
        missing_days = tuple(day for day in requested_days if day not in trusted_days)
        full_request = DailyBarRequest(
            request.instrument,
            min(request.start, trusted.index[0].date()),
            max(request.end, trusted.index[-1].date()),
        )
        full_expected = self._expected_sessions(full_request)
        quality_mode = QualityMode.parse(mode)

        if not missing_days:
            report = DailyQualityGate(expected_sessions=full_expected).evaluate(
                trusted,
                quality_mode,
            )
            if not report.accepted:
                raise DataQualityError(loaded.provider, report) from None
            provenance = loaded.provenance
            refreshed_manifest = self.store.write_daily(
                str(request.instrument),
                report.frame,
                provider=loaded.provider,
                quality_report_json=report.to_json(),
                provenance_json=loaded.provenance_json,
                instrument_name=loaded.instrument_name,
            )
            return SyncResult(
                provider=loaded.provider,
                attempts=(),
                manifest=refreshed_manifest,
                quality_report=report,
                corporate_actions=(() if provenance is None else provenance.corporate_actions),
                degradation_codes=(
                    ("CORPORATE_ACTIONS_UNAVAILABLE",)
                    if self._require_corporate_actions
                    and (provenance is None or provenance.corporate_actions_status == "unavailable")
                    else ()
                ),
                reused_rows=len(
                    trusted.loc[pd.Timestamp(request.start) : pd.Timestamp(request.end)]
                ),
            )

        provider_list = tuple(providers)
        self._validate_providers(provider_list)
        fetched_results: list[SyncResult] = []
        skipped_quality_attempts: list[DailyProviderAttempt] = []
        skipped_attempt_names: list[str] = []
        for missing_request in self._missing_requests(
            request,
            requested_days,
            missing_days,
            split_by_year=provider_list[0].name == "baostock",
        ):
            try:
                fetched_results.append(
                    self.sync_daily(missing_request, provider_list, quality_mode)
                )
            except DataQualityError as error:
                if quality_mode is not QualityMode.DEGRADED:
                    raise
                names = error.attempts or (error.provider,)
                skipped_attempt_names.extend(names)
                skipped_quality_attempts.extend(
                    DailyProviderAttempt(name, "quality_rejected", "data_quality") for name in names
                )

        merge_frames = [
            trusted,
            *(result.quality_report.frame for result in fetched_results),
        ]
        shared_columns = tuple(
            column
            for column in merge_frames[0].columns
            if all(column in frame.columns for frame in merge_frames[1:])
        )
        merged = pd.concat(
            [frame.loc[:, list(shared_columns)] for frame in merge_frames],
            axis=0,
            sort=False,
        )
        merged = merged.loc[~merged.index.duplicated(keep="last")].sort_index()
        merged = merged.loc[pd.Timestamp(full_request.start) : pd.Timestamp(full_request.end)]
        report = DailyQualityGate(expected_sessions=full_expected).evaluate(
            merged,
            quality_mode,
        )
        selected_provider_name = (
            fetched_results[-1].provider if fetched_results else loaded.provider
        )
        if not report.accepted:
            raise DataQualityError(
                selected_provider_name,
                report,
                tuple(
                    [item for result in fetched_results for item in result.attempts]
                    + skipped_attempt_names
                ),
            ) from None
        if self._require_rule_attestation and not self._has_rule_attestation(
            report.frame,
            full_request,
        ):
            raise ProviderCapabilityError(
                selected_provider_name,
                "price_limit_rule_attestation",
            ) from None

        selected_provider = next(
            provider for provider in provider_list if provider.name == selected_provider_name
        )
        corporate_actions, action_provider, action_attempts = self._corporate_actions(
            full_request,
            provider_list,
            selected=selected_provider,
            mode=quality_mode,
        )
        degradation_codes = (
            ("CORPORATE_ACTIONS_UNAVAILABLE",)
            if self._require_corporate_actions and action_provider is None
            else ()
        )
        provenances = tuple(
            provenance
            for result in fetched_results
            if (provenance := result.manifest.provenance) is not None
        )
        attempts = tuple(
            [attempt for provenance in provenances for attempt in provenance.daily_attempts]
            + skipped_quality_attempts
        )
        selected_attempt = DailyProviderAttempt(
            selected_provider_name,
            "selected",
            None,
        )
        if not attempts or attempts[-1] != selected_attempt:
            attempts = (*attempts, selected_attempt)
        trusted_provenance = loaded.provenance
        fetched_at = max(
            (
                provenance.fetched_at
                for provenance in (
                    *provenances,
                    *(() if trusted_provenance is None else (trusted_provenance,)),
                )
            ),
            default=self._checked_moment(self._clock(), label="data fetch clock"),
        )
        source_values = tuple(
            provenance.source_at
            for provenance in (
                *provenances,
                *(() if trusted_provenance is None else (trusted_provenance,)),
            )
            if provenance.source_at is not None
        )
        source_at = max(source_values, default=None)
        calendar = self._calendar_identity(full_request)
        completed_through = self._completed_session(full_request)
        self._validate_calendar(calendar, full_request, completed_through)
        calendar_sessions = tuple(timestamp.date() for timestamp in full_expected)
        stored_days = {timestamp.date() for timestamp in report.frame.index}
        missing_sessions = tuple(day for day in calendar_sessions if day not in stored_days)
        provenance = DatasetProvenance(
            attempts,
            selected_provider_name,
            fetched_at,
            source_at,
            calendar,
            completed_through,
            completed_through in stored_days,
            "available" if action_provider is not None else "unavailable",
            action_provider,
            corporate_actions,
            calendar_sessions,
            missing_sessions,
            action_attempts,
        )
        instrument_name = next(
            (
                result.manifest.instrument_name
                for result in reversed(fetched_results)
                if result.manifest.instrument_name is not None
            ),
            loaded.instrument_name,
        )
        manifest = self.store.write_daily(
            str(request.instrument),
            report.frame,
            provider=selected_provider_name,
            quality_report_json=report.to_json(),
            provenance_json=provenance.to_json(),
            instrument_name=instrument_name,
        )
        requested_stored_days = {
            timestamp.date()
            for timestamp in report.frame.loc[
                pd.Timestamp(request.start) : pd.Timestamp(request.end)
            ].index
        }
        return SyncResult(
            provider=selected_provider_name,
            attempts=tuple(
                [item for result in fetched_results for item in result.attempts]
                + skipped_attempt_names
            ),
            manifest=manifest,
            quality_report=report,
            corporate_actions=corporate_actions,
            degradation_codes=degradation_codes,
            downloaded_rows=sum(result.quality_report.output_rows for result in fetched_results),
            reused_rows=len(trusted.loc[pd.Timestamp(request.start) : pd.Timestamp(request.end)]),
            remaining_requested_sessions=sum(
                day not in requested_stored_days for day in requested_days
            ),
        )

    @staticmethod
    def _missing_requests(
        request: DailyBarRequest,
        requested_days: tuple[date, ...],
        missing_days: tuple[date, ...],
        *,
        split_by_year: bool,
    ) -> tuple[DailyBarRequest, ...]:
        positions = {day: position for position, day in enumerate(requested_days)}
        groups: list[list[date]] = []
        for day in missing_days:
            if (
                not groups
                or positions[day] != positions[groups[-1][-1]] + 1
                or (split_by_year and day.year != groups[-1][-1].year)
            ):
                groups.append([day])
            else:
                groups[-1].append(day)
        return tuple(DailyBarRequest(request.instrument, group[0], group[-1]) for group in groups)

    def _corporate_actions(
        self,
        request: DailyBarRequest,
        providers: tuple[MarketDataProvider, ...],
        *,
        selected: MarketDataProvider,
        mode: QualityMode,
    ) -> tuple[
        tuple[CorporateAction, ...],
        str | None,
        tuple[CorporateActionProviderAttempt, ...],
    ]:
        ordered = (selected,) + tuple(item for item in providers if item is not selected)
        attempts: list[CorporateActionProviderAttempt] = []
        last_error: ProviderError | None = None
        for provider in ordered:
            operation = getattr(provider, "fetch_corporate_actions", None)
            if not callable(operation):
                attempts.append(
                    CorporateActionProviderAttempt(provider.name, "unsupported", "capability")
                )
                continue
            try:
                raw = operation(request)
                if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
                    raise TypeError("corporate actions must be a sequence")
                actions = tuple(raw)
                if any(type(item) is not CorporateAction for item in actions):
                    raise TypeError("corporate actions contain an invalid value")
                if any(
                    item.instrument != request.instrument
                    or not request.start <= item.ex_date <= request.end
                    for item in actions
                ):
                    raise ValueError("corporate action is outside the request")
                ordered_actions = tuple(
                    sorted(
                        actions,
                        key=lambda item: (
                            item.ex_date,
                            item.split_ratio,
                            item.cash_dividend_per_share,
                        ),
                    )
                )
                if len(set(ordered_actions)) != len(ordered_actions):
                    raise ValueError("corporate actions contain duplicates")
                attempts.append(CorporateActionProviderAttempt(provider.name, "selected", None))
                return ordered_actions, provider.name, tuple(attempts)
            except (NotImplementedError, ProviderCapabilityError):
                attempts.append(
                    CorporateActionProviderAttempt(provider.name, "unsupported", "capability")
                )
                continue
            except Exception as raw_error:
                translated = translate_provider_error(provider.name, raw_error)
                last_error = ProviderError(
                    translated.kind,
                    provider.name,
                    translated.message,
                )
                attempts.append(
                    CorporateActionProviderAttempt(provider.name, "failed", last_error.kind.value)
                )
                continue
        if self._require_corporate_actions and mode is QualityMode.STRICT:
            if last_error is not None:
                raise last_error from None
            raise ProviderCapabilityError("data_service", "corporate_actions") from None
        return (), None, tuple(attempts)

    @staticmethod
    def _has_rule_attestation(
        frame: pd.DataFrame,
        request: DailyBarRequest,
    ) -> bool:
        asset_type = default_instrument_type(request.instrument)
        if asset_type is AssetType.INDEX:
            return True
        if {"limit_up", "limit_down"}.issubset(frame.columns):
            if frame[["limit_up", "limit_down"]].notna().all().all():
                return True
        required = {"price_limit_rate", "price_limit_rule_id"}
        if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
            return False
        if asset_type is AssetType.ETF:
            return True
        return (
            "listing_regime_known" in frame
            and frame["listing_regime_known"].map(lambda value: type(value) is bool and value).all()
            and "risk_warning" in frame
            and frame["risk_warning"].map(lambda value: type(value) is bool).all()
        )

    @staticmethod
    def _checked_moment(value: object, *, label: str) -> datetime:
        if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
            raise TypeError(f"{label} must return a timezone-aware datetime")
        return value

    @classmethod
    def _source_at(cls, value: object, fetched_at: datetime) -> datetime | None:
        if isinstance(value, pd.Timestamp):
            value = value.to_pydatetime()
        if type(value) is not datetime:
            return None
        if value.tzinfo is None or value.utcoffset() is None or value > fetched_at:
            return None
        return value

    @staticmethod
    def _validate_calendar(
        calendar: object,
        request: DailyBarRequest,
        completed_through: object,
    ) -> None:
        if type(calendar) is not CalendarIdentity:
            raise TypeError("calendar identity must be an exact CalendarIdentity")
        if type(completed_through) is not date:
            raise TypeError("completed session must be an exact date")
        if not calendar.covered_from <= request.start <= request.end <= calendar.covered_to:
            raise ValueError("calendar identity does not cover the daily request")
        if not request.start <= completed_through <= request.end:
            raise ValueError("completed session is outside the daily request")

    @staticmethod
    def _validate_providers(providers: tuple[MarketDataProvider, ...]) -> None:
        if not providers:
            raise ValueError("daily synchronization requires at least one provider")
        names = [provider.name for provider in providers]
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("provider names must be non-empty text")
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate provider name: {duplicates[0]}")
