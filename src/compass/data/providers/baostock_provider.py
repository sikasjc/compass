from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from threading import RLock
from types import TracebackType
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from compass.data.base import (
    DailyBarRequest,
    InstrumentTypeResolver,
    ProviderCapabilityError,
    ProviderError,
    ProviderErrorKind,
    default_instrument_type,
    raise_translated_provider_error,
    resolve_instrument_type,
)
from compass.data.normalize import normalize_daily
from compass.data.network_timeout import (
    DEFAULT_MARKET_TIMEOUT_SECONDS,
    TimedSocketModule,
    validate_market_timeout_seconds,
)
from compass.data.instrument_rule_attestation import (
    attach_attestation_columns,
    attest_etf_name,
    attest_stock_session,
)
from compass.domain.market import AssetType, Exchange, InstrumentId
from compass.domain.trading import CorporateAction
from compass.services.diagnostic_log import diagnostic_request


_BAOSTOCK_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,adjustflag,tradestatus,isST"
)
_BAOSTOCK_INDEX_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,pctChg"
_BAOSTOCK_DAILY_MAPPING = {
    "date": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "preclose": "previous_close",
    "volume": "volume",
    "amount": "amount",
    "adjustflag": "adjust_flag",
    "suspended": "suspended",
    "risk_warning": "risk_warning",
}

_AUTHENTICATION_CODES = {
    "10001001",
    "10001002",
    "10001003",
    "10001006",
    "10001007",
    "10001008",
    "10001009",
    "10001011",
}
_RATE_LIMIT_CODES = {"10001005", "10004016"}
_NETWORK_CODES = {f"1000200{suffix}" for suffix in range(1, 9)}
_BAOSTOCK_ENDPOINT = "tcp://public-api.baostock.com:10030"


def _baostock_symbol(instrument: InstrumentId) -> str:
    prefix = "sh" if instrument.exchange is Exchange.SSE else "sz"
    return f"{prefix}.{instrument.code}"


class BaostockProvider:
    name = "baostock"
    calendar_version = "baostock-query-trade-dates-v1"

    def __init__(
        self,
        *,
        client: Any | None = None,
        instrument_type_resolver: InstrumentTypeResolver = default_instrument_type,
        request_timeout_seconds: int = DEFAULT_MARKET_TIMEOUT_SECONDS,
    ) -> None:
        checked_timeout = validate_market_timeout_seconds(request_timeout_seconds)
        socket_module: TimedSocketModule | None = None
        socket_context: Any | None = None
        if client is None:
            import baostock as baostock_client  # type: ignore[import-untyped]
            import baostock.common.context as baostock_context  # type: ignore[import-untyped]
            import baostock.util.socketutil as socket_util  # type: ignore[import-untyped]

            resolved_client: Any = baostock_client
            socket_module = TimedSocketModule(checked_timeout)
            socket_util.socket = socket_module
            socket_context = baostock_context
        else:
            resolved_client = client
        self._client = resolved_client
        self._instrument_type_resolver = instrument_type_resolver
        self._authenticated = False
        self._session_lock = RLock()
        self._request_timeout_seconds = checked_timeout
        self._socket_module = socket_module
        self._socket_context = socket_context

    def set_request_timeout(self, seconds: int) -> None:
        checked = validate_market_timeout_seconds(seconds)
        with self._session_lock:
            self._request_timeout_seconds = checked
            if self._socket_module is not None:
                self._socket_module.set_timeout(checked)
            active_socket = (
                None
                if self._socket_context is None
                else getattr(self._socket_context, "default_socket", None)
            )
            if active_socket is not None:
                reset = getattr(active_socket, "reset_deadline", None)
                if callable(reset):
                    reset(checked)
                else:
                    active_socket.settimeout(checked)

    def _reset_active_socket_deadline(self) -> None:
        active_socket = (
            None
            if self._socket_context is None
            else getattr(self._socket_context, "default_socket", None)
        )
        if active_socket is None:
            return
        reset = getattr(active_socket, "reset_deadline", None)
        if callable(reset):
            reset(self._request_timeout_seconds)
        else:
            active_socket.settimeout(self._request_timeout_seconds)

    def __enter__(self) -> BaostockProvider:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _result_error(self, code: object, operation: str) -> ProviderError:
        normalized = str(code)
        if normalized in _AUTHENTICATION_CODES:
            kind = ProviderErrorKind.AUTHENTICATION
        elif normalized in _RATE_LIMIT_CODES:
            kind = ProviderErrorKind.RATE_LIMIT
        elif normalized in _NETWORK_CODES:
            kind = ProviderErrorKind.NETWORK
        else:
            kind = ProviderErrorKind.MALFORMED_RESPONSE
        return ProviderError(kind, self.name, f"upstream {operation} was rejected")

    def _authenticate(self) -> None:
        if self._authenticated:
            return
        login = getattr(self._client, "login", None)
        if login is None:
            self._authenticated = True
            return
        result = diagnostic_request(
            provider="BaoStock",
            transport="TCP",
            operation="login",
            endpoint=_BAOSTOCK_ENDPOINT,
            details=f"session=anonymous timeout={self._request_timeout_seconds}s",
            call=login,
        )
        if str(result.error_code) != "0":
            raise self._result_error(result.error_code, "login")
        self._authenticated = True

    def close(self) -> None:
        if not self._authenticated:
            return
        logout = getattr(self._client, "logout", None)
        if logout is None:
            self._authenticated = False
            return
        try:
            self._reset_active_socket_deadline()
            result = diagnostic_request(
                provider="BaoStock",
                transport="TCP",
                operation="logout",
                endpoint=_BAOSTOCK_ENDPOINT,
                details=f"session=current timeout={self._request_timeout_seconds}s",
                call=logout,
            )
            if result is not None and str(result.error_code) != "0":
                raise self._result_error(result.error_code, "logout")
            self._authenticated = False
        except Exception as error:
            raise_translated_provider_error(self.name, error)

    def fetch_daily(self, request: DailyBarRequest) -> pd.DataFrame:
        with self._session_lock:
            return self._fetch_daily_locked(request)

    def _fetch_daily_locked(self, request: DailyBarRequest) -> pd.DataFrame:
        try:
            asset_type = resolve_instrument_type(
                self.name, self._instrument_type_resolver, request.instrument
            )
            self._authenticate()
            symbol = _baostock_symbol(request.instrument)
            fields = (
                _BAOSTOCK_INDEX_FIELDS if asset_type is AssetType.INDEX else _BAOSTOCK_FIELDS
            )
            self._reset_active_socket_deadline()
            source = diagnostic_request(
                provider="BaoStock",
                transport="TCP",
                operation="query_history_k_data_plus",
                endpoint=_BAOSTOCK_ENDPOINT,
                details=(
                    f"instrument={request.instrument} symbol={symbol} "
                    f"start={request.start} end={request.end} frequency=d adjustflag=3 "
                    f"timeout={self._request_timeout_seconds}s"
                ),
                call=lambda: self._client.query_history_k_data_plus(
                    symbol,
                    fields,
                    start_date=request.start.isoformat(),
                    end_date=request.end.isoformat(),
                    frequency="d",
                    adjustflag="3",
                ),
            )
            if str(source.error_code) != "0":
                raise self._result_error(source.error_code, "query")
            rows: list[list[str]] = []
            while source.next():
                rows.append(source.get_row_data())
            frame = pd.DataFrame(rows, columns=source.fields)
            required_numeric = [
                "open",
                "high",
                "low",
                "close",
                "preclose",
                "volume",
                "amount",
            ]
            for column in required_numeric:
                frame[column] = pd.to_numeric(frame[column], errors="raise")
            if asset_type is AssetType.INDEX:
                frame["suspended"] = False
                frame["risk_warning"] = False
            else:
                frame["suspended"] = frame["tradestatus"].astype(str) != "1"
                if not frame["isST"].astype(str).isin(("0", "1")).all():
                    raise ValueError("BaoStock isST values must be 0 or 1")
                frame["risk_warning"] = frame["isST"].astype(str) == "1"
            result = normalize_daily(frame, _BAOSTOCK_DAILY_MAPPING).loc[
                pd.Timestamp(request.start) : pd.Timestamp(request.end)
            ]
            result["exchange_reference_price"] = result["previous_close"]
            basic = (
                None if asset_type is AssetType.INDEX else self._basic_metadata(request.instrument)
            )
            if basic is not None:
                name, ipo_date = basic
                result.attrs["instrument_name"] = name.strip()
                if asset_type is AssetType.ETF:
                    attestation = attest_etf_name(name)
                    attach_attestation_columns(
                        result,
                        (attestation,) * len(result.index),
                    )
                elif ipo_date is not None:
                    standard_from = self._stock_standard_from(ipo_date)
                    attach_attestation_columns(
                        result,
                        tuple(
                            attest_stock_session(
                                request.instrument,
                                timestamp.date(),
                                standard_from=standard_from,
                                risk_warning=bool(result.loc[timestamp, "risk_warning"]),
                            )
                            for timestamp in result.index
                        ),
                    )
            return result
        except Exception as error:
            self._authenticated = False
            raise_translated_provider_error(self.name, error)

    def _basic_metadata(self, instrument: InstrumentId) -> tuple[str, date | None] | None:
        operation = getattr(self._client, "query_stock_basic", None)
        if not callable(operation):
            return None
        symbol = _baostock_symbol(instrument)
        self._reset_active_socket_deadline()
        source = diagnostic_request(
            provider="BaoStock",
            transport="TCP",
            operation="query_stock_basic",
            endpoint=_BAOSTOCK_ENDPOINT,
            details=(
                f"instrument={instrument} symbol={symbol} "
                f"timeout={self._request_timeout_seconds}s"
            ),
            call=lambda: operation(code=symbol),
        )
        if str(source.error_code) != "0":
            raise self._result_error(source.error_code, "stock basic query")
        fields = tuple(source.fields)
        required = {"code", "code_name", "ipoDate"}
        if not required.issubset(fields):
            raise TypeError("BaoStock basic metadata response is malformed")
        rows: list[list[str]] = []
        while source.next():
            rows.append(source.get_row_data())
        if len(rows) != 1 or len(rows[0]) != len(fields):
            raise ValueError("BaoStock basic metadata must identify one instrument")
        row = dict(zip(fields, rows[0], strict=True))
        if row["code"] != _baostock_symbol(instrument) or not row["code_name"]:
            raise ValueError("BaoStock basic metadata instrument mismatch")
        ipo_date = None if not row["ipoDate"] else date.fromisoformat(row["ipoDate"])
        return row["code_name"], ipo_date

    def _stock_standard_from(self, ipo_date: date) -> date:
        sessions = self.fetch_exchange_sessions(ipo_date, ipo_date + timedelta(days=45))
        if len(sessions) < 6:
            raise ValueError("BaoStock listing sessions are unavailable")
        return sessions[5]

    def fetch_snapshot(self, instruments: Sequence[InstrumentId]) -> pd.DataFrame:
        raise ProviderCapabilityError(self.name, "snapshot")

    def fetch_exchange_sessions(self, start: date, end: date) -> tuple[date, ...]:
        try:
            if type(start) is not date or type(end) is not date or start > end:
                raise ValueError("invalid exchange calendar range")
            self._authenticate()
            self._reset_active_socket_deadline()
            source = diagnostic_request(
                provider="BaoStock",
                transport="TCP",
                operation="query_trade_dates",
                endpoint=_BAOSTOCK_ENDPOINT,
                details=(
                    f"start={start} end={end} timeout={self._request_timeout_seconds}s"
                ),
                call=lambda: self._client.query_trade_dates(
                    start_date=start.isoformat(),
                    end_date=end.isoformat(),
                ),
            )
            if str(source.error_code) != "0":
                raise self._result_error(source.error_code, "calendar query")
            if tuple(source.fields) != ("calendar_date", "is_trading_day"):
                raise TypeError("BaoStock calendar response is malformed")
            sessions: list[date] = []
            while source.next():
                row = source.get_row_data()
                if type(row) is not list or len(row) != 2 or row[1] not in {"0", "1"}:
                    raise TypeError("BaoStock calendar row is malformed")
                day = date.fromisoformat(row[0])
                if row[1] == "1":
                    sessions.append(day)
            checked = tuple(sessions)
            if not checked or tuple(sorted(set(checked))) != checked:
                raise ValueError("BaoStock calendar response has invalid sessions")
            return checked
        except Exception as error:
            self._authenticated = False
            raise_translated_provider_error(self.name, error)

    def fetch_corporate_actions(self, request: DailyBarRequest) -> Sequence[CorporateAction]:
        raise ProviderCapabilityError(self.name, "corporate_actions")
