from __future__ import annotations

from collections.abc import Sequence
from datetime import date
import json
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import requests

from compass.data.base import (
    DailyBarRequest,
    InstrumentTypeResolver,
    ProviderCapabilityError,
    default_instrument_type,
    raise_translated_provider_error,
    resolve_instrument_type,
)
from compass.data.instrument_rule_attestation import (
    attach_attestation_columns,
    attest_etf_name,
)
from compass.data.network_timeout import (
    DEFAULT_MARKET_TIMEOUT_SECONDS,
    validate_market_timeout_seconds,
)
from compass.data.normalize import normalize_daily
from compass.domain.market import AssetType, Exchange, InstrumentId
from compass.domain.trading import CorporateAction
from compass.services.diagnostic_log import diagnostic_request


_TENCENT_HISTORY_ENDPOINT = (
    "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
)
_TENCENT_DAILY_COLUMNS = (
    "date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
)
_MAX_YEARS_PER_REQUEST = 2


def _tencent_symbol(instrument: InstrumentId) -> str:
    prefix = "sh" if instrument.exchange is Exchange.SSE else "sz"
    return f"{prefix}{instrument.code}"


def _date_chunks(start: date, end: date) -> tuple[tuple[date, date], ...]:
    chunks: list[tuple[date, date]] = []
    first_year = start.year
    while first_year <= end.year:
        last_year = min(first_year + _MAX_YEARS_PER_REQUEST - 1, end.year)
        chunk_start = max(start, date(first_year, 1, 1))
        chunk_end = min(end, date(last_year, 12, 31))
        chunks.append((chunk_start, chunk_end))
        first_year = last_year + 1
    return tuple(chunks)


def _instrument_name(payload: object, symbol: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    quote = payload.get("qt")
    if not isinstance(quote, dict):
        return None
    fields = quote.get(symbol)
    if not isinstance(fields, list) or len(fields) < 3:
        return None
    name = fields[1]
    code = fields[2]
    if type(name) is not str or not name.strip() or str(code) != symbol[2:]:
        return None
    return name.strip()


def _parse_history_response(response: requests.Response, symbol: str) -> pd.DataFrame:
    response.raise_for_status()
    marker = response.text.find("={")
    if marker < 0:
        raise ValueError("Tencent history response is missing its JSON payload")
    decoded = json.loads(response.text[marker + 1 :])
    if not isinstance(decoded, dict) or not isinstance(decoded.get("data"), dict):
        raise TypeError("Tencent history response data is malformed")
    payload = decoded["data"].get(symbol)
    if not isinstance(payload, dict):
        raise ValueError("Tencent history response does not contain the requested instrument")
    raw_rows = payload.get("day", [])
    if not isinstance(raw_rows, list):
        raise TypeError("Tencent history rows are malformed")
    records: list[dict[str, object]] = []
    for row in raw_rows:
        if not isinstance(row, list) or len(row) < 9:
            raise TypeError("Tencent history row is malformed")
        records.append(
            {
                "date": row[0],
                "open": row[1],
                "close": row[2],
                "high": row[3],
                "low": row[4],
                # Tencent reports volume in hands and turnover amount in 10,000 yuan.
                "volume": row[5],
                "amount": row[8],
            }
        )
    frame = pd.DataFrame.from_records(records, columns=_TENCENT_DAILY_COLUMNS)
    for column in ("open", "close", "high", "low", "volume", "amount"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["volume"] *= 100.0
    frame["amount"] *= 10_000.0
    name = _instrument_name(payload, symbol)
    if name is not None:
        frame.attrs["instrument_name"] = name
    return frame


class TencentProvider:
    """Tencent daily history adapter for Shanghai and Shenzhen instruments."""

    name = "tencent"

    def __init__(
        self,
        *,
        session: Any | None = None,
        instrument_type_resolver: InstrumentTypeResolver = default_instrument_type,
        request_timeout_seconds: int = DEFAULT_MARKET_TIMEOUT_SECONDS,
    ) -> None:
        self._session = requests if session is None else session
        self._instrument_type_resolver = instrument_type_resolver
        self._request_timeout_seconds = validate_market_timeout_seconds(
            request_timeout_seconds
        )

    def set_request_timeout(self, seconds: int) -> None:
        self._request_timeout_seconds = validate_market_timeout_seconds(seconds)

    def fetch_daily(self, request: DailyBarRequest) -> pd.DataFrame:
        try:
            asset_type = resolve_instrument_type(
                self.name,
                self._instrument_type_resolver,
                request.instrument,
            )
            symbol = _tencent_symbol(request.instrument)
            frames: list[pd.DataFrame] = []
            instrument_name: str | None = None
            for chunk_start, chunk_end in _date_chunks(request.start, request.end):
                parameters = {
                    "_var": f"kline_day{chunk_start.year}",
                    "param": (
                        f"{symbol},day,{chunk_start.isoformat()},"
                        f"{chunk_end.isoformat()},640,"
                    ),
                    "r": "0.8205512681390605",
                }

                def fetch_chunk() -> pd.DataFrame:
                    response = self._session.get(
                        _TENCENT_HISTORY_ENDPOINT,
                        params=parameters,
                        timeout=self._request_timeout_seconds,
                    )
                    return _parse_history_response(response, symbol)

                frame = diagnostic_request(
                    provider="腾讯证券",
                    transport="HTTP",
                    operation="stock_zh_a_hist_tx",
                    endpoint=_TENCENT_HISTORY_ENDPOINT,
                    details=(
                        f"instrument={request.instrument} symbol={symbol} "
                        f"start={chunk_start} end={chunk_end} adjust=none "
                        f"timeout={self._request_timeout_seconds}s"
                    ),
                    call=fetch_chunk,
                )
                raw_name = frame.attrs.get("instrument_name")
                if type(raw_name) is str and raw_name.strip():
                    instrument_name = raw_name.strip()
                frames.append(frame)

            combined = pd.concat(frames, ignore_index=True)
            combined = combined.drop_duplicates(subset="date", keep="last")
            result = normalize_daily(combined, {}).loc[
                pd.Timestamp(request.start) : pd.Timestamp(request.end)
            ]
            if instrument_name is not None:
                result.attrs["instrument_name"] = instrument_name
            if asset_type is AssetType.ETF:
                if instrument_name is None:
                    raise ValueError("Tencent ETF name metadata is unavailable")
                attestation = attest_etf_name(instrument_name)
                attach_attestation_columns(result, (attestation,) * len(result.index))
            return result
        except Exception as error:
            raise_translated_provider_error(self.name, error)

    def fetch_snapshot(self, instruments: Sequence[InstrumentId]) -> pd.DataFrame:
        del instruments
        raise ProviderCapabilityError(self.name, "snapshot")

    def fetch_corporate_actions(self, request: DailyBarRequest) -> Sequence[CorporateAction]:
        del request
        raise ProviderCapabilityError(self.name, "corporate_actions")
