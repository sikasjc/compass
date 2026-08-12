from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, time
from math import isfinite
from numbers import Real
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd  # type: ignore[import-untyped]

from compass.data.base import (
    DailyBarRequest,
    InstrumentTypeResolver,
    ProviderCapabilityError,
    default_instrument_type,
    raise_translated_provider_error,
    resolve_instrument_type,
)
from compass.data.normalize import normalize_daily
from compass.data.network_timeout import (
    DEFAULT_MARKET_TIMEOUT_SECONDS,
    requests_call_with_timeout,
    validate_market_timeout_seconds,
)
from compass.data.instrument_rule_attestation import (
    attach_attestation_columns,
    attest_etf_name,
)
from compass.domain.market import AssetType, InstrumentId
from compass.domain.trading import CorporateAction
from compass.services.diagnostic_log import diagnostic_request


_AKSHARE_DAILY_MAPPING = {
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
}
_AKSHARE_INDEX_DAILY_MAPPING = {
    "date": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "amount": "amount",
}
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_SNAPSHOT_COLUMNS = (
    "instrument",
    "timestamp",
    "source_at",
    "fetched_at",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
)
_STOCK_SNAPSHOT_MAPPING = {
    "今开": "open",
    "最高": "high",
    "最低": "low",
    "最新价": "close",
    "成交量": "volume",
    "成交额": "amount",
}
_ETF_SNAPSHOT_MAPPING = {
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
    "最新价": "close",
    "成交量": "volume",
    "成交额": "amount",
}
_EASTMONEY_HISTORY_ENDPOINT = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
_EASTMONEY_ETF_SPOT_ENDPOINT = "https://88.push2.eastmoney.com/api/qt/clist/get"
_EASTMONEY_STOCK_SPOT_ENDPOINT = "https://82.push2.eastmoney.com/api/qt/clist/get"
_SINA_CALENDAR_ENDPOINT = "https://finance.sina.com.cn/realstock/company/klc_td_sh.txt"


def _akshare_symbol(instrument: InstrumentId) -> str:
    return instrument.code


def _akshare_index_symbol(instrument: InstrumentId) -> str:
    prefix = "sh" if instrument.exchange.value == "SSE" else "sz"
    return f"{prefix}{instrument.code}"


def _etf_name_from_spot(
    client: Any,
    instrument: InstrumentId,
    request_timeout_seconds: int,
) -> str:
    operation = getattr(client, "fund_etf_spot_em", None)
    if not callable(operation):
        raise ProviderCapabilityError("akshare", "etf_rule_metadata")
    source = diagnostic_request(
        provider="东方财富",
        transport="HTTP",
        operation="fund_etf_spot_em",
        endpoint=_EASTMONEY_ETF_SPOT_ENDPOINT,
        details=f"instrument={instrument} timeout={request_timeout_seconds}s",
        call=lambda: requests_call_with_timeout(request_timeout_seconds, operation),
    )
    if not isinstance(source, pd.DataFrame) or source.columns.has_duplicates:
        raise TypeError("AKShare ETF metadata response is malformed")
    code_column = next((item for item in ("代码", "基金代码") if item in source.columns), None)
    name_column = next((item for item in ("名称", "基金简称") if item in source.columns), None)
    if code_column is None or name_column is None:
        raise TypeError("AKShare ETF metadata fields are unavailable")
    matches = source.loc[source[code_column].map(str) == instrument.code]
    if len(matches) != 1:
        raise ValueError("AKShare ETF metadata must identify one instrument")
    name = matches.iloc[0][name_column]
    if type(name) is not str:
        raise TypeError("AKShare ETF name metadata is invalid")
    return name.strip()


def _stock_name_from_spot(
    client: Any,
    instrument: InstrumentId,
    request_timeout_seconds: int,
) -> str | None:
    operation = getattr(client, "stock_zh_a_spot_em", None)
    if not callable(operation):
        return None
    source = diagnostic_request(
        provider="东方财富",
        transport="HTTP",
        operation="stock_zh_a_spot_em",
        endpoint=_EASTMONEY_STOCK_SPOT_ENDPOINT,
        details=f"instrument={instrument} timeout={request_timeout_seconds}s",
        call=lambda: requests_call_with_timeout(request_timeout_seconds, operation),
    )
    if not isinstance(source, pd.DataFrame) or source.columns.has_duplicates:
        return None
    if not {"代码", "名称"}.issubset(source.columns):
        return None
    matches = source.loc[source["代码"].map(str) == instrument.code]
    if len(matches) != 1:
        return None
    name = matches.iloc[0]["名称"]
    if type(name) is not str or not name.strip():
        return None
    return name.strip()


def _shanghai_now() -> datetime:
    return datetime.now(tz=_SHANGHAI)


def _validated_snapshot_time(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("AKShare snapshot clock must return a datetime")
    if value.tzinfo != _SHANGHAI or value.utcoffset() is None:
        raise ValueError("AKShare snapshot clock must use Asia/Shanghai")
    if value > datetime.now(tz=_SHANGHAI):
        raise ValueError("AKShare snapshot clock must not be in the future")
    return value


def _snapshot_number(value: object, *, field: str) -> Real:
    if isinstance(value, bool):
        raise TypeError(f"AKShare snapshot {field} must be numeric")
    parsed = pd.to_numeric(value, errors="raise")
    if not isinstance(parsed, Real) or not isfinite(parsed):
        raise ValueError(f"AKShare snapshot {field} must be finite")
    return parsed


def _snapshot_record(
    source: pd.DataFrame,
    mapping: dict[str, str],
    instrument: InstrumentId,
    fetched_at: datetime,
    source_columns: tuple[str, str] | None,
) -> dict[str, object]:
    if source.columns.has_duplicates:
        raise ValueError("AKShare snapshot columns must be unique")
    required = {"代码", *mapping}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"AKShare snapshot response is missing {sorted(missing)[0]}")
    matches = source.loc[source["代码"].map(str) == instrument.code]
    if len(matches.index) != 1:
        raise ValueError("AKShare snapshot must contain one requested instrument row")
    row = matches.iloc[0]
    values = {
        canonical: _snapshot_number(row[native], field=canonical)
        for native, canonical in mapping.items()
    }
    prices = tuple(values[column] for column in ("open", "high", "low", "close"))
    if any(value <= 0 for value in prices):
        raise ValueError("AKShare snapshot prices must be positive")
    if values["high"] < max(prices) or values["low"] > min(prices):
        raise ValueError("AKShare snapshot OHLC values are inconsistent")
    if values["volume"] < 0 or values["amount"] < 0:
        raise ValueError("AKShare snapshot volume and amount must be non-negative")
    source_at = _source_timestamp(row, source_columns, fetched_at)
    return {
        "instrument": str(instrument),
        "timestamp": source_at,
        "source_at": source_at,
        "fetched_at": fetched_at,
        **values,
    }


def _source_timestamp(
    row: pd.Series,
    columns: tuple[str, str] | None,
    fetched_at: datetime,
) -> object:
    if columns is None:
        return pd.NaT
    date_column, time_column = columns
    if date_column not in row.index or time_column not in row.index:
        return pd.NaT
    try:
        source_day = pd.Timestamp(row[date_column]).date()
        raw_time = row[time_column]
        if isinstance(raw_time, time):
            parsed = datetime.combine(source_day, raw_time, tzinfo=_SHANGHAI)
        elif isinstance(raw_time, (datetime, pd.Timestamp)):
            stamp = pd.Timestamp(raw_time)
            if stamp.tzinfo is None:
                parsed = datetime.combine(source_day, stamp.time(), tzinfo=_SHANGHAI)
            else:
                parsed = stamp.to_pydatetime().astimezone(_SHANGHAI)
        elif type(raw_time) is str:
            parsed_time = time.fromisoformat(raw_time)
            parsed = datetime.combine(source_day, parsed_time, tzinfo=_SHANGHAI)
        else:
            return pd.NaT
        if parsed.date() != source_day or parsed > fetched_at:
            return pd.NaT
        return parsed
    except (TypeError, ValueError, OverflowError):
        return pd.NaT


class AkshareProvider:
    name = "akshare"
    calendar_version = "akshare-tool-trade-date-hist-sina-v1"

    def __init__(
        self,
        *,
        client: Any | None = None,
        instrument_type_resolver: InstrumentTypeResolver = default_instrument_type,
        clock: Callable[[], datetime] = _shanghai_now,
        request_timeout_seconds: int = DEFAULT_MARKET_TIMEOUT_SECONDS,
    ) -> None:
        if client is None:
            import akshare as akshare_client  # type: ignore[import-untyped]

            resolved_client: Any = akshare_client
        else:
            resolved_client = client
        self._client = resolved_client
        self._instrument_type_resolver = instrument_type_resolver
        self._clock = clock
        self._request_timeout_seconds = validate_market_timeout_seconds(
            request_timeout_seconds
        )

    def set_request_timeout(self, seconds: int) -> None:
        self._request_timeout_seconds = validate_market_timeout_seconds(seconds)

    def fetch_daily(self, request: DailyBarRequest) -> pd.DataFrame:
        try:
            asset_type = resolve_instrument_type(
                self.name, self._instrument_type_resolver, request.instrument
            )
            if asset_type is AssetType.INDEX:
                symbol = _akshare_index_symbol(request.instrument)
                source = diagnostic_request(
                    provider="东方财富",
                    transport="HTTP",
                    operation="stock_zh_index_daily_em",
                    endpoint=_EASTMONEY_HISTORY_ENDPOINT,
                    details=(
                        f"instrument={request.instrument} symbol={symbol} "
                        f"start={request.start} end={request.end} "
                        f"timeout={self._request_timeout_seconds}s"
                    ),
                    call=lambda: requests_call_with_timeout(
                        self._request_timeout_seconds,
                        lambda: self._client.stock_zh_index_daily_em(
                            symbol=symbol,
                            start_date=request.start.strftime("%Y%m%d"),
                            end_date=request.end.strftime("%Y%m%d"),
                        ),
                    ),
                )
                mapping = _AKSHARE_INDEX_DAILY_MAPPING
            else:
                parameters = {
                    "symbol": _akshare_symbol(request.instrument),
                    "period": "daily",
                    "start_date": request.start.strftime("%Y%m%d"),
                    "end_date": request.end.strftime("%Y%m%d"),
                    "adjust": "",
                }
                operation_name = (
                    "stock_zh_a_hist"
                    if asset_type is AssetType.STOCK
                    else "fund_etf_hist_em"
                )

                def fetch_history() -> object:
                    if asset_type is AssetType.STOCK:
                        return self._client.stock_zh_a_hist(**parameters)
                    return self._client.fund_etf_hist_em(**parameters)

                source = diagnostic_request(
                    provider="东方财富",
                    transport="HTTP",
                    operation=operation_name,
                    endpoint=_EASTMONEY_HISTORY_ENDPOINT,
                    details=(
                        f"instrument={request.instrument} symbol={parameters['symbol']} "
                        f"start={request.start} end={request.end} period=daily adjust=none "
                        f"timeout={self._request_timeout_seconds}s"
                    ),
                    call=lambda: requests_call_with_timeout(
                        self._request_timeout_seconds,
                        fetch_history,
                    ),
                )
                mapping = _AKSHARE_DAILY_MAPPING
            if not isinstance(source, pd.DataFrame):
                raise TypeError("AKShare daily response is not a DataFrame")
            source = source.copy()
            volume_column = "volume" if asset_type is AssetType.INDEX else "成交量"
            source[volume_column] = pd.to_numeric(source[volume_column], errors="raise") * 100.0
            result = normalize_daily(source, mapping).loc[
                pd.Timestamp(request.start) : pd.Timestamp(request.end)
            ]
            if asset_type is AssetType.ETF:
                instrument_name = _etf_name_from_spot(
                    self._client,
                    request.instrument,
                    self._request_timeout_seconds,
                )
                result.attrs["instrument_name"] = instrument_name
                attestation = attest_etf_name(instrument_name)
                attach_attestation_columns(
                    result,
                    (attestation,) * len(result.index),
                )
            else:
                try:
                    stock_name = _stock_name_from_spot(
                        self._client,
                        request.instrument,
                        self._request_timeout_seconds,
                    )
                except Exception:
                    stock_name = None
                if stock_name is not None:
                    result.attrs["instrument_name"] = stock_name
            return result
        except Exception as error:
            raise_translated_provider_error(self.name, error)

    def fetch_snapshot(self, instruments: Sequence[InstrumentId]) -> pd.DataFrame:
        try:
            ordered = tuple(sorted(tuple(instruments), key=str))
            if not ordered or len(set(ordered)) != len(ordered):
                raise ProviderCapabilityError(self.name, "snapshot_request")
            by_type: dict[AssetType, list[InstrumentId]] = {
                AssetType.STOCK: [],
                AssetType.ETF: [],
            }
            asset_type_by_instrument: dict[InstrumentId, AssetType] = {}
            for instrument in ordered:
                if type(instrument) is not InstrumentId:
                    raise TypeError("AKShare snapshot request must contain instruments")
                if InstrumentId.parse(str(instrument)) != instrument:
                    raise ValueError("AKShare snapshot request contains a noncanonical instrument")
                asset_type = resolve_instrument_type(
                    self.name,
                    self._instrument_type_resolver,
                    instrument,
                )
                if type(asset_type) is not AssetType:
                    raise ProviderCapabilityError(self.name, "instrument_type_resolution")
                if asset_type is AssetType.INDEX:
                    raise ProviderCapabilityError(self.name, "index_snapshot")
                by_type[asset_type].append(instrument)
                asset_type_by_instrument[instrument] = asset_type
            sources: dict[AssetType, pd.DataFrame] = {}
            if by_type[AssetType.STOCK]:
                sources[AssetType.STOCK] = diagnostic_request(
                    provider="东方财富",
                    transport="HTTP",
                    operation="stock_zh_a_spot_em",
                    endpoint=_EASTMONEY_STOCK_SPOT_ENDPOINT,
                    details=(
                        f"instruments={','.join(map(str, by_type[AssetType.STOCK]))} "
                        f"timeout={self._request_timeout_seconds}s"
                    ),
                    call=lambda: requests_call_with_timeout(
                        self._request_timeout_seconds,
                        self._client.stock_zh_a_spot_em,
                    ),
                )
            if by_type[AssetType.ETF]:
                sources[AssetType.ETF] = diagnostic_request(
                    provider="东方财富",
                    transport="HTTP",
                    operation="fund_etf_spot_em",
                    endpoint=_EASTMONEY_ETF_SPOT_ENDPOINT,
                    details=(
                        f"instruments={','.join(map(str, by_type[AssetType.ETF]))} "
                        f"timeout={self._request_timeout_seconds}s"
                    ),
                    call=lambda: requests_call_with_timeout(
                        self._request_timeout_seconds,
                        self._client.fund_etf_spot_em,
                    ),
                )
            fetched_at = _validated_snapshot_time(self._clock())
            records: list[dict[str, object]] = []
            mappings = {
                AssetType.STOCK: _STOCK_SNAPSHOT_MAPPING,
                AssetType.ETF: _ETF_SNAPSHOT_MAPPING,
            }
            source_columns = {
                AssetType.STOCK: None,
                AssetType.ETF: ("数据日期", "更新时间"),
            }
            for instrument in ordered:
                asset_type = asset_type_by_instrument[instrument]
                source = sources[asset_type]
                if not isinstance(source, pd.DataFrame):
                    raise TypeError("AKShare snapshot response is not a DataFrame")
                records.append(
                    _snapshot_record(
                        source,
                        mappings[asset_type],
                        instrument,
                        fetched_at,
                        source_columns[asset_type],
                    )
                )
            return pd.DataFrame.from_records(records, columns=_SNAPSHOT_COLUMNS)
        except Exception as error:
            raise_translated_provider_error(self.name, error)

    def fetch_exchange_sessions(self, start: date, end: date) -> tuple[date, ...]:
        try:
            if type(start) is not date or type(end) is not date or start > end:
                raise ValueError("invalid exchange calendar range")
            source = diagnostic_request(
                provider="新浪财经",
                transport="HTTP",
                operation="tool_trade_date_hist_sina",
                endpoint=_SINA_CALENDAR_ENDPOINT,
                details=(
                    f"start={start} end={end} timeout={self._request_timeout_seconds}s"
                ),
                call=lambda: requests_call_with_timeout(
                    self._request_timeout_seconds,
                    self._client.tool_trade_date_hist_sina,
                ),
            )
            if not isinstance(source, pd.DataFrame) or set(source.columns) != {"trade_date"}:
                raise TypeError("AKShare calendar response is malformed")
            parsed = pd.to_datetime(source["trade_date"], errors="raise")
            sessions = tuple(
                sorted(
                    {timestamp.date() for timestamp in parsed if start <= timestamp.date() <= end}
                )
            )
            if not sessions:
                raise ValueError("AKShare calendar response has no requested sessions")
            return sessions
        except Exception as error:
            raise_translated_provider_error(self.name, error)

    def fetch_corporate_actions(self, request: DailyBarRequest) -> Sequence[CorporateAction]:
        raise ProviderCapabilityError(self.name, "corporate_actions")
