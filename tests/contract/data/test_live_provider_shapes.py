from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
import requests

from compass.data.base import (
    DailyBarRequest,
    ProviderCapabilityError,
    ProviderConfigurationError,
    ProviderError,
    ProviderErrorKind,
)
from compass.data.providers.akshare_provider import AkshareProvider
from compass.data.providers.baostock_provider import BaostockProvider
from compass.data.providers.http_provider import HttpProvider
from compass.domain.market import AssetType, InstrumentId
from compass.data.quality import QualityMode
from compass.services.data_service import DataService
from compass.storage.market_store import MarketStore


def _request(instrument: str = "SSE.600000") -> DailyBarRequest:
    return DailyBarRequest(InstrumentId.parse(instrument), date(2026, 7, 20), date(2026, 7, 21))


def _assert_canonical(result: pd.DataFrame) -> None:
    assert result.columns[:6].tolist() == ["open", "high", "low", "close", "volume", "amount"]
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.tz is None
    assert result.index.is_monotonic_increasing


class FakeAkshare:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def stock_zh_a_hist(self, **kwargs: str) -> pd.DataFrame:
        self.calls.append(("stock", kwargs["symbol"]))
        if self.error is not None:
            raise self.error
        return _akshare_frame()

    def fund_etf_hist_em(self, **kwargs: str) -> pd.DataFrame:
        self.calls.append(("etf", kwargs["symbol"]))
        if self.error is not None:
            raise self.error
        return _akshare_frame()

    def stock_zh_index_daily_em(self, **kwargs: str) -> pd.DataFrame:
        self.calls.append(("index", kwargs["symbol"]))
        if self.error is not None:
            raise self.error
        return _akshare_index_frame()

    def fund_etf_spot_em(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "代码": ["510300", "159915"],
                "名称": ["沪深300ETF", "创业板ETF"],
            }
        )

    def stock_zh_a_spot_em(self) -> pd.DataFrame:
        return pd.DataFrame({"代码": ["600000"], "名称": ["浦发银行"]})


def _akshare_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "日期": ["2026-07-21", "2026-07-20"],
            "开盘": [10.1, 10.0],
            "收盘": [10.2, 10.1],
            "最高": [10.3, 10.2],
            "最低": [10.0, 9.9],
            "成交量": [12.0, 10.0],
            "成交额": [12240.0, 10100.0],
            "换手率": [0.3, 0.2],
        }
    )


def _akshare_index_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-07-21", "2026-07-20"],
            "open": [4001.0, 4000.0],
            "close": [4002.0, 4001.0],
            "high": [4003.0, 4002.0],
            "low": [4000.0, 3999.0],
            "volume": [12.0, 10.0],
            "amount": [4802400.0, 4001000.0],
        }
    )


def test_akshare_maps_native_stock_shape() -> None:
    client = FakeAkshare()

    result = AkshareProvider(client=client).fetch_daily(_request())

    _assert_canonical(result)
    assert client.calls == [("stock", "600000")]
    assert result["volume"].tolist() == [1000.0, 1200.0]
    assert (result["amount"] / result["volume"]).tolist() == [10.1, 10.2]
    assert result.attrs["instrument_name"] == "浦发银行"


def test_akshare_default_instrument_resolver_routes_etfs() -> None:
    client = FakeAkshare()

    AkshareProvider(client=client).fetch_daily(_request("SSE.510300"))

    assert client.calls == [("etf", "510300")]


def test_akshare_default_instrument_resolver_routes_indices() -> None:
    client = FakeAkshare()

    result = AkshareProvider(client=client).fetch_daily(_request("SSE.000300"))

    _assert_canonical(result)
    assert client.calls == [("index", "sh000300")]
    assert result["volume"].tolist() == [1000.0, 1200.0]


def test_akshare_overrides_upstream_requests_timeout_per_call() -> None:
    observed: list[object] = []

    class Adapter(requests.adapters.BaseAdapter):
        def send(self, request: requests.PreparedRequest, **kwargs: Any) -> requests.Response:
            del request
            observed.append(kwargs.get("timeout"))
            response = requests.Response()
            response.status_code = 200
            response._content = b"{}"
            return response

        def close(self) -> None:
            return

    class Client(FakeAkshare):
        def stock_zh_index_daily_em(self, **kwargs: str) -> pd.DataFrame:
            del kwargs
            with requests.Session() as session:
                session.mount("https://", Adapter())
                session.get("https://push2his.eastmoney.com/test", timeout=15)
            return _akshare_index_frame()

    AkshareProvider(client=Client(), request_timeout_seconds=7).fetch_daily(
        _request("SSE.000300")
    )

    assert observed == [7]


def test_baostock_socket_factory_uses_configured_timeout() -> None:
    import baostock.util.socketutil as socket_util  # type: ignore[import-untyped]

    original = socket_util.socket
    provider = BaostockProvider(request_timeout_seconds=9)
    try:
        created = socket_util.socket.socket(socket_util.socket.AF_INET, socket_util.socket.SOCK_STREAM)
        try:
            assert created.gettimeout() == 9
        finally:
            created.close()
    finally:
        provider.close()
        socket_util.socket = original


def test_akshare_etf_volume_uses_shares_not_eastmoney_hands() -> None:
    result = AkshareProvider(client=FakeAkshare()).fetch_daily(_request("SSE.510300"))

    assert result["volume"].tolist() == [1000.0, 1200.0]
    assert (result["amount"] / result["volume"]).tolist() == [10.1, 10.2]


def test_akshare_etf_source_name_attests_effective_limit_regime() -> None:
    result = AkshareProvider(client=FakeAkshare()).fetch_daily(_request("SZSE.159915"))

    assert result["price_limit_rate"].tolist() == [Decimal("0.20")] * 2
    assert result["listing_regime_known"].tolist() == [True, True]
    assert result.attrs["instrument_name"] == "创业板ETF"
    assert all(
        value.startswith("cn-price-limit-v1:etf-name:") for value in result["price_limit_rule_id"]
    )


class FakeBaoResult:
    fields = [
        "date",
        "code",
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
        "adjustflag",
        "tradestatus",
        "isST",
    ]

    def __init__(self, error_code: str = "0", error_msg: str = "success") -> None:
        self.error_code = error_code
        self.error_msg = error_msg
        self._rows = iter(
            [
                [
                    "2026-07-21",
                    "sh.600000",
                    "10.1",
                    "10.3",
                    "10.0",
                    "10.2",
                    "10.1",
                    "1200",
                    "12240",
                    "3",
                    "1",
                    "0",
                ],
                [
                    "2026-07-20",
                    "sh.600000",
                    "10.0",
                    "10.2",
                    "9.9",
                    "10.1",
                    "10.0",
                    "1000",
                    "10100",
                    "3",
                    "0",
                    "1",
                ],
            ]
        )
        self._current: list[str] | None = None

    def next(self) -> bool:
        self._current = next(self._rows, None)
        return self._current is not None

    def get_row_data(self) -> list[str]:
        assert self._current is not None
        return self._current


class FakeBaoIndexResult(FakeBaoResult):
    fields = [
        "date",
        "code",
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
        "pctChg",
    ]

    def __init__(self) -> None:
        self.error_code = "0"
        self.error_msg = "success"
        self._rows = iter(
            [
                [
                    "2026-07-21",
                    "sh.000300",
                    "4001",
                    "4003",
                    "4000",
                    "4002",
                    "4001",
                    "1200",
                    "4802400",
                    "0.02",
                ],
                [
                    "2026-07-20",
                    "sh.000300",
                    "4000",
                    "4002",
                    "3999",
                    "4001",
                    "4000",
                    "1000",
                    "4001000",
                    "0.03",
                ],
            ]
        )
        self._current = None


class FakeBaoStock:
    def __init__(self, result: FakeBaoResult | Exception | None = None) -> None:
        self.result = result or FakeBaoResult()
        self.code: str | None = None
        self.fields: str | None = None

    def query_history_k_data_plus(self, code: str, fields: str, **kwargs: str) -> FakeBaoResult:
        self.code = code
        self.fields = fields
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def query_stock_basic(self, *, code: str) -> object:
        return _RowsResult(
            ("code", "code_name", "ipoDate"),
            ([code, "浦发银行", "1999-11-10"],),
        )

    def query_trade_dates(self, **kwargs: str) -> object:
        return _RowsResult(
            ("calendar_date", "is_trading_day"),
            tuple(
                [item.isoformat(), "1"]
                for item in pd.date_range("1999-11-10", periods=6, freq="D").date
            ),
        )


class _RowsResult:
    error_code = "0"
    error_msg = "success"

    def __init__(self, fields: tuple[str, ...], rows: tuple[list[str], ...]) -> None:
        self.fields = fields
        self._rows = iter(rows)
        self._current: list[str] | None = None

    def next(self) -> bool:
        self._current = next(self._rows, None)
        return self._current is not None

    def get_row_data(self) -> list[str]:
        assert self._current is not None
        return self._current


def test_baostock_maps_native_shape_and_trade_state() -> None:
    client = FakeBaoStock()

    result = BaostockProvider(client=client).fetch_daily(_request())

    _assert_canonical(result)
    assert client.code == "sh.600000"
    assert result["suspended"].tolist() == [True, False]
    assert result["adjust_flag"].tolist() == ["3", "3"]
    assert result["previous_close"].tolist() == [10.0, 10.1]
    assert result["exchange_reference_price"].tolist() == [10.0, 10.1]
    assert result["risk_warning"].tolist() == [True, False]
    assert result["listing_regime_known"].tolist() == [True, True]
    assert result["price_limit_rate"].tolist() == [Decimal("0.05"), Decimal("0.10")]


def test_baostock_default_instrument_resolver_routes_indices() -> None:
    client = FakeBaoStock(FakeBaoIndexResult())

    result = BaostockProvider(client=client).fetch_daily(_request("SSE.000300"))

    _assert_canonical(result)
    assert client.code == "sh.000300"
    assert client.fields == "date,code,open,high,low,close,preclose,volume,amount,pctChg"
    assert result["suspended"].tolist() == [False, False]
    assert result["risk_warning"].tolist() == [False, False]


def test_default_shaped_etf_and_stock_paths_persist_attested_rules(
    tmp_path: Path,
) -> None:
    def expected(request: DailyBarRequest) -> pd.DatetimeIndex:
        del request
        return pd.DatetimeIndex(["2026-07-20", "2026-07-21"])

    akshare = AkshareProvider(client=FakeAkshare())
    baostock = BaostockProvider(client=FakeBaoStock())

    etf = DataService(
        MarketStore(tmp_path / "etf"),
        expected_sessions=expected,
        require_rule_attestation=True,
        sleeper=lambda delay: None,
    ).sync_daily(
        _request("SZSE.159915"),
        (akshare, baostock),
        QualityMode.STRICT,
    )
    stock = DataService(
        MarketStore(tmp_path / "stock"),
        expected_sessions=expected,
        require_rule_attestation=True,
        sleeper=lambda delay: None,
    ).sync_daily(
        _request("SSE.600000"),
        (akshare, baostock),
        QualityMode.STRICT,
    )

    assert etf.provider == "akshare"
    assert stock.provider == "baostock"
    assert stock.manifest.provenance is not None
    assert tuple(
        (item.provider, item.failure_category) for item in stock.manifest.provenance.daily_attempts
    ) == (("akshare", "market_rules"), ("baostock", None))


def test_baostock_authenticates_before_using_the_default_style_client() -> None:
    events: list[str] = []

    class Client(FakeBaoStock):
        def login(self) -> object:
            events.append("login")
            return type("LoginResult", (), {"error_code": "0", "error_msg": "success"})()

        def query_history_k_data_plus(self, code: str, fields: str, **kwargs: str) -> FakeBaoResult:
            events.append("query")
            return super().query_history_k_data_plus(code, fields, **kwargs)

    BaostockProvider(client=Client()).fetch_daily(_request())

    assert events == ["login", "query"]


def test_baostock_context_manager_logs_out_and_reauthenticates_after_close() -> None:
    events: list[str] = []

    class Client(FakeBaoStock):
        def login(self) -> object:
            events.append("login")
            return type("Result", (), {"error_code": "0", "error_msg": "success"})()

        def logout(self) -> object:
            events.append("logout")
            return type("Result", (), {"error_code": "0", "error_msg": "success"})()

    client = Client()
    with BaostockProvider(client=client) as provider:
        provider.fetch_daily(_request())
    provider.fetch_daily(_request())
    provider.close()

    assert events == ["login", "logout", "login", "logout"]


def test_baostock_reauthenticates_after_an_invalid_session_result() -> None:
    login_count = 0
    query_count = 0

    class Client(FakeBaoStock):
        def login(self) -> object:
            nonlocal login_count
            login_count += 1
            return type("Result", (), {"error_code": "0", "error_msg": "success"})()

        def query_history_k_data_plus(self, code: str, fields: str, **kwargs: str) -> FakeBaoResult:
            nonlocal query_count
            query_count += 1
            if query_count == 1:
                return FakeBaoResult("10001001", "not logged in")
            return super().query_history_k_data_plus(code, fields, **kwargs)

    provider = BaostockProvider(client=Client())
    with pytest.raises(ProviderError) as first:
        provider.fetch_daily(_request())
    result = provider.fetch_daily(_request())

    assert first.value.kind is ProviderErrorKind.AUTHENTICATION
    assert login_count == 2
    _assert_canonical(result)


@pytest.mark.parametrize(
    ("code", "kind"),
    [
        ("10001001", ProviderErrorKind.AUTHENTICATION),
        ("10001002", ProviderErrorKind.AUTHENTICATION),
        ("10001005", ProviderErrorKind.RATE_LIMIT),
        ("10004016", ProviderErrorKind.RATE_LIMIT),
        ("10004011", ProviderErrorKind.MALFORMED_RESPONSE),
        ("10002001", ProviderErrorKind.NETWORK),
        ("10002002", ProviderErrorKind.NETWORK),
        ("10002003", ProviderErrorKind.NETWORK),
        ("10002008", ProviderErrorKind.NETWORK),
        ("10005001", ProviderErrorKind.MALFORMED_RESPONSE),
    ],
)
def test_baostock_response_codes_have_explicit_error_kinds(
    code: str, kind: ProviderErrorKind
) -> None:
    with pytest.raises(ProviderError) as error:
        BaostockProvider(client=FakeBaoStock(FakeBaoResult(code, "opaque"))).fetch_daily(_request())

    assert error.value.kind is kind


def test_baostock_failed_logout_keeps_session_retryable_and_secret_safe() -> None:
    logout_count = 0

    class Client(FakeBaoStock):
        def login(self) -> object:
            return type("Result", (), {"error_code": "0", "error_msg": "success"})()

        def logout(self) -> object:
            nonlocal logout_count
            logout_count += 1
            if logout_count == 1:
                return type(
                    "Result",
                    (),
                    {
                        "error_code": "10002007",
                        "error_msg": "token=sentinel-logout-secret",
                    },
                )()
            return type("Result", (), {"error_code": "0", "error_msg": "success"})()

    provider = BaostockProvider(client=Client())
    provider.fetch_daily(_request())

    with pytest.raises(ProviderError) as first:
        provider.close()
    provider.close()

    assert first.value.kind is ProviderErrorKind.NETWORK
    assert "sentinel-logout-secret" not in str(first.value)
    assert "sentinel-logout-secret" not in repr(first.value)
    assert logout_count == 2


def test_http_maps_explicit_enveloped_shape_without_network() -> None:
    captured_url: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_url.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "day": "2026-07-21",
                        "o": 10.1,
                        "h": 10.3,
                        "l": 10.0,
                        "c": 10.2,
                        "v": 1200.0,
                        "a": 12240.0,
                    },
                    {
                        "day": "2026-07-20",
                        "o": 10.0,
                        "h": 10.2,
                        "l": 9.9,
                        "c": 10.1,
                        "v": 1000.0,
                        "a": 10100.0,
                    },
                ],
                "status": "ok",
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = HttpProvider(
            url_template="https://market.example/bars/{exchange}/{symbol}?start={start}&end={end}&type={asset_type}",
            field_mapping={
                "day": "date",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "a": "amount",
            },
            client=client,
            instrument_type_resolver=lambda _: AssetType.STOCK,
        )
        result = provider.fetch_daily(_request())

    _assert_canonical(result)
    assert captured_url == [
        "https://market.example/bars/SSE/600000?start=2026-07-20&end=2026-07-21&type=stock"
    ]


@pytest.mark.parametrize(
    "url",
    [
        "https://user:password@market.example/bars/{symbol}",
        "https://market.example/bars/{symbol}?token=secret",
        "https://market.example/bars/{symbol}?api_key=secret",
    ],
)
def test_http_rejects_credentials_and_secrets_in_url_templates(url: str) -> None:
    with pytest.raises(ProviderConfigurationError) as error:
        HttpProvider(url_template=url, field_mapping={}, client=httpx.Client())

    assert "secret" not in str(error.value)
    assert "password" not in repr(error.value)


def test_http_rejects_a_malformed_response_envelope() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {"not": "records"}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = HttpProvider(
            url_template="https://market.example/bars/{symbol}",
            field_mapping={},
            client=client,
        )
        with pytest.raises(ProviderError) as error:
            provider.fetch_daily(_request())

    assert error.value.kind is ProviderErrorKind.MALFORMED_RESPONSE


def test_provider_error_redacts_bearer_credentials() -> None:
    error = ProviderError(
        ProviderErrorKind.AUTHENTICATION,
        "example",
        "Authorization: Bearer highly-secret",
    )

    assert "highly-secret" not in str(error)
    assert "highly-secret" not in repr(error)


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (PermissionError("token=highly-secret"), ProviderErrorKind.AUTHENTICATION),
        (RuntimeError("您没有访问该接口的权限"), ProviderErrorKind.AUTHENTICATION),
        (RuntimeError("rate limit exceeded: token=highly-secret"), ProviderErrorKind.RATE_LIMIT),
        (RuntimeError("每分钟最多访问该接口"), ProviderErrorKind.RATE_LIMIT),
        (ConnectionError("https://example.test/?token=highly-secret"), ProviderErrorKind.NETWORK),
    ],
)
@pytest.mark.parametrize(
    "provider_factory",
    [AkshareProvider, BaostockProvider],
)
def test_upstream_failures_have_stable_safe_error_kinds(
    provider_factory: Any, error: Exception, kind: ProviderErrorKind
) -> None:
    client: object
    if provider_factory is AkshareProvider:
        client = FakeAkshare(error=error)
    else:
        client = FakeBaoStock(result=error)

    with pytest.raises(ProviderError) as caught:
        provider_factory(client=client).fetch_daily(_request())

    assert caught.value.kind is kind
    assert "highly-secret" not in str(caught.value)
    assert "highly-secret" not in repr(caught.value)


@pytest.mark.parametrize(
    "provider", [AkshareProvider(client=FakeAkshare()), BaostockProvider(client=FakeBaoStock())]
)
def test_unsupported_company_actions_raise_typed_capability_error(provider: object) -> None:
    with pytest.raises(ProviderCapabilityError) as error:
        provider.fetch_corporate_actions(_request())

    assert error.value.kind is ProviderErrorKind.CAPABILITY
