from __future__ import annotations

from datetime import date
from decimal import Decimal
import json
from typing import Any

import pandas as pd
import pytest
import requests

from compass.data.base import DailyBarRequest, ProviderError, ProviderErrorKind
from compass.data.providers.tencent_provider import TencentProvider
from compass.domain.market import InstrumentId


def _request(instrument: str = "SSE.600000") -> DailyBarRequest:
    return DailyBarRequest(
        InstrumentId.parse(instrument),
        date(2026, 7, 20),
        date(2026, 7, 21),
    )


class Response:
    def __init__(self, symbol: str, name: str) -> None:
        payload = {
            "data": {
                symbol: {
                    "day": [
                        [
                            "2026-07-21",
                            "10.1",
                            "10.2",
                            "10.3",
                            "10.0",
                            "12",
                            {},
                            "0.3",
                            "1.224",
                        ],
                        [
                            "2026-07-20",
                            "10.0",
                            "10.1",
                            "10.2",
                            "9.9",
                            "10",
                            {},
                            "0.2",
                            "1.01",
                        ],
                    ],
                    "qt": {symbol: ["1", name, symbol[2:]]},
                }
            }
        }
        self.text = "kline_day2026=" + json.dumps(payload, ensure_ascii=False)

    def raise_for_status(self) -> None:
        return


class Session:
    def __init__(self, name: str = "浦发银行") -> None:
        self.name = name
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> Response:
        self.calls.append({"url": url, **kwargs})
        symbol = str(kwargs["params"]["param"]).split(",", 1)[0]
        return Response(symbol, self.name)


def test_tencent_maps_daily_prices_volume_and_turnover_units() -> None:
    session = Session()

    result = TencentProvider(session=session).fetch_daily(_request())

    assert result.columns[:6].tolist() == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    assert result.index.tolist() == [pd.Timestamp("2026-07-20"), pd.Timestamp("2026-07-21")]
    assert result["volume"].tolist() == [1000.0, 1200.0]
    assert result["amount"].tolist() == [10_100.0, 12_240.0]
    assert result.attrs["instrument_name"] == "浦发银行"
    assert session.calls[0]["timeout"] == 5
    assert session.calls[0]["params"]["param"].startswith("sh600000,day,")


def test_tencent_attests_etf_rules_from_returned_name() -> None:
    result = TencentProvider(session=Session("创业板ETF易方达")).fetch_daily(
        _request("SZSE.159915")
    )

    assert result["price_limit_rate"].tolist() == [Decimal("0.20")] * 2
    assert result["listing_regime_known"].tolist() == [True, True]
    assert all(
        item.startswith("cn-price-limit-v1:etf-name:")
        for item in result["price_limit_rule_id"]
    )


def test_tencent_uses_exchange_prefix_for_indices() -> None:
    session = Session("沪深300指数")

    result = TencentProvider(session=session).fetch_daily(_request("SSE.000300"))

    assert not result.empty
    assert session.calls[0]["params"]["param"].startswith("sh000300,day,")


def test_tencent_splits_long_ranges_into_two_year_requests() -> None:
    session = Session()
    request = DailyBarRequest(
        InstrumentId.parse("SSE.600000"),
        date(2020, 7, 1),
        date(2024, 7, 1),
    )

    TencentProvider(session=session).fetch_daily(request)

    assert [call["params"]["param"].split(",")[2:4] for call in session.calls] == [
        ["2020-07-01", "2021-12-31"],
        ["2022-01-01", "2023-12-31"],
        ["2024-01-01", "2024-07-01"],
    ]


def test_tencent_translates_network_failures() -> None:
    class FailedSession:
        def get(self, url: str, **kwargs: Any) -> object:
            del url, kwargs
            raise requests.ConnectionError("upstream unavailable")

    with pytest.raises(ProviderError) as error:
        TencentProvider(session=FailedSession()).fetch_daily(_request())

    assert error.value.kind is ProviderErrorKind.NETWORK
    assert error.value.provider == "tencent"
