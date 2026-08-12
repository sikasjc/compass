from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from compass.config import Settings
from compass.data.base import DailyBarRequest
from compass.domain.market import InstrumentId
from compass.services.local_crud_gateways import LocalSettingsGateway
from compass.services.local_application import (
    _MarketProxyEnvironment,
    _test_market_connections,
)
from compass.storage.database import Database
from compass.ui.pages.settings import (
    ConnectionTestStatus,
    MarketProxyMode,
    MarketProxySetting,
)


class AvailableProvider:
    def __init__(self, name: str) -> None:
        self.name = name

    def fetch_daily(self, request: DailyBarRequest) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open": [4.0],
                "high": [4.2],
                "low": [3.9],
                "close": [4.1],
                "volume": [1000.0],
                "amount": [4100.0],
            },
            index=pd.DatetimeIndex([request.start.isoformat()], name="date"),
        )

    def fetch_snapshot(self, instruments: Sequence[InstrumentId]) -> pd.DataFrame:
        raise NotImplementedError

    def fetch_corporate_actions(self, request: DailyBarRequest) -> Sequence[Any]:
        return ()


def test_market_proxy_environment_can_switch_between_direct_custom_and_system(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("HTTP_PROXY", "http://system-proxy:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://system-proxy:8080")
    monkeypatch.delenv("NO_PROXY", raising=False)
    environment = _MarketProxyEnvironment()

    environment.apply(MarketProxySetting(MarketProxyMode.NONE))
    assert os.environ["NO_PROXY"] == "*"
    assert "HTTP_PROXY" not in os.environ
    assert "HTTPS_PROXY" not in os.environ

    environment.apply(MarketProxySetting(MarketProxyMode.CUSTOM, "127.0.0.1", 7897))
    assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:7897"
    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7897"
    assert "NO_PROXY" not in os.environ

    environment.apply(MarketProxySetting(MarketProxyMode.SYSTEM))
    assert os.environ["HTTP_PROXY"] == "http://system-proxy:8080"
    assert os.environ["HTTPS_PROXY"] == "http://system-proxy:8080"
    assert "NO_PROXY" not in os.environ


def test_connection_test_reports_proxy_and_each_market_source() -> None:
    results = _test_market_connections(
        (AvailableProvider("akshare"), AvailableProvider("baostock")),
        MarketProxySetting(MarketProxyMode.NONE),
        clock=lambda: datetime(2026, 8, 9, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    assert tuple(item.target for item in results) == ("proxy", "akshare", "baostock")
    assert results[0].status is ConnectionTestStatus.SKIPPED
    assert results[0].detail_code == "PROXY_DISABLED"
    assert all(item.status is ConnectionTestStatus.SUCCEEDED for item in results[1:])


def test_market_sources_use_fixed_configured_order(tmp_path) -> None:  # type: ignore[no-untyped-def]
    database = Database(Settings.from_env(tmp_path))
    database.create_schema()
    legacy = LocalSettingsGateway(
        database,
        clock=lambda: datetime(2026, 8, 9, tzinfo=ZoneInfo("Asia/Shanghai")),
        providers=(
            ("akshare", "东方财富", True, None),
            ("baostock", "BaoStock", True, None),
        ),
    )
    assert tuple(item.provider for item in legacy.state().providers) == (
        "akshare",
        "baostock",
    )

    upgraded = LocalSettingsGateway(
        database,
        clock=lambda: datetime(2026, 8, 9, tzinfo=ZoneInfo("Asia/Shanghai")),
        providers=(
            ("tencent", "腾讯证券", True, None),
            ("akshare", "东方财富", True, None),
            ("baostock", "BaoStock", True, None),
        ),
    )

    assert tuple(item.provider for item in upgraded.state().providers) == (
        "tencent",
        "akshare",
        "baostock",
    )
    database.engine.dispose()
