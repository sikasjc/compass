from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pandas as pd

from compass.data.providers.akshare_provider import AkshareProvider
from compass.domain.market import AssetType, InstrumentId
from compass.domain.trading import TargetIntent
from compass.services.intraday_service import IntradayService
from compass.strategies.base import StrategyContext


SHANGHAI = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 7, 21, 10, 5, tzinfo=SHANGHAI)
FETCHED_AT = NOW + timedelta(microseconds=1)
COMPLETED_AT = NOW + timedelta(microseconds=2)
ETF = InstrumentId.parse("SSE.510300")


class OfflineAkshare:
    def fund_etf_spot_em(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "代码": ["510300"],
                "名称": ["沪深300ETF"],
                "最新价": [4.12],
                "涨跌额": [0.01],
                "涨跌幅": [0.2],
                "成交量": [456789],
                "成交额": [1882050.68],
                "开盘价": [4.11],
                "最高价": [4.13],
                "最低价": [4.10],
                "昨收": [4.11],
                "换手率": [0.7],
                "流通市值": [200000000.0],
                "总市值": [200000000.0],
                "数据日期": [pd.Timestamp("2026-07-21")],
                "更新时间": [pd.Timestamp("2026-07-21 10:05:00")],
            }
        )


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [4.00, 4.05],
            "high": [4.08, 4.12],
            "low": [3.98, 4.03],
            "close": [4.05, 4.10],
            "volume": [100000.0, 120000.0],
            "amount": [405000.0, 492000.0],
        },
        index=pd.DatetimeIndex(["2026-07-17", "2026-07-20"], name="date"),
    )


def _temporary_intent(context: StrategyContext) -> tuple[TargetIntent, ...]:
    return (
        TargetIntent(
            strategy_id="snapshot-integration",
            instrument=ETF,
            target_weight=Decimal("0.5"),
            score=1.0,
            confidence=0.8,
            reason_code="PRICE_UPDATE",
            valid_until=context.as_of,
        ),
    )


def test_akshare_snapshot_drives_real_intraday_service_without_network() -> None:
    provider = AkshareProvider(client=OfflineAkshare(), clock=lambda: FETCHED_AT)
    service = IntradayService(
        instruments=(ETF,),
        provider=provider,
        daily_history={ETF: _history()},
        calculator=_temporary_intent,
        is_trading_day=lambda _: True,
        account_equity=Decimal("100000"),
        asset_types={ETF: AssetType.ETF},
        completion_clock=lambda: COMPLETED_AT,
    )

    state = service.refresh(NOW)

    assert state.failure_code is None
    assert state.active_session is True
    assert state.stale is False
    assert state.observed_at == COMPLETED_AT
    assert state.source_at == NOW
    assert state.quotes[0].volume == Decimal("456789")
    assert state.signals[0].status == "temporary"
    assert state.signals[0].confirmed is False
    assert state.persist_to_daily_results is False
