from __future__ import annotations

from datetime import datetime
from math import inf, nan
import traceback
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from compass.data.base import (
    ProviderCapabilityError,
    ProviderError,
    ProviderErrorKind,
)
from compass.data.providers.akshare_provider import AkshareProvider
from compass.domain.market import AssetType, InstrumentId


SHANGHAI = ZoneInfo("Asia/Shanghai")
FETCHED_AT = datetime(2026, 7, 21, 10, 5, tzinfo=SHANGHAI)
SOURCE_AT = datetime(2026, 7, 21, 10, 4, tzinfo=SHANGHAI)
STOCK = InstrumentId.parse("SSE.600000")
ETF = InstrumentId.parse("SZSE.159915")
SSE_ETF = InstrumentId.parse("SSE.510300")
SNAPSHOT_COLUMNS = [
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
]


def _stock_spot() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "序号": [1, 2],
            "代码": ["600000", "000001"],
            "名称": ["浦发银行", "平安银行"],
            "最新价": [10.20, 11.30],
            "涨跌幅": [0.5, -0.2],
            "涨跌额": [0.05, -0.02],
            "成交量": [123456, 234567],
            "成交额": [1259251.2, 2650607.1],
            "振幅": [1.1, 1.2],
            "最高": [10.30, 11.40],
            "最低": [10.00, 11.20],
            "今开": [10.10, 11.35],
            "昨收": [10.15, 11.32],
            "量比": [1.0, 0.9],
            "换手率": [0.3, 0.4],
            "市盈率-动态": [5.0, 6.0],
            "市净率": [0.8, 0.9],
            "总市值": [1000000000.0, 2000000000.0],
            "流通市值": [900000000.0, 1800000000.0],
            "涨速": [0.1, -0.1],
            "5分钟涨跌": [0.2, -0.2],
            "60日涨跌幅": [1.5, 2.5],
            "年初至今涨跌幅": [3.5, 4.5],
        }
    )


def _etf_spot() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "代码": ["159915", "510300"],
            "名称": ["创业板ETF", "沪深300ETF"],
            "最新价": [2.50, 4.12],
            "IOPV实时估值": [2.50, 4.12],
            "基金折价率": [0.0, 0.0],
            "涨跌额": [0.02, 0.01],
            "涨跌幅": [0.8, 0.2],
            "成交量": [345678, 456789],
            "成交额": [864195.0, 1882050.68],
            "开盘价": [2.48, 4.11],
            "最高价": [2.52, 4.13],
            "最低价": [2.47, 4.10],
            "昨收": [2.48, 4.11],
            "振幅": [2.0, 0.7],
            "换手率": [1.2, 0.7],
            "量比": [1.1, 0.9],
            "委比": [3.0, 4.0],
            "外盘": [1000, 2000],
            "内盘": [900, 1800],
            "主力净流入-净额": [10000.0, 20000.0],
            "主力净流入-净占比": [1.0, 2.0],
            "超大单净流入-净额": [1000.0, 2000.0],
            "超大单净流入-净占比": [0.1, 0.2],
            "大单净流入-净额": [2000.0, 3000.0],
            "大单净流入-净占比": [0.2, 0.3],
            "中单净流入-净额": [3000.0, 4000.0],
            "中单净流入-净占比": [0.3, 0.4],
            "小单净流入-净额": [4000.0, 5000.0],
            "小单净流入-净占比": [0.4, 0.5],
            "现手": [100, 200],
            "买一": [2.49, 4.11],
            "卖一": [2.50, 4.12],
            "最新份额": [40000000.0, 50000000.0],
            "流通市值": [100000000.0, 200000000.0],
            "总市值": [100000000.0, 200000000.0],
            "数据日期": [pd.Timestamp("2026-07-21")] * 2,
            "更新时间": [pd.Timestamp(SOURCE_AT)] * 2,
        }
    )


class FakeAkshareSnapshotClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def stock_zh_a_spot_em(self) -> pd.DataFrame:
        self.calls.append("stock")
        return _stock_spot()

    def fund_etf_spot_em(self) -> pd.DataFrame:
        self.calls.append("etf")
        return _etf_spot()


def test_akshare_stock_snapshot_maps_native_shape_without_rescaling_spot_volume() -> None:
    client = FakeAkshareSnapshotClient()

    result = AkshareProvider(client=client, clock=lambda: FETCHED_AT).fetch_snapshot(
        (STOCK,)
    )

    assert client.calls == ["stock"]
    assert result.columns.tolist() == SNAPSHOT_COLUMNS
    assert result.to_dict(orient="records") == [
        {
            "instrument": "SSE.600000",
            "timestamp": pd.NaT,
            "source_at": pd.NaT,
            "fetched_at": FETCHED_AT,
            "open": 10.10,
            "high": 10.30,
            "low": 10.00,
            "close": 10.20,
            "volume": 123456,
            "amount": 1259251.2,
        }
    ]


def test_akshare_etf_snapshot_maps_native_shape_without_calling_stock_endpoint() -> None:
    client = FakeAkshareSnapshotClient()

    result = AkshareProvider(client=client, clock=lambda: FETCHED_AT).fetch_snapshot((ETF,))

    assert client.calls == ["etf"]
    assert result.columns.tolist() == SNAPSHOT_COLUMNS
    assert result.to_dict(orient="records") == [
        {
            "instrument": "SZSE.159915",
            "timestamp": SOURCE_AT,
            "source_at": SOURCE_AT,
            "fetched_at": FETCHED_AT,
            "open": 2.48,
            "high": 2.52,
            "low": 2.47,
            "close": 2.50,
            "volume": 345678,
            "amount": 864195.0,
        }
    ]


def test_akshare_mixed_snapshot_calls_each_required_endpoint_once_and_sorts_output() -> None:
    client = FakeAkshareSnapshotClient()

    result = AkshareProvider(client=client, clock=lambda: FETCHED_AT).fetch_snapshot(
        (ETF, STOCK)
    )

    assert client.calls == ["stock", "etf"]
    assert result["instrument"].tolist() == ["SSE.600000", "SZSE.159915"]
    assert result["volume"].tolist() == [123456, 345678]


def test_akshare_snapshot_never_represents_missing_upstream_time_as_fresh() -> None:
    result = AkshareProvider(
        client=FakeAkshareSnapshotClient(),
        clock=lambda: FETCHED_AT,
    ).fetch_snapshot((STOCK,))

    assert pd.isna(result.iloc[0]["timestamp"])
    assert pd.isna(result.iloc[0]["source_at"])
    assert result.iloc[0]["fetched_at"] == FETCHED_AT


def test_akshare_mixed_snapshot_uses_global_canonical_order_across_asset_types() -> None:
    result = AkshareProvider(
        client=FakeAkshareSnapshotClient(),
        clock=lambda: FETCHED_AT,
    ).fetch_snapshot((STOCK, SSE_ETF))

    assert result["instrument"].tolist() == ["SSE.510300", "SSE.600000"]


def test_akshare_snapshot_resolves_each_requested_instrument_once() -> None:
    calls: list[InstrumentId] = []
    asset_types = {STOCK: AssetType.STOCK, ETF: AssetType.ETF}

    def resolver(instrument: InstrumentId) -> AssetType:
        calls.append(instrument)
        return asset_types[instrument]

    AkshareProvider(
        client=FakeAkshareSnapshotClient(),
        clock=lambda: FETCHED_AT,
        instrument_type_resolver=resolver,
    ).fetch_snapshot((ETF, STOCK))

    assert calls == [STOCK, ETF]


@pytest.mark.parametrize("instruments", [(), (STOCK, STOCK)])
def test_akshare_snapshot_rejects_empty_or_duplicate_requests(
    instruments: tuple[InstrumentId, ...],
) -> None:
    client = FakeAkshareSnapshotClient()

    with pytest.raises(ProviderCapabilityError) as error:
        AkshareProvider(client=client, clock=lambda: FETCHED_AT).fetch_snapshot(
            instruments
        )

    assert error.value.kind is ProviderErrorKind.CAPABILITY
    assert client.calls == []


@pytest.mark.parametrize(
    "stock_frame",
    [
        _stock_spot().loc[lambda frame: frame["代码"] != "600000"],
        pd.concat((_stock_spot(), _stock_spot().iloc[[0]]), ignore_index=True),
    ],
    ids=("missing-requested-row", "duplicate-requested-row"),
)
def test_akshare_snapshot_rejects_missing_or_duplicate_requested_rows(
    stock_frame: pd.DataFrame,
) -> None:
    class Client(FakeAkshareSnapshotClient):
        def stock_zh_a_spot_em(self) -> pd.DataFrame:
            self.calls.append("stock")
            return stock_frame

    with pytest.raises(ProviderError) as error:
        AkshareProvider(client=Client(), clock=lambda: FETCHED_AT).fetch_snapshot(
            (STOCK,)
        )

    assert error.value.kind is ProviderErrorKind.MALFORMED_RESPONSE


@pytest.mark.parametrize(
    "response",
    [
        object(),
        pd.DataFrame({"代码": ["600000"], "最新价": [10.2]}),
        pd.concat(
            (
                _stock_spot(),
                _stock_spot()[["今开"]].rename(columns={"今开": "最新价"}),
            ),
            axis=1,
        ),
    ],
    ids=("non-dataframe", "missing-native-columns", "duplicate-native-column"),
)
def test_akshare_snapshot_rejects_non_dataframe_or_malformed_native_shapes(
    response: object,
) -> None:
    class Client:
        def stock_zh_a_spot_em(self) -> object:
            return response

    with pytest.raises(ProviderError) as error:
        AkshareProvider(client=Client(), clock=lambda: FETCHED_AT).fetch_snapshot(
            (STOCK,)
        )

    assert error.value.kind is ProviderErrorKind.MALFORMED_RESPONSE


def _stock_value(column: str, value: object) -> pd.DataFrame:
    frame = _stock_spot()
    if not isinstance(value, (int, float)):
        frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    return frame


@pytest.mark.parametrize(
    "stock_frame",
    [
        _stock_value("今开", nan),
        _stock_value("最高", inf),
        _stock_value("最低", 0),
        _stock_value("最新价", "not-numeric"),
        _stock_value("最高", 10.19),
        _stock_value("最低", 10.21),
        _stock_value("成交量", -1),
        _stock_value("成交额", -1),
    ],
    ids=(
        "non-finite-open",
        "non-finite-high",
        "zero-low",
        "non-numeric-close",
        "high-below-close",
        "low-above-close",
        "negative-volume",
        "negative-amount",
    ),
)
def test_akshare_snapshot_rejects_invalid_requested_market_values(
    stock_frame: pd.DataFrame,
) -> None:
    class Client(FakeAkshareSnapshotClient):
        def stock_zh_a_spot_em(self) -> pd.DataFrame:
            return stock_frame

    with pytest.raises(ProviderError) as error:
        AkshareProvider(client=Client(), clock=lambda: FETCHED_AT).fetch_snapshot(
            (STOCK,)
        )

    assert error.value.kind is ProviderErrorKind.MALFORMED_RESPONSE


@pytest.mark.parametrize(
    "timestamp",
    [
        datetime(2026, 7, 21, 10, 5),
        datetime(2026, 7, 21, 2, 5, tzinfo=ZoneInfo("UTC")),
        datetime(2099, 7, 21, 10, 5, tzinfo=SHANGHAI),
    ],
    ids=("naive", "wrong-timezone", "future"),
)
def test_akshare_snapshot_rejects_an_untrustworthy_clock(timestamp: datetime) -> None:
    with pytest.raises(ProviderError) as error:
        AkshareProvider(
            client=FakeAkshareSnapshotClient(),
            clock=lambda: timestamp,
        ).fetch_snapshot((STOCK,))

    assert error.value.kind is ProviderErrorKind.MALFORMED_RESPONSE


def test_akshare_default_snapshot_clock_is_shanghai_aware() -> None:
    result = AkshareProvider(client=FakeAkshareSnapshotClient()).fetch_snapshot((STOCK,))

    fetched_at = result.iloc[0]["fetched_at"]
    assert pd.isna(result.iloc[0]["timestamp"])
    assert isinstance(fetched_at, datetime)
    assert fetched_at.tzinfo == SHANGHAI
    assert fetched_at <= datetime.now(tz=SHANGHAI)


def test_akshare_snapshot_rejects_an_unsupported_resolver_result_without_network() -> None:
    client = FakeAkshareSnapshotClient()

    with pytest.raises(ProviderCapabilityError) as error:
        AkshareProvider(
            client=client,
            clock=lambda: FETCHED_AT,
            instrument_type_resolver=lambda _: "bond",  # type: ignore[return-value]
        ).fetch_snapshot((STOCK,))

    assert error.value.kind is ProviderErrorKind.CAPABILITY
    assert client.calls == []


def test_akshare_snapshot_translates_upstream_errors_without_secret_context() -> None:
    secret = "sentinel-snapshot-secret"

    class Client:
        def stock_zh_a_spot_em(self) -> pd.DataFrame:
            raise ConnectionError(
                f"https://upstream.invalid/spot?token={secret}"
            )

    with pytest.raises(ProviderError) as caught:
        AkshareProvider(client=Client(), clock=lambda: FETCHED_AT).fetch_snapshot(
            (STOCK,)
        )

    formatted = "".join(traceback.format_exception(caught.value))
    assert caught.value.kind is ProviderErrorKind.NETWORK
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    assert secret not in formatted
    assert secret not in str(caught.value)
    assert secret not in repr(caught.value)
