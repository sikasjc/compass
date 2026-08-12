import pandas as pd
import pytest

from compass.domain.market import AssetType, BarFrame, Exchange, Instrument, InstrumentId


def valid_bar_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [1000.0],
            "amount": [10500.0],
        },
        index=pd.DatetimeIndex(["2026-07-20"], name="date"),
    )


def test_instrument_id_is_canonical() -> None:
    instrument = InstrumentId.parse("sse.510300")
    assert instrument.exchange is Exchange.SSE
    assert instrument.code == "510300"
    assert str(instrument) == "SSE.510300"


def test_instrument_metadata_carries_market_mechanics() -> None:
    instrument = Instrument(InstrumentId.parse("SSE.510300"), AssetType.ETF, 100, False)
    assert instrument.asset_type is AssetType.ETF
    assert instrument.lot_size == 100
    assert instrument.same_day_sell is False


def test_bar_frame_rejects_impossible_ohlc() -> None:
    frame = pd.DataFrame(
        {
            "open": [10.0],
            "high": [9.0],
            "low": [8.0],
            "close": [9.5],
            "volume": [1000.0],
            "amount": [9500.0],
        },
        index=pd.DatetimeIndex(["2026-07-20"], name="date"),
    )
    with pytest.raises(ValueError, match="high"):
        BarFrame.validate(frame)


def test_bar_frame_rejects_timezone_aware_daily_dates() -> None:
    frame = valid_bar_frame()
    frame.index = pd.DatetimeIndex(["2026-07-20"], tz="Asia/Shanghai", name="date")

    with pytest.raises(ValueError, match="timezone-naive"):
        BarFrame.validate(frame)


def test_bar_frame_rejects_intraday_timestamps_on_daily_axis() -> None:
    frame = valid_bar_frame()
    frame.index = pd.DatetimeIndex(["2026-07-20 09:30"], name="date")

    with pytest.raises(ValueError, match="midnight"):
        BarFrame.validate(frame)


def test_bar_frame_preserves_provider_adjustment_mode_without_treating_it_as_a_factor() -> None:
    frame = valid_bar_frame().assign(adjust_flag=["3"])

    result = BarFrame.validate(frame)

    assert result["adjust_flag"].tolist() == ["3"]


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("open", float("nan")),
        ("high", float("inf")),
        ("low", float("-inf")),
        ("volume", float("nan")),
        ("volume", float("inf")),
        ("volume", float("-inf")),
        ("volume", -1.0),
        ("amount", float("nan")),
        ("amount", float("inf")),
        ("amount", float("-inf")),
        ("amount", -1.0),
    ],
)
def test_bar_frame_rejects_non_finite_or_negative_market_data(column: str, value: float) -> None:
    frame = valid_bar_frame()
    frame.loc[:, column] = value

    with pytest.raises(ValueError, match="finite|non-negative"):
        BarFrame.validate(frame)
