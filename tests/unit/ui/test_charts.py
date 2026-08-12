from __future__ import annotations

from compass.ui.components.charts import (
    CurvePoint,
    equity_chart_options,
    thaw_chart_options,
)


def test_equity_chart_supports_trade_markers_and_zoom() -> None:
    equity = (
        CurvePoint("2026-08-06", 1.0),
        CurvePoint("2026-08-07", 1.1),
    )

    options = thaw_chart_options(
        equity_chart_options(
            equity,
            benchmark=equity,
            buy_markers=(equity[0],),
            sell_markers=(equity[1],),
            buy_signal_markers=(equity[0],),
            sell_signal_markers=(equity[1],),
            unfilled_markers=(equity[1],),
        )
    )

    assert [item["name"] for item in options["series"]] == [
        "净值",
        "基准",
        "买入信号",
        "卖出信号",
        "未成交",
        "B 买入",
        "S 卖出",
    ]
    assert options["series"][2]["label"]["formatter"] == "△"
    assert options["series"][4]["label"]["formatter"] == "×"
    assert options["series"][5]["label"]["formatter"] == "B"
    assert options["series"][6]["label"]["formatter"] == "S"
    assert [item["type"] for item in options["dataZoom"]] == ["inside", "slider"]
    assert options["legend"]["top"] == 8
    assert options["grid"]["top"] == 88
