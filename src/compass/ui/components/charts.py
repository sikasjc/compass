from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from math import isfinite
from types import MappingProxyType

from compass.services.safe_display import safe_display_text, safe_identifier


ChartOptions = Mapping[str, object]


def _finite_float(value: object, *, label: str) -> float:
    if type(value) is not float or not isfinite(value):
        raise ValueError(f"{label} must be an exact finite float")
    return value


def _iso_day(value: object) -> str:
    if type(value) is not str:
        raise TypeError("chart day must be exact text")
    assert isinstance(value, str)
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise ValueError("chart day must be an ISO date") from None
    if parsed.isoformat() != value:
        raise ValueError("chart day must be a canonical ISO date")
    return value


@dataclass(frozen=True, slots=True)
class CurvePoint:
    day: str
    value: float

    def __post_init__(self) -> None:
        _iso_day(self.day)
        _finite_float(self.value, label="curve value")


@dataclass(frozen=True, slots=True)
class MarketBarPoint:
    day: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        _iso_day(self.day)
        for label, value in (
            ("open", self.open),
            ("high", self.high),
            ("low", self.low),
            ("close", self.close),
        ):
            if _finite_float(value, label=f"market bar {label}") <= 0:
                raise ValueError(f"market bar {label} must be positive")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("market bar OHLC values are inconsistent")
        if self.low > self.high:
            raise ValueError("market bar low must not exceed high")
        if _finite_float(self.volume, label="market bar volume") < 0:
            raise ValueError("market bar volume must be non-negative")


@dataclass(frozen=True, slots=True)
class AttributionPoint:
    day: str
    sleeve: str
    value: float

    def __post_init__(self) -> None:
        _iso_day(self.day)
        safe_identifier(self.sleeve, label="attribution sleeve")
        _finite_float(self.value, label="attribution value")


@dataclass(frozen=True, slots=True)
class MonthlyReturnPoint:
    month: str
    value: float | None

    def __post_init__(self) -> None:
        if type(self.month) is not str:
            raise ValueError("monthly return month must use YYYY-MM")
        try:
            parsed = date.fromisoformat(f"{self.month}-01")
        except ValueError:
            raise ValueError("monthly return month must use YYYY-MM") from None
        if parsed.strftime("%Y-%m") != self.month:
            raise ValueError("monthly return month must use YYYY-MM")
        if self.value is not None:
            _finite_float(self.value, label="monthly return value")


def _freeze_options(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_options(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_options(item) for item in value)
    return value


def _thaw_options(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_options(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_thaw_options(item) for item in value]
    return value


def thaw_chart_options(options: ChartOptions) -> dict[str, object]:
    if not isinstance(options, Mapping):
        raise TypeError("chart options must be a mapping")
    thawed = _thaw_options(options)
    assert isinstance(thawed, dict)
    return thawed


def _curve(values: Sequence[CurvePoint], *, label: str) -> tuple[CurvePoint, ...]:
    points = tuple(values)
    if any(type(point) is not CurvePoint for point in points):
        raise TypeError(f"{label} must contain exact CurvePoint values")
    days = tuple(point.day for point in points)
    if len(set(days)) != len(days) or days != tuple(sorted(days)):
        raise ValueError(f"{label} dates must be unique and increasing")
    return points


def _series(name: str, points: tuple[CurvePoint, ...], *, axis: int) -> dict[str, object]:
    safe_display_text(name, label="chart series name")
    return {
        "name": name,
        "type": "line",
        "showSymbol": False,
        "yAxisIndex": axis,
        "data": [[point.day, point.value] for point in points],
    }


def _trade_marker_series(
    name: str,
    marker: str,
    points: tuple[CurvePoint, ...],
    *,
    color: str,
    position: str,
) -> dict[str, object]:
    safe_display_text(name, label="trade marker series name")
    return {
        "name": name,
        "type": "scatter",
        "symbol": "circle",
        "symbolSize": 14,
        "yAxisIndex": 0,
        "data": [[point.day, point.value] for point in points],
        "itemStyle": {"color": color},
        "label": {
            "show": True,
            "formatter": marker,
            "position": position,
            "fontWeight": "bold",
            "color": color,
        },
    }


def _signal_marker_series(
    name: str,
    marker: str,
    points: tuple[CurvePoint, ...],
    *,
    color: str,
    symbol: str,
    position: str,
) -> dict[str, object]:
    safe_display_text(name, label="signal marker series name")
    return {
        "name": name,
        "type": "scatter",
        "symbol": symbol,
        "symbolSize": 11,
        "yAxisIndex": 0,
        "data": [[point.day, point.value] for point in points],
        "itemStyle": {"color": "#ffffff", "borderColor": color, "borderWidth": 2},
        "label": {
            "show": True,
            "formatter": marker,
            "position": position,
            "fontSize": 9,
            "color": color,
        },
    }


def equity_chart_options(
    equity: Sequence[CurvePoint],
    *,
    benchmark: Sequence[CurvePoint] = (),
    drawdown: Sequence[CurvePoint] = (),
    buy_markers: Sequence[CurvePoint] = (),
    sell_markers: Sequence[CurvePoint] = (),
    buy_signal_markers: Sequence[CurvePoint] = (),
    sell_signal_markers: Sequence[CurvePoint] = (),
    unfilled_markers: Sequence[CurvePoint] = (),
) -> ChartOptions:
    equity_points = _curve(equity, label="equity curve")
    benchmark_points = _curve(benchmark, label="benchmark curve")
    drawdown_points = _curve(drawdown, label="drawdown curve")
    buy_points = _curve(buy_markers, label="buy markers")
    sell_points = _curve(sell_markers, label="sell markers")
    buy_signal_points = _curve(buy_signal_markers, label="buy signal markers")
    sell_signal_points = _curve(sell_signal_markers, label="sell signal markers")
    unfilled_points = _curve(unfilled_markers, label="unfilled markers")
    equity_days = tuple(point.day for point in equity_points)
    for label, points in (
        ("benchmark curve", benchmark_points),
        ("drawdown curve", drawdown_points),
    ):
        if points and tuple(point.day for point in points) != equity_days:
            raise ValueError(f"{label} dates must exactly match the equity curve")
    equity_day_set = set(equity_days)
    for label, points in (
        ("buy markers", buy_points),
        ("sell markers", sell_points),
        ("buy signal markers", buy_signal_points),
        ("sell signal markers", sell_signal_points),
        ("unfilled markers", unfilled_points),
    ):
        if any(point.day not in equity_day_set for point in points):
            raise ValueError(f"{label} dates must belong to the equity curve")
    series = [_series("净值", equity_points, axis=0)]
    if benchmark_points:
        series.append(_series("基准", benchmark_points, axis=0))
    if drawdown_points:
        series.append(_series("回撤", drawdown_points, axis=1))
    if buy_signal_points:
        series.append(
            _signal_marker_series(
                "买入信号", "△", buy_signal_points,
                color="#ef4444", symbol="triangle", position="top",
            )
        )
    if sell_signal_points:
        series.append(
            _signal_marker_series(
                "卖出信号", "▽", sell_signal_points,
                color="#10b981", symbol="triangle", position="bottom",
            )
        )
    if unfilled_points:
        series.append(
            _signal_marker_series(
                "未成交", "×", unfilled_points,
                color="#d97706", symbol="rect", position="right",
            )
        )
    if buy_points:
        series.append(
            _trade_marker_series(
                "B 买入", "B", buy_points, color="#dc2626", position="top"
            )
        )
    if sell_points:
        series.append(
            _trade_marker_series(
                "S 卖出", "S", sell_points, color="#059669", position="bottom"
            )
        )
    options: dict[str, object] = {
        "animation": False,
        "tooltip": {"trigger": "axis"},
        "legend": {
            "data": [item["name"] for item in series],
            "type": "scroll",
            "top": 8,
            "left": "center",
        },
        "xAxis": {"type": "time", "name": "日期"},
        "grid": {"left": 64, "right": 32, "top": 88, "bottom": 72},
        "dataZoom": (
            {"type": "inside", "xAxisIndex": 0, "start": 0, "end": 100},
            {
                "type": "slider",
                "xAxisIndex": 0,
                "start": 0,
                "end": 100,
                "bottom": 16,
            },
        ),
        "yAxis": (
            {"type": "value", "name": "净值 / 基准"},
            {"type": "value", "name": "回撤", "max": 0},
        ),
        "series": series,
    }
    frozen = _freeze_options(options)
    assert isinstance(frozen, Mapping)
    return frozen


def market_data_chart_options(points: Sequence[MarketBarPoint]) -> ChartOptions:
    checked = tuple(points)
    if not checked or any(type(point) is not MarketBarPoint for point in checked):
        raise TypeError("market data must contain exact MarketBarPoint values")
    days = tuple(point.day for point in checked)
    if len(set(days)) != len(days) or days != tuple(sorted(days)):
        raise ValueError("market bar dates must be unique and increasing")
    frozen = _freeze_options(
        {
            "animation": False,
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
            "axisPointer": {"link": ({"xAxisIndex": "all"},)},
            "grid": (
                {"left": 64, "right": 24, "top": 28, "height": "58%"},
                {"left": 64, "right": 24, "top": "74%", "height": "16%"},
            ),
            "xAxis": (
                {
                    "type": "category",
                    "data": days,
                    "boundaryGap": True,
                    "axisLine": {"onZero": False},
                    "axisLabel": {"show": False},
                    "min": "dataMin",
                    "max": "dataMax",
                },
                {
                    "type": "category",
                    "gridIndex": 1,
                    "data": days,
                    "boundaryGap": True,
                    "min": "dataMin",
                    "max": "dataMax",
                },
            ),
            "yAxis": (
                {"scale": True, "name": "价格"},
                {"scale": True, "gridIndex": 1, "name": "成交量"},
            ),
            "dataZoom": (
                {"type": "inside", "xAxisIndex": (0, 1), "start": 60, "end": 100},
                {
                    "type": "slider",
                    "xAxisIndex": (0, 1),
                    "top": "93%",
                    "start": 60,
                    "end": 100,
                },
            ),
            "series": (
                {
                    "name": "日 K 线",
                    "type": "candlestick",
                    "data": tuple(
                        (point.open, point.close, point.low, point.high) for point in checked
                    ),
                    "itemStyle": {
                        "color": "#dc2626",
                        "color0": "#059669",
                        "borderColor": "#dc2626",
                        "borderColor0": "#059669",
                    },
                },
                {
                    "name": "成交量",
                    "type": "bar",
                    "xAxisIndex": 1,
                    "yAxisIndex": 1,
                    "data": tuple(point.volume for point in checked),
                    "itemStyle": {"color": "#64748b"},
                },
            ),
        }
    )
    assert isinstance(frozen, Mapping)
    return frozen


def attribution_chart_options(points: Sequence[AttributionPoint]) -> ChartOptions:
    checked = tuple(points)
    if any(type(point) is not AttributionPoint for point in checked):
        raise TypeError("attribution must contain exact AttributionPoint values")
    keys = tuple((point.day, point.sleeve) for point in checked)
    if len(set(keys)) != len(keys) or keys != tuple(sorted(keys)):
        raise ValueError("attribution points must be unique and deterministically sorted")
    sleeves = tuple(sorted({point.sleeve for point in checked}))
    series = [
        {
            "name": sleeve,
            "type": "bar",
            "stack": "归因",
            "data": [[point.day, point.value] for point in checked if point.sleeve == sleeve],
        }
        for sleeve in sleeves
    ]
    frozen = _freeze_options(
        {
            "animation": False,
            "tooltip": {"trigger": "axis"},
            "legend": {"data": list(sleeves)},
            "xAxis": {"type": "time", "name": "日期"},
            "yAxis": {"type": "value", "name": "策略贡献"},
            "series": series,
        }
    )
    assert isinstance(frozen, Mapping)
    return frozen


def monthly_return_chart_options(points: Sequence[MonthlyReturnPoint]) -> ChartOptions:
    checked = tuple(points)
    if any(type(point) is not MonthlyReturnPoint for point in checked):
        raise TypeError("monthly returns must contain exact MonthlyReturnPoint values")
    months = tuple(point.month for point in checked)
    if len(set(months)) != len(months) or months != tuple(sorted(months)):
        raise ValueError("monthly returns must be unique and increasing")
    frozen = _freeze_options(
        {
            "animation": False,
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "name": "月份", "data": list(months)},
            "yAxis": {"type": "value", "name": "月收益"},
            "series": [
                {
                    "name": "月收益",
                    "type": "bar",
                    "data": [point.value for point in checked],
                }
            ],
        }
    )
    assert isinstance(frozen, Mapping)
    return frozen
