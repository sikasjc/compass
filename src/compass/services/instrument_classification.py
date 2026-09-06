from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from compass.domain.market import InstrumentId
from compass.services.instrument_names import common_instrument_name
from compass.services.safe_display import safe_display_text


class InstrumentCategory(StrEnum):
    BROAD = "宽基"
    INDUSTRY = "行业"
    DIVIDEND = "红利"
    GROWTH = "成长科技"
    BOND = "债券"
    COMMODITY = "商品"
    OVERSEAS = "海外"
    OTHER = "其他"


@dataclass(frozen=True, slots=True)
class InstrumentClassification:
    category: InstrumentCategory
    theme: str

    def __post_init__(self) -> None:
        if type(self.category) is not InstrumentCategory:
            raise TypeError("instrument category must be exact")
        safe_display_text(self.theme, label="instrument theme", maximum=64)


_EXPLICIT = MappingProxyType(
    {
        "SSE.000015": (InstrumentCategory.DIVIDEND, "红利"),
        "SSE.000016": (InstrumentCategory.BROAD, "上证50"),
        "SSE.000300": (InstrumentCategory.BROAD, "沪深300"),
        "SSE.000688": (InstrumentCategory.GROWTH, "科创50"),
        "SSE.000852": (InstrumentCategory.BROAD, "中证1000"),
        "SSE.000905": (InstrumentCategory.BROAD, "中证500"),
        "SSE.510050": (InstrumentCategory.BROAD, "上证50"),
        "SSE.510300": (InstrumentCategory.BROAD, "沪深300"),
        "SSE.510500": (InstrumentCategory.BROAD, "中证500"),
        "SSE.510880": (InstrumentCategory.DIVIDEND, "红利"),
        "SSE.512100": (InstrumentCategory.BROAD, "中证1000"),
        "SSE.517520": (InstrumentCategory.INDUSTRY, "黄金股"),
        "SSE.560860": (InstrumentCategory.INDUSTRY, "工业有色"),
        "SSE.588000": (InstrumentCategory.GROWTH, "科创50"),
        "SZSE.159326": (InstrumentCategory.INDUSTRY, "电网设备"),
        "SZSE.159382": (InstrumentCategory.INDUSTRY, "人工智能"),
        "SZSE.159915": (InstrumentCategory.GROWTH, "创业板"),
        "SZSE.159949": (InstrumentCategory.GROWTH, "创业板50"),
        "SZSE.159967": (InstrumentCategory.GROWTH, "创业板成长"),
        "SZSE.399006": (InstrumentCategory.GROWTH, "创业板"),
        "SZSE.399673": (InstrumentCategory.GROWTH, "创业板50"),
    }
)

_INDUSTRY_THEMES = (
    (("人工智能", "AI"), "人工智能"),
    (("半导体", "芯片"), "半导体"),
    (("电网", "电力设备"), "电网设备"),
    (("工业有色",), "工业有色"),
    (("黄金股",), "黄金股"),
    (("证券", "券商"), "证券"),
    (("银行",), "银行"),
    (("医药", "医疗"), "医药医疗"),
    (("消费",), "消费"),
    (("新能源", "光伏", "储能"), "新能源"),
    (("军工",), "军工"),
    (("房地产", "地产"), "房地产"),
    (("传媒",), "传媒"),
    (("通信",), "通信"),
    (("汽车",), "汽车"),
    (("煤炭",), "煤炭"),
    (("钢铁",), "钢铁"),
    (("农业", "养殖"), "农业"),
)


def classify_instrument(
    instrument: InstrumentId,
    instrument_name: str | None = None,
) -> InstrumentClassification:
    if type(instrument) is not InstrumentId:
        raise TypeError("instrument must be an exact InstrumentId")
    if instrument_name is not None:
        safe_display_text(instrument_name, label="instrument name", maximum=128)
    explicit = _EXPLICIT.get(str(instrument))
    if explicit is not None:
        return InstrumentClassification(*explicit)
    name = instrument_name or common_instrument_name(instrument) or instrument.code
    if "红利" in name or "股息" in name:
        return InstrumentClassification(InstrumentCategory.DIVIDEND, "红利")
    if any(
        keyword in name
        for keyword in ("国债", "债券", "短债", "长债", "政金债", "信用债", "可转债")
    ):
        return InstrumentClassification(InstrumentCategory.BOND, "债券")
    if any(
        keyword in name
        for keyword in ("纳指", "纳斯达克", "标普", "恒生", "港股", "日经", "德国", "法国")
    ):
        return InstrumentClassification(InstrumentCategory.OVERSEAS, "海外市场")
    if "黄金ETF" in name and "黄金股" not in name:
        return InstrumentClassification(InstrumentCategory.COMMODITY, "黄金")
    if any(keyword in name for keyword in ("原油", "豆粕", "有色期货", "商品")):
        return InstrumentClassification(InstrumentCategory.COMMODITY, "商品")
    for keywords, theme in _INDUSTRY_THEMES:
        if any(keyword in name for keyword in keywords):
            return InstrumentClassification(InstrumentCategory.INDUSTRY, theme)
    if any(keyword in name for keyword in ("创业板", "科创", "科技", "成长")):
        return InstrumentClassification(InstrumentCategory.GROWTH, "成长科技")
    if any(
        keyword in name
        for keyword in (
            "沪深300",
            "中证500",
            "中证1000",
            "上证50",
            "中证A",
            "A500",
            "全指",
            "宽基",
        )
    ):
        return InstrumentClassification(InstrumentCategory.BROAD, "宽基")
    return InstrumentClassification(InstrumentCategory.OTHER, "其他")
