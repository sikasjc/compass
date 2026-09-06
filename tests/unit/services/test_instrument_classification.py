from __future__ import annotations

from compass.domain.market import InstrumentId
from compass.services.instrument_classification import (
    InstrumentCategory,
    classify_instrument,
)


def test_common_etfs_are_classified_by_asset_theme() -> None:
    cases = (
        ("SSE.510300", "沪深300ETF华泰柏瑞", InstrumentCategory.BROAD, "沪深300"),
        ("SSE.510880", "红利ETF", InstrumentCategory.DIVIDEND, "红利"),
        ("SZSE.159326", "电网设备ETF华夏", InstrumentCategory.INDUSTRY, "电网设备"),
        ("SZSE.159382", "创业板人工智能ETF南方", InstrumentCategory.INDUSTRY, "人工智能"),
        ("SZSE.159949", "创业板50ETF华安", InstrumentCategory.GROWTH, "创业板50"),
    )

    for code, name, category, theme in cases:
        result = classify_instrument(InstrumentId.parse(code), name)
        assert result.category is category
        assert result.theme == theme


def test_unknown_etf_uses_name_rules_and_safe_other_fallback() -> None:
    bond = classify_instrument(InstrumentId.parse("SSE.511999"), "短债ETF")
    unknown = classify_instrument(InstrumentId.parse("SZSE.159999"), "示例ETF")

    assert bond.category is InstrumentCategory.BOND
    assert unknown.category is InstrumentCategory.OTHER
