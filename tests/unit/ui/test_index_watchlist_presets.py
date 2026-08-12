from __future__ import annotations

from compass.data.base import default_instrument_type
from compass.domain.market import AssetType, InstrumentId
from compass.services.instrument_names import (
    common_index_etf_pairs,
    common_instrument_name,
    preferred_etf_for_index,
)
from compass.ui.pages.watchlists import (
    QUICK_SELECTIONS,
    WatchlistFormModel,
    _append_symbols,
    _market_symbols,
)


EXPECTED_PRESETS = (
    ("大盘核心", ("SSE.000016", "SSE.000300")),
    ("全市场宽基", ("SSE.000300", "SSE.000905", "SSE.000852")),
    ("成长科技", ("SZSE.399006", "SZSE.399673", "SSE.000688")),
    ("红利风格", ("SSE.000015",)),
)


def test_quick_selections_prefer_indices_with_names() -> None:
    assert QUICK_SELECTIONS == EXPECTED_PRESETS
    for _, codes in QUICK_SELECTIONS:
        for code in codes:
            instrument = InstrumentId.parse(code)
            assert default_instrument_type(instrument) is AssetType.INDEX
            assert common_instrument_name(instrument) is not None


def test_index_preset_appends_without_replacing_an_existing_etf() -> None:
    updated = _append_symbols("SSE.510300", QUICK_SELECTIONS[0][1])

    assert updated == "SSE.510300\nSSE.000016\nSSE.000300"
    validation = WatchlistFormModel("关注标的", updated).validate()
    assert validation.errors == {}
    assert validation.draft is not None
    assert validation.draft.instruments == (
        InstrumentId.parse("SSE.000016"),
        InstrumentId.parse("SSE.000300"),
        InstrumentId.parse("SSE.510300"),
    )


def test_market_first_input_builds_canonical_codes() -> None:
    assert _market_symbols("SSE", "000300, 510300") == (
        "SSE.000300",
        "SSE.510300",
    )
    assert _market_symbols("SZSE", "399006") == ("SZSE.399006",)


def test_every_preset_index_has_one_named_preferred_etf() -> None:
    pairs = dict(common_index_etf_pairs())
    preset_indices = {
        InstrumentId.parse(code) for _, codes in QUICK_SELECTIONS for code in codes
    }

    assert set(pairs) == preset_indices
    for index in preset_indices:
        etf = preferred_etf_for_index(index)
        assert etf == pairs[index]
        assert default_instrument_type(etf) is AssetType.ETF
        assert common_instrument_name(etf) is not None
