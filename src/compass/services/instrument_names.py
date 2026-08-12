from __future__ import annotations

from types import MappingProxyType

from compass.domain.market import InstrumentId


_COMMON_INSTRUMENT_NAMES = MappingProxyType(
    {
        "SSE.000015": "上证红利指数",
        "SSE.000016": "上证50指数",
        "SSE.000300": "沪深300指数",
        "SSE.000688": "科创50指数",
        "SSE.000852": "中证1000指数",
        "SSE.000905": "中证500指数",
        "SSE.510050": "上证50ETF",
        "SSE.510300": "沪深300ETF",
        "SSE.510500": "中证500ETF",
        "SSE.510880": "红利ETF",
        "SSE.512100": "中证1000ETF",
        "SSE.588000": "科创50ETF",
        "SZSE.159915": "创业板ETF",
        "SZSE.159949": "华安创业板50ETF",
        "SZSE.399006": "创业板指数",
        "SZSE.399673": "创业板50指数",
    }
)

_PREFERRED_INDEX_ETFS = MappingProxyType(
    {
        "SSE.000015": "SSE.510880",
        "SSE.000016": "SSE.510050",
        "SSE.000300": "SSE.510300",
        "SSE.000688": "SSE.588000",
        "SSE.000852": "SSE.512100",
        "SSE.000905": "SSE.510500",
        "SZSE.399006": "SZSE.159915",
        "SZSE.399673": "SZSE.159949",
    }
)


def common_instrument_name(instrument: InstrumentId) -> str | None:
    if type(instrument) is not InstrumentId:
        raise TypeError("instrument must be an exact InstrumentId")
    return _COMMON_INSTRUMENT_NAMES.get(str(instrument))


def preferred_etf_for_index(instrument: InstrumentId) -> InstrumentId | None:
    if type(instrument) is not InstrumentId:
        raise TypeError("instrument must be an exact InstrumentId")
    mapped = _PREFERRED_INDEX_ETFS.get(str(instrument))
    return None if mapped is None else InstrumentId.parse(mapped)


def common_index_etf_pairs() -> tuple[tuple[InstrumentId, InstrumentId], ...]:
    return tuple(
        (InstrumentId.parse(index), InstrumentId.parse(etf))
        for index, etf in _PREFERRED_INDEX_ETFS.items()
    )
