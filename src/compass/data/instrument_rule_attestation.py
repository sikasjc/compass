from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from compass.domain.market import Exchange, InstrumentId


RULE_ATTESTATION_VERSION = "cn-price-limit-v1"


@dataclass(frozen=True, slots=True)
class InstrumentRuleAttestation:
    price_limit_rate: Decimal
    listing_regime_known: bool
    rule_id: str


def attest_etf_name(name: object) -> InstrumentRuleAttestation:
    if type(name) is not str or not name.strip():
        raise ValueError("ETF name metadata is unavailable")
    normalized = "".join(name.upper().split())
    twenty_percent_markers = ("创业板", "科创", "双创", "STAR", "CHINEXT")
    rate = (
        Decimal("0.20")
        if any(marker in normalized for marker in twenty_percent_markers)
        else Decimal("0.10")
    )
    return InstrumentRuleAttestation(
        rate,
        True,
        f"{RULE_ATTESTATION_VERSION}:etf-name:{rate}",
    )


def attest_stock_session(
    instrument: InstrumentId,
    day: date,
    *,
    standard_from: date,
    risk_warning: bool,
) -> InstrumentRuleAttestation | None:
    if day < standard_from:
        return None
    if risk_warning:
        rate = Decimal("0.05")
    elif (
        instrument.exchange is Exchange.SSE
        and instrument.code.startswith(("688", "689"))
        and day >= date(2019, 7, 22)
    ) or (
        instrument.exchange is Exchange.SZSE
        and instrument.code.startswith(("300", "301"))
        and day >= date(2020, 8, 24)
    ):
        rate = Decimal("0.20")
    else:
        rate = Decimal("0.10")
    return InstrumentRuleAttestation(
        rate,
        True,
        f"{RULE_ATTESTATION_VERSION}:stock-session:{rate}",
    )


def attach_attestation_columns(
    frame: object,
    attestations: tuple[InstrumentRuleAttestation | None, ...],
) -> None:
    # Kept here so providers share one field contract; the concrete frame type is
    # intentionally duck-typed to avoid making this small rules module own pandas.
    frame["price_limit_rate"] = [  # type: ignore[index]
        None if item is None else item.price_limit_rate for item in attestations
    ]
    frame["listing_regime_known"] = [  # type: ignore[index]
        False if item is None else item.listing_regime_known for item in attestations
    ]
    frame["price_limit_rule_id"] = [  # type: ignore[index]
        None if item is None else item.rule_id for item in attestations
    ]
