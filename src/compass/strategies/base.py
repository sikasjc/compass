from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
import re
from types import MappingProxyType
from typing import Protocol, overload, runtime_checkable

import pandas as pd  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict

from compass.domain.market import AssetType, BarFrame, InstrumentId
from compass.domain.trading import TargetIntent


_SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
_FIELD_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_CANONICAL_BAR_FIELDS = frozenset((*BarFrame.REQUIRED_COLUMNS, *BarFrame.OPTIONAL_COLUMNS))


class StrategyFrequency(StrEnum):
    DAILY = "daily"
    FIVE_MINUTES = "5m"


class StrategyDecisionStatus(StrEnum):
    GENERATED = "GENERATED"
    SKIPPED = "SKIPPED"
    CASH = "CASH"


def _freeze_detail(value: object) -> object:
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not pd.notna(value) or value in (float("inf"), float("-inf")):
            raise ValueError("decision detail floats must be finite")
        return value
    if isinstance(value, tuple):
        return tuple(_freeze_detail(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_detail(item) for item in value)
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("decision detail mapping keys must be exact strings")
        return MappingProxyType(
            {key: _freeze_detail(value[key]) for key in sorted(value)}
        )
    raise TypeError("decision details must contain deterministic scalar, tuple, or mapping values")


@dataclass(frozen=True, slots=True)
class StrategyDecision(Sequence[TargetIntent]):
    """Immutable explained strategy result that remains Sequence-compatible."""

    intents: tuple[TargetIntent, ...]
    status: StrategyDecisionStatus
    reason_code: str
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        intents = tuple(self.intents)
        if any(not isinstance(intent, TargetIntent) for intent in intents):
            raise TypeError("decision intents must contain TargetIntent values")
        if not isinstance(self.status, StrategyDecisionStatus):
            raise TypeError("decision status must be a StrategyDecisionStatus")
        if type(self.reason_code) is not str or not re.fullmatch(
            r"[A-Z][A-Z0-9_]*", self.reason_code
        ):
            raise ValueError("decision reason_code must be a stable upper-snake identifier")
        if self.status is StrategyDecisionStatus.GENERATED and not intents:
            raise ValueError("generated decision must contain at least one intent")
        if self.status is not StrategyDecisionStatus.GENERATED and intents:
            raise ValueError("non-generated decision must not contain intents")
        if not isinstance(self.details, Mapping):
            raise TypeError("decision details must be a mapping")
        frozen_details = _freeze_detail(self.details)
        object.__setattr__(self, "intents", intents)
        object.__setattr__(self, "details", frozen_details)

    @classmethod
    def generated(
        cls,
        intents: Sequence[TargetIntent],
        *,
        details: Mapping[str, object] | None = None,
    ) -> StrategyDecision:
        return cls(tuple(intents), StrategyDecisionStatus.GENERATED, "GENERATED", details or {})

    @classmethod
    def empty(
        cls,
        status: StrategyDecisionStatus,
        reason_code: str,
        *,
        details: Mapping[str, object] | None = None,
    ) -> StrategyDecision:
        return cls((), status, reason_code, details or {})

    def __len__(self) -> int:
        return len(self.intents)

    def __iter__(self) -> Iterator[TargetIntent]:
        return iter(self.intents)

    @overload
    def __getitem__(self, index: int) -> TargetIntent: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[TargetIntent, ...]: ...

    def __getitem__(self, index: int | slice) -> TargetIntent | tuple[TargetIntent, ...]:
        return self.intents[index]


@dataclass(frozen=True, slots=True)
class StrategyMetadata:
    """Stable, serializable strategy definition available before instantiation."""

    strategy_type: str
    version: str
    display_name: str
    description: str
    supported_asset_types: frozenset[AssetType]
    supported_frequencies: frozenset[StrategyFrequency]
    required_fields: frozenset[str]
    minimum_history: int
    default_required_history: int
    parameters_type: type[StrategyParameters]

    def __post_init__(self) -> None:
        if not isinstance(self.strategy_type, str) or not re.fullmatch(
            r"[a-z][a-z0-9_]*", self.strategy_type
        ):
            raise ValueError("strategy_type must be a lower snake identifier")
        if not isinstance(self.version, str) or not _SEMVER.fullmatch(self.version):
            raise ValueError("version must be a Semantic Version")
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise ValueError("display_name must be non-empty")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be non-empty")

        asset_types = frozenset(self.supported_asset_types)
        if not asset_types or any(not isinstance(asset_type, AssetType) for asset_type in asset_types):
            raise ValueError("supported_asset_types must be non-empty AssetType values")
        frequencies = frozenset(self.supported_frequencies)
        if not frequencies or any(
            not isinstance(frequency, StrategyFrequency) for frequency in frequencies
        ):
            raise ValueError("supported_frequencies must be non-empty StrategyFrequency values")
        fields = frozenset(self.required_fields)
        if any(
            not isinstance(field, str)
            or not _FIELD_NAME.fullmatch(field)
            or field not in _CANONICAL_BAR_FIELDS
            for field in fields
        ):
            raise ValueError("required_fields must be canonical daily BarFrame fields")
        if (
            isinstance(self.minimum_history, bool)
            or not isinstance(self.minimum_history, int)
            or self.minimum_history <= 0
        ):
            raise ValueError("minimum_history must be a positive integer")
        if (
            isinstance(self.default_required_history, bool)
            or not isinstance(self.default_required_history, int)
            or self.default_required_history < self.minimum_history
        ):
            raise ValueError(
                "default_required_history must be an integer at least minimum_history"
            )
        if not isinstance(self.parameters_type, type) or not issubclass(
            self.parameters_type, StrategyParameters
        ):
            raise TypeError("parameters_type must subclass StrategyParameters")

        object.__setattr__(self, "supported_asset_types", asset_types)
        object.__setattr__(self, "supported_frequencies", frequencies)
        object.__setattr__(self, "required_fields", fields)


@dataclass(frozen=True, slots=True)
class HoldingSummary:
    """Immutable account holding data used by strategies that need holding age or stops."""

    instrument: InstrumentId
    quantity: int
    available_quantity: int
    average_cost: Decimal
    mark_price: Decimal
    holding_since: date | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.instrument, InstrumentId):
            raise ValueError("holding instrument must be an InstrumentId")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int) or self.quantity < 0:
            raise ValueError("holding quantity must be a non-negative integer")
        if (
            isinstance(self.available_quantity, bool)
            or not isinstance(self.available_quantity, int)
            or not 0 <= self.available_quantity <= self.quantity
        ):
            raise ValueError("available quantity must be between zero and quantity")
        for name, value in (("average cost", self.average_cost), ("mark price", self.mark_price)):
            if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
                raise ValueError(f"holding {name} must be a finite non-negative Decimal")
        if self.holding_since is not None and (
            isinstance(self.holding_since, datetime) or not isinstance(self.holding_since, date)
        ):
            raise ValueError("holding_since must be a date or None")


@dataclass(frozen=True, slots=True, init=False)
class StrategyContext:
    """A defensive, as-of bounded view of daily market data for one decision."""

    as_of: date
    instruments: tuple[InstrumentId, ...]
    account_equity: Decimal
    cash: Decimal
    _bars: Mapping[InstrumentId, pd.DataFrame]
    _holdings: Mapping[InstrumentId, HoldingSummary]
    _asset_types: Mapping[InstrumentId, AssetType]

    def __init__(
        self,
        as_of: date,
        bars: Mapping[InstrumentId, pd.DataFrame],
        instruments: Sequence[InstrumentId],
        account_equity: Decimal,
        cash: Decimal = Decimal("0"),
        holdings: Mapping[InstrumentId, HoldingSummary] | Sequence[HoldingSummary] = (),
        asset_types: Mapping[InstrumentId, AssetType] | None = None,
    ) -> None:
        if isinstance(as_of, datetime) or not isinstance(as_of, date):
            raise ValueError("as_of must be a date, not a datetime")
        if not isinstance(account_equity, Decimal) or not account_equity.is_finite():
            raise ValueError("account equity must be a finite Decimal")
        if account_equity < 0:
            raise ValueError("account equity must be non-negative")
        if not isinstance(cash, Decimal) or not cash.is_finite() or cash < 0:
            raise ValueError("cash must be a finite non-negative Decimal")

        ordered_instruments = tuple(instruments)
        if any(not isinstance(instrument, InstrumentId) for instrument in ordered_instruments):
            raise ValueError("instruments must contain InstrumentId values")
        if len(set(ordered_instruments)) != len(ordered_instruments):
            raise ValueError("instruments must be unique")
        if set(bars) != set(ordered_instruments):
            raise ValueError("bars must exist for exactly the configured instruments")

        as_of_timestamp = pd.Timestamp(as_of)
        bounded_bars: dict[InstrumentId, pd.DataFrame] = {}
        for instrument in ordered_instruments:
            frame = bars[instrument]
            if not isinstance(frame, pd.DataFrame):
                raise ValueError(f"bars for {instrument} must be a DataFrame")
            if not isinstance(frame.index, pd.DatetimeIndex):
                raise ValueError("bar index must be DatetimeIndex")
            if frame.index.tz is not None:
                raise ValueError("daily bar index must be timezone-naive trading dates")
            visible = frame.loc[frame.index.normalize() <= as_of_timestamp].copy(deep=True)
            bounded_bars[instrument] = BarFrame.validate(visible)

        bounded_holdings = self._bounded_holdings(holdings, set(ordered_instruments), as_of)
        bounded_asset_types = self._bounded_asset_types(asset_types, set(ordered_instruments))

        object.__setattr__(self, "as_of", as_of)
        object.__setattr__(self, "instruments", ordered_instruments)
        object.__setattr__(self, "account_equity", account_equity)
        object.__setattr__(self, "cash", cash)
        object.__setattr__(self, "_bars", MappingProxyType(bounded_bars))
        object.__setattr__(self, "_holdings", MappingProxyType(bounded_holdings))
        object.__setattr__(self, "_asset_types", MappingProxyType(bounded_asset_types))

    @staticmethod
    def _bounded_asset_types(
        asset_types: Mapping[InstrumentId, AssetType] | None,
        instruments: set[InstrumentId],
    ) -> dict[InstrumentId, AssetType]:
        if asset_types is None:
            return {}
        if not isinstance(asset_types, Mapping) or set(asset_types) != instruments:
            raise ValueError("asset_types must exist for exactly the configured instruments")
        result: dict[InstrumentId, AssetType] = {}
        for instrument, asset_type in asset_types.items():
            if not isinstance(instrument, InstrumentId) or not isinstance(asset_type, AssetType):
                raise ValueError("asset_types must map InstrumentId to AssetType")
            result[instrument] = asset_type
        return result

    @staticmethod
    def _bounded_holdings(
        holdings: Mapping[InstrumentId, HoldingSummary] | Sequence[HoldingSummary],
        instruments: set[InstrumentId],
        as_of: date,
    ) -> dict[InstrumentId, HoldingSummary]:
        pairs: Sequence[tuple[InstrumentId, HoldingSummary]]
        if isinstance(holdings, Mapping):
            pairs = tuple(holdings.items())
        else:
            pairs = tuple((holding.instrument, holding) for holding in holdings)

        result: dict[InstrumentId, HoldingSummary] = {}
        for instrument, holding in pairs:
            if not isinstance(instrument, InstrumentId) or not isinstance(holding, HoldingSummary):
                raise ValueError("holdings must contain InstrumentId and HoldingSummary values")
            if instrument != holding.instrument:
                raise ValueError("holding key must match HoldingSummary.instrument")
            if instrument not in instruments:
                raise ValueError("holdings must belong to configured instruments")
            if holding.holding_since is not None and holding.holding_since > as_of:
                raise ValueError("holding_since must not be after context as_of")
            if instrument in result:
                raise ValueError("holdings must be unique by instrument")
            result[instrument] = holding
        return result

    @property
    def bars(self) -> Mapping[InstrumentId, pd.DataFrame]:
        """Return a read-only mapping whose frames cannot mutate the context."""

        return MappingProxyType(
            {instrument: frame.copy(deep=True) for instrument, frame in self._bars.items()}
        )

    def history(self, instrument: InstrumentId) -> pd.DataFrame:
        """Return an independent, daily-bar slice visible at ``as_of``."""

        if instrument not in self._bars:
            raise KeyError(f"unknown instrument: {instrument}")
        return self._bars[instrument].copy(deep=True)

    @property
    def holdings(self) -> Mapping[InstrumentId, HoldingSummary]:
        """Return a fresh read-only holding map; summaries themselves are frozen."""

        return MappingProxyType(dict(self._holdings))

    def holding(self, instrument: InstrumentId) -> HoldingSummary | None:
        """Return the immutable summary for a configured instrument, if held."""

        if instrument not in self._bars:
            raise KeyError(f"unknown instrument: {instrument}")
        return self._holdings.get(instrument)

    @property
    def asset_types(self) -> Mapping[InstrumentId, AssetType]:
        return MappingProxyType(dict(self._asset_types))


class StrategyParameters(BaseModel):
    """Base model for immutable, explicitly declared strategy parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)


@runtime_checkable
class Strategy(Protocol):
    """A deterministic daily-decision strategy that emits target intents only."""

    strategy_type: str
    metadata: StrategyMetadata
    parameters_type: type[StrategyParameters]
    minimum_history: int
    required_history: int

    def generate_targets(self, context: StrategyContext) -> StrategyDecision:
        """Generate non-executable portfolio target intents for one decision date."""
