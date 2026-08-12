from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
import re
from types import MappingProxyType

from compass.domain.market import AssetType, InstrumentId
from compass.domain.weights import (
    WEIGHT_SCALE,
    largest_remainder_units,
    round_ratio_half_even,
    weight_to_units,
)


ContributionKey = tuple[str, str]


class AllocationStage(StrEnum):
    NORMALIZATION = "NORMALIZATION"
    STRATEGY_BUDGET = "STRATEGY_BUDGET"
    ASSET_BUDGET = "ASSET_BUDGET"
    CASH_RESERVE = "CASH_RESERVE"
    SLEEVE_ATTRIBUTION = "SLEEVE_ATTRIBUTION"


@dataclass(frozen=True, slots=True)
class AllocationAdjustment:
    """Exact, replayable record of one canonical fixed-point transformation."""

    stage: AllocationStage
    group: str
    input_keys: tuple[str, ...]
    before_units: int
    limit_units: int
    after_units: int
    residue_recipients: tuple[str, ...]
    reason_code: str

    def __post_init__(self) -> None:
        if type(self.stage) is not AllocationStage:
            raise TypeError("adjustment stage must be an exact AllocationStage")
        if type(self.group) is not str or not self.group or self.group != self.group.strip():
            raise ValueError("adjustment group must be a non-empty stable string")
        if type(self.input_keys) is not tuple:
            raise TypeError("adjustment input_keys must be an exact tuple")
        if any(type(value) is not str or not value for value in self.input_keys):
            raise TypeError("adjustment input keys must be non-empty strings")
        if (
            len(set(self.input_keys)) != len(self.input_keys)
            or tuple(sorted(self.input_keys)) != self.input_keys
        ):
            raise ValueError("adjustment input keys must be unique and canonically sorted")
        for name, value in (
            ("before_units", self.before_units),
            ("limit_units", self.limit_units),
            ("after_units", self.after_units),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"adjustment {name} must be a non-negative integer")
        if self.after_units > self.before_units or self.after_units > self.limit_units:
            raise ValueError("adjustment after_units must not exceed before or limit units")
        if type(self.residue_recipients) is not tuple:
            raise TypeError("adjustment residue_recipients must be an exact tuple")
        if any(type(value) is not str or not value for value in self.residue_recipients):
            raise TypeError("adjustment residue recipients must be non-empty strings")
        if len(set(self.residue_recipients)) != len(self.residue_recipients):
            raise ValueError("adjustment residue recipients must be unique")
        if any(value not in self.input_keys for value in self.residue_recipients):
            raise ValueError("adjustment residue recipients must identify input keys")
        if type(self.reason_code) is not str or not re.fullmatch(
            r"[A-Z][A-Z0-9_]*", self.reason_code
        ):
            raise ValueError("adjustment reason_code must be a stable upper-snake identifier")


def _stable_strategy_id(value: object) -> str:
    if type(value) is not str:
        raise TypeError("strategy id must be an exact string")
    if not value or value != value.strip():
        raise ValueError("strategy id must be a non-empty stable string")
    return value


def _stable_symbol(value: object) -> str:
    if type(value) is not str:
        raise TypeError("weight keys must be canonical instrument strings")
    try:
        canonical = str(InstrumentId.parse(value))
    except (TypeError, ValueError):
        raise ValueError("weight keys must be canonical instrument strings") from None
    if value != canonical:
        raise ValueError("weight keys must be canonical instrument strings")
    return value


@dataclass(frozen=True, slots=True)
class AllocationPolicy:
    strategy_budgets: Mapping[str, Decimal]
    asset_class_budgets: Mapping[AssetType, Decimal]
    asset_types: Mapping[InstrumentId, AssetType]
    cash_reserve: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.strategy_budgets, Mapping):
            raise TypeError("strategy budgets must be a mapping")
        strategy_units: dict[str, int] = {}
        for strategy_id, budget in self.strategy_budgets.items():
            stable_id = _stable_strategy_id(strategy_id)
            strategy_units[stable_id] = weight_to_units(budget, label="strategy budget")
        if sum(strategy_units.values()) > WEIGHT_SCALE:
            raise ValueError("strategy budgets must sum to at most one")

        if not isinstance(self.asset_class_budgets, Mapping):
            raise TypeError("asset-class budgets must be a mapping")
        asset_units: dict[AssetType, int] = {}
        for asset_type, budget in self.asset_class_budgets.items():
            if type(asset_type) is not AssetType:
                raise TypeError("asset-class budget keys must be exact AssetType values")
            asset_units[asset_type] = weight_to_units(budget, label="asset-class budget")
        if sum(asset_units.values()) > WEIGHT_SCALE:
            raise ValueError("asset-class budgets must sum to at most one")

        if not isinstance(self.asset_types, Mapping):
            raise TypeError("asset types must be a mapping")
        checked_asset_types: dict[InstrumentId, AssetType] = {}
        for instrument, asset_type in self.asset_types.items():
            if type(instrument) is not InstrumentId:
                raise TypeError("asset type keys must be exact InstrumentId values")
            if type(asset_type) is not AssetType:
                raise TypeError("asset type values must be exact AssetType values")
            if asset_type not in asset_units:
                raise ValueError(f"missing asset-class budget for {asset_type.value}")
            checked_asset_types[instrument] = asset_type

        weight_to_units(self.cash_reserve, label="cash reserve")
        object.__setattr__(
            self,
            "strategy_budgets",
            MappingProxyType(
                {
                    strategy_id: self.strategy_budgets[strategy_id]
                    for strategy_id in sorted(strategy_units)
                }
            ),
        )
        object.__setattr__(
            self,
            "asset_class_budgets",
            MappingProxyType(
                {
                    asset_type: self.asset_class_budgets[asset_type]
                    for asset_type in sorted(asset_units, key=lambda item: item.value)
                }
            ),
        )
        object.__setattr__(
            self,
            "asset_types",
            MappingProxyType(
                {
                    instrument: checked_asset_types[instrument]
                    for instrument in sorted(checked_asset_types, key=lambda item: str(item))
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class StrategyTrace:
    normalized: Mapping[ContributionKey, int]
    budgeted: Mapping[ContributionKey, int]
    adjustments: tuple[AllocationAdjustment, AllocationAdjustment]


@dataclass(frozen=True, slots=True)
class AllocationTrace:
    requested: Mapping[ContributionKey, int]
    normalized: Mapping[ContributionKey, int]
    budgeted: Mapping[ContributionKey, int]
    asset_limited: Mapping[ContributionKey, int]
    final_contributions: Mapping[ContributionKey, int]
    symbol_asset_types: Mapping[str, AssetType]
    final_symbols: Mapping[str, int]
    asset_units: Mapping[AssetType, int]
    cash_units: int
    adjustments: tuple[AllocationAdjustment, ...]


def _contribution_name(key: ContributionKey) -> str:
    return f"{key[0]}/{key[1]}"


def _canonical_input_keys(keys: Mapping[ContributionKey, int]) -> tuple[str, ...]:
    return tuple(sorted(_contribution_name(key) for key in keys))


def _adjustment(
    stage: AllocationStage,
    group: str,
    input_keys: tuple[str, ...],
    before_units: int,
    limit_units: int,
    after_units: int,
    recipients: tuple[str, ...],
    reason_code: str,
) -> AllocationAdjustment:
    return AllocationAdjustment(
        stage=stage,
        group=group,
        input_keys=input_keys,
        before_units=before_units,
        limit_units=limit_units,
        after_units=after_units,
        residue_recipients=recipients,
        reason_code=reason_code,
    )


def replay_strategy_stages(
    requested: Mapping[ContributionKey, int],
    strategy_id: str,
    budget_units: int,
) -> StrategyTrace:
    strategy_id = _stable_strategy_id(strategy_id)
    group = {key: requested[key] for key in sorted(requested) if key[0] == strategy_id}
    input_keys = _canonical_input_keys(group)
    before = sum(group.values())
    normalized_total = min(before, WEIGHT_SCALE)
    normalized, normalization_recipients = largest_remainder_units(
        group, normalized_total, canonical_key=_contribution_name
    )
    budgeted_total = round_ratio_half_even(normalized_total * budget_units, WEIGHT_SCALE)
    budgeted, budget_recipients = largest_remainder_units(
        normalized, budgeted_total, canonical_key=_contribution_name
    )
    return StrategyTrace(
        normalized=MappingProxyType(normalized),
        budgeted=MappingProxyType(budgeted),
        adjustments=(
            _adjustment(
                AllocationStage.NORMALIZATION,
                f"strategy:{strategy_id}",
                input_keys,
                before,
                WEIGHT_SCALE,
                normalized_total,
                normalization_recipients,
                "RAW_TARGET_CAP",
            ),
            _adjustment(
                AllocationStage.STRATEGY_BUDGET,
                f"strategy:{strategy_id}",
                input_keys,
                normalized_total,
                budget_units,
                budgeted_total,
                budget_recipients,
                "STRATEGY_BUDGET_APPLIED",
            ),
        ),
    )


def replay_allocation(
    policy: AllocationPolicy,
    requested: Mapping[ContributionKey, int],
) -> AllocationTrace:
    if not isinstance(policy, AllocationPolicy):
        raise TypeError("policy must be an AllocationPolicy")
    checked_requested: dict[ContributionKey, int] = {}
    symbol_asset_types: dict[str, AssetType] = {}
    for key, units in requested.items():
        if type(key) is not tuple or len(key) != 2:
            raise TypeError("contribution keys must be exact strategy/symbol tuples")
        strategy_id = _stable_strategy_id(key[0])
        symbol = _stable_symbol(key[1])
        if strategy_id not in policy.strategy_budgets:
            raise ValueError(f"unknown strategy: {strategy_id}")
        instrument = InstrumentId.parse(symbol)
        if instrument not in policy.asset_types:
            raise ValueError(f"unknown instrument asset type: {symbol}")
        if isinstance(units, bool) or not isinstance(units, int) or not 0 <= units <= WEIGHT_SCALE:
            raise ValueError("requested contribution units must be fixed-point unit integers")
        stable_key = (strategy_id, symbol)
        checked_requested[stable_key] = units
        symbol_asset_types[symbol] = policy.asset_types[instrument]
    requested_units = {key: checked_requested[key] for key in sorted(checked_requested)}
    strategy_ids = sorted({key[0] for key in requested_units})
    normalized: dict[ContributionKey, int] = {}
    budgeted: dict[ContributionKey, int] = {}
    adjustments: list[AllocationAdjustment] = []
    for strategy_id in strategy_ids:
        strategy_trace = replay_strategy_stages(
            requested_units,
            strategy_id,
            weight_to_units(policy.strategy_budgets[strategy_id], label="strategy budget"),
        )
        normalized.update(strategy_trace.normalized)
        budgeted.update(strategy_trace.budgeted)
        adjustments.extend(strategy_trace.adjustments)

    asset_limited = dict(budgeted)
    used_asset_types = sorted(
        {symbol_asset_types[key[1]] for key in budgeted}, key=lambda item: item.value
    )
    for asset_type in used_asset_types:
        group = {
            key: asset_limited[key]
            for key in sorted(asset_limited)
            if symbol_asset_types[key[1]] is asset_type
        }
        input_keys = _canonical_input_keys(group)
        before = sum(group.values())
        limit = weight_to_units(policy.asset_class_budgets[asset_type], label="asset-class budget")
        after = min(before, limit)
        scaled, recipients = largest_remainder_units(group, after, canonical_key=_contribution_name)
        asset_limited.update(scaled)
        adjustments.append(
            _adjustment(
                AllocationStage.ASSET_BUDGET,
                f"asset:{asset_type.value}",
                input_keys,
                before,
                limit,
                after,
                recipients,
                "ASSET_CLASS_CAP",
            )
        )

    symbols = sorted({key[1] for key in asset_limited})
    pre_cash_symbols = {
        symbol: sum(value for key, value in asset_limited.items() if key[1] == symbol)
        for symbol in symbols
    }
    before_cash = sum(pre_cash_symbols.values())
    invested_limit = WEIGHT_SCALE - weight_to_units(policy.cash_reserve, label="cash reserve")
    after_cash = min(before_cash, invested_limit)
    final_symbols, cash_recipients = largest_remainder_units(
        pre_cash_symbols, after_cash, canonical_key=lambda symbol: symbol
    )
    adjustments.append(
        _adjustment(
            AllocationStage.CASH_RESERVE,
            "portfolio",
            tuple(symbols),
            before_cash,
            invested_limit,
            after_cash,
            cash_recipients,
            "CASH_RESERVE_FLOOR",
        )
    )

    final_contributions: dict[ContributionKey, int] = {}
    for symbol in symbols:
        group = {key: asset_limited[key] for key in sorted(asset_limited) if key[1] == symbol}
        input_keys = _canonical_input_keys(group)
        before = sum(group.values())
        limit = final_symbols[symbol]
        attributed, recipients = largest_remainder_units(
            group, limit, canonical_key=_contribution_name
        )
        final_contributions.update(attributed)
        adjustments.append(
            _adjustment(
                AllocationStage.SLEEVE_ATTRIBUTION,
                f"symbol:{symbol}",
                input_keys,
                before,
                limit,
                limit,
                recipients,
                "CASH_LIMIT_ATTRIBUTION",
            )
        )

    asset_units = {asset_type: 0 for asset_type in used_asset_types}
    for symbol, units in final_symbols.items():
        asset_units[symbol_asset_types[symbol]] += units
    return AllocationTrace(
        requested=MappingProxyType(requested_units),
        normalized=MappingProxyType(dict(sorted(normalized.items()))),
        budgeted=MappingProxyType(dict(sorted(budgeted.items()))),
        asset_limited=MappingProxyType(dict(sorted(asset_limited.items()))),
        final_contributions=MappingProxyType(dict(sorted(final_contributions.items()))),
        symbol_asset_types=MappingProxyType(
            {symbol: symbol_asset_types[symbol] for symbol in symbols}
        ),
        final_symbols=MappingProxyType(dict(sorted(final_symbols.items()))),
        asset_units=MappingProxyType(
            {
                asset_type: asset_units[asset_type]
                for asset_type in sorted(asset_units, key=lambda item: item.value)
            }
        ),
        cash_units=WEIGHT_SCALE - sum(final_symbols.values()),
        adjustments=tuple(adjustments),
    )
