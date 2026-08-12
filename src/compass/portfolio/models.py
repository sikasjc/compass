from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType

from compass.domain.market import AssetType
from compass.domain.weights import weight_to_units
from compass.portfolio.trace import (
    AllocationAdjustment,
    AllocationPolicy,
    AllocationStage,
    ContributionKey,
    _stable_strategy_id,
    _stable_symbol,
    replay_allocation,
    replay_strategy_stages,
)


def _freeze_symbol_weights(
    values: Mapping[str, Decimal], *, label: str
) -> Mapping[str, Decimal]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} must be a mapping")
    for symbol in values:
        _stable_symbol(symbol)
        weight_to_units(values[symbol], label=f"{label} weight")
    return MappingProxyType({symbol: values[symbol] for symbol in sorted(values)})


def _freeze_asset_weights(
    values: Mapping[AssetType, Decimal], *, label: str
) -> Mapping[AssetType, Decimal]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} must be a mapping")
    for asset_type, value in values.items():
        if type(asset_type) is not AssetType:
            raise TypeError(f"{label} keys must be exact AssetType values")
        weight_to_units(value, label=f"{label} weight")
    return MappingProxyType(
        {
            asset_type: values[asset_type]
            for asset_type in sorted(values, key=lambda item: item.value)
        }
    )


def _symbol_units(values: Mapping[str, Decimal], *, label: str) -> dict[str, int]:
    return {
        symbol: weight_to_units(weight, label=label) for symbol, weight in values.items()
    }


@dataclass(frozen=True, slots=True)
class SleeveTarget:
    """Replay-validated strategy stages ending in the final attributed contribution."""

    strategy_id: str
    strategy_budget: Decimal
    requested_weights: Mapping[str, Decimal]
    normalized_weights: Mapping[str, Decimal]
    budgeted_weights: Mapping[str, Decimal]
    asset_limited_weights: Mapping[str, Decimal]
    final_weights: Mapping[str, Decimal]

    def __post_init__(self) -> None:
        strategy_id = _stable_strategy_id(self.strategy_id)
        strategy_budget_units = weight_to_units(self.strategy_budget, label="strategy budget")
        requested = _freeze_symbol_weights(self.requested_weights, label="requested")
        normalized = _freeze_symbol_weights(self.normalized_weights, label="normalized")
        budgeted = _freeze_symbol_weights(self.budgeted_weights, label="budgeted")
        asset_limited = _freeze_symbol_weights(
            self.asset_limited_weights, label="asset-limited"
        )
        final = _freeze_symbol_weights(self.final_weights, label="final")
        stages = (requested, normalized, budgeted, asset_limited, final)
        if any(set(stage) != set(requested) for stage in stages[1:]):
            raise ValueError("all sleeve weight stages must contain the same instruments")

        requested_contributions: dict[ContributionKey, int] = {
            (strategy_id, symbol): weight_to_units(weight, label="requested weight")
            for symbol, weight in requested.items()
        }
        strategy_trace = replay_strategy_stages(
            requested_contributions, strategy_id, strategy_budget_units
        )
        expected_normalized = {
            key[1]: units for key, units in strategy_trace.normalized.items()
        }
        expected_budgeted = {
            key[1]: units for key, units in strategy_trace.budgeted.items()
        }
        if _symbol_units(normalized, label="normalized weight") != expected_normalized:
            raise ValueError("normalized weights do not match fixed-point replay")
        if _symbol_units(budgeted, label="budgeted weight") != expected_budgeted:
            raise ValueError("budgeted weights do not match fixed-point replay")

        unit_stages = [
            _symbol_units(stage, label="sleeve stage weight") for stage in stages
        ]
        for earlier, later in zip(unit_stages[2:], unit_stages[3:]):
            if any(later[symbol] > earlier[symbol] for symbol in earlier):
                raise ValueError("post-budget sleeve weights must be non-increasing")

        object.__setattr__(self, "strategy_id", strategy_id)
        object.__setattr__(self, "requested_weights", requested)
        object.__setattr__(self, "normalized_weights", normalized)
        object.__setattr__(self, "budgeted_weights", budgeted)
        object.__setattr__(self, "asset_limited_weights", asset_limited)
        object.__setattr__(self, "final_weights", final)

    @property
    def weights(self) -> Mapping[str, Decimal]:
        return self.final_weights

    @property
    def pre_net_weights(self) -> Mapping[str, Decimal]:
        """Compatibility alias; final weights are reattributed after symbol netting."""

        return self.final_weights


@dataclass(frozen=True, slots=True)
class PortfolioTarget:
    """Policy-bound target whose complete fixed-point trace is replayed on construction."""

    policy: AllocationPolicy
    sleeves: Mapping[str, SleeveTarget]
    weights: Mapping[str, Decimal]
    symbol_asset_types: Mapping[str, AssetType]
    asset_weights: Mapping[AssetType, Decimal]
    cash_weight: Decimal
    adjustments: tuple[AllocationAdjustment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.policy, AllocationPolicy):
            raise TypeError("policy must be an AllocationPolicy")
        if not isinstance(self.sleeves, Mapping):
            raise TypeError("sleeves must be a mapping")
        sleeves: dict[str, SleeveTarget] = {}
        for strategy_id, sleeve in self.sleeves.items():
            stable_id = _stable_strategy_id(strategy_id)
            if not isinstance(sleeve, SleeveTarget):
                raise TypeError("sleeves must contain SleeveTarget values")
            if sleeve.strategy_id != stable_id:
                raise ValueError("sleeve key must match its strategy id")
            if stable_id not in self.policy.strategy_budgets:
                raise ValueError("sleeve strategy is not configured in policy")
            if weight_to_units(
                sleeve.strategy_budget, label="sleeve strategy budget"
            ) != weight_to_units(
                self.policy.strategy_budgets[stable_id], label="policy strategy budget"
            ):
                raise ValueError("sleeve strategy budget does not match policy")
            if not sleeve.requested_weights:
                raise ValueError("sleeve requested weights must not be empty")
            sleeves[stable_id] = sleeve

        requested: dict[ContributionKey, int] = {
            (strategy_id, symbol): weight_to_units(weight, label="requested weight")
            for strategy_id, sleeve in sleeves.items()
            for symbol, weight in sleeve.requested_weights.items()
        }
        replay = replay_allocation(self.policy, requested)

        for strategy_id, sleeve in sleeves.items():
            keys = [key for key in replay.requested if key[0] == strategy_id]
            expected_stages = (
                ("requested", sleeve.requested_weights, replay.requested),
                ("normalized", sleeve.normalized_weights, replay.normalized),
                ("budgeted", sleeve.budgeted_weights, replay.budgeted),
                ("asset-limited", sleeve.asset_limited_weights, replay.asset_limited),
                ("final", sleeve.final_weights, replay.final_contributions),
            )
            for stage_name, actual, expected in expected_stages:
                expected_units = {key[1]: expected[key] for key in keys}
                if _symbol_units(actual, label=f"{stage_name} weight") != expected_units:
                    raise ValueError(
                        f"sleeve {stage_name} weights do not match fixed-point replay"
                    )

        weights = _freeze_symbol_weights(self.weights, label="portfolio")
        if _symbol_units(weights, label="portfolio weight") != dict(replay.final_symbols):
            raise ValueError("portfolio symbol weights do not match fixed-point replay")

        if not isinstance(self.symbol_asset_types, Mapping):
            raise TypeError("symbol asset types must be a mapping")
        checked_symbol_assets: dict[str, AssetType] = {}
        for symbol, asset_type in self.symbol_asset_types.items():
            stable_symbol = _stable_symbol(symbol)
            if type(asset_type) is not AssetType:
                raise TypeError("symbol asset type values must be exact AssetType values")
            checked_symbol_assets[stable_symbol] = asset_type
        if checked_symbol_assets != dict(replay.symbol_asset_types):
            raise ValueError("symbol asset types do not match fixed-point replay")

        asset_weights = _freeze_asset_weights(self.asset_weights, label="asset weights")
        actual_asset_units = {
            asset_type: weight_to_units(weight, label="asset weight")
            for asset_type, weight in asset_weights.items()
        }
        if actual_asset_units != dict(replay.asset_units):
            raise ValueError("asset weights do not match fixed-point replay")
        if weight_to_units(self.cash_weight, label="cash weight") != replay.cash_units:
            raise ValueError("cash weight does not match fixed-point replay")

        if type(self.adjustments) is not tuple:
            raise TypeError("adjustments must be an exact tuple")
        if any(not isinstance(adjustment, AllocationAdjustment) for adjustment in self.adjustments):
            raise TypeError("adjustments must contain AllocationAdjustment values")
        if self.adjustments != replay.adjustments:
            raise ValueError("adjustment trace does not match fixed-point replay")

        object.__setattr__(self, "sleeves", MappingProxyType(dict(sorted(sleeves.items()))))
        object.__setattr__(self, "weights", weights)
        object.__setattr__(
            self,
            "symbol_asset_types",
            MappingProxyType(dict(sorted(checked_symbol_assets.items()))),
        )
        object.__setattr__(self, "asset_weights", asset_weights)


__all__ = [
    "AllocationAdjustment",
    "AllocationPolicy",
    "AllocationStage",
    "PortfolioTarget",
    "SleeveTarget",
]
