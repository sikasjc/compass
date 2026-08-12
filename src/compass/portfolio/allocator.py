from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal

from compass.domain.market import InstrumentId
from compass.domain.trading import TargetIntent
from compass.domain.weights import units_to_weight, weight_to_units
from compass.portfolio.models import PortfolioTarget, SleeveTarget
from compass.portfolio.trace import (
    AllocationPolicy,
    ContributionKey,
    replay_allocation,
)


def _weights(values: Mapping[str, int]) -> dict[str, Decimal]:
    return {key: units_to_weight(values[key]) for key in sorted(values)}


class DeterministicAllocator:
    """Validate intents and materialize the shared fixed-point replay trace."""

    def allocate(
        self, intents: Sequence[TargetIntent], policy: AllocationPolicy
    ) -> PortfolioTarget:
        if not isinstance(intents, Sequence) or isinstance(intents, (str, bytes)):
            raise TypeError("intents must be a sequence of TargetIntent values")
        if not isinstance(policy, AllocationPolicy):
            raise TypeError("policy must be an AllocationPolicy")

        requested: dict[ContributionKey, int] = {}
        for intent in intents:
            if not isinstance(intent, TargetIntent):
                raise TypeError("intents must contain TargetIntent values")
            if (
                type(intent.strategy_id) is not str
                or not intent.strategy_id
                or intent.strategy_id != intent.strategy_id.strip()
            ):
                raise ValueError("intent must use a stable strategy id")
            if intent.strategy_id not in policy.strategy_budgets:
                raise ValueError(f"unknown strategy: {intent.strategy_id}")
            if type(intent.instrument) is not InstrumentId:
                raise TypeError("intent instrument must be an exact InstrumentId")
            if intent.instrument not in policy.asset_types:
                raise ValueError(f"unknown instrument asset type: {intent.instrument}")
            key = (intent.strategy_id, str(intent.instrument))
            if key in requested:
                raise ValueError(
                    f"duplicate strategy/instrument intent: {intent.strategy_id}/{intent.instrument}"
                )
            requested[key] = weight_to_units(
                intent.target_weight, label="intent target weight"
            )

        trace = replay_allocation(policy, requested)
        strategy_ids = sorted({key[0] for key in trace.requested})
        sleeves: dict[str, SleeveTarget] = {}
        for strategy_id in strategy_ids:
            keys = [key for key in trace.requested if key[0] == strategy_id]
            sleeves[strategy_id] = SleeveTarget(
                strategy_id=strategy_id,
                strategy_budget=policy.strategy_budgets[strategy_id],
                requested_weights=_weights(
                    {key[1]: trace.requested[key] for key in keys}
                ),
                normalized_weights=_weights(
                    {key[1]: trace.normalized[key] for key in keys}
                ),
                budgeted_weights=_weights(
                    {key[1]: trace.budgeted[key] for key in keys}
                ),
                asset_limited_weights=_weights(
                    {key[1]: trace.asset_limited[key] for key in keys}
                ),
                final_weights=_weights(
                    {key[1]: trace.final_contributions[key] for key in keys}
                ),
            )

        return PortfolioTarget(
            policy=policy,
            sleeves=sleeves,
            weights=_weights(trace.final_symbols),
            symbol_asset_types=trace.symbol_asset_types,
            asset_weights={
                asset_type: units_to_weight(units)
                for asset_type, units in trace.asset_units.items()
            },
            cash_weight=units_to_weight(trace.cash_units),
            adjustments=trace.adjustments,
        )
