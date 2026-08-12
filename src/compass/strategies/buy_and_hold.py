from __future__ import annotations

from decimal import Decimal

from pydantic import Field, model_validator

from compass.domain.market import AssetType
from compass.domain.trading import TargetIntent
from compass.domain.weights import weight_to_units
from compass.strategies.base import (
    StrategyContext,
    StrategyDecision,
    StrategyFrequency,
    StrategyMetadata,
    StrategyParameters,
)
from compass.strategies.momentum import (
    _equal_weights,
    _normalize_strategy_id,
    _prepare_context,
)


class BuyAndHoldParameters(StrategyParameters):
    target_weight: Decimal = Field(
        default=Decimal("1"),
        strict=True,
        allow_inf_nan=False,
        gt=0,
        le=1,
        description="策略袖套内所有买入持有标的的合计目标权重。",
    )

    @model_validator(mode="after")
    def validate_weight(self) -> BuyAndHoldParameters:
        weight_to_units(self.target_weight, label="target_weight")
        return self


class BuyAndHoldStrategy:
    strategy_type = "buy_and_hold"
    parameters_type = BuyAndHoldParameters
    minimum_history = 1
    metadata = StrategyMetadata(
        strategy_type=strategy_type,
        version="1.0.0",
        display_name="买入持有",
        description="按设定权重买入候选标的并长期持有，作为主动策略的基础对照。",
        supported_asset_types=frozenset({AssetType.ETF, AssetType.STOCK}),
        supported_frequencies=frozenset({StrategyFrequency.DAILY}),
        required_fields=frozenset({"close"}),
        minimum_history=minimum_history,
        default_required_history=1,
        parameters_type=parameters_type,
    )

    def __init__(
        self,
        parameters: BuyAndHoldParameters | None = None,
        strategy_id: str = "buy_and_hold",
    ) -> None:
        self.parameters = parameters or BuyAndHoldParameters()
        self.required_history = 1
        self.strategy_id = _normalize_strategy_id(strategy_id)

    def generate_targets(self, context: StrategyContext) -> StrategyDecision:
        prepared = _prepare_context(context, self.metadata.supported_asset_types)
        if isinstance(prepared, StrategyDecision):
            return prepared
        weights = _equal_weights(len(prepared.instruments), self.parameters.target_weight)
        intents = tuple(
            TargetIntent(
                strategy_id=self.strategy_id,
                instrument=instrument,
                target_weight=weight,
                score=1.0,
                confidence=1.0,
                reason_code="BUY_AND_HOLD_TARGET",
                valid_until=context.as_of,
            )
            for instrument, weight in zip(prepared.instruments, weights, strict=True)
        )
        return StrategyDecision.generated(
            intents,
            details={**prepared.details, "selected_count": len(intents)},
        )
