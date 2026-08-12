from __future__ import annotations

from decimal import Decimal
from math import isfinite

from pydantic import Field, model_validator

from compass.domain.market import AssetType, InstrumentId
from compass.domain.trading import TargetIntent
from compass.domain.weights import weight_to_units
from compass.strategies.base import (
    StrategyContext,
    StrategyDecision,
    StrategyDecisionStatus,
    StrategyFrequency,
    StrategyMetadata,
    StrategyParameters,
)
from compass.strategies.indicators import simple_moving_average
from compass.strategies.momentum import (
    _equal_weights,
    _normalize_strategy_id,
    _prepare_context,
)


class DualMaParameters(StrategyParameters):
    short_window: int = Field(
        default=20, strict=True, ge=1, le=10_000, description="短期简单移动平均交易日窗口。"
    )
    long_window: int = Field(
        default=60, strict=True, ge=2, le=10_000, description="长期简单移动平均交易日窗口。"
    )
    confirmation_days: int = Field(
        default=1, strict=True, ge=1, le=10_000, description="连续收盘确认趋势所需交易日数。"
    )
    target_weight: Decimal = Field(
        default=Decimal("1"),
        strict=True,
        allow_inf_nan=False,
        ge=0,
        le=1,
        description="策略袖套内所有多头目标合计权重。",
    )

    @model_validator(mode="after")
    def validate_windows(self) -> DualMaParameters:
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be less than long_window")
        weight_to_units(self.target_weight, label="target_weight")
        return self


class DualMaStrategy:
    strategy_type = "dual_ma"
    parameters_type = DualMaParameters
    minimum_history = 2
    metadata = StrategyMetadata(
        strategy_type=strategy_type,
        version="1.0.0",
        display_name="双均线趋势",
        description="使用收盘确认的短期与长期均线关系生成趋势目标。",
        supported_asset_types=frozenset({AssetType.ETF, AssetType.STOCK}),
        supported_frequencies=frozenset({StrategyFrequency.DAILY}),
        required_fields=frozenset({"close"}),
        minimum_history=minimum_history,
        default_required_history=60,
        parameters_type=parameters_type,
    )

    def __init__(
        self, parameters: DualMaParameters | None = None, strategy_id: str = "dual_ma"
    ) -> None:
        self.parameters = parameters or DualMaParameters()
        self.required_history = (
            self.parameters.long_window + self.parameters.confirmation_days - 1
        )
        self.strategy_id = _normalize_strategy_id(strategy_id)

    def generate_targets(self, context: StrategyContext) -> StrategyDecision:
        prepared = _prepare_context(context, self.metadata.supported_asset_types)
        if isinstance(prepared, StrategyDecision):
            return prepared
        candidates: list[tuple[InstrumentId, float, str, bool]] = []
        skipped_insufficient = 0
        for instrument in prepared.instruments:
            close = prepared.histories[instrument]["close"]
            if len(close) < self.required_history:
                skipped_insufficient += 1
                continue
            short = simple_moving_average(close, self.parameters.short_window)
            long = simple_moving_average(close, self.parameters.long_window)
            spread = short - long
            recent = spread.iloc[-self.parameters.confirmation_days :]
            if recent.isna().any():
                skipped_insufficient += 1
                continue
            score = float(spread.iloc[-1]) / float(long.iloc[-1])
            if not isfinite(score):
                continue
            holding = context.holding(instrument)
            if (recent > 0).all():
                candidates.append((instrument, score, "MA_BULL_CONFIRMED", True))
            elif holding is not None and holding.quantity > 0 and (recent < 0).all():
                candidates.append((instrument, score, "MA_BEAR_EXIT", False))
        candidates.sort(key=lambda item: (-item[1], str(item[0])))
        active_count = sum(is_active for _, _, _, is_active in candidates)
        active_weights = iter(_equal_weights(active_count, self.parameters.target_weight))
        intents = tuple(
            TargetIntent(
                strategy_id=self.strategy_id,
                instrument=instrument,
                target_weight=next(active_weights) if is_active else Decimal("0"),
                score=score,
                confidence=min(1.0, abs(score)),
                reason_code=reason,
                valid_until=context.as_of,
            )
            for instrument, score, reason, is_active in candidates
        )
        details = {
            **prepared.details,
            "active_count": active_count,
            "required_history": self.required_history,
            "skipped_insufficient_history": skipped_insufficient,
        }
        if intents:
            return StrategyDecision.generated(intents, details=details)
        if skipped_insufficient == len(prepared.instruments):
            return StrategyDecision.empty(
                StrategyDecisionStatus.SKIPPED,
                "INSUFFICIENT_HISTORY",
                details=details,
            )
        return StrategyDecision.empty(
            StrategyDecisionStatus.CASH, "NO_CANDIDATES_CASH", details=details
        )
