from __future__ import annotations

from decimal import Decimal
from math import isfinite
from typing import Annotated

import pandas as pd  # type: ignore[import-untyped]
from pydantic import Field, model_validator

from compass.domain.market import AssetType, InstrumentId
from compass.domain.trading import TargetIntent
from compass.domain.weights import weight_to_units
from compass.strategies.base import (
    HoldingSummary,
    StrategyContext,
    StrategyDecision,
    StrategyDecisionStatus,
    StrategyFrequency,
    StrategyMetadata,
    StrategyParameters,
)
from compass.strategies.indicators import bollinger_bands, rsi
from compass.strategies.momentum import (
    _equal_weights,
    _normalize_strategy_id,
    _prepare_context,
)


FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class MeanReversionParameters(StrategyParameters):
    rsi_window: int = Field(
        default=14, strict=True, ge=2, le=10_000, description="Wilder RSI 的交易日窗口。"
    )
    bollinger_window: int = Field(
        default=20, strict=True, ge=2, le=10_000, description="布林带均值和波动窗口。"
    )
    bollinger_std: FiniteFloat = Field(
        default=2.0, gt=0, le=100, description="布林带标准差倍数。"
    )
    entry_rsi: FiniteFloat = Field(
        default=30.0, ge=0, le=100, description="入场所需的最高 RSI。"
    )
    exit_rsi: FiniteFloat = Field(
        default=60.0, ge=0, le=100, description="指标退出所需的最低 RSI。"
    )
    stop_loss: FiniteFloat = Field(
        default=0.08, gt=0, lt=1, description="相对平均成本的止损比例。"
    )
    max_holding_sessions: int = Field(
        default=20,
        strict=True,
        ge=1,
        le=10_000,
        description="包含建仓日在内允许持有的最长可见交易日数。",
    )
    target_weight: Decimal = Field(
        default=Decimal("0.2"),
        strict=True,
        allow_inf_nan=False,
        gt=0,
        le=1,
        description="策略袖套内所有正权重目标的合计权重。",
    )

    @model_validator(mode="after")
    def validate_thresholds(self) -> MeanReversionParameters:
        if self.entry_rsi >= self.exit_rsi:
            raise ValueError("entry_rsi must be less than exit_rsi")
        weight_to_units(self.target_weight, label="target_weight")
        return self


class MeanReversionStrategy:
    strategy_type = "mean_reversion"
    parameters_type = MeanReversionParameters
    minimum_history = 3
    metadata = StrategyMetadata(
        strategy_type=strategy_type,
        version="1.0.0",
        display_name="RSI/布林均值回归",
        description="结合 RSI 与布林带，按持仓成本和交易日龄管理退出。",
        supported_asset_types=frozenset({AssetType.ETF, AssetType.STOCK}),
        supported_frequencies=frozenset({StrategyFrequency.DAILY}),
        required_fields=frozenset({"close"}),
        minimum_history=minimum_history,
        default_required_history=20,
        parameters_type=parameters_type,
    )

    def __init__(
        self,
        parameters: MeanReversionParameters | None = None,
        strategy_id: str = "mean_reversion",
    ) -> None:
        self.parameters = parameters or MeanReversionParameters()
        self.required_history = max(
            self.parameters.rsi_window + 1,
            self.parameters.bollinger_window,
            self.parameters.max_holding_sessions,
        )
        self.strategy_id = _normalize_strategy_id(strategy_id)

    def _holding_reason(
        self,
        holding: HoldingSummary,
        close: float,
        indicator_rsi: float,
        middle: float,
        index: pd.DatetimeIndex,
    ) -> tuple[str, bool]:
        if holding.average_cost > 0 and Decimal(str(close)) <= holding.average_cost * (
            Decimal("1") - Decimal(str(self.parameters.stop_loss))
        ):
            return "MEAN_REVERSION_STOP_LOSS", False
        if holding.holding_since is not None:
            sessions = int((index.date >= holding.holding_since).sum())
            if sessions >= self.parameters.max_holding_sessions:
                return "MEAN_REVERSION_MAX_HOLDING", False
        if indicator_rsi >= self.parameters.exit_rsi or close >= middle:
            return "MEAN_REVERSION_SIGNAL_EXIT", False
        return "MEAN_REVERSION_HOLD", True

    def generate_targets(self, context: StrategyContext) -> StrategyDecision:
        prepared = _prepare_context(context, self.metadata.supported_asset_types)
        if isinstance(prepared, StrategyDecision):
            return prepared
        candidates: list[tuple[InstrumentId, float, str, bool]] = []
        skipped_insufficient = 0
        skipped_invalid_indicator = 0
        for instrument in prepared.instruments:
            history = prepared.histories[instrument]
            close_series = history["close"]
            if len(close_series) < self.required_history:
                skipped_insufficient += 1
                continue
            try:
                indicator_rsi = float(rsi(close_series, self.parameters.rsi_window).iloc[-1])
                bands = bollinger_bands(
                    close_series,
                    self.parameters.bollinger_window,
                    self.parameters.bollinger_std,
                )
            except (ArithmeticError, ValueError):
                skipped_invalid_indicator += 1
                continue
            close = float(close_series.iloc[-1])
            middle = float(bands.middle.iloc[-1])
            lower = float(bands.lower.iloc[-1])
            if not all(isfinite(value) for value in (indicator_rsi, close, middle, lower)):
                skipped_insufficient += 1
                continue
            holding = context.holding(instrument)
            if holding is not None and holding.quantity > 0:
                reason, is_active = self._holding_reason(
                    holding, close, indicator_rsi, middle, history.index
                )
            elif indicator_rsi <= self.parameters.entry_rsi and close <= lower:
                reason, is_active = "MEAN_REVERSION_ENTRY", True
            else:
                continue
            score = (self.parameters.entry_rsi - indicator_rsi) / 100.0
            if isfinite(score):
                candidates.append((instrument, score, reason, is_active))
        candidates.sort(key=lambda item: (-item[1], str(item[0])))
        active_count = sum(is_active for _, _, _, is_active in candidates)
        active_weights = iter(_equal_weights(active_count, self.parameters.target_weight))
        intents = tuple(
            TargetIntent(
                self.strategy_id,
                instrument,
                next(active_weights) if is_active else Decimal("0"),
                score,
                min(1.0, abs(score)),
                reason,
                context.as_of,
            )
            for instrument, score, reason, is_active in candidates
        )
        details = {
            **prepared.details,
            "active_count": active_count,
            "required_history": self.required_history,
            "skipped_insufficient_history": skipped_insufficient,
            "skipped_invalid_indicator": skipped_invalid_indicator,
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
