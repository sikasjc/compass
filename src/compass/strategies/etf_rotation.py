from __future__ import annotations

from decimal import Decimal
from math import fsum, isfinite
from typing import Annotated

import pandas as pd  # type: ignore[import-untyped]
from pydantic import Field, model_validator

from compass.domain.market import AssetType, InstrumentId
from compass.domain.trading import TargetIntent
from compass.domain.weights import (
    WEIGHT_SCALE,
    largest_remainder_units,
    units_to_weight,
)
from compass.strategies.base import (
    StrategyContext,
    StrategyDecision,
    StrategyDecisionStatus,
    StrategyFrequency,
    StrategyMetadata,
    StrategyParameters,
)
from compass.strategies.indicators import annualized_volatility, simple_moving_average
from compass.strategies.momentum import (
    NonNegativeFiniteFloat,
    RebalanceFrequency,
    WindowInt,
    _buffered_selection,
    _is_rebalance_session,
    _normalize_strategy_id,
    _prepare_context,
    _weighted_momentum,
)


FinitePenalty = Annotated[float, Field(strict=True, ge=0, le=100, allow_inf_nan=False)]


class EtfRotationParameters(StrategyParameters):
    lookbacks: tuple[WindowInt, ...] = Field(
        default=(20, 60, 120), description="用于计算 ETF 动量的递增交易日窗口。"
    )
    lookback_weights: tuple[NonNegativeFiniteFloat, ...] = Field(
        default=(1.0, 1.0, 1.0), description="各动量窗口的非负有限组合权重。"
    )
    trend_window: int = Field(
        default=120,
        strict=True,
        ge=2,
        le=10_000,
        description="收盘价趋势过滤均线窗口。",
    )
    volatility_window: int = Field(
        default=20, strict=True, ge=2, le=10_000, description="年化波动率惩罚窗口。"
    )
    volatility_penalty: FinitePenalty = Field(
        default=0.5, description="从综合动量中扣除年化波动率的系数。"
    )
    top_n: int = Field(
        default=3, strict=True, ge=1, le=10_000, description="通过过滤后选择的 ETF 数量。"
    )
    turnover_buffer: int = Field(
        default=0,
        strict=True,
        ge=0,
        le=10_000,
        description="允许真实持仓 ETF 保留的额外排名缓冲位数。",
    )
    risk_alternative: str | None = Field(
        default=None,
        strict=True,
        description="无风险候选时使用且必须存在于标的池的 ETF 替代标的。",
    )
    rebalance_frequency: RebalanceFrequency = Field(
        default=RebalanceFrequency.WEEKLY, description="日、周或月调仓频率。"
    )

    @model_validator(mode="after")
    def validate_windows_and_weights(self) -> EtfRotationParameters:
        if not self.lookbacks:
            raise ValueError("lookbacks must not be empty")
        if tuple(sorted(set(self.lookbacks))) != self.lookbacks:
            raise ValueError("lookbacks must be unique and strictly increasing")
        if len(self.lookback_weights) != len(self.lookbacks):
            raise ValueError("lookback_weights must match lookbacks")
        try:
            total = fsum(self.lookback_weights)
        except OverflowError:
            raise ValueError("lookback_weights sum must be finite and positive") from None
        if not isfinite(total) or total <= 0:
            raise ValueError("lookback_weights sum must be finite and positive")
        if self.risk_alternative is not None:
            try:
                InstrumentId.parse(self.risk_alternative)
            except (TypeError, ValueError):
                raise ValueError("risk_alternative must be a canonical instrument id") from None
        return self


def _score_weights(selected: list[tuple[InstrumentId, float]]) -> tuple[Decimal, ...]:
    positive = [Decimal(str(score)) for _, score in selected]
    exponents = [value.as_tuple().exponent for value in positive]
    if any(not isinstance(exponent, int) for exponent in exponents):
        raise ValueError("ETF score weights must be finite")
    integer_exponents = [exponent for exponent in exponents if isinstance(exponent, int)]
    minimum_exponent = min(integer_exponents)
    integer_scores: dict[InstrumentId, int] = {}
    for (instrument, _), value in zip(selected, positive, strict=True):
        _, digits, exponent = value.as_tuple()
        if not isinstance(exponent, int):  # guarded above for mypy and defensive clarity
            raise ValueError("ETF score weights must be finite")
        coefficient = int("".join(str(digit) for digit in digits))
        integer_scores[instrument] = coefficient * (10 ** (exponent - minimum_exponent))
    allocated, _ = largest_remainder_units(
        integer_scores,
        WEIGHT_SCALE,
        canonical_key=lambda instrument: str(instrument),
    )
    return tuple(units_to_weight(allocated[instrument]) for instrument, _ in selected)


class EtfRotationStrategy:
    strategy_type = "etf_rotation"
    parameters_type = EtfRotationParameters
    minimum_history = 3
    metadata = StrategyMetadata(
        strategy_type=strategy_type,
        version="1.0.0",
        display_name="ETF 轮动",
        description="通过多周期动量、趋势过滤和波动率惩罚选择 ETF。",
        supported_asset_types=frozenset({AssetType.ETF}),
        supported_frequencies=frozenset({StrategyFrequency.DAILY}),
        required_fields=frozenset({"close"}),
        minimum_history=minimum_history,
        default_required_history=121,
        parameters_type=parameters_type,
    )

    def __init__(
        self,
        parameters: EtfRotationParameters | None = None,
        strategy_id: str = "etf_rotation",
    ) -> None:
        self.parameters = parameters or EtfRotationParameters()
        self.required_history = max(
            max(self.parameters.lookbacks) + 1,
            self.parameters.trend_window,
            self.parameters.volatility_window + 1,
        )
        self.strategy_id = _normalize_strategy_id(strategy_id)

    def generate_targets(self, context: StrategyContext) -> StrategyDecision:
        prepared = _prepare_context(context, self.metadata.supported_asset_types)
        if isinstance(prepared, StrategyDecision):
            return prepared
        if not _is_rebalance_session(prepared.calendar, self.parameters.rebalance_frequency):
            return StrategyDecision.empty(
                StrategyDecisionStatus.SKIPPED,
                "NOT_REBALANCE_SESSION",
                details=prepared.details,
            )
        alternative = (
            InstrumentId.parse(self.parameters.risk_alternative)
            if self.parameters.risk_alternative is not None
            else None
        )
        configured_primary = tuple(
            instrument
            for instrument in context.instruments
            if context.asset_types[instrument] is AssetType.ETF and instrument != alternative
        )
        fresh_primary = tuple(
            instrument for instrument in prepared.instruments if instrument != alternative
        )
        stale_primary_count = len(configured_primary) - len(fresh_primary)
        if stale_primary_count > 0:
            return StrategyDecision.empty(
                StrategyDecisionStatus.SKIPPED,
                "STALE_DATA",
                details={
                    **prepared.details,
                    "stale_primary_instruments": stale_primary_count,
                },
            )
        ranked: list[tuple[InstrumentId, float]] = []
        skipped_insufficient = 0
        skipped_invalid_indicator = 0
        primary_count = 0
        for instrument in prepared.instruments:
            if instrument == alternative:
                continue
            primary_count += 1
            history = prepared.histories[instrument]
            close = history["close"]
            if len(close) < self.required_history:
                skipped_insufficient += 1
                continue
            momentum = _weighted_momentum(
                close, self.parameters.lookbacks, self.parameters.lookback_weights
            )
            if momentum is None:
                continue
            try:
                trend = float(simple_moving_average(close, self.parameters.trend_window).iloc[-1])
                volatility = float(
                    annualized_volatility(close, self.parameters.volatility_window).iloc[-1]
                )
            except (ArithmeticError, ValueError):
                skipped_invalid_indicator += 1
                continue
            if not isfinite(trend) or not isfinite(volatility) or float(close.iloc[-1]) <= trend:
                continue
            score = momentum - self.parameters.volatility_penalty * volatility
            if isfinite(score) and score > 0:
                ranked.append((instrument, score))
        ranked.sort(key=lambda item: (-item[1], str(item[0])))
        selected = _buffered_selection(
            ranked, context, self.parameters.top_n, self.parameters.turnover_buffer
        )
        risk_alternative_usable = (
            alternative is not None
            and alternative in prepared.instruments
            and context.asset_types[alternative] is AssetType.ETF
            and prepared.histories[alternative].index[-1] == pd.Timestamp(context.as_of)
            and len(prepared.histories[alternative]) >= self.parameters.volatility_window + 1
        )
        details = {
            **prepared.details,
            "required_history": self.required_history,
            "risk_alternative_usable": risk_alternative_usable,
            "selected_count": len(selected),
            "skipped_insufficient_history": skipped_insufficient,
            "skipped_invalid_indicator": skipped_invalid_indicator,
        }
        if selected:
            weights = _score_weights(selected)
            intents = tuple(
                TargetIntent(
                    self.strategy_id,
                    instrument,
                    weight,
                    score,
                    1.0,
                    "MOMENTUM_TOP_N",
                    context.as_of,
                )
                for (instrument, score), weight in zip(selected, weights, strict=True)
            )
            return StrategyDecision.generated(intents, details=details)
        if primary_count == 0:
            return StrategyDecision.empty(
                StrategyDecisionStatus.CASH,
                "NO_CANDIDATES_CASH",
                details=details,
            )
        if skipped_insufficient == primary_count:
            return StrategyDecision.empty(
                StrategyDecisionStatus.SKIPPED,
                "INSUFFICIENT_HISTORY",
                details=details,
            )
        if risk_alternative_usable and alternative is not None:
            intent = TargetIntent(
                self.strategy_id,
                alternative,
                Decimal("1"),
                0.0,
                1.0,
                "RISK_ALTERNATIVE",
                context.as_of,
            )
            return StrategyDecision.generated((intent,), details=details)
        return StrategyDecision.empty(
            StrategyDecisionStatus.CASH, "NO_CANDIDATES_CASH", details=details
        )
