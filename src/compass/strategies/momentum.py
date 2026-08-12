from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from math import fsum, isfinite
from typing import Annotated

import pandas as pd  # type: ignore[import-untyped]
from pydantic import Field, model_validator

from compass.domain.market import AssetType, InstrumentId
from compass.domain.trading import TargetIntent
from compass.domain.weights import (
    largest_remainder_units,
    units_to_weight,
    weight_to_units,
)
from compass.strategies.base import (
    StrategyContext,
    StrategyDecision,
    StrategyDecisionStatus,
    StrategyFrequency,
    StrategyMetadata,
    StrategyParameters,
)


class RebalanceFrequency(StrEnum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


WindowInt = Annotated[int, Field(strict=True, gt=0, le=10_000)]
NonNegativeFiniteFloat = Annotated[
    float, Field(strict=True, ge=0, allow_inf_nan=False)
]


class CrossSectionalMomentumParameters(StrategyParameters):
    lookbacks: tuple[WindowInt, ...] = Field(
        default=(60, 120), description="用于计算截面收益率的递增交易日窗口。"
    )
    lookback_weights: tuple[NonNegativeFiniteFloat, ...] = Field(
        default=(1.0, 1.0), description="各动量窗口的非负有限组合权重。"
    )
    top_n: int = Field(
        default=3, strict=True, ge=1, le=10_000, description="每次调仓选择的最高排名标的数。"
    )
    turnover_buffer: int = Field(
        default=0,
        strict=True,
        ge=0,
        le=10_000,
        description="允许真实持仓标的保留的额外排名缓冲位数。",
    )
    rebalance_frequency: RebalanceFrequency = Field(
        default=RebalanceFrequency.WEEKLY, description="日、周或月调仓频率。"
    )

    @model_validator(mode="after")
    def validate_windows_and_weights(self) -> CrossSectionalMomentumParameters:
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
        return self


@dataclass(frozen=True, slots=True)
class _PreparedContext:
    histories: dict[InstrumentId, pd.DataFrame]
    calendar: pd.DatetimeIndex
    instruments: tuple[InstrumentId, ...]
    details: dict[str, object]


def _visible_calendar(histories: dict[InstrumentId, pd.DataFrame]) -> pd.DatetimeIndex:
    dates = sorted({timestamp for history in histories.values() for timestamp in history.index})
    return pd.DatetimeIndex(dates)


def _prepare_context(
    context: StrategyContext, supported_assets: frozenset[AssetType]
) -> _PreparedContext | StrategyDecision:
    if not context.instruments:
        return StrategyDecision.empty(StrategyDecisionStatus.SKIPPED, "NO_INSTRUMENTS")
    if set(context.asset_types) != set(context.instruments):
        return StrategyDecision.empty(
            StrategyDecisionStatus.SKIPPED,
            "MISSING_ASSET_METADATA",
            details={
                "missing_asset_metadata": len(context.instruments) - len(context.asset_types)
            },
        )
    histories = {instrument: context.history(instrument) for instrument in context.instruments}
    calendar = _visible_calendar(histories)
    if len(calendar) == 0 or calendar[-1] != pd.Timestamp(context.as_of):
        return StrategyDecision.empty(
            StrategyDecisionStatus.SKIPPED,
            "STALE_DATA",
            details={
                "current_session": str(calendar[-1].date()) if len(calendar) else "NONE",
                "expected_session": str(context.as_of),
            },
        )
    supported = tuple(
        instrument
        for instrument in context.instruments
        if context.asset_types[instrument] in supported_assets
    )
    skipped_unsupported = len(context.instruments) - len(supported)
    if not supported:
        return StrategyDecision.empty(
            StrategyDecisionStatus.SKIPPED,
            "UNSUPPORTED_ASSET",
            details={
                "eligible_instruments": 0,
                "skipped_stale": 0,
                "skipped_unsupported_asset": skipped_unsupported,
            },
        )
    fresh = tuple(
        instrument
        for instrument in supported
        if len(histories[instrument]) > 0
        and histories[instrument].index[-1] == pd.Timestamp(context.as_of)
    )
    skipped_stale = len(supported) - len(fresh)
    details: dict[str, object] = {
        "eligible_instruments": len(fresh),
        "skipped_stale": skipped_stale,
        "skipped_unsupported_asset": skipped_unsupported,
    }
    if not fresh:
        if skipped_stale:
            return StrategyDecision.empty(
                StrategyDecisionStatus.SKIPPED, "STALE_DATA", details=details
            )
        return StrategyDecision.empty(
            StrategyDecisionStatus.CASH, "NO_CANDIDATES_CASH", details=details
        )
    return _PreparedContext(histories, calendar, fresh, details)


def _normalize_strategy_id(strategy_id: object) -> str:
    if type(strategy_id) is not str:
        raise TypeError("strategy_id must be an exact str")
    normalized = strategy_id.strip()
    if not normalized:
        raise ValueError("strategy_id must be non-empty after stripping")
    return normalized


def _is_rebalance_session(index: pd.DatetimeIndex, frequency: RebalanceFrequency) -> bool:
    if frequency is RebalanceFrequency.DAILY:
        return True
    if len(index) < 2:
        return False
    current = index[-1]
    previous = index[-2]
    if frequency is RebalanceFrequency.WEEKLY:
        return (current.isocalendar().year, current.isocalendar().week) != (
            previous.isocalendar().year,
            previous.isocalendar().week,
        )
    return (current.year, current.month) != (previous.year, previous.month)


def _weighted_momentum(
    close: pd.Series, lookbacks: tuple[int, ...], weights: tuple[float, ...]
) -> float | None:
    if len(close) <= max(lookbacks):
        return None
    try:
        total = fsum(weights)
        if not isfinite(total) or total <= 0:
            return None
        normalized_weights = tuple(weight / total for weight in weights)
        score = fsum(
            normalized_weight
            * (float(close.iloc[-1]) / float(close.iloc[-lookback - 1]) - 1.0)
            for lookback, normalized_weight in zip(
                lookbacks, normalized_weights, strict=True
            )
        )
    except (ArithmeticError, ValueError):
        return None
    return score if isfinite(score) else None


def _is_held(context: StrategyContext, instrument: InstrumentId) -> bool:
    holding = context.holding(instrument)
    return holding is not None and holding.quantity > 0


def _buffered_selection(
    ranked: list[tuple[InstrumentId, float]],
    context: StrategyContext,
    top_n: int,
    buffer: int,
) -> list[tuple[InstrumentId, float]]:
    selected = ranked[:top_n]
    if buffer == 0:
        return selected
    held_in_buffer = [
        candidate
        for candidate in ranked[top_n : top_n + buffer]
        if _is_held(context, candidate[0])
    ]
    for candidate in held_in_buffer:
        replace_at = next(
            (
                index
                for index in range(len(selected) - 1, -1, -1)
                if not _is_held(context, selected[index][0])
            ),
            None,
        )
        if replace_at is None:
            break
        selected[replace_at] = candidate
    return sorted(selected, key=lambda item: (-item[1], str(item[0])))


def _equal_weights(count: int, total: Decimal = Decimal("1")) -> tuple[Decimal, ...]:
    if count <= 0:
        return ()
    total_units = weight_to_units(total, label="strategy target weight")
    allocated, _ = largest_remainder_units(
        {index: 1 for index in range(count)},
        total_units,
        canonical_key=lambda index: f"{index:012d}",
    )
    return tuple(units_to_weight(allocated[index]) for index in range(count))


class CrossSectionalMomentumStrategy:
    strategy_type = "cross_sectional_momentum"
    parameters_type = CrossSectionalMomentumParameters
    minimum_history = 2
    metadata = StrategyMetadata(
        strategy_type=strategy_type,
        version="1.0.0",
        display_name="横截面动量",
        description="按多周期收益综合得分选择排名靠前的股票或 ETF。",
        supported_asset_types=frozenset({AssetType.ETF, AssetType.STOCK}),
        supported_frequencies=frozenset({StrategyFrequency.DAILY}),
        required_fields=frozenset({"close"}),
        minimum_history=minimum_history,
        default_required_history=121,
        parameters_type=parameters_type,
    )

    def __init__(
        self,
        parameters: CrossSectionalMomentumParameters | None = None,
        strategy_id: str = "cross_sectional_momentum",
    ) -> None:
        self.parameters = parameters or CrossSectionalMomentumParameters()
        self.required_history = max(self.parameters.lookbacks) + 1
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
        ranked: list[tuple[InstrumentId, float]] = []
        skipped_insufficient = 0
        for instrument in prepared.instruments:
            history = prepared.histories[instrument]
            if len(history) < self.required_history:
                skipped_insufficient += 1
                continue
            score = _weighted_momentum(
                history["close"], self.parameters.lookbacks, self.parameters.lookback_weights
            )
            if score is not None:
                ranked.append((instrument, score))
        ranked.sort(key=lambda item: (-item[1], str(item[0])))
        selected = _buffered_selection(
            ranked, context, self.parameters.top_n, self.parameters.turnover_buffer
        )
        details = {
            **prepared.details,
            "required_history": self.required_history,
            "selected_count": len(selected),
            "skipped_insufficient_history": skipped_insufficient,
        }
        if not selected:
            if skipped_insufficient == len(prepared.instruments):
                return StrategyDecision.empty(
                    StrategyDecisionStatus.SKIPPED,
                    "INSUFFICIENT_HISTORY",
                    details=details,
                )
            return StrategyDecision.empty(
                StrategyDecisionStatus.CASH, "NO_CANDIDATES_CASH", details=details
            )
        weights = _equal_weights(len(selected))
        intents = tuple(
            TargetIntent(
                strategy_id=self.strategy_id,
                instrument=instrument,
                target_weight=weight,
                score=score,
                confidence=1.0,
                reason_code="CROSS_SECTIONAL_TOP_N",
                valid_until=context.as_of,
            )
            for (instrument, score), weight in zip(selected, weights, strict=True)
        )
        return StrategyDecision.generated(intents, details=details)
