from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from math import isfinite
from types import MappingProxyType

import pandas as pd  # type: ignore[import-untyped]

from compass.analytics.attribution import attribute_sleeves
from compass.backtest.engine import BacktestResult
from compass.backtest.orders import Fill, Order, OrderSide
from compass.domain.market import InstrumentId


def _freeze_values(values: Mapping[str, float], *, label: str) -> Mapping[str, float]:
    checked: dict[str, float] = {}
    for key, value in sorted(values.items()):
        if type(key) is not str or not key:
            raise TypeError(f"{label} keys must be non-empty strings")
        if type(value) is not float or not isfinite(value):
            raise ValueError(f"{label} values must be exact finite floats")
        checked[key] = value
    return MappingProxyType(checked)


@dataclass(frozen=True, slots=True)
class SleeveAccountingPeriod:
    trading_day: date
    returns: Mapping[str, float]
    beginning_weights: Mapping[str, float]
    contributions: Mapping[str, float]
    portfolio_return: float
    total_contribution: float
    residual: float

    def __post_init__(self) -> None:
        if type(self.trading_day) is not date:
            raise TypeError("sleeve accounting day must be an exact date")
        returns = _freeze_values(self.returns, label="sleeve returns")
        weights = _freeze_values(self.beginning_weights, label="sleeve beginning weights")
        contributions = _freeze_values(self.contributions, label="sleeve contributions")
        if set(returns) != set(weights) or set(returns) != set(contributions):
            raise ValueError("sleeve accounting mappings must expose identical sleeves")
        for label in ("portfolio_return", "total_contribution", "residual"):
            value = getattr(self, label)
            if type(value) is not float or not isfinite(value):
                raise ValueError(f"{label} must be an exact finite float")
        if self.total_contribution != sum(contributions.values()):
            raise ValueError("total contribution must equal the sleeve contribution sum")
        if self.residual != self.portfolio_return - self.total_contribution:
            raise ValueError("sleeve residual must reconcile the portfolio return")
        object.__setattr__(self, "returns", returns)
        object.__setattr__(self, "beginning_weights", weights)
        object.__setattr__(self, "contributions", contributions)


@dataclass(frozen=True, slots=True)
class SleeveAccounting:
    sleeves: tuple[str, ...]
    periods: tuple[SleeveAccountingPeriod, ...]
    combined_residual: float

    def __post_init__(self) -> None:
        sleeves = tuple(self.sleeves)
        if sleeves != tuple(sorted(set(sleeves))) or any(
            type(item) is not str or not item for item in sleeves
        ):
            raise ValueError("accounting sleeves must be unique and sorted")
        periods = tuple(self.periods)
        if any(type(item) is not SleeveAccountingPeriod for item in periods):
            raise TypeError("accounting periods must contain exact values")
        days = tuple(item.trading_day for item in periods)
        if days != tuple(sorted(set(days))):
            raise ValueError("accounting periods must be unique and increasing")
        if any(tuple(item.returns) != sleeves for item in periods):
            raise ValueError("every accounting period must expose every sleeve")
        if type(self.combined_residual) is not float or not isfinite(self.combined_residual):
            raise ValueError("combined sleeve residual must be an exact finite float")
        if self.combined_residual != sum(item.residual for item in periods):
            raise ValueError("combined sleeve residual must equal the period residual sum")
        object.__setattr__(self, "sleeves", sleeves)
        object.__setattr__(self, "periods", periods)


def _effective_allocation(
    orders: tuple[Order, ...],
    instrument: InstrumentId,
    as_of: date,
) -> Mapping[str, Decimal]:
    candidates = tuple(
        order
        for order in orders
        if order.instrument == instrument
        and order.filled_quantity > 0
        and sum(order.sleeve_weights.values(), Decimal("0")) > 0
        and order.scheduled_for is not None
        and order.scheduled_for <= as_of
    )
    if not candidates:
        return MappingProxyType({})
    latest = max(candidates, key=lambda item: (item.scheduled_for, item.order_id))
    total = sum(latest.sleeve_weights.values(), Decimal("0"))
    return MappingProxyType(
        {
            sleeve: weight / total
            for sleeve, weight in sorted(latest.sleeve_weights.items())
        }
    )


def calculate_sleeve_accounting(result: BacktestResult) -> SleeveAccounting:
    if type(result) is not BacktestResult:
        raise TypeError("sleeve accounting requires an exact BacktestResult")
    result.verify_integrity()
    sleeves = tuple(
        sorted(
            {
                sleeve
                for order in result.orders
                if order.filled_quantity > 0
                for sleeve in order.sleeve_weights
            }
        )
    )
    index = pd.to_datetime(tuple(item.trading_day for item in result.ledger))
    returns = pd.DataFrame(0.0, index=index, columns=sleeves)
    weights = pd.DataFrame(0.0, index=index, columns=sleeves)
    fills_by_day: dict[date, tuple[Fill, ...]] = {}
    for day in tuple(item.trading_day for item in result.ledger):
        fills_by_day[day] = tuple(fill for fill in result.fills if fill.trading_day == day)

    for period_index in range(1, len(result.ledger)):
        previous = result.ledger[period_index - 1]
        current = result.ledger[period_index]
        current_positions = {item.instrument: item for item in current.positions}
        sleeve_values = {sleeve: 0.0 for sleeve in sleeves}
        sleeve_profit = {sleeve: 0.0 for sleeve in sleeves}
        for position in previous.positions:
            allocation = _effective_allocation(
                result.orders,
                position.instrument,
                previous.trading_day,
            )
            if not allocation:
                continue
            beginning_value = position.market_value
            sell_fills = tuple(
                fill
                for fill in fills_by_day[current.trading_day]
                if fill.instrument == position.instrument and fill.side is OrderSide.SELL
            )
            sold_quantity = min(position.quantity, sum(fill.quantity for fill in sell_fills))
            sale_proceeds = sum(
                (fill.gross_amount for fill in sell_fills),
                start=beginning_value * 0,
            )
            remaining_quantity = position.quantity - sold_quantity
            current_position = current_positions.get(position.instrument)
            marked_remainder = (
                beginning_value * 0
                if remaining_quantity == 0 or current_position is None
                else current_position.mark_price * remaining_quantity
            )
            profit = sale_proceeds + marked_remainder - beginning_value
            for sleeve, fraction in allocation.items():
                sleeve_values[sleeve] += float(beginning_value * fraction)
                sleeve_profit[sleeve] += float(profit * fraction)
        previous_equity = float(previous.equity)
        for sleeve in sleeves:
            value = sleeve_values[sleeve]
            weights.loc[index[period_index], sleeve] = (
                0.0 if previous_equity == 0 else value / previous_equity
            )
            returns.loc[index[period_index], sleeve] = (
                0.0 if value == 0 else sleeve_profit[sleeve] / value
            )

    contribution_frame = attribute_sleeves(returns, weights)
    equity = pd.Series(tuple(float(item.equity) for item in result.ledger), index=index)
    portfolio_returns = equity.pct_change().fillna(0.0)
    periods = tuple(
        SleeveAccountingPeriod(
            trading_day=snapshot.trading_day,
            returns={sleeve: float(returns.loc[day, sleeve]) for sleeve in sleeves},
            beginning_weights={sleeve: float(weights.loc[day, sleeve]) for sleeve in sleeves},
            contributions={
                sleeve: float(contribution_frame.loc[day, sleeve]) for sleeve in sleeves
            },
            portfolio_return=float(portfolio_returns.loc[day]),
            total_contribution=float(contribution_frame.loc[day, "total_contribution"]),
            residual=float(
                portfolio_returns.loc[day]
                - contribution_frame.loc[day, "total_contribution"]
            ),
        )
        for day, snapshot in zip(index, result.ledger, strict=True)
    )
    return SleeveAccounting(
        sleeves=sleeves,
        periods=periods,
        combined_residual=float(sum(item.residual for item in periods)),
    )
