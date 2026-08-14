from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import StrEnum
from math import isfinite
import re
from types import MappingProxyType
from typing import Protocol, runtime_checkable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from compass.backtest.broker import (
    Broker,
    DailyExecutionBar,
    InitialPosition,
)
from compass.backtest.effective_rules import effective_price_limits
from compass.backtest.market_rules import (
    MarketRuleBook,
    MarketRuleProfile,
    OddLotSellPolicy,
)
from compass.backtest.orders import (
    CancellationReason,
    Fill,
    LedgerSnapshot,
    Order,
    OrderSide,
    OrderStatus,
    exact_decimal,
    exact_product,
    round_money,
)
from compass.domain.market import (
    AssetType,
    BarFrame,
    Instrument,
    InstrumentId,
)
from compass.domain.trading import CorporateAction
from compass.domain.weights import WEIGHT_SCALE, units_to_weight, weight_to_units
from compass.portfolio.models import PortfolioTarget
from compass.risk.base import RiskAdjustment, RiskContext, RiskResult, RiskTarget
from compass.risk.engine import RiskEngine
from compass.strategies.base import HoldingSummary, StrategyContext


SHANGHAI = ZoneInfo("Asia/Shanghai")


class ExecutionTiming(StrEnum):
    NEXT_OPEN = "next_open"
    NEXT_CLOSE = "next_close"


def _exact_date(value: object, *, label: str) -> date:
    if type(value) is not date:
        raise TypeError(f"{label} must be an exact date")
    assert isinstance(value, date)
    return value


def _validate_exact_tuple(label: str, values: object, item_type: type[object]) -> None:
    if type(values) is not tuple:
        raise TypeError(f"{label} must be a tuple")
    assert isinstance(values, tuple)
    if any(type(item) is not item_type for item in values):
        raise TypeError(f"{label} contains an invalid value")


def _validate_string_tuple(label: str, values: object) -> None:
    if type(values) is not tuple:
        raise TypeError(f"{label} must be a tuple of non-empty strings")
    assert isinstance(values, tuple)
    if any(type(value) is not str or not value for value in values):
        raise TypeError(f"{label} must be a tuple of non-empty strings")
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"{label} must be unique and sorted")


_WARNING_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,127}\Z")
_WARNING_PARAMETER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def _validate_warning_tuple(values: object) -> None:
    _validate_string_tuple("warnings", values)
    assert isinstance(values, tuple)
    for warning in values:
        parts = warning.split(":")
        if len(parts) not in {1, 2}:
            raise ValueError("warnings must use CODE or CODE:safe-id")
        if _WARNING_CODE.fullmatch(parts[0]) is None:
            raise ValueError("backtest warning code is invalid")
        if len(parts) == 2 and _WARNING_PARAMETER.fullmatch(parts[1]) is None:
            raise ValueError("backtest warning parameter is invalid")
        if len(warning) > 256:
            raise ValueError("backtest warning exceeds the display boundary")


def _ratio_weight(numerator: Decimal, denominator: Decimal) -> Decimal:
    if denominator == 0:
        return Decimal("0")
    numerator_value, numerator_scale = numerator.as_integer_ratio()
    denominator_value, denominator_scale = denominator.as_integer_ratio()
    units = (
        numerator_value * denominator_scale * WEIGHT_SCALE // (numerator_scale * denominator_value)
    )
    return units_to_weight(min(WEIGHT_SCALE, units))


def _directional_trade(
    weight: Decimal,
    equity: Decimal,
    price: Decimal,
    current_quantity: int,
    lot: int,
    odd_lot_policy: OddLotSellPolicy,
) -> tuple[OrderSide, int] | None:
    weight_units = weight_to_units(weight, label="target weight")
    equity_value, equity_scale = equity.as_integer_ratio()
    price_value, price_scale = price.as_integer_ratio()
    target_numerator = weight_units * equity_value * price_scale
    target_denominator = WEIGHT_SCALE * equity_scale * price_value
    current_numerator = current_quantity * target_denominator
    if target_numerator > current_numerator:
        raw_buy = (target_numerator - current_numerator) // target_denominator
        quantity = raw_buy // lot * lot
        return None if quantity == 0 else (OrderSide.BUY, quantity)
    if target_numerator >= current_numerator:
        return None
    if weight_units == 0 and odd_lot_policy in (
        OddLotSellPolicy.ALLOWED,
        OddLotSellPolicy.POSITION_REMAINDER_ONLY,
    ):
        quantity = current_quantity
    else:
        quantity = (current_numerator - target_numerator) // target_denominator
        if odd_lot_policy is not OddLotSellPolicy.ALLOWED:
            quantity = quantity // lot * lot
    return None if quantity == 0 else (OrderSide.SELL, quantity)


def _positive_bar_decimal(value: object, *, label: str) -> Decimal:
    if type(value) is Decimal:
        result = value
    elif isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    elif isinstance(value, (int, float, np.integer, np.floating)):
        result = Decimal(str(value))
    else:
        raise TypeError(f"{label} must be numeric")
    return exact_decimal(result, label=label, positive=True)


def _freeze_instruments(
    values: Mapping[InstrumentId, Instrument],
) -> Mapping[InstrumentId, Instrument]:
    if not isinstance(values, Mapping):
        raise TypeError("instruments must be a mapping")
    checked: dict[InstrumentId, Instrument] = {}
    for key, instrument in values.items():
        if type(key) is not InstrumentId or type(instrument) is not Instrument:
            raise TypeError("instruments must map exact InstrumentId to Instrument values")
        if key != instrument.instrument_id:
            raise ValueError("instrument key must match its identifier")
        checked[key] = instrument
    if not checked:
        raise ValueError("at least one instrument is required")
    return MappingProxyType(dict(sorted(checked.items(), key=lambda item: str(item[0]))))


def _freeze_sleeves(
    values: Mapping[InstrumentId, Mapping[str, Decimal]],
    configured: set[InstrumentId],
) -> Mapping[InstrumentId, Mapping[str, Decimal]]:
    if not isinstance(values, Mapping):
        raise TypeError("sleeve_weights must be a mapping")
    checked: dict[InstrumentId, Mapping[str, Decimal]] = {}
    for symbol, sleeves in values.items():
        if type(symbol) is not InstrumentId or symbol not in configured:
            raise ValueError("sleeve weight instrument must be configured")
        if not isinstance(sleeves, Mapping):
            raise TypeError("per-instrument sleeve weights must be a mapping")
        sleeve_values: dict[str, Decimal] = {}
        for sleeve, weight in sleeves.items():
            if type(sleeve) is not str or not sleeve or sleeve != sleeve.strip():
                raise ValueError("sleeve id must be a stable non-empty string")
            weight_to_units(weight, label="sleeve weight")
            sleeve_values[sleeve] = weight
        checked[symbol] = MappingProxyType(dict(sorted(sleeve_values.items())))
    return MappingProxyType(dict(sorted(checked.items(), key=lambda item: str(item[0]))))


@dataclass(frozen=True, slots=True)
class ForecastTrace:
    """A persisted explanation for one model forecast and portfolio decision."""

    decision_date: date
    strategy_id: str
    instrument: InstrumentId
    action: str
    expected_return: float
    path_positive_ratio: float
    rank: int
    close: float
    trend_value: float
    trend_passed: bool
    target_weight: Decimal
    reason_code: str

    def __post_init__(self) -> None:
        _exact_date(self.decision_date, label="forecast trace decision_date")
        if type(self.strategy_id) is not str or not self.strategy_id.strip():
            raise ValueError("forecast trace strategy_id must be a non-empty string")
        if type(self.instrument) is not InstrumentId:
            raise TypeError("forecast trace instrument must be an exact InstrumentId")
        if type(self.action) is not str or self.action not in {"BUY", "HOLD", "SELL", "CASH"}:
            raise ValueError("forecast trace action is invalid")
        for label, value in (
            ("expected_return", self.expected_return),
            ("path_positive_ratio", self.path_positive_ratio),
            ("close", self.close),
            ("trend_value", self.trend_value),
        ):
            if type(value) is not float or not isfinite(value):
                raise ValueError(f"forecast trace {label} must be a finite float")
        if not 0 <= self.path_positive_ratio <= 1:
            raise ValueError("forecast trace path_positive_ratio must be between zero and one")
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("forecast trace rank must be a positive integer")
        if type(self.trend_passed) is not bool:
            raise TypeError("forecast trace trend_passed must be an exact bool")
        weight_to_units(self.target_weight, label="forecast trace target weight")
        if type(self.reason_code) is not str or _WARNING_CODE.fullmatch(self.reason_code) is None:
            raise ValueError("forecast trace reason_code is invalid")


@dataclass(frozen=True, slots=True)
class DecisionTarget:
    """Narrow deterministic adapter for already-allocated portfolio weights."""

    weights: Mapping[InstrumentId, Decimal]
    sleeve_weights: Mapping[InstrumentId, Mapping[str, Decimal]]
    preserve_unspecified: bool = False
    forecast_traces: tuple[ForecastTrace, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.weights, Mapping):
            raise TypeError("weights must be a mapping")
        checked: dict[InstrumentId, Decimal] = {}
        total_units = 0
        for symbol, weight in self.weights.items():
            if type(symbol) is not InstrumentId:
                raise TypeError("weight keys must be exact InstrumentId values")
            units = weight_to_units(weight, label="target weight")
            checked[symbol] = weight
            total_units += units
        if total_units > WEIGHT_SCALE:
            raise ValueError("decision target weights must sum to at most one")
        configured = set(checked) | set(self.sleeve_weights)
        sleeves = _freeze_sleeves(self.sleeve_weights, configured)
        if not set(sleeves).issubset(checked):
            raise ValueError("sleeve weights must belong to target instruments")
        object.__setattr__(
            self,
            "weights",
            MappingProxyType(dict(sorted(checked.items(), key=lambda item: str(item[0])))),
        )
        object.__setattr__(self, "sleeve_weights", sleeves)
        if type(self.preserve_unspecified) is not bool:
            raise TypeError("preserve_unspecified must be an exact bool")
        traces = tuple(self.forecast_traces)
        if any(type(item) is not ForecastTrace for item in traces):
            raise TypeError("forecast_traces must contain exact ForecastTrace values")
        trace_keys = tuple(
            (item.decision_date, item.strategy_id, str(item.instrument)) for item in traces
        )
        if trace_keys != tuple(sorted(set(trace_keys))):
            raise ValueError("forecast traces must be unique and sorted")
        object.__setattr__(self, "forecast_traces", traces)


@runtime_checkable
class DecisionSource(Protocol):
    def targets(self, context: StrategyContext) -> DecisionTarget | PortfolioTarget:
        """Return targets using only the supplied close-bounded context."""


@dataclass(frozen=True, slots=True, init=False)
class BacktestRequest:
    run_id: str
    sessions: tuple[date, ...]
    instruments: Mapping[InstrumentId, Instrument]
    _bars: Mapping[InstrumentId, pd.DataFrame]
    _strategy_bars: Mapping[InstrumentId, pd.DataFrame]
    _data_warnings: tuple[str, ...]
    initial_cash: Decimal
    initial_positions: tuple[InitialPosition, ...]
    corporate_actions: tuple[CorporateAction, ...]
    decision_source: DecisionSource
    risk_engine: RiskEngine
    rule_book: MarketRuleBook
    execution_timing: ExecutionTiming

    def __init__(
        self,
        *,
        run_id: str,
        sessions: Sequence[date],
        instruments: Mapping[InstrumentId, Instrument],
        bars: Mapping[InstrumentId, pd.DataFrame],
        initial_cash: Decimal,
        initial_positions: Sequence[InitialPosition],
        corporate_actions: Sequence[CorporateAction],
        decision_source: DecisionSource,
        risk_engine: RiskEngine,
        rule_book: MarketRuleBook,
        execution_timing: ExecutionTiming = ExecutionTiming.NEXT_OPEN,
    ) -> None:
        if type(run_id) is not str or not run_id or run_id != run_id.strip():
            raise ValueError("run_id must be a stable non-empty string")
        ordered_sessions = tuple(sessions)
        if not ordered_sessions:
            raise ValueError("at least one session is required")
        for session in ordered_sessions:
            _exact_date(session, label="session")
        if tuple(sorted(set(ordered_sessions))) != ordered_sessions:
            raise ValueError("sessions must be unique and increasing")
        checked_instruments = _freeze_instruments(instruments)
        if not isinstance(bars, Mapping) or set(bars) != set(checked_instruments):
            raise ValueError("bars must exist for exactly the configured instruments")
        checked_bars: dict[InstrumentId, pd.DataFrame] = {}
        strategy_bars: dict[InstrumentId, pd.DataFrame] = {}
        data_warnings: set[str] = set()
        all_dates: set[date] = set()
        for symbol in checked_instruments:
            frame = bars[symbol]
            if not isinstance(frame, pd.DataFrame):
                raise TypeError("bars must contain DataFrame values")
            validated = BarFrame.validate(frame.copy(deep=True))
            checked_bars[symbol] = validated
            if "adjust_flag" in validated:
                for flag in validated["adjust_flag"]:
                    if type(flag) is not str or flag != "3":
                        raise ValueError(
                            "execution bars must use raw prices with adjust_flag code '3'"
                        )
            comparable = validated.copy(deep=True)
            if "adjust_factor" in validated:
                factors = tuple(
                    _positive_bar_decimal(value, label="adjust_factor")
                    for value in validated["adjust_factor"]
                )
                for column in ("open", "high", "low", "close"):
                    comparable[column] = pd.Series(
                        [
                            exact_decimal(
                                exact_product(_positive_bar_decimal(value, label=column), factor),
                                label=f"comparable {column}",
                                positive=True,
                            )
                            for value, factor in zip(validated[column], factors, strict=True)
                        ],
                        index=validated.index,
                        dtype=object,
                    )
                comparable = BarFrame.validate(comparable)
            else:
                data_warnings.add(f"COMPARABLE_PRICE_FACTOR_MISSING:{symbol}")
            strategy_bars[symbol] = comparable
            all_dates.update(timestamp.date() for timestamp in validated.index)
        if (
            not all_dates
            or min(ordered_sessions) < min(all_dates)
            or max(ordered_sessions) > max(all_dates)
        ):
            raise ValueError("requested sessions are outside available bars")
        cash = exact_decimal(initial_cash, label="initial_cash")
        if cash != round_money(cash):
            raise ValueError("initial_cash must be rounded to cents")
        positions = tuple(initial_positions)
        if any(type(position) is not InitialPosition for position in positions):
            raise TypeError("initial_positions must contain exact InitialPosition values")
        if len({position.instrument for position in positions}) != len(positions):
            raise ValueError("initial_positions must be unique")
        if any(position.instrument not in checked_instruments for position in positions):
            raise ValueError("initial position instrument is not configured")
        actions = tuple(corporate_actions)
        if any(type(action) is not CorporateAction for action in actions):
            raise TypeError("corporate_actions must contain exact CorporateAction values")
        action_keys = [(action.instrument, action.ex_date) for action in actions]
        if len(set(action_keys)) != len(action_keys):
            raise ValueError("corporate actions must be unique by instrument and ex-date")
        if any(
            action.instrument not in checked_instruments or action.ex_date not in ordered_sessions
            for action in actions
        ):
            raise ValueError("corporate actions must belong to configured sessions/instruments")
        if not isinstance(decision_source, DecisionSource):
            raise TypeError("decision_source must implement targets(context)")
        if type(risk_engine) is not RiskEngine:
            raise TypeError("risk_engine must be an exact RiskEngine")
        if type(rule_book) is not MarketRuleBook:
            raise TypeError("rule_book must be an exact MarketRuleBook")
        if type(execution_timing) is not ExecutionTiming:
            raise TypeError("execution_timing must be an exact ExecutionTiming")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "sessions", ordered_sessions)
        object.__setattr__(self, "instruments", checked_instruments)
        object.__setattr__(self, "_bars", MappingProxyType(checked_bars))
        object.__setattr__(self, "_strategy_bars", MappingProxyType(strategy_bars))
        object.__setattr__(self, "_data_warnings", tuple(sorted(data_warnings)))
        object.__setattr__(self, "initial_cash", cash)
        object.__setattr__(self, "initial_positions", positions)
        object.__setattr__(self, "corporate_actions", actions)
        object.__setattr__(self, "decision_source", decision_source)
        object.__setattr__(self, "risk_engine", risk_engine)
        object.__setattr__(self, "rule_book", rule_book)
        object.__setattr__(self, "execution_timing", execution_timing)

    @property
    def bars(self) -> Mapping[InstrumentId, pd.DataFrame]:
        return MappingProxyType(
            {symbol: frame.copy(deep=True) for symbol, frame in self._bars.items()}
        )


@dataclass(frozen=True, slots=True)
class RiskTrace:
    decision_date: date
    instrument: InstrumentId
    result: RiskResult

    def __post_init__(self) -> None:
        _exact_date(self.decision_date, label="decision_date")
        if type(self.instrument) is not InstrumentId:
            raise TypeError("risk trace instrument must be an exact InstrumentId")
        if type(self.result) is not RiskResult:
            raise TypeError("risk trace result must be an exact RiskResult")

    @property
    def final_weight(self) -> Decimal:
        return self.result.final_weight

    @property
    def adjustments(self) -> tuple[RiskAdjustment, ...]:
        return self.result.adjustments


@dataclass(frozen=True, slots=True)
class BacktestResult:
    run_id: str
    orders: tuple[Order, ...]
    fills: tuple[Fill, ...]
    ledger: tuple[LedgerSnapshot, ...]
    risk_traces: tuple[RiskTrace, ...]
    used_profile_ids: tuple[str, ...]
    warnings: tuple[str, ...]
    forecast_traces: tuple[ForecastTrace, ...] = ()

    def __post_init__(self) -> None:
        if type(self.run_id) is not str or not self.run_id or self.run_id != self.run_id.strip():
            raise ValueError("run_id must be a stable non-empty string")
        expected: tuple[tuple[str, object, type[object]], ...] = (
            ("orders", self.orders, Order),
            ("fills", self.fills, Fill),
            ("ledger", self.ledger, LedgerSnapshot),
            ("risk_traces", self.risk_traces, RiskTrace),
            ("forecast_traces", self.forecast_traces, ForecastTrace),
        )
        for label, values, item_type in expected:
            _validate_exact_tuple(label, values, item_type)
        if not self.ledger:
            raise ValueError("ledger must not be empty")
        ledger_days = tuple(snapshot.trading_day for snapshot in self.ledger)
        if tuple(sorted(set(ledger_days))) != ledger_days:
            raise ValueError("ledger sessions must be unique and increasing")
        order_ids = tuple(order.order_id for order in self.orders)
        if len(set(order_ids)) != len(order_ids):
            raise ValueError("result orders must have unique ids")
        if any(order.status is OrderStatus.PENDING for order in self.orders):
            raise ValueError("backtest results cannot contain pending orders")
        fill_ids = tuple(fill.fill_id for fill in self.fills)
        if len(set(fill_ids)) != len(fill_ids):
            raise ValueError("result fills must have unique ids")
        _validate_string_tuple("used_profile_ids", self.used_profile_ids)
        _validate_warning_tuple(self.warnings)
        forecast_keys = tuple(
            (item.decision_date, item.strategy_id, str(item.instrument))
            for item in self.forecast_traces
        )
        if forecast_keys != tuple(sorted(set(forecast_keys))):
            raise ValueError("forecast traces must be unique and sorted")
        orders_by_id = {order.order_id: order for order in self.orders}
        fills_by_order: dict[str, list[Fill]] = {}
        ledger_days_set = set(ledger_days)
        previous_fill_day: date | None = None
        fill_order_keys: list[tuple[date, int, str, str]] = []
        for fill in self.fills:
            order = orders_by_id.get(fill.order_id)
            if order is None:
                raise ValueError(f"ghost fill references unknown order: {fill.order_id}")
            if fill.instrument != order.instrument:
                raise ValueError("fill instrument does not match its order")
            if fill.side is not order.side:
                raise ValueError("fill side does not match its order")
            if order.scheduled_for is None or fill.trading_day != order.scheduled_for:
                raise ValueError("fill date does not match the order execution date")
            if fill.trading_day not in ledger_days_set:
                raise ValueError("fill date has no ledger session")
            if fill.profile_id not in self.used_profile_ids:
                raise ValueError("fill profile is missing from used profile provenance")
            if previous_fill_day is not None and fill.trading_day < previous_fill_day:
                raise ValueError("fills must be ordered by trading date")
            previous_fill_day = fill.trading_day
            fill_order_keys.append(
                (
                    fill.trading_day,
                    0 if fill.side is OrderSide.SELL else 1,
                    str(fill.instrument),
                    fill.order_id,
                )
            )
            fills_by_order.setdefault(fill.order_id, []).append(fill)
        if fill_order_keys != sorted(fill_order_keys):
            raise ValueError("fills must follow stable sell-before-buy ordering")
        risk_codes_by_decision: dict[tuple[date, InstrumentId], tuple[str, ...]] = {}
        for trace in self.risk_traces:
            key = (trace.decision_date, trace.instrument)
            if key in risk_codes_by_decision:
                raise ValueError("duplicate risk trace for decision date and instrument")
            risk_codes_by_decision[key] = tuple(adjustment.code for adjustment in trace.adjustments)
        for order in self.orders:
            filled = sum(fill.quantity for fill in fills_by_order.get(order.order_id, ()))
            if filled != order.filled_quantity:
                raise ValueError("fill quantity does not match its order outcome")
            expected_codes = risk_codes_by_decision.get((order.created_on, order.instrument), ())
            if order.risk_codes != expected_codes:
                raise ValueError("order risk codes must exactly match its decision trace")

    def verify_integrity(self) -> None:
        self.__post_init__()
        for order in self.orders:
            order.__post_init__()
        for fill in self.fills:
            fill.__post_init__()
        for snapshot in self.ledger:
            for position in snapshot.positions:
                position.__post_init__()
            snapshot.__post_init__()
        for risk_trace in self.risk_traces:
            risk_trace.__post_init__()
        for forecast_trace in self.forecast_traces:
            forecast_trace.__post_init__()


def _cell_decimal(value: object, *, label: str, optional: bool = False) -> Decimal | None:
    if optional and pd.isna(value):
        return None
    if type(value) is Decimal:
        result = value
    elif isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    elif isinstance(value, (int, float)):
        result = Decimal(str(value))
    else:
        raise TypeError(f"{label} must be numeric")
    return exact_decimal(result, label=label, positive=True)


def _execution_bar(
    frame: pd.DataFrame,
    day: date,
    instrument: Instrument,
    profile: MarketRuleProfile,
    *,
    has_corporate_action: bool = False,
    execution_timing: ExecutionTiming = ExecutionTiming.NEXT_OPEN,
) -> DailyExecutionBar | None:
    timestamp = pd.Timestamp(day)
    if timestamp not in frame.index:
        return None
    row = frame.loc[timestamp]
    volume_value = row["volume"]
    if isinstance(volume_value, bool) or not isinstance(
        volume_value, (int, float, np.integer, np.floating)
    ):
        raise ValueError("bar volume must be numeric")
    rounded_volume = round(float(volume_value))
    tolerance = max(1e-6, abs(float(volume_value)) * 1e-12)
    if not np.isfinite(volume_value) or abs(float(volume_value) - rounded_volume) > tolerance:
        raise ValueError(
            "bar volume must be an exact integer "
            f"instrument={instrument.instrument_id} day={day.isoformat()} "
            f"value={volume_value!r}"
        )
    suspended_value = row.get("suspended", False)
    if pd.isna(suspended_value):
        # Missing optional suspension metadata is modeled as not suspended; callers
        # can distinguish this limitation through their data-quality provenance.
        suspended = False
    elif type(suspended_value) is bool or isinstance(suspended_value, np.bool_):
        suspended = bool(suspended_value)
    else:
        raise TypeError("suspended must be an exact bool or nullable missing value")
    limits = effective_price_limits(
        frame,
        timestamp,
        row,
        day,
        instrument,
        profile,
        has_corporate_action=has_corporate_action,
    )
    execution_column = "open" if execution_timing is ExecutionTiming.NEXT_OPEN else "close"
    return DailyExecutionBar(
        open=_cell_decimal(row[execution_column], label=execution_column),  # type: ignore[arg-type]
        close=_cell_decimal(row["close"], label="close"),  # type: ignore[arg-type]
        volume=rounded_volume,
        suspended=suspended,
        limit_up=limits.limit_up,
        limit_down=limits.limit_down,
        price_limit_state_known=limits.state_known,
    )


def _source_target(
    target: DecisionTarget | PortfolioTarget,
    instruments: Mapping[InstrumentId, Instrument],
) -> DecisionTarget:
    if type(target) is DecisionTarget:
        if not set(target.weights).issubset(instruments):
            raise ValueError("decision target contains an unconfigured instrument")
        return target
    if type(target) is not PortfolioTarget:
        raise TypeError("decision source must return DecisionTarget or PortfolioTarget")
    weights: dict[InstrumentId, Decimal] = {}
    sleeves: dict[InstrumentId, dict[str, Decimal]] = {}
    for symbol_text, weight in target.weights.items():
        symbol = InstrumentId.parse(symbol_text)
        if symbol not in instruments:
            raise ValueError("portfolio target contains an unconfigured instrument")
        weights[symbol] = weight
        sleeves[symbol] = {
            strategy_id: sleeve.final_weights[symbol_text]
            for strategy_id, sleeve in target.sleeves.items()
            if symbol_text in sleeve.final_weights and sleeve.final_weights[symbol_text] > 0
        }
    return DecisionTarget(weights, sleeves)


class BacktestEngine:
    def _context(
        self,
        request: BacktestRequest,
        day: date,
        snapshot: LedgerSnapshot,
        holding_since: Mapping[InstrumentId, date],
    ) -> StrategyContext:
        holdings = {
            position.instrument: HoldingSummary(
                instrument=position.instrument,
                quantity=position.quantity,
                available_quantity=position.available_quantity,
                average_cost=position.average_cost,
                mark_price=position.mark_price,
                holding_since=holding_since[position.instrument],
            )
            for position in snapshot.positions
        }
        return StrategyContext(
            as_of=day,
            bars=request._strategy_bars,
            instruments=tuple(request.instruments),
            account_equity=snapshot.equity,
            cash=snapshot.cash,
            holdings=holdings,
            asset_types={
                symbol: instrument.asset_type for symbol, instrument in request.instruments.items()
            },
        )

    def _orders_at_close(
        self,
        request: BacktestRequest,
        day: date,
        next_day: date | None,
        snapshot: LedgerSnapshot,
        decision: DecisionTarget,
    ) -> tuple[
        tuple[Order, ...],
        tuple[RiskTrace, ...],
        tuple[str, ...],
        tuple[str, ...],
    ]:
        positions = {position.instrument: position for position in snapshot.positions}
        target_symbols = set(decision.weights) | set(positions)
        orders: list[Order] = []
        traces: list[RiskTrace] = []
        profile_ids: set[str] = set()
        warnings: set[str] = set()
        equity = snapshot.equity
        as_of = datetime.combine(day, time(15, 0), tzinfo=SHANGHAI)
        total_target_units = sum(
            weight_to_units(weight, label="target weight") for weight in decision.weights.values()
        )
        approved_strategy_units = 0
        approved_buy_turnover_units = 0
        # Risk is replayed in canonical instrument order so aggregate limits do
        # not depend on decision-source mapping insertion order.
        for symbol in sorted(target_symbols, key=str):
            instrument = request.instruments[symbol]
            frame = request._bars[symbol]
            decision_profile = request.rule_book.profile_for(day, instrument)
            bar = _execution_bar(frame, day, instrument, decision_profile)
            position = positions.get(symbol)
            current_quantity = 0 if position is None else position.quantity
            available_quantity = 0 if position is None else position.available_quantity
            current_value = Decimal("0") if position is None else position.market_value
            current_weight = _ratio_weight(current_value, equity)
            requested_weight = decision.weights.get(
                symbol,
                current_weight if decision.preserve_unspecified else Decimal("0"),
            )
            execution_profile = request.rule_book.profile_for(
                day if next_day is None else next_day, instrument
            )
            for profile in (decision_profile, execution_profile):
                profile_ids.add(profile.profile_id)
                if not profile.fee_profile_confirmed:
                    warnings.add(f"FEE_PROFILE_UNCONFIRMED:{profile.profile_id}")
            reference_price = (
                position.mark_price
                if bar is None and position is not None
                else None
                if bar is None
                else bar.close
            )
            preliminary_quantity = 0
            if reference_price is not None and equity > 0:
                preliminary = _directional_trade(
                    requested_weight,
                    equity,
                    reference_price,
                    current_quantity,
                    execution_profile.buy_lot_size,
                    execution_profile.odd_lot_sell_policy,
                )
                preliminary_quantity = 0 if preliminary is None else preliminary[1]
            data_as_of: datetime | None = None
            if not frame.empty:
                visible = frame.loc[frame.index <= pd.Timestamp(day)]
                if not visible.empty:
                    data_as_of = datetime.combine(
                        visible.index[-1].date(), time(15, 0), tzinfo=SHANGHAI
                    )
            other_units = max(
                0,
                total_target_units - weight_to_units(requested_weight, label="target weight"),
            )
            other_stock_units = sum(
                weight_to_units(weight, label="stock target weight")
                for other_symbol, weight in decision.weights.items()
                if other_symbol != symbol
                and request.instruments[other_symbol].asset_type is AssetType.STOCK
            )
            risk_context = RiskContext(
                as_of=as_of,
                data_as_of=data_as_of,
                data_valid=bar is not None,
                tradable=bar is not None and not bar.suspended,
                other_invested_weight=units_to_weight(min(WEIGHT_SCALE, other_units)),
                other_stock_weight=units_to_weight(min(WEIGHT_SCALE, other_stock_units)),
                strategy_other_weight=units_to_weight(approved_strategy_units),
                turnover_used_weight=units_to_weight(approved_buy_turnover_units),
                observed_volume=None if bar is None else bar.volume,
                expected_order_quantity=preliminary_quantity,
                available_cash=snapshot.cash,
                account_equity=equity,
                available_sell_quantity=available_quantity,
                lot_size=execution_profile.buy_lot_size,
                allow_odd_lot_sell=(
                    execution_profile.odd_lot_sell_policy is not OddLotSellPolicy.FORBIDDEN
                ),
                reference_price=reference_price,
                minimum_trade_amount=Decimal("0"),
                price_constraint_ok=True,
            )
            result = request.risk_engine.apply(
                risk_context,
                RiskTarget(
                    instrument=instrument,
                    requested_weight=requested_weight,
                    current_weight=current_weight,
                    strategy_id="portfolio",
                ),
            )
            traces.append(RiskTrace(day, symbol, result))
            trade = (
                None
                if result.blocked or reference_price is None or equity == 0
                else _directional_trade(
                    result.final_weight,
                    equity,
                    reference_price,
                    current_quantity,
                    execution_profile.buy_lot_size,
                    execution_profile.odd_lot_sell_policy,
                )
            )
            current_units = weight_to_units(current_weight, label="current weight")
            approved_units = current_units
            if trade is not None and next_day is not None:
                assert reference_price is not None
                planned_side, planned_quantity = trade
                target_quantity = current_quantity + (
                    planned_quantity
                    if planned_side is OrderSide.BUY
                    else -planned_quantity
                )
                planned_value = round_money(exact_product(reference_price, target_quantity))
                approved_units = weight_to_units(
                    _ratio_weight(planned_value, equity),
                    label="approved strategy weight",
                )
            approved_strategy_units = min(
                WEIGHT_SCALE, approved_strategy_units + approved_units
            )
            approved_buy_turnover_units = min(
                WEIGHT_SCALE,
                approved_buy_turnover_units + max(0, approved_units - current_units),
            )
            if trade is None:
                continue
            side, quantity = trade
            order_id = f"{request.run_id}:{day.isoformat()}:{symbol}:{side.value}"
            risk_codes = tuple(item.code for item in result.adjustments)
            if next_day is None:
                order = Order(
                    order_id=order_id,
                    instrument=symbol,
                    side=side,
                    quantity=quantity,
                    created_on=day,
                    scheduled_for=None,
                    sleeve_weights=decision.sleeve_weights.get(symbol, {}),
                    risk_codes=risk_codes,
                    status=OrderStatus.CANCELLED,
                    cancellation_reason=CancellationReason.NO_NEXT_SESSION,
                )
            else:
                order = Order(
                    order_id=order_id,
                    instrument=symbol,
                    side=side,
                    quantity=quantity,
                    created_on=day,
                    scheduled_for=next_day,
                    sleeve_weights=decision.sleeve_weights.get(symbol, {}),
                    risk_codes=risk_codes,
                )
            orders.append(order)
        return (
            tuple(orders),
            tuple(traces),
            tuple(sorted(profile_ids)),
            tuple(sorted(warnings)),
        )

    def run(self, request: BacktestRequest) -> BacktestResult:
        if type(request) is not BacktestRequest:
            raise TypeError("request must be an exact BacktestRequest")
        broker = Broker(
            request.initial_cash,
            request.instruments,
            request.rule_book,
            request.initial_positions,
        )
        all_orders: dict[str, Order] = {}
        order_sequence: list[str] = []
        pending: tuple[Order, ...] = ()
        fills: list[Fill] = []
        ledger: list[LedgerSnapshot] = []
        traces: list[RiskTrace] = []
        forecast_traces: list[ForecastTrace] = []
        used_profiles: set[str] = set()
        warnings: set[str] = set(request._data_warnings)
        actions_by_day: dict[date, list[CorporateAction]] = {}
        for action in request.corporate_actions:
            actions_by_day.setdefault(action.ex_date, []).append(action)
        holding_since = {
            position.instrument: request.sessions[0]
            for position in request.initial_positions
        }
        for index, day in enumerate(request.sessions):
            day_bars = {
                symbol: bar
                for symbol, frame in request._bars.items()
                if (
                    bar := _execution_bar(
                        frame,
                        day,
                        request.instruments[symbol],
                        request.rule_book.profile_for(
                            day, request.instruments[symbol]
                        ),
                        has_corporate_action=any(
                            action.instrument == symbol
                            for action in actions_by_day.get(day, ())
                        ),
                        execution_timing=request.execution_timing,
                    )
                )
                is not None
            }
            execution = broker.execute_session(
                day,
                day_bars,
                pending,
                actions=tuple(actions_by_day.get(day, ())),
            )
            for order in execution.orders:
                all_orders[order.order_id] = order
            fills.extend(execution.fills)
            used_profiles.update(execution.used_profile_ids)
            warnings.update(execution.warnings)
            snapshot = execution.snapshot
            ledger.append(snapshot)
            held_symbols = {position.instrument for position in snapshot.positions}
            holding_since = {
                symbol: holding_since.get(symbol, day)
                for symbol in held_symbols
            }
            context = self._context(request, day, snapshot, holding_since)
            target = _source_target(request.decision_source.targets(context), request.instruments)
            forecast_traces.extend(target.forecast_traces)
            next_day = request.sessions[index + 1] if index + 1 < len(request.sessions) else None
            created, decision_traces, decision_profiles, decision_warnings = self._orders_at_close(
                request, day, next_day, snapshot, target
            )
            traces.extend(decision_traces)
            used_profiles.update(decision_profiles)
            warnings.update(decision_warnings)
            for order in created:
                all_orders[order.order_id] = order
                order_sequence.append(order.order_id)
            pending = tuple(order for order in created if order.status is OrderStatus.PENDING)
        return BacktestResult(
            run_id=request.run_id,
            orders=tuple(all_orders[order_id] for order_id in order_sequence),
            fills=tuple(fills),
            ledger=tuple(ledger),
            risk_traces=tuple(traces),
            used_profile_ids=tuple(sorted(used_profiles)),
            warnings=tuple(sorted(warnings)),
            forecast_traces=tuple(forecast_traces),
        )
