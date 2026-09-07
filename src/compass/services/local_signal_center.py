from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal, DecimalException
from numbers import Number
from types import MappingProxyType

from compass.data.base import default_instrument_type
from compass.domain.market import AssetType, InstrumentId
from compass.domain.trading import AccountSnapshot, Position
from compass.services.export_service import DecisionExportRecord
from compass.services.instrument_names import common_instrument_name
from compass.services.local_crud_gateways import IdFactory, LocalStrategyGateway
from compass.services.local_decision_gateway import (
    LocalDecisionGateway,
    SelectedDecisionStrategy,
)
from compass.storage.account_repository import (
    SHANGHAI,
    AccountRepository,
    StoredAccountSnapshot,
)
from compass.storage.dataset_bundle_repository import DatasetBundleRepository
from compass.storage.signal_account_repository import (
    ShadowExecutionTiming,
    SignalAccountProfile,
    SignalAccountRepository,
    SignalAccountStrategySetting,
    SignalShadowSimulationSetting,
)
from compass.storage.signal_execution_repository import (
    SignalExecutionFill,
    SignalExecutionRecord,
    SignalExecutionRepository,
    SignalExecutionStatus,
)


@dataclass(frozen=True, slots=True)
class SignalInstrumentChoice:
    instrument: InstrumentId
    name: str
    asset_type: AssetType
    data_day: date
    close: Decimal


@dataclass(frozen=True, slots=True)
class SignalStrategyChoice:
    instance_id: str
    name: str
    strategy_type: str


@dataclass(frozen=True, slots=True)
class AccountPositionInput:
    instrument: str
    quantity: object
    available_quantity: object
    average_cost: object


@dataclass(frozen=True, slots=True)
class SignalExecutionFillInput:
    instrument: str
    quantity_delta: object
    execution_price: object


@dataclass(frozen=True, slots=True)
class SignalDecisionFreshness:
    stale: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SignalComparisonPoint:
    day: date
    adopted_equity: Decimal
    ignored_equity: Decimal


@dataclass(frozen=True, slots=True)
class SignalDecisionComparison:
    decision_id: str
    points: tuple[SignalComparisonPoint, ...]
    adopted_return: Decimal
    ignored_return: Decimal
    relative_impact: Decimal


@dataclass(frozen=True, slots=True)
class SignalAccountValuationPoint:
    day: date
    cash: Decimal
    market_value: Decimal
    equity: Decimal
    source_snapshot_row_id: int
    position_values: Mapping[InstrumentId, Decimal]

    def __post_init__(self) -> None:
        if type(self.day) is not date:
            raise TypeError("valuation day must be an exact date")
        for label, value in (
            ("cash", self.cash),
            ("market value", self.market_value),
            ("equity", self.equity),
        ):
            if type(value) is not Decimal or not value.is_finite() or value < 0:
                raise ValueError(f"valuation {label} must be finite and non-negative")
            if value != value.quantize(Decimal("0.01")):
                raise ValueError(f"valuation {label} must use cents")
        if self.cash + self.market_value != self.equity:
            raise ValueError("valuation components must add up to equity")
        if (
            isinstance(self.source_snapshot_row_id, bool)
            or not isinstance(self.source_snapshot_row_id, int)
            or self.source_snapshot_row_id <= 0
        ):
            raise ValueError("valuation source snapshot id must be positive")
        values = dict(self.position_values)
        if any(
            type(instrument) is not InstrumentId
            or type(value) is not Decimal
            or not value.is_finite()
            or value < 0
            or value != value.quantize(Decimal("0.01"))
            for instrument, value in values.items()
        ):
            raise ValueError("valuation position values are invalid")
        if sum(values.values(), Decimal("0")) != self.market_value:
            raise ValueError("valuation position values must add up to market value")
        object.__setattr__(
            self,
            "position_values",
            MappingProxyType(dict(sorted(values.items(), key=lambda item: str(item[0])))),
        )


@dataclass(frozen=True, slots=True)
class SignalShadowPoint:
    day: date
    actual_equity: Decimal
    shadow_equity: Decimal
    hold_equity: Decimal

    def __post_init__(self) -> None:
        if type(self.day) is not date:
            raise TypeError("shadow point day must be exact")
        for value in (self.actual_equity, self.shadow_equity, self.hold_equity):
            if (
                type(value) is not Decimal
                or not value.is_finite()
                or value < 0
                or value != value.quantize(Decimal("0.01"))
            ):
                raise ValueError("shadow point equity must be non-negative cents")


@dataclass(frozen=True, slots=True)
class SignalShadowExecution:
    decision_id: str
    signal_day: date
    execution_day: date | None
    status: str
    trade_count: int
    costs: Decimal

    def __post_init__(self) -> None:
        if type(self.decision_id) is not str or not self.decision_id:
            raise ValueError("shadow decision id must be non-empty")
        if type(self.signal_day) is not date:
            raise TypeError("shadow signal day must be exact")
        if self.execution_day is not None and type(self.execution_day) is not date:
            raise TypeError("shadow execution day must be exact or None")
        if self.status not in {"executed", "no_trade", "pending", "unfilled"}:
            raise ValueError("shadow execution status is invalid")
        if type(self.trade_count) is not int or self.trade_count < 0:
            raise ValueError("shadow trade count must be non-negative")
        if (
            type(self.costs) is not Decimal
            or not self.costs.is_finite()
            or self.costs < 0
            or self.costs != self.costs.quantize(Decimal("0.01"))
        ):
            raise ValueError("shadow costs must be non-negative cents")


@dataclass(frozen=True, slots=True)
class SignalShadowSimulation:
    setting: SignalShadowSimulationSetting
    start_day: date
    points: tuple[SignalShadowPoint, ...]
    executions: tuple[SignalShadowExecution, ...]
    unavailable_instruments: tuple[InstrumentId, ...]

    def __post_init__(self) -> None:
        if type(self.setting) is not SignalShadowSimulationSetting:
            raise TypeError("shadow setting must be exact")
        if type(self.start_day) is not date:
            raise TypeError("shadow start day must be exact")
        points = tuple(self.points)
        executions = tuple(self.executions)
        unavailable = tuple(self.unavailable_instruments)
        if any(type(item) is not SignalShadowPoint for item in points):
            raise TypeError("shadow points must be exact")
        if tuple(item.day for item in points) != tuple(
            sorted({item.day for item in points})
        ):
            raise ValueError("shadow point days must be unique and sorted")
        if any(type(item) is not SignalShadowExecution for item in executions):
            raise TypeError("shadow executions must be exact")
        if unavailable != tuple(sorted(set(unavailable), key=str)):
            raise ValueError("unavailable shadow instruments must be unique and sorted")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "executions", executions)
        object.__setattr__(self, "unavailable_instruments", unavailable)


def _decimal(value: object, *, label: str, cents: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, Decimal, Number)):
        raise ValueError(f"{label} must be numeric")
    try:
        parsed = Decimal(str(value))
    except DecimalException:
        raise ValueError(f"{label} must be numeric") from None
    if not parsed.is_finite() or parsed < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    if cents and parsed != parsed.quantize(Decimal("0.01")):
        raise ValueError(f"{label} must use cents")
    return parsed


def _quantity(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    if type(value) is int:
        parsed = value
    elif type(value) is float and value.is_integer():
        parsed = int(value)
    elif type(value) is str and value.isascii() and value.isdigit():
        parsed = int(value)
    else:
        raise ValueError(f"{label} must be a non-negative integer")
    if parsed < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return parsed


def _signed_quantity(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-zero integer")
    if type(value) is int:
        parsed = value
    elif type(value) is float and value.is_integer():
        parsed = int(value)
    elif type(value) is str:
        try:
            parsed = int(value)
        except ValueError:
            raise ValueError(f"{label} must be a non-zero integer") from None
    else:
        raise ValueError(f"{label} must be a non-zero integer")
    if parsed == 0:
        raise ValueError(f"{label} must be a non-zero integer")
    return parsed


class LocalSignalCenter:
    """Compose local account snapshots, saved strategies and close decisions."""

    def __init__(
        self,
        *,
        accounts: AccountRepository,
        bundles: DatasetBundleRepository,
        strategies: LocalStrategyGateway,
        decisions: LocalDecisionGateway,
        id_factory: IdFactory,
        account_profiles: SignalAccountRepository,
        executions: SignalExecutionRepository,
        account_factory: Callable[[str], AccountRepository],
    ) -> None:
        self._accounts = accounts
        self._bundles = bundles
        self._strategies = strategies
        self._decisions = decisions
        self._id_factory = id_factory
        self._account_profiles = account_profiles
        self._executions = executions
        self._account_factory = account_factory
        self._bound_account_id: str | None = None
        self._expected_snapshot_id: int | None = -1

    def for_account(self, account_id: str | None = None) -> LocalSignalCenter:
        """Bind a page to an account without changing any other page's selection."""
        scoped = copy(self)
        scoped._bound_account_id = account_id or self.active_account_profile().account_id
        scoped.active_account_profile()  # Reject unknown/deleted accounts, never fall back.
        latest = scoped.latest_account()
        scoped._expected_snapshot_id = None if latest is None else latest.row_id
        return scoped

    def account_profiles(self) -> tuple[SignalAccountProfile, ...]:
        return self._account_profiles.state().profiles

    def active_account_profile(self) -> SignalAccountProfile:
        state = self._account_profiles.state()
        if self._bound_account_id is None:
            return state.active
        for profile in state.profiles:
            if profile.account_id == self._bound_account_id:
                return profile
        raise LookupError("账户已删除，请重新选择账户。")

    def select_account(self, account_id: str) -> SignalAccountProfile:
        return self._account_profiles.select(account_id).active

    def create_account(
        self,
        name: str,
        holdings_account_id: str | None = None,
    ) -> SignalAccountProfile:
        return self._account_profiles.create(
            self._id_factory("account"),
            name,
            holdings_account_id=holdings_account_id,
        )

    def delete_account(self, account_id: str) -> SignalAccountProfile:
        return self._account_profiles.delete(account_id).active

    def save_strategy_configuration(
        self,
        selections: Sequence[SelectedDecisionStrategy],
        *,
        cash_reserve: Decimal,
        minimum_trade_amount: Decimal,
    ) -> SignalAccountProfile:
        return self._account_profiles.save_configuration(
            self.active_account_profile().account_id,
            tuple(
                SignalAccountStrategySetting(item.strategy_instance_id, item.budget)
                for item in selections
            ),
            cash_reserve=cash_reserve,
            minimum_trade_amount=minimum_trade_amount,
        )

    def _active_accounts(self) -> AccountRepository:
        profile = self.active_account_profile()
        assert profile.holdings_account_id is not None
        if profile.holdings_account_id == self._accounts.account_id:
            return self._accounts
        repository = self._account_factory(profile.holdings_account_id)
        if type(repository) is not AccountRepository:
            raise TypeError("account factory must return an exact AccountRepository")
        return repository

    def instruments(self) -> tuple[SignalInstrumentChoice, ...]:
        bundle = self._bundles.latest()
        if bundle is None:
            return ()
        references = self._bundles.references_by_instrument(bundle)
        choices: list[SignalInstrumentChoice] = []
        for instrument in bundle.instruments:
            asset_type = default_instrument_type(instrument)
            if asset_type not in {AssetType.ETF, AssetType.STOCK}:
                continue
            reference = references[instrument]
            manifest = self._bundles.load_manifest(reference.manifest_id)
            frame = self._bundles.read_manifest(reference.manifest_id)
            if frame.empty:
                continue
            close = _decimal(frame.iloc[-1]["close"], label="latest close")
            if close == 0:
                continue
            choices.append(
                SignalInstrumentChoice(
                    instrument,
                    manifest.instrument_name
                    or common_instrument_name(instrument)
                    or instrument.code,
                    asset_type,
                    frame.index[-1].date(),
                    close,
                )
            )
        return tuple(sorted(choices, key=lambda item: str(item.instrument)))

    def strategies(self) -> tuple[SignalStrategyChoice, ...]:
        return tuple(
            SignalStrategyChoice(item.instance_id, item.name, item.strategy_type)
            for item in self._strategies.list()
            if item.enabled and self._strategies.is_watchlist_enabled(item.watchlist_id)
        )

    def latest_account(self) -> StoredAccountSnapshot | None:
        return self._active_accounts().latest()

    def account_history(self) -> tuple[StoredAccountSnapshot, ...]:
        return self._active_accounts().history()

    def account_valuation_history(self) -> tuple[SignalAccountValuationPoint, ...]:
        records = tuple(
            sorted(
                self._active_accounts().history(),
                key=lambda item: (item.snapshot.as_of, item.row_id),
            )
        )
        if not records:
            return ()
        bundle = self._bundles.latest()
        if bundle is None:
            latest_by_day = {item.snapshot.as_of: item for item in records}
            return tuple(
                SignalAccountValuationPoint(
                    item.snapshot.as_of,
                    item.snapshot.cash,
                    item.snapshot.equity - item.snapshot.cash,
                    item.snapshot.equity,
                    item.row_id,
                    {
                        position.instrument: position.market_value
                        for position in item.snapshot.positions
                    },
                )
                for item in latest_by_day.values()
            )
        references = self._bundles.references_by_instrument(bundle)
        required = tuple(
            sorted(
                {
                    position.instrument
                    for record in records
                    for position in record.snapshot.positions
                },
                key=str,
            )
        )
        selected = tuple(item for item in required if item in references)
        if not selected:
            selected = (bundle.instruments[0],)
        price_days: dict[InstrumentId, tuple[date, ...]] = {}
        prices: dict[InstrumentId, tuple[Decimal, ...]] = {}
        market_days: set[date] = set()
        for instrument in selected:
            frame = self._bundles.read_manifest(references[instrument].manifest_id)
            rows = tuple(
                (timestamp.date(), _decimal(row["close"], label="valuation close"))
                for timestamp, row in frame.iterrows()
            )
            if not rows:
                continue
            price_days[instrument] = tuple(item[0] for item in rows)
            prices[instrument] = tuple(item[1] for item in rows)
            market_days.update(price_days[instrument])
        first_day = records[0].snapshot.as_of
        valuation_days = tuple(
            sorted(
                {item.snapshot.as_of for item in records}
                | {day for day in market_days if day >= first_day}
            )
        )
        points: list[SignalAccountValuationPoint] = []
        record_index = 0
        active: StoredAccountSnapshot | None = None
        for day in valuation_days:
            while (
                record_index < len(records)
                and records[record_index].snapshot.as_of <= day
            ):
                active = records[record_index]
                record_index += 1
            if active is None:
                continue
            marked_positions: list[Position] = []
            for position in active.snapshot.positions:
                days = price_days.get(position.instrument)
                values = prices.get(position.instrument)
                mark_price = position.mark_price
                if days is not None and values is not None:
                    price_index = bisect_right(days, day) - 1
                    if price_index >= 0:
                        mark_price = values[price_index]
                marked_positions.append(
                    Position(
                        position.instrument,
                        position.quantity,
                        position.available_quantity,
                        position.average_cost,
                        mark_price,
                    )
                )
            valuation = AccountSnapshot(day, active.snapshot.cash, marked_positions)
            points.append(
                SignalAccountValuationPoint(
                    day,
                    valuation.cash,
                    valuation.equity - valuation.cash,
                    valuation.equity,
                    active.row_id,
                    {
                        position.instrument: position.market_value
                        for position in valuation.positions
                    },
                )
            )
        return tuple(points)

    def enable_shadow_simulation(
        self,
        execution_timing: ShadowExecutionTiming,
        *,
        commission_rate: Decimal,
        minimum_commission: Decimal,
        slippage_bps: int,
    ) -> SignalAccountProfile:
        latest = self._active_accounts().latest()
        if latest is None:
            raise LookupError("SIGNAL_SHADOW_ACCOUNT_MISSING")
        setting = SignalShadowSimulationSetting(
            latest.row_id,
            execution_timing,
            commission_rate,
            minimum_commission,
            slippage_bps,
        )
        return self._account_profiles.save_shadow_simulation(
            self.active_account_profile().account_id,
            setting,
        )

    def disable_shadow_simulation(self) -> SignalAccountProfile:
        return self._account_profiles.save_shadow_simulation(
            self.active_account_profile().account_id,
            None,
        )

    def shadow_simulation(self) -> SignalShadowSimulation | None:
        setting = self.active_account_profile().shadow_simulation
        if setting is None:
            return None
        start = self._active_accounts().get(setting.start_snapshot_row_id)
        if start is None:
            raise LookupError("SIGNAL_SHADOW_START_MISSING")
        bundle = self._bundles.latest()
        if bundle is None:
            raise LookupError("SIGNAL_SHADOW_MARKET_DATA_MISSING")
        daily_decisions: dict[date, DecisionExportRecord] = {}
        for record in sorted(
            self.decision_history(),
            key=lambda item: (item.result.decision_at, item.decision_id),
        ):
            if record.result.decision_date >= start.snapshot.as_of:
                daily_decisions[record.result.decision_date] = record
        decisions = tuple(daily_decisions.values())
        required = {
            position.instrument for position in start.snapshot.positions
        } | {
            recommendation.instrument
            for record in decisions
            for recommendation in record.result.recommendations
        }
        references = self._bundles.references_by_instrument(bundle)
        unavailable = tuple(sorted(required - set(references), key=str))
        selected = tuple(sorted(required & set(references), key=str))
        if not selected:
            selected = (bundle.instruments[0],)
        opens: dict[InstrumentId, dict[date, Decimal]] = {}
        closes: dict[InstrumentId, dict[date, Decimal]] = {}
        close_days: dict[InstrumentId, tuple[date, ...]] = {}
        market_days: set[date] = {start.snapshot.as_of}
        for instrument in selected:
            frame = self._bundles.read_manifest(references[instrument].manifest_id)
            opens[instrument] = {
                timestamp.date(): _decimal(row["open"], label="shadow open")
                for timestamp, row in frame.iterrows()
            }
            closes[instrument] = {
                timestamp.date(): _decimal(row["close"], label="shadow close")
                for timestamp, row in frame.iterrows()
            }
            close_days[instrument] = tuple(closes[instrument])
            market_days.update(
                day for day in close_days[instrument] if day >= start.snapshot.as_of
            )
        days = tuple(sorted(market_days))
        latest_market_day = days[-1]
        fallback_prices = {
            position.instrument: position.mark_price
            for position in start.snapshot.positions
        }

        def close_at(instrument: InstrumentId, day: date) -> Decimal | None:
            instrument_days = close_days.get(instrument)
            if instrument_days:
                index = bisect_right(instrument_days, day) - 1
                if index >= 0:
                    return closes[instrument][instrument_days[index]]
            return fallback_prices.get(instrument)

        scheduled: dict[date, list[DecisionExportRecord]] = {}
        execution_rows: list[SignalShadowExecution] = []
        for record in decisions:
            actionable = tuple(
                item
                for item in record.result.recommendations
                if not item.blocked and item.quantity_delta != 0
            )
            if not actionable:
                execution_rows.append(
                    SignalShadowExecution(
                        record.decision_id,
                        record.result.decision_date,
                        record.result.decision_date,
                        "no_trade",
                        0,
                        Decimal("0.00"),
                    )
                )
                continue
            candidates: tuple[date, ...]
            if setting.execution_timing is ShadowExecutionTiming.DECISION_CLOSE:
                candidates = (record.result.decision_date,)
                price_rows = closes
            else:
                candidates = tuple(
                    day
                    for day in days
                    if record.result.decision_date < day <= record.result.valid_until
                )
                price_rows = opens
            execution_day = next(
                (
                    day
                    for day in candidates
                    if all(day in price_rows.get(item.instrument, {}) for item in actionable)
                ),
                None,
            )
            if execution_day is None:
                pending = latest_market_day < record.result.valid_until
                execution_rows.append(
                    SignalShadowExecution(
                        record.decision_id,
                        record.result.decision_date,
                        None,
                        "pending" if pending else "unfilled",
                        0,
                        Decimal("0.00"),
                    )
                )
                continue
            scheduled.setdefault(execution_day, []).append(record)

        shadow_cash = start.snapshot.cash
        shadow_quantities = {
            position.instrument: position.quantity
            for position in start.snapshot.positions
        }
        hold_cash = start.snapshot.cash
        hold_quantities = dict(shadow_quantities)
        actual = tuple(
            item
            for item in self.account_valuation_history()
            if item.day >= start.snapshot.as_of
        )
        actual_days = tuple(item.day for item in actual)

        def marked_equity(
            cash: Decimal,
            quantities: Mapping[InstrumentId, int],
            day: date,
        ) -> Decimal:
            total = cash
            for instrument, quantity in quantities.items():
                price = close_at(instrument, day)
                if price is not None:
                    total += price * quantity
            return total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        def costs(gross: Decimal) -> Decimal:
            if gross <= 0:
                return Decimal("0.00")
            return max(
                setting.minimum_commission,
                gross * setting.commission_rate,
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        points: list[SignalShadowPoint] = []
        for day in days:
            for record in scheduled.get(day, ()):
                recommendations = tuple(
                    item
                    for item in record.result.recommendations
                    if not item.blocked and item.quantity_delta != 0
                )
                base_equity = marked_equity(
                    shadow_cash,
                    shadow_quantities,
                    record.result.decision_date,
                )
                price_rows = (
                    closes
                    if setting.execution_timing is ShadowExecutionTiming.DECISION_CLOSE
                    else opens
                )
                execution_prices: dict[InstrumentId, Decimal] = {}
                targets: dict[InstrumentId, int] = {}
                for item in recommendations:
                    raw_price = price_rows[item.instrument][day]
                    provisional_target = int(
                        item.final_weight * base_equity / raw_price / 100
                    ) * 100
                    current_quantity = shadow_quantities.get(item.instrument, 0)
                    slip = Decimal(setting.slippage_bps) / Decimal("10000")
                    multiplier = Decimal("1") + (
                        slip if provisional_target > current_quantity else -slip
                    )
                    price = (raw_price * multiplier).quantize(
                        Decimal("0.0001"), rounding=ROUND_HALF_UP
                    )
                    execution_prices[item.instrument] = price
                    targets[item.instrument] = int(
                        item.final_weight * base_equity / price / 100
                    ) * 100
                trade_count = 0
                total_costs = Decimal("0.00")
                for item in recommendations:
                    instrument = item.instrument
                    current_quantity = shadow_quantities.get(instrument, 0)
                    target = targets[instrument]
                    if target >= current_quantity:
                        continue
                    quantity = current_quantity - target
                    gross = execution_prices[instrument] * quantity
                    fee = costs(gross)
                    shadow_cash += gross - fee
                    total_costs += fee
                    trade_count += 1
                    if target:
                        shadow_quantities[instrument] = target
                    else:
                        shadow_quantities.pop(instrument, None)
                    fallback_prices[instrument] = execution_prices[instrument]
                for item in recommendations:
                    instrument = item.instrument
                    current_quantity = shadow_quantities.get(instrument, 0)
                    desired = max(0, targets[instrument] - current_quantity)
                    if desired <= 0:
                        continue
                    price = execution_prices[instrument]
                    quantity = min(desired, int(shadow_cash / price / 100) * 100)
                    while quantity > 0:
                        gross = price * quantity
                        fee = costs(gross)
                        if gross + fee <= shadow_cash:
                            break
                        quantity -= 100
                    if quantity <= 0:
                        continue
                    gross = price * quantity
                    fee = costs(gross)
                    shadow_cash -= gross + fee
                    total_costs += fee
                    trade_count += 1
                    shadow_quantities[instrument] = current_quantity + quantity
                    fallback_prices[instrument] = price
                execution_rows.append(
                    SignalShadowExecution(
                        record.decision_id,
                        record.result.decision_date,
                        day,
                        "executed" if trade_count else "no_trade",
                        trade_count,
                        total_costs,
                    )
                )
            actual_index = bisect_right(actual_days, day) - 1
            actual_equity = (
                start.snapshot.equity
                if actual_index < 0
                else actual[actual_index].equity
            )
            points.append(
                SignalShadowPoint(
                    day,
                    actual_equity,
                    marked_equity(shadow_cash, shadow_quantities, day),
                    marked_equity(hold_cash, hold_quantities, day),
                )
            )
        return SignalShadowSimulation(
            setting,
            start.snapshot.as_of,
            tuple(points),
            tuple(
                sorted(
                    execution_rows,
                    key=lambda item: (item.signal_day, item.decision_id),
                )
            ),
            unavailable,
        )

    def compact_account_history(self) -> int:
        protected = self._decisions.referenced_account_snapshot_ids().union(
            item.resulting_snapshot_row_id
            for item in self._executions.history()
            if item.resulting_snapshot_row_id is not None
        )
        return self._active_accounts().compact_duplicates(frozenset(protected))

    def save_account(
        self,
        cash: object,
        positions: Sequence[AccountPositionInput],
    ) -> StoredAccountSnapshot:
        parsed_cash = _decimal(cash, label="cash", cents=True)
        choices = {str(item.instrument): item for item in self.instruments()}
        rows = tuple(positions)
        if any(type(item) is not AccountPositionInput for item in rows):
            raise TypeError("positions must contain AccountPositionInput values")
        if not choices and rows:
            raise LookupError("SIGNAL_MARKET_DATA_MISSING")
        as_of = min(
            (item.data_day for item in choices.values()),
            default=datetime.now(SHANGHAI).date(),
        )
        # A cash-only snapshot may predate the first market sync. Saving today's
        # holdings must not move behind an existing snapshot when prices are older.
        as_of = max(
            as_of,
            max((item.snapshot.as_of for item in self._active_accounts().history()), default=as_of),
        )
        parsed_positions: list[Position] = []
        seen: set[str] = set()
        for row in rows:
            choice = choices.get(row.instrument)
            if choice is None:
                raise LookupError("SIGNAL_POSITION_DATA_MISSING")
            if row.instrument in seen:
                raise ValueError("SIGNAL_POSITION_DUPLICATE")
            seen.add(row.instrument)
            quantity = _quantity(row.quantity, label="quantity")
            available = _quantity(row.available_quantity, label="available quantity")
            if available > quantity:
                raise ValueError("SIGNAL_AVAILABLE_QUANTITY_EXCEEDS_POSITION")
            if quantity == 0:
                continue
            average_cost = _decimal(row.average_cost, label="average cost")
            parsed_positions.append(
                Position(
                    choice.instrument,
                    quantity,
                    available,
                    average_cost,
                    choice.close,
                )
            )
        snapshot = AccountSnapshot(as_of, parsed_cash, parsed_positions)
        if snapshot.equity == 0:
            raise ValueError("SIGNAL_ACCOUNT_EQUITY_REQUIRED")
        saved = self._active_accounts().save(
            snapshot, expected_row_id=self._expected_snapshot_id
        )
        if self._bound_account_id is not None:
            self._expected_snapshot_id = saved.row_id
        return saved

    def generate(
        self,
        selections: Sequence[SelectedDecisionStrategy],
        *,
        cash_reserve: Decimal,
        minimum_trade_amount: Decimal,
    ) -> DecisionExportRecord:
        selected = tuple(selections)
        profile = self.active_account_profile()
        self.save_strategy_configuration(
            selected,
            cash_reserve=cash_reserve,
            minimum_trade_amount=minimum_trade_amount,
        )
        return self._decisions.generate_latest(
            f"{self._id_factory('decision')}:{profile.account_id}",
            selected,
            cash_reserve=cash_reserve,
            minimum_trade_amount=minimum_trade_amount,
            accounts=self._active_accounts(),
        )

    def latest_decision(self) -> DecisionExportRecord | None:
        records, _ = self.readable_decisions()
        return records[0] if records else None

    def decision_history(self) -> tuple[DecisionExportRecord, ...]:
        records, _ = self.readable_decisions()
        return records

    def readable_decisions(self) -> tuple[tuple[DecisionExportRecord, ...], int]:
        records, invalid_count = self._decisions.readable_history(
            self._adopted_decision_ids()
        )
        profile = self.active_account_profile()
        return (
            tuple(item for item in records if self._decision_belongs_to(item, profile)),
            invalid_count,
        )

    def decision(self, decision_id: str) -> DecisionExportRecord | None:
        record = self._decisions.get(decision_id)
        if record is None or not self._decision_belongs_to(
            record,
            self.active_account_profile(),
        ):
            return None
        return record

    def delete_decision(self, decision_id: str) -> bool:
        record = self.decision(decision_id)
        if record is None:
            raise LookupError("SIGNAL_DECISION_NOT_FOUND")
        execution = self._executions.get(decision_id)
        if execution is not None and execution.status is not SignalExecutionStatus.IGNORED:
            raise ValueError("SIGNAL_DECISION_ADOPTED_DELETE_FORBIDDEN")
        deleted = self._decisions.delete(decision_id)
        self._executions.delete((decision_id,))
        return deleted

    def clear_decisions(self) -> int:
        records, _ = self.readable_decisions()
        adopted_ids = self._adopted_decision_ids()
        decision_ids = tuple(
            item.decision_id for item in records if item.decision_id not in adopted_ids
        )
        deleted = sum(self._decisions.delete(item) for item in decision_ids)
        self._executions.delete(decision_ids)
        return deleted

    def clear_invalid_decisions(self) -> int:
        invalid_ids = self._decisions.clear_invalid(self._adopted_decision_ids())
        self._executions.delete(invalid_ids)
        return len(invalid_ids)

    def _adopted_decision_ids(self) -> frozenset[str]:
        return frozenset(
            item.decision_id
            for item in self._executions.history()
            if item.status in {
                SignalExecutionStatus.EXECUTED,
                SignalExecutionStatus.PARTIAL,
            }
        )

    def compare_decision(self, decision_id: str) -> SignalDecisionComparison:
        record = self.decision(decision_id)
        if record is None:
            raise LookupError("SIGNAL_DECISION_NOT_FOUND")
        account = self._active_accounts().get(record.result.account_snapshot_row_id)
        if account is None or account.content_hash != record.result.account_snapshot_hash:
            raise LookupError("SIGNAL_COMPARISON_ACCOUNT_MISSING")
        bundle = self._bundles.latest()
        if bundle is None:
            raise LookupError("SIGNAL_COMPARISON_MARKET_DATA_MISSING")
        references = self._bundles.references_by_instrument(bundle)
        baseline_quantities = {
            item.instrument: item.quantity for item in account.snapshot.positions
        }
        adopted_quantities = dict(baseline_quantities)
        for item in record.result.recommendations:
            if item.target_quantity:
                adopted_quantities[item.instrument] = item.target_quantity
            else:
                adopted_quantities.pop(item.instrument, None)
        instruments = tuple(
            sorted(set(baseline_quantities) | set(adopted_quantities), key=str)
        )
        if not instruments or any(item not in references for item in instruments):
            raise LookupError("SIGNAL_COMPARISON_MARKET_DATA_MISSING")
        frames = {
            item: self._bundles.read_manifest(references[item].manifest_id)
            for item in instruments
        }
        closes = {
            instrument: {
                timestamp.date(): Decimal(str(row["close"]))
                for timestamp, row in frame.iterrows()
            }
            for instrument, frame in frames.items()
        }
        common_days = set.intersection(
            *(set(values) for values in closes.values())
        )
        days = tuple(sorted(day for day in common_days if day >= record.result.decision_date))
        if not days:
            raise LookupError("SIGNAL_COMPARISON_RANGE_MISSING")

        def equity(
            day: date,
            cash: Decimal,
            quantities: Mapping[InstrumentId, int],
        ) -> Decimal:
            total = cash
            for instrument, quantity in quantities.items():
                total += closes[instrument][day] * quantity
            return total.quantize(Decimal("0.01"))

        points = tuple(
            SignalComparisonPoint(
                day,
                equity(day, record.result.remaining_cash, adopted_quantities),
                equity(day, account.snapshot.cash, baseline_quantities),
            )
            for day in days
        )
        last = points[-1]

        def total_return(end: Decimal, start: Decimal) -> Decimal:
            if start == 0:
                return Decimal("0")
            return ((end / start) - Decimal("1")).quantize(Decimal("0.0001"))

        return SignalDecisionComparison(
            record.decision_id,
            points,
            total_return(last.adopted_equity, record.result.decision_equity),
            total_return(last.ignored_equity, record.result.decision_equity),
            (last.adopted_equity - last.ignored_equity).quantize(Decimal("0.01")),
        )

    def execution(self, decision_id: str) -> SignalExecutionRecord | None:
        record = self._executions.get(decision_id)
        if record is None or record.profile_id != self.active_account_profile().account_id:
            return None
        return record

    def execution_history(self) -> tuple[SignalExecutionRecord, ...]:
        profile_id = self.active_account_profile().account_id
        return tuple(
            item for item in self._executions.history() if item.profile_id == profile_id
        )

    def decision_freshness(self, record: DecisionExportRecord) -> SignalDecisionFreshness:
        reasons: list[str] = []
        account = self._active_accounts().latest()
        if account is None or account.content_hash != record.result.account_snapshot_hash:
            reasons.append("HOLDINGS_CHANGED")
        profile = self.active_account_profile()
        configured_budgets = {
            item.strategy_instance_id: item.budget for item in profile.strategies
        }
        raw_budgets = record.snapshot.allocator_configuration["strategy_budgets"]
        if not isinstance(raw_budgets, Mapping):
            raise ValueError("DECISION_EXPORT_INTEGRITY")
        snapshot_budgets = dict(raw_budgets)
        snapshot_reserve = record.snapshot.allocator_configuration["cash_reserve"]
        snapshot_minimum = record.snapshot.risk_configuration.get(
            "minimum_trade_amount"
        )
        if (
            configured_budgets != snapshot_budgets
            or profile.cash_reserve != snapshot_reserve
            or (
                snapshot_minimum is not None
                and profile.minimum_trade_amount != snapshot_minimum
            )
        ):
            reasons.append("STRATEGY_CONFIGURATION_CHANGED")
        bundle = self._bundles.latest()
        if bundle is None:
            reasons.append("MARKET_DATA_CHANGED")
        else:
            current = self._bundles.references_by_instrument(bundle)
            for manifest in record.market_manifests:
                loaded = self._bundles.load_manifest(manifest.manifest_id)
                reference = current.get(InstrumentId.parse(loaded.instrument))
                if reference is None or reference.manifest_id != manifest.manifest_id:
                    reasons.append("MARKET_DATA_CHANGED")
                    break
        return SignalDecisionFreshness(bool(reasons), tuple(reasons))

    def record_execution(
        self,
        decision_id: str,
        status: SignalExecutionStatus,
        fills: Sequence[SignalExecutionFillInput],
        *,
        fees: object,
        recorded_at: datetime,
    ) -> SignalExecutionRecord:
        decision = self.decision(decision_id)
        if decision is None:
            raise LookupError("SIGNAL_DECISION_NOT_FOUND")
        if type(status) is not SignalExecutionStatus:
            raise TypeError("execution status must be exact")
        parsed_fees = _decimal(fees, label="execution fees", cents=True)
        if type(recorded_at) is not datetime or recorded_at.tzinfo is None:
            raise ValueError("execution time must be timezone-aware")
        existing_execution = self._executions.get(decision_id)
        if existing_execution is not None:
            raise ValueError("SIGNAL_EXECUTION_ALREADY_RECORDED")
        if status is SignalExecutionStatus.IGNORED:
            record = SignalExecutionRecord(
                decision_id,
                self.active_account_profile().account_id,
                status,
                (),
                parsed_fees,
                recorded_at,
            )
            return self._executions.save(record)
        if self.decision_freshness(decision).stale:
            raise ValueError("SIGNAL_DECISION_STALE")
        if decision.result.valid_until < recorded_at.astimezone(SHANGHAI).date():
            raise ValueError("SIGNAL_DECISION_EXPIRED")
        recommendation_by_instrument = {
            str(item.instrument): item for item in decision.result.recommendations
        }
        parsed_fills: list[SignalExecutionFill] = []
        current = self._active_accounts().latest()
        if current is None:
            raise LookupError("ACCOUNT_SNAPSHOT_MISSING")
        positions = {str(item.instrument): item for item in current.snapshot.positions}
        cash = current.snapshot.cash
        for fill in fills:
            if type(fill) is not SignalExecutionFillInput:
                raise TypeError("execution fills must be exact inputs")
            recommendation = recommendation_by_instrument.get(fill.instrument)
            if recommendation is None:
                raise LookupError("SIGNAL_EXECUTION_INSTRUMENT_UNKNOWN")
            delta = _signed_quantity(fill.quantity_delta, label="execution quantity")
            if delta * recommendation.quantity_delta <= 0 or abs(delta) > abs(
                recommendation.quantity_delta
            ):
                raise ValueError("SIGNAL_EXECUTION_QUANTITY_INVALID")
            price = _decimal(fill.execution_price, label="execution price")
            if price == 0:
                raise ValueError("SIGNAL_EXECUTION_PRICE_INVALID")
            existing = positions.get(fill.instrument)
            current_quantity = 0 if existing is None else existing.quantity
            new_quantity = current_quantity + delta
            if new_quantity < 0:
                raise ValueError("SIGNAL_EXECUTION_POSITION_NEGATIVE")
            instrument = recommendation.instrument
            if new_quantity == 0:
                positions.pop(fill.instrument, None)
            else:
                old_cost = Decimal("0") if existing is None else existing.average_cost
                average_cost = (
                    price
                    if existing is None
                    else (
                        (old_cost * current_quantity + price * delta) / new_quantity
                        if delta > 0
                        else old_cost
                    )
                )
                available = (
                    new_quantity
                    if existing is None
                    else min(existing.available_quantity + delta, new_quantity)
                )
                positions[fill.instrument] = Position(
                    instrument,
                    new_quantity,
                    max(0, available),
                    average_cost,
                    price,
                )
            cash -= price * delta
            parsed_fills.append(SignalExecutionFill(fill.instrument, delta, price))
        cash -= parsed_fees
        if cash < 0:
            raise ValueError("SIGNAL_EXECUTION_CASH_NEGATIVE")
        executed_quantities = {
            item.instrument: item.quantity_delta for item in parsed_fills
        }
        recommended_quantities = {
            instrument: item.quantity_delta
            for instrument, item in recommendation_by_instrument.items()
            if item.quantity_delta != 0
        }
        is_complete = executed_quantities == recommended_quantities
        if status is SignalExecutionStatus.EXECUTED and not is_complete:
            raise ValueError("SIGNAL_EXECUTION_INCOMPLETE")
        if status is SignalExecutionStatus.PARTIAL and is_complete:
            raise ValueError("SIGNAL_EXECUTION_ALREADY_COMPLETE")
        snapshot = AccountSnapshot(current.snapshot.as_of, cash, tuple(positions.values()))
        saved = self._active_accounts().save(snapshot)
        record = SignalExecutionRecord(
            decision_id,
            self.active_account_profile().account_id,
            status,
            tuple(sorted(parsed_fills, key=lambda item: item.instrument)),
            parsed_fees,
            recorded_at,
            saved.row_id,
        )
        try:
            return self._executions.save(record)
        except Exception:
            current = self._active_accounts().latest()
            if current is not None and current.row_id == saved.row_id:
                self._active_accounts().compact_duplicates(
                    frozenset(
                        self._decisions.referenced_account_snapshot_ids()
                    )
                )
            raise

    @staticmethod
    def _decision_belongs_to(
        record: DecisionExportRecord,
        profile: SignalAccountProfile,
    ) -> bool:
        _, separator, tagged_profile_id = record.decision_id.rpartition(":")
        if separator:
            return tagged_profile_id == profile.account_id
        return record.result.account_id == profile.account_id
