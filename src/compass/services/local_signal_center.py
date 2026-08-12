from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, DecimalException
from numbers import Number

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
    SignalAccountProfile,
    SignalAccountRepository,
    SignalAccountStrategySetting,
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

    def account_profiles(self) -> tuple[SignalAccountProfile, ...]:
        return self._account_profiles.state().profiles

    def active_account_profile(self) -> SignalAccountProfile:
        return self._account_profiles.state().active

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
        if not choices:
            raise LookupError("SIGNAL_MARKET_DATA_MISSING")
        as_of = min(item.data_day for item in choices.values())
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
        return self._active_accounts().save(snapshot)

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
