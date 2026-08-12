from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, cast
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from compass.backtest.market_rules import MarketRuleBook, MarketRuleProfile
from compass.backtest.snapshot import ManifestReference, RunSnapshot, StrategySnapshot
from compass.domain.market import AssetType, InstrumentId
from compass.portfolio.trace import AllocationPolicy
from compass.risk.base import RiskRule
from compass.risk.engine import RiskEngine
from compass.services.decision_service import (
    CloseDecisionRequest,
    DecisionService,
    InstrumentRiskMetadata,
)
from compass.services.dataset_provenance import (
    snapshot_data_quality,
    validate_dataset_provenance,
)
from compass.services.export_service import (
    DecisionExportRecord,
    DecisionManifestProvenance,
    DecisionStrategyProvenance,
    decision_snapshot_run_id,
)
from compass.services.local_crud_gateways import (
    LocalSettingsGateway,
    LocalStrategyGateway,
)
from compass.services.local_market_configuration import (
    local_instruments,
    local_risk_engine,
    local_rule_book,
)
from compass.services.safe_display import safe_identifier
from compass.storage.account_repository import AccountRepository
from compass.storage.dataset_bundle_repository import DatasetBundleRepository
from compass.storage.decision_export_repository import DecisionExportRepository
from compass.risk.rules import MinimumTradeAmountRule
from compass.strategies.base import Strategy, StrategyContext, StrategyDecision
from compass.strategies.registry import StrategyRegistry
from compass.ui.pages.settings import FeeProfileSetting, RiskTemplateSetting
from compass.ui.pages.strategies import strategy_parameters_json


SHANGHAI = ZoneInfo("Asia/Shanghai")
Clock = Callable[[], datetime]
TradingCalendar = Callable[[date], bool]
CalendarCoverage = Callable[[date, date], object]


@dataclass(slots=True)
class _ConfiguredDecisionStrategy:
    strategy_id: str
    strategy: Strategy
    instrument_pool: tuple[InstrumentId, ...]

    def generate_targets(self, context: StrategyContext) -> StrategyDecision:
        return self.strategy.generate_targets(context)


@dataclass(frozen=True, slots=True)
class SelectedDecisionStrategy:
    strategy_instance_id: str
    budget: Decimal

    def __post_init__(self) -> None:
        safe_identifier(self.strategy_instance_id, label="strategy instance id")
        if type(self.budget) is not Decimal:
            raise TypeError("strategy budget must be an exact Decimal")
        if not self.budget.is_finite() or not Decimal("0") < self.budget <= Decimal("1"):
            raise ValueError("strategy budget must be between zero and one")


def _profile_payload(profile: MarketRuleProfile) -> dict[str, object]:
    return {
        "asset_type": profile.asset_type.value,
        "buy_lot_size": profile.buy_lot_size,
        "commission_rate": profile.commission_rate,
        "effective_from": profile.effective_from,
        "effective_to": profile.effective_to,
        "exchange": profile.exchange.value,
        "fee_profile_confirmed": profile.fee_profile_confirmed,
        "maximum_volume_participation": profile.maximum_volume_participation,
        "minimum_commission": profile.minimum_commission,
        "odd_lot_sell_policy": profile.odd_lot_sell_policy.value,
        "price_limit_mode": profile.price_limit_mode.value,
        "price_limit_rate": profile.price_limit_rate,
        "profile_id": profile.profile_id,
        "risk_warning_price_limit_rate": profile.risk_warning_price_limit_rate,
        "same_day_sell_eligible": profile.same_day_sell_eligible,
        "sell_stamp_duty_rate": profile.sell_stamp_duty_rate,
        "settlement_mode": profile.settlement_mode.value,
        "slippage_bps": profile.slippage_bps,
        "transfer_fee_rate": profile.transfer_fee_rate,
    }


def _rule_payload(rule: RiskRule) -> dict[str, object]:
    """Freeze all dataclass instance fields of a configured local risk rule."""

    if not is_dataclass(rule):
        raise TypeError("local risk rule must be a dataclass")
    parameters = {
        field.name: getattr(rule, field.name)
        for field in fields(cast(Any, rule))
    }
    return {
        "code": rule.code,
        "enabled": rule.enabled,
        "parameters": parameters,
        "priority": rule.priority,
        "stage": rule.stage.value,
        "type": f"{type(rule).__module__}.{type(rule).__qualname__}",
    }


def _allocation_payload(policy: AllocationPolicy) -> dict[str, object]:
    return {
        "asset_class_budgets": {
            asset_type.value: budget
            for asset_type, budget in policy.asset_class_budgets.items()
        },
        "asset_types": {
            str(instrument): asset_type.value
            for instrument, asset_type in policy.asset_types.items()
        },
        "cash_reserve": policy.cash_reserve,
        "kind": "deterministic_allocator",
        "strategy_budgets": dict(policy.strategy_budgets),
    }


def _risk_payload(
    engine: RiskEngine,
    templates: tuple[RiskTemplateSetting, ...],
    minimum_trade_amount: Decimal | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "active": bool(engine.rules),
        "rules": tuple(_rule_payload(rule) for rule in engine.rules),
        "templates": tuple(
            {
                "active": template.active,
                "template_id": template.template_id,
            }
            for template in templates
        ),
    }
    if minimum_trade_amount is not None:
        payload["minimum_trade_amount"] = minimum_trade_amount
    return payload


def _market_rule_payload(rule_book: MarketRuleBook) -> dict[str, object]:
    return {
        "execution": "close_decision",
        "profiles": tuple(_profile_payload(profile) for profile in rule_book.profiles),
    }


def _fee_payload(
    fee: FeeProfileSetting,
    rule_book: MarketRuleBook,
) -> dict[str, object]:
    return {
        "confirmed": fee.confirmed,
        "profile_id": fee.profile_id,
        "schedules": tuple(_profile_payload(profile) for profile in rule_book.profiles),
    }


class LocalDecisionGateway:
    def __init__(
        self,
        *,
        accounts: AccountRepository,
        strategies: LocalStrategyGateway,
        registry: StrategyRegistry,
        bundles: DatasetBundleRepository,
        settings: LocalSettingsGateway,
        repository: DecisionExportRepository,
        app_git_commit: str | None,
        clock: Clock,
        trading_calendar: TradingCalendar,
        calendar_coverage: CalendarCoverage | None = None,
    ) -> None:
        self._accounts = accounts
        self._strategies = strategies
        self._registry = registry
        self._bundles = bundles
        self._settings = settings
        self._repository = repository
        self._app_git_commit = app_git_commit
        self._clock = clock
        self._trading_calendar = trading_calendar
        self._calendar_coverage = calendar_coverage

    def generate(
        self,
        decision_id: str,
        strategy_instance_id: str,
        manifest_id: str,
        *,
        valid_until: date | None = None,
    ) -> DecisionExportRecord:
        checked_decision = safe_identifier(decision_id, label="decision id")
        checked_strategy = safe_identifier(
            strategy_instance_id,
            label="strategy instance id",
        )
        app_git_commit = self._app_git_commit
        if app_git_commit is None:
            raise LookupError("DECISION_APP_VERSION_UNKNOWN")
        instance = next(
            (item for item in self._strategies.list() if item.instance_id == checked_strategy),
            None,
        )
        if instance is None or not instance.enabled:
            raise LookupError("DECISION_STRATEGY_UNAVAILABLE")
        if not self._strategies.is_watchlist_enabled(instance.watchlist_id):
            raise LookupError("DECISION_WATCHLIST_DISABLED")
        bundle = self._bundles.for_manifest(manifest_id)
        pool = self._strategies.pool_instruments(instance.instance_id)
        manifest_by_instrument = self._bundles.references_by_instrument(bundle)
        if not set(pool).issubset(manifest_by_instrument):
            raise LookupError("DECISION_POOL_DATA_MISSING")
        bars = {
            instrument: self._bundles.read_manifest(manifest_by_instrument[instrument].manifest_id)
            for instrument in pool
        }
        dataset_manifests = tuple(
            self._bundles.load_manifest(
                manifest_by_instrument[instrument].manifest_id
            )
            for instrument in pool
        )
        instruments = local_instruments(pool)
        now = self._clock()
        if type(now) is not datetime or now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("decision clock must return a timezone-aware datetime")
        decision_at = now.astimezone(SHANGHAI)
        decision_date = decision_at.date()
        dataset_provenance = validate_dataset_provenance(
            dataset_manifests,
            required_through=decision_date,
            failure_prefix="DECISION",
        )
        settings = self._settings.state()
        fee = settings.fee_profile
        if fee is None or not fee.confirmed:
            raise ValueError("DECISION_FEE_PROFILE_UNCONFIRMED")
        final_valid_until = (
            self._next_session(decision_date) if valid_until is None else valid_until
        )
        metadata = self._registry.describe(instance.strategy_type)
        parameters = metadata.parameters_type.model_validate_json(
            strategy_parameters_json(instance.parameters),
            strict=True,
        )
        strategy = self._registry.create(
            instance.strategy_type,
            parameters,
            instance.instance_id,
        )
        allocation_policy = AllocationPolicy(
            strategy_budgets={instance.instance_id: Decimal("0.80")},
            asset_class_budgets={
                AssetType.ETF: Decimal("0.80"),
                AssetType.STOCK: Decimal("0.20"),
            },
            asset_types={
                symbol: instrument.asset_type for symbol, instrument in instruments.items()
            },
            cash_reserve=Decimal("0.10"),
        )
        risk_engine = local_risk_engine(
            active=any(item.active for item in settings.risk_templates)
        )
        rule_book = local_rule_book(
            instruments,
            fee_confirmed=True,
        )
        result = DecisionService(self._accounts).generate_close_decision(
            CloseDecisionRequest(
                account_id=self._accounts.account_id,
                decision_at=decision_at,
                valid_until=final_valid_until,
                data_accepted=cast(bool, bundle.data_quality["accepted"]),
                daily_close_complete=decision_at.timetz().replace(tzinfo=None) >= time(15, 0),
                market_data_source_at=dataset_provenance.fetched_at,
                instruments=instruments,
                bars=bars,
                strategies=(
                    _ConfiguredDecisionStrategy(
                        strategy_id=instance.instance_id,
                        strategy=strategy,
                        instrument_pool=tuple(pool),
                    ),
                ),
                allocation_policy=allocation_policy,
                risk_engine=risk_engine,
                rule_book=rule_book,
                risk_metadata={},
                strategy_pools={instance.instance_id: tuple(pool)},
            )
        )
        manifests = []
        for reference in bundle.market_manifests:
            manifest = self._bundles.load_manifest(reference.manifest_id)
            manifests.append(
                DecisionManifestProvenance(
                    manifest_id=manifest.manifest_id,
                    provider=manifest.provider,
                    content_hash=manifest.content_hash,
                    relative_data_path=manifest.relative_data_path,
                )
            )
        snapshot = RunSnapshot(
            run_id=decision_snapshot_run_id(checked_decision),
            schema_version=1,
            market_manifests=tuple(
                ManifestReference(item.manifest_id, item.content_hash)
                for item in sorted(manifests, key=lambda item: item.manifest_id)
            ),
            data_quality=snapshot_data_quality(bundle.data_quality, dataset_provenance),
            strategies=(
                StrategySnapshot(
                    sleeve_id=instance.instance_id,
                    strategy_type=instance.strategy_type,
                    strategy_version=instance.strategy_version,
                    parameters=parameters.model_dump(mode="json"),
                ),
            ),
            instrument_pool={
                "instruments": tuple(str(item) for item in pool),
                "snapshot_id": instance.pool_snapshot_id,
            },
            survivorship_bias={"mode": "static_pool", "warning": True},
            allocator_configuration=_allocation_payload(allocation_policy),
            risk_configuration=_risk_payload(risk_engine, settings.risk_templates),
            market_rule_configuration=_market_rule_payload(rule_book),
            fee_profile_configuration=_fee_payload(fee, rule_book),
            app_git_commit=app_git_commit,
            random_seed=0,
        )
        record = DecisionExportRecord(
            decision_id=checked_decision,
            result=result,
            market_manifests=tuple(sorted(manifests, key=lambda item: item.manifest_id)),
            strategies=(
                DecisionStrategyProvenance(
                    strategy_instance_id=instance.instance_id,
                    strategy_type=instance.strategy_type,
                    strategy_version=instance.strategy_version,
                    parameters=parameters.model_dump(mode="json"),
                ),
            ),
            snapshot=snapshot,
        )

        def require_live_watchlist(session: Session) -> None:
            self._strategies.acquire_enabled_watchlist_write_lock(
                session,
                instance.watchlist_id,
                disabled_error="DECISION_WATCHLIST_DISABLED",
            )

        self._repository.save(record, before_persist=require_live_watchlist)
        return record

    def latest(self) -> DecisionExportRecord | None:
        return self._repository.latest()

    def history(self) -> tuple[DecisionExportRecord, ...]:
        return self._repository.history()

    def readable_history(
        self,
        protected_decision_ids: frozenset[str] = frozenset(),
    ) -> tuple[tuple[DecisionExportRecord, ...], int]:
        return self._repository.readable_history(protected_decision_ids)

    def referenced_account_snapshot_ids(self) -> frozenset[int]:
        return self._repository.referenced_account_snapshot_ids()

    def get(self, decision_id: str) -> DecisionExportRecord | None:
        return self._repository.get(decision_id)

    def delete(self, decision_id: str) -> bool:
        return self._repository.delete(decision_id)

    def clear_invalid(
        self,
        protected_decision_ids: frozenset[str] = frozenset(),
    ) -> tuple[str, ...]:
        return self._repository.clear_invalid(protected_decision_ids)

    def generate_latest(
        self,
        decision_id: str,
        selections: Sequence[SelectedDecisionStrategy],
        *,
        cash_reserve: Decimal = Decimal("0.10"),
        minimum_trade_amount: Decimal = Decimal("5000"),
        accounts: AccountRepository | None = None,
    ) -> DecisionExportRecord:
        """Generate an auditable close signal on the latest common local bar.

        This is deliberately advisory: it persists a frozen decision record but
        never creates an order or talks to a broker.
        """

        checked_decision = safe_identifier(decision_id, label="decision id")
        account_repository = self._accounts if accounts is None else accounts
        if type(account_repository) is not AccountRepository:
            raise TypeError("accounts must be an exact AccountRepository")
        selected = tuple(selections)
        if not selected or any(type(item) is not SelectedDecisionStrategy for item in selected):
            raise ValueError("DECISION_STRATEGY_SELECTION_REQUIRED")
        selected_ids = tuple(item.strategy_instance_id for item in selected)
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("DECISION_STRATEGY_DUPLICATE")
        for label, value in (
            ("cash reserve", cash_reserve),
            ("minimum trade amount", minimum_trade_amount),
        ):
            if type(value) is not Decimal or not value.is_finite() or value < 0:
                raise ValueError(f"{label} must be a finite non-negative Decimal")
        if cash_reserve >= Decimal("1"):
            raise ValueError("cash reserve must be below one")
        if sum((item.budget for item in selected), Decimal("0")) + cash_reserve > Decimal("1"):
            raise ValueError("DECISION_BUDGET_EXCEEDS_AVAILABLE_CAPITAL")
        app_git_commit = self._app_git_commit
        if app_git_commit is None:
            raise LookupError("DECISION_APP_VERSION_UNKNOWN")

        instances_by_id = {item.instance_id: item for item in self._strategies.list()}
        instances = []
        pools: dict[str, tuple[InstrumentId, ...]] = {}
        for selection in selected:
            instance = instances_by_id.get(selection.strategy_instance_id)
            if instance is None or not instance.enabled:
                raise LookupError("DECISION_STRATEGY_UNAVAILABLE")
            if not self._strategies.is_watchlist_enabled(instance.watchlist_id):
                raise LookupError("DECISION_WATCHLIST_DISABLED")
            instances.append(instance)
            pools[instance.instance_id] = self._strategies.pool_instruments(instance.instance_id)

        account = account_repository.latest()
        if account is None:
            raise LookupError(f"ACCOUNT_SNAPSHOT_MISSING:{account_repository.account_id}")
        held = tuple(position.instrument for position in account.snapshot.positions)
        required_symbols = tuple(
            sorted(
                set(held).union(*(set(pool) for pool in pools.values())),
                key=str,
            )
        )
        bundle = self._bundles.latest()
        if bundle is None:
            raise LookupError("DECISION_DATA_BUNDLE_MISSING")
        manifest_by_instrument = self._bundles.references_by_instrument(bundle)
        if not set(required_symbols).issubset(manifest_by_instrument):
            raise LookupError("DECISION_POOL_DATA_MISSING")
        raw_bars = {
            symbol: self._bundles.read_manifest(manifest_by_instrument[symbol].manifest_id)
            for symbol in required_symbols
        }
        common_days: set[date] | None = None
        for frame in raw_bars.values():
            frame_days = {item.date() for item in frame.index}
            common_days = frame_days if common_days is None else common_days & frame_days
        if not common_days:
            raise LookupError("DECISION_COMMON_MARKET_DAY_MISSING")
        decision_date = max(common_days)
        if account.snapshot.as_of > decision_date:
            raise ValueError("ACCOUNT_SNAPSHOT_FROM_FUTURE")
        bars = {
            symbol: frame.loc[frame.index.date <= decision_date].copy(deep=True)
            for symbol, frame in raw_bars.items()
        }
        manifests = tuple(
            self._bundles.load_manifest(manifest_by_instrument[symbol].manifest_id)
            for symbol in required_symbols
        )
        dataset_provenance = validate_dataset_provenance(
            manifests,
            required_through=decision_date,
            failure_prefix="DECISION",
            allow_quality_gaps=bundle.data_quality["mode"] == "degraded",
        )
        decision_at = datetime.combine(decision_date, time(15, 0), SHANGHAI)
        valid_until = self._next_session(decision_date)
        instruments = local_instruments(required_symbols)

        configured_strategies: list[_ConfiguredDecisionStrategy] = []
        parameters_by_id: dict[str, Any] = {}
        for instance in instances:
            metadata = self._registry.describe(instance.strategy_type)
            parameters = metadata.parameters_type.model_validate_json(
                strategy_parameters_json(instance.parameters),
                strict=True,
            )
            parameters_by_id[instance.instance_id] = parameters
            configured_strategies.append(
                _ConfiguredDecisionStrategy(
                    instance.instance_id,
                    self._registry.create(
                        instance.strategy_type,
                        parameters,
                        instance.instance_id,
                    ),
                    pools[instance.instance_id],
                )
            )

        asset_types = {symbol: item.asset_type for symbol, item in instruments.items()}
        distinct_asset_types = tuple(sorted(set(asset_types.values()), key=lambda item: item.value))
        available = Decimal("1") - cash_reserve
        class_budget = available / len(distinct_asset_types)
        allocation_policy = AllocationPolicy(
            strategy_budgets={item.strategy_instance_id: item.budget for item in selected},
            asset_class_budgets={item: class_budget for item in distinct_asset_types},
            asset_types=asset_types,
            cash_reserve=cash_reserve,
        )
        risk_engine = RiskEngine((cast(RiskRule, MinimumTradeAmountRule()),))
        rule_book = local_rule_book(
            instruments,
            fee_confirmed=True,
            slippage_bps=Decimal("2"),
        )
        result = DecisionService(account_repository).generate_close_decision(
            CloseDecisionRequest(
                account_id=account_repository.account_id,
                decision_at=decision_at,
                valid_until=valid_until,
                data_accepted=cast(bool, bundle.data_quality["accepted"]),
                daily_close_complete=True,
                market_data_source_at=decision_at,
                instruments=instruments,
                bars=bars,
                strategies=configured_strategies,
                allocation_policy=allocation_policy,
                risk_engine=risk_engine,
                rule_book=rule_book,
                risk_metadata={
                    symbol: InstrumentRiskMetadata(minimum_trade_amount=minimum_trade_amount)
                    for symbol in required_symbols
                },
                strategy_pools=pools,
            )
        )
        manifest_provenance = tuple(
            sorted(
                (
                    DecisionManifestProvenance(
                        manifest_id=item.manifest_id,
                        provider=item.provider,
                        content_hash=item.content_hash,
                        relative_data_path=item.relative_data_path,
                    )
                    for item in manifests
                ),
                key=lambda item: item.manifest_id,
            )
        )
        ordered_instances = tuple(sorted(instances, key=lambda item: item.instance_id))
        strategy_provenance = tuple(
            DecisionStrategyProvenance(
                strategy_instance_id=item.instance_id,
                strategy_type=item.strategy_type,
                strategy_version=item.strategy_version,
                parameters=parameters_by_id[item.instance_id].model_dump(mode="json"),
            )
            for item in ordered_instances
        )
        snapshot = RunSnapshot(
            run_id=decision_snapshot_run_id(checked_decision),
            schema_version=1,
            market_manifests=tuple(
                ManifestReference(item.manifest_id, item.content_hash)
                for item in manifest_provenance
            ),
            data_quality=snapshot_data_quality(bundle.data_quality, dataset_provenance),
            strategies=tuple(
                StrategySnapshot(
                    sleeve_id=item.strategy_instance_id,
                    strategy_type=item.strategy_type,
                    strategy_version=item.strategy_version,
                    parameters=item.parameters,
                )
                for item in strategy_provenance
            ),
            instrument_pool={
                "instruments": tuple(str(item) for item in required_symbols),
                "strategy_pools": {
                    strategy_id: tuple(str(item) for item in pools[strategy_id])
                    for strategy_id in sorted(pools)
                },
                "snapshot_ids": {
                    item.instance_id: item.pool_snapshot_id for item in ordered_instances
                },
            },
            survivorship_bias={"mode": "static_pool", "warning": True},
            allocator_configuration=_allocation_payload(allocation_policy),
            risk_configuration=_risk_payload(
                risk_engine,
                (),
                minimum_trade_amount,
            ),
            market_rule_configuration=_market_rule_payload(rule_book),
            fee_profile_configuration=_fee_payload(
                FeeProfileSetting("local-default-v1", True),
                rule_book,
            ),
            app_git_commit=app_git_commit,
            random_seed=0,
        )
        record = DecisionExportRecord(
            checked_decision,
            result,
            manifest_provenance,
            strategy_provenance,
            snapshot,
        )

        watchlist_ids = tuple(sorted({item.watchlist_id for item in instances}))

        def require_live_watchlists(session: Session) -> None:
            for watchlist_id in watchlist_ids:
                self._strategies.acquire_enabled_watchlist_write_lock(
                    session,
                    watchlist_id,
                    disabled_error="DECISION_WATCHLIST_DISABLED",
                )

        self._repository.save(record, before_persist=require_live_watchlists)
        return record

    def _next_session(self, day: date) -> date:
        if self._calendar_coverage is not None:
            try:
                self._calendar_coverage(day + timedelta(days=1), day + timedelta(days=14))
            except Exception:
                raise LookupError("DECISION_CALENDAR_UNAVAILABLE") from None
        candidate = day + timedelta(days=1)
        for _ in range(14):
            if self._trading_calendar(candidate):
                return candidate
            candidate += timedelta(days=1)
        raise LookupError("DECISION_NEXT_SESSION_UNAVAILABLE")
