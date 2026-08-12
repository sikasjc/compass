from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from compass.domain.market import InstrumentId
from compass.data.exchange_calendar import CalendarIdentity
from compass.domain.trading import TargetIntent
from compass.services.intraday_service import (
    CompletionClock,
    IntradayService,
    SnapshotProvider,
    TradingDayCalendar,
)
from compass.services.local_crud_gateways import LocalStrategyGateway, LocalWatchlistGateway
from compass.services.local_market_configuration import local_instruments
from compass.services.local_read_gateways import LocalDashboardGateway
from compass.storage.account_repository import AccountRepository
from compass.storage.dataset_bundle_repository import DatasetBundleRepository
from compass.strategies.base import (
    HoldingSummary,
    Strategy,
    StrategyContext,
)
from compass.strategies.registry import StrategyRegistry
from compass.ui.pages.strategies import (
    StrategyInstance,
    strategy_parameters_json,
)


@dataclass(frozen=True, slots=True)
class _ConfiguredStrategy:
    strategy: Strategy
    instruments: tuple[InstrumentId, ...]


class _CombinedCalculator:
    def __init__(self, configured: Sequence[_ConfiguredStrategy]) -> None:
        self._configured = tuple(configured)

    def __call__(self, context: StrategyContext) -> tuple[TargetIntent, ...]:
        intents: list[TargetIntent] = []
        for configured in self._configured:
            instruments = configured.instruments
            subcontext = StrategyContext(
                as_of=context.as_of,
                bars={instrument: context.history(instrument) for instrument in instruments},
                instruments=instruments,
                account_equity=context.account_equity,
                cash=context.cash,
                holdings={
                    instrument: holding
                    for instrument, holding in context.holdings.items()
                    if instrument in instruments
                },
                asset_types={
                    instrument: context.asset_types[instrument] for instrument in instruments
                },
            )
            decision = configured.strategy.generate_targets(subcontext)
            intents.extend(decision.intents)
        return tuple(intents)


class LocalIntradayRuntime:
    """Build the real intraday service lazily from exact persisted prerequisites."""

    def __init__(
        self,
        *,
        provider: SnapshotProvider | None,
        strategies: LocalStrategyGateway,
        watchlists: LocalWatchlistGateway,
        registry: StrategyRegistry,
        accounts: AccountRepository,
        bundles: DatasetBundleRepository,
        dashboard: LocalDashboardGateway,
        trading_calendar: TradingDayCalendar,
        completion_clock: CompletionClock,
        calendar_identity: Callable[[], CalendarIdentity] | None = None,
    ) -> None:
        self._provider = provider
        self._strategies = strategies
        self._watchlists = watchlists
        self._registry = registry
        self._accounts = accounts
        self._bundles = bundles
        self._dashboard = dashboard
        self._trading_calendar = trading_calendar
        self._completion_clock = completion_clock
        self._calendar_identity = calendar_identity
        self._cache_key: tuple[object, ...] | None = None
        self._service: IntradayService | None = None

    def refresh(self, boundary: datetime) -> None:
        service = self._service_for_current_state(boundary)
        if service is None:
            self._dashboard.clear_intraday()
            return
        self._dashboard.set_intraday(service.refresh(boundary))

    def _service_for_current_state(self, boundary: datetime) -> IntradayService | None:
        provider = self._provider
        bundle = self._bundles.latest()
        account = self._accounts.latest()
        enabled = tuple(item for item in self._strategies.list() if item.enabled)
        if provider is None or bundle is None or account is None or not enabled:
            return None
        manifest_by_instrument = self._bundles.references_by_instrument(bundle)
        configured: list[tuple[StrategyInstance, tuple[InstrumentId, ...]]] = []
        for instance in enabled:
            if not self._watchlists.is_enabled(instance.watchlist_id):
                self._cache_key = None
                self._service = None
                return None
            pool = self._strategies.pool_instruments(instance.instance_id)
            if not set(pool).issubset(manifest_by_instrument):
                return None
            configured.append((instance, pool))
        instruments = tuple(
            sorted(
                {instrument for _, pool in configured for instrument in pool},
                key=str,
            )
        )
        held = {item.instrument for item in account.snapshot.positions}
        if not held.issubset(instruments):
            return None
        identity_source = self._calendar_identity
        if identity_source is not None:
            try:
                current_identity = identity_source()
                if (
                    type(current_identity) is not CalendarIdentity
                    or not current_identity.covered_from
                    <= boundary.date()
                    <= current_identity.covered_to
                ):
                    raise ValueError
                manifests = tuple(
                    self._bundles.load_manifest(
                        manifest_by_instrument[instrument].manifest_id
                    )
                    for instrument in instruments
                )
                if any(
                    manifest.provenance is None
                    or manifest.provenance.calendar != current_identity
                    for manifest in manifests
                ):
                    raise ValueError
            except Exception:
                raise LookupError("INTRADAY_CALENDAR_MISMATCH") from None
        cache_key = (
            bundle.bundle_id,
            tuple(item.instance_id for item, _ in configured),
            account.row_id,
            account.content_hash,
        )
        if self._service is not None and cache_key == self._cache_key:
            return self._service
        daily_history = {
            instrument: self._bundles.read_manifest(manifest_by_instrument[instrument].manifest_id)
            for instrument in instruments
        }
        strategy_values = []
        for instance, pool in configured:
            metadata = self._registry.describe(instance.strategy_type)
            parameters = metadata.parameters_type.model_validate_json(
                strategy_parameters_json(instance.parameters),
                strict=True,
            )
            strategy_values.append(
                _ConfiguredStrategy(
                    strategy=self._registry.create(
                        instance.strategy_type,
                        parameters,
                        instance.instance_id,
                    ),
                    instruments=pool,
                )
            )
        instrument_values = local_instruments(instruments)
        holding_since = self._accounts.holding_since(account.snapshot.as_of)
        holdings = tuple(
            HoldingSummary(
                instrument=item.instrument,
                quantity=item.quantity,
                available_quantity=item.available_quantity,
                average_cost=item.average_cost,
                mark_price=item.mark_price,
                holding_since=holding_since.get(item.instrument, account.snapshot.as_of),
            )
            for item in account.snapshot.positions
        )
        service = IntradayService(
            instruments=instruments,
            provider=provider,
            daily_history=daily_history,
            calculator=_CombinedCalculator(strategy_values),
            is_trading_day=self._trading_calendar,
            account_equity=account.snapshot.equity,
            cash=account.snapshot.cash,
            holdings=holdings,
            asset_types={
                instrument: value.asset_type for instrument, value in instrument_values.items()
            },
            completion_clock=self._completion_clock,
        )
        self._cache_key = cache_key
        self._service = service
        return service
