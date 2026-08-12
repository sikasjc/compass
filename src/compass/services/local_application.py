from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import os
from pathlib import Path
import socket
import subprocess
from threading import RLock
from time import monotonic
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from compass.config import Settings
import pandas as pd  # type: ignore[import-untyped]

from compass.data.base import DailyBarRequest, MarketDataProvider, ProviderError
from compass.data.exchange_calendar import (
    ExchangeCalendarProvider,
    PersistedExchangeCalendar,
)
from compass.data.providers.akshare_provider import AkshareProvider
from compass.data.providers.baostock_provider import BaostockProvider
from compass.data.providers.tencent_provider import TencentProvider
from compass.data.quality import QualityMode
from compass.domain.market import InstrumentId
from compass.services.data_service import DataService, ExpectedSessions
from compass.services.automatic_market_sync import AutomaticMarketSync
from compass.services.diagnostic_log import (
    configure_application_logging,
    set_application_log_level,
)
from compass.services.local_crud_gateways import (
    IdFactory,
    LocalSettingsGateway,
    LocalStrategyGateway,
    LocalWatchlistGateway,
)
from compass.services.local_read_gateways import LocalDataGateway
from compass.services.local_decision_gateway import LocalDecisionGateway
from compass.services.local_signal_center import LocalSignalCenter
from compass.services.local_strategy_lab import LocalStrategyLabGateway
from compass.services.strategy_optimizer import LocalStrategyOptimizer
from compass.services.scheduler import LocalScheduler
from compass.services.task_manager import TaskManager
from compass.storage.database import Database
from compass.storage.account_repository import AccountRepository
from compass.storage.backtest_report_repository import BacktestReportRepository
from compass.storage.decision_export_repository import DecisionExportRepository
from compass.storage.dataset_bundle_repository import DatasetBundleRepository
from compass.storage.market_store import MarketStore
from compass.storage.run_snapshot_repository import RunSnapshotRepository
from compass.storage.signal_account_repository import SignalAccountRepository
from compass.storage.signal_execution_repository import SignalExecutionRepository
from compass.strategies.buy_and_hold import BuyAndHoldStrategy
from compass.strategies.dual_ma import DualMaStrategy
from compass.strategies.etf_rotation import EtfRotationStrategy
from compass.strategies.mean_reversion import MeanReversionStrategy
from compass.strategies.momentum import CrossSectionalMomentumStrategy
from compass.strategies.registry import StrategyFactory, StrategyRegistry
from compass.strategies.rule_dsl import RuleDslStrategy
from compass.ui.pages.data import DataPageModel, DataSourceSnapshot
from compass.ui.pages.logs import LogsPageModel
from compass.ui.pages.settings import (
    ConnectionTestResult,
    ConnectionTestStatus,
    MarketProxyMode,
    MarketProxySetting,
    SettingsPageModel,
)
from compass.ui.pages.signals import SignalPageModel
from compass.ui.pages.strategy_lab import StrategyLabPageModel
from compass.ui.pages.strategies import StrategyPageModel
from compass.ui.pages.watchlists import WatchlistDataRange, WatchlistPageModel


if TYPE_CHECKING:
    from compass.ui.app import AppViewModels


SHANGHAI = ZoneInfo("Asia/Shanghai")
Clock = Callable[[], datetime]
SyncWindow = Callable[[date], tuple[date, date]]
_PROXY_ENVIRONMENT_KEYS = (
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
)


def _now() -> datetime:
    return datetime.now(SHANGHAI)


def _id(kind: str) -> str:
    from uuid import uuid4

    return f"{kind}-{uuid4().hex}"


def _sync_window(today: date) -> tuple[date, date]:
    return today - timedelta(days=550), today


def _app_git_commit() -> str:
    configured = os.environ.get("COMPASS_GIT_COMMIT", "").strip().lower()
    if configured and 7 <= len(configured) <= 64 and all(item in "0123456789abcdef" for item in configured):
        return configured
    try:
        completed = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=Path(__file__).resolve().parents[3],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        resolved = completed.stdout.strip().lower()
        if 7 <= len(resolved) <= 64 and all(item in "0123456789abcdef" for item in resolved):
            return resolved
    except (OSError, subprocess.SubprocessError):
        pass
    return "0000000"


class _MarketProxyEnvironment:
    def __init__(self) -> None:
        self._original = {key: os.environ.get(key) for key in _PROXY_ENVIRONMENT_KEYS}

    def apply(self, setting: MarketProxySetting) -> None:
        if type(setting) is not MarketProxySetting:
            raise TypeError("market proxy must be an exact MarketProxySetting")
        self._restore()
        if setting.mode is MarketProxyMode.SYSTEM:
            return
        if setting.mode is MarketProxyMode.NONE:
            os.environ["NO_PROXY"] = "*"
            os.environ["no_proxy"] = "*"
            for key in _PROXY_ENVIRONMENT_KEYS[:6]:
                os.environ.pop(key, None)
            return
        assert setting.host is not None and setting.port is not None
        host = f"[{setting.host}]" if ":" in setting.host else setting.host
        proxy_url = f"http://{host}:{setting.port}"
        os.environ["HTTP_PROXY"] = proxy_url
        os.environ["http_proxy"] = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url
        os.environ["https_proxy"] = proxy_url
        os.environ.pop("ALL_PROXY", None)
        os.environ.pop("all_proxy", None)

    def _restore(self) -> None:
        for key, value in self._original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def close(self) -> None:
        self._restore()


def _default_providers() -> tuple[MarketDataProvider, ...]:
    created: list[MarketDataProvider] = []
    try:
        created.append(TencentProvider())
        created.append(AkshareProvider())
        created.append(BaostockProvider())
    except Exception:
        for provider in reversed(created):
            close = getattr(provider, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        raise
    return tuple(created)


def _provider_configuration(
    providers: Sequence[MarketDataProvider],
) -> tuple[tuple[str, str, bool, bool | None], ...]:
    display_names = {
        "akshare": "东方财富",
        "tencent": "腾讯证券",
        "baostock": "BaoStock",
    }
    return tuple(
        (provider.name, display_names.get(provider.name, provider.name), True, None)
        for provider in providers
    )


def _configure_market_request_timeout(
    providers: Sequence[MarketDataProvider],
    seconds: int,
) -> None:
    for provider in providers:
        setter = getattr(provider, "set_request_timeout", None)
        if callable(setter):
            setter(seconds)


def _elapsed_ms(started_at: float) -> int:
    return max(0, round((monotonic() - started_at) * 1000))


def _test_proxy_connection(setting: MarketProxySetting) -> ConnectionTestResult:
    if setting.mode is MarketProxyMode.NONE:
        return ConnectionTestResult(
            "proxy",
            "代理连接",
            ConnectionTestStatus.SKIPPED,
            detail_code="PROXY_DISABLED",
        )
    if setting.mode is MarketProxyMode.CUSTOM:
        assert setting.host is not None and setting.port is not None
        host, port = setting.host, setting.port
    else:
        raw_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
        if not raw_url:
            return ConnectionTestResult(
                "proxy",
                "系统代理",
                ConnectionTestStatus.SKIPPED,
                detail_code="SYSTEM_PROXY_NOT_CONFIGURED",
            )
        parsed = urlsplit(raw_url)
        if parsed.hostname is None:
            return ConnectionTestResult(
                "proxy",
                "系统代理",
                ConnectionTestStatus.FAILED,
                detail_code="SYSTEM_PROXY_INVALID",
            )
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    started_at = monotonic()
    try:
        connection = socket.create_connection((host, port), timeout=3.0)
        connection.close()
    except OSError:
        return ConnectionTestResult(
            "proxy",
            "代理连接",
            ConnectionTestStatus.FAILED,
            _elapsed_ms(started_at),
            "PROXY_NETWORK",
        )
    return ConnectionTestResult(
        "proxy",
        "代理连接",
        ConnectionTestStatus.SUCCEEDED,
        _elapsed_ms(started_at),
    )


def _test_provider_connection(
    provider: MarketDataProvider,
    *,
    clock: Clock,
) -> ConnectionTestResult:
    display_names = {
        "akshare": "东方财富",
        "tencent": "腾讯证券",
        "baostock": "BaoStock",
    }
    display_name = display_names.get(provider.name, provider.name)
    end = clock().date()
    request = DailyBarRequest(
        InstrumentId.parse("SSE.000300"),
        end - timedelta(days=14),
        end,
    )
    started_at = monotonic()
    try:
        frame = provider.fetch_daily(request)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return ConnectionTestResult(
                provider.name,
                display_name,
                ConnectionTestStatus.FAILED,
                _elapsed_ms(started_at),
                "PROVIDER_EMPTY_RESPONSE",
            )
    except ProviderError as error:
        return ConnectionTestResult(
            provider.name,
            display_name,
            ConnectionTestStatus.FAILED,
            _elapsed_ms(started_at),
            f"PROVIDER_{error.kind.value.upper()}",
        )
    except Exception:
        return ConnectionTestResult(
            provider.name,
            display_name,
            ConnectionTestStatus.FAILED,
            _elapsed_ms(started_at),
            "CONNECTION_TEST_FAILED",
        )
    return ConnectionTestResult(
        provider.name,
        display_name,
        ConnectionTestStatus.SUCCEEDED,
        _elapsed_ms(started_at),
    )


def _test_market_connections(
    providers: Sequence[MarketDataProvider],
    setting: MarketProxySetting,
    *,
    clock: Clock,
) -> tuple[ConnectionTestResult, ...]:
    return (
        _test_proxy_connection(setting),
        *(_test_provider_connection(provider, clock=clock) for provider in providers),
    )


@dataclass(slots=True)
class LocalApplication:
    settings: Settings
    models: AppViewModels
    database: Database
    market_store: MarketStore
    task_manager: TaskManager
    scheduler: LocalScheduler
    automatic_market_sync: AutomaticMarketSync
    watchlists: LocalWatchlistGateway
    strategies: LocalStrategyGateway
    data_gateway: LocalDataGateway
    strategy_lab_gateway: LocalStrategyLabGateway
    strategy_optimizer: LocalStrategyOptimizer
    accounts: AccountRepository
    signal_accounts: SignalAccountRepository
    signal_executions: SignalExecutionRepository
    decisions: LocalDecisionGateway
    signal_center: LocalSignalCenter
    settings_gateway: LocalSettingsGateway
    proxy_environment: _MarketProxyEnvironment
    owned_providers: tuple[object, ...]
    _closed: bool = field(default=False, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def start(self) -> None:
        self.automatic_market_sync.start()

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.automatic_market_sync.stop()
        self.task_manager.shutdown()
        for provider in reversed(self.owned_providers):
            close = getattr(provider, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        self.proxy_environment.close()
        self.database.engine.dispose()


def build_local_application(
    settings: Settings,
    *,
    providers: Sequence[MarketDataProvider] | None = None,
    clock: Clock = _now,
    task_manager: TaskManager | None = None,
    executor: Executor | None = None,
    id_factory: IdFactory = _id,
    expected_sessions: ExpectedSessions | None = None,
    sync_window: SyncWindow = _sync_window,
    sync_quality_mode: QualityMode = QualityMode.DEGRADED,
) -> LocalApplication:
    if type(settings) is not Settings:
        raise TypeError("settings must be an exact Settings value")
    owned_providers: tuple[object, ...] = ()
    configured_providers: tuple[MarketDataProvider, ...]
    if providers is None:
        configured_providers = _default_providers()
        owned_providers = tuple(configured_providers)
    else:
        configured_providers = tuple(providers)
    if not configured_providers:
        raise ValueError("at least one market data provider is required")

    database: Database | None = None
    tasks: TaskManager | None = None
    proxy_environment: _MarketProxyEnvironment | None = None
    try:
        database = Database(settings)
        database.create_schema()
        log_path = configure_application_logging(settings.logs_dir, "INFO")
        market_store = MarketStore(settings.market_data_dir, database)
        exchange_calendar = PersistedExchangeCalendar(
            settings.market_data_dir / "exchange_calendar.json",
            providers=cast(
                tuple[ExchangeCalendarProvider, ...],
                tuple(
                    provider
                    for provider in configured_providers
                    if callable(getattr(provider, "fetch_exchange_sessions", None))
                ),
            ),
        )
        resolved_expected_sessions = (
            exchange_calendar.expected_sessions if expected_sessions is None else expected_sessions
        )
        tasks = task_manager or TaskManager(executor=executor, clock=clock)
        watchlists = LocalWatchlistGateway(database, clock=clock, id_factory=id_factory)
        strategies = LocalStrategyGateway(
            database,
            watchlists,
            clock=clock,
            id_factory=id_factory,
        )
        strategy_registry = StrategyRegistry()
        strategy_registry.register(
            "buy_and_hold", cast(StrategyFactory, BuyAndHoldStrategy)
        )
        strategy_registry.register(
            "cross_sectional_momentum",
            cast(StrategyFactory, CrossSectionalMomentumStrategy),
        )
        strategy_registry.register("dual_ma", cast(StrategyFactory, DualMaStrategy))
        strategy_registry.register(
            "etf_rotation", cast(StrategyFactory, EtfRotationStrategy)
        )
        strategy_registry.register(
            "mean_reversion", cast(StrategyFactory, MeanReversionStrategy)
        )
        strategy_registry.register("rule_dsl", cast(StrategyFactory, RuleDslStrategy))
        provider_configuration = _provider_configuration(configured_providers)
        proxy_environment = _MarketProxyEnvironment()
        settings_gateway = LocalSettingsGateway(
            database,
            clock=clock,
            providers=provider_configuration,
            proxy_configurer=proxy_environment.apply,
            connection_tester=lambda setting: _test_market_connections(
                configured_providers,
                setting,
                clock=clock,
            ),
            log_path=log_path,
            log_level_configurer=set_application_log_level,
            request_timeout_configurer=lambda seconds: _configure_market_request_timeout(
                configured_providers,
                seconds,
            ),
        )
        set_application_log_level(settings_gateway.state().log_level)
        bundles = DatasetBundleRepository(
            database,
            market_store,
            clock=clock,
            id_factory=id_factory,
        )
        snapshots = RunSnapshotRepository(database, market_store, clock=clock)
        source_views = tuple(
            DataSourceSnapshot(
                provider=provider_id,
                source_name=display_name,
                available=available,
                last_update=None,
                latest_manifest_id=None,
                latest_source=None,
                quality=None,
                cache_bytes=0,
            )
            for provider_id, display_name, available, _ in provider_configuration
        )
        data_gateway = LocalDataGateway(
            source_views,
            providers=configured_providers,
            service=DataService(
                market_store,
                expected_sessions=resolved_expected_sessions,
                calendar_identity=(
                    (lambda request: exchange_calendar.identity)
                    if expected_sessions is None
                    else None
                ),
                completed_session=lambda request: request.end,
                clock=clock,
                require_corporate_actions=True,
                require_rule_attestation=providers is None,
            ),
            bundles=bundles,
            watchlists=watchlists,
            settings=settings_gateway,
            sync_window=sync_window,
            clock=clock,
            refresh_calendar=(
                exchange_calendar.ensure_coverage if expected_sessions is None else None
            ),
            latest_completed_session=(
                exchange_calendar.latest_completed_session if expected_sessions is None else None
            ),
            quality_mode=sync_quality_mode,
            protected_manifest_ids=snapshots.referenced_manifest_ids,
        )
        reports = BacktestReportRepository(database, snapshots, clock=clock)
        accounts = AccountRepository(database, "main", clock)
        signal_accounts = SignalAccountRepository(
            settings.root / "data" / "signal_accounts.json"
        )
        signal_executions = SignalExecutionRepository(
            settings.root / "data" / "signal_executions.json"
        )
        decision_repository = DecisionExportRepository(
            database,
            market_store,
            snapshots,
            clock=clock,
        )
        decisions = LocalDecisionGateway(
            accounts=accounts,
            strategies=strategies,
            registry=strategy_registry,
            bundles=bundles,
            settings=settings_gateway,
            repository=decision_repository,
            app_git_commit=_app_git_commit(),
            clock=clock,
            trading_calendar=exchange_calendar.is_session,
            calendar_coverage=(
                exchange_calendar.ensure_coverage if expected_sessions is None else None
            ),
        )
        signal_center = LocalSignalCenter(
            accounts=accounts,
            bundles=bundles,
            strategies=strategies,
            decisions=decisions,
            id_factory=id_factory,
            account_profiles=signal_accounts,
            executions=signal_executions,
            account_factory=lambda account_id: AccountRepository(
                database,
                account_id,
                clock,
            ),
        )
        strategy_lab_gateway = LocalStrategyLabGateway(
            bundles=bundles,
            reports=reports,
            strategies=strategies,
            app_git_commit=_app_git_commit(),
            id_factory=id_factory,
        )
        strategy_optimizer = LocalStrategyOptimizer(
            settings.root / "data" / "strategy_optimizations.json",
            strategies=strategies,
            backtests=strategy_lab_gateway,
            clock=clock,
            id_factory=id_factory,
        )

        from compass.ui.app import AppViewModels
        from compass.ui.pages.account_overview import AccountOverviewPageModel

        data_model = DataPageModel(data_gateway, tasks)
        scheduler = LocalScheduler(clock=clock)
        automatic_market_sync = AutomaticMarketSync(
            settings=settings_gateway,
            data=data_model,
            scheduler=scheduler,
            clock=clock,
            latest_completed_session=(
                exchange_calendar.latest_completed_session
                if expected_sessions is None
                else None
            ),
        )

        def retry_instrument(instrument: InstrumentId) -> object:
            provider = next(
                item.provider for item in settings_gateway.state().providers if item.available
            )
            return data_model.start_sync(provider, instruments=(instrument,))

        models = AppViewModels(
            watchlists=WatchlistPageModel(
                watchlists,
                lambda: tuple(
                    WatchlistDataRange(
                        preview.instrument,
                        date.fromisoformat(preview.first_day),
                        date.fromisoformat(preview.last_day),
                    )
                    for preview in data_gateway.latest_market_previews()
                ),
                retry_instrument,
            ),
            data=data_model,
            strategies=StrategyPageModel(
                strategy_registry,
                strategies,
                tasks,
                optimizer=strategy_optimizer,
            ),
            backtests=StrategyLabPageModel(strategy_lab_gateway, tasks),
            account=AccountOverviewPageModel(signal_center),
            signals=SignalPageModel(signal_center),
            settings=SettingsPageModel(settings_gateway, settings.root),
            logs=LogsPageModel(settings_gateway),
            task_manager=tasks,
        )
        return LocalApplication(
            settings=settings,
            models=models,
            database=database,
            market_store=market_store,
            task_manager=tasks,
            scheduler=scheduler,
            automatic_market_sync=automatic_market_sync,
            watchlists=watchlists,
            strategies=strategies,
            data_gateway=data_gateway,
            strategy_lab_gateway=strategy_lab_gateway,
            strategy_optimizer=strategy_optimizer,
            accounts=accounts,
            signal_accounts=signal_accounts,
            signal_executions=signal_executions,
            decisions=decisions,
            signal_center=signal_center,
            settings_gateway=settings_gateway,
            proxy_environment=proxy_environment,
            owned_providers=owned_providers,
        )
    except Exception:
        if tasks is not None:
            tasks.shutdown()
        for provider in reversed(owned_providers):
            close = getattr(provider, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        if database is not None:
            database.engine.dispose()
        if proxy_environment is not None:
            proxy_environment.close()
        raise
