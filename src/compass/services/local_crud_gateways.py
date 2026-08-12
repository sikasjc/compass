from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import cast

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from compass.data.base import default_instrument_type
from compass.data.network_timeout import (
    DEFAULT_MARKET_TIMEOUT_SECONDS,
    validate_market_timeout_seconds,
)
from compass.domain.market import AssetType, InstrumentId
from compass.services.diagnostic_log import (
    DiagnosticLogEntry,
    read_application_logs,
)
from compass.services.safe_display import safe_identifier
from compass.storage.canonical_json import (
    canonical_json,
    content_hash,
    decode_canonical_json,
)
from compass.storage.database import Database
from compass.storage.models import (
    LocalSettingsRecord,
    LocalStrategyVersionRecord,
    LocalWatchlistRecord,
)
from compass.strategies.base import StrategyFrequency
from compass.ui.pages.settings import (
    AUTOMATIC_SYNC_INTERVALS,
    ConnectionTestResult,
    MarketProxyMode,
    MarketProxySetting,
    ProviderSetting,
    SettingsSnapshot,
)
from compass.ui.pages.strategies import (
    StrategyDraft,
    StrategyInstance,
    StrategyPool,
    StrategyPoolChoice,
)
from compass.ui.pages.watchlists import (
    WatchlistDraft,
    WatchlistEntry,
)


Clock = Callable[[], datetime]
IdFactory = Callable[[str], str]
_SCHEMA_VERSION = 1
_SETTINGS_ID = "application"
_FEE_PROFILE_ID = "standard-a-share-v1"
_RISK_TEMPLATE_ID = "balanced-etf-v1"


def _timestamp(clock: Clock) -> datetime:
    value = clock()
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("local application clock must return a timezone-aware datetime")
    return value


def _new_id(factory: IdFactory, kind: str) -> str:
    return safe_identifier(factory(kind), label=f"{kind} id")


def _parameters_json(value: object) -> object:
    if value is None or type(value) in {str, bool, int, float}:
        return value
    if isinstance(value, Enum):
        return _parameters_json(value.value)
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("strategy parameter keys must be exact strings")
            result[key] = _parameters_json(item)
        return dict(sorted(result.items()))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_parameters_json(item) for item in value]
    raise TypeError("strategy parameters contain an unsupported value")


class LocalWatchlistGateway:
    def __init__(self, database: Database, *, clock: Clock, id_factory: IdFactory) -> None:
        self._database = database
        self._clock = clock
        self._id_factory = id_factory

    def list(self) -> tuple[WatchlistEntry, ...]:
        with self._database.session_factory() as session:
            rows = session.scalars(
                select(LocalWatchlistRecord).order_by(LocalWatchlistRecord.watchlist_id)
            ).all()
            return tuple(self._entry(row) for row in rows)

    def primary(self) -> WatchlistEntry | None:
        entries = self.list()
        selected = tuple(item for item in entries if item.enabled) or entries[:1]
        if not selected:
            return None
        instruments = tuple(
            sorted(
                {instrument for item in selected for instrument in item.instruments},
                key=str,
            )
        )
        return WatchlistEntry(selected[0].watchlist_id, "关注标的", instruments, True)

    def save_primary(self, draft: WatchlistDraft) -> None:
        if type(draft) is not WatchlistDraft:
            raise TypeError("draft must be an exact WatchlistDraft")
        current = self.primary()
        if current is None:
            self.create(draft)
            return
        self.update(current.watchlist_id, draft)
        for entry in self.list():
            if entry.watchlist_id != current.watchlist_id and entry.enabled:
                self.disable(entry.watchlist_id)

    def create(self, draft: WatchlistDraft) -> None:
        if type(draft) is not WatchlistDraft:
            raise TypeError("draft must be an exact WatchlistDraft")
        watchlist_id = _new_id(self._id_factory, "watchlist")
        self._write_new(watchlist_id, draft.name, draft.instruments, True)

    def update(self, watchlist_id: str, draft: WatchlistDraft) -> None:
        checked = safe_identifier(watchlist_id, label="watchlist id")
        if type(draft) is not WatchlistDraft:
            raise TypeError("draft must be an exact WatchlistDraft")
        payload = self._payload(draft.name, draft.instruments, self._enabled(checked))
        payload_json = canonical_json(payload)
        with self._database.session_factory.begin() as session:
            row = session.get(LocalWatchlistRecord, checked)
            if row is None:
                raise LookupError("WATCHLIST_UNKNOWN")
            row.payload_json = payload_json
            row.content_hash = content_hash(payload_json)
            row.updated_at = _timestamp(self._clock)

    def copy(self, watchlist_id: str) -> None:
        source = self._get(watchlist_id)
        copy_draft = WatchlistDraft(
            f"{source.name} 副本",
            source.instruments,
        )
        self._write_new(
            _new_id(self._id_factory, "watchlist"),
            copy_draft.name,
            copy_draft.instruments,
            True,
        )

    def disable(self, watchlist_id: str) -> None:
        source = self._get(watchlist_id)
        payload_json = canonical_json(self._payload(source.name, source.instruments, False))
        with self._database.session_factory.begin() as session:
            row = session.get(LocalWatchlistRecord, source.watchlist_id)
            if row is None:
                raise LookupError("WATCHLIST_UNKNOWN")
            row.payload_json = payload_json
            row.content_hash = content_hash(payload_json)
            row.updated_at = _timestamp(self._clock)

    def content_hash(self, watchlist_id: str) -> str:
        checked = safe_identifier(watchlist_id, label="watchlist id")
        with self._database.session_factory() as session:
            row = session.get(LocalWatchlistRecord, checked)
            if row is None:
                raise LookupError("WATCHLIST_UNKNOWN")
            self._entry(row)
            return row.content_hash

    def is_enabled(self, watchlist_id: str) -> bool:
        return self._get(watchlist_id).enabled

    def acquire_enabled_write_lock(
        self,
        session: Session,
        watchlist_id: str,
        *,
        disabled_error: str = "WATCHLIST_DISABLED",
    ) -> None:
        """Atomically require a live watchlist before a dependent local write."""

        checked = safe_identifier(watchlist_id, label="watchlist id")
        session.execute(
            update(LocalWatchlistRecord)
            .where(LocalWatchlistRecord.watchlist_id == checked)
            .values(updated_at=LocalWatchlistRecord.updated_at)
        )
        row = session.get(LocalWatchlistRecord, checked)
        if row is None:
            raise LookupError("WATCHLIST_UNKNOWN")
        entry = self._entry(row)
        if not entry.enabled:
            raise LookupError(disabled_error)

    def _get(self, watchlist_id: str) -> WatchlistEntry:
        checked = safe_identifier(watchlist_id, label="watchlist id")
        with self._database.session_factory() as session:
            row = session.get(LocalWatchlistRecord, checked)
            if row is None:
                raise LookupError("WATCHLIST_UNKNOWN")
            return self._entry(row)

    def _enabled(self, watchlist_id: str) -> bool:
        return self._get(watchlist_id).enabled

    def _write_new(
        self,
        watchlist_id: str,
        name: str,
        instruments: Sequence[InstrumentId],
        enabled: bool,
    ) -> None:
        payload_json = canonical_json(self._payload(name, instruments, enabled))
        with self._database.session_factory.begin() as session:
            if session.get(LocalWatchlistRecord, watchlist_id) is not None:
                raise ValueError("WATCHLIST_ID_CONFLICT")
            session.add(
                LocalWatchlistRecord(
                    watchlist_id=watchlist_id,
                    schema_version=_SCHEMA_VERSION,
                    payload_json=payload_json,
                    content_hash=content_hash(payload_json),
                    updated_at=_timestamp(self._clock),
                )
            )

    @staticmethod
    def _payload(
        name: str,
        instruments: Sequence[InstrumentId],
        enabled: bool,
    ) -> dict[str, object]:
        return {
            "enabled": enabled,
            "instruments": [str(item) for item in instruments],
            "name": name,
            "schema_version": _SCHEMA_VERSION,
        }

    @staticmethod
    def _entry(row: LocalWatchlistRecord) -> WatchlistEntry:
        decoded = decode_canonical_json(row.payload_json, row.content_hash)
        if (
            type(row.schema_version) is not int
            or row.schema_version != _SCHEMA_VERSION
            or set(decoded) != {"enabled", "instruments", "name", "schema_version"}
            or type(decoded["schema_version"]) is not int
            or decoded["schema_version"] != _SCHEMA_VERSION
            or type(decoded["name"]) is not str
            or type(decoded["enabled"]) is not bool
            or type(decoded["instruments"]) is not list
            or any(type(item) is not str for item in cast(list[object], decoded["instruments"]))
        ):
            raise ValueError("WATCHLIST_INTEGRITY")
        instruments = tuple(
            InstrumentId.parse(cast(str, item))
            for item in cast(list[object], decoded["instruments"])
        )
        return WatchlistEntry(
            row.watchlist_id,
            decoded["name"],
            instruments,
            decoded["enabled"],
        )


class LocalStrategyGateway:
    def __init__(
        self,
        database: Database,
        watchlists: LocalWatchlistGateway,
        *,
        clock: Clock,
        id_factory: IdFactory,
    ) -> None:
        self._database = database
        self._watchlists = watchlists
        self._clock = clock
        self._id_factory = id_factory

    def list(self) -> tuple[StrategyInstance, ...]:
        with self._database.session_factory() as session:
            rows = session.scalars(
                select(LocalStrategyVersionRecord).order_by(LocalStrategyVersionRecord.instance_id)
            ).all()
            return tuple(self._instance(row) for row in rows)

    def pools(self) -> tuple[StrategyPoolChoice, ...]:
        choices = []
        for entry in self._watchlists.list():
            if not entry.enabled:
                continue
            pool = self.pool(entry.watchlist_id)
            choices.append(
                StrategyPoolChoice(
                    entry.watchlist_id,
                    entry.name,
                    pool.instruments,
                    pool.asset_type,
                    pool.frequency,
                )
            )
        return tuple(sorted(choices, key=lambda item: item.watchlist_id))

    def pool(self, watchlist_id: str) -> StrategyPool:
        entry = self._watchlists._get(watchlist_id)
        if not entry.enabled:
            raise LookupError("WATCHLIST_DISABLED")
        instruments = tuple(
            item
            for item in entry.instruments
            if default_instrument_type(item) in {AssetType.ETF, AssetType.STOCK}
        )
        if not instruments:
            raise LookupError("STRATEGY_POOL_HAS_NO_TRADABLE_INSTRUMENTS")
        asset_types = {default_instrument_type(item) for item in instruments}
        if len(asset_types) != 1:
            raise ValueError("WATCHLIST_ASSET_TYPES_MIXED")
        digest = self._watchlists.content_hash(watchlist_id)
        return StrategyPool(
            watchlist_id=entry.watchlist_id,
            snapshot_id=f"{entry.watchlist_id}-{digest[:16]}",
            instruments=instruments,
            asset_type=next(iter(asset_types)),
            frequency=StrategyFrequency.DAILY,
        )

    def create(self, draft: StrategyDraft) -> StrategyInstance:
        if type(draft) is not StrategyDraft:
            raise TypeError("draft must be an exact StrategyDraft")
        pool = self.pool(draft.watchlist_id)
        self._check_pool(draft, pool)
        lineage_id = _new_id(self._id_factory, "strategy")
        return self._insert(lineage_id, 1, draft, pool.instruments)

    def copy(self, instance_id: str) -> StrategyInstance:
        source, instruments = self._load(instance_id)
        if not self.is_watchlist_enabled(source.watchlist_id):
            raise LookupError("WATCHLIST_DISABLED")
        lineage_id = _new_id(self._id_factory, "strategy")
        draft = StrategyDraft(
            name=f"{source.name} 副本",
            strategy_type=source.strategy_type,
            strategy_version=source.strategy_version,
            watchlist_id=source.watchlist_id,
            pool_snapshot_id=source.pool_snapshot_id,
            frequency=source.frequency,
            parameters=source.parameters,
        )
        return self._insert(lineage_id, 1, draft, instruments)

    def create_version(
        self,
        instance_id: str,
        draft: StrategyDraft,
    ) -> StrategyInstance:
        source, _ = self._load(instance_id)
        if not self.is_watchlist_enabled(source.watchlist_id):
            raise LookupError("WATCHLIST_DISABLED")
        if type(draft) is not StrategyDraft:
            raise TypeError("draft must be an exact StrategyDraft")
        pool = self.pool(draft.watchlist_id)
        self._check_pool(draft, pool)
        with self._database.session_factory() as session:
            lineage_rows = session.scalars(
                select(LocalStrategyVersionRecord)
                .where(LocalStrategyVersionRecord.lineage_id == source.lineage_id)
                .order_by(LocalStrategyVersionRecord.version)
            ).all()
            if not lineage_rows or lineage_rows[-1].instance_id != source.instance_id:
                raise ValueError("STRATEGY_VERSION_SOURCE_NOT_LATEST")
            next_version = lineage_rows[-1].version + 1
        return self._insert(
            source.lineage_id,
            next_version,
            draft,
            pool.instruments,
            disable_lineage=True,
            required_live_watchlist_ids=(source.watchlist_id,),
        )

    def disable(self, instance_id: str) -> None:
        checked = safe_identifier(instance_id, label="strategy instance id")
        with self._database.session_factory.begin() as session:
            row = session.get(LocalStrategyVersionRecord, checked)
            if row is None:
                raise LookupError("STRATEGY_UNKNOWN")
            row.enabled = 0

    def delete(self, instance_id: str) -> bool:
        checked = safe_identifier(instance_id, label="strategy instance id")
        with self._database.session_factory.begin() as session:
            source = session.get(LocalStrategyVersionRecord, checked)
            if source is None:
                return False
            rows = session.scalars(
                select(LocalStrategyVersionRecord).where(
                    LocalStrategyVersionRecord.lineage_id == source.lineage_id
                )
            ).all()
            for row in rows:
                session.delete(row)
            return True

    def quick_backtest(self, instance_id: str) -> None:
        self._load(instance_id)
        raise LookupError("BACKTEST_DATA_BUNDLE_MISSING")

    def pool_instruments(self, instance_id: str) -> tuple[InstrumentId, ...]:
        return self._load(instance_id)[1]

    def is_watchlist_enabled(self, watchlist_id: str) -> bool:
        return self._watchlists.is_enabled(watchlist_id)

    def acquire_enabled_watchlist_write_lock(
        self,
        session: Session,
        watchlist_id: str,
        *,
        disabled_error: str = "WATCHLIST_DISABLED",
    ) -> None:
        self._watchlists.acquire_enabled_write_lock(
            session,
            watchlist_id,
            disabled_error=disabled_error,
        )

    def _insert(
        self,
        lineage_id: str,
        version: int,
        draft: StrategyDraft,
        pool_instruments: Sequence[InstrumentId],
        *,
        disable_lineage: bool = False,
        required_live_watchlist_ids: Sequence[str] = (),
    ) -> StrategyInstance:
        instance_id = f"{lineage_id}-v{version}"
        payload = {
            "frequency": draft.frequency.value,
            "name": draft.name,
            "parameters": _parameters_json(draft.parameters),
            "pool_instruments": [str(item) for item in pool_instruments],
            "pool_snapshot_id": draft.pool_snapshot_id,
            "schema_version": _SCHEMA_VERSION,
            "strategy_type": draft.strategy_type,
            "strategy_version": draft.strategy_version,
            "watchlist_id": draft.watchlist_id,
        }
        payload_json = canonical_json(payload)
        created_at = _timestamp(self._clock)
        with self._database.session_factory.begin() as session:
            for watchlist_id in sorted({draft.watchlist_id, *required_live_watchlist_ids}):
                self.acquire_enabled_watchlist_write_lock(session, watchlist_id)
            if session.get(LocalStrategyVersionRecord, instance_id) is not None:
                raise ValueError("STRATEGY_ID_CONFLICT")
            if disable_lineage:
                rows = session.scalars(
                    select(LocalStrategyVersionRecord).where(
                        LocalStrategyVersionRecord.lineage_id == lineage_id
                    )
                ).all()
                for row in rows:
                    row.enabled = 0
            session.add(
                LocalStrategyVersionRecord(
                    instance_id=instance_id,
                    lineage_id=lineage_id,
                    version=version,
                    enabled=1,
                    schema_version=_SCHEMA_VERSION,
                    payload_json=payload_json,
                    content_hash=content_hash(payload_json),
                    created_at=created_at,
                )
            )
        return StrategyInstance(
            instance_id=instance_id,
            lineage_id=lineage_id,
            version=version,
            name=draft.name,
            strategy_type=draft.strategy_type,
            strategy_version=draft.strategy_version,
            watchlist_id=draft.watchlist_id,
            pool_snapshot_id=draft.pool_snapshot_id,
            frequency=draft.frequency,
            parameters=draft.parameters,
            enabled=True,
            created_at=created_at,
        )

    def _load(
        self,
        instance_id: str,
    ) -> tuple[StrategyInstance, tuple[InstrumentId, ...]]:
        checked = safe_identifier(instance_id, label="strategy instance id")
        with self._database.session_factory() as session:
            row = session.get(LocalStrategyVersionRecord, checked)
            if row is None:
                raise LookupError("STRATEGY_UNKNOWN")
            return self._instance(row), self._pool_instruments(row)

    @staticmethod
    def _check_pool(draft: StrategyDraft, pool: StrategyPool) -> None:
        if (
            draft.watchlist_id != pool.watchlist_id
            or draft.pool_snapshot_id != pool.snapshot_id
            or draft.frequency is not pool.frequency
        ):
            raise ValueError("STRATEGY_POOL_CHANGED")

    @staticmethod
    def _decoded(row: LocalStrategyVersionRecord) -> dict[str, object]:
        decoded = decode_canonical_json(row.payload_json, row.content_hash)
        expected = {
            "frequency",
            "name",
            "parameters",
            "pool_instruments",
            "pool_snapshot_id",
            "schema_version",
            "strategy_type",
            "strategy_version",
            "watchlist_id",
        }
        if (
            type(row.schema_version) is not int
            or row.schema_version != _SCHEMA_VERSION
            or set(decoded) != expected
            or type(decoded["schema_version"]) is not int
            or decoded["schema_version"] != _SCHEMA_VERSION
            or type(row.enabled) is not int
            or row.enabled not in {0, 1}
        ):
            raise ValueError("STRATEGY_INTEGRITY")
        return decoded

    @classmethod
    def _instance(cls, row: LocalStrategyVersionRecord) -> StrategyInstance:
        decoded = cls._decoded(row)
        parameters = decoded["parameters"]
        if not isinstance(parameters, Mapping):
            raise ValueError("STRATEGY_INTEGRITY")
        try:
            frequency = StrategyFrequency(cast(str, decoded["frequency"]))
            return StrategyInstance(
                instance_id=row.instance_id,
                lineage_id=row.lineage_id,
                version=row.version,
                name=cast(str, decoded["name"]),
                strategy_type=cast(str, decoded["strategy_type"]),
                strategy_version=cast(str, decoded["strategy_version"]),
                watchlist_id=cast(str, decoded["watchlist_id"]),
                pool_snapshot_id=cast(str, decoded["pool_snapshot_id"]),
                frequency=frequency,
                parameters=cast(Mapping[str, object], parameters),
                enabled=bool(row.enabled),
                created_at=row.created_at,
            )
        except (TypeError, ValueError):
            raise ValueError("STRATEGY_INTEGRITY") from None

    @classmethod
    def _pool_instruments(
        cls,
        row: LocalStrategyVersionRecord,
    ) -> tuple[InstrumentId, ...]:
        decoded = cls._decoded(row)
        raw = decoded["pool_instruments"]
        if type(raw) is not list or any(type(item) is not str for item in raw):
            raise ValueError("STRATEGY_INTEGRITY")
        try:
            return tuple(InstrumentId.parse(cast(str, item)) for item in raw)
        except (TypeError, ValueError):
            raise ValueError("STRATEGY_INTEGRITY") from None


class LocalSettingsGateway:
    def __init__(
        self,
        database: Database,
        *,
        clock: Clock,
        providers: Sequence[tuple[str, str, bool, bool | None]],
        proxy_configurer: Callable[[MarketProxySetting], None] | None = None,
        connection_tester: (
            Callable[[MarketProxySetting], Sequence[ConnectionTestResult]] | None
        ) = None,
        log_path: Path | None = None,
        log_level_configurer: Callable[[str], None] | None = None,
        request_timeout_configurer: Callable[[int], None] | None = None,
    ) -> None:
        self._database = database
        self._clock = clock
        self._providers = tuple(providers)
        self._proxy_configurer = proxy_configurer
        self._connection_tester = connection_tester
        self._log_path = log_path
        self._log_level_configurer = log_level_configurer
        self._request_timeout_configurer = request_timeout_configurer
        self._ensure()
        payload = self._read()
        self._apply_market_proxy(self._proxy_from_payload(payload))
        if self._request_timeout_configurer is not None:
            self._request_timeout_configurer(
                cast(int, payload["market_request_timeout_seconds"])
            )

    def state(self) -> SettingsSnapshot:
        payload = self._read()
        provider_order = tuple(item[0] for item in self._providers)
        by_id = {item[0]: item for item in self._providers}
        providers = tuple(
            ProviderSetting(
                provider=provider_id,
                display_name=by_id[provider_id][1],
                available=by_id[provider_id][2],
                priority=index,
                credential_present=by_id[provider_id][3],
            )
            for index, provider_id in enumerate(provider_order)
        )
        return SettingsSnapshot(
            providers=providers,
            log_level=cast(str, payload["log_level"]),
            market_request_timeout_seconds=cast(
                int,
                payload["market_request_timeout_seconds"],
            ),
            market_proxy=self._proxy_from_payload(payload),
            automatic_sync_on_startup=cast(
                bool,
                payload["automatic_sync_on_startup"],
            ),
            automatic_sync_interval_minutes=cast(
                int | None,
                payload["automatic_sync_interval_minutes"],
            ),
            automatic_sync_after_close=cast(
                bool,
                payload["automatic_sync_after_close"],
            ),
        )

    def confirm_fee_profile(self, profile_id: str) -> None:
        if profile_id != _FEE_PROFILE_ID:
            raise ValueError("FEE_PROFILE_UNKNOWN")
        self._mutate(fee_confirmed=True)

    def select_risk_template(self, template_id: str) -> None:
        if template_id != _RISK_TEMPLATE_ID:
            raise ValueError("RISK_TEMPLATE_UNKNOWN")
        self._mutate(active_risk_template=template_id)

    def set_log_level(self, level: str) -> None:
        self._mutate(log_level=level)
        if self._log_level_configurer is not None:
            self._log_level_configurer(level)

    def set_market_request_timeout(self, seconds: int) -> None:
        checked = validate_market_timeout_seconds(seconds)
        self._mutate(market_request_timeout_seconds=checked)
        if self._request_timeout_configurer is not None:
            self._request_timeout_configurer(checked)

    def set_market_proxy(self, setting: MarketProxySetting) -> None:
        if type(setting) is not MarketProxySetting:
            raise TypeError("market proxy must be an exact MarketProxySetting")
        self._mutate(
            proxy_mode=setting.mode.value,
            proxy_host=setting.host,
            proxy_port=setting.port,
        )
        self._apply_market_proxy(setting)

    def set_automatic_sync(
        self,
        on_startup: bool,
        interval_minutes: int | None,
        after_close: bool,
    ) -> None:
        if type(on_startup) is not bool or type(after_close) is not bool:
            raise TypeError("automatic sync flags must be exact bools")
        if interval_minutes is not None and interval_minutes not in AUTOMATIC_SYNC_INTERVALS:
            raise ValueError("AUTOMATIC_SYNC_INTERVAL_INVALID")
        self._mutate(
            automatic_sync_on_startup=on_startup,
            automatic_sync_interval_minutes=interval_minutes,
            automatic_sync_after_close=after_close,
        )

    def test_connections(self) -> tuple[ConnectionTestResult, ...]:
        if self._connection_tester is None:
            raise RuntimeError("SETTINGS_CONNECTION_TEST_UNAVAILABLE")
        return tuple(self._connection_tester(self.state().market_proxy))

    def read_logs(
        self,
        limit: int,
        level: str | None,
        query: str,
    ) -> tuple[DiagnosticLogEntry, ...]:
        if self._log_path is None:
            return ()
        return read_application_logs(
            self._log_path,
            limit=limit,
            level=level,
            query=query,
        )

    def _ensure(self) -> None:
        with self._database.session_factory() as session:
            if session.get(LocalSettingsRecord, _SETTINGS_ID) is not None:
                return
        self._write(self._default_payload())

    def _read(self) -> dict[str, object]:
        with self._database.session_factory() as session:
            row = session.get(LocalSettingsRecord, _SETTINGS_ID)
            if row is None:
                raise LookupError("SETTINGS_MISSING")
            return self._decoded(row)

    @staticmethod
    def _decoded(row: LocalSettingsRecord) -> dict[str, object]:
        decoded = decode_canonical_json(row.payload_json, row.content_hash)
        expected_keys = {
            "active_risk_template",
            "fee_confirmed",
            "log_level",
            "provider_priority",
            "schema_version",
        }
        proxy_keys = {"proxy_mode", "proxy_host", "proxy_port"}
        decoded_keys = frozenset(decoded)
        if (
            row.settings_id != _SETTINGS_ID
            or type(row.schema_version) is not int
            or row.schema_version != _SCHEMA_VERSION
            or not frozenset(expected_keys).issubset(decoded_keys)
            or not decoded_keys.issubset(
                frozenset(
                    expected_keys
                    | {
                        "sync_concurrency",
                        "market_request_timeout_seconds",
                        "automatic_sync_on_startup",
                        "automatic_sync_interval_minutes",
                        "automatic_sync_after_close",
                    }
                    | proxy_keys
                )
            )
            or bool(decoded_keys & proxy_keys) != proxy_keys.issubset(decoded_keys)
            or type(decoded["schema_version"]) is not int
            or decoded["schema_version"] != _SCHEMA_VERSION
            or type(decoded["provider_priority"]) is not list
            or any(
                type(item) is not str for item in cast(list[object], decoded["provider_priority"])
            )
            or type(decoded["fee_confirmed"]) is not bool
            or decoded["active_risk_template"] is not None
            and type(decoded["active_risk_template"]) is not str
            or type(decoded["log_level"]) is not str
            or "sync_concurrency" in decoded
            and (
                type(decoded["sync_concurrency"]) is not int
                or not 1 <= decoded["sync_concurrency"] <= 8
            )
            or "market_request_timeout_seconds" in decoded
            and (
                type(decoded["market_request_timeout_seconds"]) is not int
                or not 3 <= decoded["market_request_timeout_seconds"] <= 60
            )
            or "automatic_sync_on_startup" in decoded
            and type(decoded["automatic_sync_on_startup"]) is not bool
            or "automatic_sync_interval_minutes" in decoded
            and decoded["automatic_sync_interval_minutes"] is not None
            and decoded["automatic_sync_interval_minutes"] not in {30, 60, 240, 720, 1440}
            or "automatic_sync_after_close" in decoded
            and type(decoded["automatic_sync_after_close"]) is not bool
        ):
            raise ValueError("SETTINGS_INTEGRITY")
        normalized = dict(decoded)
        normalized.pop("sync_concurrency", None)
        normalized.setdefault(
            "market_request_timeout_seconds",
            DEFAULT_MARKET_TIMEOUT_SECONDS,
        )
        normalized.setdefault("proxy_mode", MarketProxyMode.SYSTEM.value)
        normalized.setdefault("proxy_host", None)
        normalized.setdefault("proxy_port", None)
        normalized.setdefault("automatic_sync_on_startup", False)
        normalized.setdefault("automatic_sync_interval_minutes", None)
        normalized.setdefault("automatic_sync_after_close", False)
        try:
            LocalSettingsGateway._proxy_from_payload(normalized)
        except (TypeError, ValueError):
            raise ValueError("SETTINGS_INTEGRITY") from None
        return normalized

    @staticmethod
    def _proxy_from_payload(payload: Mapping[str, object]) -> MarketProxySetting:
        mode_value = payload.get("proxy_mode", MarketProxyMode.SYSTEM.value)
        if type(mode_value) is not str:
            raise ValueError("SETTINGS_INTEGRITY")
        try:
            mode = MarketProxyMode(mode_value)
        except ValueError:
            raise ValueError("SETTINGS_INTEGRITY") from None
        return MarketProxySetting(
            mode,
            cast(str | None, payload.get("proxy_host")),
            cast(int | None, payload.get("proxy_port")),
        )

    def _apply_market_proxy(self, setting: MarketProxySetting) -> None:
        if self._proxy_configurer is not None:
            self._proxy_configurer(setting)

    def _mutate(self, **changes: object) -> None:
        payload = self._read()
        payload.update(changes)
        self._write(payload)

    def _write(self, payload: Mapping[str, object]) -> None:
        payload_json = canonical_json(payload)
        with self._database.session_factory.begin() as session:
            row = session.get(LocalSettingsRecord, _SETTINGS_ID)
            if row is None:
                session.add(
                    LocalSettingsRecord(
                        settings_id=_SETTINGS_ID,
                        schema_version=_SCHEMA_VERSION,
                        payload_json=payload_json,
                        content_hash=content_hash(payload_json),
                        updated_at=_timestamp(self._clock),
                    )
                )
            else:
                row.payload_json = payload_json
                row.content_hash = content_hash(payload_json)
                row.updated_at = _timestamp(self._clock)

    def _default_payload(self) -> dict[str, object]:
        return {
            "active_risk_template": None,
            "fee_confirmed": False,
            "log_level": "INFO",
            "provider_priority": [item[0] for item in self._providers],
            "schema_version": _SCHEMA_VERSION,
            "market_request_timeout_seconds": DEFAULT_MARKET_TIMEOUT_SECONDS,
            "proxy_mode": MarketProxyMode.SYSTEM.value,
            "proxy_host": None,
            "proxy_port": None,
            "automatic_sync_on_startup": False,
            "automatic_sync_interval_minutes": None,
            "automatic_sync_after_close": False,
        }
