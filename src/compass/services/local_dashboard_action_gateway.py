from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from compass.services.export_service import ExportService
from compass.services.local_crud_gateways import LocalStrategyGateway
from compass.services.local_decision_gateway import LocalDecisionGateway
from compass.services.safe_display import safe_identifier
from compass.storage.dataset_bundle_repository import DatasetBundleRepository
from compass.ui.pages.dashboard import (
    DashboardActionReceipt,
    DashboardExportReceipt,
    DashboardManifestChoice,
    DashboardStrategyChoice,
)


Clock = Callable[[], datetime]
IdFactory = Callable[[str], str]


class LocalDashboardActionGateway:
    """Narrow local Web boundary for formal close decisions and their exports."""

    def __init__(
        self,
        *,
        strategies: LocalStrategyGateway,
        bundles: DatasetBundleRepository,
        decisions: LocalDecisionGateway,
        exports: ExportService,
        clock: Clock,
        id_factory: IdFactory,
    ) -> None:
        if not callable(clock) or not callable(id_factory):
            raise TypeError("dashboard action clock and id factory must be callable")
        self._strategies = strategies
        self._bundles = bundles
        self._decisions = decisions
        self._exports = exports
        self._clock = clock
        self._id_factory = id_factory

    def list_strategies(self) -> tuple[DashboardStrategyChoice, ...]:
        choices = [
            DashboardStrategyChoice(item.instance_id, item.name)
            for item in self._strategies.list()
            if item.enabled and self._strategies.is_watchlist_enabled(item.watchlist_id)
        ]
        return tuple(sorted(choices, key=lambda item: item.strategy_instance_id))

    def list_manifests(
        self,
        strategy_instance_id: str,
    ) -> tuple[DashboardManifestChoice, ...]:
        checked = safe_identifier(strategy_instance_id, label="strategy instance id")
        instance = next(
            (
                item
                for item in self._strategies.list()
                if item.instance_id == checked and item.enabled
            ),
            None,
        )
        if instance is None or not self._strategies.is_watchlist_enabled(instance.watchlist_id):
            raise LookupError("DECISION_STRATEGY_UNAVAILABLE")
        pool = set(self._strategies.pool_instruments(checked))
        choices = [
            DashboardManifestChoice(
                bundle.primary_manifest_id,
                bundle.bundle_id,
                bundle.instruments,
                bundle.created_at,
            )
            for bundle in self._bundles.list()
            if pool.issubset(bundle.instruments)
        ]
        return tuple(sorted(choices, key=lambda item: item.manifest_id))

    def generate_close_decision(
        self,
        strategy_instance_id: str,
        manifest_id: str,
    ) -> DashboardActionReceipt:
        decision_id = safe_identifier(
            self._id_factory("decision"),
            label="decision id",
        )
        record = self._decisions.generate(
            decision_id,
            strategy_instance_id,
            manifest_id,
        )
        if record.decision_id != decision_id:
            raise ValueError("DECISION_RECEIPT_MISMATCH")
        return DashboardActionReceipt(
            decision_id,
            strategy_instance_id,
            manifest_id,
            record.result.decision_at,
        )

    def export_decision(self, decision_id: str | None) -> DashboardExportReceipt:
        record = (
            self._decisions.latest()
            if decision_id is None
            else self._decisions.get(
                safe_identifier(decision_id, label="decision id")
            )
        )
        if record is None:
            raise LookupError("DECISION_EXPORT_MISSING")
        csv_path, json_path = self._exports.export_decision(record.decision_id)
        exported_at = self._clock()
        if (
            type(exported_at) is not datetime
            or exported_at.tzinfo is None
            or exported_at.utcoffset() is None
        ):
            raise ValueError("dashboard action clock must return a timezone-aware datetime")
        return DashboardExportReceipt(
            record.decision_id,
            csv_path.name,
            json_path.name,
            exported_at,
        )
