from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING
from uuid import uuid4
from collections.abc import Mapping, Sequence
import json
import os

from pydantic import TypeAdapter

from compass.storage.canonical_json import canonical_json, content_hash, decode_canonical_json

if TYPE_CHECKING:
    from compass.ui.pages.strategy_lab import StrategyLabConfiguration
    from compass.ui.pages.backtests import BacktestReport


@dataclass(frozen=True)
class SavedResearchConfiguration:
    key: str
    name: str
    configuration: StrategyLabConfiguration


class ResearchWorkspace:
    """Persist validated research configurations as checksummed JSON, never pickle."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = RLock()

    def list(self) -> tuple[SavedResearchConfiguration, ...]:
        from compass.ui.pages.strategy_lab import StrategyLabConfiguration

        with self._lock:
            if not self.path.exists():
                return ()
            try:
                outer = json.loads(self.path.read_text("utf-8"))
                payload = decode_canonical_json(outer["payload"], outer["content_hash"])
                if payload["schema_version"] != 1 or not isinstance(payload["items"], list):
                    raise ValueError
                adapter = TypeAdapter(StrategyLabConfiguration)
                items = tuple(
                    SavedResearchConfiguration(
                        item["key"],
                        item["name"],
                        adapter.validate_json(item["configuration"], strict=True),
                    )
                    for item in payload["items"]
                )
                if any(type(item.key) is not str or type(item.name) is not str for item in items):
                    raise ValueError
                if len({item.key for item in items}) != len(items):
                    raise ValueError
                return items
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("研究配置文件校验失败，请从备份恢复。") from error

    def get(self, key: str) -> SavedResearchConfiguration:
        item = next((item for item in self.list() if item.key == key), None)
        if item is None:
            raise LookupError("研究配置不存在。")
        return item

    def save(
        self, name: str, configuration: StrategyLabConfiguration, *, key: str | None = None
    ) -> SavedResearchConfiguration:
        from compass.ui.pages.strategy_lab import StrategyLabConfiguration

        if not name.strip() or len(name) > 100:
            raise ValueError("配置名称须为 1 至 100 个字符。")
        adapter = TypeAdapter(StrategyLabConfiguration)
        validated = adapter.validate_json(adapter.dump_json(configuration), strict=True)
        entry = SavedResearchConfiguration(key or uuid4().hex, name.strip(), validated)
        with self._lock:
            items = [item for item in self.list() if item.key != entry.key] + [entry]
            payload = canonical_json(
                {
                    "schema_version": 1,
                    "items": [
                        {
                            "key": item.key,
                            "name": item.name,
                            "configuration": adapter.dump_json(item.configuration).decode("utf-8"),
                        }
                        for item in items
                    ],
                }
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
            try:
                temporary.write_text(
                    canonical_json({"payload": payload, "content_hash": content_hash(payload)}),
                    "utf-8",
                )
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)
        return entry


def _sequence(value: object) -> Sequence[object]:
    if not isinstance(value, (tuple, list)):
        raise ValueError("历史配置字段应为列表。")
    return value


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError("历史配置字段应为对象。")
    return dict(value)


def configuration_from_report(report: BacktestReport) -> StrategyLabConfiguration:
    """Reconstruct the editable inputs from the frozen run snapshot."""
    from datetime import date
    from decimal import Decimal
    from compass.backtest.engine import ExecutionTiming
    from compass.domain.market import InstrumentId
    from compass.strategies.rule_dsl import DslVariable, DslExecutableRule
    from compass.strategies.kronos_forecast import KronosForecastParameters
    from compass.ui.pages.strategy_lab import (
        StrategyLabConfiguration,
        StrategyLegConfiguration,
        StrategyLabKind,
        StrategyLabInitialPosition,
        StrategyLabRebalanceMode,
    )

    snapshot = report.snapshot
    pool, fees, allocator = (
        snapshot.instrument_pool,
        snapshot.fee_profile_configuration,
        snapshot.allocator_configuration,
    )
    legs = []
    for strategy in snapshot.strategies:
        params = strategy.parameters
        legs.append(
            StrategyLegConfiguration(
                strategy_id=strategy.sleeve_id,
                strategy=StrategyLabKind(strategy.strategy_type),
                instruments=tuple(
                    InstrumentId.parse(str(item)) for item in _sequence(params["trade_instruments"])
                ),
                budget=Decimal(str(params["budget"])),
                signal_instrument=(
                    InstrumentId.parse(str(params["signal_instrument"]))
                    if params.get("signal_instrument")
                    else None
                ),
                short_window=int(str(params.get("short_window", 20))),
                long_window=int(str(params.get("long_window", 60))),
                confirmation_days=int(str(params.get("confirmation_days", 1))),
                buy_expression=str(params.get("buy_expression", "")),
                sell_expression=str(params.get("sell_expression", "")),
                variables=tuple(
                    DslVariable.model_validate(_mapping(item))
                    for item in _sequence(params.get("variables", ()))
                ),
                dsl_rules=tuple(
                    DslExecutableRule.model_validate(_mapping(item))
                    for item in _sequence(params.get("dsl_rules", ()))
                ),
                kronos_parameters=(
                    KronosForecastParameters.model_validate(_mapping(params["kronos_parameters"]))
                    if params.get("kronos_parameters")
                    else None
                ),
                template_instance_id=(
                    str(params["template_instance_id"])
                    if params.get("template_instance_id")
                    else None
                ),
                template_name=str(params["template_name"]) if params.get("template_name") else None,
            )
        )
    return StrategyLabConfiguration(
        strategies=tuple(legs),
        benchmark=InstrumentId.parse(str(pool["benchmark"])),
        start=date.fromisoformat(str(pool["start"])),
        end=date.fromisoformat(str(pool["end"])),
        initial_cash=Decimal(str(allocator["initial_cash"])),
        commission_rate=Decimal(str(fees["commission_rate"])),
        minimum_commission=Decimal(str(fees["minimum_commission"])),
        slippage_bps=Decimal(str(fees["slippage_bps"])),
        execution_timing=ExecutionTiming(str(snapshot.market_rule_configuration["execution"])),
        rebalance_mode=StrategyLabRebalanceMode(
            str(allocator.get("rebalance_mode", "signal_change"))
        ),
        rebalance_drift=Decimal(str(allocator.get("rebalance_drift", "0.02"))),
        minimum_trade_amount=Decimal(str(allocator.get("minimum_trade_amount", "5000"))),
        initial_cash_weight=Decimal(str(pool.get("initial_cash_weight", "1"))),
        initial_positions=tuple(
            StrategyLabInitialPosition(
                InstrumentId.parse(str(_mapping(item)["instrument"])),
                Decimal(str(_mapping(item)["target_weight"])),
            )
            for item in _sequence(pool.get("initial_positions", ()))
        ),
    )
