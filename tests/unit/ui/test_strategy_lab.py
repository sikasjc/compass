from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from compass.backtest.engine import ExecutionTiming
from compass.domain.market import AssetType, InstrumentId
from compass.services.task_manager import Operation, TaskSnapshot, TaskStatus
from compass.ui.pages.strategy_lab import (
    StrategyLabConfiguration,
    StrategyLabInstrument,
    StrategyLabInitialPosition,
    StrategyLabKind,
    StrategyLabPageModel,
    StrategyLabRebalanceMode,
    StrategyLabTemplate,
    StrategyLegConfiguration,
)
from compass.strategies.kronos_forecast import KronosForecastParameters


INSTRUMENT = InstrumentId.parse("SSE.510300")
NOW = datetime(2026, 8, 9, tzinfo=ZoneInfo("Asia/Shanghai"))


def configuration() -> StrategyLabConfiguration:
    return StrategyLabConfiguration(
        strategies=(
            StrategyLegConfiguration(
                strategy_id="strategy-1",
                strategy=StrategyLabKind.DUAL_MA,
                instruments=(INSTRUMENT,),
                budget=Decimal("1"),
                signal_instrument=INSTRUMENT,
                short_window=20,
                long_window=60,
                confirmation_days=1,
            ),
        ),
        benchmark=INSTRUMENT,
        start=date(2020, 1, 1),
        end=date(2026, 8, 7),
        initial_cash=Decimal("1000000.00"),
        commission_rate=Decimal("0.0003"),
        minimum_commission=Decimal("5"),
        slippage_bps=Decimal("2"),
        execution_timing=ExecutionTiming.NEXT_OPEN,
    )


def test_strategy_configuration_rejects_lookahead_prone_or_invalid_values() -> None:
    with pytest.raises(ValueError, match="short_window"):
        replace(configuration().strategies[0], short_window=60, long_window=20)


def test_named_research_configurations_survive_restart_and_preserve_parameters(tmp_path) -> None:
    from compass.services.research_workspace import ResearchWorkspace

    path = tmp_path / "research.json"
    repository = ResearchWorkspace(path)
    original = replace(
        configuration(),
        initial_cash_weight=Decimal("0.7"),
        initial_positions=(StrategyLabInitialPosition(INSTRUMENT, Decimal("0.3")),),
    )
    saved = repository.save("研究方案", original)
    restarted = ResearchWorkspace(path)
    assert restarted.get(saved.key).configuration == original
    second = restarted.save("另一组参数", replace(original, slippage_bps=Decimal("3.25")))
    assert second.key != saved.key
    assert len(restarted.list()) == 2
    path.write_text(path.read_text("utf-8").replace("3.25", "9.25"), "utf-8")
    with pytest.raises(ValueError, match="校验失败"):
        restarted.list()


@pytest.mark.parametrize("kind", [StrategyLabKind.KRONOS_FORECAST, StrategyLabKind.RULE_DSL])
def test_research_storage_preserves_model_and_rule_parameters(tmp_path, kind) -> None:
    from compass.services.research_workspace import ResearchWorkspace
    from compass.strategies.rule_dsl import DslVariable

    leg = replace(
        configuration().strategies[0], strategy=kind,
        kronos_parameters=(KronosForecastParameters(model_size="mini", lookback=64)
                           if kind is StrategyLabKind.KRONOS_FORECAST else None),
        buy_expression="close > sma(close, window)", sell_expression="close < sma(close, window)",
        variables=(DslVariable(
            name="window", value=Decimal("20"), minimum=Decimal("5"),
            maximum=Decimal("100"), step=Decimal("5"),
        ),),
        template_instance_id="strategy-template-v1", template_name="研究模板",
    )
    original = replace(configuration(), strategies=(leg,))
    repository = ResearchWorkspace(tmp_path / "research.json")
    entry = repository.save("规则与模型配置", original)
    assert repository.get(entry.key).configuration == original


def test_strategy_configuration_rejects_duplicate_ids_and_excess_budget() -> None:
    first = replace(configuration().strategies[0], budget=Decimal("0.6"))
    duplicate = replace(first, instruments=(InstrumentId.parse("SZSE.159949"),))
    with pytest.raises(ValueError, match="unique"):
        replace(configuration(), strategies=(first, duplicate))

    second = replace(duplicate, strategy_id="strategy-2")
    with pytest.raises(ValueError, match="sum to at most one"):
        replace(configuration(), strategies=(first, second))


def test_strategy_configuration_validates_initial_positions_and_cash_weight() -> None:
    configured = replace(
        configuration(),
        initial_cash_weight=Decimal("0.4"),
        initial_positions=(StrategyLabInitialPosition(INSTRUMENT, Decimal("0.6")),),
    )
    assert configured.initial_positions[0].instrument == INSTRUMENT

    with pytest.raises(ValueError, match="sum to one"):
        replace(configured, initial_cash_weight=Decimal("0.5"))


def test_strategy_experiment_validates_rebalance_controls() -> None:
    configured = replace(
        configuration(),
        rebalance_mode=StrategyLabRebalanceMode.WEEKLY,
        rebalance_drift=Decimal("0.02"),
        minimum_trade_amount=Decimal("5000"),
    )
    assert configured.rebalance_mode is StrategyLabRebalanceMode.WEEKLY

    with pytest.raises(ValueError, match="rebalance_drift"):
        replace(configured, rebalance_drift=Decimal("0.30"))
    with pytest.raises(ValueError, match="minimum_trade_amount"):
        replace(configured, minimum_trade_amount=Decimal("-1"))


def test_kronos_strategy_leg_requires_and_preserves_model_parameters() -> None:
    with pytest.raises(ValueError, match="Kronos parameters"):
        StrategyLegConfiguration(
            strategy_id="strategy-kronos",
            strategy=StrategyLabKind.KRONOS_FORECAST,
            instruments=(INSTRUMENT,),
            budget=Decimal("1"),
        )

    parameters = KronosForecastParameters(model_size="mini", lookback=64)
    configured = StrategyLegConfiguration(
        strategy_id="strategy-kronos",
        strategy=StrategyLabKind.KRONOS_FORECAST,
        instruments=(INSTRUMENT,),
        budget=Decimal("1"),
        kronos_parameters=parameters,
    )

    assert configured.kronos_parameters == parameters


class Gateway:
    def __init__(self) -> None:
        self.runs: list[tuple[str, StrategyLabConfiguration]] = []
        self.latest_report_reads = 0

    def instruments(self):  # type: ignore[no-untyped-def]
        return (
            StrategyLabInstrument(
                INSTRUMENT,
                "沪深300ETF",
                AssetType.ETF,
                date(2020, 1, 1),
                date(2026, 8, 7),
                1600,
            ),
        )

    def templates(self):  # type: ignore[no-untyped-def]
        return (
            StrategyLabTemplate(
                "strategy-saved-v1",
                "双均线模板",
                StrategyLabKind.DUAL_MA,
                "1.1.0",
                (INSTRUMENT,),
                {
                    "short_window": 10,
                    "long_window": 30,
                    "confirmation_days": 2,
                    "target_weight": "0.8",
                },
            ),
        )

    def latest_report(self):  # type: ignore[no-untyped-def]
        self.latest_report_reads += 1
        return None

    def report(self, run_id: str):  # type: ignore[no-untyped-def]
        del run_id
        return None

    def new_run_id(self) -> str:
        return "backtest-one"

    def run(self, run_id: str, value: StrategyLabConfiguration) -> None:
        self.runs.append((run_id, value))


class Tasks:
    def __init__(self) -> None:
        self.operation: Operation | None = None
        self.snapshot = TaskSnapshot(
            "task-one",
            "backtest:backtest-one",
            True,
            TaskStatus.QUEUED,
            NOW,
        )

    def submit(self, name: str, heavy: bool, operation: Operation) -> TaskSnapshot:
        assert name == self.snapshot.name
        assert heavy is True
        self.operation = operation
        return self.snapshot

    def status(self, task_id: str) -> TaskSnapshot:
        assert task_id == self.snapshot.task_id
        return self.snapshot


def test_strategy_lab_model_submits_one_heavy_reproducible_run() -> None:
    gateway = Gateway()
    tasks = Tasks()
    model = StrategyLabPageModel(gateway, tasks)

    task = model.start(configuration())

    assert task == tasks.snapshot
    assert tasks.operation is not None
    tasks.operation()
    assert gateway.runs == [("backtest-one", configuration())]


def test_strategy_lab_model_does_not_load_latest_report_by_default() -> None:
    gateway = Gateway()
    model = StrategyLabPageModel(gateway, Tasks())

    state = model.state()

    assert state.latest_report is None
    assert state.active_report is None
    assert gateway.latest_report_reads == 0
    assert state.templates[0].name == "双均线模板"


def test_reopened_research_page_tracks_existing_task_without_resubmission() -> None:
    class SharedTasks(Tasks):
        def snapshots(self):
            return (self.snapshot,)

    tasks = SharedTasks()
    page = StrategyLabPageModel(Gateway(), tasks)
    assert page.state().active_task == tasks.snapshot
    assert tasks.operation is None
    with pytest.raises(RuntimeError, match="BACKTEST_ALREADY_RUNNING"):
        page.start(configuration())


def test_configuration_update_save_as_delete_and_recovery(tmp_path):
    from compass.services.research_workspace import ResearchWorkspace
    path = tmp_path / "research.json"
    workspace = ResearchWorkspace(path)
    first = workspace.save("研究", configuration())
    changed = replace(configuration(), slippage_bps=Decimal("3"))
    workspace.save("研究改名", changed, key=first.key)
    assert len(workspace.list()) == 1
    assert workspace.get(first.key).configuration == changed
    with pytest.raises(ValueError, match="同名"):
        workspace.save("研究改名", configuration())
    second = workspace.save("另一份", configuration())
    assert len(workspace.list()) == 2
    workspace.delete(second.key)
    assert workspace.list()[0].key == first.key
    with pytest.raises(ValueError, match="无需重建"):
        workspace.reset_corrupt()
    path.write_text("{broken", "utf-8")
    backup = workspace.reset_corrupt()
    assert backup.read_text("utf-8") == "{broken"
    assert workspace.list() == ()
    workspace.save("重建", configuration())
    assert len(workspace.list()) == 1


def test_corrupt_configuration_does_not_abort_backtest_render(tmp_path):
    from nicegui import ui
    from compass.services.research_workspace import ResearchWorkspace
    from compass.ui.pages.strategy_lab import render_strategy_lab_page

    path = tmp_path / "research.json"
    path.write_text("{broken", "utf-8")
    model = StrategyLabPageModel(Gateway(), Tasks(), workspace=ResearchWorkspace(path))
    with ui.column() as container:
        render_strategy_lab_page(model, configuration_key="saved-by-old-link")
    elements = list(container.descendants())
    texts = [getattr(item, "text", "") for item in elements]
    assert any("已保存配置暂不可用" in text for text in texts)
    assert "运行回测" in texts
    assert "保留损坏文件并重建列表" in texts
    container.delete()
    assert path.read_text("utf-8") == "{broken"
